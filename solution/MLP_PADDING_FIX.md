# T4 MLP padding fallback

Для commit `092d9e020f5d87889015b841a5151c1a3bda4383` пользователь сообщил
результат Kaggle NVIDIA T4: 210 passed, 1 skipped, 1 failed. В тесте
`test_production_qwen_nf4_mlp_padding_values_gradients_and_rng` входной градиент
packed MLP не совпал с исходным при `rtol=1e-2, atol=2e-4`: 3040/393216
элементов, максимальная абсолютная разница 0.015625. Это результат старого
commit, а не проверка исправления.

## Что установлено

`active_mlp_rows` строит уникальные возрастающие индексы. Causal labels
сдвигаются внутри каждой последовательности до flatten. Сохраняются позиции
с ненулевой attention mask и позиции, влияющие на supervised loss, включая
query вне attention mask. В примере остаётся 161 из 256 строк.

Для построчного MLP с frozen весами и без dropout обозначим gather через P.
Тогда packed forward равен `Pᵀ f(Px)`, а его VJP —
`Pᵀ J_f(Px)ᵀ P v`. На сохранённых строках это тот же математический MLP;
на исключённых строках output и градиент равны нулю. У dense MLP градиент
исключённых строк также нулевой, если upstream там нулевой. Его outputs
на этих строках в общем случае ненулевые. `index_copy_` пишет в свежий
нулевой тензор, индексы не повторяются; `reshape` сохраняет логический порядок
и для неконтинуального входа. Ошибки gather/scatter/backward не обнаружено.
CPU-регрессия сверяет оба VJP с независимо выписанной аналитической
производной SwiGLU в FP64 с допуском `1e-12`, включая исключённые строки,
supervised query вне mask, неконтинуальный вход и повторный backward.

В конкретном случае обе формы проходят `mlp_casts.MIN_ROWS=128` и используют
один `_CastElidedMLP`. NF4 decoder получает одинаковые веса и quant state,
не зависит от числа токенов и возвращает отдельные FP16 буферы.
`compact_swiglu` сохраняет FP16-origin значения без потерь. `mlp_casts`
оставляет FP16 GEMM и отдельные округления gate/up gradient до их сложения
в FP32. Упаковка меняет M у трёх forward и трёх backward GEMM с 256 на 161.

Дефект — использование смены формы FP16 GEMM без подтверждённой численной
эквивалентности на T4. Разные формы могут выбирать разные порядки накопления,
а округления промежуточных значений и компенсация двух ветвей backward
могут усиливать расхождение около нуля. Это согласуется с отчётом, но
конкретный первый расходящийся GEMM **ещё не измерен на T4**.
По одному максимальному отклонению нельзя доказать источник ошибки или
обосновать ослабление допуска. PyTorch также не гарантирует идентичность
вычисления на срезе и среза полного результата:
[Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
Cast и backward сверены с исходниками
[Linear4bit 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py)
и [MatMul4Bit 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py).

## Исправление

На SM75 wrapper вызывает предыдущий MLP forward с **тем же объектом входа**,
полной формой и stride. Он не вызывает `_packed_call`. Это сохраняет все
GEMM-размеры, cast boundaries и исходный backward, включая Jacobian
исключённых строк. На других архитектурах существующий dispatch сохранён;
данное исправление не подтверждает его GPU-корректность.

Active-row context остаётся доступен head. Параметры модели, NF4 веса,
LoRA, dropout, attention, метрики и `rewind.jsonl` не изменены. `harness/`,
`task/` и `tests/` не редактировались. Допуски не ослаблены: CUDA-регрессия
теперь требует точного совпадения полного forward и двух VJP с предыдущим
MLP, один с нулевым upstream на исключённых строках, другой — с ненулевым.
Она также проверяет RNG, неизменность frozen весов и отсутствие их gradients
для FP32 и FP16 hidden states. CPU metadata-тест доказывает вызов исходного
объекта с полной формой без выполнения CUDA.

## Проверки и воспроизведение

Локально, 1 ноября 2026: Windows, Python 3.14.3, PyTorch 2.15.0+cpu,
Triton 3.6.0, pytest 9.2.0.

- `python -m pytest solution -x -vv --tb=short`: **165 passed, 219 skipped**.
  CUDA и проверки отсутствующих training packages пропущены.
- FP64 helper диагностического скрипта отдельно совпал с независимым CPU
  autograd для output/VJP при `rtol=atol=1e-12`.
- `SOUP_APPLY_SOLUTION=1 python -m pytest tests -x -q --tb=short`:
  загрузка conftest остановилась на отсутствующем `bitsandbytes`; 79 Soup
  regression tests не выполнены.
- `git diff --check` и Python syntax checks прошли.

**Исправленный commit не запускался на Kaggle T4.** Также не выполнены
training benchmark и проверки правил по новым измерениям. Отказ от MLP
packing может увеличить время и память относительно старого кандидата;
это необходимо измерить обычным harness без изменения его проверок.

На Kaggle T4 с фиксированными training dependencies:

```bash
python -m pytest solution -x -vv --tb=short
python -m solution.diagnose_mlp_padding > /kaggle/working/mlp_padding_diagnostic.json
```

Диагностика запускается только явно, не импортируется `solution.apply()` и
не добавляет работу в обучение. Она воспроизводит seed 743 и принудительно
измеряет старый packed путь; отдельно сравнивает selected/dropped outputs
и gradients с dense и production fallback. Шесть GEMM probes меняют только
M при одинаковых выбранных операндах. Проверка native/patched NF4 decode
отделяет декодирование весов от GEMM. Независимый аналитический FP64 эталон
использует те же native-decoded NF4 веса, но не повторяет промежуточные
FP16 округления: он оценивает точность обеих версий относительно реальной
формулы, а не заменяет обязательное сравнение с исходным FP16 MLP.
По умолчанию эталон считается на CPU; `--reference-device cuda` переносит
его на GPU. JSON содержит версии, GPU и флаг reduced-precision reduction.
