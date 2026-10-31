# Soup Kernel Challenge — Triton vocabulary reductions

Первая версия оптимизации обучения **Qwen2.5-1.5B-Instruct на Kaggle Tesla T4** для
Decentrathon 6.0: **Soup 0.77.0**, PyTorch и собственные Triton-кернелы.

Последний переданный командой отчёт Kaggle T4: **79.9 → 79.0 с**,
**14345 → 9243 MiB**, eval loss **2.04969 → 2.04967**. Прошли 9/10 проверок
и 79 Soup-тестов; check 4 не прошёл: собственные кернелы заняли **2.5%**.
Это результаты предыдущего запуска, не подтверждение текущего кандидата.

**Текущий отдельный кандидат: `codex/t4-mlp-cast-elision`.** Он основан на
`6908b39` (`codex/t4-strided-accuracy`), результат которого ещё ожидается.
Новый autograd-путь убирает внутренние FP32↔FP16-копии замороженного NF4 MLP,
сохраняя native bitsandbytes GEMM и decode, отдельное округление градиентов
ветвей и их FP32-суммирование. Штатные SwiGLU-кернелы используются без изменений
и не засчитываются как собственные. Детали и команды:
[solution/MLP_CAST_CANDIDATE.md](solution/MLP_CAST_CANDIDATE.md).

Родительский эксперимент устраняет копию сдвинутых logits для accuracy:
[solution/ACCURACY_CANDIDATE.md](solution/ACCURACY_CANDIDATE.md).
Ветки `codex/t4-strided-accuracy` и `codex/triton-vocab-t4` сохранены.
Обычный `tl.dot` на T4 остаётся отключён: native Tensor Core GEMM сохраняется
([SM75-аудит](solution/SM75_AUDIT.md)). Остальные оптимизации описаны в
[solution/OPTIMIZATION_NOTES.md](solution/OPTIMIZATION_NOTES.md).

**Цель ≤60 секунд, дополнительная экономия памяти, GPU-корректность нового пути
и доля ≥5% пока не подтверждены.** Локально выполняются CPU-проверки и offline
SM75-компиляция; они не заменяют обучение и тесты на Kaggle T4.
Публичный Kaggle notebook пока не опубликован. После Save & Run All его ссылку
нужно внести сюда и в [SUBMISSION.md](SUBMISSION.md).

## Архитектура и первая оптимизация

Все 24 исходных файла сверены по Git blob SHA с
[официальным шаблоном](https://github.com/MakazhanAlpamys/soup-kernel-challenge):
до изменений они полностью совпадали.
Подробный обзор: [solution/ARCHITECTURE.md](solution/ARCHITECTURE.md).

Harness запускает baseline и solution в отдельных процессах. `solution.apply()`
вызывается до создания модели. Soup строит NF4 double-quant модель с FP16 compute
и LoRA r=16 на q/v. Transformers считает causal cross-entropy в FP32, TRL считает
entropy/accuracy, Soup пишет per-row loss и token counts в `rewind.jsonl`.

Первая оптимизация заменяет редукции по словарю из **151936** классов:

| Собственный кернел | Работа |
|---|---|
| `vocab_partition_kernel` | Частичные max, sum(exp) и первый момент по тайлам из 4096 классов |
| `vocab_ce_finalize_kernel` | Объединение частичных сумм и token CE |
| `vocab_ce_backward_kernel` | Пересчёт softmax и градиент `(p - one_hot) * grad_loss` |
| `vocab_entropy_finalize_kernel` | Энтропия через partition function и первый момент |
| `rmsnorm_row_forward_kernel` | Нормализация и масштабирование frozen FP32 RMSNorm |
| `rmsnorm_row_backward_kernel` | Входной градиент RMSNorm с пересчётом normalized values |
| `mlp_input_grad_kernel` | FP16 GEMM с FP32 accumulation для MLP backward только на SM80+; на T4 native |
| `vocab_argmax_tile_kernel` | Максимум и первый индекс внутри vocabulary-тайла без копии logits |
| `vocab_argmax_finish_kernel` | Объединение максимумов в точные token predictions для accuracy |

Код написан самостоятельно: [solution/vocab_ops.py](solution/vocab_ops.py).
FP32-редукции и сдвинутые экспоненты обеспечивают численную устойчивость.
Backward сохраняет logits и два скаляра на токен, пересчитывает вероятности.
Выделение отдельной матрицы `log_softmax` в forward не требуется.
Entropy и rewind также обходятся без промежуточных матриц размером со словарь.
**Фактический peak памяти и скорость нужно измерить на T4.**

[solution/patches.py](solution/patches.py) подключает кернелы к трём операциям:

- `transformers.loss.loss_utils.fixed_cross_entropy`: исходный label shifting
  и нормировка `num_items_in_batch` сохраняются. `LOSS_MAPPING` и identity
  `ForCausalLMLoss` сохраняются, чтобы Trainer правильно считал supervised tokens.
- Оба alias `entropy_from_logits` в TRL utils/SFTTrainer.
- `soup_cli.trainer.rewind_hf.row_losses`: прежнее выравнивание `t -> t+1`,
  строки и token counts; пустая supervised-строка даёт loss 0 и count 0.

Metrics hooks ограничены CUDA FP32 logits нужного словаря. Остальные случаи используют
оригинальные функции. FP32 в реальном main CE обеспечивает исходный
`ForCausalLMLoss`, в возвращаемых logits для метрик — штатный Accelerate wrapper.
Повторный `apply()` безопасен.

Новый instance loss hook Qwen2 принимает исходные FP16 logits и накапливает CE
в FP32 внутри Triton, сохраняя FP16 gradient. Он заменяет `CE(logits.float())`
без отдельной плотной FP32-копии для loss; forward/backward сравниваются с
исходным Transformers. Полные logits, FP32 metrics и token counting сохраняются.

Штатный SwiGLU оставлен включённым. Для proven FP16-origin выходов frozen NF4
gate/up проекций backward-кэш хранится в FP16 без потери значений; forward и
градиенты остаются FP32. При полном батче объём кэша расчётно уменьшается на
1.914 GiB; это не измеренное изменение peak памяти.
**Его кернелы не считаются нашими**, включая долю 5%.
Модель, данные, batch order/size, LoRA/dropout, optimizer, scheduler и настройки
обучения не менялись. `harness/`, `tests/`, `task/` не изменены.

## Запуск в Kaggle T4

1. Импортировать [kaggle_t4.ipynb](kaggle_t4.ipynb) в новый Kaggle notebook.
2. Выбрать **GPU T4 x2**, включить Internet. Все GPU-процессы используют только
   GPU 0. Перед измерениями обе GPU должны быть свободны.
3. URL репозитория заполнен. Для разработки `COMMIT` указывает на ветку решения;
   для итогового воспроизведения заменить его полным SHA проверяемого commit.
4. Run All. `subprocess.run(check=True)` останавливает notebook при ошибках.
5. Сохранить публичную версию с фиксированным SHA и заполнить ссылки/результаты.

Зависимости устанавливаются до измерений. Версии соответствуют шаблону;
torch/triton используются как совместимая пара Kaggle и записываются harness.

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Команды рассчитаны на Linux/CUDA. Regression должен показать ровно **79 passed**.
Новые тесты находятся в `solution/`, отдельно от защищённого `tests/`.
Для анализа после основных измерений:
`python harness/run_task.py --only solution --profile --out runs_profile`.
Этот режим не является полным benchmark: в нём нет baseline.

## Корректность и правила

[solution/test_vocab_ops.py](solution/test_vocab_ops.py) проверяет CE forward и
first-order backward против PyTorch: словарь 151936, хвост 4103, произвольный
upstream gradient, `none/sum/mean`, mixed/all ignored labels, row stride,
повторный backward, неизменность входов, CPU/FP16 fallback. Для entropy проверяются
формы, большие смещения logits и native autograd. Integration tests проверяют
hooks, denominator, rewind и идемпотентность.
**Пропуск CUDA-тестов не подтверждает корректность кернелов.**

Для допуска нужны все 10 проверок `harness/check_rules.py`:

| Проверка | Требование |
|---|---|
| 1–2 | Запуски завершились; нет запрещённых библиотек/установки пакетов во время обучения |
| 3–4 | Наши кернелы запускаются в каждом solution-run; их доля ≥5% |
| 5 | Совпадают task, adapter, settings, steps, tokens и Soup outputs |
| 6 | First-step loss в пределах 1%, mean loss-curve gap ≤3% |
| 7 | Eval loss не более +0.5% к baseline |
| 8 | Время и память не хуже baseline более чем на 10% |
| 9 | T4, GPU 0, idle start, штатный allocator, согласованные memory peaks |
| 10 | 79 regression tests Soup с включённым `apply()` |

Доля — сумма CUDA-event `own_kernel_ms` оригинальных кернелов, запущенных во всех
solution-runs, делённая на `1000 * timed_loop_s` шагов 3+. Берётся худший результат
повторных запусков. Она определяется только реальным `runs/result.json`.
Ускорение операции само по себе не гарантирует 5%. Если проверка 4 не пройдёт,
следующий полезный кернел выбирается по profiler.

## Результаты и ограничения

| Проверка/метрика | Текущий результат |
|---|---|
| Сверка исходного шаблона | 24/24 blob SHA совпадают |
| Штатный `source_kernels()` | Проверить текущий кандидат; определение кернела не подтверждает его запуск или долю |
| Python / notebook syntax, `git diff --check` | Успешно; GPU-код не запускался |
| GPU correctness на T4 | Для `6e6781b` команда сообщила 51 PASS / 2 FAIL; FP16 entropy исправлен, повторная GPU-проверка ожидается |
| 79 Soup regression tests | Не запускалось: локально нет зависимостей |
| Предыдущая версия, по отчёту команды | 1.084x; 12385 vs 14345 MiB; eval 2.04965 vs 2.04967 |
| Новая версия: время / peak MiB / eval / доля | Требуется повторный T4 benchmark |
| `6b175fb`: правила, по отчёту команды | 9/10 PASS; check 4 FAIL при 2.4%; 79 Soup tests PASS |
| Все 10 правил | Не подтверждено |

Это первая реализация без T4 autotuning, с фиксированным tile 4096. Холодная
компиляция учитывается в benchmark. Custom CE поддерживает first-order backward;
double backward не реализован. FP16/BF16 и неподходящие layouts идут через native
fallback. Изменение пика памяти и скорости не гарантируется до измерения.

Разбор FP16 entropy, сравнение actual/reference на CPU и границы проверки
исправления: [solution/ENTROPY_FIX.md](solution/ENTROPY_FIX.md).

## Источники и лицензии

Код кернелов не заимствован. Математика: стандартные stable log-sum-exp, categorical
entropy и производная CE. Контракты сверены по первичным источникам:
[Transformers 5.16.1 loss_utils](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py),
[Trainer token counting](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py),
[TRL 0.29.1 entropy](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py),
[Soup 0.77.0 rewind](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py).

Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE). SwiGLU и защищённые тесты из шаблона.
Данные: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT на карточке), перевод [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
(CC BY-NC 4.0). Модель:
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).
