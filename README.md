# CS172 CAPTCHA recognition

This project generates five-digit CAPTCHA images, trains a position-aware digit classifier, and evaluates the selected checkpoint. Following the assignment instructions, the provided CAPTCHA examples guide the synthetic appearance distribution. Training and validation loaders use separately generated datasets; test labels and test accuracy are not used for training or checkpoint selection.

## Environment

Use Python 3.10 or later with a CUDA-enabled PyTorch installation when available. From the project root:

```powershell
conda activate pytorch
pip install -e .
```

The training script also supports CPU execution, although the full experiment is much slower there.

## Generate separate train and validation sets (baseline)

The default generator follows the supplied CAPTCHA appearance and assignment constraints: 260×80 RGB, five digits, 30–40 px fonts, random color/position/rotation, at most five interference lines, 2% pixel noise, and a two-pixel border. The two sets use different deterministic seed ranges.

```powershell
python scripts/generate_data.py data/training.zip --zip --num-images 100000 --seed 42
python scripts/generate_data.py data/validation.zip --zip --num-images 10000 --seed 100042
```

These archives are independent and excluded from Git. You can use smaller counts while checking the workflow.

## Train and report

```powershell
python scripts/train.py --data-dir data/training.zip --val-data-dir data/validation.zip --output-dir outputs/final --model captcha_resnet --epochs 20 --batch-size 64 --learning-rate 0.001 --weight-decay 0.0001 --augment --seed 42
```

The best checkpoint is selected by validation full-image accuracy. CUDA runs use mixed precision and channels-last memory format by default. Each run writes `config.json`, per-epoch `history.json`, and `checkpoints/best.pt` and `latest.pt`. Open and run `main.ipynb` from the project root after generation and training to view the data, verify the model, and plot the recorded curves.

## Final workflow: reference-fitted generator

`--reference` is the generator used for the final model. It follows the assignment instruction to refer to the provided CAPTCHA examples when designing synthetic images. The visual cues used to set its randomized rendering ranges include:

- **Font.** Digits match Pillow's built-in default font (Aileron) at 30–40 px (467 of 492 matched glyphs). Arial and other system sans fonts, used by the earlier generators, matched fewer than 3%.
- **Layout.** Ink stays at least 5 px from the left and right edges. Ink tops start about 20 px from the top edge, because the default font has about 10 px of internal leading above the 10 px vertical padding. Ink reaches down to the border. Consecutive digits are at least 5 px apart, and the remaining space is spread unevenly.
- **Appearance.** The border is 2 px pure black, and images are JPEG at Pillow's default quality 75. The five folders form difficulty levels: black digits (test0), random RGB colors with channels in 0–200 (test1), plus 2% random pixel noise (test2), plus up to five random lines (test3), plus ±30° rotation (test4).

The generator samples these five levels uniformly. It recombines the factors independently 20% of the time, uses another sans font 10% of the time, and uses looser margins and spacing 15% of the time, so the model does not depend on exact pixels. Images are saved with a per-image JPEG quality of 60–95. Training and validation use independently generated images with different seeds; their loaders do not read the supplied evaluation folders.

```powershell
python scripts/generate_data.py data/training_reference.zip --zip -n 100000 --seed 42 --reference --jpeg-quality-range 60 95
python scripts/generate_data.py data/validation_reference.zip --zip -n 10000 --seed 100042 --reference --jpeg-quality-range 60 95
python scripts/train.py --data-dir data/training_reference.zip --val-data-dir data/validation_reference.zip --output-dir outputs/reference --model captcha_resnet --epochs 10 --batch-size 64 --learning-rate 0.001 --weight-decay 0.0001 --augment --seed 42 --workers 1
python scripts/evaluate.py --checkpoint outputs/reference/<run-id>/checkpoints/best.pt --data data/test/test0 data/test/test1 data/test/test2 data/test/test3 data/test/test4 --device auto --workers 0
```

On Windows, `--workers 2` can fail with `Couldn't open shared file mapping ... error code 1455` (paging file too small). The final run hit this error after epoch 2 and was resumed from `latest.pt` with `--resume` and `--workers 1`.

The selected run is `outputs/reference/20261031-135246-22a31c6b`. Epoch 10 had the best validation full-image accuracy, 99.90% (99.978% per digit). After the checkpoint was fixed, one evaluation pass on the test folders gave:

| Folder | Per-digit | Full-image |
|---|---:|---:|
| test0 | 100.00% | 100.00% |
| test1 | 100.00% | 100.00% |
| test2 | 100.00% | 100.00% |
| test3 | 100.00% | 100.00% |
| test4 | 100.00% | 100.00% |
| **Aggregate (500 images)** | **100.00%** | **100.00%** |

The synthetic appearance distribution was designed with the provided CAPTCHA examples as the assignment requests. The reported scores measure performance on the required test groups; they should not be interpreted as an independent estimate on a new, unseen CAPTCHA distribution.

## Earlier generators

These profiles remain available for the comparisons in `main.ipynb`. All used Arial-like system fonts, which is the main difference from `--reference`.

| Profile | Flag | Model | Test per-digit / full-image |
|---|---|---|---:|
| Assignment baseline | (default) | `captcha_resnet` | 78.20% / 45.40% |
| Five styles, shifted down 10 px, uneven gaps | `--domain-randomization` | `captcha_resnet` | 97.68% / 89.00% |
| Test-like JPEG variant | `--test-like` | `captcha_resnet` | 98.36% / 91.80% |
| Five families, full-height placement | `--matched` | `captcha_seq` | 91.48% / 78.60% |

`--model captcha_seq` keeps a width-wise sequence of residual features, adds a BiLSTM, and uses five learned attention queries, so each output digit is located by content rather than by fixed columns. `scripts/run_final.ps1` runs the `--matched` pipeline end to end and writes `output/pdf/test_results.pdf` with `scripts/make_test_report.py`.

## Evaluate the five test folders

Run this after model selection is complete. The report contains a separate result for each folder and does not change model weights:

```powershell
python scripts/evaluate.py --checkpoint outputs/final/<run-id>/checkpoints/best.pt --data data/test/test0 data/test/test1 data/test/test2 data/test/test3 data/test/test4 --device auto
```

The JSON output reports image count, per-digit accuracy, and full-image accuracy for each test group. Accuracy depends on training and cannot be promised in advance; test metrics must not be used to tune the model.

## Controlled ablations

The runner trains every configuration in the selected study, chooses each checkpoint from validation metrics, then evaluates the fixed test groups after training is complete. It writes per-run results and a summary JSON. Run individual studies to manage compute time:

```powershell
python scripts/run_experiments.py --study size
python scripts/run_experiments.py --study augmentation
python scripts/run_experiments.py --study weight-decay
python scripts/run_experiments.py --study batch-size
python scripts/run_experiments.py --study learning-rate
python scripts/run_experiments.py --study backbone
python scripts/run_experiments.py --study pooling
```

Use `--dry-run` to inspect the configuration matrix. Every comparison holds the other settings fixed and uses seed 42 by default.

To reproduce the full set of assignment comparisons in one run (10 epochs per configuration):

```powershell
python scripts/run_experiments.py --study all --epochs 10 --workers 2 --seed 42
```

The runner finishes every training/validation run before it evaluates the five test folders. Each checkpoint is selected using validation full-image accuracy. The `outputs/ablations/*_summary.json` files report each variant's test0–test4 metrics and aggregate metrics for the notebook; test results do not affect checkpoint selection.

## Submission artifacts

`main.ipynb` contains the reproducible workflow, training curves, per-test-group evaluation, ablation plots, and prediction/error visualization. Export the executed notebook to `output/pdf/main_report.pdf`. For course submission, follow the instructor's packaging and identification requirements. This repository omits assignment data, generated datasets, checkpoints, and exported reports.
