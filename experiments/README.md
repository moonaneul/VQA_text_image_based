# 실험 기록 규칙

`run_vlm_baseline.py`는 실행이 끝날 때 `experiments.csv`에 한 줄을 추가합니다. leaderboard 점수와 해석은 제출 후 같은 행에 수동으로 기록합니다.

## 비교 대상으로 인정되는 run

- seed와 validation CSV가 기록되어 있음
- 모델, prompt, resolution, quantization, decision method 중 무엇을 바꿨는지 명확함
- 전체 accuracy와 질문 유형별 accuracy가 존재함
- OCR-heavy aggregate accuracy와 parse failure rate가 기록됨
- 예측 CSV와 `run.json` 경로가 남아 있음
- latency, throughput, peak VRAM이 가능하면 기록됨
- git commit SHA가 남아 있어 실행 코드를 복원할 수 있음
- 실패한 실험도 삭제하지 않고 notes에 원인을 기록함

권장 이름은 `YYYYMMDD_HHMMSS_stage_prompt_resolution`입니다.

## W&B

W&B는 선택 사항입니다. 로컬 CSV/JSON 기록은 항상 남고, `--wandb`를 지정한 경우 동일 run을 W&B에도 기록합니다.

```powershell
pip install wandb
wandb login

.\baseline\Scripts\python.exe .\scripts\run_vlm_baseline.py `
  --csv .\splits\val_random.csv `
  --prompt direct `
  --resolution standard `
  --run-id B0_random_direct_standard `
  --wandb `
  --wandb-group B0
```

W&B artifact에는 최소 `predictions.csv`와 `run.json`이 포함됩니다.

## Pairwise ablation comparison

두 실험의 accuracy 차이만 보지 말고 같은 샘플에서 무엇이 뒤집혔는지 확인합니다.

```powershell
.\baseline\Scripts\python.exe .\scripts\compare_runs.py `
  --baseline .\output\baseline\B0_random_direct_standard\predictions.csv `
  --candidate .\output\baseline\A1_random_ocr_standard\predictions.csv `
  --output .\output\comparisons\B0_vs_A1.json
```

출력에는 다음 값이 포함됩니다.

- baseline / candidate accuracy
- percentage-point delta
- baseline wrong -> candidate right
- baseline right -> candidate wrong
- net sample gain
- prediction disagreement rate
- 질문 유형별 동일 통계
- 실제로 prediction이 바뀐 sample 목록

작은 accuracy 차이는 이 paired 결과와 오류 유형을 함께 보고 해석합니다.
