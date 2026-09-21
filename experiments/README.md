# 실험 기록 규칙

`run_vlm_baseline.py`는 실행이 끝날 때 `experiments.csv`에 한 줄을 추가합니다. leaderboard 점수와 해석은 제출 후 같은 행에 수동으로 기록합니다.

각 실험은 다음 조건을 만족해야 비교 대상으로 인정합니다.

- seed와 validation CSV가 기록되어 있음
- 모델, prompt, resolution, quantization 중 무엇을 바꿨는지 명확함
- 전체 accuracy와 질문 유형별 accuracy가 존재함
- 예측 CSV와 `run.json` 경로가 남아 있음
- 실패한 실험도 삭제하지 않고 notes에 원인을 기록함

권장 이름은 `YYYYMMDD_HHMMSS_stage_prompt_resolution`입니다.

