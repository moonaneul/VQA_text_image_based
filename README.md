# SSAFY 16기 2회차 VQA Challenge

이미지와 한국어 질문, 네 개의 선택지에서 정답 하나를 고르는 VQA 대회 프로젝트입니다. 원본 데이터는 `data/`에 그대로 두고, 재현 가능한 EDA·검증 분할·베이스라인 실행·실험 기록을 분리합니다.

## 현재 결론

- train/dev/test 이미지와 CSV는 모두 존재하며 손상되거나 누락된 이미지는 없습니다.
- train의 68.7%, test의 67.7%가 질문 문구 기준으로 장면 텍스트·가격·전화번호·메뉴 판독형입니다.
- 실제 표본도 간판, 메뉴판, 포스터, 가격표, 날짜, 전화번호처럼 작은 글자를 정확히 읽어야 풀리는 문제가 중심입니다.
- 첫 모델은 이미 로컬에 준비된 `Qwen2.5-VL-3B-Instruct`의 zero-shot 추론으로 고정합니다. LoRA보다 먼저 prompt와 해상도의 효과를 측정합니다.
- 주 평가는 정답과 질문 유형을 함께 층화한 random 80/20 split입니다. 동일 질문 템플릿과 동일 이미지가 양쪽에 걸치지 않는 grouped split을 함께 보고 일반화 차이를 측정합니다.
- dev의 5개 응답은 정답으로 간주하지 않습니다. 2,026행에서 최다 득표가 3개이고, 4개 이상 일치는 한 건도 없으며 465행은 최다 득표 동률입니다.

상세 근거는 `reports/eda/README.md`에 있습니다.

## 폴더 구조

```text
data/                         원본 CSV와 이미지. 수정하지 않음
downloads/models/             로컬 사전학습 모델
scripts/run_eda.py            CSV·이미지 무결성 및 분포 분석
scripts/make_splits.py        random/grouped validation 생성
scripts/run_vlm_baseline.py   validation 평가와 test 추론
configs/                      고정된 실험 설정과 실행 예시
splits/                       생성된 train/validation CSV
reports/eda/                  EDA 요약, 표본, 이미지 메타데이터
experiments/experiments.csv   실험 결과 누적 기록
output/baseline/              실행별 예측, 지표, 제출 파일
```

## 권장 실행 순서

프로젝트 루트에서 기존 가상환경을 사용합니다.

```powershell
.\baseline\Scripts\python.exe .\scripts\run_eda.py --data-dir .\data --output-dir .\reports\eda
.\baseline\Scripts\python.exe .\scripts\make_splits.py --data-dir .\data --output-dir .\splits --metadata .\reports\eda\image_metadata.csv
```

먼저 50개로 모델 로딩과 출력 파싱을 확인합니다.

```powershell
.\baseline\Scripts\python.exe .\scripts\run_vlm_baseline.py --csv .\splits\val_random.csv --prompt direct --resolution standard --max-samples 50
```

정상 동작하면 전체 random validation을 실행하고, 같은 설정으로 grouped validation을 실행합니다.

```powershell
.\baseline\Scripts\python.exe .\scripts\run_vlm_baseline.py --csv .\splits\val_random.csv --prompt direct --resolution standard
.\baseline\Scripts\python.exe .\scripts\run_vlm_baseline.py --csv .\splits\val_grouped.csv --prompt direct --resolution standard
```

그다음 `direct × standard`를 기준으로 prompt만 바꾼 실험과 resolution만 바꾼 실험을 한 번에 하나씩 비교합니다. test 추론은 validation 설정을 확정한 후 수행합니다.

```powershell
.\baseline\Scripts\python.exe .\scripts\run_vlm_baseline.py --csv .\data\test.csv --prompt ocr_deliberate --resolution high
```

## 실험 원칙

1. 한 번에 한 요소만 바꿉니다.
2. random과 grouped validation 점수를 함께 기록합니다.
3. 전체 accuracy와 질문 유형별 accuracy를 함께 봅니다.
4. test 제출은 validation에서 선택한 설정으로만 만듭니다.
5. dev 응답은 신뢰도 분석 전까지 학습 정답으로 사용하지 않습니다.
6. LoRA를 시작할 때는 assistant 정답 토큰만 loss에 포함합니다. 기존 실습 노트북처럼 전체 prompt를 labels로 복사하지 않습니다.

