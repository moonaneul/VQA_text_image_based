# SSAFY 16기 2회차 VQA Challenge

> **프로젝트 상태: 종료 (2026-09-28)**  
> 이미지 + 한국어 질문 + 4지선다(a~d) 정답을 맞히는 VQA 경진대회 프로젝트입니다.

이 저장소는 더 이상 실험을 진행하지 않습니다.  
나중에 다시 볼 때는 **[최종 정리](reports/FINAL_REPORT.md)** 를 먼저 읽으면 됩니다.

## 최종 결과

- 최종 기준 Public LB: **0.95263**
- 최종 방식: 여러 VLM의 4지선다 확률을 **weighted log probability ensemble**로 결합
- 최종 로컬 제출 파일:
  `output/ENSEMBLE_TEAM722_PERM1805_B3P3_0475_FT8B_05.csv`

최종 ensemble:

| Component | Weight |
|---|---:|
| TEAM-C896 | 72.20% |
| Permutation-8B | 18.05% |
| B3P3 | 4.75% |
| FT Qwen3-VL-8B 640 P0 | 5.00% |

> Public LB는 공개 리더보드 관측값입니다. test 정답은 제공되지 않았으므로 item-level 정답 여부는 알 수 없습니다.

## 가장 중요한 결론

처음에는 OCR-heavy 데이터라서 "글자를 더 잘 읽는 것"이 핵심이라고 생각했지만, 실제 오답 125개를 전수 분류한 결과는 달랐습니다.

- number/text binding: **33.6%**
- OCR recognition: **16.8%**
- question understanding: **16.0%**
- visual/spatial reasoning: **14.4%**
- exact-string confusion: **10.4%**
- target localization: **5.6%**
- ambiguity/label: **3.2%**

특히 binding + question understanding + spatial reasoning을 합친 reasoning/association 계열이 **64%**였습니다.

즉 이 프로젝트에서 가장 큰 학습은:

> **OCR이 많이 등장하는 데이터셋이라고 해서 OCR 자체가 가장 큰 병목인 것은 아니다.  
> 실제 오류를 직접 분류해야 무엇을 개선해야 하는지 알 수 있다.**

## 실제로 효과가 있었던 것

1. **Qwen3-VL-8B QLoRA**
   - base 947/1007 = 94.04%
   - fine-tuned 952/1007 = 94.54%
   - rescue 22 / regression 17 / net +5

2. **서로 다른 모델을 작은 비중으로 ensemble**
   - TEAM-C896 + Permutation-8B: Public LB **0.95144**
   - + B3P3 5%: **0.95204**
   - + FT-8B 5%: **0.95263**

3. **작은 변화도 paired comparison으로 검증**
   - 단순 accuracy만 보지 않고
   - wrong→right(rescue)
   - right→wrong(regression)
   - disagreement
   - category별 변화
   를 함께 확인했습니다.

## 효과가 없거나 채택하지 않은 것

- OCR 강조 prompt
- global high-resolution만 적용
- output-format / constrained choice scoring
- price/phone prompt routing
- tiled crop/zoom
- FT-8B P0+P1 replacement
- FT-8B 896-profile replacement
- MiniCPM-V-4.5
- InternVL3-8B
- Step3-VL-10B

실패 실험을 억지로 살리지 않고, 미리 정한 gate를 넘지 못하면 종료했습니다.

## 나중에 다시 볼 문서

읽는 순서:

1. **[reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)** — 전체 결과와 공부용 정리
2. **[reports/model_evolution.md](reports/model_evolution.md)** — 모델 변화만 빠르게 보기
3. **[experiments/README.md](experiments/README.md)** — 실험을 어떻게 비교했는지
4. `reports/experiments/` — 세부 raw experiment archive

## 폴더

```text
data/                 원본 데이터
splits/               validation / holdout split
scripts/              학습·평가·비교 스크립트
experiments/          실험 기록 규칙과 experiments.csv
reports/FINAL_REPORT.md
                      최종 정리
reports/experiments/  상세 실험 archive
output/               로컬 실행 결과 및 제출 파일 (Git 비추적 항목 포함)
```

## 프로젝트에서 익힌 핵심 개념

- VLM / VQA
- zero-shot baseline
- QLoRA / LoRA
- 4-bit NF4 quantization
- choice-token scoring
- paired evaluation
- McNemar test
- validation / confirmation / holdout 분리
- error taxonomy
- ensemble diversity
- weighted log probability ensemble
- Public LB overfitting 방지

세부 설명은 [최종 정리](reports/FINAL_REPORT.md)에 남겨 두었습니다.
