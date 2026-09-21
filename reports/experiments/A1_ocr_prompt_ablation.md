# A1 — OCR-aware Prompt Ablation

Date: 2026-09-21

## 한눈에 보기

| 항목 | B0 | A1 | 변화 |
|---|---:|---:|---:|
| 모델 | Qwen2.5-VL-3B-Instruct | 동일 | - |
| 해상도 | standard (~0.8MP) | 동일 | - |
| Prompt | direct | **ocr_deliberate** | **이것만 변경** |
| Random accuracy | **90.68%** | **90.68%** | **0.00 pp** |
| OCR-heavy accuracy | **89.91%** | **89.70%** | **-0.22 pp** |
| scene_text | 91.17% | 90.87% | **-0.30 pp** |
| spatial | 84.21% | 87.72% | **+3.51 pp** |
| Runtime | 792.3s | 805.1s | **+1.6% slower** |
| Parse failure | 0 | 0 | 변화 없음 |

> **결론:** OCR을 더 꼼꼼히 보라고 지시한 prompt는 전체 accuracy를 올리지 못했고, 정작 OCR-heavy accuracy는 소폭 하락했다. 따라서 A1 prompt는 채택하지 않고 B0의 direct prompt를 유지한다.

![A1 delta vs B0](../figures/a1_delta_vs_b0.svg)

---

## 1. 왜 이 실험을 했나?

B0에서 random validation 오답 125개 중 93개(74.4%)가 scene_text / price / phone / menu에 몰려 있었다.

그래서 첫 번째 가설은 단순했다.

> **가설:** 모델이 이미지 속 글자와 숫자를 충분히 주의 깊게 보지 않아서 틀린다면, OCR을 강조한 prompt만으로도 OCR-heavy 정확도가 올라갈 것이다.

이 가설을 확인하기 위해 **prompt만 바꾸고 나머지는 모두 고정**했다.

## 2. 실험 조건

### 고정한 것
- Model: Qwen2.5-VL-3B-Instruct
- Resolution: standard
  - min_pixels: 200,704
  - max_pixels: 802,816
- Quantization: none
- Decision: free generation
- Seed: 20260921
- Validation: val_random.csv
- Samples: 1,341

### 바꾼 것
B0:

```
이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요.
최종 답은 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요.
```

A1:

```
이미지 속 글자와 숫자를 확대해 읽듯이 세밀하게 확인한 뒤 질문에 답하세요.
간판, 메뉴, 가격, 전화번호, 날짜의 정확한 표기를 선택지와 비교하세요.
최종 답은 설명 없이 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요.
```

## 3. 결과

### 전체

- B0: 1,216 / 1,341 correct = **90.68%**
- A1: 1,216 / 1,341 correct = **90.68%**
- 변화: **0.00 percentage points**

즉, 전체 정답 개수는 하나도 늘지 않았다.

### 유형별 순증감

| 유형 | B0 correct | A1 correct | 순증감 |
|---|---:|---:|---:|
| scene_text | 609 | 607 | **-2** |
| price | 158 | 158 | 0 |
| phone | 36 | 36 | 0 |
| menu | 26 | 26 | 0 |
| spatial | 48 | 50 | **+2** |
| other | 325 | 325 | 0 |
| count | 10 | 10 | 0 |
| color | 4 | 4 | 0 |

OCR-heavy 합계:

- B0: 829 / 922 = **89.91%**
- A1: 827 / 922 = **89.70%**
- 변화: **-0.22 pp**

### 속도

- B0: 792.3s
- A1: 805.1s
- A1이 약 **1.6% 느림**

속도 차이는 크지 않지만 성능 이득도 없기 때문에 A1을 유지할 이유가 없다.

## 4. 무엇을 배웠나?

### 측정된 사실
- OCR-aware prompt는 전체 accuracy를 개선하지 않았다.
- OCR-heavy accuracy는 오히려 0.22 pp 낮아졌다.
- scene_text에서 순수하게 2문제를 잃었다.
- spatial에서는 2문제를 얻었다.
- parse failure는 여전히 0이다.

### 해석
이 결과는 **“모델이 OCR을 더 꼼꼼히 보라고 지시받지 않아서 틀린다”**는 단순 가설을 지지하지 않는다.

다음 가능성이 더 중요해졌다.

1. 작은 글자가 실제 입력 해상도에서 충분히 보이지 않는다.
2. 글자는 보이지만 모델이 다른 간판/영역을 선택한다.
3. 텍스트는 읽지만 가격·상품·대상을 잘못 연결한다.
4. prompt보다 image detail 자체가 병목이다.

즉 다음 실험은 prompt를 더 길게 만드는 것이 아니라 **이미지 해상도**를 검증하는 것이 정보 가치가 높다.

## 5. 결정

### A1 판정: **Reject**

A1을 최종 pipeline에 넣지 않는다.

**현재 best 설정은 여전히 B0다.**

```
Qwen2.5-VL-3B-Instruct
+ direct prompt
+ standard ~0.8MP
+ free generation
```

Random accuracy: **90.68%**

Grouped accuracy: **91.73%**

## 6. 다음 실험

### A2 — High-resolution ablation

이번에도 변수 하나만 변경한다.

```
standard max ~0.8MP
        ↓
high max ~1.0MP
```

고정:
- Qwen2.5-VL-3B-Instruct
- **direct prompt**
- no quantization
- free generation
- seed 20260921
- random validation

### 가설

> 작은 장면 텍스트가 주요 병목이라면 prompt를 바꾸는 것보다 입력 이미지의 세부 정보를 더 보존하는 것이 scene_text / price / phone / menu 성능을 개선할 것이다.

### 성공 기준

우선 다음을 본다.

- overall: 약 **+0.7 pp 이상**이면 강한 채택 후보
- 또는 OCR-heavy: 약 **+1.5 pp 이상**이면 의미 있는 후보
- 작은 차이는 paired prediction 비교 후 판단

## 7. 아직 확인할 것

A1과 B0의 aggregate accuracy만으로는 실제로 어떤 **개별 샘플**이 바뀌었는지 알 수 없다.

따라서 로컬 prediction CSV가 있으면 다음 paired 비교를 추가로 실행한다.

```powershell
.\baseline\Scripts\python.exe .\scripts\compare_runs.py `
  --baseline .\output\baseline\B0_random_direct_standard\predictions.csv `
  --candidate .\output\baseline\A1_random_ocr_standard\predictions.csv `
  --output .\output\comparisons\B0_vs_A1.json
```

이 결과에서 wrong→right / right→wrong를 확인하면 A1이 어떤 문제를 바꿨는지 더 정확히 설명할 수 있다.

## 8. 포트폴리오 포인트

이 실험에서 중요한 것은 “prompt engineering을 했다”가 아니다.

- OCR-heavy 오답 분석에서 출발해 명확한 prompt 가설을 세웠다.
- 한 번에 하나의 변수만 바꿨다.
- 효과가 없었던 접근을 **채택하지 않았다**.
- 실패 결과를 다음 해상도 실험의 근거로 사용했다.

즉 성능 개선뿐 아니라 **실패한 실험도 의사결정 근거로 남기는 ML 실험 설계**를 보여준다.
