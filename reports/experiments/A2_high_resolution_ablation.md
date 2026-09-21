# A2 — High-resolution Cap Ablation

Date: 2026-09-21

## 한눈에 보기

| 항목 | 결과 |
|---|---|
| 기준 모델 | Qwen2.5-VL-3B-Instruct |
| 이전 best | B0: direct + standard (~0.8MP) |
| 이번에 바꾼 것 | **max resolution cap만 ~0.8MP → ~1.0MP** |
| Random accuracy | **90.68% → 90.75% (+0.07 pp)** |
| 정답 수 | **1,216 → 1,217 (+1)** |
| OCR-heavy accuracy | **89.91% → 90.02% (+0.11 pp)** |
| Runtime | **792.3s → 814.0s (+2.74%)** |
| 결론 | **Global default로는 보류/Reject** |
| 다음 행동 | **B0 vs A2 sample-level paired comparison** |

> **결론:** high 설정은 1문제만 더 맞혔다. 비용은 늘었지만 개선폭이 너무 작아 현재 근거만으로 채택할 수 없다.

## 왜 이 실험을 했나?

B0와 A1 결과에서 단순한 OCR-aware prompt는 도움이 되지 않았다.

다음 가설은 이미지 정보량이다.

> 작은 간판·전화번호·가격표를 틀리는 이유가 downsampling 때문이라면, Qwen의 최대 이미지 해상도 cap을 높였을 때 OCR-heavy 성능이 개선될 수 있다.

## 무엇을 바꿨나?

**resolution 한 가지 변수만** 변경했다.

```text
B0: direct + standard
max_pixels = 802,816 (~0.8MP)

A2: direct + high
max_pixels = 1,003,520 (~1.0MP)
```

고정:
- Qwen2.5-VL-3B-Instruct
- direct prompt
- no quantization
- free generation
- seed 20260921
- same random validation

## 결과

| Category | B0 standard | A2 high | Δ |
|---|---:|---:|---:|
| Overall | **90.68%** | **90.75%** | **+0.07 pp** |
| OCR-heavy | 89.91% | 90.02% | +0.11 pp |
| scene_text | 91.17% | 91.02% | -0.15 pp |
| price | 87.78% | 88.33% | +0.56 pp |
| phone | 83.72% | 86.05% | +2.33 pp |
| menu | 83.87% | 83.87% | 0.00 pp |
| spatial | 84.21% | 84.21% | 0.00 pp |
| other | 93.39% | 93.39% | 0.00 pp |

집계상:
- scene_text: 609 → 608 (**-1**)
- price: 158 → 159 (**+1**)
- phone: 36 → 37 (**+1**)
- 전체: 1,216 → 1,217 (**+1**)

![A2 delta vs B0](../figures/a2_delta_vs_b0.svg)

## 중요한 해석: high cap은 '업스케일'이 아니다

EDA에서 이미지 중앙값은 약 **0.69MP**였다.

standard cap은 약 **0.80MP**, high cap은 약 **1.00MP**다.

따라서 high 설정은 모든 이미지를 1.0MP로 키우는 실험이 아니다. 주로 standard cap보다 큰 이미지를 덜 축소하는 효과가 있다.

즉 A2의 결과를:

> "해상도는 중요하지 않다"

라고 해석하면 안 된다.

정확한 해석은:

> **Qwen의 max_pixels cap을 0.8MP에서 1.0MP로 올리는 것만으로는 현재 validation에서 의미 있는 전체 개선을 만들지 못했다.**

작은 글자 자체를 실제로 확대하는 **crop / tiling / explicit upscaling**은 별도의 가설이다.

## 비용 대비 효과

- Accuracy: **+0.07 pp**
- 추가 정답: **+1 / 1,341**
- Runtime: **+2.74%**
- Throughput: 약 **-2.67%**

현재 단계에서는 global default를 high로 바꾸기엔 근거가 약하다.

## 왜 grouped를 바로 돌리지 않는가?

미리 정한 규칙은 random에서 의미 있는 이득이 있을 때 grouped로 확인하는 것이었다.

A2의 개선은 **1문제, +0.07 pp**뿐이므로 지금 grouped 전체 추론에 GPU 시간을 쓰는 정보 가치가 낮다.

대신 먼저 이미 생성된 predictions를 sample-level로 비교한다.

## 다음 행동 — GPU를 쓰지 않는 paired diagnostic

`scripts/compare_runs.py`로 다음을 확인한다.

- B0 wrong → A2 right
- B0 right → A2 wrong
- prediction disagreement rate
- 어떤 category에서 실제 prediction이 바뀌었는지

Aggregate score는 +1이지만, 내부적으로 많은 샘플이 서로 뒤집혔을 수도 있다.

그 경우 high-resolution path가 ensemble/routing에 보완적인지 검토할 가치가 생긴다.

반대로 거의 동일한 prediction이라면 high cap 실험은 종료한다.

## 포트폴리오 포인트

- 단순 score가 아니라 비용 대비 gain을 평가
- high resolution의 의미를 "업스케일"과 구분
- +0.07 pp를 과장하지 않고 보류
- 다음 GPU 실험 전에 sample-level diagnostic으로 정보 효율을 높임
