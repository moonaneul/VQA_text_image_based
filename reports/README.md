# Modeling Dashboard

> **현재 reference:** Qwen2.5-VL-3B-Instruct + direct prompt + standard (~0.8MP). Random 90.68%, Grouped 91.73%.

## 지금까지의 모델 변화

| Stage | 변경 | Random 결과 | 판단 |
|---|---|---:|---|
| **B0** | direct + ~0.8MP | **90.68%** | Reference |
| A1 | OCR-aware prompt | 90.68% | Reject |
| A2 | max cap ~1.0MP | 90.75% (+1 sample) | Reject as global |
| A2-D | B0 vs A2 paired | 5 wins / 4 losses | High/ensemble 종료 |
| A4 | constrained scoring 사전진단 | 오류 98.4%가 clean one-letter | Full run 생략 |
| **B1** | price+phone binding-aware prompt | TBD | **Next** |

## B0 오답 125개 전수 분석

![B0 root causes](figures/b0_error_root_causes.svg)

| Root cause | Errors | Share |
|---|---:|---:|
| **number_text_binding** | **42** | **33.6%** |
| OCR recognition | 21 | 16.8% |
| question understanding | 20 | 16.0% |
| visual/spatial reasoning | 18 | 14.4% |
| exact-string confusion | 13 | 10.4% |
| target localization | 7 | 5.6% |
| ambiguity/label | 4 | 3.2% |

Reasoning/association 계열(binding + question understanding + spatial)이 **64%**다.

### 가장 중요한 발견

- price 오답 **22/22 = binding**
- phone 오답 **7/7 = binding**

즉 현재 첫 병목은 “숫자를 못 읽는다”보다 **“질문의 대상과 올바른 숫자를 연결한다”**에 가깝다.

## 다음 실험: B1

```text
val_random
   ↓
price + phone만 추출 (223)
   ↓
B0 direct            B1 binding-aware
   │                       │
   └──── paired comparison ┘
              ↓
      wrong→right / right→wrong
      net gain / McNemar
```

B0 subset:
- **194 / 223 correct**
- **87.00%**
- errors **29**

B1은 model, resolution, seed, decoding을 그대로 두고 **prompt만 변경**한다.

### 성공 기준

**net +5 samples 이상**

+5는 전체 validation 기준 약 **+0.37pp**에 해당한다.

- +0~2 → Reject
- +3~4 → 신호는 있으나 추가 근거 필요
- **+5 이상 → grouped price+phone confirmation**

## 실패하면

prompt를 계속 미세조정하지 않는다.

다음은 **B2 layout-aware grounding**:
- target crop / zoom
- OCR + bbox
- same-row / nearest-value 구조

그 이후에도 binding/reasoning이 남으면 QLoRA를 검토한다.

## Validation 원칙

Random split은 반복 실험에 사용되었으므로 **development/tuning set**으로 취급한다.

Grouped split은 candidate가 생겼을 때만 쓰는 **confirmation set**으로 아껴 둔다.


## B1 결과

| Metric | Result |
|---|---:|
| B0 price+phone | 194 / 223 (**87.00%**) |
| B1 price+phone | 196 / 223 (**87.89%**) |
| Wrong→Right | **2** |
| Right→Wrong | **0** |
| Net gain | **+2** |
| McNemar exact p | **0.50** |

사전 gate였던 **net +5**를 넘지 못했다.

따라서:
- B1 채택 안 함
- grouped 확인 안 함
- prompt 미세조정 중단

다만 A2 high-resolution도 phone +1, price +1이었다.

### 다음은 GPU 없이 확인

**A2와 B1이 같은 2개를 고쳤는지 확인한다.**

- same fixes → prompt/high-res 둘 다 종료 → B2 layout grounding
- different fixes → complementarity 가능 → 작은 조합 실험 검토
