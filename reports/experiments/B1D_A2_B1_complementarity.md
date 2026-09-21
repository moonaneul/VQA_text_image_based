# B1-D — A2 vs B1 Complementarity Diagnostic

Date: 2026-09-21

## 한눈에 보기

| 항목 | A2 high/direct | B1 standard/binding |
|---|---:|---:|
| price+phone accuracy | **87.89%** | **87.89%** |
| correct / 223 | **196** | **196** |
| 서로 다른 prediction | **4 (1.79%)** | |
| A2만 정답 | **2** | |
| B1만 정답 | **2** | |
| 둘 다 정답 | **194** | |
| 둘 다 오답 | **25** | |
| McNemar exact p | **1.00** | |

> **핵심:** A2와 B1은 점수는 같지만 **서로 다른 2개씩을 고쳤다.** 두 intervention의 개선 sample이 겹치지 않는다.

## B0까지 함께 보면

B0 price+phone:
- 194 / 223 correct
- 29 errors

A2 vs B0:
- +2
- regressions 0

B1 vs B0:
- +2
- regressions 0

A2 vs B1:
- A2-only correct 2
- B1-only correct 2
- both correct 194

따라서 A2와 B1이 B0에서 회수한 **각 2개가 서로 다른 sample**이라는 구조가 성립한다.

## Oracle upper bound

A2와 B1 중 정답인 쪽을 완벽하게 고를 수 있다고 가정하면:

- 198 / 223 correct
- **88.79%**
- B0 대비 +4 samples = **+1.79 pp on subset**
- full random validation 기준 잠재 +4/1341 = **+0.30 pp**

이는 실제 routing 성능이 아니라 **두 intervention의 보완성 upper bound**다.

## 다음 실험이 명확해짐: 2×2 factorial의 마지막 칸

현재까지:

| Resolution | Prompt | Result |
|---|---|---:|
| standard | direct | B0 = 194/223 |
| high | direct | A2 = 196/223 |
| standard | binding | B1 = 196/223 |
| **high** | **binding** | **미실험** |

따라서 다음은 새로운 아이디어가 아니라 **이미 시작한 2×2 실험의 마지막 cell**을 채우는 것이다.

이 실험은 두 효과가:
- additive인지
- redundant인지
- 서로 방해하는지

를 가장 싸게 확인한다.

## B1C success gate

Target: price+phone 223 samples.

- **198+ correct**: 두 intervention이 거의 additive → grouped confirmation 가치 있음
- **197 correct**: 약한 상호보완 → 추가 근거 없이는 채택 보류
- **196 이하**: 결합 이득 없음 → prompt/high-res branch 종료

추가로 B0 대비 right→wrong이 생기는지 반드시 확인한다.

## 다음

B1C:
- Qwen2.5-VL-3B-Instruct
- prompt = binding_aware
- resolution = high
- same seed
- same 223 sample subset

그 뒤 B0/A2/B1 세 기준과 paired comparison한다.
