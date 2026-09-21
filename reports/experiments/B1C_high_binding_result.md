# B1C — High Resolution + Binding-aware Prompt

Date: 2026-09-21

## 한눈에 보기

| Setup | Correct / 223 | Accuracy | vs B0 |
|---|---:|---:|---:|
| B0 standard + direct | 194 | 87.00% | baseline |
| A2 high + direct | 196 | 87.89% | +2 |
| B1 standard + binding | 196 | 87.89% | +2 |
| **B1C high + binding** | **198** | **88.79%** | **+4** |

> **핵심:** B1C가 사전에 정한 gate인 **198/223**을 정확히 충족했다. A2와 B1이 각각 고친 서로 다른 2개씩을 모두 보존했고, B0 대비 새 regression은 0개였다.

## Paired comparison

### B0 → B1C
- Wrong→Right: **4**
- Right→Wrong: **0**
- Net: **+4**
- Delta: **+1.79 pp on price+phone subset**
- Exact McNemar p-value: **0.125**

### A2 → B1C
- Wrong→Right: **2**
- Right→Wrong: **0**
- Net: **+2**

### B1 → B1C
- Wrong→Right: **2**
- Right→Wrong: **0**
- Net: **+2**

## 중요한 구조적 발견

A2 vs B1 진단에서:
- A2-only correct = 2
- B1-only correct = 2
- both correct = 194

즉 A2와 B1의 oracle union은 **198 correct**였다.

B1C는:
- A2의 정답을 하나도 잃지 않고 +2
- B1의 정답을 하나도 잃지 않고 +2
- 최종 **198 correct**

따라서 B1C는 이 validation subset에서 **A2와 B1의 서로 다른 4개 gain을 정확히 모두 결합했다.**

이는 단순히 점수가 +4라는 것보다 더 중요한 evidence다. 두 intervention이 서로 방해하지 않고 결합될 수 있다는 뜻이다.

## Category

### Phone
- B0: 36/43 = 83.72%
- B1C: **38/43 = 88.37%**
- net +2
- regressions 0

### Price
- B0: 158/180 = 87.78%
- B1C: **160/180 = 88.89%**
- net +2
- regressions 0

두 category 모두 같은 방향으로 개선됐다.

## 통계 해석

Exact McNemar p=0.125이므로 0.05 기준의 통계적 유의성을 확보했다고 말할 수는 없다.

또한 random split은 여러 번 tuning에 사용되었으므로 현재 결과는 **development evidence**다.

따라서 이 결과를 최종 채택 근거로 쓰지 않고 grouped split에서 confirmation한다.

## Grouped confirmation gate — 사전 고정

Grouped B0 price+phone baseline:
- phone: 36/41
- price: 166/182
- total: **202/223 = 90.58%**

Random에서 B1C gain은 +4/223였다.

Grouped confirmation 판정:

| Grouped B1C | 해석 |
|---|---|
| **206+ correct (net +4 이상)** | strong replication |
| **204–205 correct (net +2~3)** | directional replication |
| **203 이하 (net +1 이하)** | not confirmed |

추가 조건:
- right→wrong이 2개 이상이면 stability warning
- category 하나에서만 gain이 몰리고 다른 category가 regress하면 selective rule 재검토

## 다음

Grouped price+phone 223개에 **high + binding-aware**를 딱 한 번 실행한다.

그 결과까지 확인한 뒤:
- confirmation 성공 → selective routing 후보로 승격
- confirmation 실패 → prompt/high-resolution branch 종료, B2 layout-aware grounding으로 이동
