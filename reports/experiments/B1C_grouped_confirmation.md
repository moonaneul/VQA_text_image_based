# B1C Grouped Confirmation

Date: 2026-09-21

## 한눈에 보기

| 항목 | B0 grouped | B1C grouped | 변화 |
|---|---:|---:|---:|
| price+phone 전체 | 202/223 = 90.58% | **205/223 = 91.93%** | **+3 / +1.35 pp** |
| Wrong→Right | - | **6** | |
| Right→Wrong | - | **3** | |
| McNemar exact p | - | **0.508** | |
| phone | 36/41 = 87.80% | **35/41 = 85.37%** | **-1 / -2.44 pp** |
| price | 166/182 = 91.21% | **170/182 = 93.41%** | **+4 / +2.20 pp** |

> **결론:** 사전 gate 기준으로는 **directional replication(204–205)**에 해당한다. 하지만 phone이 regress했고 right→wrong이 3개라 stability warning 조건도 동시에 충족했다.

## 사전 gate와 대조

사전 기준:
- 206+ = strong replication
- 204–205 = directional replication
- 203 이하 = not confirmed
- right→wrong >= 2 = stability warning
- category 하나만 gain이고 다른 category regress = selective rule 재검토

실제:
- 205 correct → **directional replication**
- right→wrong 3 → **stability warning**
- price +4, phone -1 → **category heterogeneity**

따라서 price+phone 전체에 B1C를 적용하는 routing rule은 채택하지 않는다.

## Random + Grouped를 함께 보면

### Price
- Random: **+2**, regressions 0
- Grouped: **+4**, wins 6 / losses 2
- 방향: **두 split 모두 positive**

### Phone
- Random: **+2**, regressions 0
- Grouped: **-1**, wins 0 / losses 1
- 방향: **불일치**

따라서 현재 evidence는 **price에만 일관되게 긍정적**이고 phone에는 불안정하다.

## Price-only router의 관측 성능

B1C를 price에만 적용하고 phone은 B0를 유지한다고 가정하면:

### Random price+phone subset
- price B1C: 160/180
- phone B0: 36/43
- hybrid: **196/223 = 87.89%**
- B0 대비 **+2**

### Grouped price+phone subset
- price B1C: 170/182
- phone B0: 36/41
- hybrid: **206/223 = 92.38%**
- B0 대비 **+4**

Full validation으로 환산하면:
- Random: 1216 → **1218 / 1341 = 90.83%** (+0.15 pp)
- Grouped: 1232 → **1236 / 1343 = 92.03%** (+0.30 pp)

하지만 이 price-only rule은 grouped category 결과를 본 뒤 정제되었으므로 **최종 확인 없이 채택하면 confirmation set까지 tuning에 사용한 셈**이 된다.

## 다음: fresh audit holdout

새로운 아이디어를 더 튜닝하기 전에, train에서:
- random val ID 제외
- grouped val ID 제외
- 그 뒤 남은 데이터에서 fresh price/phone audit subset 생성

하여 price-only 효과를 독립적으로 확인한다.

권장 audit:
- price 200
- phone 100
- seed 20260921

같은 audit subset에:
1. B0 standard + direct
2. B1C high + binding-aware

를 각각 한 번만 실행한다.

### Audit 판정 원칙

Price:
- net positive + regressions manageable → price-only router 후보 유지
- net <= 0 → routing branch 종료

Phone:
- 별도로 기록만 하고, grouped에서 이미 regress했으므로 강한 독립 gain이 없으면 B0 유지

이 audit은 leaderboard를 쓰지 않고도 validation tuning bias를 줄이기 위한 마지막 confirmation이다.
