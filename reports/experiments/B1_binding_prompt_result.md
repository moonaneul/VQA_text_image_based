# B1 — Selective Binding-aware Prompt Result

Date: 2026-09-21

## 한눈에 보기

| 항목 | B0 | B1 | 변화 |
|---|---:|---:|---:|
| price+phone 전체 | 86.996% | **87.892%** | **+0.90 pp** |
| correct / 223 | 194 | **196** | **+2** |
| phone | 83.72% | **86.05%** | **+2.33 pp / +1** |
| price | 87.78% | **88.33%** | **+0.56 pp / +1** |
| Wrong→Right | - | **2** | |
| Right→Wrong | - | **0** | |
| McNemar exact p | - | **0.50** | |

> **결론:** 방향성은 긍정적이지만, 사전에 고정한 성공 기준인 **net +5**를 충족하지 못했다. B1을 채택하지 않고 다음 단계로 이동한다.

## 해석

좋은 점:
- 2개를 고쳤고 새로 틀린 샘플은 0개
- phone과 price에서 각각 1개씩 회수
- binding-aware instruction이 완전히 무효는 아님

하지만:
- prediction이 바뀐 샘플은 **2 / 223 = 0.90%**뿐
- 전체 validation 환산 순이득은 **2 / 1341 ≈ +0.15 pp**
- exact McNemar p-value는 0.50
- random split은 이미 반복 tuning에 사용됨

따라서 이 결과를 실제 개선으로 채택하기에는 근거가 약하다.

## A2와 흥미로운 공통점

A2 high-resolution에서도:
- phone +1
- price +1

이 나왔다.

Aggregate 결과는 B1과 동일하지만, **같은 샘플을 고쳤는지는 아직 모른다.**

이는 GPU 없이 확인할 수 있는 중요한 진단이다.

- 같은 2개를 고쳤다면: 두 intervention은 같은 쉬운/borderline case에 반응
- 서로 다른 샘플을 고쳤다면: resolution과 binding prompt가 보완적일 가능성

따라서 B2 전에 A2와 B1의 price+phone changed sample을 비교한다.

## 결정

- B1 global/selective routing: **Reject for now**
- Grouped confirmation: **실행하지 않음**
- Prompt 추가 미세조정: **중단**
- 다음 GPU 실험: 바로 하지 않음
- 다음 진단: **A2 vs B1 complementarity on price+phone**

## 다음 단계

1. A2 prediction에서 price+phone 223개만 추출
2. B1과 paired compare
3. 보완성이 없으면 B2 layout-aware grounding
4. 보완성이 있으면 high-res + binding 조합을 작은 subset에서만 검토

## 포트폴리오 포인트

- 사전 성공 기준(net +5)을 실험 후 바꾸지 않음
- +2 / 0 regressions라는 긍정 신호도 과장하지 않음
- grouped split을 불필요하게 소비하지 않음
- 기존 A2 결과와의 complementarity를 먼저 확인해 다음 GPU 실험을 정함
