# A4-P1 — Generation-format Diagnostic

Date: 2026-09-21

## 한눈에 보기

| 항목 | 결과 |
|---|---:|
| B0 validation samples | 1,341 |
| exact one-letter output | **1,240 (92.47%)** |
| non-exact output | **101 (7.53%)** |
| empty output | 0 |
| non-exact 중 choice-like | **101 / 101 (100%)** |
| parse failure | 0 |
| scoring 실험 가치 | **있음 — 단, 101개 subset부터** |

> **핵심:** B0의 92.5%는 이미 a/b/c/d 한 글자를 직접 생성한다. 따라서 constrained scoring이 전체 1,341개를 크게 바꿀 가능성은 낮다. 하지만 101개는 첫 token이 '(' 또는 다른 token일 수 있어 label logits가 다른 답을 선택할 여지가 있다.

## 실제 raw-output 패턴

가장 많은 출력:

- `c`: 357
- `a`: 349
- `b`: 314
- `d`: 220
- `(d)`: 46
- `(c)`: 13
- `(a)`: 2

일부는 다음처럼 선택지 뒤에 문자열을 더 생성했다.

- `(c) 다사`
- `(d) 에어`
- `(b) 한술`
- `(d) BRIS`

현재 parser는 이들에서 a/b/c/d를 정상 추출하므로 **parsing 자체는 병목이 아니다.**

## 왜 scoring 가능성이 아직 남아 있나?

Exact output 예:

```text
c
```

greedy generation의 첫 token이 이미 `c`였으므로 a/b/c/d token만 비교해도 c가 이길 가능성이 사실상 확정적이다.

반면 non-exact output:

```text
(d)
```

에서는 첫 생성 token이 `(`일 수 있다.

이 경우 현재 generation은:

```text
전체 vocabulary argmax -> "("
다음 token -> "d"
```

형태일 수 있다.

Constrained choice scoring은 처음부터:

```text
argmax(logit[a], logit[b], logit[c], logit[d])
```

를 선택하기 때문에 현재 parser 결과인 d와 다른 답을 낼 수 있다.

즉 scoring이 바꿀 수 있는 실질적인 target은 우선 **101개의 non-exact subset**이다.

## 전략적 결정

전체 validation에 scoring을 바로 돌리지 않는다.

먼저 두 단계로 간다.

### Step 1 — CPU diagnostic
101개 non-exact subset의 현재 accuracy와 category 구성을 측정한다.

질문:

> non-exact generation이 실제로 더 어려운/불안정한 샘플인가?

### Step 2 — targeted GPU scoring
Step 1 결과가 가치 있으면 101개 subset에서만 constrained label scoring을 실행한다.

이렇게 하면 전체 1,341개 scoring보다 훨씬 적은 비용으로 scoring의 가능성을 판단할 수 있다.

## 포트폴리오 포인트

- tokenizer와 generation format을 연결해 scoring 실험의 실제 적용 범위를 계산
- parse failure와 decision instability를 구분
- 전체 GPU run 전에 7.5% target subset으로 실험 범위를 축소
- 실험 가능성보다 정보 효율을 우선
