# A4-P2 — Exact vs Non-exact Accuracy Diagnostic

Date: 2026-09-21

## 한눈에 보기

| 구분 | 샘플 | Accuracy | Errors |
|---|---:|---:|---:|
| 전체 | 1,341 | **90.68%** | 125 |
| exact one-letter | 1,240 | **90.08%** | **123** |
| non-exact | 101 | **98.02%** | **2** |

> **핵심:** 모델 오류 125개 중 **123개(98.4%)**가 이미 정확히 a/b/c/d 한 글자를 출력한 샘플에서 발생했다. 출력 형식이나 parsing은 병목이 아니다.

## 예상과 반대였던 결과

처음에는 `(d)`, `(c) 문자열` 같은 non-exact output이 decision instability의 신호일 수 있다고 봤다.

하지만 실제로는 non-exact 101개 중 99개를 맞혔다.

- non-exact accuracy: **98.02%**
- exact-one-letter accuracy: **90.08%**

즉 non-exact output은 오히려 더 정확했다.

## Category별 non-exact 성능

| Category | Samples | Accuracy |
|---|---:|---:|
| scene_text | 63 | 96.83% |
| other | 26 | 100% |
| phone | 4 | 100% |
| price | 3 | 100% |
| spatial | 4 | 100% |
| menu | 1 | 100% |

non-exact의 유일한 오류 2개는 모두 scene_text였다.

## 전략적 해석

이 결과는 constrained choice scoring의 기대가 낮다는 강한 근거다.

왜냐하면:

1. parser failure가 0이다.
2. non-exact subset이 이미 98% 정확하다.
3. 전체 오류의 98.4%가 exact one-letter output에 있다.
4. exact one-letter 샘플은 greedy generation 첫 token이 이미 a/b/c/d 중 하나였으므로 단순 next-token constrained scoring이 답을 바꿀 가능성이 사실상 없다.

따라서 문제는 **답을 어떤 형식으로 출력하느냐**가 아니라, 이미지/질문을 보고 **어떤 선택지를 옳다고 판단하느냐**에 있다.

## 결정

**A4 constrained choice scoring은 full validation 실험을 생략한다.**

이 실험은 구현할 수 있지만, 현재 evidence상 gain 가능성이 매우 낮다.

GPU 시간을 쓰지 않고 다음 병목 분석으로 이동한다.

## 다음 단계 — Error Root-cause Review

B0 오류 125개를 failure cause로 분류한다.

예상 taxonomy:

- OCR recognition
- target localization
- number/text binding
- exact-string confusion
- question understanding
- visual/spatial reasoning
- ambiguity / label issue

이제 질문은:

> "모델이 왜 틀렸는가?"

이다.

이 분류 결과에 따라 다음 전략을 선택한다.

- recognition 많음 → crop / tiling / OCR
- localization 많음 → region-focused inference
- binding/reasoning 많음 → QLoRA
- mixed → selective routing

## 포트폴리오 포인트

- 출력 형식 문제와 실제 reasoning error를 분리
- 예상과 반대 결과도 그대로 기록
- 구현 가능하다는 이유만으로 scoring을 밀어붙이지 않음
- 125개 오류의 root cause 분석으로 다음 전략을 데이터 기반으로 선택
