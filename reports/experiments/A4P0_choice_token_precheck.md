# A4-P0 — Choice-token Pre-check

Date: 2026-09-21

## 한눈에 보기

| 항목 | 결과 |
|---|---|
| 목적 | constrained choice scoring 구현 가능성 확인 |
| Model tokenizer | Qwen2.5-VL-3B-Instruct |
| `a` | token id **64**, 1 token |
| `b` | token id **65**, 1 token |
| `c` | token id **66**, 1 token |
| `d` | token id **67**, 1 token |
| Assistant context에서 suffix | 모두 **1 token** |
| 구현 난이도 | 낮음 |
| 다음 행동 | **현재 generation이 이미 exact one-letter인지 먼저 확인** |

> **결론:** a/b/c/d는 assistant answer 위치에서 각각 단일 token이다. 따라서 next-token logits만으로 4-choice constrained scoring을 구현할 수 있다.

## 왜 이 확인이 필요했나?

Multiple-choice scoring을 구현할 때 label이 여러 token이면 각 후보 sequence의 conditional log-probability를 따로 계산해야 한다.

하지만 이번 tokenizer에서는:

```text
a -> [64]
b -> [65]
c -> [66]
d -> [67]
```

이고 실제 assistant-generation prefix 뒤에서도 동일하게 한 token이다.

따라서 한 번의 forward pass에서 마지막 위치 logits의

```text
logit[64], logit[65], logit[66], logit[67]
```

만 비교하면 된다.

## 중요한 전략적 포인트

이 결과가 곧바로 "scoring을 전체 validation에 돌려야 한다"는 뜻은 아니다.

현재 generation은 greedy decoding(`do_sample=False`)이다.

만약 B0의 raw output이 모든 sample에서 정확히 `a`, `b`, `c`, `d` 한 글자라면:

1. greedy generation의 첫 token이 이미 a/b/c/d 중 하나였고
2. 그 token은 전체 vocabulary에서 argmax였으며
3. 따라서 a/b/c/d 네 token만 비교해도 같은 token이 argmax다.

즉 **그 경우 constrained label scoring은 prediction을 바꿀 수 없다.**

그래서 다음 단계는 GPU inference가 아니라 기존 `predictions.csv`의 raw output 형식을 확인하는 것이다.

## 의사결정

- exact one-letter output = 100% → A4 full scoring run 생략
- exact one-letter가 거의 100% → non-exact subset만 우선 진단
- 설명/괄호/기타 prefix가 의미 있게 존재 → full constrained scoring 실험 가치 있음

이 검사를 통해 불필요한 13분짜리 GPU run을 피할 수 있다.
