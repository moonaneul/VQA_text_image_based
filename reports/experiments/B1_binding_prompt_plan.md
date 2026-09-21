# B1 — Selective Binding-aware Prompt Plan

Date: 2026-09-21

## 한눈에 보기

| 항목 | 설정 |
|---|---|
| 목적 | price/phone의 target ↔ number binding 오류 감소 |
| Model | Qwen2.5-VL-3B-Instruct |
| Resolution | standard (~0.8MP) |
| Decision | greedy generation |
| 변경 변수 | **prompt only** |
| Random target | price + phone **223 samples** |
| B0 subset baseline | **194/223 = 87.00%** |
| B0 subset errors | **29** |
| 1차 성공 기준 | **net +5 samples 이상** |
| 전체 validation 환산 | +5 samples ≈ **+0.37 pp** |
| Grouped | Random 성공 시에만 confirmation |

## 왜 price + phone부터 하나?

125개 B0 오답을 전수 리뷰한 결과:

- price errors: **22/22 number_text_binding**
- phone errors: **7/7 number_text_binding**

따라서 이 두 category는 root cause가 가장 깨끗하게 한 가지로 모인다.

scene_text는 OCR, exact-string, understanding, localization 등이 섞여 있어 첫 타깃으로는 덜 적합하다.

## B1 prompt

> 질문에서 묻는 대상이 무엇인지 먼저 특정하세요. 이미지에서 그 대상과 같은 행, 같은 박스, 또는 가장 가까이 직접 연결된 숫자나 문구를 찾으세요. 주변의 다른 상품, 가게, 가격, 전화번호와 섞지 말고 대상과 값의 공간적 연결을 확인하세요. 선택지와 정확히 비교한 뒤 최종 답은 설명 없이 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요.

A1 OCR-aware prompt와 다른 점:

- A1: 글자/숫자를 **잘 읽기**
- B1: 읽은 값이 **어떤 대상에 연결되는지 판단하기**

## 판정

B0 subset과 B1을 동일 ID로 paired comparison한다.

필수 지표:

- wrong → right
- right → wrong
- net gain
- price net
- phone net
- disagreement rate
- exact McNemar p-value

### Adopt gate

**net +5 samples 이상**을 1차 실용 기준으로 둔다.

이는 random 전체 기준 약 +0.37pp에 해당한다.

단, +5를 넘더라도 wins/losses 구조와 category concentration을 확인하고, grouped price+phone subset에서 confirmation 후 채택한다.

## 현실적인 기대 범위

29개 baseline error가 모두 회수되는 것은 비현실적이다.

이번 prompt-only 실험의 실용적 목표는 **3~8 net recovery** 정도를 탐색 범위로 본다. 이는 예측이 아니라 실험 설계를 위한 범위다.

- +0~2: prompt path 종료
- +3~4: 흥미로운 신호, 추가 evidence 필요
- **+5 이상: grouped confirmation**
- 큰 gain: selective routing 후보

## 실패 시

B1이 실패하면 binding을 prompt로만 해결하기 어렵다고 본다.

다음은 **layout-aware grounding**:

1. target crop / region proposal
2. OCR text + bbox coordinates
3. same-row / nearest-neighbor layout representation

그 뒤에도 readable text에서 binding/reasoning 오류가 남으면 QLoRA로 이동한다.
