[object Object]

## Root-cause review 완료: 125 / 125

![B0 root causes](figures/b0_error_root_causes.svg)

가장 큰 병목은 **number/text binding**이다.

- binding: **42 (33.6%)**
- OCR recognition: 21
- question understanding: 20
- visual/spatial reasoning: 18
- exact-string confusion: 13
- localization: 7
- ambiguity/label: 4

특히:
- price 오답 22개 → 전부 binding
- phone 오답 7개 → 전부 binding

즉 **숫자를 못 읽는 것보다, 어떤 숫자가 질문의 대상에 붙어 있는지 연결하는 문제가 더 크다.**

### 다음 실험

**B1 — price+phone selective binding-aware prompt**

전체 validation이 아니라 price+phone 223개만 먼저 테스트한다.

목표:
- target entity를 먼저 특정
- 같은 행/박스/가까운 위치의 숫자를 연결
- 주변 다른 가격/번호와 혼동하지 않기

성공하면 grouped subset에서 확인한다.

### Validation 원칙 업데이트

Random split은 반복 실험에 사용되었으므로 이제 **development set**으로 취급한다.

Grouped split은 candidate가 생겼을 때만 확인하는 **confirmation set**으로 아껴 쓴다.
