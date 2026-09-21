# B0 Error Root-cause Review — 125/125 Complete

Date: 2026-09-21

## 한눈에 보기

| Root cause | Errors | Share | 최대 회수 가능폭* |
|---|---:|---:|---:|
| **number_text_binding** | **42** | **33.6%** | **+3.13 pp** |
| ocr_recognition | 21 | 16.8% | +1.57 pp |
| question_understanding | 20 | 16.0% | +1.49 pp |
| visual_spatial_reasoning | 18 | 14.4% | +1.34 pp |
| exact_string_confusion | 13 | 10.4% | +0.97 pp |
| target_localization | 7 | 5.6% | +0.52 pp |
| ambiguity_or_label_issue | 4 | 3.2% | +0.30 pp |

\* 해당 유형을 완전히 해결한다고 가정한 이론적 upper bound. 실제 기대 개선폭은 더 작다.

![B0 root causes](../figures/b0_error_root_causes.svg)

> **핵심 결론:** B0의 가장 큰 병목은 글자를 아예 못 읽는 OCR 문제가 아니라, **읽은 숫자/문구를 질문의 대상과 올바르게 연결하는 binding 문제**다.

## 1. 리뷰 품질

- Total errors: **125**
- Labeled: **125**
- Unlabeled: **0**
- Completion: **100%**
- Human-visible text: **121 / 125**
- Correct text present in image: **83 yes / 38 no / 4 unsure**

사람 기준으로 텍스트 자체는 거의 항상 볼 수 있었다. 따라서 단순한 global resolution 증가만으로 대부분의 오류를 해결할 가능성은 낮다.

## 2. 오류를 큰 가족으로 묶으면

### Reasoning / association
- number_text_binding: 42
- question_understanding: 20
- visual_spatial_reasoning: 18

합계: **80 / 125 = 64.0%**

### Text perception / grounding fidelity
- ocr_recognition: 21
- exact_string_confusion: 13
- target_localization: 7

합계: **41 / 125 = 32.8%**

### Data ambiguity
- ambiguity_or_label_issue: 4

합계: **3.2%**

따라서 현재 모델의 우선 병목은 **perception보다 reasoning/association 쪽이 더 크다.**

## 3. Category와 root cause의 관계

### Price
오답 22개 전부 **number_text_binding**.

즉 가격 숫자를 전혀 못 읽는 문제라기보다, 여러 숫자 중 질문 대상에 해당하는 가격을 연결하는 문제가 핵심이다.

### Phone
오답 7개 전부 **number_text_binding**.

전화번호 문제 역시 OCR recognition 자체보다 대상-번호 연결 문제가 우선이다.

### Scene text
가장 다양한 failure가 섞여 있다.

- OCR recognition: 21
- exact-string confusion: 11
- question understanding: 11
- visual/spatial reasoning: 6
- binding: 5
- localization: 3
- ambiguity: 2

따라서 scene_text 전체에 하나의 처방을 적용하는 것은 비효율적이다.

### Spatial
오답 9개 전부 visual/spatial reasoning.

## 4. 중요한 경고: likely_fix는 독립 근거로 사용하지 않음

수동 리뷰의 `likely_fix` 분포가 failure_type과 정확히 1:1 대응했다.

예:
- binding 42 → bbox_localization 42
- OCR 21 → external_ocr 21
- question understanding 20 → QLoRA 20
- exact string 13 → choice_scoring 13

이 값은 **독립적인 실험 결과가 아니라 리뷰어의 해결책 가설**이다.

특히 `exact_string_confusion → choice_scoring`은 앞선 A4 진단에서 scoring이 핵심 병목이 아니라는 근거와 충돌한다.

따라서 다음 전략은 `likely_fix` 개수보다 **failure_type + 기존 ablation 결과**를 우선해 결정한다.

## 5. 지금까지의 증거를 합치면

1. OCR-aware prompt: 개선 없음
2. full-image 0.8→1.0MP cap: +1 net sample뿐
3. constrained scoring: 출력 형식이 병목이 아님
4. manual root-cause: binding/reasoning 계열이 64%

따라서 다음 실험은 외부 OCR 전체 적용이나 더 높은 global resolution이 아니라 **binding을 직접 겨냥해야 한다.**

## 6. 다음 전략 — B1 Binding-aware selective prompt

첫 타깃은 price + phone이다.

이유:
- 두 category의 B0 오답 29개가 모두 binding으로 분류됨
- 질문 text만으로 안정적으로 routing 가능
- 총 223개 샘플이라 전체 validation보다 훨씬 저렴하게 테스트 가능
- previous OCR-aware prompt는 recognition을 강조했지만, 이번 prompt는 **target ↔ number association**을 직접 강조

가설:

> 질문이 가리키는 대상(상품/서비스/가게)을 먼저 특정하고, 같은 행·같은 박스·가까운 위치에 연결된 숫자만 선택하도록 지시하면 price/phone binding error가 줄어든다.

성공 기준:
- price+phone subset에서 **net +5 samples 이상**을 1차 실용 기준으로 사용
- regressions가 wins를 거의 상쇄하면 reject
- 유의미한 gain이 있으면 grouped price+phone subset에서 확인

## 7. Validation discipline 업데이트

Random validation은 이제 여러 실험 결정을 내리는 데 반복 사용되었으므로 사실상 **development/tuning set**으로 본다.

앞으로:
- random: 아이디어 개발/ablation
- grouped: 후보가 생겼을 때만 confirmation
- QLoRA 단계: random/grouped 모두 train에서 제외

이렇게 해야 반복 튜닝으로 random score에 과적합되는 것을 줄일 수 있다.

## 포트폴리오 포인트

- 125개 오답을 전수 수동 분석
- category와 failure cause를 분리
- OCR-heavy dataset이라는 초기 가설을 더 정교하게 수정
- 단순 OCR 문제가 아니라 number/text binding이 핵심 병목임을 발견
- validation 재사용으로 인한 tuning bias까지 관리
