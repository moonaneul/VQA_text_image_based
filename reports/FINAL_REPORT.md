# VQA Challenge — Final Report & Study Notes

> **Status: CLOSED**  
> 마지막 업데이트: 2026-09-28  
> 이 문서는 이 프로젝트의 최종 canonical summary입니다.  
> 세부 실험 로그는 `reports/experiments/`에 archive로 남겨 두되, 나중에 복습할 때는 이 문서만 먼저 보면 됩니다.

---

## 1. 문제 한 줄 요약

이미지와 한국어 질문, 네 개의 선택지(a~d)가 주어졌을 때 정답 하나를 선택하는 **4-choice VQA** 문제입니다.

데이터 규모:
- train: **6,714**
- dev: **2,683**
- test: **6,714**

평가 지표:
- **Accuracy**

문제 특성:
- 간판
- 메뉴판
- 가격
- 전화번호
- 포스터
- 날짜
- 작은 장면 텍스트

처럼 **scene text를 읽어야 하는 문제 비중이 높았습니다.**

---

## 2. 최종 결과

### 최종 Public LB

**0.95263**

최종 ensemble:

| Component | Weight | 역할 |
|---|---:|---|
| TEAM-C896 | 72.20% | 주력 fine-tuned Qwen3-VL-4B 계열 |
| Permutation-8B | 18.05% | Qwen3-VL-8B permutation 기반 보완 |
| B3P3 | 4.75% | Qwen2.5-VL-3B QLoRA diversity |
| FT Qwen3-VL-8B 640 P0 | 5.00% | 로컬 fine-tuned 8B diversity |

결합 방식:
- 각 모델의 a/b/c/d 확률을 사용
- **weighted log probability** 합산
- 가장 큰 최종 score의 choice를 선택

최종 로컬 제출 파일:
`output/ENSEMBLE_TEAM722_PERM1805_B3P3_0475_FT8B_05.csv`

### 점수 변화

| 단계 | Public LB |
|---|---:|
| D-PERM | 0.95055 |
| TEAM-C896 80% + Permutation-8B 20% | 0.95144 |
| + B3P3 5% | 0.95204 |
| + FT-8B 5% | **0.95263** |

중요:
- 이 값은 **Public leaderboard 관측값**
- test gold는 없으므로 어떤 test item이 실제로 개선됐는지는 알 수 없음
- 마지막에는 dense weight search를 하지 않음

---

## 3. 처음 baseline

초기 로컬 baseline:

**Qwen2.5-VL-3B-Instruct**
- direct prompt
- standard resolution
- random validation: **90.68%**
- grouped validation: **91.73%**

처음 가설은 단순했습니다.

> "OCR-heavy 데이터니까 OCR을 더 잘하게 만들면 점수가 오르지 않을까?"

하지만 실제 실험은 이 가설을 거의 부정했습니다.

---

## 4. 가장 중요했던 분석: 오답 125개 전수 분류

B0 random validation:
- 전체 1,341
- 정답 1,216
- 오답 **125**

125개를 직접 원인별로 분류했습니다.

| Failure type | Count | Share |
|---|---:|---:|
| number_text_binding | 42 | **33.6%** |
| ocr_recognition | 21 | 16.8% |
| question_understanding | 20 | 16.0% |
| visual_spatial_reasoning | 18 | 14.4% |
| exact_string_confusion | 13 | 10.4% |
| target_localization | 7 | 5.6% |
| ambiguity_or_label_issue | 4 | 3.2% |

### 여기서 배운 것

binding + question understanding + spatial reasoning:

**80 / 125 = 64%**

즉 주 병목은 단순 OCR recognition이 아니었습니다.

예:
- 숫자 여러 개를 읽기는 했지만 질문 대상에 맞는 숫자를 못 고름
- 글자는 읽었지만 무엇과 연결해야 하는지 틀림
- 질문이 묻는 대상/위치를 잘못 이해함

### 핵심 교훈

**데이터의 겉모습과 모델의 실제 failure mode는 다를 수 있다.**

"텍스트가 많다"  
≠  
"OCR recognition이 가장 큰 문제다"

---

## 5. 실험 중 실제로 배운 것

### 5.1 Prompt만 바꾼다고 해결되지 않는다

OCR-aware prompt:
- overall improvement 없음
- OCR-heavy accuracy도 개선되지 않음

binding-aware prompt:
- 일부 price/phone 문제를 rescue했지만
- 독립 holdout에서 안정적으로 재현되지 않음

교훈:

> prompt engineering은 싸고 빠르지만, 모델이 실제로 부족한 perception/reasoning 능력을 자동으로 만들어 주지는 않는다.

---

### 5.2 해상도를 무조건 올리는 것도 답이 아니다

global high-resolution:
- B0 대비 +1 sample 수준
- paired: rescue 5 / regression 4
- 거의 동일한 모델

교훈:

> resolution을 높이는 것은 "필요한 정보가 원래 잘리지 않았을 때" 효과가 거의 없다.  
> 작은 글씨 문제라면 global resize보다 target localization/crop이 더 논리적이다.

하지만 실제 tiled/crop 실험도 사전 gate를 넘지 못해 종료했습니다.

---

### 5.3 Output format은 병목이 아니었다

B0 generation:
- exact one-letter: 1,240 / 1,341
- non-exact: 101
- parse failure: 0

오류 125개 중 **123개가 exact one-letter 출력**에서 발생했습니다.

즉:

> 모델이 "a만 출력하지 못해서" 틀린 것이 아니다.

그래서 constrained choice scoring을 전체 데이터에 돌리는 것은 중단했습니다.

### 교훈

구현할 수 있는 기술이 있다고 해서 반드시 실험할 필요는 없습니다.

먼저:

> "이 변경이 실제 prediction을 바꿀 가능성이 있는가?"

를 확인하는 편이 낫습니다.

---

## 6. QLoRA에서 배운 것

### B3P3 — Qwen2.5-VL-3B

설정:
- NF4 4bit
- BF16
- q/k/v/o LoRA
- r=8
- alpha=16
- dropout=0.05
- lr=5e-5
- 1 epoch
- answer-token-only loss

inner dev:
- B0 271/300
- QLoRA 275/300
- rescue 10
- regression 6
- net +4

사전 gate의 regression 조건을 통과하지 못했기 때문에 clean promotion은 아니었습니다.

하지만 one-time frozen holdout에서는:
- B0: **361/400 = 90.25%**
- B3P3: **369/400 = 92.25%**
- rescue 13
- regression 5
- net +8
- McNemar p ≈ 0.096

통계적으로 p<0.05라고 말할 수는 없지만, 이후 ensemble 5% component로 넣었을 때:

**0.95144 → 0.95204**

로 실제 Public LB가 올랐습니다.

### 여기서 배운 점

약한 standalone 실험도 **ensemble diversity**로 가치가 생길 수 있습니다.

---

## 7. Qwen3-VL-8B QLoRA

로컬 8B fine-tuning:

- train: 5,707
- VAL-A: 1,007
- QLoRA r=32 / alpha=64
- q/k/v/o
- lr=2e-4
- 1 epoch
- 50% choice permutation
- four-choice next-token CE
- 640 profile

결과:

| | Accuracy |
|---|---:|
| Base 8B | 947/1007 = 94.04% |
| FT-8B | **952/1007 = 94.54%** |

paired:
- rescue 22
- regression 17
- net +5
- McNemar p ≈ 0.522

큰 유의미한 차이라고 말할 수는 없지만 방향은 positive였습니다.

이 FT-8B를 기존 ensemble에 **5%만** 추가:
- test prediction 변경 9 / 6,714
- Public LB **0.95204 → 0.95263**

### 교훈

강한 ensemble에서는:
- component를 통째로 교체하는 것보다
- **작은 비중의 diverse component**가 더 안전할 수 있습니다.

---

## 8. 왜 다른 모델들을 최종적으로 버렸나

### MiniCPM-V-4.5 INT4

confirmation 907:
- FT-8B: 94.60%
- MiniCPM: 86.77%
- rescue 17
- regression 88

FT가 틀린 문제를 꽤 rescue했지만, 맞는 문제를 너무 많이 망가뜨렸습니다.

FT confidence가 낮을 때 rescue가 더 많다는 신호는 있었지만, post-hoc low-margin routing의 개선은 작고 같은 validation에서 threshold를 본 결과라 채택하지 않았습니다.

### InternVL3-8B

첫 100:
- 92/100
- rescue 5 / regression 7

confirmation 907:
- FT: 94.60%
- InternVL: 87.21%
- rescue 22
- regression 89

pilot에서 보였던 scene_text 강점도 confirmation에서 재현되지 않았습니다.

### Step3-VL-10B

첫 100:
- FT: 94/100
- Step3: 91/100
- rescue 6
- regression 9

사전 gate:
- >=94, 또는
- >=92 + rescue>=4

에 미달해서 즉시 종료했습니다.

### 공통 교훈

> diversity가 있다는 것과 ensemble에 쓸 가치가 있다는 것은 다르다.

다른 답을 많이 내는 모델이 필요한 것이 아니라:

> **주력 모델이 틀릴 때 더 자주 맞는 모델**

이 필요합니다.

---

## 9. 채택하지 않은 주요 실험

| 실험 | 결과 | 결정 |
|---|---|---|
| OCR prompt | 개선 없음 | Reject |
| global high resolution | +1 sample 수준 | Reject |
| constrained choice scoring | format이 병목 아님 | Skip |
| binding-aware prompt | 작은 local signal | 미채택 |
| price router | fresh holdout에서 재현 실패 | Reject |
| tiled crop | rescue gate 미달 | Reject |
| B3P3 standalone | protocol상 clean promotion 아님 | ensemble diversity로만 사용 |
| FT-8B P0+P1 | Public LB 0.95233 | Reject |
| FT-8B 896 | Public LB 0.95233 | Reject |
| cross-view consensus | validation 신호, test 영향 거의 없음 | 미제출 |
| category router | Public LB 0.95263 | 기존 best와 동점, 복잡도 때문에 미채택 |
| MiniCPM | standalone 약함 | Reject |
| InternVL | confirmation 실패 | Reject |
| Step3 | first-100 gate 실패 | Reject |

---

## 10. 이 프로젝트에서 가장 중요했던 실험 습관

### 10.1 accuracy 하나만 보지 않기

두 모델 비교 시:

- both right
- rescue: baseline wrong → candidate right
- regression: baseline right → candidate wrong
- both wrong
- disagreement

을 봤습니다.

예를 들어 accuracy +0.2%라도:
- rescue 10
- regression 8

이면 매우 불안정할 수 있습니다.

반대로:
- rescue 4
- regression 0

이면 작은 변화라도 의미가 다릅니다.

---

### 10.2 validation을 계속 재사용하면 holdout이 아니다

프로젝트 중:
- random validation
- grouped confirmation
- fresh audit
- QLoRA inner dev
- frozen holdout
- VAL-A

등을 분리하려고 했습니다.

완벽하게 pristine하게 유지되지 못한 경우도 있었고, 그런 경우 보고서에 명시했습니다.

### 교훈

> 데이터 이름이 holdout이라고 해서 자동으로 holdout이 되는 것이 아니다.  
> 의사결정에 사용한 순간부터 그 데이터는 사실상 tuning 정보가 된다.

---

### 10.3 사전에 gate를 정하기

예:
- rescue >= N
- regression <= N
- accuracy >= X

처럼 결과를 보기 전에 기준을 정했습니다.

이유:

결과를 본 뒤:
- threshold 변경
- subset 변경
- category 변경
- weight 변경

을 반복하면 validation도 leaderboard처럼 과적합됩니다.

---

### 10.4 Public LB를 item-level oracle처럼 쓰지 않기

Public LB가:
- 0.95204 → 0.95263

으로 올랐더라도 test gold가 없기 때문에:

"어떤 9개 중 몇 개를 고쳤는지"

는 알 수 없습니다.

따라서 제출 점수만 보고 특정 test sample을 맞았다/틀렸다 추론하지 않았습니다.

---

## 11. 공부용 개념 정리

### Rescue / Regression

두 모델을 같은 sample에서 비교할 때:

- **rescue**: 기존 모델은 틀리고 새 모델은 맞음
- **regression**: 기존 모델은 맞고 새 모델은 틀림

net gain:

`rescue - regression`

단순 accuracy delta보다 모델 변화의 성격을 이해하기 좋습니다.

---

### McNemar test

같은 sample을 두 모델이 풀었을 때:

- A만 맞음
- B만 맞음

의 비대칭이 우연인지 확인하는 paired test입니다.

이 프로젝트에서는 작은 validation에서 p-value가 대부분 높았습니다.

따라서:
- "통계적으로 유의하다"라고 과장하지 않고
- practical gate + 방향성 정도로 해석했습니다.

---

### QLoRA

큰 모델 전체를 업데이트하지 않고:
1. base model을 4bit로 양자화
2. 일부 linear layer에 작은 LoRA adapter 추가
3. adapter만 학습

장점:
- VRAM 절약
- 로컬 GPU에서도 8B급 학습 가능

주의:
- 학습이 된다고 반드시 성능이 좋아지는 것은 아님
- loss 감소보다 holdout paired comparison이 중요

---

### Weighted log probability ensemble

모델 m의 choice c 확률을 `p_m(c)`, weight를 `w_m`이라고 하면:

`score(c) = Σ_m w_m log(p_m(c))`

가장 높은 score의 choice를 선택했습니다.

일반 arithmetic probability average와 달리 log-space에서는 모델 간 probability 비율을 곱하는 형태와 연결됩니다.

중요:
- 모델 confidence calibration이 완벽하지 않음
- weight를 Public LB에 계속 맞추면 leaderboard overfitting 위험

그래서 이 프로젝트에서는 작은 수의 bounded candidate만 제출했습니다.

---

## 12. 최종적으로 무엇이 점수를 올렸나

한 문장으로 정리하면:

> **한 모델을 크게 바꾸는 것보다, 오류 분석으로 쓸모없는 방향을 빠르게 버리고, 서로 조금 다른 강한 모델을 작은 비중으로 결합한 것이 최종 개선으로 이어졌다.**

실제로:
- generic OCR prompt → 실패
- resolution 증가 → 거의 실패
- routing → 재현성 부족
- 타 VLM → standalone 약함
- QLoRA + conservative ensemble → 최종 improvement

이었습니다.

---

## 13. 나중에 다시 본다면 기억할 것

### 하지 말아야 할 것

- Public LB weight grid search
- validation 결과를 보고 gate 변경
- "OCR 데이터니까 OCR 모델"처럼 데이터 표면만 보고 전략 결정
- 작은 category sample의 +1을 일반화
- p>0.05 결과를 유의하다고 표현
- test gold 없이 item-level rescue를 추정

### 다시 해볼 가치가 있는 것

프로젝트를 다시 시작한다면 순서는:

1. 강한 최신 VLM baseline
2. 처음부터 고정된 train/dev/holdout 설계
3. error taxonomy 자동화 + manual audit
4. calibration까지 포함한 ensemble
5. 필요하면 specialist router를 **별도 holdout**으로 검증
6. 마지막에만 leaderboard 제출

---

## 14. 최종 상태

프로젝트는 여기서 종료합니다.

최종 기록:
- best observed Public LB: **0.95263**
- final ensemble file: `output/ENSEMBLE_TEAM722_PERM1805_B3P3_0475_FT8B_05.csv`
- broad model search: 종료
- 추가 weight tuning: 하지 않음
- raw experiment logs: `reports/experiments/`
- canonical summary: **이 문서**

실험 자체보다 더 중요했던 것은:

> **가설 → 작은 실험 → paired 분석 → gate → 채택/폐기**

의 흐름을 반복한 경험입니다.
