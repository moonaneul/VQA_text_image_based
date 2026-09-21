[object Object]

## A4-P1 generation-format 진단

B0의 raw output을 확인했다.

- 정확히 a/b/c/d 한 글자: **92.47%**
- `(d)`, `(c)` 등 non-exact: **7.53% (101개)**
- 101개 모두 parser가 choice를 찾을 수 있음
- parse failure: **0**

따라서 scoring은 parsing을 고치는 실험이 아니다.

대신 `(d)`처럼 첫 token이 괄호인 101개에서는 **처음부터 a/b/c/d logits만 비교하는 방식**이 다른 결정을 낼 수 있다.

### 다음

전체 GPU run 전에 101개 subset의 현재 accuracy와 category를 분석한다.

이 subset이 실제로 더 약하면 그때 **101개만 constrained scoring**한다.
