[object Object]

## A4 scoring 사전 점검

Tokenizer 검사 결과 a/b/c/d는 assistant answer 위치에서 각각 **단일 token**이다.

그래서 next-token logits scoring 자체는 쉽게 구현할 수 있다.

하지만 B0가 이미 greedy generation으로 정확히 한 글자만 출력했다면 scoring은 같은 결과가 된다.

따라서 다음은 **GPU를 쓰지 않는 raw-output 진단**이다.

```text
B0 predictions.csv
→ raw_output이 정말 100% a/b/c/d 한 글자인가?
→ Yes: scoring full run 생략
→ No: scoring 실험 진행
```

이렇게 해야 "해볼 수 있는 기법"이 아니라 **실제로 정보를 줄 기법**만 실행한다.
