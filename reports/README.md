[object Object]

## A4-P2 결론: scoring은 건너뛴다

정확히 한 글자만 출력한 1,240개가 오히려 더 어려웠다.

| Output type | Accuracy | Errors |
|---|---:|---:|
| exact a/b/c/d | **90.08%** | **123** |
| non-exact | **98.02%** | **2** |

즉 전체 오류의 **98.4%가 이미 clean one-letter decision**이었다.

따라서 다음 문제는 output format이 아니라 **왜 잘못된 선택지를 골랐는지**다.

### 다음 단계

**125개 오답 root-cause review**

```text
OCR recognition?
target localization?
number/text binding?
exact-string confusion?
question understanding?
visual reasoning?
label ambiguity?
```

이 결과를 보고 crop/OCR/QLoRA 중 어디에 투자할지 결정한다.
