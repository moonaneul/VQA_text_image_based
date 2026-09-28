# Model Evolution — Final

> 프로젝트 종료 시점의 간단한 모델 변화 기록입니다.  
> 상세 수치는 [FINAL_REPORT.md](FINAL_REPORT.md)를 참고하세요.

## 한눈에 보기

```text
Qwen2.5-VL-3B baseline
  90.68% random / 91.73% grouped
        │
        ├─ OCR prompt ───────────── Reject
        ├─ global high-res ──────── Reject
        ├─ choice-format/scoring ── 병목 아님
        ├─ binding prompt/router ── 재현 실패
        ├─ tiled grounding ──────── gate 실패
        │
        ├─ Qwen2.5-VL-3B QLoRA (B3P3)
        │        └─ ensemble 5% → LB 0.95204
        │
Qwen3-VL-8B QLoRA
  94.04% → 94.54% on VAL-A
        └─ ensemble 5% → LB 0.95263
        │
        ├─ P0+P1 replacement ───── 0.95233 Reject
        ├─ 896 replacement ─────── 0.95233 Reject
        ├─ category router ──────── 0.95263 Tie / 복잡도 때문에 미채택
        │
        ├─ MiniCPM ──────────────── Reject
        ├─ InternVL3 ────────────── Reject
        └─ Step3-VL ─────────────── Reject

FINAL: 0.95263
```

## 핵심 단계

| Stage | 결과 | 의미 |
|---|---|---|
| B0 Qwen2.5-VL-3B | 90.68% random | 초기 기준 |
| Error review | reasoning/association 64% | OCR 자체보다 binding/reasoning이 큼 |
| B3P3 QLoRA | holdout 92.25% vs 90.25% | standalone 근거는 제한적이나 diversity 확보 |
| TEAM-C896 + Perm8B | LB 0.95144 | 강한 ensemble 기준 |
| + B3P3 5% | LB 0.95204 | 작은 diverse component 효과 |
| Qwen3-VL-8B FT | 94.54% vs base 94.04% | positive paired signal |
| + FT-8B 5% | **LB 0.95263** | 최종 best |

## 최종 결론

이 프로젝트에서 가장 잘 작동한 전략은:

1. 오답을 직접 분류하고
2. 효과 없는 prompt/resolution 실험을 빠르게 종료하고
3. QLoRA로 약간 다른 모델을 만들고
4. 강한 기존 모델을 크게 교체하지 않고
5. 작은 비중으로 diversity를 추가하는 것

이었습니다.

더 많은 모델을 계속 추가한다고 개선되지 않았습니다.  
MiniCPM, InternVL, Step3 실험은 모두 **다른 답을 내는 것만으로는 충분하지 않다**는 점을 보여줬습니다.
