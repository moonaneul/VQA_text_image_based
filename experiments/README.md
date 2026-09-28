# Experiments Archive

이 폴더는 프로젝트 진행 당시의 **raw experiment 기록**입니다.

프로젝트는 종료되었고, 최종 결론은 아래 문서를 기준으로 합니다.

- 전체 최종 정리: [../reports/FINAL_REPORT.md](../reports/FINAL_REPORT.md)
- 모델 변화 요약: [../reports/model_evolution.md](../reports/model_evolution.md)

## 이 폴더를 남겨 두는 이유

실패한 실험도 삭제하지 않았습니다.

나중에 공부할 때 다음을 확인할 수 있기 때문입니다.

- 어떤 가설을 세웠는지
- 어떤 조건을 고정했는지
- 왜 paired comparison을 했는지
- 왜 어떤 실험은 중단했는지
- validation을 계속 재사용하는 것이 왜 위험한지

## 실험 비교 원칙

두 run을 비교할 때 accuracy만 보지 않았습니다.

확인한 값:
- baseline accuracy
- candidate accuracy
- rescue: baseline wrong → candidate right
- regression: baseline right → candidate wrong
- net gain
- disagreement rate
- category별 paired 결과
- 필요 시 McNemar exact p-value

## 기록 원칙

- 한 번에 한 요소만 바꾸기
- 가능한 경우 결과 보기 전에 gate 정하기
- 실패 실험도 남기기
- validation / confirmation / holdout 역할 구분하기
- test gold가 없으면 item-level 정답을 추측하지 않기
- Public LB로 dense tuning하지 않기

## experiments.csv

`experiments.csv`는 초기 baseline 계열의 run registry입니다.

프로젝트 후반부의 모든 실험이 이 CSV 하나에 완전히 통합되어 있지는 않습니다.  
후반 실험의 authoritative summary는 `reports/FINAL_REPORT.md`와 `reports/experiments/`의 개별 기록을 참고합니다.

이 폴더는 더 이상 업데이트하지 않습니다.
