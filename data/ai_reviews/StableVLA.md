# StableVLA: Towards Robust Vision-Language-Action Models without Extra Data 세미나 리뷰

> **한 줄 요약**: 학습 데이터에 포함되지 않은 시각적 disturbance에 취약한 기존 VLA 문제를, 정보 이론 기반의 lightweight Information Bottleneck Adapter(IB-Adapter)로 해결. 0.5B 백본에 <10M 파라미터만 추가해 OpenPi를 능가하는 visual robustness를 확보한 ICML 2026 논문.

---

## 1. 배경 및 동기

- 모든 가능한 disturbance를 학습 데이터에 담는 것은 현실적으로 불가능하며, 최신 VLA 모델조차 학습 분포에서 벗어난 실제 환경의 시각적 disturbance(예: 조명 변화, occlusion, 물리적 corruption)에 노출되면 큰 성능 저하를 보인다.
- 저자들은 최신 SOTA VLA에 대한 systematic study를 통해 이 robustness gap을 정량적으로 드러내고, **추가 데이터나 augmentation 없이도** 이를 완화할 수 있는 경량 어댑터를 제안한다.

## 2. 핵심 아이디어

- **Information Bottleneck Adapter (IB-Adapter)**: 정보 이론(IB principle)에 기반해 시각 입력에서 잠재적 noise를 선택적으로 걸러내는 lightweight adapter 모듈.
- **No extra data, no augmentation**: 별도의 augmentation pipeline이나 추가 수집 데이터를 요구하지 않으며, 베이스라인 대비 평균 **약 30%** 성능 향상.
- **Parameter-efficient**: 추가 파라미터가 **10M 미만**으로, 기존 VLA에 거의 무비용으로 부착 가능.
- **Backbone downsizing**: **0.5B** 파라미터 백본만 사용하고 Open X-Embodiment에 대한 pre-training 없이도 7B-scale SOTA VLA에 견주는 robustness를 달성.
- **Long-horizon 보존**: long-horizon task의 정확도를 유지하면서, 합성/물리적 visual corruption 모두에서 OpenPi를 상회.

## 3. 방법론 요약

- IB-Adapter는 시각 인코더의 representation에 information bottleneck을 부과해, task-relevant 신호는 유지하되 학습 분포에서 벗어난 시각적 잡음 성분은 압축/제거하도록 학습된다.
- 추가 파라미터 < 10M으로 0.5B 베이스라인 위에 끼워 넣으며, 학습 자체에는 별도의 augmentation 데이터셋이 필요 없다.
- 결과 모델 **StableVLA**는 14배 작은 백본과 OXE 비-사전학습 조건임에도 7B-급 SOTA와 비교 가능한 robustness 곡선을 갖는다.
- 평가는 (a) synthetic visual corruption (b) physical visual corruption 두 축에서 진행되며, baseline 및 OpenPi와 비교된다.

## 4. 실험 결과

- **베이스라인 대비 평균 향상**: 약 **+30%** (구체 benchmark 별 수치는 abstract에 미명시).
- **추가 파라미터**: < **10M** (전체 모델 0.5B 백본 + IB-Adapter).
- **백본 축소**: **14×** 작은 백본(0.5B), Open X-Embodiment **사전학습 없음**.
- **vs OpenPi**: synthetic / physical visual corruption 모두에서 OpenPi를 상회.
- **Long-horizon task**: 정확도 유지(구체 점수 abstract에 미명시).
- **LIBERO/CALVIN/SimplerEnv** 등 표준 benchmark 점수: abstract에 미명시.

## 5. 한계 및 의의

- **한계**:
  - Abstract 단계에서 정량 점수가 "평균 +30%"와 정성적 비교(OpenPi 상회) 위주로만 제시되어, benchmark별 per-suite 비교는 본문/표 확인이 필요하다.
  - "Visual disturbance" 카테고리(조명, occlusion, blur, sensor noise 등)별 robustness profile이 abstract에서는 분리되지 않는다.
  - IB-Adapter의 정보 이론적 정의(어떤 mutual information을 최소화/유지하는지, β 스케줄, surrogate 등)는 본문 확인 필요.
  - 0.5B 백본이라는 강한 제약 하에서의 결과이므로, 7B 백본에 IB-Adapter를 그대로 부착했을 때의 ceiling은 별도 검증이 요구된다.
- **의의**:
  - "추가 데이터 없이" robustness를 끌어올린다는 점에서, 데이터 스케일링이 어려운 실세계 시나리오에서 매우 실용적이다.
  - <10M 파라미터 어댑터로 14× 작은 모델을 7B-급 SOTA에 수렴시킨다는 결과는, VLA 영역에서 IB 류 information-theoretic regularization의 가치를 강하게 시사한다.
  - ICML 2026 accept라는 점에서, robustness-oriented VLA 연구 흐름의 대표 사례 중 하나로 자리잡을 가능성이 있다.

<!-- VERIFIED: abstract-only -->
