# Logic-VLA: A Temporal Logic Conditioned Vision-Language-Action Model

> **한 줄 요약**: 자연어 지시만으로는 안전·시공간 요구사항(장애물 이격 거리, 방문 순서, 데드라인 등)을 정확히 지정할 수 없다는 문제에서 출발해, 추론 시점에 주어지는 **Signal Temporal Logic(STL) 명세를 추가 입력으로 받는 π0.5 기반 VLA**를 제안. Robust semantics로 사전학습한 syntax-graph STL encoder + (1) STL-conditioned SFT, (2) 만족/위반 rollout 쌍에 대한 trajectory-level IPO(flow-matching loss 차이를 likelihood ratio surrogate로 사용)의 2단계 post-training으로, Isaac Sim 쿼드콥터 내비게이션에서 STL-blind base 대비 **STL 만족률 +24.8~40.7pp**, NL task 성공률 손실은 **최대 1.8pp**.

---

## 1. 배경 및 동기

- VLA는 "무엇을" 할지는 자연어로 잘 따르지만, "어떻게" 수행해야 하는지(최소 이격 거리 유지, 특정 순서로 영역 방문, 조건 충족 전까지 대기 등)는 자연어로 모호하게만 표현된다.
- STL은 궤적 수준 요구사항을 형식적으로 표현하고, **robust semantics** ρ로 만족/위반 정도를 정량 평가할 수 있다.
- 핵심 요구: 배포 시 요구사항이 바뀔 때마다 별도 정책을 학습하지 않고, **하나의 정책이 NL task를 유지하면서 입력으로 받은 STL 명세에 맞춰 행동을 조정**해야 한다.
- 기존 접근 비교: GRAPE·FlowPRO 등 preference 기반 VLA post-training은 고정된 선호/보상에 정책을 정렬; CBF·MPC·safety filter는 배포 시 외부 필터로 동작. Logic-VLA는 요구사항 자체를 정책 입력으로 두고 post-training에서 STL을 조건 + 감독 신호로 동시에 사용한다는 점에서 다르다.

---

## 2. 문제 정식화

- 잘 학습된 VLA π_θ (NL 분포 L, 환경 분포 E)가 주어졌을 때, post-training 절차 θ̃ = A(θ)를 설계해 π_θ̃(·|o, l, φ)가 관측 o, NL task l, STL φ를 동시에 조건으로 받아 s ⊨ φ와 NL task를 모두 만족하도록 한다 (Problem III.1).
- 도전 과제: (1) base rollout 분포를 크게 바꾸면 NL 성능 저하, (2) SFT만으로는 STL의 시공간 의미 학습이 어려움, (3) STL은 장기 궤적에 대한 전역 함수이므로 action chunk 단위 선호로 환원하면 안 됨.
- 가정: in-domain rollout 데이터셋 D 존재(Assumption III.2), NL과 STL 명세가 상충하지 않음(Assumption III.3).

---

## 3. 방법론: 데이터 구성

- STL 분포 P_STL에서 후보 formula bank Φ를 샘플.
- 오프라인 monitor로 각 rollout과 formula의 robust semantics를 계산 → 명확히 만족하는 rollout에서 **D+** (STL-conditioned action demonstration) 구성.
- 동일 NL task·유사 환경/초기조건을 공유하는 **만족–위반 rollout 쌍**을 매칭해 preference 데이터셋 **P** 구성.
- 실험 규모: 1,449 formula-group, 1,224개 고유 formula (87개 구조) → 만족 rollout 8,886개, preference-pair instance 13,494개.

---

## 4. 방법론: Logic Encoder

- **TeLoGraF**의 syntax-graph backbone 채택: 연산자(AND/OR/NOT/F/G/≤/≥)를 노드로, operand→parent 방향 edge. 노드 feature x_v = [연산자 타입, 시간 구간(t_s, t_e), 신호 one-hot(±1), threshold c, 보조 필드].
- TeLoGraF와 달리 객체 geometry나 초기 상태를 그래프에 넣지 않고 **기호적 요구사항만 인코딩**; 장애물 위치 등은 VLA의 관측을 통해 grounding.
- GCN(child→parent) + mean pooling → formula 표현 E_STL(φ).
- **사전학습**: 보조 trajectory encoder + MLP head로 tanh(ρ/c)를 Huber loss로 회귀 → 임계값·시간 경계·논리 조합 정보를 embedding에 주입. 사전학습 후 보조 모듈은 폐기.
- VLA 연결: 선형 projection 후 N_spec개의 specification token으로 reshape, 이미지·NL 토큰 뒤 VLM prefix에 append (양방향 attention으로 perception이 명세를 참조).

---

## 5. 방법론: 2단계 Post-training

### Stage 1 — STL-conditioned SFT
- D+에 대해 π0.5 action expert의 표준 conditional flow-matching loss로 학습. STL encoder와 VLA를 함께 fine-tune, STL 전용 loss는 없음.

### Stage 2 — Trajectory-level Preference Optimization
- **IPO(Identity Preference Optimization)** 채택. Flow-matching 정책은 정확한 likelihood 비율 계산이 비싸므로, **reference 대비 flow-matching loss 차이**를 log-policy ratio surrogate로 사용.
- STL 만족은 궤적 수준 속성이므로 여러 temporal window에 걸쳐 loss를 평균한 trajectory-level 점수 q±를 정의하고, 동일한 (flow time, noise) 실현값을 현재/참조 정책과 모든 window에 공유.
- margin Δ = β(q+ − q−)를 목표 c로 보내는 Huber화된 IPO loss.
- **One-sided preferred-rollout anchor**: 현재 정책이 만족 rollout에서 Stage-1 참조 정책보다 loss가 커질 때만 활성화 → "위반 rollout을 더 망가뜨려서 margin을 키우는" 퇴화 해를 방지.
- 최종 L_pref = E[ℓ_IPO + λ·A].

---

## 6. 실험 설정

- **환경**: NVIDIA Isaac Sim 5.1, 10개의 무작위 photorealistic 창고 환경, 6개 NL 내비게이션 task(포장 테이블, 지게차, 젖은 바닥 표지, 틈 통과 후 오렌지 배럴, G1 휴머노이드, 상자 더미로 비행).
- **데이터**: 충돌 없는 참조 궤적 3,000개, DJI Mavic 2 Pro 모델, 1인칭 onboard 카메라 + 3인칭 room-view 카메라, action = 절대 드론 위치.
- **Base 정책**: π0.5 체크포인트를 STL 없이 D로 fine-tune한 STL-blind 정책 → 모든 baseline의 공통 초기화.
- **학습 설정**: 2B VLM은 LoRA rank 16(α=16), 300M action expert는 LoRA rank 32(α=32), vision encoder와 action in/out projection은 full fine-tune. 40 action 실행 후 재질의, 10Hz 제어.
- **평가 설정**: Seen(60 entry), Unseen Parameter(99 entry, 87개 구조 모두 포함·파라미터만 새로), Unseen Structure(56 entry, 사전학습·post-training 모두에서 제외된 3개 템플릿의 49개 formula). entry당 초기상태 10개 rollout.
- **지표**: STL 만족률(ρ ≥ 0), ρ 평균±표준편차, 최소 ρ(worst-case), NL task 성공률(목표 영역을 정해진 순서로 모두 도달).
- **Baseline**: Base, STL-SFT(Stage 1만), Smooth Robust Semantics(1×/2×; flow-matching loss + 미분가능 smooth robust semantics 최대화).

---

## 7. 주요 결과 (Table I)

| Method | Seen STL Sat. | Seen NL | Unseen Param STL | Unseen Param NL | Unseen Struct STL | Unseen Struct NL |
|---|---|---|---|---|---|---|
| Base (STL-blind) | 41.3 | 90.5 | 50.0 | 93.9 | 56.8 | 89.3 |
| STL-SFT | 61.7 | 87.2 | 59.5 | 91.2 | 64.5 | 85.2 |
| Smooth RS (1×) | 76.2 | 68.5 | 71.1 | 75.9 | 77.5 | 68.6 |
| Smooth RS (2×) | 78.8 | 45.0 | 72.1 | 47.4 | 81.8 | 42.0 |
| **Logic-VLA** | **82.0** | 89.0 | **74.8** | 92.2 | **82.0** | 87.5 |

- ρ 통계(Logic-VLA): Seen 0.81±1.33 / min −1.62, Unseen Param 0.89±1.64 / min −3.79, Unseen Struct 1.24±1.71 / min −2.46. Base의 최소 ρ는 −9.11 / −5.80 / −6.98로 Logic-VLA가 worst-case 위반 폭을 크게 줄임.
- Smooth RS는 만족률은 올리지만 NL task 성공률이 급락(2×에서 Seen 45.0) — 직접적 robust semantics 최대화가 정책을 필요 이상으로 변형.
- Logic-VLA는 세 설정 모두에서 최고 만족률을 달성하면서 NL 성공률은 base/STL-SFT 수준 유지. Unseen Structure에서도 82.0%로 formula 구조 수준의 일반화를 보여줌.

---

## 8. Ablation 분석 (Figure 3)

- **STL encoder 사전학습** (STL-SFT 설정): 랜덤 초기화 대비 semantic 사전학습 시 STL 만족률 Seen 46.7 → 61.7, Unseen Param 53.0 → 59.5.
- **구조화 인코딩 vs 텍스트 프롬프트** (2단계 전체 절차 동일): 텍스트로 formula를 NL 지시에 붙이는 경우 대비 syntax-graph encoder가 Seen 66.5 → 82.0, Unseen Param 70.1 → 74.8, Unseen Param NL task 89.7 → 92.2.
- 해석: 텍스트 프롬프트도 일정 정보를 전달하지만, 술어·시간 연산자·논리 조합을 명시적으로 인코딩하는 것이 더 강한 조건 신호.
- 참고: ablation 수치는 막대 그래프(Figure 3)에만 표기되어 있음.

---

## 9. 관련 연구와의 위치

- **Preference 기반 VLA 정렬** (GRAPE, FlowPRO, SafeVLA): 고정된 안전/선호 목표로 정렬 → Logic-VLA는 요구사항이 입력으로 변하는 "조건부 정렬".
- **STL 조건 생성 모델** (TeLoGraF, Vision-TL-Action, S-MSP): 궤적 생성기에 STL을 조건으로 주지만 closed-loop VLA가 아님 → Logic-VLA는 반복 관측·행동하는 vision-language 정책을 적응.
- **배포 시 필터** (CBF, STL-guided sampling, hierarchical replanning): 실행 중 추가 최적화 필요 → Logic-VLA는 추론 시 추가 최적화 없이 단일 forward.

---

## 10. 강점

1. **문제 설정의 명확성**: "NL task는 유지하면서 형식 명세는 입력으로 바뀐다"는 조건부 post-training 문제를 깔끔히 정식화.
2. **Flow-matching 정책용 trajectory-level IPO**: likelihood가 없는 flow-matching action expert에 loss-difference surrogate + 공통 noise 실현 + 다중 window 집계로 궤적 수준 선호학습을 구현한 점이 실용적.
3. **Anchor 설계**: relative preference 목표의 알려진 퇴화 모드(선호 샘플도 같이 나빠짐)를 one-sided anchor로 직접 막음.
4. **일반화 평가 설계**: Seen / Unseen Parameter / Unseen Structure 3단계로 formula 일반화를 체계적으로 측정.
5. **Trade-off 가시화**: Smooth RS baseline과의 비교로 "만족률 vs NL 성공률"의 trade-off를 수치로 명확히 보여줌.

---

## 11. 한계 및 논의

1. **시뮬레이션 단일 도메인**: Isaac Sim 창고 환경의 쿼드콥터 내비게이션만 평가. 매니퓰레이션이나 실제 기체 검증 없음.
2. **제한된 STL 어휘**: 술어가 드론 위치 x, y, z에 대한 임계값 비교로 제한되어, 객체 상대 거리 등 복합 술어로의 확장은 미검증.
3. **제어 인터페이스 단순화**: 절대 위치를 Isaac Sim API로 직접 추종 — 실제 동역학·저수준 제어 오차가 배제됨.
4. **데이터 요구**: 만족/위반 쌍 매칭을 위해 대량 rollout + 오프라인 monitor가 필요. 새 도메인에서의 데이터 수집 비용이 큼.
5. **보장 부재**: 만족률 82%는 여전히 약 18% 위반을 의미하며, 안전-임계 용도에서는 형식적 보장이 없다는 점이 남음. 코드 공개 언급도 없음.
6. **NL–STL 충돌 가정**: 두 명세가 상충하는 경우의 처리(Assumption III.3)는 범위 밖.

---

## 12. 총평 및 예상 질문

- **총평**: VLA에 형식 논리 요구사항을 "입력 조건"으로 주입하는 드문 시도로, 사전학습된 STL encoder와 flow-matching용 trajectory-level IPO를 결합해 NL 성능을 거의 해치지 않고 STL 만족률을 크게 끌어올렸다. 평가 범위는 좁지만 방법론은 다른 flow-matching VLA에 이식 가능한 템플릿이다.
- **예상 질문**
  - Q1. Unseen Structure에서 만족률(82.0)이 Unseen Parameter(74.8)보다 높은 이유는? 템플릿 난이도 차이인지, 샘플 수 차이인지?
  - Q2. Flow-matching loss 차이를 log-ratio surrogate로 쓸 때의 분산은 어떻게 통제했는가(공통 noise 외에)?
  - Q3. 안전-임계 명세에 대해 배포 시 filter(CBF 등)와 결합하면 추가 이득이 있는가?
  - Q4. 실제 드론 또는 매니퓰레이션 VLA(예: LIBERO)로의 이식 시 STL 술어 어휘는 어떻게 설계해야 하는가?

<!-- VERIFIED: pdf -->
