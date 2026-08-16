# DreamTrajectory: Trajectory-Guided Action Generation with World Model Alignment for Mobile Manipulation

- arXiv: 2608.01381 (2026-08-02)
- 기관: Hong Kong University of Science and Technology (Guangzhou) / Ola Dimensions
- 코드: 공개 정보 없음

## 1. 한 줄 요약

모바일 매니퓰레이션에서 end-effector 궤적을 "지각과 전신 제어 사이의 명시적 인터페이스"로 삼아, 하나의 action expert가 계획 궤적과 전신 action chunk를 동시에 디노이징하고, 경량 궤적 world model이 후보 action이 실제로 유발할 궤적을 예측해 계획과 가장 잘 맞는 후보를 실행 전에 고르는 프레임워크(DT)다.

## 2. 문제 정의

모바일 매니퓰레이터는 시점이 계속 바뀌고 접촉 조건이 변하는 상황에서 베이스와 팔의 움직임을 동시에 결정해야 하며, 고정 베이스 조작보다 유효 행동 공간이 훨씬 크다. 저자들은 기존 VLA가 두 가지 점에서 부족하다고 본다. 첫째, 관측에서 전신 action chunk로 직접 매핑하기 때문에 태스크 공간의 명시적 운동 계획 없이 고차원 행동 공간을 탐색하고, chunk가 실현하려는 운동이 암묵적으로만 존재해 베이스-팔 협응 예측이 부정확해진다. 둘째, 예측한 chunk를 open-loop로 실행해 그 행동이 의도한 운동을 실제로 만들어내는지 아무도 검증하지 않으며, 제어 오차·충돌·분포 외 접촉은 이미 실패한 뒤에야 드러난다. 내비게이션 후 조작으로 분해하거나 베이스와 팔의 컨트롤러를 분리하는 전통적 대안은 최적화는 쉬워지지만 이동 중 도달, 베이스 재배치 중 관절체 조작처럼 시간적으로 강하게 결합된 행동을 배제한다.

## 3. 핵심 아이디어

중심 가설은 "간결한 end-effector 궤적이 전신 제어를 위한 물리적으로 의미 있는 중간 표현"이라는 것이다. 궤적은 베이스와 팔의 명령이 그 운동을 *어떻게* 실현할지에 앞서 그리퍼가 *어디로* 가야 하는지를 규정한다. 여기서 두 가지가 파생된다. (i) 궤적을 정책 내부에서 행동과 함께 생성해 태스크 공간 레퍼런스로 삼는다. (ii) 같은 궤적 표현을 후보 행동의 결과 예측에도 쓰면, 계획 궤적과 유발 궤적을 직접 비교하는 것만으로 실행 전 정합성 검사가 가능해진다. 표현을 공유한다는 점이 이 논문의 구조적 요점이다.

## 4. 아키텍처

pi0.5 백본 위에 dual-stream action expert를 얹는다. 입력은 egocentric head 카메라와 wrist 카메라 이미지, 고유수용 상태 s_t, 언어 지시 l이다. 미래 운동은 tau_{t:t+H-1}로 표현되며 각 waypoint는 7D 포즈, 즉 위치 p와 쿼터니언 q를 현재 베이스 포즈에 고정된 chunk-local 프레임 B_t에서 나타낸 것이다. 이 프레임은 예측 지평 동안 고정되므로 궤적이 미래 베이스 운동과 팔 운동의 합성 효과를 함께 담게 된다. 궤적과 action chunk는 지평 H를 공유하며 단일 expert가 결합 분포로 생성한다.

## 5. 학습 목적함수

pi0의 conditional flow matching을 따른다. 학습 샘플의 궤적 chunk tau와 action chunk a에 대해 노이즈 eps_tau, eps_a를 독립적으로 샘플링하되 flow time sigma ~ U(0,1)는 **공유**한다. 보간은 tau_sigma = sigma*eps_tau + (1-sigma)*tau, a_sigma = sigma*eps_a + (1-sigma)*a이며, 같은 flow time을 쓰기 때문에 두 디노이징이 동기적으로 진행된다. 손실은 L_VLA = E[lambda_tau*||v_tau - u_tau||^2 + lambda_a*||v_a - u_a||^2]이고 목표 속도는 u_tau = eps_tau - tau, u_a = eps_a - a다. 추론 시에는 sigma=1에서 0까지 Euler step으로 두 속도장을 함께 적분한다.

## 6. Group-causal attention

두 스트림 사이에 비대칭 의존성을 강제한다. 궤적 토큰은 멀티모달 prefix와 이전 궤적 토큰만 볼 수 있고, action 토큰은 거기에 더해 궤적 스트림 **전체**와 이전 action 토큰을 본다. 계획 궤적이 action 디노이징을 인도하되 action 정보가 궤적 생성으로 역류하지 못하게 막는 설계다. 궤적이 "행동에 맞춰 사후 합리화되는" 것을 구조적으로 차단한다는 점에서, 단순히 보조 손실을 붙이는 방식과 구분된다.

## 7. 궤적 world model과 test-time refinement

world model W_phi(o_t, s_t, a_{t:t+H-1})은 dual-view 관측, 상태, 후보 action chunk를 받아 그 chunk가 실제로 유발할 궤적 tilde-tau(a) in R^{H x 7}을 예측한다. 계획 궤적이 아니라 실행 결과의 예측이라는 점이 핵심이다. 이 매핑을 해석적으로 유도하지 않는 이유는 명령 운동과 실현 운동의 편차가 외부 물체와의 접촉, 팔의 자기 충돌, 저수준 컨트롤러의 추종 오차에서 오기 때문이고, 어떤 것도 해석적 동역학 모델이 신뢰성 있게 잡아내지 못한다.

학습 샘플은 d_s = (I_head, I_wrist, s_t, a_{t:t+H-1}, tau_exec)이며, 기록된 평면 베이스 속도 채널을 chunk에 걸쳐 적분하고 순간 베이스 프레임의 end-effector 포즈를 시각 t의 chunk-local 프레임으로 변환해 만든다. 한 스텝의 시간 오프셋을 두어 step h의 목표 waypoint가 a_{t+h} 실행 *후* 관측된 포즈가 되도록 한다. 성공과 실패 시행을 모두 포함한 상호작용 데이터에서 sliding window로 조밀하게 샘플을 만든다.

추론 시 search-predict-score는 다음과 같다. 원 chunk를 유지한 채 denormalized action 공간에서 N-1개의 섭동 chunk를 만드는데, 각 행동 차원이 주변 표준편차 sigma=0.05, lag-one 상관 rho=0.9의 독립 AR(1) Gaussian 과정을 따른다. N=30으로 후보 집합 C를 만들고 world model이 배치 차원에서 30개를 병렬 평가한 뒤, tilde-a = argmax [lambda*S_traj(a) + (1-lambda)*eta*S_smooth(a)]로 선택한다(lambda=0.5, eta=1e-3). S_traj는 계획-유발 궤적 일치도를, S_smooth는 급격한 운동에 대한 페널티를 담당한다. world model에는 언어를 넣지 않는데, 실행 결과 예측은 태스크 비의존적이고 태스크 의도는 이미 계획 궤적이 대표하기 때문이다.

## 8. 실험 설정

시뮬레이션은 ManiSkill-HAB(MS-HAB)의 set_table suite 6개 서브태스크(pick apple, pick bowl, open fridge, close fridge, open counter, close counter)를 Fetch 모바일 매니퓰레이터로 수행한다. 저자들은 "모바일 매니퓰레이션"을 순차적 내비게이션 후 고정 베이스 조작이 아니라 베이스와 팔의 동시 운동을 요구하는 것으로 정의하고, 조사한 공개 벤치마크 중 MS-HAB이 이 기준을 가장 잘 만족한다고 밝힌다. 서브태스크당 100 에피소드, 방법당 600 에피소드다. 실기는 ARX LIFT 모바일 매니퓰레이터에서 과일 pick-and-place(사과, 오렌지, 배, 망고스틴), 서랍 열기, 서랍 닫기 3개 태스크를 태스크당 20 에피소드로 평가한다. 세 태스크 모두 팔 작업 공간이 제한되어 조작 중 베이스 조정이 필수다. 학습은 두 단계로 분리 최적화된다 — Stage I은 pi0.5 사전학습 가중치에서 시작한 VLA 파인튜닝, Stage II는 world model 지도학습이며, 배포 시점에만 refinement로 결합된다.

## 9. 주요 결과

MS-HAB 성공률(%, Table 2):

| Method | Pick Apple | Pick Bowl | Open Fridge | Close Fridge | Open Counter | Close Counter | Avg |
|---|---|---|---|---|---|---|---|
| ACT | 1.0 | 0.0 | 28.0 | 24.0 | 22.0 | 93.0 | 28.0 |
| Diffusion Policy | 19.0 | 22.0 | 19.0 | 61.0 | 17.0 | 28.0 | 27.7 |
| RDT-1B | 0.0 | 0.0 | 5.0 | 14.0 | 0.0 | 33.0 | 8.7 |
| GR00T N1 | 3.0 | 1.0 | 43.0 | 16.0 | 5.0 | 59.0 | 21.2 |
| pi0.5 | 37.0 | 19.0 | 5.0 | 8.0 | 87.0 | 38.0 | 32.3 |
| **DT (ours)** | 39.0 | 35.0 | 51.0 | 33.0 | 91.0 | 80.0 | **54.8** |

실기 성공률(%, Table 4, 태스크당 20 에피소드):

| Method | Fruit Pick-and-Place | Open Drawer | Close Drawer | Avg |
|---|---|---|---|---|
| pi0.5 | 45.0 | 60.0 | 85.0 | 63.3 |
| DT w/o refiner | 70.0 | 75.0 | 100.0 | 81.7 |
| **DT** | 80.0 | 90.0 | 100.0 | **90.0** |

## 10. 어블레이션

MS-HAB 어블레이션(Table 3)에서 궤적 유도를 action-only pi0.5 베이스라인에 추가하면 평균이 32.3%에서 47.5%로 오른다. 가장 큰 이득은 접촉이 많은 관절체 태스크에서 나온다 — open fridge 5.0 → 44.0, close counter 38.0 → 72.0. 반면 pick apple(37.0 → 34.0)과 open counter(87.0 → 80.0)는 소폭 하락해, 궤적 유도가 정밀한 전신 접촉 협응을 요구하는 태스크에 선별적으로 유리함을 보여준다. world model refinement는 평균을 54.8%로 끌어올리며 6개 서브태스크 **전부**를 개선하고 7.3%p를 더한다. close counter가 72.0 → 80.0으로 오른 것은 초기 정책이 이미 강할 때도 잔여 오차를 교정한다는 뜻이다. 결국 베이스라인 대비 22.5%p 이득은 궤적 유도 15.2%p와 refinement 7.3%p로 분해되며, 두 요소가 상보적이라는 주장을 뒷받침한다.

world model 아키텍처 비교(Table 5, 4,096 궤적, H=16, 낮을수록 좋음):

| Model | xyz ADE (m) | xyz FDE (m) | Angular ADE (deg) | Geodesic ADE |
|---|---|---|---|---|
| Analytical FK (open-loop) | 0.241 | 0.345 | 27.0 | 0.057 |
| Diffusion | 0.061 | 0.086 | 8.1 | 0.014 |
| One-shot Transformer | 0.036 | 0.049 | 7.7 | 0.016 |
| Cross Attention | 0.033 | 0.045 | 7.4 | 0.016 |
| **GRU (DT 채택)** | **0.028** | **0.035** | **6.2** | **0.014** |

학습된 모델이 모두 해석적 FK를 큰 차이로 앞선다는 점이 "접촉과 컨트롤러 오차는 명목 기구학으로 잡히지 않는다"는 논지의 직접 증거다.

## 11. 계산 비용

RTX 4090 단일 GPU, BF16 기준(Table 6). 궤적 head는 0.017M 파라미터(<0.01%)에 3.65ms, 궤적 world model은 49.02M(3.5B pi0.5 대비 1.40%)에 8.10ms, DT 전체는 49.04M(1.40%)에 replanning step당 11.75ms를 더한다. 픽셀 공간이 아니라 압축된 궤적 공간에서 후보를 평가하기 때문에 30개 후보를 배치 병렬로 처리해도 이 정도로 끝난다는 것이 설계상 실질적 이득이다. 또한 refiner는 정책 재학습이 필요 없는 plug-in 단계다.

## 12. 강점, 약점, 시사점

강점. 계획과 실행 결과를 같은 태스크 공간 표현으로 통일해 "실행 전 정합성 검사"를 저비용으로 구현한 점이 깔끔하다. group-causal mask로 정보 역류를 막은 것은 궤적을 진짜 사전 계획으로 유지하는 데 필요한 최소한의 구조적 장치다. world model을 픽셀/비디오가 아닌 궤적 공간에 둔 선택은 GigaWorld-Policy, MotuBrain, DreamZero 같은 영상 기반 world-action model 계열과 대비되며 1.40% 오버헤드라는 실측치로 정당화된다. 해석적 FK 대비 학습 모델의 우위를 직접 측정한 것도 좋다.

약점과 의문점. 첫째, 시뮬레이션 평가가 MS-HAB set_table 6개 서브태스크에 국한되고 LIBERO/CALVIN 같은 표준 벤치마크 수치가 없어 다른 VLA와의 직접 비교가 어렵다. 저자들은 future-prediction VLA들이 고정 베이스 임베디먼트라 MS-HAB과 인터페이스가 맞지 않아 정량 비교에서 제외했다고 밝히지만, 반대로 DT를 고정 베이스 벤치마크에 올린 결과도 없다. 둘째, 궤적 유도가 pick apple과 open counter에서 성능을 떨어뜨렸는데 그 원인 분석이 얕다. 셋째, N=30, sigma=0.05, rho=0.9, lambda=0.5 같은 refinement 하이퍼파라미터의 민감도 분석이 본문에 없다. 넷째, S_traj/S_smooth의 정확한 정의, attention mask, world model 세부 구조가 모두 보충자료로 넘어가 본문만으로는 재현이 불가능하다. 다섯째, world model 학습용 상호작용 데이터의 규모가 명시되지 않아 refiner의 데이터 비용을 가늠하기 어렵다. 여섯째, 실기 태스크가 3개, 태스크당 20 에피소드로 표본이 작고 close drawer는 refinement 없이 이미 100%라 헤드룸이 없다.

시사점. VLA의 test-time scaling을 "여러 행동을 뽑아 고른다"로 놓을 때, 채점 기준을 어디에 둘 것인가가 핵심 설계 변수다. DT는 보상 모델이나 영상 예측 대신 정책 자신이 생성한 계획 궤적을 채점 기준으로 재사용한다. 즉 추가 감독 신호 없이 정책의 자기 일관성만으로 test-time 선택을 구현했다. 이 아이디어는 궤적이라는 표현이 특별해서 성립하는 것이 아니라, 정책이 명시적 중간 목표를 뱉기만 하면 일반화될 수 있다는 점에서 모바일 매니퓰레이션 밖으로도 확장 여지가 있다.

<!-- VERIFIED: pdf -->
