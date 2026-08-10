# DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation

**arXiv**: 2608.06374 (2026-08-06) · **소속**: The Hong Kong University of Science and Technology (Guangzhou), COCO Matrix (Shanghai) · **공동 1저자**: Junfeng Li, Junjie He, Zhide Zhong(프로젝트 리드), Yangyang Zheng · **교신**: Haoang Li · **프로젝트 페이지**: https://livfour.github.io/DyPES-VLA_RELEASE/

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
단일 generalist 정책으로 이질적인 로봇 embodiment를 모두 다루는 것은 VLA 분야의 미해결 문제다. 저자들은 기존 cross-embodiment VLA의 한계를 두 가지로 정리한다.

1. **감독 신호의 빈곤**: 기존 방법은 action prediction을 유일한 supervision으로 사용한다. 따라서 embodiment 간에 공유되는 dynamics prior가 오직 action label로부터만 학습된다. 물체 운동, 접촉(contact), 상호작용에 의한 장면 변화라는 반복적 패턴을 담고 있는 대규모 human/robot manipulation 비디오는 action label이 없다는 이유로 활용되지 못한다.
2. **수작업 action 정렬(alignment)**: 많은 방법이 이질적 제어를 공통 포맷(common action format)으로 수작업 매핑해 action 레벨에서 정보를 공유한다. 이는 좌표 변환이나 inverse kinematics를 요구하고 morphology 수가 늘수록 확장성이 나쁘다. 더 본질적으로, 이 강제 통합은 **embodiment 간 공유되어야 할 상호작용 규칙성**과 **각 로봇 신체에 고유한 제어 의미론**을 뒤섞어버린다.

### 핵심 질문
> 단일 정책에서 **무엇을 공유하고, 무엇을 embodiment-specific으로 남겨야 하는가?**

DyPES-VLA의 답은 "공유 인터페이스를 action space가 아니라 그 **상류(upstream)의 query state**에 둔다"는 것이다.

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처 (Fig. 2)
세 컴포넌트로 구성된다.
1. **사전학습 VLM** (Qwen3-VL-2B): 시각 관측 `o_t`, 언어 지시 `l`, embodiment metadata `m_e`, 그리고 학습 가능한 query token `Q ∈ R^{N×d}`를 한 번의 forward로 처리 → **query states** `Z ∈ R^{N×d}`.
2. **Future generation head** (SANA-600M image generator): `Z`를 조건으로 미래 프레임을 합성 → shared dynamics priors 학습.
3. **Embodiment-specific MoE action head** (flow-matching DiT): 동일한 `Z`를 조건으로 각 embodiment의 **native action space**에서 직접 action chunk 생성.

핵심은 `Z`가 두 head의 **유일한 공유 인터페이스**라는 점이다.

### 2.2 Shared Query Interface (§3.2)
`Z = f_θ([φ_v(o_t), φ_l(l), φ_l(m_e), Q])`

Embodiment metadata는 텍스트로 verbalize된다 (`data_source=libero, robot_type=Single Arm Gripper, robot_model=Franka Panda, camera_views=[main, wrist], control_freq=20.00 Hz, action_dim=7, action_horizon=8, action_space=[Δx,...,gripper]`). 즉 embodiment 정보가 VLM에게 **자연어로 명시**된다. 특기할 점: **proprioceptive 입력을 전혀 쓰지 않는다.**

### 2.3 Dynamics Priors 학습 (§3.3)
현재 시각 `t`에서 primary camera의 미래 프레임 `x_{t+Δ_e}`가 타깃이며, temporal offset `Δ_e`는 embodiment별 action horizon `H_e`에 맞춘다. Frozen autoencoder가 `z = AE_w(x_{t+Δ_e})`로 latent를 만들고, projector `p_ω(Z)`가 SANA transformer `g_ψ`의 cross-attention conditioning token이 된다.

**설계상 가장 중요한 제약**: 생성기는 **현재 관측도, 그 latent도 받지 않는다.** 미래를 합성하는 데 필요한 정보가 오직 N개의 query state를 통과해야만 하도록 information bottleneck을 강제한 것이다. 이것이 "future prediction이 표현을 실제로 형성한다"는 주장의 구조적 근거다.

Rectified-flow 목적함수: `z_τ = τz + (1-τ)ε`, `L_future = E[||g_ψ(z_τ, τ, p_ω(Z)) - (z-ε)||²]`.

### 2.4 Embodiment-Specific Control (§3.4)
Flow-matching DiT가 action 생성을 두 축으로 분해한다.
- **Per-Embodiment Interfaces**: 각 embodiment가 경량 encoder-decoder 쌍을 소유. `Enc_{r(e)}`는 noisy action chunk와 flow timestep을 공유 DiT width로 임베딩하고, `Dec_{r(e)}`는 DiT 출력을 native action chunk `A_e ∈ R^{H_e×d_e}` 위의 velocity로 되돌린다.
- **Shared Attention, Routed Experts**: L개 블록에서 attention(`Z`에 대한 cross-attn + action sequence self-attn, AdaLN(τ) 변조)은 **모든 embodiment가 공유**하고, FFN만 K개 expert 뱅크에서 라우팅 인덱스로 선택한다.

  `X̄ = X + Attn(AdaLN(X,τ); Z)`, `X' = X̄ + FFN^{(r(e))}(AdaLN(X̄,τ))`

라우팅은 **static**이다. metadata `m_e`가 `r(e) ∈ {1,...,K}`를 결정론적으로 지정한다 (학습된 gating 아님).

### 2.5 두 단계 학습과 추론 (§3.5)
- **Stage 1 (Dynamics Priors Pretraining)**: action label이 불필요하므로 action-free 비디오(EgoDex + 시뮬레이션 비디오)로 VLM + query token + SANA head 사전학습.
- **Stage 2 (Cross-Embodiment Co-Training)**: `L = L_action + λ_w · L_future`로 두 head 공동 최적화. future prediction은 공유 표현의 **regularizer** 역할을 계속한다.
- **추론**: VLM 1회 forward로 `Z` 생성 → action head가 Gaussian noise에서 few Euler steps로 flow 적분 → native action chunk. **Future generation head는 추론 시 제거된다** (연산 비용 0).

## 3. 데이터 전략

- **Stage 1 혼합**: 50% EgoDex(egocentric human manipulation) / 20% RoboTwin 2.0 / 20% RoboCasa-GR1 / 10% LIBERO. EgoDex는 full corpus 사용.
- **Stage 2 혼합**: 40% RoboTwin 2.0 / 40% RoboCasa-GR1 / 20% LIBERO.
- **실기 finetuning**: 3개 task × 3개 embodiment × 200 demo = 1,800 teleoperated demonstrations.

3개 embodiment family는 각각 시뮬레이션-실기 짝을 이룬다: single-arm(Franka Panda ↔ FR3), dual-arm(ALOHA-AgileX ↔ COBOT Magic), humanoid(Fourier GR-1 ↔ Unitree G1 + Inspire RH56DFQ hands). LIBERO의 Panda와 실기 FR3가 7-DoF 운동학을 밀접하게 공유하므로 expert가 그대로 재사용된다는 점이 실기 전이의 설계적 근거다.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| VLM | Qwen3-VL-2B |
| Future head | SANA-600M (사전학습 가중치로 초기화) |
| Action head | 16-layer DiT, from scratch, K=3 experts |
| Query token 수 | N = 96 |
| 입력 해상도 | 모든 view 256×256 |
| Camera views | single-arm 2개(3인칭+wrist), GR-1/G1 humanoid 1개(egocentric), dual-arm 3개(egocentric+양 wrist) |
| Action horizon `H_e` (= `Δ_e`) | single-arm 8, humanoid 16, dual-arm 50 |
| Proprioception | 미사용 |
| Action 정규화 | 차원별 min-max |
| Stage 1 / Stage 2 steps | 100,000 / 200,000 |
| Batch / GPU | effective batch 512, 16× H100, AdamW |
| `λ_w` (future loss weight) | 0.05 |
| Flow timestep schedule | `τ = s^{(1-u)}`, `u ~ Beta(1.5, 1.0)`, `s = 0.999` (고노이즈 구간 집중, GR00T-N1 방식) |
| 추론 Euler steps | 4 |
| 실기 finetuning | 1,800 demo, 5,000 steps |

## 5. 실험 설계 및 평가 프로토콜

저자들은 baseline을 **checkpoint scope** 기준으로 나눈다는 점이 중요하다.
1. **Per-benchmark specialists**: 벤치마크마다 별도로 학습/파인튜닝/포스트트레이닝. multi-robot 데이터로 사전학습한 방법도 보고 결과가 벤치마크별 checkpoint에서 나오면 이 그룹에 포함.
2. **Single-checkpoint generalists**: 벤치마크별 파인튜닝 없이 단일 checkpoint를 전 벤치마크에 평가. DyPES-VLA와 동시기 Qwen-VLA가 여기 속한다.

**이 구분이 결과 해석의 핵심**이다. DyPES-VLA의 수치는 훨씬 불리한 조건(단일 checkpoint)에서 나온 것이다.

평가 규모: RoboTwin 2.0 50 tasks × 100 rollouts × 2 settings, RoboCasa-GR1 24 tasks × 50 rollouts(총 1,200), 실기 3 tasks × 3 embodiments × 25 rollouts.

## 6. 실험 결과 심층 분석

### RoboTwin 2.0 (Table 1, 14-DoF ALOHA-AgileX)
Clean = 벤치마크의 Easy, Randomized = Hard 설정.

| Method | Clean | Randomized | Average |
|---|---|---|---|
| Diffusion Policy | 28.0 | 0.6 | 14.30 |
| RDT-1B | 34.5 | 13.7 | 24.10 |
| π0 | 46.4 | 16.3 | 31.35 |
| X-VLA | 70.0 | 39.0 | 54.50 |
| π0.5 | 82.7 | 76.8 | 79.75 |
| ABot-M0 | 86.0 | 85.0 | 85.50 |
| Qwen-VLA (generalist) | 86.1 | 87.2 | 86.65 |
| **DyPES-VLA (generalist)** | **88.78** | **89.26** | **89.02** |

주목할 점: **randomized가 clean보다 높다**(89.26 > 88.78). 배경/조명/물체 배치를 교란해도 성능이 떨어지지 않는다는 것은 정책이 시각적 표면 통계가 아니라 dynamics에 의존한다는 간접 증거다. Diffusion Policy(28.0 → 0.6)나 X-VLA(70.0 → 39.0)의 붕괴와 대조된다. 부록 Table S2/S3의 per-task 결과를 보면 Hanging Mug(35/36), Blocks Ranking Size(63/58), Place Can Basket(58/74)이 하위권이고, Adjust Bottle·Shake Bottle·Move Pillbottle Pad 등은 100%다.

### RoboCasa-GR1 (Table 2, 29-DoF Fourier GR-1)
DyPES-VLA **59.25%**. 최강 per-benchmark specialist인 ABot-M0(58.3) 대비 +0.95, WAM baseline LDA-1B(55.4) 대비 +3.85, 동시기 generalist Qwen-VLA(56.7) 대비 +2.55. GR00T-N1.5는 48.2, GR00T-N1.6은 47.6.

부록 Table S1의 24개 task 분해: articulated receptacle로 pick-and-place하는 6개 task 평균 55.3%, container-to-container 18개 task 평균 60.6%로 두 task family 간 일관적이다. 최고는 TrayToPlate 82, 최저는 PotatoToMicrowaveClose 32 / PlacematToTieredshelf 34.

### LIBERO (Table 3, 7-DoF Franka Panda)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | **98.0** | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| Fast-WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| X-VLA | 98.2 | 98.6 | 97.8 | **97.6** | **98.1** |
| Qwen-VLA (generalist) | – | – | – | – | 97.9 |
| **DyPES-VLA (generalist)** | **98.8** | 99.4 | 97.0 | 96.8 | 98.0 |

X-VLA(98.1)에 0.1점 뒤지지만, X-VLA는 **벤치마크별 파인튜닝된** specialist다. 단일 checkpoint로 이 수준을 유지한다는 것이 요점.

### 실기 (Table 4, 25 rollouts × 3 tasks × 3 embodiments)

| | ACT | GR00T-N1.6 | DyPES-VLA |
|---|---|---|---|
| FR3 Kiwi→basket | 68 | 84 | **100** |
| FR3 Pour water | 44 | 76 | **92** |
| FR3 Book→shelf | 36 | 64 | **80** |
| Magic Kiwi→basket | 60 | 80 | **92** |
| Magic Pour water | 36 | 64 | **76** |
| Magic Book→shelf | 28 | 60 | **80** |
| G1 Kiwi→basket | 20 | 52 | **72** |
| G1 Pour water | 0 | 32 | **52** |
| G1 Book→shelf | 0 | 24 | **36** |
| **Average** | 32.4 | 59.6 | **75.6** |

ACT는 task×embodiment마다 별도 9개 checkpoint, GR00T-N1.6과 DyPES-VLA는 단일 joint checkpoint. G1 humanoid에서 ACT가 두 task에서 0%인 것과 대비해 DyPES-VLA가 52/36%를 내는 것이 gap이 가장 큰 지점 — 데이터가 적고 morphology가 어려울수록 cross-embodiment prior의 이득이 커진다는 해석이 가능하다.

## 7. Ablation 분석 (Table 5, 단일 co-trained checkpoint)

| Variant | RoboTwin | RoboCasa-GR1 | LIBERO |
|---|---|---|---|
| **DyPES-VLA (full)** | **89.02** | **59.25** | **98.0** |
| w/o future prediction | 86.67 (−2.35) | 56.75 (−2.50) | 96.1 (−1.9) |
| w/o Stage-1 pretraining | 87.76 (−1.26) | 58.33 (−0.92) | 96.7 (−1.3) |
| Shared dense head | 87.85 (−1.17) | 57.17 (−2.08) | 96.8 (−1.2) |
| w/o metadata | 88.50 (−0.52) | 58.75 (−0.50) | 97.6 (−0.4) |

기여도 순서가 명확하다: **future prediction(Q2) > MoE action head(Q3) > metadata(Q4)**. 특히 `w/o future prediction`(objective 자체 제거)이 `w/o Stage-1`(사전학습만 제거)보다 훨씬 큰 하락을 낳는다는 점이 흥미롭다 — 즉 future prediction의 이득은 "대규모 비디오 사전학습"보다 **Stage 2의 공동 regularization**에서 더 크게 온다.

### Future-Contact Probe (§4.5, Table 6)
행동 지표를 넘어 표현 내용을 직접 검증한 실험. Frozen query state `Z` 위에 single-layer linear probe를 학습해, action chunk horizon H=8 내 **contact onset**(현재 비접촉 `c_t=0`에서 접촉 시작 여부)과 **release**(현재 접촉 `c_t=1`에서 종료 여부)를 예측한다. 각 subset이 현재 접촉 상태를 고정하므로, 현재 상태를 그대로 되뇌는 trivial predictor는 판별력이 없다 — 성능은 오직 상태 **전이 예측**에서 와야 한다.

| Metric | w/o future pred. | Full DyPES-VLA |
|---|---|---|
| Onset AUROC ↑ | 95.2 | **97.3** |
| Onset AUPRC ↑ | 70.8 | **86.3** |
| Release AUROC ↑ | 92.3 | **94.4** |
| Release AUPRC ↑ | 64.8 | **72.8** |
| Weighted BCE ↓ | 0.227 | **0.142** |

Onset AUPRC +15.5점이 가장 큰 격차다. "dynamics prior를 학습한다"는 주장을 행동 성능이 아닌 표현 수준에서 뒷받침하는, 이 논문에서 가장 설득력 있는 증거.

## 8. 관련 연구 비교

저자들은 cross-embodiment VLA를 **embodiment specificity가 모델의 어디로 들어오는가**로 분류한다.
1. **공통 action space로 통합**: hand-engineered(공유 end-effector frame, interpretable action vector — RDT-1B) 또는 학습된 latent/universal action space(UniVLA, Moto). → 공유되는 것과 고유한 것이 뒤섞임.
2. **정책은 공유하고 embodiment context를 별도 공급**: soft prompt(X-VLA), motion-transfer training(Gemini Robotics), 단일 공유 action head.
3. **아키텍처 자체를 분할**: per-embodiment stem/head, readout(Octo), state encoder/action decoder(GR00T-N1), heterogeneity factor 기반 expert routing.

DyPES-VLA는 3번 계열이되, **공유 인터페이스를 future-supervised query state로 상류 이동**시킨 것이 차별점이다.

Predictive learning 축에서는 WAM 계열과의 구분이 명시적이다. UWM은 video/action diffusion을 **결합(couple)**하고, Fast-WAM은 추론 시 명시적 future 합성을 제거하며, LDA-1B는 dynamics/forecasting/policy를 이질 데이터 위에서 함께 스케일한다. DyPES-VLA는 **future prediction을 오직 shared prior 학습에만 쓰고 action 생성은 전용 head에 위임**한다 — 즉 결합이 아니라 **분리**가 설계 철학이다.

## 9. 한계 및 미해결 문제

- **K=3, static routing**: expert 수가 embodiment 수와 같고 라우팅이 결정론적이다. 새 embodiment 추가 시 새 expert + encoder/decoder를 붙이고 재학습해야 하며, "임의 embodiment로 확장 가능"이라는 주장의 비용이 정량화되지 않았다. Zero-shot으로 미학습 embodiment에 대응하는 실험은 없다.
- **Proprioception 미사용**: 설계 단순화에는 유리하나, 접촉이 풍부하거나 force feedback이 필요한 task에서 상한이 될 수 있다. 실기 pour water가 76~92%에 머무는 것과 관련이 있을 수 있다.
- **Metadata ablation이 약함(−0.5)**: 저자가 강조하는 "verbalized embodiment context"의 실질 기여가 작다. static routing이 이미 embodiment 구분 정보를 대부분 제공하기 때문일 가능성이 높은데, 이 상호작용은 분석되지 않았다.
- **LIBERO에서 X-VLA에 여전히 뒤짐**: 0.1점 차이지만, 단일 checkpoint의 이점이 "가장 잘 튜닝된 specialist를 능가"까지는 못 간다.
- **`λ_w = 0.05` 민감도 부재**: future loss 가중치 스윕이 없어, 2.4점을 좌우하는 하이퍼파라미터의 견고성을 알 수 없다.
- **실기 task가 3개**: kiwi/water/book 모두 pick-place 계열이라 morphology 다양성에 비해 task 다양성이 좁다.
- **동시기 Qwen-VLA와의 비교 공정성**: 학습 데이터/규모가 다를 수 있는데 통제되지 않았다.

## 10. 총평

이 논문의 기여는 새로운 loss나 새로운 아키텍처 블록이 아니라 **"공유 인터페이스의 위치"에 대한 재배치**다. 기존 연구가 action space에서 공유를 시도해 좌표 변환/IK/포맷 정렬이라는 부담을 지고 상호작용 규칙성과 제어 의미론을 뒤섞었다면, DyPES-VLA는 공유 지점을 query state로 끌어올리고 그 아래를 embodiment별 expert에 완전히 위임한다. Future generation head를 "표현을 만드는 도구"로만 쓰고 추론 시 버리는 결정도 깔끔하다 — WAM 계열의 추론 비용을 지지 않으면서 예측 감독의 이득만 취한다.

설득력을 만드는 것은 두 가지다. 하나는 **checkpoint scope를 명시한 정직한 baseline 분류**로, 단일 checkpoint가 파인튜닝된 specialist를 세 벤치마크에서 모두 매치/상회한다는 주장이 뭉개지지 않는다. 다른 하나는 **future-contact linear probe**로, "dynamics prior를 학습한다"를 성능 상승이라는 순환 논증이 아니라 표현의 선형 디코딩 가능성으로 검증했다. RoboTwin randomized(89.26)가 clean(88.78)보다 높은 것도 이 서사와 일관된다.

약점은 확장성 주장이 실증되지 않은 것이다. "임의 embodiment 확장 가능"이라 하지만 K=3 static routing에서 새 신체는 새 파라미터와 재학습을 요구하고, 미학습 embodiment 일반화 실험은 없다. metadata 기여가 −0.5에 그친 것도 저자 서사와 결과 사이의 작은 균열이다. 그럼에도 cross-embodiment VLA 설계에서 "무엇을 어디서 공유할 것인가"라는 질문을 명확한 실험으로 좁힌, 잘 통제된 논문이다.

## 11. 🔥 예상 날카로운 질문 모음

1. **Future generation head를 추론 시 버릴 거면, 왜 SANA-600M 같은 무거운 이미지 생성기가 필요한가?** 더 가벼운 latent 예측(예: DINO feature regression)으로 대체하면 어디서 손해가 나는가? — 생성 품질이 곧 prior 품질인지에 대한 근거가 없다.
2. **"생성기가 현재 관측을 받지 않는다"는 bottleneck이 정말 결정적인가?** 현재 관측을 함께 주면 query state가 dynamics를 인코딩할 유인이 사라진다는 주장의 ablation이 없다. 이 설계가 논문의 핵심 메커니즘인데 검증이 비어 있다.
3. **`w/o future prediction`(−2.4)이 `w/o Stage-1`(−1.3)보다 큰 이유는?** 대규모 EgoDex 사전학습의 순 기여가 절반 이하라면, 50% EgoDex 혼합 비율의 정당성은 무엇인가? Stage 1을 아예 시뮬레이션 비디오만으로 해도 되는가?
4. **Static routing 대비 learned gating을 왜 안 썼는가?** MoE의 통상적 이점(입력 의존적 특화)을 포기하고 embodiment ID로 고정한 것은 사실상 per-embodiment adapter인데, 이를 "MoE"라 부르는 것이 정당한가?
5. **Metadata 기여가 −0.5에 불과한데 §3.2에서 이를 핵심 설계로 서술하는 것이 과대평가 아닌가?** Static routing이 이미 embodiment를 알려주므로 metadata는 중복 정보일 수 있다. Routing 없이 metadata만 있는 조건은 왜 없는가?
6. **RoboTwin randomized > clean 역전을 어떻게 설명하는가?** 학습 데이터 혼합이 randomized 쪽에 치우쳤을 가능성, 혹은 clean setting 고유의 실패 모드(예: Hanging Mug 35 vs 36)는 없는가?
7. **Proprioception을 쓰지 않고 min-max 정규화된 action만 예측하는데, 실기에서 초기 자세가 다르면 어떻게 되는가?** Action horizon 50인 dual-arm에서 open-loop drift는 없는가?
8. **`Δ_e = H_e`로 future offset을 action horizon에 묶은 이유는?** dual-arm은 50 step 뒤 프레임을 예측하는데, 그 정도 미래는 예측 불가능에 가까워 학습 신호가 노이즈가 되지 않는가? single-arm(8)과 dual-arm(50)의 예측 난이도 격차는 어떻게 다뤄지는가?
9. **LIBERO Goal에서 97.0으로 π0.5(98.0)에 뒤지는데, goal-conditioned 세팅에서 dynamics prior가 덜 유용한 이유가 있는가?**
10. **K를 3에서 늘려 embodiment를 추가할 때 기존 embodiment 성능이 유지되는가?** Catastrophic forgetting과 expert 간 간섭에 대한 실험이 필요하다.
11. **Query token 수 N=96은 어떻게 정했는가?** N이 bottleneck 폭을 직접 결정하는데, N 스윕이 없으면 "정보가 query를 통과해야 한다"는 설계의 tightness를 알 수 없다.
12. **Beta(1.5, 1.0) 고노이즈 편향 스케줄이 4 Euler step 추론과 어떤 관계인가?** 스케줄과 step 수를 함께 튜닝했는가, 아니면 GR00T-N1 설정을 그대로 차용했는가?

## 12. 세미나 토론 포인트

- **"공유 인터페이스의 위치"라는 축**: latent action space(UniVLA/Moto) → soft prompt(X-VLA) → query state(DyPES-VLA)로 이어지는 상류 이동의 흐름을 놓고, 다음 이동 지점은 어디인가? VLM 내부 레이어 자체로 더 올라갈 수 있는가, 아니면 query state가 적정 지점인가?
- **분리 vs 결합**: UWM/LDA-1B처럼 future와 action을 결합하는 계열과, DyPES-VLA처럼 분리하는 계열 중 어느 쪽이 스케일에서 유리한가? Fast-WAM이 추론 시 future 합성을 제거한 것과 DyPES-VLA가 head 자체를 버리는 것은 실질적으로 같은 결론에 도달한 것인가?
- **Linear probe를 표준 평가 도구로**: contact onset/release AUPRC 같은 표현 수준 지표가 VLA 논문의 기본 평가로 자리잡을 수 있는가? 성공률만으로는 "무엇을 학습했는가"를 알 수 없다는 문제의식과 연결해 논의.
- **Static routing의 재해석**: 이것이 MoE인가 per-embodiment adapter인가라는 명명 문제를 넘어, 결정론적 라우팅이 오히려 cross-embodiment 학습의 안정성(load balancing loss 불필요, 간섭 없음)을 준다는 실용적 이점을 어떻게 평가할 것인가?
- **Randomized ≥ clean 현상**: domain randomization이 충분히 강하면 정책이 시각 통계 대신 dynamics에 의존하게 된다는 가설을 다른 벤치마크에서도 검증할 방법은?
- **실기 gap의 구조**: G1 humanoid에서 ACT 0% → DyPES-VLA 52%라는 격차가 "cross-embodiment prior의 이득"인지 "단순히 더 큰 사전학습 모델의 이득"인지 분리하려면 어떤 실험이 필요한가?

<!-- VERIFIED: pdf -->
