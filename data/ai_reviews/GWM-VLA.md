# GWM-VLA: Geometry-Aware Latent World Modeling for Vision-Language-Action Learning

> **한 줄 요약**: 여러 카메라 뷰를 **각각** 인코딩하던 기존 latent world model(VLA-JEPA)과 달리, 동결된 **VGGT-Ω** aggregator로 매 timestep의 multi-view 관측을 **함께** 집계해 geometry-aware latent state를 만들고, 그 중 **wrist view의 patch token만** 다음 스텝으로 예측한다(register token은 global geometric context). 나아가 world model과 flow-matching action head를 **동일한 latent action token**으로 조건화한다. LIBERO 97.1%(OpenVLA-OFT와 공동 1위), LIBERO-Plus **76.9%** SOTA(robot-only VLA-JEPA 대비 +14.0pt), 실기(SO-101) 평균 0.67.

- arXiv: 2608.07619 (2026-08-07, cs.RO) · Tongji University · AAAI 2027 저작권 표기

---

## 1. 배경 및 동기

### 문제 정의
- VLA policy는 강력한 조작 성능을 보이지만 **카메라 시점, 조명, 배경 텍스처, 물체 배치, 센서 노이즈**의 통제된 변화만으로도 크게 무너진다(LIBERO-Plus, Fei et al. 2025).
- 이는 policy가 "행동에 따라 장면이 어떻게 변하는가"를 포착하기보다 **appearance-specific correlation**에 의존하고 있음을 시사한다.

### 기존 world modeling의 두 갈래와 한계
1. **Explicit world model** — DreamZero(미래 비디오+행동 동시 생성), π₀.₇(별도 world model이 multi-view subgoal 이미지를 생성해 VLA에 제공). 미래 시각 관측을 실제로 생성해야 하므로 **텍스처·조명·배경 같은 control과 무관한 디테일까지 모델링**해야 한다.
2. **Latent world model** — VLA-JEPA(Sun et al. 2026). 미래 이미지를 만들지 않고 표현을 예측하므로 훨씬 compact. **다만** 사전학습된 V-JEPA2 인코더가 **각 카메라 뷰를 독립적으로** 인코딩하고 결과를 concat할 뿐이어서, cross-view 대응관계·카메라 상대 장면 구조 같은 **기하 정보는 암묵적으로만** 남는다.

### 이 논문의 진입점
VGGT / VGGT-Ω 같은 feed-forward geometric foundation model은 여러 뷰를 **공동 집계**하며 cross-view 기하를 명시적으로 인코딩한다. 그러나 기존 연구(3D-Mix, Yang et al. 2026)는 이를 **지각 입력 / 공간 prior**로만 썼을 뿐, **world-model objective로 그 시간적 변화를 모델링한 사례는 없었다.** GWM-VLA는 정확히 그 빈 칸을 채운다.

📌 [Figure 1] — VLA-JEPA(단일뷰 독립 인코딩 + concat, 별도 embodied action token) vs GWM-VLA(multi-view 기하 집계 + target-view 예측 + 공유 latent action)의 mechanism-level 비교

---

## 2. 방법론 심층 분석

궤적 세그먼트: τ = (O_{0:T}, ℓ, s_{0:T}, u_{0:T-1}), 여기서 O_t = {I_t^v}_{v=1}^V 는 multi-view 관측, ℓ은 language instruction, s_t는 proprioceptive state, u_t는 실행된 action. **T는 prediction horizon과 action horizon이 공유하는 값**이라는 점이 뒤에서 중요해진다.

### 2.1 Geometry-Aware Multi-View State Encoding

$$\{R_t, P_t\} = E_\Omega(\{I_t^v\}_{v=1}^V)$$

- `E_Ω` = **VGGT-Ω 인코더, 학습 내내 동결(frozen)**
- `R_t` = register token, `P_t` = patch token
- **핵심 비대칭**: 식은 **timestep마다 독립적으로** 적용되지만, **같은 timestep의 모든 뷰는 공동 집계**된다. 즉 공간적으로는 뷰를 묶고, 시간적으로는 풀어놓은 뒤 그 시간축을 world model이 담당하게 만든 설계다.

### 2.2 Global Context-Conditioned Target-View Prediction — 이 논문의 심장

전체 multi-view state를 예측하면 예측 목표가 **모든 카메라 스트림에 분산**된다. GWM-VLA는 대신 target view `v*` 하나를 골라

$$r_t = R_t^{v^\star},\qquad p_t = P_t^{v^\star}$$

- `p_t` = **예측 대상**이 되는 target-view latent state
- `r_t` = latent world model에 주는 **global geometric context**
- 두 토큰 집합 모두 **cross-view interaction 이후에** 뽑히기 때문에, "한 뷰만 예측하면서도 multi-view 기하는 유지한다"는 절충이 성립한다. 이 문장이 방법론 전체의 논리적 축이다.

실험에서는 **wrist view**를 선택한다 — end-effector 운동과 국소 gripper-object 상호작용에 가장 밀접하기 때문.

world model:

$$\hat{p}_{1:T} = F_\phi(p_{0:T-1},\, r_{0:T-1},\, A_{0:T-1})$$

**Time-causal attention mask**:

$$M_{ij} = \begin{cases} 0, & \tau(j) \le \tau(i) \\ -\infty, & \tau(j) > \tau(i)\end{cases}$$

같은 timestep 내 토큰은 양방향으로 상호작용하고, timestep t의 토큰은 t 이하의 토큰만 attend한다.

**Teacher forcing**: p_{t+1} 예측 시 ground-truth p_t를 제공하며, 지도 타깃은 동일한 per-timestep geometry encoder로 미래 관측을 인코딩해 얻는다 — p_{t+1} = E_Ω^{P,v*}(O_{t+1}).

손실은 ℓ₁:

$$\mathcal{L}_{wm} = \frac{1}{T}\sum_{t=0}^{T-1}\lVert \hat{p}_{t+1} - p_{t+1}\rVert_1$$

### 2.3 Unified Latent Action Representation

VLM 백본은 **Qwen3-VL-2B**. 현재 관측과 instruction에 **timestep-grouped learnable latent action query**를 삽입한다:

$$Q^A = (Q_0^A, \dots, Q_{T-1}^A),\qquad Q_t^A = (q_{t,1}^A, \dots, q_{t,K}^A)$$
$$A_{0:T-1} = Q_\theta(O_0,\, \ell,\, Q^A),\qquad A_t = [a_{t,1},\dots,a_{t,K}]$$

> ⚠️ **자주 오해하는 지점**: 모든 timestep-grouped latent action은 **현재 관측 O₀ 하나에서만** 생성된다. t는 "따로 관측된 policy 입력"이 아니라 **미래 action horizon 내의 위치 인덱스**다. horizon T를 latent prediction과 action generation이 공유하기 때문에, 예측되는 각 transition과 action-chunk 위치에 latent-action group이 **1:1로** 대응된다.

VLA-JEPA는 action 생성을 위해 **별도의 embodied action query**를 추가하지만, GWM-VLA는 `A_{0:T-1}`을 **그대로 재사용**해 world model과 action head를 모두 조건화한다. 그 결과 latent prediction objective와 robot-action objective가 **같은 표현을 함께 조형**한다.

### 2.4 Conditional Flow-Matching Action Head

u_γ = (1-γ)ε + γ·u_{0:T-1}, ε ~ N(0,I), γ ~ U(0,1)일 때

$$\mathcal{L}_{action} = \mathbb{E}\big[\lVert v_\psi(u_\gamma,\gamma \mid A_{0:T-1}, s_0) - (u_{0:T-1}-\epsilon)\rVert_2^2\big]$$

### 2.5 Joint Objective

$$\mathcal{L} = \mathcal{L}_{action} + \lambda\mathcal{L}_{wm},\qquad \lambda = 0.1\ \text{(default)}$$

**world-model loss는 auxiliary training objective이며 배포 시에는 사용되지 않는다.** 즉 추론 비용 증가가 없다.

📌 [Figure 2] — Qwen3-VL → latent action token → (Latent World Model + Action Head)로 갈라지는 전체 아키텍처. 눈송이=동결(VGGT-Ω), 불꽃=학습 가능.

---

## 3. 실험 설정

| 항목 | 내용 |
|---|---|
| VLM 백본 | Qwen3-VL-2B (fully trainable) |
| Geometry encoder | VGGT-Ω, **frozen**, 기본 Layer 24 표현 |
| λ (WM loss weight) | 0.1 |
| Pretraining | **DROID** — 약 76,000 demonstration trajectory (다기관·다작업자, 다양한 시점/배치/배경) |
| Sim fine-tuning | LIBERO 표준 demonstration set |
| 하드웨어 | 8× NVIDIA A800 (pretrain+finetune) / 1× RTX 6000D (ablation) |
| Ablation 조건 | LIBERO-Spatial only, batch 16, 10,000 step |

**Benchmarks**: LIBERO(Franka Panda, Spatial/Object/Goal/LIBERO-10 4개 suite, in-distribution) + LIBERO-Plus(동일 task family에 camera viewpoint / robot initial state / language / illumination / background / visual noise / object layout 7개 perturbation, OOD robustness).

**Baselines**: LAPA, UniVLA, OpenVLA-OFT, π₀, π₀-FAST, CoT-VLA, WorldVLA, villa-X, GR00T N1, π₀.₅, VLA-JEPA. **GWM-VLA는 robot demonstration만으로 사전학습되므로, human-video pretraining이 없는 robot-only VLA-JEPA가 가장 직접적인 비교 대상**이다.

---

## 4. 주요 결과

### 4.1 LIBERO (Table 1, 단위 %)

| Method | Spatial | Object | Goal | LIBERO-10 | **Avg** |
|---|---|---|---|---|---|
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| VLA-JEPA (w/o human videos) | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |
| **GWM-VLA** | 96.8 | 99.0 | 98.0 | **94.4** | **97.1** |

→ OpenVLA-OFT와 **공동 1위(97.1)**, robot-only VLA-JEPA 대비 **+1.0pt**. In-distribution 성능을 훼손하지 않으면서 기하·예측 supervision을 얹었다는 것이 핵심 메시지다.

### 4.2 LIBERO-Plus (Table 2, 단위 %) — 진짜 승부처

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg** |
|---|---|---|---|---|---|---|---|---|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | **93.3** | **75.8** | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | **79.0** | 68.9 | 53.6 |
| π₀-FAST | **65.1** | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| VLA-JEPA (w/o human videos) | 40.3 | **55.7** | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |
| **GWM-VLA** | 57.9 | 54.7 | **89.8** | **95.4** | 90.8 | 72.5 | **77.1** | **76.9** |

- **SOTA 평균 76.9%** — OpenVLA-OFT 대비 **+7.3pt**, robot-only VLA-JEPA 대비 **+14.0pt**
- Language / Light / Layout에서 **1위**, Camera / Robot / Background에서 **2위**
- VLA-JEPA 대비 **가장 큰 개선은 visual noise(38.2→72.5, +34.3), background(70.5→90.8, +20.3), camera(40.3→57.9, +17.6), language** — 정확히 "공동 집계된 기하 상태가 시각·환경 변화에 더 안정적인 공간 표현을 준다"는 가설이 예측하는 패턴이다.

### 4.3 Real-World (SO-101, Figure 3)

LeRobot 배포 스택, 5개 pick-and-place task에 대해 **100개 teleoperated demo**, 각 평가 task 10회 시도.

| Method | ID | OOD (task) | OOD (object layouts) | Average |
|---|---|---|---|---|
| π₀ | 0.57 | 0.07 | 0.17 | 0.27 |
| π₀.₅ | 0.70 | **0.60** | 0.60 | 0.63 |
| **GWM-VLA** | **0.80** | 0.53 | **0.67** | **0.67** |

→ 평균 최고, 그리고 **ID와 OOD-layout에서 최고**. 반면 held-out object–receptacle 재조합을 요구하는 **OOD-task에서는 π₀.₅(60.0%)에 뒤진다(53.3%)**. 저자들은 이를 두고 **"공간적 robustness와 compositional instruction generalization은 서로 다른 도전 과제"**라고 정직하게 적는다.

---

## 5. Ablation Study

모두 LIBERO-Spatial only, 1× RTX 6000D, batch 16, 10K step의 **경량 진단 실험**이며 본 실험 수치와 직접 비교 불가.

### 5.1 Target View 선택 (Figure 4 좌)
| Target view | Success rate |
|---|---|
| Third-person | 87.8% |
| Mixed (wrist 80% / third 20%) | 92.4% |
| **Wrist** | **92.8%** |

Wrist가 최선. Mixed가 wrist 단독보다 아주 근소하게 낮다는 점이 흥미롭다 — 예측 목표를 분산시키는 것 자체가 손해라는 본문 논리와 일관된다.

### 5.2 VGGT-Ω Representation Depth (Figure 5)
| Layer | 5 | 12 | 18 | **24** |
|---|---|---|---|---|
| Success | 89.0% | 89.4% | 92.0% | **92.8%** |

층이 깊어질수록 단조 증가. 선행 연구(Bratulić et al. 2026; You et al. 2026)가 보고한 "cross-view correspondence와 epipolar geometry는 중간층에서 출현해 후반층에서 정제된다"는 관찰과 부합하며, **latent world modeling은 더 정제된 후반층 기하 표현에서 이득**을 본다.

### 5.3 Shared Latent Action의 효과 (Figure 4 우)
| Action conditioning | Success rate |
|---|---|
| Additional embodied query | 88.0% |
| **Direct latent-action conditioning** | **92.8%** |

**+4.8pt.** VLA-JEPA식 별도 embodied query보다 latent action을 그대로 공유하는 편이 낫다 — 이 논문의 세 번째 설계 주장을 직접 뒷받침한다.

---

## 6. 정성 분석: 예측된 Latent State는 기하를 담고 있는가

**관측된** 다음 스텝 VGGT-Ω 토큰으로 시각화 디코더를 학습시켜 frozen dense head의 depth map을 맞추게 한 뒤, 그 디코더를 **동결하고 예측된 토큰에 적용**한다. 디코딩 결과는 **주요 장면 레이아웃, gripper 주변과 조작 대상 물체의 기하를 보존**하되 미세 디테일은 뭉개진다.

> ⚠️ 저자들 스스로 이것이 **3D reconstruction accuracy의 측정이 아니라 qualitative probe**임을 명시한다. 과대해석 금지.

📌 [Figure 6] — Simulation / Real World 각각에 대해 Current / Predicted Next / Ground-Truth Next의 depth·point cloud 비교

---

## 7. 이 논문의 기여 정리

1. **Geometry-aware latent world modeling 개념 제안** — 공동 집계된 multi-view 기하 표현을 *predictive state*로 사용하고 그 action-conditioned 시간 전이를 모델링.
2. **GWM-VLA 프레임워크** — geometry-aware latent world modeling + flow-matching action generation을 *공유 latent action 표현*으로 통합.
3. **시뮬레이션·실기 검증** — LIBERO 97.1, LIBERO-Plus 76.9(SOTA), SO-101 실기 평균 최고.

---

## 8. 비판적 검토

### 강점
- **설계 3요소가 각각 ablation으로 뒷받침된다** (target view / depth / shared latent action). 서사와 증거의 대응이 깔끔하다.
- **추론 비용 중립**: WM loss는 auxiliary이며 배포 시 제거된다. robustness 개선을 latency 없이 얻는 구조.
- **공정한 비교 기준 설정**: robot-only VLA-JEPA를 "가장 직접 비교 가능한 baseline"으로 명시해, human-video pretraining 유무의 교란을 스스로 통제한다.
- **음성 결과를 숨기지 않는다**: OOD-task에서 π₀.₅에 진다는 사실을 본문에 그대로 적고 해석한다.

### 약점 및 의문
- **Ablation 규모가 매우 작다.** 10K step, batch 16, 단일 suite, 단일 GPU. 92.8 vs 92.4(mixed) 같은 0.4pt 차이는 seed noise와 구분되지 않는다. seed variance나 error bar가 보고되지 않는다.
- **LIBERO 97.1은 사실상 포화 구간**이다. OpenVLA-OFT와 동률이며, 이 벤치마크만으로는 기여를 변별할 수 없다. 논문의 실질적 주장은 전부 LIBERO-Plus에 실려 있다.
- **Camera perturbation에서 π₀-FAST(65.1)에 진다.** 기하 인식을 핵심으로 내세운 모델이 정작 시점 변화 항목에서 1위가 아니라는 점은 설명이 더 필요하다. VGGT-Ω가 카메라 상대 구조를 인코딩한다면 이 항목이 가장 강해야 한다는 자연스러운 기대와 어긋난다.
- **λ = 0.1의 근거**가 본문에 없다(appendix의 sensitivity analysis로 미룸). world-model loss가 실제로 얼마나 기여하는지 판단하려면 λ=0 대조가 본문에 있어야 한다 — **엄밀히 말해 "world model 자체의 ablation"이 본문에 없다.**
- **Multi-view 요구가 구조적 제약**이다. 저자들도 인정하듯, 동시 multi-view 관측이 필요하므로 **large-scale human-video pretraining에 직접 적용할 수 없다** — 이는 VLA-JEPA가 취할 수 있었던 확장 경로를 스스로 닫는 선택이다.
- **Target view가 고정**이다. LIBERO-Spatial에서는 wrist가 최선이지만 task·센서 구성에 따라 최적 뷰는 달라질 수 있다.
- **실기 규모가 작다**: task당 10회 시도, 총 100 demo. 0.80 vs 0.70 같은 차이의 신뢰구간은 넓다.

---

## 9. 예상 Q&A

> ❓ **VGGT-Ω를 동결한 채로 그 표현공간에서 예측하는 것이 왜 유효한가?**
> 동결 인코더는 **안정적인 타깃 분포**를 제공한다. 인코더가 함께 학습되면 latent 예측 손실이 표현 붕괴(collapse)로 최소화될 수 있다. 또한 teacher-forcing 타깃 p_{t+1} = E_Ω^{P,v*}(O_{t+1})이 동일한 동결 인코더에서 나오므로 예측과 타깃의 좌표계가 일치한다.

> ❓ **왜 wrist view만 예측하는가? 정보 손실 아닌가?**
> 손실이 아니다 — 예측 대상 토큰이 **cross-view aggregation 이후에** 추출되기 때문에, wrist patch token 안에 이미 다른 뷰의 기하 정보가 섞여 있다. 여기에 register token이 global context로 추가된다. 즉 "예측 범위를 좁히되 정보 범위는 좁히지 않는" 구조다.

> ❓ **VLA-JEPA와의 본질적 차이는 결국 인코더 교체인가?**
> 아니다. 세 가지가 동시에 바뀐다: (1) 독립 인코딩+concat → 공동 기하 집계, (2) 전체 multi-view 예측 → register 조건부 target-view 예측, (3) 별도 embodied action query → 공유 latent action. 5.3의 +4.8pt는 (3)만으로도 유의미한 기여가 있음을 보여준다.

> ❓ **world model 없이 VGGT-Ω만 백본으로 써도 되지 않나?**
> 그것이 바로 3D-Mix / Yang et al. 2026 계열이며, 논문은 이들을 "기하 표현을 지각 입력·공간 prior로만 사용"한다고 위치시킨다. GWM-VLA의 주장은 그 표현의 **시간적 전이를 예측 목표로 삼는 것**이 추가 이득을 준다는 것이다. 다만 앞서 지적했듯 λ=0 대조가 본문에 없어 이 주장의 직접 증거는 상대적으로 약하다.

> ❓ **teacher forcing인데 배포 시 exposure bias 문제는?**
> WM은 배포에 사용되지 않으므로 **문제가 되지 않는다.** WM은 순수히 학습 시 표현 조형 장치이며, 추론 경로는 Qwen3-VL → latent action → flow-matching head뿐이다.

---

## 10. 관련 연구 지형

| 축 | 대표 연구 | GWM-VLA의 위치 |
|---|---|---|
| Geometry for VLA | Spatial Forcing, SpatialVLA, SUGAR, Lift3D, PointVLA, 3D-VLA, 3D-Mix | 기하를 **입력/prior**가 아니라 **predictive state**로 사용 |
| Explicit world model | DreamZero(WAM), π₀.₇, WorldVLA, CoT-VLA | 미래 픽셀 생성 회피 |
| Latent world model | **VLA-JEPA** | 직접적 전신(前身). 인코딩·예측 범위·action 조건화 모두 재설계 |
| Latent action learning | LAPA, UniVLA, villa-X, MVP-LAM | 이들은 action 표현 학습에 집중; GWM-VLA는 **latent 환경 상태의 미래 전개**를 함께 모델링 |

---

## 11. 재현 및 실무 관점

- **코드 미공개**(2026-08 기준), open_source: false.
- 재현 장벽: DROID 76K trajectory 사전학습 + 8×A800. 사전학습 없이 LIBERO fine-tune만으로는 본문 수치 재현 불가.
- 실무 적용 시 검토할 점:
  - **동시 multi-view 관측이 필수**다. 단일 카메라 셋업에는 그대로 적용 불가.
  - VGGT-Ω가 동결이므로 **inference에 geometry encoder forward가 매 timestep 추가**된다(WM은 제거되지만 인코더는 남는다). 논문에 inference Hz 보고가 없어 실시간성 판단이 어렵다.
  - Layer 24 사용이 기본이라는 점, λ=0.1이 기본이라는 점은 그대로 채택 가능한 실용적 기본값이다.

---

## 12. 총평

**기여의 성격**: 완전히 새로운 패러다임이라기보다, VLA-JEPA라는 명확한 전신에 대해 **"무엇을 latent state로 삼을 것인가"를 기하 기반으로 교체하고, "무엇을 얼마나 예측할 것인가"를 좁히고, "예측과 행동이 같은 표현을 쓰게 한" 세 겹의 정제**다. 각 결정이 ablation과 1:1로 대응한다는 점에서 논문의 구성은 모범적이다.

**설득력의 무게중심**은 LIBERO가 아니라 **LIBERO-Plus 76.9%(+14.0pt over robot-only VLA-JEPA)** 에 있다. visual noise +34.3pt, background +20.3pt라는 개선 패턴은 "appearance correlation 의존을 기하 표현으로 대체한다"는 서사와 정합적이며, 우연한 튜닝 결과로 보기 어렵다.

**가장 아쉬운 점**은 본문에 **world-model loss 자체의 on/off 대조(λ=0)가 없다**는 것이다. 세 설계 요소 중 (2)(3)은 ablation이 있으나, "latent world modeling을 한다"는 제목 그 자체의 기여는 appendix로 밀려 있다. 또한 기하 인식을 표방하면서 camera perturbation 1위를 놓친 점은 후속 연구가 파고들 지점이다.

**한계의 정직성**은 높이 평가할 만하다 — multi-view 요구로 인한 human-video pretraining 확장 불가, 고정 target view, OOD-task에서의 열위를 모두 본문에 명시한다. 저자들이 제시한 후속 방향(단일뷰 비디오로의 확장, adaptive target-view selection, multi-target prediction)은 정확히 이 한계들을 겨냥한다.

**한 문장으로**: *"뷰를 따로 보지 말고 함께 보되, 예측은 한 뷰만 하고, 예측과 행동은 같은 표현을 공유하라"* — 이 세 문장이 LIBERO-Plus에서 14포인트로 환산된 논문.

<!-- VERIFIED: pdf -->
