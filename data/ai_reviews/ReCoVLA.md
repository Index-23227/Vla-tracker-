# ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in Vision-Language-Action Policies

> **한 줄 요약**: 사전학습된 VLA(π0.5/OpenVLA)를 **얼린 채로** 외부 Qwen3-VL-8B VLM이 실패 모드를 인식해 **structured recovery descriptor**(category, stage, entities, confidence, reward mask)를 출력하고, 결정론적 reward compiler가 그것을 stage-aware gate가 붙은 PPO residual reward로 변환해 시뮬레이션에서 학습한 residual policy 라이브러리를 실제 Fetch 로봇에 zero-shot으로 배포 — 물리 평균 성공률 27%→62%, OOD 10%→53%까지 끌어올린 failure-conditioned residual recovery 프레임워크.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **VLA imitation 학습의 분포 외 취약성**: π0/π0.5 같은 flow-matching VLA는 nominal task에는 강하지만 객체를 떨어뜨리거나 잘못된 receptacle에 놓는 등 **action-induced off-distribution state**에서는 corrective behavior가 부족 (Ross et al. DAgger 문헌, [3]).
- **Recovery 데이터 수집의 비용·망각 문제**: VLA를 recovery 데이터로 fine-tuning하면 수집 비용이 크고, **catastrophic forgetting** (LoRA/EWC 한계, [6–8])이 발생.
- **VLA RL refinement의 어려움**:
  - Flow-matching VLA는 **action likelihood**가 없어 policy-gradient RL이 직접 적용 불가
  - π0.6 [11]은 offline RL로 우회하지만 **value function 별도 학습 + 비싼 human intervention** 필요
  - 일반 task-level reward는 sparse하고 ([9]), 모든 hand-designed reward를 동시에 켜면 reward hacking ([10])

### 핵심 질문
- VLA를 **건드리지 않은 채** 실패 모드별 corrective behavior를 어떻게 학습시킬 것인가?
- VLM에게 **action**이나 **scalar reward**를 직접 생성시키지 않고, 어떤 형태로 reward shaping에 활용해야 stable한가?
- Simulation에서 학습한 recovery를 **zero-shot으로 물리 로봇**에 옮길 수 있는가?

📌 [Figure 1 삽입] — Frozen VLA가 (a_t^b, h_t)를 출력. 별도 Qwen3-VL-8B-Instruct가 RGB+prompt 스트림에서 recovery descriptor를 추출. Reward compiler가 reward library와 결합해 residual-RL reward를 생성. Residual policy는 h_t를 입력받아 a_t^r을 출력, a_t = Π_A(a_t^b + β_t a_t^r)로 합성.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

세 단계 파이프라인:
1. **Failure catalog 구축 (sim)**: 얼린 base VLA(M1)의 실패 rollout을 VLM이 분석해 ξ_i = (c_i, z_i, E_i, ρ_i, m_i)를 추출. 고신뢰 카테고리 C_train = {c_i : ρ_i ≥ τ_cat} 구성.
2. **Reward 컴파일 + residual policy 학습 (sim)**: 각 c ∈ C_train마다 reward mask m_c와 active entity E_c로 컴파일된 reward로 PPO residual policy 학습.
3. **실세계 dispatch (real)**: VLM이 5-frame history (Δ=15s)로 monitoring → β_t = 1[ρ_t ≥ τ_deploy ∧ c_t ∈ C_train]일 때만 해당 residual 호출.

### 2.2 Structured Recovery Descriptor (핵심)

VLM은 **action도 아니고 scalar reward도 아닌** 다음 5개 필드를 출력:

| 필드 | 의미 | 예시 |
|------|------|------|
| c | failure category | wrong-broccoli-receptacle |
| z | recovery stage | regrasp / place |
| E | active entities | {gripper, broccoli, yellow_bowl, red_dish} |
| ρ | confidence | 0.91 |
| m ∈ {0,1}^K | reward mask | [1, 1, 1, 0] for {dist, grasp, place, close} |

> ❓ **예상 질문**: 왜 VLM에게 reward를 직접 출력시키지 않나?
> **답변**: 자유 형식 scalar reward는 **(1) 단위·스케일이 일관되지 않고 (2) reward hacking에 취약하며 (3) replay·재학습 시 비결정적**. 저자는 VLM을 **semantic field producer**로만 쓰고 결정론적 compiler가 grounding/binding을 책임지게 함으로써 "안전한 인터페이스"를 만든다 (논문 §3.3, Algorithm 1).

### 2.3 Stage-Gated Reward Compilation (M4의 핵심)

Reward library R = {φ_k}는 4개 primitive potential을 가짐 (Eq. 8):
- **φ_dist**(a,b) = max(-1, 1 - ‖p(a)-p(b)‖/d_init)
- **φ_grasp**(gripper, obj) = q_grasp ∈ [0,1] (시뮬레이터 grasp 품질 신호)
- **φ_place**(obj, target) = max(-1, 1 - ‖p(obj)-p(target)‖/d_init)
- **φ_close**(obj) = max(-1, 1 - |θ(obj)-θ_closed|/|θ_init-θ_closed|)

각 component k는 stage gate g_{c,k}(s_t; E_c) ∈ {0,1}를 가짐. 예) "place 단계의 reward는 broccoli가 잡힌 다음에만 활성화" — 이게 **grasp-before-place**, **contact-before-articulation** 같은 fixed gate template.

최종 reward (Eq. 7):
```
R_M4^c(s_t, a_t, s_{t+1}) = Σ_k m_c^(k) · g_{c,k}(s_t; E_c) · Δφ_k(s_t, s_{t+1}; E_c) - Ω_t(a^r)
```
where Ω_t(a^r) = λ_1‖a_t^r‖² + λ_2‖a_t^r - a_{t-1}^r‖² (smoothness regularization).

### 2.4 Deterministic Reward Compiler (Algorithm 1)

VLM 출력의 **불완전성을 보수적으로 거부**하는 게 핵심:
1. label normalize → c ∈ C_train, ρ ≥ τ, m ∈ {0,1}^K 검증
2. entity canonicalize (시뮬레이터 object map O로 grounding)
3. semantic role 할당 (end-effector / object / target / articulated part)
4. 각 component k의 **signature σ_k**에 따라 필요한 role만 binding (gripper를 articulation term에 넘기지 않음)
5. stage gate g_{c,k}(·) retrieve
6. 어느 단계든 실패 시 → **fallback reward를 합성하지 않고 recovery skip**

> ❓ **예상 질문**: VLM이 잘못된 mask를 내면 보호장치는?
> **답변**: ρ < τ거나 entity가 resolve 안 되면 descriptor 자체를 reject (Algorithm 1 line 3, 7, 16, 22). 잘못된 reward로 학습하지 않고 nominal VLA가 계속 실행. 단, **VLM이 그럴듯한 오분류를 했을 때** (table-vs-ground confusion 등)는 잘못된 residual policy를 호출할 수 있음 (Figure 7 confusion matrix에서 실제 관찰).

### 2.5 Residual Policy (Mixture of Experts)

- **입력**: 얼린 VLA의 latent h_t만 (시뮬레이터 state는 critic 학습에만 사용)
- **구조**: MLP (4096, 1024, 128) + Tanh
- **출력**: a_t^r ∈ [-1, 1] (스케일 0.25, mobile base 0.03, torso 0.01 — 작은 corrective motion)
- **합성**: a_t = Π_A(a_t^b + β_t a_t^r)
- **학습**: PPO, 300K timesteps per residual policy, γ=0.99, GAE λ=0.95, clip=0.2, lr=3e-4

> ❓ **예상 질문**: residual policy를 latent h_t에서만 받게 한 이유?
> **답변**: critic은 simulator state를 보지만 actor가 deployment 가능하려면 robot이 실시간으로 받을 수 있는 정보만 의존해야 함. h_t는 VLA가 이미 만드는 representation이므로 추가 perception 없이 사용 가능.

---

## 3. 데이터 전략

### Base VLA fine-tuning
- **태스크당 40 expert demo** (sim 20 + physical 20)
- π0.5 base (`gs://openpi-assets/checkpoints/pi05_base/params`) 위에 20K gradient step, batch 4, cosine decay, peak lr 2.5e-5, action horizon 32
- 동일 절차로 OpenVLA도 fine-tune

### Residual training
- Sim only (PPO 300K steps × residual policy 개수)
- **Failure-state initialization**: OmniGibson 시뮬레이터 state restoration으로 "잘못 놓인 broccoli", "떨어진 soda can" 등을 강제 재현

### Deployment
- VLM 5-frame history H_t = {I_{t-4Δ}, ..., I_t}, Δ=15s
- 학습된 카테고리 외 실패는 recovery 없음

> ❓ **예상 질문**: 태스크당 40 demo가 너무 적지 않나? Sim 20+Phys 20만으로 π0.5를 어떻게 fine-tune하나?
> **답변**: base VLA는 이미 사전학습된 일반 능력을 갖고 있고, fine-tuning은 nominal task에 정렬시키는 용도. 실패 데이터 자체는 fine-tuning에 들어가지 않고, **모두 residual policy 학습용 sim rollout으로 처리**된다는 게 핵심 디자인.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base VLA | π0.5 (M1–M4), OpenVLA (M5–M6) |
| VLM analyzer | Qwen3-VL-8B-Instruct (frozen, prompted) |
| Action horizon (π0.5) | 32 |
| Peak lr (π0.5 FT) | 2.5×10⁻⁵, cosine decay |
| Demo/task | 40 (20 sim + 20 real) |
| Residual actor/critic | MLP (4096, 1024, 128), Tanh |
| PPO clip / γ / GAE | 0.2 / 0.99 / 0.95 |
| Residual steps / policy | 300,000 |
| Residual action clip | [-1, 1], scale 0.25 (arm) / 0.03 (mobile base) / 0.01 (torso) |
| Action-delta penalty | 0.005 |
| Sign-flip penalty | 0.01 |
| Robot platform | Fetch mobile manipulator (main) + R1 Pro Galaxea (B1K) |
| Hardware | 본문 미명시 |

---

## 5. 실험 설계 및 평가 프로토콜

### 평가 변형 (M1–M6)
| Variant | Base VLA | Reward design |
|---------|----------|--------------|
| M1 | π0.5 | no recovery |
| M2 | π0.5 | task-level mask m_ℓ, no gates |
| M3 | π0.5 | failure mask m_c, no gates (ablation) |
| **M4 (ReCoVLA)** | **π0.5** | **failure mask m_c + stage gates g_{c,k}** |
| M5 | OpenVLA | no recovery |
| M6 | OpenVLA | failure mask + stage gates |

### 태스크 (Fetch tabletop, language-conditioned)
1. **Organizing toolbox** (contact-rich): 케이블+박스 toolbox에 넣고 뚜껑 닫기
2. **Sorting vegetables** (long-horizon): broccoli→red dish, corn→other red dish, pumpkin→yellow bowl
3. **Picking up trash cans** (short-horizon): 소다캔 3개를 trash can에 투입

각 method×task = **20 trials**. 평가 metric: binary success + **Q-score** (partial-credit, e.g. 케이블 0.30 + 박스 0.30 + 뚜껑 0.40 = 1.0).

3분간 Q-score 증가가 없거나 충돌 시 trial 종료, 인간 expert demo 평균의 2배가 max time.

---

## 6. 실험 결과 심층 분석

### Simulation (Table 4, in-distribution)

| Task | M1 | M2 | M3 | **M4** | M5 | M6 |
|------|----|----|----|--------|----|----|
| Sort veg | 0.30/0.48 | 0.20/0.28 | 0.45/0.55 | **0.65/0.82** | 0.20/0.30 | 0.45/0.52 |
| Trash cans | 0.55/0.67 | 0.70/0.79 | 0.60/0.71 | **0.75/0.88** | 0.40/0.45 | 0.60/0.68 |
| Toolbox | 0.25/0.54 | 0.30/0.59 | 0.40/0.64 | **0.60/0.78** | 0.10/0.23 | 0.30/0.45 |
| **Avg** | 0.37/0.56 | 0.40/0.55 | 0.48/0.63 | **0.67/0.83** | 0.23/0.33 | 0.45/0.55 |

- **M4 vs M1**: +30.0pp 성공률, +0.27 Q-score (π0.5 위에서)
- **M4 vs M3 (stage gate ablation)**: +18.4pp — gate가 핵심
- **M6 vs M5 (OpenVLA에 적용)**: +21.7pp — backbone-agnostic 증거

### Physical Fetch (Table 4)

| Task | M1 | M4 | gain |
|------|----|----|------|
| Sort veg | 0.25/0.36 | **0.60/0.72** | +35pp |
| Trash cans | 0.45/0.54 | **0.75/0.83** | +30pp |
| Toolbox | 0.10/0.30 | **0.50/0.69** | +40pp |
| **Avg** | 0.27/0.40 | **0.62/0.75** | **+34.7pp** |

- 물리 환경에서 M4가 모든 태스크 1위
- **toolbox contact-rich task에서 가장 큰 향상** — gate가 lid-close 같은 multi-stage 행동을 정확히 ordering

### OOD stress test (물리, object substitution)

| Substitution | M1 | M4 |
|--------------|----|----|
| pumpkin → tomato | 0.00/0.10 | **0.50/0.65** |
| can → tall can | 0.30/0.50 | **0.70/0.76** |
| cable → tape | 0.00/0.05 | **0.40/0.55** |
| **Avg** | 0.10/0.22 | **0.53/0.65** |

- M1은 두 태스크에서 **success 0**, M4는 +43.3pp 회복
- Recovery 디자인이 visual/geometric perturbation에 robust

### Behavior-1K (다른 robot, R1 Pro)

| Task | ReCoVLA | OpenPI Comet |
|------|---------|--------------|
| Sort veg | **0.20/0.51** | 0.05/0.26 |
| Bring wood | **0.65/0.79** | 0.50/0.62 |
| Lunch box | **0.25/0.49** | 0.10/0.33 |
| Avg | **0.37/0.60** | 0.22/0.40 |

- 다른 시뮬레이터·로봇으로의 transferability 일부 입증

### Recent recovery baselines (Table 2, sim)

| Task | ReCoVLA | RLinf [46] | RACER [43] |
|------|---------|-----------|-----------|
| Sort veg | **0.65/0.82** | 0.00/0.12 | 0.35/0.49 |
| Soda can | **0.75/0.88** | 0.35/0.40 | 0.65/0.74 |
| Toolbox | **0.60/0.78** | 0.00/0.05 | 0.30/0.56 |
| **Avg** | **0.67/0.83** | 0.12/0.19 | 0.43/0.60 |

- 최신 RL/recovery baseline 대비도 일관된 우위

### VLM Failure Detector Accuracy (Figure 7)

| Task | Macro acc | Aggregate acc | Std (pp) |
|------|-----------|--------------|----------|
| Sort veg (5 modes) | 87.4% | 87.5% | 1.0 |
| Soda can (2 modes) | 86.7% | 86.8% | 3.3 |
| Toolbox (3 modes) | 80.0% | 79.5% | 8.5 |
| **All** | **85.0%** | **84.6%** | 6.0 |

- VLM detector가 ~85% accuracy로 의미 있는 dispatch 신호 제공
- 오분류는 대부분 의미적으로 유사한 실패 간 confusion (table-vs-ground 등)

> ❓ **예상 질문**: M4 vs M3 격차의 정체는 진짜 "stage gate" 덕분인가?
> **답변**: M4와 M3는 **동일한 mask m_c, 동일한 reward components**를 사용. 차이는 binary gate g_{c,k}뿐. M4가 M3보다 sim +18.4pp, real +22pp 우위라는 건 gate가 "place reward를 grasp 전에 활성화하면 reward hacking이 발생"을 막는다는 가설과 일치. 다만 gate 자체는 fixed template라 데이터로 학습된 게 아님.

---

## 7. Ablation 분석

### 핵심 Ablation (M1 → M2 → M3 → M4 누적 디자인)
| 디자인 추가 | sim avg | Δ |
|-------------|---------|---|
| M1 (no recovery) | 0.37 | — |
| M2 (+ residual w/ task-level reward) | 0.40 | +0.03 |
| M3 (+ failure-relevant mask) | 0.48 | +0.08 |
| M4 (+ stage gates) | **0.67** | **+0.19** |

- 단순히 residual을 켜는 것(M2)으로는 거의 향상 없음 — sorting task에선 오히려 0.30→0.20으로 **하락** (모든 component 동시 활성화 → conflicting objective)
- **VLM-selected failure mask**가 의미 있는 구조를 주지만 (M3), **stage gate가 본체**
- M3→M4의 격차가 M1→M3의 격차보다 큼

### Backbone Transfer (M5/M6, OpenVLA)
| Setting | M5 (no recovery) | M6 (+ stage gates) |
|---------|----------------|-------------------|
| Sim avg | 0.23/0.33 | 0.45/0.55 |
| Real avg | 0.15/0.25 | 0.35/0.42 |

- π0.5만의 트릭이 아니라 OpenVLA에도 +21.7pp/+20pp 적용 가능

### Residual 정규화 (Table 5 설정)
- Action-delta penalty 0.005, sign-flip 0.01, residual scale 0.25 (arm) / 0.03 (mobile) / 0.01 (torso)
- 의도: residual이 base controller를 **대체하지 말고 보정**만 하도록 — 실제 sim-to-real 안정성에 기여한다고 §A.4에서 언급

---

## 8. 관련 연구 비교

| 방법 | Backbone 동결 | VLM 역할 | Recovery training | Online human-in-loop |
|------|--------------|---------|-------------------|---------------------|
| π0.6 RL [11] | ✗ (offline RL) | – | full policy | ✓ (비싸다) |
| RACER [43] | ✗ (LLM language guidance) | text guidance | imitation | – |
| FailSafe [44] | △ | reasoning + action | – | – |
| RLinf [46] | ✗ | – | full policy RL | – |
| **ReCoVLA** | **✓** | **structured descriptor** | **residual MoE in sim** | **✗ (sim-to-real zero-shot)** |

### 핵심 차이
- **VLM이 action도 reward scalar도 출력하지 않음** — semantic field만 출력하고 결정론적 compiler가 grounding
- **Backbone freeze + residual in latent space** — catastrophic forgetting 회피
- **Mixture-of-experts dispatch** — 각 failure category마다 별도 residual

---

## 9. 한계 및 미해결 문제

### 명시적 한계 (저자 §6)
1. **Sim reproducibility requirement**: 모든 recoverable failure는 simulator에서 재현 가능해야 함. 새로운 실패 모드는 sim에 추가 후 새 residual을 학습해야 하며 **online policy synthesis는 불가**.
2. **Empirical scale**: Fetch 3 태스크 × 20 trials. 본격적인 manipulation diversity는 미평가.
3. **Sim-to-real gap**: residual이 sim only로 학습되어 perception/contact variation에 영향받음.
4. **VLM failure classification error**: 평균 85% accuracy. 오분류 시 잘못된 residual policy 호출 가능 — 보호장치는 ρ threshold뿐.

### 분석가 추가 지적
5. **Stage gate가 hand-crafted template**: grasp-before-place 같은 gate는 사람이 작성한 fixed template (§3.3). **학습 가능한 gate 또는 LLM 생성 gate**가 future work라고 인정.
6. **Reward library R의 확장성**: K=4 primitive (dist, grasp, place, close)로는 deformable object, fluid pouring 같은 태스크를 다루기 어려움.
7. **Residual scale tuning이 robot-specific**: arm 0.25 / mobile 0.03 / torso 0.01은 Fetch에 맞춰진 값.
8. **β_t binary gate**: continuous blending이 아니라 hard switch — switching 순간 jerk 가능성.
9. **20 trials × 1 robot platform** — binomial std error bar가 크다 (논문 figure에서도 명시).
10. **Compute disclosure 부족**: GPU 종류·개수·총 학습 시간 미보고.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLM을 action/scalar reward generator가 아니라 **structured descriptor producer**로 쓰고 결정론적 compiler가 grounding하는 인터페이스 설계가 깔끔 |
| **Technical depth** | ★★★★☆ — Algorithm 1의 conservative rejection, stage gate, role-binding signature 등 시스템 디테일 풍부 |
| **Experimental rigor** | ★★★★☆ — M1–M6 ablation + OOD + B1K + 두 base VLA + recent baseline 비교 + confusion matrix까지 체계적. 다만 trial 수 20개로 통계력 제한 |
| **Practical impact** | ★★★★☆ — Backbone freeze + sim-to-real zero-shot은 실제 산업 환경에서 매력적. residual MoE가 demand되는 failure mode마다 incremental하게 확장 가능 |
| **Writing quality** | ★★★★☆ — 명료한 6 variant 구성과 reward 수식 (Eq. 5–7)의 누적 비교가 ablation 흐름을 잘 보여줌 |

**강점**: VLA를 동결한 채 **외부 VLM + structured reward compilation**으로 recovery를 분리한 게 깔끔한 디자인. **+30pp simulation / +35pp physical** 향상은 실용적으로 의미 있는 수치. Stage gate가 단순 mask 대비 +18pp 추가 이득을 낸다는 점이 reward hacking 회피의 중요성을 명확히 보여준다. OpenVLA로의 transfer (M6)와 B1K (다른 로봇)에서도 작동하는 게 backbone/robot agnostic 가능성을 시사.

**약점**: Recovery catalog가 sim에서 미리 학습 가능한 failure mode로 한정. Stage gate가 hand-crafted template. VLM 오분류 시 wrong residual dispatch 가능성. 20 trials의 통계적 한계.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VLM에게 직접 reward를 생성시키지 않은 이유? | Free-form scalar reward는 단위·스케일 불일치, reward hacking, 비재현성 문제. Structured field + 결정론적 compiler가 안전한 인터페이스 |
| 2 | M3 vs M4 격차가 진짜 stage gate 덕인가? | M3와 M4는 동일 mask·동일 components, gate만 다름. sim +18.4pp / real +22pp 우위가 gate 효과의 직접 측정 |
| 3 | 20 trial은 통계적으로 충분한가? | 저자도 binomial std error bar로 표시. 실제로 일부 차이는 1-sigma 안에 들 수 있음. 하지만 OOD/B1K/baseline 비교까지 일관되게 우위 |
| 4 | Real-world에서 VLM latency는? | Δ=15s 주기로 5-frame을 분석 — 8B VLM이라 단일 inference 수 초. 빠른 실패 모드(예: 미끄러지는 grasp)에는 부적합 |
| 5 | β_t binary switching이 jerk를 만들지 않나? | 가능. 논문은 residual scale을 작게 (arm 0.25) + action-delta penalty 0.005로 완화. continuous β_t blending은 미구현 |
| 6 | Stage gate template은 누가 정하나? | 본 논문에서는 fixed template (grasp-before-place 등). 저자도 "learn or generate reward gates"를 future work로 언급 |
| 7 | OpenVLA에서 21.7pp 향상이 base가 약해서 (M5=0.23) 쉽게 얻은 게 아닌가? | 일부 사실. 다만 같은 절대 수준(M6=0.45)이 M3(0.48 sim, π0.5 위)와 비슷한 영역이므로 design transferability는 입증됨 |
| 8 | VLM 85% accuracy로도 dispatch가 의미 있는가? | Figure 7에서 오분류 대부분이 의미적으로 유사한 실패 (table-vs-ground), 이 경우 trained residual이 부분적으로 generalize. 다만 toolbox는 std 8.5pp로 더 위험 |
| 9 | π0.5 fine-tuning이 demo 40개라는 게 너무 적지 않은가? | Base는 사전학습 능력 보유. 40 demo는 task alignment용. Residual에는 실패 데이터가 들어가지 않고 sim rollout만 사용 — 데이터 부담을 imitation에서 RL로 옮긴 디자인 |
| 10 | Reward library K=4로 어디까지 커버되는가? | Pick/place/articulation 위주. Pouring, deformable, soft body는 미커버. φ_k 추가는 새로운 simulator state signal 필요 |
| 11 | π0.6의 offline RL 대비 장점은? | π0.6은 value function 별도 학습 + human intervention 필요. ReCoVLA는 base를 동결하므로 forgetting 없고, recovery만 분리 학습 |
| 12 | Residual을 latent h_t로 받게 한 trade-off? | 장점: deployment 시 추가 perception 불필요. 단점: VLA가 latent에 충분한 정보를 담고 있어야 함 — π0.5/OpenVLA에선 동작하지만 latent가 빈약한 backbone에선 어려울 수 있음 |

<!-- VERIFIED: pdf -->
