# RedFlow: Redirect Failure into Action-Level Corrections for Flow-matching VLA Policy

> **한 줄 요약**: 배포 중 발생한 **실패 롤아웃을 버리지 않고**, GRM 기반 task-progress + proprioceptive state로 정의한 execution context에서 HDBSCAN 클러스터링을 돌려 "비슷한 상황의 성공 action"을 **action-chunk 단위 corrective target**으로 회수하고, flow-matching velocity field 위에서 attraction / suppression / correction 3항 비대칭 목적함수로 재주입하는 **strictly offline** post-training 프레임워크 — LIBERO 평균 56.2%→68.2%, 실기 56.7%→74.7%, 온폴리시 RL 대비 약 1/10 샘플.

- **arXiv**: 2607.27782 (2026-07-30, preprint)
- **기관**: The Hong Kong University of Science and Technology (HKUST)
- **저자**: Zhengyang Yan†, Junhao Li†, Fangqi Zhu†, Zijun Wang, Quanxin Shou, Yikun Miao, Zicong Hong, Xiaoyi Pang, Song Guo* († equal contribution)

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Flow-matching VLA의 compounding error**: π0/π0.5 계열은 noise→action velocity field를 학습해 multimodal action 분포를 잘 모델링하지만, 결국 human demo에 대한 **imitation learning**으로 훈련되므로 behavior cloning의 근본 한계인 distribution shift 하 오차 누적을 그대로 물려받는다. 배포 중 학습 분포를 벗어난 state에 들어가면 작은 예측 오차가 스텝마다 쌓여 회복 불가능한 실패로 수렴한다.
- **Online RL은 비용 문제**: PPO/GRPO/DDPO 계열은 지속적 개선이 가능하지만 실로봇에서 fresh rollout을 반복 수집해야 해 스케일이 안 된다.
- **Offline RL 진영의 이분법 (Fig. 1)**:
  - **Classical offline RL (AWR 등)**: 성공 행동을 reweight/모방 → **실패에서 배우지 못함**. 실패 안에 담긴 진단 정보가 통째로 버려진다.
  - **Preference-based (DPO 등)**: 실패를 쓰긴 하지만 **trajectory-level 비교**에 그침 → "무엇을 피하라"만 말하고 "**어떻게 고쳐라**"는 말하지 못한다 (action-level guidance 부재).
  - **Human-in-the-loop 개입**: 두 성질을 다 만족하지만 전문가 노동이 필요해 확장 불가.

### 핵심 질문
> 사람 개입 없이, 이미 수집된 실패 경험을 **단순한 "나쁜 action으로부터의 반발"을 넘어 명시적 supervisory target을 제공하는 action-level correction**으로 정밀하게 재지향(redirect)할 수 있는가?

### 저자가 짚은 두 가지 근본 난제
1. **Granularity mismatch**: trajectory-level 실패 라벨은 실제로 붕괴를 유발한 희소한 action-level 오류를 가린다. 실패한 롤아웃 안의 중간 action 다수는 완전히 합리적이다. dense human supervision 없이 실패 유발 스텝을 격리하고 그 정확한 교정 타깃을 뽑아내는 것이 어렵다.
2. **Uniform imitation 함정**: 기존 flow-matching 목적함수는 주어진 모든 action을 모방해야 할 유효 샘플로 취급한다. mixed-quality 데이터에서는 suboptimal 행동을 그대로 학습하거나 over-correction으로 무너진다.

---

## 2. 방법론 심층 분석

### 2.1 문제 설정
언어 조건부 조작을 MDP로 둔다. action-chunk step $t$에서 관측 $o_t = (I_t, q_t)$ (시각 입력 + proprioceptive state $q_t \in \mathbb{R}^{D_q}$), 지시 $l$을 받아 policy $\pi_\theta(a_t | o_t, l)$가 **action chunk** $a_t \in \mathbb{R}^{K \times D}$ ($K$개 연속 $D$-DoF 명령)를 예측한다. 로봇이 chunk를 실행한 뒤 다음 관측을 받는다. 에피소드는 궤적 $\tau = \{(o_t, a_t)\}$와 **이진 결과 라벨** $y_\tau \in \{0,1\}$을 낳는다.

Flow-matching policy는 noise $x_1 \sim \mathcal{N}(0,I)$에서 시작해 velocity field $v_\theta(x_n, n, o_t, l)$를 $n=1 \to 0$으로 적분해 $a_t = x_0$를 얻는다. 사전학습은 표준 flow-matching loss $\mathcal{L}_{FM} = \mathbb{E}\|v_\theta - u_n\|^2$.

문제: 고정 버퍼 $\mathcal{D}$는 chunk-level advantage $A^\pi(o_t, a_t, l) = Q^\pi - V^\pi$를 주지 않는다. 있는 건 trajectory-level 결과뿐.

### 2.2 Context-Aware Corrective Matching (Stage II)

**(a) Action-level advantage 추정.**
사전학습된 **General Reward Model (GRM)** $R(o_t, l) \in [0,1]$을 task-progress 추정기로 쓴다. raw GRM 점수가 인접 chunk 간 요동치므로 half-window $W$의 box filter로 평활:

$$\bar{p}_t = \frac{1}{2W+1}\sum_{j=t-W}^{t+W} R(o_j, l)$$

(경계는 one-sided averaging). 그 다음 **local progress 변화 + trajectory-level outcome bias**를 결합해 signed score를 만든다:

$$\hat{A}_t = \bar{p}_{t+W} - \bar{p}_{t-W} + b \cdot (2\cdot\mathbb{1}[y_\tau=1] - 1)$$

- 앞 항: chunk $a_t$ 주변에서 궤적이 실제로 전진했는가
- 뒤 항 ($b>0$): local progress가 모호할 때 성공/실패라는 coarse 정보를 주입
- $\hat{A}_t > 0$ → positive, $< 0$ → negative. 0인 경우는 corrective-target 할당에 쓰지 않고 soft weight로만 처리.

**(b) Execution context 정의와 클러스터링.**
저자의 핵심 관찰: **progress-state 공간에서 가까운 chunk는 최종 결과가 달라도 같은 subtask에 대응하는 경우가 많다.**

$$f_t = [\tilde{q}_t \,;\, \beta \bar{p}_t]$$

$\tilde{q}_t$는 정규화된 proprioception, $\beta > 0$는 progress와 state의 스케일 균형. task별로 $\{f_t\}$ 위에서 **HDBSCAN** 클러스터 $\{\mathcal{C}_c\}$를 만든다.
→ 고차원 visual space에서 직접 클러스터링하는 것을 회피하면서도, GRM progress가 task-relevant 시각 정보를 암묵적으로 담게 하는 영리한 설계.

**(c) Corrective target 구성.**
클러스터 $\mathcal{C}_c$의 positive 부분집합 $\mathcal{C}_c^+ = \{i \in \mathcal{C}_c : \hat{A}_i > 0\}$이 비어있지 않으면, 그 안의 negative chunk $a_t$에 대해 **advantage-softmax 가중 centroid**를 타깃으로:

$$\alpha_i = \frac{\exp(\hat{A}_i/\kappa)}{\sum_{j \in \mathcal{C}_c^+}\exp(\hat{A}_j/\kappa)}, \qquad a_t^\star = \sum_{i \in \mathcal{C}_c^+} \alpha_i a_i$$

$\kappa$가 작으면 최고 advantage action에 가까워지고, 크면 균일 centroid에 가까워진다.

> **개념적으로 중요한 지점**: 저자는 $a_t^\star$를 **1:1 counterfactual replacement로 해석하지 말라**고 명시적으로 못박는다. 이것은 "empirical positive barycenter"이며, 확률 질량을 failure mode에서 밀어내는 **local transport direction**을 정의할 뿐이다.

**(d) Uncorrectable 판정.**
HDBSCAN이 outlier로 표시했거나 $\mathcal{C}_c^+ = \emptyset$인 negative chunk는 **uncorrectable**로 지정 — 타깃 없이 suppression만 받는다. (Ablation에서 이 분리가 가장 큰 기여를 한다.)

### 2.3 Adaptive Redirection Objective (Stage III)

**(a) Quality-weighted attraction.**
$$w_t = \sigma(\hat{A}_t / T_w) \in (0,1), \qquad \mathcal{L}_{att} = w_t \cdot \|v_\theta(x_n, n, o_t, l) - u_n\|^2$$
확실히 positive면 weight≈1, 확실히 negative면 ≈0. 저품질 chunk에도 약한 data-support 신호는 남긴다.

**(b) Failure suppression (repulsive hinge).**
linear flow-matching 보간 하에서 예측된 clean action $\hat{x}_0 = x_n - n \cdot v_\theta(x_n, n, o_t, l)$, 재구성 오차 $e_t = \|\hat{x}_0 - a_t\|^2$에 대해:

$$\mathcal{L}_{sup} = \lambda_{sup}(1-w_t)\max(0, m - e_t)$$

$m$은 **adaptive margin** — 재구성 오차의 stop-gradient running average로 구현. 예측이 negative chunk에서 충분히 멀어지면 반발 페널티가 자동으로 비활성화된다(유한 범위 배제).

**(c) Target-guided correction.**
$$\mathcal{L}_{cor} = c_t \cdot \lambda_{cor}(1-w_t)\|\hat{x}_0 - a_t^\star\|^2, \qquad c_t \in \{0,1\}$$
$c_t$는 correctable failure 여부 지시자. suppression은 "여기서 나가라"만 말하고 방향을 안 주므로, 이 항이 "어디로 갈지"를 지정한다.

**(d) 최종 목적함수**
$$\mathcal{L} = \mathbb{E}_{(o_t,a_t,l)\sim\mathcal{D},\, n}[\mathcal{L}_{att} + \mathcal{L}_{sup} + \mathcal{L}_{cor}]$$

### 2.4 Theorem 1 (Bounded endpoint redirection)
예측 endpoint $h$, 저품질 chunk $a^-$, corrective target $b$에 대해 endpoint energy
$$\phi(h) = \lambda_{cor}\|h-b\|^2 + \lambda_{sup}[m - \|h-a^-\|^2]_+$$
의 최소값은 **"$a^-$ 주변 유한 margin ball 바깥에 머무르면서 corrective target에 가장 가까운 endpoint"** — 즉 obstacle ball $\{h: \|h-a^-\|^2 < m\}$의 여집합 위로의 $b$의 projection. (형식적 진술과 증명은 Appendix A, Theorem 6.)

논문 표현으로는 **local Wasserstein push-pull transport**: $\mathcal{L}_{att}$가 positive endpoint에 anchoring, $\mathcal{L}_{sup}$이 유한 범위 배제, $\mathcal{L}_{cor}$이 corrective transport를 담당.

---

## 3. 데이터 전략

### LIBERO (시뮬레이션)
- 4개 suite: Spatial, Object, Goal, Long — 각 10개 task.
- Base policy: **π0**를 **πRL** 체크포인트로 초기화 (`RLinf-Pi0-SFT-Spatial-Object-Goal`, `RLinf-SFT-Pi0-LIBERO-Long`).
- Offline RL 전, base policy는 **pruned expert set** $\mathcal{D}_{exp}$로 학습: Spatial/Object/Goal는 **58 demos**, Long은 **208 demos**. (일부러 성능을 낮춰 개선 여지를 만드는 세팅.)
- Offline fine-tuning: $\mathcal{D} = \mathcal{D}_{exp} \cup \mathcal{D}_{roll}$, $\mathcal{D}_{roll}$은 suite당 **1,536 mixed-quality rollout**.
- 평가: suite당 **500 에피소드**.

### 실로봇
- **dual-arm Agilex Cobot Magic** — 전면 1대 + 손목 2대, 총 3 카메라.
- 3개 task: clothes folding (양팔 dexterous), object sweeping (도구 매개 상호작용), table cleaning (pick-and-place).
- Base: OpenPi 공식 `pi0_base` 체크포인트에서 task별 expert demo로 **50,000 step** fine-tune.
- Expert demos: folding 600 / sweeping 200 / cleaning 100.
- Offline RL rollout: 200 / 100 / 100. 평가 100 에피소드/task.

---

## 4. 시스템/학습 세부사항

| 항목 | Spatial / Object / Goal | Long |
|---|---|---|
| Policy update epochs | 30 | 30 |
| Progress smoothing window $W$ | 10 | **20** |
| Outcome-bias coefficient $b$ | 0.15 | **0.25** |
| Soft-weight temperature $T_w$ | 3.0 | 3.0 |
| Attraction coefficient | 1 | 1 |
| Suppression strength $\lambda_{sup}$ | 0.3 | **0.1** |
| Correction strength $\lambda_{cor}$ | 0.3 | **0.1** |
| Corrective-target temperature $\kappa$ | 1.0 | 1.0 |
| Max episode length | 240 | **480** |
| Action-chunk length $K$ | 5 | 5 |
| Training environments | 64 | 64 |
| SFT demonstrations | 58 | **208** |

**Optimizer**: AdamW, lr $5\times10^{-5}$, $(\beta_1,\beta_2)=(0.9,0.95)$, weight decay 0.01, grad clip 1.0, micro/global batch **128 / 2048**. 하드웨어: base policy fine-tuning에 **8×H20**.

**HDBSCAN 설정 (4개 suite 공통)**: `min_cluster_size=15`, `min_samples=5`, `progress_weight β=5.0`, `state_dim=7` (gripper 차원 제외), StandardScaler z-score 정규화, centroid softmax temperature 1.0.

**Long suite 하이퍼파라미터 조정 논리 (저자 설명)**:
- $W$↑, $b$↑: 긴 궤적은 단일 스텝 progress 신호가 더 noisy → 평활 창을 넓히고 episode-level outcome 비중을 올림.
- $\lambda_{sup}$↓: 실패 에피소드 안에도 "해롭지 않고 그저 진전에 기여 안 하는 neutral chunk"가 상당수 있는데, $\lambda_{sup}$이 크면 이들까지 무차별 억제해 policy 분포가 과집중되며 학습이 불안정해진다. 긴 horizon에서는 억제 신호가 더 누적되므로 0.3→0.1.
- $\lambda_{cor}$↓: $a^\star$는 attraction 항을 대체하려는 게 아니라 mild bias일 뿐 → 작은 값이 오히려 선호됨.

---

## 5. 실험 설계 및 평가 프로토콜

연구 질문 4가지:
1. mixed-quality 데이터에서 기존 offline RL baseline 대비 성능은?
2. 성공/실패 롤아웃 활용, Context-Aware Corrective Matching, Adaptive Redirection Objective 각각의 기여는?
3. online RL 대비 sample efficiency는?
4. 실로봇으로 전이되는가?

**공정성 통제**: 모든 방법이 **동일한 base policy, 동일한 offline buffer $\mathcal{D}$, 동일한 평가 프로토콜**을 공유. baseline은 AWR(advantage-reweighted imitation)과 DPO(trajectory-level preference).

---

## 6. 실험 결과 심층 분석

### LIBERO 4-suite (Table 1, 성공률 %)

| Method | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| Base Policy | 63.6 | 61.6 | 48.6 | 50.8 | 56.2 |
| AWR | 71.2 | 66.8 | 57.8 | 53.4 | 62.3 |
| DPO | 65.8 | 69.8 | 51.8 | 51.2 | 59.7 |
| **RedFlow (Ours)** | **75.8** | **70.4** | **71.2** | **55.2** | **68.2** |

- 4개 suite **전부** 최고. 평균 56.2 → 68.2 (**+12.0p**), AWR 대비 **+5.9p**, DPO 대비 **+8.5p**.
- **가장 극적인 격차는 LIBERO-Goal**: 71.2 vs AWR 57.8 / DPO 51.8 — **+13.4p / +19.4p**. Goal suite는 같은 장면에서 목표만 바뀌므로 "어느 chunk가 잘못된 목표로 향했는가"를 국소화하는 능력이 직접적으로 보상받는다. 저자도 실패 유발 chunk의 명시적 국소화 + 유사 context에서 회수한 타깃으로의 재지향 효과라고 해석.
- **Long은 상대적으로 약함**: 55.2 (base 50.8, +4.4p). AWR(53.4)과의 격차도 1.8p뿐. 긴 horizon에서 progress 추정 noise와 신호 누적 문제가 남아 있음을 시사 (그래서 $\lambda$를 1/3로 줄여야 했다).

### 실로봇 (Figure 6, 성공률 %)

| Method | Clothes Folding | Object Sweeping | Table Cleaning | **Avg** |
|---|---|---|---|---|
| Base Policy | 36.0 | 63.0 | 71.0 | 56.7 |
| AWR | 41.0 | 69.0 | 76.0 | 62.0 |
| DPO | 48.0 | 66.0 | 78.0 | 64.0 |
| **RedFlow** | **67.0** | **73.0** | **84.0** | **74.7** |

- 모든 task에서 전 baseline 상회, 평균 **56.7 → 74.7 (+18.0p)**.
- 최대 이득은 **clothes folding: 36.0 → 67.0 (+31.0p)** — 가장 어려운 bimanual dexterous task에서 가장 크게 개선된 것이 인상적. 보통은 어려운 task일수록 offline 방법이 힘을 못 쓰는데 반대 양상.
- 시뮬(+12.0p)보다 실기(+18.0p) 이득이 더 큼. base policy가 더 약하고 실패 모드가 더 구조적(반복적)이라 corrective matching이 걸릴 여지가 크기 때문으로 보인다.

### 창발적 recovery 행동 (Figure 7)
clothes folding에서 T-shirt가 오른팔 도달 범위 밖으로 떨어지면 base policy는 오른팔로 계속 헛잡기만 하고 회복 못 한다. RedFlow는 **왼팔로 옷을 끌어와 도달 가능한 배치로 만든 뒤 folding을 재개**한다.

> 이 행동은 추가 demo 없이 **offline buffer 재사용만으로 창발**했다: Context-Aware Corrective Matching이 유사 progress-state context에서 "왼팔 당기기" action을 회수해 타깃으로 삼고, Adaptive Redirection Objective가 실패한 grasp 시도를 그쪽으로 재지향했기 때문. — 논문에서 가장 설득력 있는 정성적 증거.

### Sample efficiency (Figure 5, LIBERO-Spatial)
- RedFlow: **1,536 offline 궤적**, 추가 환경 상호작용 0 → 75.8%.
- On-policy baseline은 매 update step마다 **1,024 fresh rollout** 수집. 같은 성공률 도달에 PPO **13** / GRPO **16** / DDPO **24** step 필요 → 각각 약 **13K / 16K / 24K 궤적**.
- 즉 **약 한 자릿수(order of magnitude) 적은 샘플**로 동급 성능. "structured failure reuse가 상당량의 on-policy 상호작용을 대체할 수 있다"는 주장의 핵심 근거.

### Corrective-target 할당 진단 (Table 7, Appendix E)

| Suite | Suppression only (%) | Supp. + Corr. (%) | Mean $\hat{A}^\star$ |
|---|---|---|---|
| Spatial | 86.3 | 13.7 | +0.216 |
| Object | 89.5 | 10.5 | +0.274 |
| Goal | 89.3 | 10.7 | +0.256 |
| **Avg** | **88.4** | **11.6** | **+0.243** |

> **가장 흥미로운 숫자**: negative chunk의 **11.6%만** 실제로 corrective target을 받는다. 나머지 88.4%는 suppression만. 즉 성능 향상의 상당 부분이 "소수의 정밀한 redirection"에서 나온다는 뜻이며, 동시에 "왜 correction만 빼도 69.1%나 나오는가"(ablation)를 설명해준다. 이 희소성은 정밀성의 증거이기도 하고, 방법의 커버리지 한계이기도 하다.

---

## 7. Ablation 분석 (Table 2, Spatial/Object/Goal 3-suite 평균)

| Method | Spatial | Object | Goal | **Avg** |
|---|---|---|---|---|
| **RedFlow (full)** | **75.8** | **70.4** | **71.2** | **72.5** |
| *Rollout data composition* | | | | |
| w/o failure rollouts | 71.4 | 67.4 | 65.2 | 68.0 (−4.5) |
| w/o success rollouts | 64.4 | 65.8 | 57.0 | 62.4 (−10.1) |
| *Context-Aware Corrective Matching* | | | | |
| w/o uncorrectable-failure separation | 66.4 | 65.8 | 50.8 | **61.0 (−11.5)** |
| *Adaptive Redirection Objective* | | | | |
| w/o $\mathcal{L}_{cor}$ | 70.4 | 68.4 | 68.6 | 69.1 (−3.4) |
| w/o $\mathcal{L}_{sup}$ | 70.8 | 66.8 | 68.8 | 68.8 (−3.7) |
| w/o $\mathcal{L}_{sup}$ & $\mathcal{L}_{cor}$ | 63.8 | 62.8 | 70.4 | 65.7 (−6.8) |

**(i) 데이터 구성**: 실패를 빼면 68.0 (억제·재지향에 필요한 negative 증거 상실), 성공을 빼면 62.4로 더 떨어짐 (고품질 anchor도 corrective target도 사라짐). 둘 다 있어야 72.5.

**(ii) Uncorrectable 분리 — 표에서 가장 큰 낙폭 (−11.5p)**: 유사한 성공 context 존재 여부와 무관하게 모든 failure chunk에 타깃을 강제 할당하면 72.5 → 61.0으로 붕괴하고, **LIBERO-Goal에서 무려 20.4p 손실**. 즉 "언제 교정하지 *않을지*를 아는 것"이 "어떻게 교정할지"보다 더 중요하다. 이 논문의 진짜 기여는 여기에 있다고 볼 수도 있다.

**(iii) 두 failure-side 항의 상보성**: $\mathcal{L}_{sup}$만 제거 68.8, $\mathcal{L}_{cor}$만 제거 69.1 — 개별 기여는 3~4p로 비슷하고 비교적 작다. 그런데 **둘 다 제거하면 65.7**로 낙폭이 개별 합보다 커진다(non-additive). 전자는 failure-prone 영역을 배제하고 후자는 확률 질량이 갈 곳을 지정하므로 중복이 아니라 상보적.
- 단, `w/o L_sup & L_cor` 조건의 **Goal 점수 70.4**는 full(71.2)에 거의 근접해 눈에 띈다. Goal suite에서는 quality-weighted attraction만으로도 상당 부분 설명되며, 3-suite 평균 낙폭은 Spatial(75.8→63.8)이 주도한다.

---

## 8. 관련 연구 비교

| 축 | Classical offline RL (AWR, IQL) | Preference (DPO/RLHF류) | Human-in-the-loop (DAgger, HG-DAgger) | Reward modeling (GRM 등) | **RedFlow** |
|---|---|---|---|---|---|
| 실패에서 학습 | ✗ | ✓ (trajectory-level) | ✓ | 평가만 | ✓ |
| Action-level 지침 | 부분 (advantage weight) | ✗ | ✓ | ✗ | ✓ |
| 명시적 corrective target | ✗ | ✗ | ✓ (사람이 제공) | ✗ | ✓ (버퍼에서 회수) |
| 사람 개입 필요 | ✗ | ✗ | **✓** | ✗ | ✗ |
| 추가 온라인 롤아웃 | ✗ | ✗ | ✓ | — | ✗ |
| Flow-matching 파라미터화 정합 | 부분 | 부분 | — | — | ✓ (velocity field 위 직접 정의) |

- **ReCoVLA / 최근 failure-recovery 계열과의 대비**: ReCoVLA는 VLA를 얼린 채 외부 VLM이 실패를 진단해 reward를 컴파일하고 **residual policy를 PPO로 온라인(시뮬) 학습**한다. RedFlow는 정반대로 **base policy 자체를 strictly offline으로 갱신**하며, 교정 신호를 외부 모델의 semantic 판단이 아니라 **자기 버퍼 안 성공 사례의 기하학적 회수**에서 얻는다. 전자는 novel failure에 semantic 일반화가 가능하나 시뮬레이터 의존, 후자는 시뮬레이터 불필요하나 버퍼 커버리지에 갇힌다.
- **GRM 활용의 차별점**: 기존 reward model 연구는 GRM을 평가자(evaluative)로만 쓴다. RedFlow는 GRM 출력을 **클러스터링 좌표축**으로 재해석해 매칭 구조를 만든다 — 이 재해석이 방법론적으로 가장 신선한 부분.

---

## 9. 한계 및 미해결 문제

**저자가 밝힌 한계 (Appendix F)**
1. Corrective target 할당이 **task-progress 추정 품질과 progress-state 클러스터링 품질에 의존**. progress 신호가 부정확하거나 클러스터링이 noisy하면 failure→success 매칭이 어긋난다.
2. **오프라인 버퍼에 인근 positive support가 있는 실패만** 건설적으로 재지향된다. novel / OOD / 성공 대응물이 없는 실패 모드는 억제만 될 뿐 명시적으로 교정되지 않는다.
3. LIBERO + 3개 실로봇 task, **고정 embodiment/세팅**에 국한. 더 다양한 로봇·시각 조건·long-horizon 검증 필요.

**리뷰어 관점의 추가 지적**
4. **Table 7의 11.6%**: 실제 correction을 받는 negative chunk 비율이 매우 낮은데, 그렇다면 ablation의 $\mathcal{L}_{cor}$ 기여(−3.4p)가 이 소수 chunk에서 나온다는 뜻이다. 인상적이지만 동시에 취약하다 — 버퍼 구성이 조금만 달라져도 이 11.6%가 흔들릴 수 있고, 분산/시드 정보가 제시되지 않았다.
5. **GRM 자체가 blackbox 의존성**: 어떤 GRM 체크포인트인지, 그 GRM이 실로봇 task를 본 적 있는지, GRM 품질 대비 성능 곡선(sensitivity)이 없다. 사람 개입을 없앴다고 하지만 **사전학습된 reward model이라는 별도 supervision을 도입**한 것이므로 "human-free"는 부분적으로만 사실이다.
6. **HDBSCAN이 7차원 proprio + 1차원 progress에서 동작**: 시각 정보는 오직 GRM progress 스칼라를 통해서만 들어온다. 로봇 자세와 진행률이 같아도 장면 배치가 다른 경우(예: 다른 물체 위치)를 구분 못 한다. LIBERO-Spatial에서 잘 되는 이유가 오히려 궁금해지는 지점.
7. **Single-iteration만 검증**: iterative offline-online 확장을 future work로 남겼는데, 재수집 없이 여러 iteration 돌리면 policy가 버퍼에서 멀어지면서 매칭이 무의미해질 가능성이 크다.
8. **Sample efficiency 비교의 공정성**: on-policy baseline은 step당 1,024 rollout을 강제하는 세팅인데, 이 batch 크기 선택이 baseline에 불리하게 작용했을 수 있다. 또한 RedFlow의 1,536 궤적은 base policy가 이미 수집한 것으로 "공짜"가 아니다.
9. **분산 미보고**: LIBERO 500 에피소드 평가지만 seed별 표준편차/신뢰구간이 없다. Long suite의 +4.4p 같은 작은 차이는 통계적 유의성 판단이 불가능하다.

---

## 10. 총평

**핵심 기여를 한 문장으로**: "실패 궤적은 버려야 할 노이즈가 아니라, **비슷한 상황에서 성공한 행동과 짝지을 수만 있다면** action-level supervision으로 변환 가능한 자산이다"는 명제를, flow-matching velocity field 파라미터화와 정합적인 목적함수로 구현해 보인 논문.

**강점**
- **개념적 명료함**: Fig. 1의 2×2 프레이밍(실패 학습 여부 × action-level 여부)이 기여를 즉시 이해시킨다. 빈 칸을 정확히 채운다.
- **저비용 매칭 설계**: 고차원 시각 임베딩 클러스터링 대신 (7-dim proprio + progress 스칼라)라는 극도로 저렴한 공간을 쓴 것이 실용적이면서도 잘 작동한다.
- **Theorem 1의 역할**: 형식적으로 강한 정리는 아니지만, hinge margin + attraction 조합의 최소해가 "obstacle ball 밖으로의 projection"이라는 기하학적 직관을 명확히 해주고 하이퍼파라미터 해석($\lambda_{sup} \geq \lambda_{cor}$)에 근거를 준다.
- **Uncorrectable 분리라는 negative result의 크기**: −11.5p라는 최대 낙폭이 "무분별한 교정은 해롭다"는 비자명한 교훈을 준다.
- **창발적 회복 행동의 정성적 증거**: 왼팔로 옷을 끌어오는 사례는 "buffer 재사용만으로 새 스킬이 나온다"는 주장을 시각적으로 설득한다.

**약점**
- GRM이라는 외부 의존성에 대한 sensitivity 분석 부재.
- LIBERO-Long에서의 미미한 개선(+4.4p)이 long-horizon 확장성에 대한 우려를 남긴다 — 그런데 저자의 미래 방향에도 long-horizon이 들어가 있다.
- 분산/시드 정보 부재로 일부 결론의 통계적 견고성이 불확실.
- correction을 받는 chunk가 11.6%뿐이라, 이름이 강조하는 "Redirect"보다 실질적으로는 "정교한 suppression + 소량의 redirect"에 가깝다.

**영향력 전망**: π0/π0.5 계열 flow-matching VLA의 **배포 후 post-training 레시피**로서 실용적 가치가 크다. 특히 "실로봇에서 이미 굴린 롤아웃이 쌓여 있는데 그걸로 뭘 할 것인가"라는 현장 문제에 직접 답한다. 온라인 RL이 어려운 실로봇 환경에서 DPO/AWR을 대체하는 기본 baseline이 될 가능성이 있다.

**점수: 9.5/10** — 문제 정의가 날카롭고, 방법이 단순하면서 파라미터화와 정합적이며, ablation이 자기 방법의 가장 취약한 가정(무분별 교정)을 스스로 겨냥해 검증했다. 실기 +18p와 창발적 양팔 회복 행동은 강력한 증거다. GRM sensitivity와 분산 보고 부재가 감점 요인.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 예상 답변 / 논점 |
|---|---|---|
| 1 | GRM이 없으면 방법이 성립하나? "human-free"라는 주장은 정직한가? | GRM은 사전학습된 외부 supervision이다. 사람의 **per-step 개입**은 없앴지만 progress supervision 자체는 외부에서 왔다. 저자는 GRM을 pretrained 자산으로 전제하며 sensitivity를 보이지 않았다 — 논문의 가장 큰 미검증 가정. |
| 2 | 7차원 proprio + 1차원 progress로 정의한 context가 정말 "같은 subtask"를 뜻하나? | 저자의 논거는 "GRM progress가 task-relevant 시각 정보를 암묵적으로 인코딩한다"는 것. 하지만 물체 배치가 다르고 자세·진행률만 같은 두 chunk는 구분되지 않는다. $\beta=5.0$으로 progress 축을 지배적으로 만든 것이 이 위험을 부분적으로 완화한다. |
| 3 | $a_t^\star$가 1:1 대체 action이 아니라면 $\|\hat{x}_0 - a_t^\star\|^2$로 직접 당기는 게 정당한가? | 저자는 $\lambda_{cor}$를 작게(0.3, Long은 0.1) 두고 $(1-w_t)$로 감쇠시켜 "mild bias"로만 작동하게 설계했다고 답한다. attraction 항이 여전히 관측 데이터에 anchoring한다. Theorem 1이 결과가 유계임을 보장. |
| 4 | LIBERO-Long 개선이 왜 그렇게 작은가 (+4.4p)? | 긴 horizon에서 (a) GRM progress 추정이 noisy해지고 (b) suppression/correction 신호가 누적돼 $\lambda$를 1/3로 낮춰야 했으며 (c) 실패 원인이 단일 chunk가 아니라 분산된 누적일 가능성이 크다. 방법의 구조적 한계. |
| 5 | negative chunk의 11.6%만 correction을 받는데, 그럼 이름값을 하는가? | 정직하게는 "정밀한 소량 redirect + 광범위 suppression". 다만 ablation은 그 11.6%가 3.4p를 만든다고 보여준다. 저자 관점에서는 무분별 할당이 −11.5p를 내므로 희소성이 곧 정밀성. |
| 6 | Uncorrectable 분리 제거가 Goal에서 −20.4p인 이유는? | Goal suite는 같은 장면에서 목표만 달라지므로 progress-state가 유사한 chunk가 많고, positive가 없는 클러스터에도 억지 타깃을 할당하면 **다른 목표의 action으로 끌려간다**. 구조적으로 오매칭에 가장 취약한 suite. |
| 7 | on-policy baseline 비교가 공정한가? | step당 1,024 rollout 고정이라는 설정이 baseline에 유리/불리하게 작용할 수 있고, RedFlow의 1,536 궤적도 base policy 배포 비용을 이미 치른 것이다. 다만 "이미 있는 데이터 재사용 vs 신규 수집"이라는 논지는 유효. |
| 8 | Adaptive margin $m$을 재구성 오차의 running average로 둔 이유는? | 고정 margin은 스케일 의존적이고 학습 초기·후기에 의미가 달라진다. running average는 현재 policy의 전형적 오차 규모에 자동 정규화되어 suppression이 과도하게 지속되거나 조기 비활성화되는 것을 막는다. stop-gradient로 두어 margin 자체가 최적화 대상이 되는 것을 방지. |
| 9 | $\hat{A}_t$의 outcome bias $b$가 실질적으로 하는 일은? | local progress 변화가 0 근처로 모호할 때 tie-breaker. $b=0.15$면 progress 변화가 ±0.15 밖일 때만 outcome 라벨을 뒤집을 수 있다. Long에서 0.25로 올린 것은 progress 신호를 덜 믿겠다는 선언. |
| 10 | 이 방법을 diffusion action head나 autoregressive VLA에 옮길 수 있나? | Corrective matching(Stage II)은 파라미터화 독립적이라 그대로 이식 가능. Adaptive Redirection Objective(Stage III)는 $\hat{x}_0 = x_n - n v_\theta$라는 linear flow-matching 보간에 의존하므로 diffusion은 재유도가 필요하고, discrete autoregressive head는 endpoint 거리 개념 자체가 달라 상당한 재설계가 필요. |
| 11 | iterative offline-online으로 확장하면 어떻게 되나? | 저자가 future work로 남김. 우려: policy가 갱신될수록 고정 버퍼와의 분포 괴리가 커져 회수된 corrective target이 stale해진다. 매 iteration 재수집이 필요하면 sample efficiency 이점이 희석된다. |
| 12 | 실로봇 clothes folding의 +31p가 과대평가는 아닌가? | base 36%로 매우 낮은 시작점이라 헤드룸이 컸다. 100 에피소드 평가이므로 표준오차는 대략 ±5%p 수준 — +31p는 그보다 훨씬 크다. 다만 seed/반복 정보가 없어 실행 간 변동은 알 수 없다. 정성적 회복 사례(왼팔 pull)가 수치 이상의 설득력을 보탠다. |

<!-- VERIFIED: pdf -->
