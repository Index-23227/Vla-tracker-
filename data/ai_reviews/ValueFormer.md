# ValueFormer: A Causal Transformer Value Function with Stage-Aware Labels for Semi-Autonomous VLA Policies

> **한 줄 요약**: Behavior cloning으로 학습된 VLA는 "지금 잘 하고 있는가"를 말할 수 없다는 문제에, **아키텍처가 아니라 per-frame 라벨이 병목**이라고 답하는 논문. 실패 에피소드의 라벨을 flat-zero로 뭉개는 대신 **실패 시점까지는 성공 곡선을 그대로 따르고 그 후에만 γ로 감쇠**하는 stage-aware Monte-Carlo return을 설계하고, 3.5M 파라미터 causal transformer(frozen DINOv3 ViT-L/16 위)가 smooth value V_mc와 sharp detection V_bin을 한 번의 forward로 뱉는다. 실제 로봇 샌드위치 조립 1,427 에피소드에서 val MSE 3e-4, 개입 라벨 추가 시 검출 AP 0.38→0.82. 그리고 이 critic이 만든 per-frame 학습 가중치로 π0.5를 post-training한 on-robot A/B에서 completion 70%→85%, repeat-pick 4→0 (단, n=20에서 Fisher p=0.45로 noise 범위).

---

## 1. 배경 및 동기

### 문제 설정
π0 계열 flow-matching VLA는 이제 실제 양팔 조작을 해내지만, **조용히 실패**한다. Action stream만 봐서는 (a) 상추 한 장에 막혀 정체된 롤아웃, (b) 두 번째 패티를 얹으려는 롤아웃, (c) 정상 진행 중인 롤아웃이 전부 똑같이 생겼다 — flow-matched action chunk, 비디오 스트림, joint command. 정책은 **calibrated된 progress 개념을 갖고 있지 않다**. 로봇 fleet을 감독하는 오퍼레이터에게 이 격차는 "쓸만한 semi-autonomous 시스템"과 "화면마다 사람이 붙어야 하는 시스템"의 차이다.

### 왜 기존 해법이 안 되나
- **RL로 학습하면 critic이 공짜**: 하지만 실제 스테이션에서 on-policy exploration은 느리고, 음식을 낭비하며, 무인 운용이 불안전하다.
- **시뮬레이션으로 우회**: 막혀 있다. 접히는 상추, 붙는 치즈, 넘어지는 패티 — **deformable/granular food는 현재 시뮬레이터가 가장 못 재현하는 것**이 하필 이 태스크의 본질이다.
- **Terminal success/failure 1비트**: 저자들은 여기서 정직하다. 이 비트는 *원리적으로 학습 가능*하고 에피소드가 충분하면 outcome classifier는 좋고 나쁨을 분리한다. 못 하는 건 **"언제"**다. 2분/6단계 에피소드에 1비트를 펼치는 것은 real-robot RL의 reward sparsity 문제가 라벨링 문제로 재등장한 것. Advantage 추정은 인접 두 프레임의 *차이*를 알아야 하는데 terminal에만 라벨이 있으면 정의되지 않고, safety filter는 실수가 시작된 프레임을 알아야 하는데 에피소드 단위 판정은 그 순간이 한참 지난 뒤 도착한다.

### 핵심 주장
Density만으로는 부족하고, **잘못된 모양의 dense 라벨은 라벨이 없는 것보다 나쁘다**. 네트워크가 그 라벨을 충실히 fitting하기 때문. 교과서적 flat-zero 타깃은 dense하지만 **평범하게 좋은 관측이 실패를 예측한다고 적극적으로 가르친다**. 필요한 것은 dense + continuous + 레벨과 기울기가 모두 의미를 갖도록 shaped된 타깃이다.

📌 [Figure 1 삽입] — 라이브 롤아웃에 오버레이된 dual head. 외란 전: V=0.69, 실패확률 0.18, 무알림. 상추에 외란 인가 후: 상추를 건너뛰고 top bun으로 진행 → 실패확률 1.00 saturate하며 VALUE ALERT, 반면 smooth V는 0.61로 거의 안 움직임. **sharp head가 actionable abort 신호를 담당하고 smooth head는 일부러 그걸 안 한다**는 설계 의도의 그림.

---

## 2. 방법론 심층 분석

### 2.1 Stage-Aware, Success-then-Decay 라벨 (논문의 진짜 코어)

γ = 0.99 고정. 15× temporal subsampling(0.5 s stride) 후 프레임 인덱스 k ∈ {0,…,N−1}.

**성공 에피소드**:
$$v_k^{succ} = \gamma^{(N-1-k)}$$
그래서 v_{N−1} = 1이고, ~73초/N=147 에피소드에서 v_0 = γ^{N−1} ≈ 0.23.

**실패 에피소드**: s_fail을 실패 전 완료 단계 수, k_fail = ⌊(s_fail/N_s)·N⌋이라 하면
$$v_k^{fail} = \begin{cases} \gamma^{(N-1-k)} & k \le k_{fail} \\ v_{k_{fail}} \cdot \gamma^{(k-k_{fail})} & k > k_{fail}\end{cases}$$

**왜 이게 전부인가**: 실패 에피소드의 pre-failure 구간 관측은 *정상 롤아웃의 관측 그 자체*다. 거기에 V ≡ 0을 강요하면 네트워크는 "이 관측들이 곧 실패"라고 배우고, 그게 배포 시 clean success로 번져 held-out에서 0 근처로 진동/붕괴하는 예측을 만든다. 실패 단계별 peak가 patty 0.33, cheese 0.41, lettuce 0.64로 남는 **partial credit**은 flat-zero가 표현할 수 없는 정보다.

**왜 exponential이고 linear가 아닌가** (이 논증이 좋다): linear 라벨 v_k = k/(N−1)도 단조 증가하지만, downstream advantage A_t = V_mc(t+H_A) − V_mc(t)가 상수 H_A/(N−1)로 붕괴해 **t에 무관**해진다. 즉 성공 롤아웃의 모든 프레임이 같은 advantage를 갖고, post-training filter가 실제 진전과 관성 주행(coasting)을 구별할 수 없다. Exponential은 A_t를 non-constant로 유지하고 terminal 근처에서 크게 만든다.

📌 [Figure 3 / Figure 4 삽입] — γ ablation(0.95 너무 가파름, 0.999 너무 평평), 성공 곡선, flat-zero 대비 stage-aware 곡선.

### 2.2 두 번째 sharp 타깃 V_bin

MC 라벨은 의도적으로 매끄럽다 — 0.5 s 샘플 사이 변화가 최대 γ=0.99 배. Critic이 원하는 성질이지만 safety filter가 원하는 것의 정반대다. 그래서 **에피소드당 mistake interval 집합** I_e = {(t_start^{(j)}, t_end^{(j)})}로부터
$$v_k^{bin} = 0 \text{ if } \exists j: t_{start}^{(j)} \le t_k < t_{end}^{(j)},\ \text{else } 1$$

두 가지 설계 선택이 실려 있다:
1. **에피소드 flag가 아니라 interval에서 프레임 단위로 정의** → 전체적으로 성공한 에피소드도 일시적 실수(놓친 grasp 후 재시도, 짧은 치즈 긁힘) 구간에서 0-라벨 프레임을 기여한다. Eq. (3)의 MC 타깃은 이걸 표현할 수 없다 (성공 에피소드는 경로가 아무리 지저분해도 v^succ로 라벨링됨).
2. **(t_start, t_end)가 단일 t_fail을 대체**. MC head는 "돌아올 수 없는 지점이 언제인가"만 알면 되지만, binary head는 **어느 프레임이 실수처럼 보이는가**를 알아야 하고 일시적 실수에는 시작과 끝이 있다. Interval에는 optional severity weight ∈ [0,1]이 붙어 V_mc에 곱셈으로 적용(trapezoid dip)되므로 회복된 실수도 smooth 곡선을 눌러준다.

### 2.3 라벨링을 어떻게 확보하나 (3개 소스)

- **수동 단일점 테이블**: 실패 에피소드당 t_fail 하나 + 자유 텍스트 사유. 88개 실패에 에피소드당 ~1분, 두 시간 미만 한 세션. 자유 텍스트("couldn't pick lettuce" → stage 4, "two patties" → stage 1, "scratching cheese" → stage 2)를 작은 키워드 매처가 s_fail로 결정론적으로 변환.
- **수동 multi-interval 테이블**: 에피소드당 최대 3개 segment (에피소드당 2–3분). 총 196개 interval, 여기엔 **회복된 실수를 포함한 성공 에피소드 ~70개**가 들어간다(보통 상추/치즈 단계의 3–8초 재시도).
- **개입 유래 segment (스케일링되는 유일한 소스)**: HIL 텔레오퍼레이션 스테이션은 per-frame intervention flag를 이미 공짜로 만든다. 연속 구간 하나가 곧 segment. 단 두 가지 보정이 필요:
  - flag는 **후행 지표**다. 오퍼레이터는 원인 행동 *이후에* 반응하므로 V_bin 타깃을 t_start보다 **λ = 1.5 s 앞으로 shift**하고, 모호한 run-up 구간 [t_start − 5s, t_start − λ)는 라벨을 추측하지 않고 **per-frame ignore mask로 loss에서 제외**한다.
  - takeover *내부* 프레임은 사람이 만든 것이므로 V_mc regression 타깃을 0.5로 down-weight (그건 정책이 아니라 교정 혼합물을 묘사한다).
  - 개입 라벨은 수동 주석을 **대체하지 않고 증강**한다: 합집합으로 학습하되 개입 프레임을 샘플러 질량의 ~20%로 capping하고, **수동 주석 세트를 유일한 validation/model-selection 신호로 유지**.

> ❓ **예상 질문**: VLM으로 자동 라벨링하면 안 되나?
> **답변**: 시도했고 실패했다. Gemma-4-26B에 에피소드당 12개 probe frame으로 최고 완료 단계를 묻는 방식. 88-failure 세트에서 최선 프롬프트가 failure fraction의 per-episode MAE ≈ 0.21에 그쳤고, 저자들이 유일 truth로 쓸 수 있다고 판단한 ~0.10 기준을 크게 넘었다. 지배적 실패 모드는 **VLM이 실제보다 높은 단계를 hallucinate**하는 것이고, per-episode peak heuristic이 그걸 증폭했다.

### 2.4 아키텍처 (의도적으로 지루하게)

| Block | 정의 | # params |
|---|---|---|
| Frozen encoder | 6 view (3 cam + cam_high ROI + 2 wrist ROI), DINOv3 ViT-L/16 CLS concat → 6144 | frozen |
| Vision proj | Linear(6144→256) | 1,573,120 |
| State proj | Linear(14→256) | 3,840 |
| Time proj | Linear(2→256) | 768 |
| Positional | 16 position × 256 | 4,096 |
| Causal transformer | 2 layer, 4 head, GELU pre-norm, FFN 4d | 1,579,520 |
| Final LayerNorm | 256 | 512 |
| V_mc head | 256→512→256→1 | 264,705 |
| V_bin head | 256→128→1 | 33,025 |
| **Trainable total** | | **3,459,586** |

- **6-view의 이유**: 두 개의 wrist ROI가 접촉 영역을 겨냥한다. **놓친 grasp는 top-down 뷰가 결과를 인지하기 수 초 전에 wrist crop에서 먼저 보인다.**
- **concat이 아니라 element-wise sum**: x_t = W_vis z_t + W_state q_t + W_time φ_t. 초기 실험에서 concat은 vision이 raw dimensionality로 지배하게 만들었다.
- **causal mask가 구조적 핵심**: 프레임 t는 ≤ t만 attend. Online per-frame 예측에서 미래 프레임은 존재하지 않으므로 bidirectional encoder는 애초에 admissible하지 않다.
- **8 s window (16 samples @ 2 Hz)**: 초기 16 s(32 sample)에서 절반으로 줄였다. 긴 윈도우는 half-window 스케일로 value를 평활화하는데, V_mc에는 바람직하지만 **V_bin에는 실수가 시작되는 순간을 적극적으로 가린다**. 저자들이 튜닝한 것 중 가장 영향이 큰 하이퍼파라미터(seq_len ∈ {8,16,32,64} sweep).

📌 [Figure 6 / Figure 7 삽입] — 6개 뷰와 end-to-end 데이터 흐름.

### 2.5 학습 목적함수

$$\mathcal{L}_{mc} = -\frac{1}{B}\sum_i w_i\left[v_i \log \sigma(\hat\ell_{v,i}) + (1-v_i)\log(1-\sigma(\hat\ell_{v,i}))\right]$$
$$\mathcal{L}_{bin} = \text{BCE}_w(\hat\ell_b, v^{bin}; \alpha_+),\quad \mathcal{L} = \mathcal{L}_{mc} + \beta\mathcal{L}_{bin},\ \beta = 1.0$$

- α_+ = clip(n_0/n_1, 0.1, 10): 배치당 ~95%/5% good/mistake 불균형에 대한 pos-weight (병리적 미니배치 방지용 clip).
- **β = 1.0은 의도적 변경**: 이전 single-head recipe에서 (당시 에피소드 단위) auxiliary loss는 β = 0.1의 작은 regularizer였다. 이제 V_bin은 regularizer가 아니라 **primary output**이므로 그에 맞게 가중.
- **왜 MSE가 아니라 BCE-on-value**: sigmoid-MSE 합성 loss는 {0,1} 끝점 근처에서 saturate하는데, 하필 거기가 성공과 실패가 가장 크게 갈리는 영역이다. MSE는 *측정치로만* 보고하고 학습 신호로 쓰지 않는다.

---

## 3. 데이터 전략

| 구성 | 규모 |
|---|---|
| Teleoperated expert 성공 | 1,249 에피소드 |
| Policy rollout | 178 (성공 90 / 실패 88) |
| **총계** | **1,427 에피소드** |
| 프레임 | 에피소드당 ~2,200 @ 30 fps, 15프레임마다 subsample → ~150 sample/ep |
| 총 샘플 | 213,102 (train 181,421 / val 31,681, **에피소드 단위 분할**로 frame leakage 없음) |
| Feature payload | ~6.5 GB (RTX-5090급에서 재빌드 ~90분) |
| HIL 세트 (별도) | 134 롤아웃 / 362k 프레임 / 129 에피소드에 333 takeover segment (105 train / 24 held-out) |

- **4-group balanced sampler** {expert-success, expert-fail, rollout-success, rollout-fail}. 실제로 expert-fail 그룹은 **비어 있다** (전문 텔레오퍼레이터는 유의미한 샘플이 될 만큼 자주 실패하지 않는다) → 3그룹으로 축약되어 ~88개 실패 롤아웃을 ~1,249 expert 성공에 대해 up-weight.
- 결과가 확인 불가능한 에피소드는 학습에서 완전히 제외.
- 오프라인 파이프라인의 비싼 부분은 **모델이 아니라 데이터**: feature 빌드 ~90분 vs 학습 3분 미만.

---

## 4. 시스템/학습 세부사항

- AdamW, lr 1e-4, weight decay 0.05, 5 epoch linear warm-up → cosine annealing to zero, batch 256, 최대 100 epoch, val MSE early stopping patience 15 (best ≈ 80 epoch).
- Dropout 0.2 (body) / 0.4 (binary head로 가는 CLS 경로).
- DINOv3는 torch.hub로 로드하되 HF q/k/v projection을 fused-qkv로 융합하는 변환 루틴 필요.
- 전체 학습 < 3분 (단일 RTX-5090급), 런타임은 transformer가 아니라 데이터 로딩이 지배.
- 하드웨어: 7-DoF Trossen WidowX 양팔 + 6개 재료통. 6단계 순서 bottom_bun → patty → cheese → tomato → lettuce → top_bun. "make a sandwich" 프롬프트로 π0-family VLA가 end-to-end 구동.

---

## 5. 실험 설계 및 평가 프로토콜

**Value 예측**: held-out에서 val MSE / MAE / mean-V-success / mean-V-fail / 분리도 ΔV.

**라벨 shape ablation**: 아키텍처·데이터·recipe를 전부 고정하고 **실패 에피소드 프레임의 라벨만** 바꾼 5개 run (MC-smooth, A: outcome-scaled, B: cliff, C-linear α-mix, C-late). 12 성공 / 9 실패 롤아웃 세트에서 평가. 중요한 방법론적 선택: 이 ablation은 **legacy 4-view single-head recipe**에서 돌렸다 — 라벨 효과를 dual head가 함께 가져오는 아키텍처 변화(추가 뷰, per-frame V_bin, 짧은 윈도우)와 섞지 않기 위해.

**검출**: HIL held-out 24 에피소드, **matched 5% per-frame FPR**에서 비교 + threshold-free 지표(AUROC, AP). Detection "lead"는 첫 sustained alert와 실제 takeover onset 사이 시간 — onset이 raw flag에서 오므로 label-independent.

**On-robot A/B**: 각 변형당 **연속 두 개 샌드위치 × 10 trial = 20 샌드위치**, 6개 subtask를 [0,1]로 채점 → 변형당 120 cell. 연속 두 개인 이유가 설득력 있다: 갓 세팅된 스테이션의 단일 샌드위치는 이미 안정적으로 되므로 변형 간 변별력이 없다. 정책 자신의 행동이 drift를 누적시킨다(첫 패티를 집으며 넘어뜨린 패티, 밀려난 치즈 더미) → **두 번째 샌드위치 비율이 자연스러운 hard-example mining**.

---

## 6. 실험 결과 심층 분석

### 6.1 메인 정량 (Table II)

| 항목 | 값 |
|---|---|
| Validation MSE | 3.0 × 10⁻⁴ |
| Validation MAE | 0.015 |
| Mean-V-success | 0.537 |
| Mean-V-fail | 0.000 |
| Separation ΔV | 0.537 |
| Best checkpoint | ~80 epoch |
| Wall-clock | < 3분 |

초기화 시 ~1.4e-3 → 80 epoch 내 ~3e-4, MAE 0.026 → 0.015. L_bin은 초기 몇 epoch 안에 수렴하고 학습을 지배하지 않는다.

### 6.2 네 가지 canonical 롤아웃 signature (Figure 10)

Held-out에서 post-hoc smoothing 없이 재현:
- **clean success**: MC 라벨을 거의 정확히 따르는 매끄러운 상승
- **success-with-retry**: 재시도 동안 V가 **dip했다가 단계 완료와 함께 회복** — 저자들이 가장 원했던 거동(재시도가 완전 붕괴로 가기 전 조기 경보)
- **early collapse**: bottom-bun/patty를 거쳐 ~0.4에서 peak 후 두 번째 치즈가 놓이며 감쇠
- **stuck-scratching**: 끝까지 낮게 유지

각 실패 모드가 대략 20–30 에피소드로만 뒷받침되므로, 재현된다는 사실 자체가 에피소드 암기가 아닌 일반적 "진전" 개념 학습을 시사한다.

### 6.3 라벨 shape ablation (Table III) — 이 논문에서 가장 정직한 부분

| Scheme | Val loss | Rollout MAE | V̄_succ | V̄_fail | ΔV |
|---|---|---|---|---|---|
| **MC-smooth (ours)** | **0.00051** | **0.024** | 0.533 | 0.390 | 0.143 |
| A: outcome-scaled | 0.00192 | 0.058 | 0.504 | 0.195 | **0.309** |
| B: cliff | 0.00202 | 0.047 | 0.529 | 0.275 | 0.254 |
| C-linear | 0.00190 | 0.058 | 0.515 | 0.266 | 0.249 |
| C-late | 0.00178 | 0.048 | 0.516 | 0.267 | 0.250 |

회귀 문제로 보면 MC-smooth가 val BCE loss 약 3배, rollout MAE 약 2배 우위. **그런데 ΔV로 보면 MC-smooth가 꼴찌(0.143)**이고 저자들은 이를 숨기지 않고 정면으로 논증한다:

A의 ΔV 0.309는 **label artifact**다. pre-failure 곡선 전체에 c = s_fail/N_s를 곱하므로 v_0^fail = c·γ^{N−1}이 이미 v_0^succ와 다르다 — **아직 어떤 실패 단서도 보이지 않은 첫 프레임에서 결과가 새어 나온다**. 배포 시 value head는 진행 중인 롤아웃만 보고 결말을 알 수 없으므로, 최종 결과에 조건화된 타깃은 **관측만으로는 학습 불가능**하다. 네트워크는 frame 0에서 결말을 찍는 것에 보상받는 셈. 게다가 A와 C-late는 k_fail 이후를 hard-zero로 만들어, **재료 4개를 올린 뒤 무너진 에피소드와 bottom bun에서 실패한 에피소드를 구별 불가능하게** 만든다. MC-smooth는 v_{k_fail}에서 감쇠하므로 post-failure tail이 "얼마나 갔는지"를 자동으로 담는다.

B(cliff)는 t=0 leakage는 없지만 k_fail에서 불연속을 강요 → Figure 12에서 cliff 직전 고주파 진동으로 가시화.

**요약**: MC-smooth만이 동시에 (i) v_0^fail = v_0^succ (결과 누출 없음), (ii) k_fail에서 불연속 없음, (iii) hard zero가 아닌 매끄러운 감쇠 credit을 만족한다. 좁은 ΔV는 결함이 아니라 설계 trade-off — downstream binary supervisor는 여전히 V(s)를 thresholding할 수 있고, smooth shape는 pure detector가 못 담는 within-episode dynamics(retry의 dip-and-recover, early collapse의 peak-and-drop)를 보존한다.

### 6.4 개입 유래 supervision (Table IV) — 가장 큰 실증적 이득

24개 held-out HIL 에피소드, matched 5% FPR:

| Metric | Manual only | +Intervention (single) | +Intervention (ensemble) |
|---|---|---|---|
| Average precision | 0.38 | 0.79 | **0.82** |
| AUROC | 0.69 | 0.93 | **0.94** |
| Detected before takeover | 35% | 95% | **95%** |
| Median lead | 1.2 s | 2.4 s | **2.4 s** |

**AP가 0.38 → 0.82로 두 배 이상**, 오퍼레이터가 반응하기 전에 잡히는 비율이 35% → 95%로, 같은 오경보 예산에서. 그리고 이건 운 좋은 하이퍼파라미터가 아니다: label shape / intervention share / context length / detection-head weight를 훑은 **54-run sweep에서 48개 manual+intervention run 전부가 6개 manual-only run 전부를 held-out AP에서 이겼고**, 두 클러스터 사이에 깨끗한 간격이 있다.

⚠️ **저자가 스스로 붙인 caveat**: 이 held-out ground truth 자체가 개입 유래(λ=1.5 s 보정 포함)이므로, **"오퍼레이터가 개입한 지점과의 일치도"를 재는 것이지 독립적인 실패 개념을 재는 게 아니다**. raw onset 대비로 측정되는 lead-time 열만 예외. 완전히 독립적인 hand-labeled onset 평가가 다음 측정으로 남아 있다.

### 6.5 Critic → 정책 학습 가중치 (Table V) 그리고 on-robot A/B (Table VI)

Critic 출력이 만들 수 있는 per-frame 가중치는 네 가지. **모두 human-takeover 프레임에 w=9 floor를 두고, takeover 없는 에피소드는 손대지 않는다. 오직 takeover 에피소드의 autonomous 프레임에서만 다르다.**

| Scheme | Signal | Weight | Status |
|---|---|---|---|
| IWR (control) | flag only | 1 | 평가됨 (대조군) |
| vf-mask | V_bin | 1[V_bin ≥ 0.5] | 평가됨 |
| soft | V_bin | 0.1 + 0.9 V_bin | **구현만, 미학습** |
| awr | A_t | clip(e^{A_t/β_awr}, 0.1, 9) | 평가됨 |

A_t = V_mc(t+50) − V_mc(t), β_awr = 0.3. **scheme (iv)만이 reweighted imitation이 아니라 진짜 RL 업데이트**다: AWR은 max_π E[A(s,a)] s.t. KL(π‖π_BC) ≤ ε의 closed-form 해이므로, flow-matching loss를 e^{A_t/β}로 가중하는 것은 V_mc를 policy-evaluation critic으로 하는 **offline policy iteration 한 스텝**. 이 관점에서 IWR은 degenerate special case.

**On-robot A/B (변형당 20 샌드위치, 동일 985k-frame 데이터, 동일 w=9 floor)**:

| Metric | IWR (control) | vf-mask | vf-awr |
|---|---|---|---|
| Completion | 70% (14/20) | **85% (17/20)** | **85% (17/20)** |
| Wilson 95% | [48, 85] | [64, 95] | [64, 95] |
| Sandwich 2 only | 60% | **90%** | 80% |
| Mean subtask | 0.91 | 0.94 | **0.96** |
| Clean cells | 88% | 88% | **92%** |
| Catastrophic cells | 7 | **2** | **2** |
| Repeat-picks | 4 | **0** | **0** |
| Median duration | 1:15 | 1:21 | 1:22 |

두 변형이 서로 다른 방식으로 도달한다. **vf-mask는 가장 평평한 바닥**을 산다: 0.89 미만 subtask 없음, 대조군이 약했던 tomato 0.82→0.97, lettuce 0.82→0.95, top bun 0.88→0.92를 끌어올리고 cheese 0.93→0.89, bottom bun 1.00→0.94를 조금 내준다. **더 어려운 두 번째 샌드위치에서 개선하는 유일한 변형(60%→90%)**. **vf-awr은 가장 높은 천장**: tomato/lettuce 완벽 1.00/1.00, 최고 mean subtask 0.96, 92% clean cell, 실패해도 부드럽게 실패(실패 샌드위치에서도 평균 0.86 vs vf-mask 0.78). 대신 cheese 0.84이고 sandwich-1에서 더 높게 출발하는 탓에 두 번째 샌드위치는 90%→80%로 하락.

**Repeat-pick 제거가 가장 선명한 신호**: 이미 성공한 pick을 recovery reflex가 다시 실행하는 이 거동은 takeover 데이터에서 물려받은 것으로, 모든 flag-only 레버(교정 데이터 추가, replay share 2배, pause frame 제거)를 견디고 대조군에서 4회 발생했는데, **두 critic 가중치 어느 쪽에서도 20 샌드위치 동안 한 번도 안 나왔다**. 저자들이 제시하는 메커니즘: 불필요한 re-pick으로 가는 run-up이 정확히 critic이 flag하는 low-value drift이므로 vf-mask는 0으로, vf-awr은 down-weight로 그 프레임을 gradient에서 빼버리고, 반면 recovery *동작*을 가르치는 takeover 프레임은 w=9를 온전히 유지한다.

**사전 등록된 예측이 반쯤 틀렸다는 것도 그대로 보고한다**: vf-mask가 가장 유망(고정 마스킹 윈도우의 timing bluntness를 고침), soft가 견고한 2순위, **vf-awr이 가장 안 될 것**(가중치가 대부분 프레임에서 1에서 거의 안 벗어나고 horizon-differenced advantage는 critic의 가장 noisy한 출력)이라고 예측했는데, vf-awr이 vf-mask의 completion을 맞추고 최고 subtask 품질까지 냈다. "예측이 vf-awr에게 유리하게 틀렸다"고 조용히 고치지 않고 명시한다.

### 6.6 서빙 비용 (Table VII/VIII)

Critic이 정책과 같은 가속기(RTX 5090 32GB)를 쓰면서 실제로 정책을 방해한다.

| Component | Isolated | Concurrent |
|---|---|---|
| π0.5 inference | 58.6 ms | 84.3 ms (p90 100.6, max 102.8) |
| Value tick (FP32) | 266 ms | ~400 ms (2 Hz, isol. duty 53%) |

30 Hz 루프 + 3-step prefetch = **~100 ms 예산**인데 concurrent p90이 정확히 거기에 닿는다 → prefetch 미완료 → 클라이언트 블로킹 → 제어 루프 stall(눈에 보이는 모션 hitch). 간섭은 대칭적이지만(정책 부하 하에서 critic tick도 300–560 ms로 팽창해 500 ms 주기를 넘김) hard deadline이 있는 건 정책 쪽뿐.

**수정은 재학습이 필요 없다.** 비용은 518 px DINOv3 ViT-L/16 6회 forward가 지배(transformer head는 sub-millisecond):

| Encode variant | Mean | Speedup | Duty @2Hz |
|---|---|---|---|
| Original: batch-1, FP32, host copies | 349.8 ms | 1.0× | 70% |
| Batched×6, FP32, on-GPU | 172.0 ms | 2.0× | 34% |
| Batched×6, TF32 | 129.9 ms | 2.7× | 26% |
| Batched×6, bf16 autocast | 85.1 ms | **4.1×** | **17%** |

실제 서빙 스텝에서 tick 266 ms → ~88 ms (batched+TF32) → ~52 ms (+bf16), **3–5× 감소**로 p90을 100 ms 데드라인 아래로 되돌린다. 수치 검증도 붙였다: batched FP32는 per-view와 수치적으로 동일(max |Δ| = 3.5e-5), bf16은 value 출력을 <1e-3만 이동(V_bin은 소수 4자리까지 불변). bf16 재검증 — 24 에피소드 held-out을 bf16으로 재인코딩해 matched 5% FPR에서: cosine 유사도 평균 0.99994/최소 0.99969, AUROC 0.943→0.942, AP 0.818→0.816, median lead 2.4 s 불변. 더 민감한 트리거에서만 onset 에피소드 20개 중 1개를 놓친다(95%→90%, +0.2 false alarm/min).

---

## 7. Ablation 분석

| Ablation | 결론 |
|---|---|
| **라벨 shape (5종)** | val BCE loss spread ≈ **4×**. 저자들이 시도한 **어떤 아키텍처 변경(layer 수, head 수, hidden size, sampler weighting)도 이에 비견되는 변화를 못 만들었다** |
| **γ (0.95 / 0.99 / 0.999)** | 0.95는 너무 가팔라 마지막 몇 초에만 상승, 0.999는 너무 평평해 이미 1 근처에서 시작, 0.99 채택 |
| **Context window (8/16/32/64 sample)** | 튜닝한 것 중 가장 영향 큼. 16 sample(8 s) 채택 — 긴 윈도우는 V_mc엔 좋지만 V_bin의 전이 순간을 가림 |
| **개입 라벨 (54-run sweep)** | 48개 manual+intervention이 6개 manual-only를 **전부** 이김, 깨끗한 간격 |
| **Auxiliary head** | val MSE 변화는 작지만 실패 클래스 예측이 일관되게 0 근처로 더 조이는 효과. 없으면 실패 클래스 평균에 작은 양의 residual (calibrated 수치는 보고 안 함) |
| **Single-head vs dual-head** | single-head는 γ와 라벨 shape를 어떻게 잡아도 **항상 너무 sharp(retry dip 안 보임) 아니면 너무 smooth(actionable abort 순간 없음)**. backbone 공유하며 타깃만 분리하면 한 forward로 양쪽 확보, single-head 대비 파라미터 몇 % 증가, wall-clock 증가 측정 불가 |

---

## 8. 관련 연구 비교

- **π0 / π0.5 [Black et al.]**: 이 논문이 감독하는 정책. flow matching action head, open-world 일반화. 둘 다 critic 없음.
- **π*0.6 / RECAP**: 정신적으로 가장 가까운 선행 연구. steps-to-success를 예측하는 distributional value function을 축적된 롤아웃에서 학습하고, advantage를 이산 good/bad 지시자로 binarize해 그 조건 하에 VLA를 재학습. **value와 policy를 교대 라운드로 학습**(value fit → advantage 계산 → policy fine-tune → 새 롤아웃 수집 → 반복). ValueFormer는 **policy-agnostic 대안**: 같은 value head가 정책의 입력 형식이나 학습 루프를 건드리지 않고 π0 계열 어떤 VLA든 감독할 수 있다.
- **QPILOTS / Guided-Action-Flow 계열과의 차이**: 그쪽은 critic gradient를 **inference-time에** denoising trajectory에 주입한다. ValueFormer는 추론 경로에 전혀 들어가지 않고(**advisory & fail-safe** — tick을 놓치거나 늦어도 오퍼레이터 알림만 지연될 뿐 제어 경로에 진입하지 않음), **오프라인 post-training loss의 per-frame 가중치**로만 정책에 영향을 준다. 서로 직교하는 사용처.
- **Failure detection 계열**과의 차이: 대부분 에피소드 단위 판정이거나 단일 failure point 감독. ValueFormer는 segment 기반 interval로 **회복된 실수까지** 신호로 쓴다.

---

## 9. 한계 및 미해결 문제

1. **Causal window lag**: 실패가 MC 라벨 전이보다 수 초 늦게 V_mc에 등록된다. 16 s 윈도우에서 육안 추정 5–10 s, 8 s 윈도우에서 대략 절반. **정밀 측정은 하지 않았다**(저자 명시). 최근 프레임을 강조하는 asymmetric attention kernel이나 mixture-of-windows가 후보.
2. **수동 실패 타이밍이 스케일하지 않음**: 88개는 감당됐지만 fleet-scale post-training loop는 주당 10³–10⁴ 실패를 만든다. §IV-C의 개입 flag가 첫걸음이지만, **takeover를 유발하지 않은 잔여 실패**는 여전히 라벨링 경로가 없다.
3. **Closed-loop 통합: 한 경로만 평가, 그마저 noise 범위**: completion 70%→85%는 n=20에서 Fisher p=0.45. Catastrophic cell 감소는 대체로 completion 카운트를 되풀이하고, *완료된* 샌드위치만 보면 subtask 평균이 수렴한다(vf-mask 0.97, vf-awr 0.98, 대조군 0.99). 두 변형이 대조군보다 1–2주 뒤에 평가되어 **재료 staging(끈적한 치즈 등)이 세션 간 통제되지 않았고**, note 유래 카운트는 heuristic. 공유 초기조건 + McNemar test의 paired 재평가가 power fix. 미평가 경로: (i) V(s) 하락 기반 abort/retry 트리거, (ii) advantage **conditioning**(A_t를 loss weight가 아니라 conditioning token으로), (iii) low-value 롤아웃을 post-training set에서 걷어내는 offline filter. `soft` 변형도 미학습.
4. **실패 taxonomy가 coarse**: 롤아웃 로그에서 경험적으로 뽑은 모드들. 하드웨어 결함은 grasping 실패와 같은 의미의 "partial credit"이 아니고 별도 decay schedule을 받아야 마땅하다.
5. **단일 태스크/단일 embodiment**: 하나의 샌드위치 스테이션, 하나의 양팔 구성. 시뮬레이션 벤치마크(LIBERO/CALVIN 등) 수치가 전혀 없어 다른 논문과의 직접 비교 불가.

---

## 10. 총평

### 강점
- **문제 진단이 정확하고, 그 진단을 실험으로 지지한다**. "라벨이 아키텍처를 지배한다"는 주장을 5-shape 통제 ablation(val loss 4× spread)으로 뒷받침하고, 아키텍처 튜닝은 비견되는 변화를 못 만들었다고 명시.
- **자기 지표가 불리한 곳에서 정직하다**. ΔV에서 자기 방법이 꼴찌라는 걸 표에 그대로 싣고, 왜 그 지표가 오도하는지를 label leakage 논증으로 설명한다. 사전 등록한 예측이 틀린 것도 고치지 않고 보고한다.
- **개입 flag 재활용이 실무적으로 가장 큰 아이디어**. HIL 스테이션이 이미 공짜로 만들고 있는 신호를, lag 보정(λ=1.5 s)과 모호 구간 ignore mask라는 두 개의 올바른 디테일과 함께 쓴다. AP 0.38→0.82는 이 논문에서 가장 큰 폭의 개선이고 54-run sweep로 robust함까지 보였다.
- **배포 현실을 다룬다**. 대부분의 논문이 무시하는 "감시 모델이 감시 대상의 deadline을 잡아먹는다"를 프로파일하고, 재학습 없는 3–5× 최적화와 그 수치적 안전성 검증까지 붙였다.

### 약점
- **핵심 정책 개선 주장이 통계적으로 약하다**. n=20, p=0.45. 저자들도 "suggestive, not established"라고 쓴다. 게다가 대조군과 변형이 1–2주 간격으로 평가되어 세션 간 교란이 통제되지 않았다.
- **평가가 완전히 in-house**. 공개 벤치마크 0개, 코드 미공개. 재현 경로가 사실상 없다.
- **Value head가 정책을 만들지 않는다**. 이 논문은 VLA가 아니라 VLA 주변 인프라다 — 트래커에 넣는 근거는 어디까지나 critic-derived weight가 π0.5의 성공률을 움직였다는 §VI-G.

### Bottom line
"작은 dual-output value head, 두 타깃을 따로 엔지니어링하되(critic엔 smooth, safety filter엔 sharp) 하나의 backbone과 하나의 forward pass를 공유한다"는 설계 패턴이 진짜 기여물이다. 정책 개선 수치는 아직 약하지만, **라벨 shape이 아키텍처를 지배한다**는 통제된 증거와 **HIL 개입 flag를 dense supervision으로 재활용**하는 레시피는 behavior cloning으로 학습된 어떤 실기 VLA 스택에도 그대로 이식 가능하다.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | Value model일 뿐 정책을 만들지 않는데, 이게 VLA 논문인가? | 엄밀히는 아니다. 하지만 §VI-G에서 critic 출력이 π0.5의 post-training loss 가중치가 되어 on-robot completion 70%→85%, repeat-pick 4→0을 만든다. 정책 개선 경로가 실기로 닫혀 있다는 점에서 순수 monitoring 논문과 구분된다. 다만 그 gain 자체는 n=20, p=0.45로 미확립. |
| 2 | Completion 70%→85%가 p=0.45면 사실상 아무것도 못 보인 것 아닌가? | Completion만 보면 그렇다. 저자들이 내세우는 근거는 두 개의 준독립 채널이다: (a) repeat-pick 4→0 — 모든 flag-only 레버를 견딘 거동이 두 critic 가중치 모두에서 0회, (b) 하드 마스크와 소프트 advantage라는 **서로 다른** 두 가중치가 같은 85%에 도달. 그래도 paired McNemar 재평가 전까지 "확립"은 아니다. |
| 3 | ΔV에서 자기 방법이 꼴찌인데 그걸 "feature"라고 하는 건 사후 합리화 아닌가? | 사후 합리화가 아니라고 볼 근거는 논증이 라벨 정의만으로 성립한다는 점이다. A는 v_0^fail = c·γ^{N−1} ≠ v_0^succ라 t=0에서 결말이 새고, 이건 관측만으로 학습 불가능한 타깃이다. 또 A/C-late는 k_fail 이후를 hard-zero로 만들어 "재료 4개 후 붕괴"와 "bottom bun 실패"를 구별 불가능하게 만든다. ΔV가 큰 것이 곧 유용한 분리가 아니라는 주장. |
| 4 | 검출 성능 AP 0.82가 개입 flag로 학습하고 개입 flag로 평가한 결과 아닌가? | 맞고, **저자가 먼저 인정한다**. held-out ground truth 자체가 λ=1.5 s 보정된 개입 유래이므로 "오퍼레이터가 개입한 지점과의 일치도"다. 예외는 raw onset 기준으로 재는 lead-time(1.2 s → 2.4 s). 독립적 hand-labeled onset 평가는 명시적으로 future work. |
| 5 | 6-view DINOv3가 tick당 266 ms인데 2 Hz critic이 실용적인가? | 최적화 전에는 실용적이지 않았다 — concurrent에서 π0.5 p90이 100.6 ms로 100 ms prefetch 예산에 닿아 제어 루프가 stall했다. 6뷰를 batch 1회로 묶고 host copy를 없애면 2.0×, TF32 2.7×, bf16 4.1×. 실서빙 tick 266→~52 ms. 정책이 이미 DINOv3 frontend를 돌린다면 원리적으론 encoder 공유도 가능하지만 현 스택은 서버가 분리되어 있다. |
| 6 | γ=0.99, 8 s window, λ=1.5 s 같은 상수가 태스크 특화 아닌가? | 그렇다. γ는 ~73 s / 6단계 에피소드 길이에 맞춰졌고(v_0 ≈ 0.23), λ=1.5 s는 사람 반응 지연, 8 s는 detection sharpness 트레이드오프. 저자들도 "매우 짧은 horizon 태스크를 노린다면 비용 균형이 뒤집힌다"고 §VII(5)에서 인정. 이식할 때 반드시 재튜닝해야 할 값들. |
| 7 | vf-mask와 vf-awr이 같은 85%인 게 정말 "critic이 레버"라는 증거인가, 아니면 둘 다 20개 샌드위치의 같은 3개를 살린 건가? | 후자의 가능성을 저자도 적는다 — catastrophic cell 감소가 대체로 completion 카운트를 되풀이하고, 완료된 샌드위치만 보면 subtask 평균이 수렴한다(0.97/0.98/0.99). 더 강한 증거는 completion이 아니라 **repeat-pick 0회**와 두 변형의 서로 다른 프로필(vf-mask는 바닥을 평평하게: sandwich-2 60→90%, vf-awr은 천장을 높게: mean 0.96, sandwich-2 90→80%). |
| 8 | Flat-zero 라벨이 그렇게 나쁘다면 왜 지금까지 쓰였나? | Dense하고 구현이 자명하기 때문. 문제는 dense가 곧 유용은 아니라는 것 — 네트워크가 잘못된 shape도 똑같이 충실히 fitting한다. 실패 에피소드의 pre-failure 프레임은 성공 롤아웃 프레임과 광학적으로 동일한데 V≡0을 강요하면 "이 관측 = 실패"를 학습하고, 배포 시 clean success에서 0 근처 진동으로 번진다. 저자들의 표현: sparse supervision은 너무 적게 말해서 실패하고, 잘못 shaped된 dense supervision은 틀린 것을 확신을 갖고 말해서 실패한다. |
| 9 | Advantage weighting(vf-awr)이 진짜 RL인가, 그냥 reweighted imitation인가? | AWR은 max_π E[A(s,a)] s.t. KL(π‖π_BC) ≤ ε의 closed-form 해이므로, flow-matching loss를 e^{A_t/β_awr}로 가중하면 V_mc를 policy-evaluation critic으로 하는 offline policy iteration 한 스텝이 된다. 이 관점에서 IWR은 사람 프레임에만 상수 log-w를 주는 degenerate case. 다만 A_t는 lookahead 50프레임 차분이라 critic의 가장 noisy한 출력이고, 실제 가중치 중앙값은 1.03(범위 0.13–7.61)으로 1에서 거의 안 벗어난다. |
| 10 | Expert-fail 그룹이 비어 있다는 게 문제 아닌가? | 4-group balanced sampler가 실질적으로 3그룹으로 축약된다. 전문 텔레오퍼레이터가 유의미한 샘플이 될 만큼 실패하지 않기 때문인데, 이는 곧 **실패 분포가 정책 롤아웃 88개에서만 온다**는 뜻이다. 실패 모드당 20–30 에피소드 수준이므로 taxonomy가 coarse할 수밖에 없고, 저자도 §VIII(d)에서 하드웨어 결함처럼 성격이 다른 실패는 별도 decay schedule이 필요하다고 적는다. |
| 11 | Linear 라벨이면 안 되는 이유가 정말 advantage 붕괴 하나뿐인가? | 그게 핵심이다. A_t = V(t+H_A) − V(t)가 linear 라벨 하에서 상수 H_A/(N−1)로 붕괴하므로 성공 롤아웃의 모든 프레임이 동일 advantage를 갖고, post-training filter가 실제 진전과 coasting을 구별할 수 없다. Exponential은 A_t를 non-constant로 유지하고 terminal 근처에서 크게 만든다 — 마지막 거리를 좁히는 것이 초반 동작보다 정보량이 크다는 직관과 일치. |
| 12 | Dual head가 아니라 head 하나로 γ만 잘 고르면 안 되나? | 저자들이 single-head로 시도했고 실패했다: γ와 라벨 shape를 어떻게 잡아도 head는 항상 너무 sharp(retry dip이 안 보임) 아니면 너무 smooth(actionable abort 순간이 없음)였다. 두 타깃은 **설계상 반대 방향으로 당긴다** — MC 라벨은 0.5 s 샘플 간 최대 γ=0.99배로만 변하는데, safety filter가 원하는 건 실수 시작 순간의 step이다. Backbone은 공유하되 타깃을 분리하는 것이 파라미터 몇 % 비용으로 두 regime을 모두 복구하는 방법. |

<!-- VERIFIED: pdf -->
