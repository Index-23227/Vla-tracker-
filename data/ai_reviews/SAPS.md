# SAPS: Shared Autonomy for Policy Steering by Blending Teleoperation with a Pretrained VLA

> **한 줄 요약**: Pretrained VLA(주로 π0.5)의 액션 청크와 실시간 인간 텔레오퍼레이션 명령을 **action level에서 선형 블렌딩**하여 OOD 실패를 복구하는 모델-불가지론적 shared-autonomy 프레임워크. 리트레이닝/auxiliary model/sampler 수정 없이, 세 가지 arbitration 정책(Takeover, Equal Blending, dynamic Cosine-similarity)으로 LIBERO-PRO 15.0%→97.4% (Cosine), CALVIN ABC→D avg-len 3.83→4.30, 실세계 Franka 26.7%→98.3%를 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **VLA의 brittleness**: OpenVLA, π0, π0.5, RT-2 같은 generalist VLA가 학습 분포 안에서는 강력하지만 객체 배치/외형/언어 perturbation에서 급격히 실패 (LIBERO-PRO에서 π0.5가 15.0%까지 떨어짐)
- **기존 policy steering의 부담**:
  - **ITPS**(Wang+ 2025): diffusion sampler를 수정해 spatial target으로 유도 → diffusion 정책 외 적용 어려움
  - **DynaGuide**(Du & Song 2025): auxiliary dynamics 모델 학습 필요, 또한 highperforming π0.5에서는 오히려 -5.7%p 손해
  - **VLS/FOREWARN/π0.7**: VLM 추론 모듈, latent alignment, verbal coaching 등 모두 architecture-specific
- **Pure teleoperation의 부담**: 100% 인간 input + 깊이 인지 부재 + 정밀 제어 → cognitive load 매우 높음 (LIBERO-PRO에서 task당 46초 소요)

### 핵심 질문
- **정책 retrain 없이, sampler 수정 없이, auxiliary model 없이도 OOD 복구가 가능한가?**
- **인간이 100% 개입하지 않으면서도 teleoperation 수준의 성공률을 낼 수 있는가?**

📌 [Figure 1 삽입] — VLA가 OOD에서 deviated path로 빠지면 사용자가 짧은 corrective input → SAPS가 frozen π0.5와 블렌딩하여 in-distribution으로 복귀

---

## 2. 방법론 심층 분석

### 2.1 전체 구조

매 control step에서:
1. 사전학습 VLA π_θ가 RGB(외부+wrist) + 7-DoF proprio + language prompt → action chunk 예측, 첫 N action 실행
2. 동시에 인간이 키보드/게임패드로 a_expert ∈ ℝ⁷ 출력 (가능)
3. Arbitration α ∈ [0,1] 계산 → blended action 실행:

$$
a_{\text{blended}}^{(1:6)} = \alpha \cdot a_{\text{VLA}}^{(1:6)} + (1-\alpha) \cdot a_{\text{expert}}^{(1:6)}, \quad a_{\text{final}}^{(7)} = \max(a_{\text{VLA}}^{(7)}, a_{\text{expert}}^{(7)})
$$

Gripper(7번째 차원)는 둘 중 더 큰 값(close bias) — opposing gripper command 충돌 방지.

> ❓ **예상 질문**: 왜 gripper는 별도 처리?
> **답변**: gripper command는 binary-ish 의도 신호라 선형 블렌딩이 의미 없음. opposing intent 시 그립을 우선시(안전한 default)하는 게 실험적으로 잘 동작.

### 2.2 세 가지 Arbitration Policy

#### (a) Takeover (hard switch)
$$
\alpha = \begin{cases} 0.0 & \|a_{\text{expert}}^{(1:6)}\| > \epsilon \\ 1.0 & \text{otherwise} \end{cases}, \quad \epsilon = 10^{-3}
$$
사용자 input이 감지되면 즉시 100% 사람 통제, 그렇지 않으면 100% 정책.

#### (b) Equal Blending (continuous 50/50)
$$
\alpha = \begin{cases} 0.5 & \|a_{\text{expert}}^{(1:6)}\| > \epsilon \\ 1.0 & \text{otherwise} \end{cases}
$$
intervention 중에는 항상 50/50 — 정책의 manipulation skill을 부분적으로 살려둠.

#### (c) Cosine Similarity (동적 confidence-based)
$$
c = \cos\theta = \frac{a_{\text{human}}^{(1:6)} \cdot a_{\text{VLA}}^{(1:6)}}{\|a_{\text{human}}\| \, \|a_{\text{VLA}}\|}, \quad \alpha = \sigma(k \cos\theta), \quad k=6
$$
- cos=+1(완전 일치) → α≈1 → 정책 주도
- cos=-1(완전 반대) → α≈0 → 사람 주도
- cos=0(직교) → α=0.5
- logistic sigmoid의 sharpness k=6 → disagreement 근처에서 민감, 명확한 동의/반대에서는 saturate

> ❓ **예상 질문**: cosine을 confidence proxy로 쓰는 근거?
> **답변**: 정책의 절대 confidence(log-prob 등)는 backbone-specific하고 calibration 문제가 큼. Cosine은 모델 불가지론적이고, "사람과 정책이 같은 방향으로 가면 정책이 옳다는 신호"라는 직관적 가정. 안전성: 정책이 confidently wrong일 때는 사람이 반대 방향으로 밀면 α가 즉시 떨어짐.

### 2.3 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| **Action-level blending** | 모든 VLA의 출력 인터페이스 = 동일한 7-DoF EE command → 모델 불가지론 |
| **Post-inference** | sampler/decoder를 건드리지 않음 → diffusion, flow-matching, AR 모두 호환 |
| **No retraining** | frozen pretrained checkpoint 그대로 사용 → 즉시 deploy |
| **Sparse human input** | ε=1e-3 threshold로 idle 감지 → 정책이 자율 진행 가능 |

---

## 3. 데이터 및 정책 백본

학습 데이터 없음 (SAPS 자체는 training-free). 사용한 frozen 백본:

| 백본 | 학습 데이터 | 평가 |
|------|------------|------|
| π0.5 | LIBERO checkpoint (Black+ 2025) | LIBERO 표준, LIBERO-PRO |
| π0.5 | DROID | 실세계 Franka |
| π0.5 | CALVIN ABC (RLinf 릴리스) | CALVIN ABC→D long-horizon |
| Diffusion Policy | DynaGuide 릴리스 | CALVIN single-subtask 11개 |

> ❓ **예상 질문**: π0.5는 어떤 action head?
> **답변**: π0.5는 vision-language backbone + **flow-matching action expert**. SAPS는 head 종류와 무관하게 출력 action에만 작용하므로 flow matching 여부는 무관.

---

## 4. 시스템/실험 세부

| 항목 | 값 |
|------|-----|
| Control rate | 20 Hz (LIBERO/CALVIN simulation) |
| Action dim | 7-DoF (3 translation + 3 rotation + 1 gripper) |
| Teleop interface | Keyboard (Blending, Takeover) / Gamepad (Cosine) |
| Idle threshold ε | 1e-3 (L2 norm of expert action) |
| Cosine sharpness k | 6 |
| Hardware | Franka Panda (operational-space pose controller) |
| Observation | 외부 + wrist RGB + EE pose + gripper state |

---

## 5. 실험 설계 및 평가 프로토콜

네 가지 평가 환경:

1. **Standard LIBERO** — Object suite의 단일 task(크림치즈 → 바스켓)를 **8개 perturbation distance**로 변형 (Appendix A4)
2. **LIBERO-PRO** — 10개 선별 task × 2 perturbation type (task / swap), n=20
3. **CALVIN ABC→D** — n=30 chains × 5 subtasks (long horizon)
4. **Real Franka** — n=20 × 3 tabletop task (Pick&Place, Close Drawer, Open Cabinet)

추가 CALVIN subtask 평가: DP 백본 + 11개 single subtask (block-lift 3 + articulated 8).

---

## 6. 실험 결과 심층 분석

### 6.1 Standard LIBERO 단일 task 결과 (Figure 3)

| Method | distance>=0.15 success |
|--------|---------------------|
| π0.5 alone | 22.9% |
| Blending | 87.9% |
| Cosine | 90.0% |

- π0.5는 distance에 강하게 negative 상관(Pearson r=-0.800, p=0.005)
- 두 SAPS 방법 모두 정책 대비 유의(p=0.004, p=0.003)

### 6.2 LIBERO-PRO (Table/Figure 4)

| Method | Avg Success | Human Intervention | Completion Time |
|--------|------------|-------------------|----------------|
| π0.5 | **15.0%** | 0% | 30.7s |
| Teleoperation | 98.8% | 100% | 46.0s |
| Blending | 92.6% | **10.8%** | **11.1s** |
| Takeover | 93.2% | 11.7% | 13.2s |
| Cosine | **97.4%** | 30.0% | 13.0s |

- **Cosine이 가장 높은 success rate** — Teleop에 +/-1.4%p 수준이면서 인간 input은 1/3
- 모든 SAPS 방법 vs π0.5: Wilcoxon p=9.77×10⁻⁴
- Completion time이 π0.5(autonomous)보다도 짧음 → 정책의 inefficient recovery 회피

### 6.3 CALVIN ABC→D (Table 1)

| Method | ST-SR | Mean/5 | Human % | ST-5 |
|--------|-------|--------|---------|------|
| π0.5 | 91.27% | 3.833 | 0% | 63.33% |
| Teleop | 47.37% | 0.900 | 100% | 0% |
| Teleop (3x timeout) | 86.78% | 3.500 | 100% | 46.67% |
| ITPS | 89.43% | 3.667 | 6.67% | 56.67% |
| DynaGuide | 85.59% | 3.167 | 0% | 46.67% |
| **Cosine** | **94.85%** | **4.300** | 13.73% | **76.67%** |

- **Cosine이 모든 metric에서 1위**, baseline π0.5 보다 5-task에서 +13.34%p
- ITPS/DynaGuide는 π0.5에 대해 오히려 -1.84%p, -5.7%p 손해 → high-performing policy에서 기존 steering 방법이 역효과
- Human input은 13.73%만 — pure teleop(100%) 대비 1/7

> ❓ **예상 질문**: 왜 DynaGuide가 π0.5에서는 손해?
> **답변**: 저자 설명대로, DynaGuide의 dynamics-guided denoising은 unguided success<50%일 때 효과가 큼. π0.5처럼 91% 수준에서는 steering signal이 올바른 trajectory를 perturb할 확률이 커짐(head-room 부재). Cosine은 cos~+1일 때 α→1로 정책을 그대로 두므로 이런 perturbation을 피함.

### 6.4 CALVIN single subtask + DP backbone (Figure 6)

| Method | Avg over 11 subtasks |
|--------|---------------------|
| DP alone | 45% |
| ITPS | 66% |
| DynaGuide | 80% |
| **Cosine** | **93%** |

- 8개 articulated task에서 모두 100%, harder block-lift에서도 70-90%
- DP 같은 lower-capacity policy에서도 SAPS 효과 유지 → 백본 무관성 입증

### 6.5 실세계 Franka (Table 2)

| Task | π0.5 | Teleop | Blending | Cosine |
|------|------|--------|---------|--------|
| Pick&Place | 50% | 100% | 100% | 100% |
| Close Drawer | 25% | 100% | 80% | 100% |
| Open Cabinet | 5% | 100% | 100% | 95% |
| **Avg** | **26.7%** | 100% | 93.3% | **98.3%** |

- π0.5가 Open Cabinet에서 5%(거의 실패)인데 Cosine으로 95%로 복구
- Cosine과 Blending이 Marker Plate에서 Teleop보다 유의하게 빠름(p=0.0002 / 0.0005)
- Close Drawer / Open Cabinet은 Teleop과 통계적 차이 없음 — real-world에서 사람은 깊이 인지가 있어 빠를 수 있음

---

## 7. Ablation 분석

논문은 별도 ablation table이 적지만 실험 자체가 ablation 역할:

### 7.1 Arbitration policy 비교 (LIBERO-PRO)

| Policy | Success | Intervention |
|--------|---------|-------------|
| Takeover (hard) | 93.2% | 11.7% |
| Equal Blending | 92.6% | 10.8% |
| **Cosine (dynamic)** | **97.4%** | 30.0% |

- Cosine은 intervention이 더 많지만 success가 +4.2%p
- Trade-off: intervention 효율 vs 절대 성공률 → 어플리케이션 dependent

### 7.2 Sigmoid sharpness k의 효과
- k=6 (논문 default): cos≈0 근처에서만 빠른 전이, |cos|>0.5에서 saturate
- (논문에 sweep 없음 — 한계)

### 7.3 Idle threshold ε
- ε=1e-3 (L2 norm) — 게임패드 noise vs 의도적 input 구분
- (논문에 sensitivity analysis 없음)

---

## 8. 관련 연구 비교

| 방법 | Modification | Auxiliary model | Real-time Human | LIBERO-PRO 결과 |
|------|-------------|----------------|-----------------|----------------|
| ITPS | diffusion sampler | spatial target input | optional | N/A (-1.84% on CALVIN π0.5) |
| DynaGuide | denoising step | dynamics model | × | N/A (-5.7% on CALVIN π0.5) |
| VLS | reward synthesis | VLM | × | — |
| FOREWARN | latent alignment | VLM + dynamics | × | — |
| π0.7 | inference-time coaching | — | verbal | — |
| **SAPS-Cosine** | **none (action-level)** | **none** | **✓ (sparse)** | **97.4% on LIBERO-PRO** |

### 핵심 차이
- **Post-inference action-level** → 모든 backbone 호환
- **Frozen policy** → 추가 학습 cost 0
- **Real-time human as steering signal** → VLM/reward 추론 latency 없음

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Standard LIBERO 4-suite 미보고**: Object suite 단일 task만 perturbation으로 평가. Spatial/Goal/Long suite의 표준 numbers는 없음 → LIBERO leaderboard에 직접 등재 어려움
2. **Cosine sharpness k의 sweep 없음**: k=6은 휴리스틱, sensitivity analysis 부재
3. **Gripper handling의 안전성**: max() rule은 "close bias"인데, "open이 필요한데 정책이 잘못 close 명령"인 시나리오 분석 없음
4. **Single backbone family**: π0.5 + DP만 평가. OpenVLA, RT-2 같은 AR token-based VLA에서의 transferability 미검증
5. **LIBERO-PRO 10/40 tasks subset**: 비용 문제로 task/swap perturbation 두 종류만, 그것도 10개 task만. coverage 부분적
6. **Operator skill의 영향**: n=? operator로 평가했는지 명시 약함 (Appendix A3 참고). Skill variance 분석 부족

### Attribution 문제
- LIBERO-PRO에서 정책 단독 15.0%는 매우 낮은 baseline. SAPS의 개선이 **arbitration 설계 덕분**인지 **사람이 보고 task 의도를 명시적으로 전달하기 때문**인지 분리 어려움
- "100% 사람 통제(teleop) 98.8% vs Cosine 97.4%" — Cosine은 단순히 teleop을 30% 시간 동안 한 것과 어떻게 다른가? completion time 빠른 게 정책 기여의 증거이긴 하나, 더 cleaner attribution 필요

### Real-world generalization
- 3개 task만, 1개 operator 추정
- 깊이 인지 가능한 real-world에서는 teleop과 시간 차이가 없거나 적음 → SAPS의 simulation 우위가 real에서 부분적으로 무력화

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Shared autonomy 자체는 오래된 패러다임. 기여는 "action-level + cosine similarity arbitration"의 단순화. |
| **Technical depth** | ★★★☆☆ — 수식은 매우 간단(2-3개 식). 깊이보다는 깔끔함과 실용성. |
| **Experimental rigor** | ★★★★☆ — 4개 평가 환경, 다양한 baseline(ITPS, DynaGuide), statistical test 일관. Standard LIBERO 미보고는 감점. |
| **Practical impact** | ★★★★★ — Plug-and-play, retrain-free, model-agnostic — 즉시 deploy 가능. assistive teleoperation / 데이터 수집 파이프라인에 큰 활용. |
| **Writing quality** | ★★★★☆ — 명확, 동기-방법-실험-한계 구조 견고. |

**강점**:
- 거의 무료(zero-cost) 개선 — 어떤 frozen VLA에든 적용 가능
- LIBERO-PRO 15.0%→97.4% 같은 극단적 개선은 실용성 강력
- ITPS/DynaGuide가 high-perf policy에서 무력함을 명확히 보여 reframing 기여
- Real-world transfer 검증

**약점**:
- 새로운 모델/loss/architecture가 아님 → "policy steering" 카테고리에 가까움
- 표준 LIBERO 4-suite 숫자 부재로 leaderboard 비교 곤란
- 사람이 항상 필요 — fully autonomous 시나리오에는 부적합

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | SAPS가 VLA 모델인가, 단순 teleop interface인가? | 둘 다 아니고 **인터페이스+정책**. 정책 단독 26.7%(real)/15.0%(LIBERO-PRO)를 98%대로 끌어올리지만, 사람 input 0%면 그대로 정책 성능. 즉 augmentation framework. |
| 2 | LIBERO-PRO Cosine 97.4% vs Teleop 98.8%의 차이는 통계적으로 유의? | 본문에 직접 비교 통계 없음. 단 Cosine은 intervention 30%로 teleop 100% 대비 1/3.3 effort. 핵심 가치는 effort-success Pareto 우월. |
| 3 | Cosine sharpness k=6의 선택 근거? | 휴리스틱 (논문에 sweep 없음). k가 너무 작으면 transition이 부드러워서 정책 추종을 잃고, 너무 크면 binary switch가 되어 Takeover와 동일. |
| 4 | Cosine similarity가 음수일 때 α→0인데, 사람이 단지 노이즈를 입력했다면? | ε=1e-3 idle threshold가 1차 필터. 그 이상의 magnitude면 의도된 input으로 간주. 게임패드의 stick noise가 1e-3 이상이면 false trigger 가능 — 논문에서 명시적 분석 없음. |
| 5 | DynaGuide가 π0.5에서 -5.7%인 이유는 SAPS 우월성의 증거인가? | 부분적으로 그렇다. DynaGuide는 unguided 50% 미만 정책에서 학습됐는데 91% 정책에서는 head-room이 없어 perturbation이 손해. SAPS는 cos≈+1에서 α≈1로 정책을 보존하는 구조라 high-perf에서도 안전. |
| 6 | Gripper max() 규칙의 위험성 (close bias)? | "정책은 open 원하는데 사람이 실수로 close"인 경우 close가 우선되어 정책 의도를 무시. 안전 default(잡는 게 떨어뜨리는 것보다 낫다)지만 fragile object 시나리오에선 위험. |
| 7 | OpenVLA처럼 AR token-based 정책에도 적용 가능? | 원리상 가능 — 출력 7-DoF action에만 의존. 단, AR token policy는 action chunk가 짧고 replan 주기 다름 → α 적용 단위가 변함. 미검증. |
| 8 | Real-world에서 Open Cabinet 100→95% 감소(Cosine)는 왜? | 논문 명시 없음. 추정: 정책 일치 시(cos~+1) α≈1로 정책 그대로 → 정책이 마지막 5%에서 contact-rich 단계 실패. Blending(50/50)은 100%인 게 그 증거. |
| 9 | 표준 LIBERO 4-suite 숫자가 없는 이유? | 비용 + scope. perturbation study에 집중. 한계로 직접 인정. |
| 10 | SAPS의 "정책 confidence"는 진짜 confidence가 아니라 "사람과의 동의"인데, 정책과 사람이 둘 다 틀린 경우(both wrong, but agree)? | 본질적 한계. cos≈+1 → α≈1로 잘못된 정책 그대로. 논문은 "operator가 정확하다"는 implicit 가정에 의존. assistive 시나리오라 합리적이나 fully autonomous로 일반화 X. |
| 11 | Inference overhead? | Cosine 계산 7-dim 내적 + sigmoid 1번 → 무시 가능 수준. control rate 20Hz 유지. |
| 12 | 이 방법의 데이터 수집 측면 활용? | 저자도 conclusion에서 언급 — sparse correction을 데이터로 모아 imitation learning에 활용 가능. Real-to-Sim-to-Real Shared Autonomy[22]와 결합 잠재력. |

<!-- VERIFIED: pdf -->
