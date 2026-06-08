# CR-VLA: Continuous Reasoning for Vision-Language-Action

> **한 줄 요약**: π0.5 백본 위에 **N_τ=2개 연속 thought slot**을 WAE로 공유 Gaussian 잠재공간에 매핑하고, **EMA teacher self-verification**으로 "다른 모델 인스턴스가 같은 잠재 코드를 소비해 더 나은 action을 예측해야 한다"는 재사용성을 학습 시 강제. 결과: LIBERO-PRO suite mean **58.0 → 64.0**, TX-G2 mean subtask success **π0.5 대비 +40.4%**, HSR **+26.3%**. 텍스트 CoT 토큰을 더하는 게 아니라 **action을 위한 내부 언어(internal language for action)** 를 학습한다.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **언어 기반 CoT (ECoT, CoT-VLA, dVLA)**: 텍스트 reasoning trace는 공유 가능(shareable)하고 검증 가능(verifiable)하지만, **연속 제어의 granularity와 시간 스케일이 불일치**. "subtask 1 → subtask 2"라는 추론 한 단계가 수많은 action chunk에 걸쳐 펼쳐지므로 *지금 이 시점의 action*에 대한 기여는 약하고 간접적.
- **시각 subgoal (SuSIE, CoT-VLA)**: 미래 이미지를 생성하지만 *어디로 가야 하는지*만 알려줄 뿐 *지금 어떤 action chunk*인지에 직접 대응하지 않고 prediction cost가 큼.
- **잠재 reasoning (ThinkAct, FAST-ThinkAct 등)**: 연속 공간으로 내부화했지만 **"좋은 action 예측 = 좋은 reasoning"이 아님**. LLM 분야 연구 (Lanham 2023, Yu 2026)에서 outcome-based RL로 학습된 reasoning trace가 신뢰성/검증가능성 없이도 final accuracy를 올릴 수 있음이 보고됨 — VLA도 같은 함정.

### 핵심 질문
- **VLA에서 "언어의 역할"을 무엇이 대신해야 하는가?** 저자는 세 조건 제시:
  1. **재사용성(reusable)**: 다른 모델 인스턴스가 그 reasoning trace로부터 이득을 봐야 함
  2. **공유성(shareable)**: 표현이 공통 잠재 공간에 존재해 전달·소비 가능
  3. **추상화 정렬(abstraction-aligned)**: low-level motor fluctuation보다 위, 자유로운 자연어보다 아래 — **chunk-level 제어와 같은 시간 스케일**

📌 [Figure 1 삽입] — VLA-A가 continuous thoughts를 만들고 공유 Gaussian 잠재 코드로 매핑, decode하여 chunk-causal action 예측. VLA-B(=EMA teacher)는 같은 잠재 코드를 자신의 WAE decoder로 풀어 action을 예측 — **재사용성의 운영적 정의**.

---

## 2. 방법론 심층 분석

### 2.1 문제 설정 및 백본
- 관측 o, 명령 x, 미래 action sequence a ∈ ℝ^(H×d), chunked flow-matching 백본 (π0.5).
- **H=16, C=4, K=4** (action horizon = num_chunks × chunk_size). 이 chunked 분해 자체가 "reasoning이 작동해야 할 추상화 레벨"의 정의.

### 2.2 Continuous Thoughts (N_τ=2 slots)
- 원시 thought vector τ = [τ_1, τ_2], 각 τ_i ∈ ℝ^D를 **순차적으로** 생성. 각 slot 계산 후 decoded interface를 prefix에 다시 써넣고 KV cache에 commit → 다음 slot이 attention 가능.
- **텍스트로 디코딩되지 않으며 action chunk와 1:1 매핑도 아님**. 대신 horizon 전체에 걸친 **compact reasoning scaffold**.

### 2.3 Shareable Latent Geometry (WAE)
- Encoder q_φ: τ → z (d_z=128), decoder p_ψ: z → τ̂_S
- **L_wae = λ_rec · ‖τ − τ̂_S‖² + λ_mmd · MMD(z, 𝒩(0, I))**
- Wasserstein autoencoder(Tolstikhin 2017) + IMQ-MMD 커널. "코드를 의미적으로 보장하기 위해서가 아니라, **공통 잠재 geometry를 강제**해 다른 인스턴스가 풀어 쓸 수 있도록 하기 위함."
- 분산 학습 시 **MMD를 device 전체에 걸쳐 글로벌 배치로 계산** (로컬 sub-batch MMD는 의도된 글로벌 분포를 보지 못함) — 구현 상의 핵심 디테일.

### 2.4 Flow-Matching Action Head
- x_t = (1−t)a + tε, target velocity u_t = ε − a
- Student: v_θ(o, x, x_t, t; τ̂_S) → **L_fm = E‖v_θ − u_t‖²**
- **Within chunk**: flow matching이 chunk 내부를 **bidirectional**로 한 번에 예측 (motor-step AR 없음)
- **Across chunks**: **block-causal attention** — 같은 chunk 내 토큰은 한 attention block 공유, 후속 chunk는 이전 chunk attend

### 2.5 Self-Verification: 진짜 핵심
- EMA teacher 파라미터 θ̄ (γ_EMA=0.994). 같은 잠재 코드 z를 teacher의 WAE decoder가 풀어 τ̂_ema 생성:
- **L_verify = E‖v_θ̄(o, x, x_t, t; τ̂_ema) − u_t‖²**
- "**다른 모델 인스턴스가 같은 공유 코드를 받아 action을 잘 예측해야 한다**"가 학습 시 강제됨. 단순히 student가 잘 활용하는지가 아니라 *다른 인스턴스에 의해서도 활용 가능해야* 한다는 reusability의 운영적 정의.

> ❓ **예상 질문**: 단순 consistency loss와 무엇이 다른가?
> **답변**: 일반적인 EMA consistency (Mean Teacher, BYOL)는 teacher의 *예측*과 student의 *예측*을 align. 여기서는 teacher가 **student가 만든 잠재 코드 z를 입력으로 받아** 자신의 디코더로 풀어 *행동을 예측*해야 함. 즉 EMA teacher는 reasoning interface의 **소비자(consumer) 역할**을 하며, z가 producer-specific shortcut으로 collapse하는 것을 방지.

### 2.6 Abstraction Alignment
- Chunk-causal mask로 *chunk 단위*에 시간 의존성 배치 (motor step 단위 X)
- **per-chunk shortcut path 없음**: action 생성이 shared reasoning prefix를 반드시 소비하도록 — private shortcut으로 bypass 불가
- → Reasoning이 low-level motor perturbation으로 붕괴하지 않도록 구조적으로 차단

### 2.7 최종 목적함수
**L = L_fm + L_wae + λ_verify · L_verify** (λ_verify = 0.1)

---

## 3. 데이터 전략

### 3.1 시뮬레이션
- **LIBERO-PRO** (Zhou et al., 2026, arXiv 2510.03827): 4 suites (spatial / object / goal / 10) × 10 tasks × 4 perturbation types (object / position / semantic / task) × 20 trials = **suite-perturbation cell당 200 에피소드**
- LIBERO-PRO를 메인 벤치마크로 선택한 이유: 표준 LIBERO success rate는 최신 VLA들이 saturate해 robustness/distribution shift를 분리할 수 없음. position과 task perturbation은 **action retargeting**을 직접 요구해 reasoning의 가치를 진단할 수 있음.

### 3.2 실제 로봇 데이터
| 플랫폼 | 트래젝토리 | 태스크 패밀리 |
|--------|-----------|--------------|
| TX-G2 (bimanual, 3 cam, 10 Hz) | 1198 | Bowl Stacking 129 / Clothes Sorting 514 / Cutlery Transfer 206 / Dish Racking 349 |
| HSR (mobile manipulation, 2 Hz) | 1205 | Mug Rectangle 399 / Coffee Bottle→Box 295 / Box Rearrangement 195 / Coffee Bottles→Table 316 |

모든 실제 로봇 baseline은 **150k step fine-tune**. 평가 hardware는 RTX 5070(OOM으로 OpenVLA-OFT는 비교 제외).

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO-PRO Main (Table 1)

Suite mean (4가지 perturbation 평균) 기준 (단위 %):

| Method | spatial | object | goal | long(10) | **Suite Mean** |
|--------|---------|--------|------|----------|---------------|
| OpenVLA-OFT | 29.5 | 41.3 | 13.6 | 11.0 | 23.9 |
| X-VLA | 49.3 | 48.5 | 43.5 | 38.0 | 44.8 |
| VLA-Adapter | 61.3 | 49.0 | 37.0 | 37.0 | 46.1 |
| π0.5 | 76.0 | 53.3 | 57.8 | 45.0 | **58.0** |
| **CR (Ours)** | **76.1** | **68.4** | **61.6** | **49.6** | **64.0** |

- **+6.0 suite mean over π0.5**, 4개 suite 모두 1위
- **Position perturbation: 26.8 → 39.3 (+12.5)**, **Task perturbation: 24.8 → 37.1 (+12.3)** — action retargeting이 필요한 perturbation에서 큰 향상
- Object/Semantic은 ±2 이내 — 외양/문구만 바뀌는 perturbation은 단순 action template replay로도 풀 수 있어 "혜택이 spurious"할 수 있다고 저자 명시. **개선의 본질은 spatial re-anchoring과 task-level adaptation**.

### 4.2 실제 로봇 — TX-G2 (Table 2, Table A5)

Subtask success rate (4-task 평균):

| Method | Cutlery | Bowl | Clothes | Dish | **Mean** |
|--------|---------|------|---------|------|----------|
| X-VLA | 0.0 | 7.5 | 5.0 | 5.0 | 4.4 |
| VLA-Adapter | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| π0.5 | 5.0 | 47.5 | 83.3 | 60.0 | **48.95** |
| **CR (Ours)** | **22.5** | **70.0** | **95.0** | **87.5** | **68.75** |

- π0.5 대비 **+19.8 절대 / +40.4% 상대** 향상 (abstract의 40.4% 수치)
- **E2E(전체 task 완료)**: Bowl Stacking 0→40, Clothes Sorting 50→80, Dish Racking 20→60
- **Cutlery Transfer 5→22.5** — 가장 어려운 thin object 조작에서도 4배 이상 향상

### 4.3 실제 로봇 — HSR (Table 2, Table A6)

Subtask success rate:

| Method | Bott.→Box | Bott.→Table | Box Rearr. | Mug Rect. | **Mean** |
|--------|-----------|-------------|------------|-----------|----------|
| X-VLA | 8.3 | 4.2 | 0.0 | 25.0 | 9.4 |
| VLA-Adapter | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| π0.5 | 83.3 | 45.8 | 41.7 | 66.7 | **59.4** |
| **CR (Ours)** | 83.3 | **83.3** | **58.3** | **75.0** | **75.0** |

- π0.5 대비 **+15.6 절대 / +26.3% 상대** 향상
- **Coffee Bottles→Table**에서 45.8→83.3 (+37.5)이 가장 큰 향상 — 가장 long-horizon한 locomotion + 반복 manipulation 태스크 (선반→테이블→복귀→재방문)

### 4.4 CALVIN ABC→D (Appendix D, Figure A2)

Average sequence length (5-task chain):

| Method | Avg Len |
|--------|---------|
| OpenVLA-OFT | 3.27 |
| UniVLA | 3.80 |
| π0.5 | 3.92 |
| VLA-Adapter | 4.42 |
| X-VLA | 4.43 |
| **CR (Ours)** | **4.27** |

- CR은 X-VLA/VLA-Adapter보다는 살짝 낮으나 π0.5 + 0.35 — "메인 클레임을 위해 일반 long-horizon 성능을 희생하지 않았다"가 저자의 메시지. 즉 CALVIN은 saturate되어 reasoning의 가치를 분리하기 어렵다는 입장.

---

## 5. Ablation 분석 (Appendix C, Table A2)

LIBERO-PRO 4-suite 평균 (object/position/semantic/task):

| Variant | Object | Position | Semantic | Task | **Overall** |
|---------|--------|----------|----------|------|------------|
| **Full CR** | **84.6** | **39.3** | **94.9** | **37.1** | **64.0** |
| w/o Gaussian latent (WAE) | 86.0 | 30.2 | 95.5 | 24.9 | 59.2 (-4.8) |
| w/o chunk-causal mask | 86.5 | 32.5 | 96.0 | 29.2 | 61.1 (-2.9) |
| w/o continuous thoughts | 84.9 | 32.5 | 96.5 | 28.6 | 60.6 (-3.4) |
| w/o self-verification | 86.0 | 32.4 | 95.5 | 28.1 | 60.5 (-3.5) |

**핵심 관찰**:
1. **Gaussian latent 제거 시 가장 큰 손실** (-4.8 overall, position -9.1, task -12.2) — WAE 구조화가 reasoning interface의 공유성을 만드는 데 결정적
2. 4개 ablation 모두 Object/Semantic은 거의 그대로 — 다시 한번 **개선의 본질이 spatial/task adaptation에 있음**을 증명
3. Self-verification 제거 시 -3.5 — 단순 추가 잠재 채널이 아니라 EMA teacher가 실제로 reusability를 강제하고 있음

> ❓ **예상 질문**: thought slots을 N_τ=2개만 쓰는 이유? 더 많이 쓰면?
> **답변**: 논문은 N_τ=2를 default로 보고하나 sweep을 명시하지는 않음. KV cache 비용 (thought당 1회 추가 forward), 그리고 chunk-level reasoning의 추상화 레벨에 맞추려는 의도로 추측. **이는 약점**으로, N_τ scaling 실험이 빠진 점은 reasoning의 expressivity-cost 트레이드오프를 보여주지 못함.

---

## 6. 잠재 공간 진단 (Figure 4-6, Appendix C)

### 6.1 PCA 시각화 (LIBERO paired scenes)
- 같은 초기 scene에서 instruction만 바꾼 paired 비교, 각 5 rollouts
- "물체 in caddy" 태스크: 초기 approach는 유사 (둘 다 가까운 물체를 잡음) → 접촉 시점에서 가장 가까운 latent 거리 (grasp geometry 유사) → placement 단계에서 분기 (큰 책은 다른 insertion 전략 필요)
- **저자 해석**: "scene identity를 단순 encoding하지 않고 task phase + object-specific control demand로 reorganize"

### 6.2 TX-G2 동적 perturbation (Figure 5, 6)
- "Pick up the green socks" 태스크에서 **녹색 양말이 episode 중간에 외부 perturbation으로 좌→우로 던져지는 OOD 시나리오**
- Perturbed rollout의 latent: 초기엔 좌측 target reasoning pattern → 마지막엔 우측 target configuration으로 이동
- **online re-anchoring**의 증거. 데이터셋에 없는 within-episode displacement에 latent가 적응
- 또한 무관한 distractor object (training data에 없는 green fruit 등)가 던져져도 latent는 t≈6.0에서 안정적으로 유지되다가 실제 target(green socks) 등장 시 sharp transition

---

## 7. 관련 연구 비교

| 모델 | Reasoning 매체 | Shareable? | Verifiable? | Chunk-aligned? | LIBERO/실제 |
|------|---------------|-----------|-------------|----------------|------------|
| ECoT | Text (4-layer) | ✓ | △ (autoregressive token) | ✗ | 79.1 / N/A |
| CoT-VLA | Visual subgoal | ✓ | △ | ✗ | 81.1 / 7x slowdown |
| dVLA | Multimodal text CoT | ✓ | △ | ✗ | 96.4 / - |
| FLARE / ThinkAct | Latent planning | △ (single producer) | ✗ | ✗ | - |
| DualCoT-VLA | Parallel V+L CoT | ✓ | △ | ✗ | - |
| **CR-VLA** | **Shared Gaussian latent + WAE** | **✓ (강제)** | **✓ (EMA teacher 소비)** | **✓ (chunk-causal)** | **64.0 (LIBERO-PRO) / TX-G2 +40.4%** |

**저자 포지셔닝**: "VLA reasoning은 '더 많은 토큰을 생성하는 것'이 아니라 '**action을 위한 공유·검증 가능한 내부 언어를 학습하는 것**'이다."

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **CALVIN ABC→D에서 SOTA 미달** — X-VLA/VLA-Adapter보다 낮은 4.27 (vs 4.43). 저자는 "saturate되어 reasoning의 가치 분리 불가"라 변호하나, **다른 long-horizon 벤치 (RoboCasa, SimplerEnv 등)에서의 검증 부재**.
2. **표준 LIBERO 수치 없음** — 모든 LIBERO 수치가 LIBERO-PRO. 기존 leaderboard와의 직접 비교 어려움.
3. **OpenVLA-OFT는 RTX 5070 OOM으로 실제 로봇 비교에서 제외** — 가장 강력한 잠재 경쟁자가 빠진 셈. 더 큰 GPU에서의 fair comparison 필요.
4. **VLA-Adapter의 TX-G2/HSR 0% 결과**가 의심스러움 — 저자 자체 검증 (30-sample replay)으로 "mean-action baseline 수준"이라 진단했으나 fine-tuning recipe 문제일 가능성 배제 못함.
5. **N_τ, d_z, λ_verify의 sensitivity 부재** — 단일 default 구성만 보고. Reasoning capacity vs cost 트레이드오프 미해명.
6. **Verification objective의 collapse 위험 미논의** — EMA decay 0.994면 teacher가 student를 거의 따라가 결국 self-distillation으로 수렴할 가능성. γ_EMA sweep 없음.

### 클레임 vs 증거의 gap
- "**shareable across model instances**"는 EMA teacher (=student의 가까운 복사본)로만 검증. **진짜 다른 architecture/seed VLA가 z를 받아 잘 사용하는지**는 실험 부재. Reusability의 가장 강한 증거는 *서로 다른 backbone* 간 latent 전이 실험.

### 인프라/재현성
- **코드 미공개** (현재). AIRoA 단독 저자. NEDO 프로젝트 자금. 재현 비용 미상.
- TX-G2가 AgiBot G2 호환 변형이라 외부 재현 불가능에 가까움.

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "Reasoning = shareable + verifiable + chunk-aligned interface"라는 재정의 자체가 신선. WAE + self-verification 조합도 VLA 맥락에서 새로움. |
| **Technical depth** | ★★★★☆ — Block-causal mask, distributed MMD, chunk-aligned curriculum 등 implementation detail이 충실. 다만 hyperparameter sensitivity 부족. |
| **Experimental rigor** | ★★★★☆ — LIBERO-PRO 200 episodes/cell, 실제 로봇 10 episodes × 4 tasks × 2 platforms, 4-way ablation. 다만 baseline VLA-Adapter 0%는 의문. |
| **Practical impact** | ★★★★☆ — π0.5 대비 실제 로봇에서 +40.4% / +26.3% 향상은 실질적. 다만 백본은 π0.5에 종속. |
| **Writing quality** | ★★★★★ — Motivation (language를 무엇이 대신해야 하는가)이 매우 명확. Figure 1이 핵심 메시지 한 컷에 압축. |

**강점**: VLA reasoning의 **기준 자체를 재정의** ("좋은 reasoning은 다른 인스턴스가 소비 가능해야 한다"). 이 명제는 LLM reasoning faithfulness 연구(Lanham et al.)와 잘 연결되며, **단순히 "토큰을 더 만든다"식 reasoning과 명확히 분기**. LIBERO-PRO position/task perturbation에서 +12점대 향상은 reasoning이 spatial re-anchoring을 실제로 돕는다는 강력한 증거.

**약점**: Reusability 클레임 검증이 EMA teacher 자기복사에 머무름. 표준 LIBERO 수치 부재로 leaderboard 직접 비교 불가. 코드 미공개.

---

## 10. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | EMA teacher가 student의 거의 복사본이라면 "다른 인스턴스가 z를 소비"한다는 클레임이 정당화되는가? | 약점. 진짜 reusability는 다른 seed/architecture VLA로의 transfer 실험이 필요. γ_EMA=0.994는 효과적으로 student의 slow copy이며 self-distillation으로 수렴할 위험. |
| 2 | N_τ=2 thought slot이 왜 충분한가? Chunk가 4개인데. | 명시적 정당화 없음. Action chunk와 1:1 매핑이 아니므로 horizon-level scaffold라는 게 답이지만, N_τ scaling 실험 부재. |
| 3 | WAE가 가장 큰 ablation 효과(-4.8)인 이유는? | Latent이 unbounded continuous thought이면 producer-specific하게 collapse. Gaussian 강제가 "공유 가능한 좌표계"를 만들어 EMA teacher가 의미 있게 decode 가능. |
| 4 | LIBERO-PRO position/task에서만 큰 향상이고 object/semantic은 동일한 이유? | Object/semantic은 action template replay로도 풀 수 있어 "spurious benefit". Position/task는 spatial re-anchoring과 goal restructure를 강제 — 이게 reasoning이 실제로 도와야 하는 영역. |
| 5 | π0.5 대신 OpenVLA-OFT/X-VLA를 backbone으로 써도 동일한 향상? | 미검증. 백본 의존성이 클레임의 일반성을 제한. |
| 6 | Self-verification 제거 시 -3.5인데, 그래도 WAE만으로 60.5는 나옴. WAE 알맹이가 다인가? | 두 기제 모두 reasoning interface의 *재사용성*을 다른 방식으로 강제. WAE는 공간을 정렬, verification은 사용을 강제. 둘이 합쳐서 64.0. |
| 7 | TX-G2 +40.4%, HSR +26.3% 차이는 어디서? | TX-G2는 bimanual + arm 선택까지 요구(reasoning의 가치 큼). HSR은 locomotion이 dominate해 reasoning 기여 비중이 상대적으로 작음. 가장 long-horizon한 Coffee Bottles→Table (+37.5)에서 가장 큰 향상이 이를 뒷받침. |
| 8 | Inference cost는? Chunk당 thought 2 slot 추가가 latency에 미치는 영향? | 명시되지 않음. KV cache에 commit하므로 thought 생성은 일회성, action 생성 중 추가 forward는 없음. 그러나 정량적 latency 측정은 부재. |
| 9 | 표준 LIBERO 수치가 없는데, 다른 SOTA (Diffusion Policy 96.4, dVLA 96.4)와 어떻게 비교? | 직접 비교 불가. LIBERO-PRO와 표준 LIBERO는 다른 perturbation을 측정. 저자의 입장은 "saturated standard LIBERO는 reasoning의 가치를 분리 못함". 그러나 leaderboard에서의 외부 인지 측면에서는 약점. |
| 10 | "Reasoning 토큰을 더 만드는 게 아니다"라는 메시지가 향후 VLA reasoning 연구에 시사하는 바는? | CoT-VLA의 7x slowdown 같은 비용 문제를 우회. **Reasoning을 "내부 representation의 properties"로 재정의**하는 패러다임 shift 가능. RL/CoT 결합 연구가 reasoning trace의 *내용* 대신 *공유성/검증성*을 직접 최적화하도록 유도. |

<!-- VERIFIED: pdf -->
