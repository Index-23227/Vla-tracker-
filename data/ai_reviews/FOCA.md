# FOCA: Future-Oriented Conditioning for Data-Efficient Vision-Language-Action Adaptation

> **한 줄 요약**: pi0 / GR00T-N1.5 같은 diffusion VLA의 fine-tuning 단계에 두 종류의 학습 가능한 future 토큰(r_exp는 객체 중심 interaction 영역의 latent patch 예측, r_imp는 미래 goal embedding과의 InfoNCE 정렬)을 plug-and-play로 끼워넣어 few-shot adaptation 효율을 끌어올린 ICML 2026 framework. LIBERO 100% 평균 96.6 (SOTA), 40% 데이터로 94.0 (pi0 100%와 동등), DreamGen 합성영상까지 쓰면 40% 데이터로 95.7. RoboCasa(GR00T-N1.5)에서 baseline 26 → FOCA 34.4.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 최신 VLA(pi0, GR00T-N1.5, EO-1, SmolVLA)는 large-scale pretraining으로 generalist 정책을 만들지만, **task-specific adaptation에 여전히 수십~수백 demonstration이 필요**.
- 본 논문이 직접 stress-test: 10% data(태스크당 약 5 demo)로 줄이면 pi0 77.6%, GR00T-N1.5 78.2%, SmolVLA 77.3%로 급락 (Table 1a) — pretraining만으론 few-shot 적응이 잘 안 됨.
- 기존 future-prediction 계열(CoT-VLA, DreamVLA, FLARE 등)은 **pixel-level future frame 재구성**(연산비/redundancy ↑) 혹은 사전학습 단계에서만 작동.
- PEFT(LoRA, DoRA, ControlVLA)는 파라미터 효율은 좋지만 supervision 신호 자체를 늘려주지 않음 → 10% 데이터에서 평탄화(~78%).

### 핵심 질문
- **demonstration 한 개당 더 많은 학습 신호**를 어떻게 뽑아낼 수 있는가?
- pixel reconstruction 없이 long-horizon 구조(미래 frame에 담긴 정보)를 representation space에서 직접 활용 가능한가?
- pseudo-action / inverse dynamics 없이 **action-free 합성 비디오**를 supervision으로 쓸 수 있는가?

📌 [Figure 1 삽입] — F_θ(VLM) 내부에 (l, o_t, r_exp, r_imp) 4종 토큰을 함께 흘리고, 그 출력이 A_ϕ(diffusion action policy)에 cross-attention으로 conditioning. r_exp는 미래 interaction patch의 latent 예측, r_imp는 InfoNCE로 미래 goal embedding 정렬.

---

## 2. 방법론 심층 분석

### 2.1 두 가지 future 토큰
- **Explicit tokens r_exp ∈ R^{n_e × d}**: 학습 가능 파라미터로 초기화 (기본 n_e=8, 시점당 분배), 비전+언어 토큰과 함께 VLM에 입력. VLM 출력 r̃_exp → 작은 Transformer predictor P_exp → 미래 시점 t+의 interaction 영역 patch embedding (R^{N_p × d_e})을 회귀.
- **Implicit tokens r_imp ∈ R^{N × d}**: 현재 frame을 frozen vision encoder + Pool + W_imp로 초기화(즉 dynamic/adaptive). VLM 출력 r̃_imp → projection head P_imp → contrastive embedding z_t.

### 2.2 두 가지 future objective

**Explicit (L_exp)** — 객체 중심 latent prediction:
- 외부 grounding 모델 G(·)로 instruction l에 해당하는 객체+gripper bbox들을 미래 frame I_{t+}에서 검출 → union B_{t+}.
- target ỹ_{t+} = Patch(I_{t+} | B_{t+}) ∈ R^{N_p × d_e} (frozen vision encoder의 패치 토큰 중 bbox 내부만).
- ŷ_{t+} = P_exp(r̃_exp, PE(B_{t+})) 와 L2 회귀.

**Implicit (L_imp)** — InfoNCE goal alignment:
- positive: 같은 episode의 미래 frame goal embedding y_{t+} = Pool(V(I_{t+}) | B_{t+}).
- negatives N_¬task: 같은 minibatch 내 **다른 instruction**의 episode goal들 (semantic hard negatives).
- 미래 offset k ~ Geom(1-γ) → discounted goal occupancy 의미.

### 2.3 Structured Token Isolation Mask
VLM attention 안에서 **Attn(r_exp ↔ r_imp) = 0**으로 강제하고 (r_exp ↔ vision/text), (r_imp ↔ vision/text)만 허용. Bidirectional / causal backbone에 따라 다르게 구현. 이로써 두 토큰 군이 서로 supervision을 leak하지 않고 상보적 역할을 학습.

### 2.4 Co-training & Action-Free 모드
- 전체 손실: **L_total = L_fm + L_exp + L_imp** (real demo).
- Action-free 모드(DreamGen 합성영상): **L_fm만 제거**, L_exp + L_imp는 그대로 — 액션 라벨 / pseudo-action / inverse dynamics 전부 불필요.
- 2단계 curriculum: (i) DreamGen 영상으로 L_imp 사전학습 → (ii) real demo로 L_total co-training.

> ❓ **예상 질문**: explicit/implicit이 정말로 다른 역할을 학습하는가?
> **답변**: §4 이론 분석으로 정당화. Geometric sampling 하에서 L_imp의 최적 score는 `log ρ_μ(g|x_t) − log p_neg(g) + c(x_t)` (Thm 4.1) — goal-conditioned discounted occupancy의 log-ratio, 즉 value-like reachability. L_exp의 최적 예측자는 `E_{g~ρ_μ(·|x_t)}[g]` (Prop 4.2) — 같은 occupancy의 **conditional mean**. 즉 implicit은 **log-density ratio**(reachability), explicit은 **mean future representation**(concrete summary) — 정보론적으로 명확히 다른 통계. Token isolation mask가 이 분업을 강제.

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO 4 suite** (Spatial, Object, Goal, Long-10) — 10% (≈5 demos/task), 40% (≈20), 100% (≈43) 세 split.
- **RoboCasa Kitchen** — 5 challenge tasks(PnPCabToCounter, PnPCounterToCab, CoffeeSetupMug, TurnOffStove, TurnOnMicrowave), 30-demo / 100-demo regime.
- **DreamGen 합성 비디오** — 실제 demo로 fine-tune된 video world model이 task description + initial frame 조건으로 prompt-conditioned rollouts 생성. LIBERO 10/40/100% 별로 각각 33,587 / 133,851 / 269,918 프레임.
- **Real robot**: bimanual ALOHA (Place Parts, Open Bag, Set Table, Tie Shoelaces) + 시뮬레이션 humanoid VinR-H3.

### 데이터 사용 패턴
- LIBERO/RoboCasa segmentation mask는 시뮬레이터에서 직접 추출 → grounding model G(·) 부담 제거.
- Real-world에서는 off-the-shelf open-vocab detector로 bbox 추출.

> ❓ **예상 질문**: grounding model G에 의존하면 detection 실패가 누적되지 않나?
> **답변**: L_imp는 grounding이 부정확해도 InfoNCE의 negative contrast로 robust하게 학습됨 (논문 §4.1 강조). 합성 영상에서도 정확한 bbox 없이 작동. 반면 L_exp는 bbox에 더 민감하나 ablation(Fig 3b)에서 object-centric latent가 full-image pixel prediction보다 일관되게 우수.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone (검증된 host VLA) | pi0 (PaliGemma + flow-matching), GR00T-N1.5 (Eagle + DiT action head) |
| 추가 모듈 | r_exp (n_e=8 default), r_imp (N pooled), P_exp (small Transformer), P_imp (MLP projector) |
| Conditioning 경로 | pi0: layer-wise cross-attention 중간 VLM features 사용 / GR00T-N1.5: 마지막 VLM embedding을 매 layer cross-attention |
| Loss | L_fm + L_exp + L_imp (action-free 모드는 L_fm 제거) |
| Future sampling | k ~ Geom(1-γ) — discounted occupancy |
| Hardware | 단일 NVIDIA H100 |
| LIBERO 학습 시간 (pi0) | baseline 18h / FOCA 21h (100k steps) |
| RoboCasa 학습 시간 (pi0) | baseline 20h / FOCA 23h (100k steps) |
| LIBERO batch | 18 (pi0) |
| RoboCasa batch | 12 (pi0), 32 (GR00T-N1.5) |
| RoboCasa lr (GR00T-N1.5) | 1e-4 |
| 평가 trial | LIBERO 4 suite × 50 trials/task; RoboCasa 50 trials/task |

---

## 5. 실험 설계 및 평가 프로토콜

세 가지 연구 질문:
- **Q1 (data efficiency)**: LIBERO에서 10% / 40% / 100% 데이터 budget으로 PEFT/SOTA 대비 비교.
- **Q2 (action-free synthetic video)**: pseudo-action(IGM, VPT-style) 방식과 FOCA(implicit only / + DreamGen) 비교.
- **Q3 (cross-architecture & real-robot generalization)**: pi0와 GR00T-N1.5 모두에 적용, RoboCasa + ALOHA + humanoid 검증.

📛 Baseline 선정: pi0, GR00T-N1.5, EO-1, SmolVLA (current SOTA generalist), 그리고 PEFT(LoRA r=64, DoRA r=64, Control-VLA), future-oriented(DreamVLA, FLARE). EO-1은 default 40h 학습이 prohibitive해 재구현 (FOCA는 20h 내 수렴).

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO few-shot stress test (Table 1a)

| Method | 100% Avg | 40% Avg | 10% Avg |
|--------|---------:|--------:|--------:|
| pi0 | 94.6 | 89.9 | 77.6 |
| GR00T-N1.5 | 94.6 | 91.4 | 78.2 |
| EO-1 | 94.1 | 91.0 | 82.2 |
| SmolVLA | 92.5 | 90.3 | 77.3 |
| **FOCA (pi0)** | **96.6** | **94.0** | **85.3** |

- **10% 데이터**에서 +7.7p (pi0 대비) — few-shot regime에서 효과 가장 큼.
- **40% 데이터**의 FOCA(94.0)가 pi0 100% 데이터(94.6)와 거의 동등 — 데이터 효율 ~2.5x.

### 6.2 LIBERO 4 suite (100% data, Fig 2 / Table 1)

| Suite | pi0 | **FOCA** |
|-------|----:|---------:|
| Libero-10 (Long) | 90.0 | **92.4** |
| Libero-Goal | 95.4 | **97.4** |
| Libero-Object | 98.2 | **99.8** |
| Libero-Spatial | 94.6 | **97.0** |
| **Avg** | 94.6 | **96.6** |

- 비교 군 광범위 (Diffusion Policy, Octo, OpenVLA, Spatial-VLA, CoT-VLA, DreamVLA, GR00T-N1.0/N1.5, EO-1, Think-Act, SmolVLA, pi0-fast 포함) — FOCA가 모두 능가.

### 6.3 PEFT 비교 (Table 1b)

| PEFT method | 100% | 40% | 10% |
|-------------|----:|----:|----:|
| Control-VLA | 95.6 | 91.3 | 78.4 |
| LoRA (r=64) | 94.2 | 90.2 | 78.2 |
| DoRA (r=64) | 94.7 | 92.0 | 78.6 |
| **FOCA** | **96.6** | **94.0** | **85.3** |

- 10% 데이터에서 PEFT는 ~78% 평탄화, FOCA만 85.3 — supervision 신호량의 문제이지 파라미터 효율의 문제가 아님을 시사.

### 6.4 Action-free synthetic video (Table 2)

| Method | 100% Avg | 40% Avg | 10% Avg |
|--------|---------:|--------:|--------:|
| pi0 | 94.6 | 89.9 | 77.6 |
| IGM (inverse generative pseudo-actions) | 94.3 | 90.2 | 76.8 |
| FOCA (Implicit only) | 95.8 | 93.0 | 83.6 |
| **FOCA + DreamGen** | **96.7** | **95.7** | **86.4** |

- **IGM은 거의 무효** (저데이터에선 오히려 -0.8p) — pseudo-action noise가 누적.
- **FOCA + DreamGen 40% (95.7) > pi0 100% (94.6)** — 합성 영상만으로 50% 이상 demo 줄이고도 SOTA.

### 6.5 RoboCasa 100-demo (Table 7) — GR00T-N1.5 backbone

| Method | PnP Cab→Counter | PnP Counter→Cab | Coffee Setup | Off Stove | On MW | **Avg** |
|--------|--:|--:|--:|--:|--:|--:|
| GR00T-N1.5 baseline | 28 | 40 | 12 | 14 | 36 | 26.0 |
| + FLARE | 24 | 34 | 20 | 14 | 50 | 28.4 |
| **+ FOCA** | **30** | **36** | **10** | **24** | **72** | **34.4** |

- TurnOnMicrowave +36p, TurnOffStove +10p — FLARE 대비 +6p 평균.
- 흥미롭게 CoffeeSetupMug에서는 FOCA가 FLARE보다 낮음(10 vs 20) → task별 trade-off 존재.

### 6.6 Real-robot (Fig 3a, Table 8)

- VinR-H3 humanoid Place Parts: GR00T-N1.5 58% → **FOCA 84%** (+26p, **논문 abstract의 "26% absolute gains" 출처**).
- ALOHA Tie Shoelaces: FOCA 95%.
- ALOHA Set Table (Table 8): pi0 10% → FOCA 45% → **FOCA + DreamGen 60%** (100% data).

---

## 7. Ablation 분석

### 7.1 Explicit module (Fig 3b)
- Latent object-centric prediction > pixel-level full-image prediction (10% data, Libero):
  - FOCA explicit only **93.3** (40%) / 79.2 (10%, 8 toks)
  - Full-images pixel **90.9 / —**
  - Object-centric pixel **91.4 / —**
- Token 수 sweep (10%): 8 toks(default) **79.2** > 18 toks 76.4 > 2 toks 76.6 — 적당한 token 수가 최적.

### 7.2 Implicit module (Fig 3e, 10% data)
- Single future frame alignment (t+10, default) **83.6**
- Multi-frames (t+10,20,30) 83.3 (≈ single)
- Full images (object-centric 제거) 81.7 (-1.9p)
- 대안 contrastive(image-text vs robot state) 78.2 / multi-frames vs text 78.3 — 모두 열등.
- **단일 well-aligned future frame + object-centric region**이 핵심.

### 7.3 Token initialization (Table 3, 4)
| 변형 | 40% Avg | 100% Avg |
|------|---:|---:|
| pi0 baseline | 89.85 | 94.6 |
| FOCA (random both) | 91.7 | — |
| FOCA (adaptive impli, random expli) | 93.95 | 96.6 |
| **FOCA (both adaptive)** | **94.3** | **97.0** |

- Vision feature로 초기화된 dynamic implicit token이 random보다 +2.6p — feature grounding 중요.

### 7.4 vs. DreamVLA / FLARE (Table 5, 6, 7)
- LIBERO 40% (pi0): DreamVLA 92.1, FLARE 91.0, **FOCA 94.0**.
- LIBERO 100% (pi0): DreamVLA 95.6, **FOCA 96.6** (+1.0).
- RoboCasa 100-demo (GR00T-N1.5): FLARE 28.4, **FOCA 34.4** (+6).

---

## 8. 관련 연구 비교

| 모델 | Future signal | Pixel? | Action-free 가능? | 적용 시점 |
|------|--------------|-------|-------------------|----------|
| CoT-VLA | Visual chain-of-thought (future image) | ✓ pixel | ✗ | 사전학습+adapt |
| DreamVLA | Dynamic future conditioning | △ | ✗ | adapt |
| FLARE | Future alignment (pretraining) | latent | ✗ | **pretraining only** |
| FOCA | Explicit latent + implicit InfoNCE | latent | ✓ (L_fm 제거) | **adaptation/fine-tuning** |

### 핵심 차이
- FLARE는 diffusion policy의 사전학습에 future alignment 주입. FOCA는 adaptation/fine-tuning 단계에 주입 → **이미 FLARE 사전학습된 GR00T-N1.5 위에서도 추가 gain**(Table 7).
- pixel reconstruction 회피 + InfoNCE의 negative contrast로 grounding 잡음에 robust.

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **외부 grounding model G 의존**: bbox 품질이 L_exp 직접 supervision. 시뮬레이터(분할 마스크)에서는 무료지만 실제 환경에선 open-vocab detector의 일반화 한계.
2. **diffusion/flow-matching backbone에 한정 검증**: pi0, GR00T-N1.5만. 순수 autoregressive VLA(OpenVLA, RT-2 등)에서의 효과 미검증 — token-isolation mask 구현이 causal attention에서 더 복잡.
3. **DreamGen 의존**: action-free 이득의 큰 부분이 video world model 품질에 좌우. DreamGen이 안 되는 도메인(고복잡 manipulation)에서 일반화 불명확.
4. **RoboCasa CoffeeSetupMug 후퇴**: FLARE 20 vs FOCA 10 — task-specific trade-off에 대한 분석 없음.
5. **hyperparameter sensitivity 미공개**: γ (geometric distribution), n_e (token 수)의 sweep은 부분적으로만 (8/18/2 비교).
6. **inference cost 명시 X**: 학습 비용은 +3h 정도지만 추론 시 r_exp/r_imp 토큰이 VLM forward에 매번 들어가는데 latency overhead 정량화 부재. 다만 미래 예측은 inference에선 안 함 (token은 conditioning만).

### Attribution 문제
- 96.6 SOTA가 explicit/implicit 두 모듈의 시너지인지 아니면 r_imp의 contrastive supervision 단독 효과인지 — Fig 3d에서 explicit-only vs implicit-only가 모두 full FOCA보다 낮음을 보였으나 정량적 분리는 부분적.
- DreamGen 효과 vs FOCA 효과 분리: Table 2는 두 변수를 분리하지만 video world model 품질이 좋은 도메인(LIBERO)이라 일반화 한계.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 두 future 토큰 + structured isolation mask + action-free DreamGen co-training 조합이 깔끔. value-like reachability 이론 분석이 동기를 정당화. |
| **Technical depth** | ★★★★★ — Thm 4.1 / Prop 4.2의 occupancy-mean / log-ratio 분리, geometric sampling, token isolation 모두 잘 짜임. |
| **Experimental rigor** | ★★★★★ — LIBERO 3 데이터 budget × 다수 baseline + RoboCasa 5 task × 2 backbone + 4종 real-robot + DreamVLA/FLARE 재구현 직접 비교. ablation 광범위. |
| **Practical impact** | ★★★★★ — 40% 데이터로 pi0 100% 추격, DreamGen 결합 시 추월. real-robot에서 +26p. drop-in module이라 기존 VLA 파이프라인에 즉시 적용. |
| **Writing quality** | ★★★★☆ — Fig 3 dense하지만 핵심 결과 명료. Appendix proofs 충실. |

**강점**: (i) Few-shot adaptation을 supervision 신호량 문제로 재정의하고 future-conditioning이라는 정공법 답안 제시. (ii) Pixel reconstruction 없이 latent에서만 작동 → 효율적. (iii) Action-free 합성 비디오를 pseudo-action 우회 없이 직접 supervision으로 사용. (iv) pi0와 GR00T-N1.5 두 이종 backbone에서 모두 작동. (v) 이론적으로 두 loss가 occupancy의 서로 다른 통계를 추정함을 명시.

**약점**: (i) Grounding model 의존, (ii) diffusion VLA 한정 검증, (iii) DreamGen 품질에 일부 결합, (iv) RoboCasa task-level 변동성 분석 부족.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | FLARE가 이미 GR00T-N1.5 사전학습에 future alignment를 넣었는데 FOCA가 또 효과를 보는 이유? | FLARE는 pretraining-stage, FOCA는 adaptation-stage 주입. Table 7에서 FLARE 28.4 vs FOCA 34.4 (RoboCasa 100-demo)로 직접 비교. 사전학습과 fine-tuning에서 잡는 future 구조가 다른 distribution이라 상보적. |
| 2 | Grounding model G가 실패하면? | L_imp는 contrastive라 noise tolerant (논문 강조). L_exp는 더 민감하나 ablation에서 object-centric latent가 full-image pixel보다 일관되게 우수 — bbox가 대략적이어도 task-relevant 영역으로의 attention biasing이 핵심. |
| 3 | autoregressive VLA(OpenVLA, RT-2)에는 못 쓰나? | 논문에서 검증 X. token isolation mask가 bidirectional 가정에서 simple하나 causal masking에선 구현 변형 필요. 원리상 적용 가능하나 실험적 검증 부재. |
| 4 | DreamGen 없이 pure implicit만으로도 충분한가? | Table 2: FOCA(Imp only) 95.8 / DreamGen 96.7 (100%), 83.6 / 86.4 (10%). DreamGen 기여 의미는 있으나 implicit만으로도 이미 pi0 능가. |
| 5 | n_e=8 token 수의 근거? | Fig 3b ablation: 2 / 8 / 18 비교에서 8이 최적 — 너무 적으면 capacity 부족, 너무 많으면 noisy negatives + sparse supervision. |
| 6 | Theorem 4.1의 가정(negatives가 x_t와 독립)이 실제 minibatch에서 성립하나? | task-mismatched episode에서 샘플링 (l′ ≠ l_m)하므로 instruction-conditional independence는 근사적으로 만족. exact independence는 아니나 InfoNCE bound가 robust. |
| 7 | 96.6 SOTA가 단순히 +파라미터 효과 아닌가? | Table 1b에서 LoRA/DoRA(r=64)는 비슷한 추가 파라미터에도 78%대 (10%)에서 평탄 — supervision 신호량의 차이임을 시사. |
| 8 | inference latency overhead는? | 학습 시간만 +3h 정도 보고. inference에선 r_exp/r_imp가 VLM forward에 포함되나 미래 prediction head는 호출 안 함. 정량 latency 보고 X — 후속 reproducibility의 quibble point. |
| 9 | CoffeeSetupMug에서 FLARE보다 떨어진 이유? | 논문에 직접 분석 없음. 해당 task가 long-horizon precision-grasp 위주라 future bbox grounding이 부정확할 가능성 추정. task-specific failure mode 분석은 future work. |
| 10 | 40% data로 pi0 100% 추격은 진짜 데이터 효율인가, 아니면 정규화 효과인가? | 둘 다. Fig 3d에서 explicit/implicit 단독 ablation도 baseline 능가 — 단순 regularization 그 이상의 future-structure supervision. |
| 11 | real-robot 26%p gain은 cherry-picking인가? | VinR-H3 Place Parts 한 task의 max gain. Aloha 평균은 더 모더레이트 (Set Table 10→60, Tie Shoelaces 95). Fig 3a 4개 task 평균을 봐야 공정. |
| 12 | DreamGen이 만든 합성영상의 unrealistic frame이 L_imp에 노이즈 되지 않나? | 저자 주장: InfoNCE의 negative contrast로 잡음에 robust. Table 2의 FOCA+DreamGen이 일관 best임이 경험적 근거. 다만 worst-case rollout quality 분석 부재. |

<!-- VERIFIED: pdf -->
