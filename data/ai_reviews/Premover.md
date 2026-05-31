# Premover: Fast Vision-Language-Action Control by Acting Before Instructions Are Complete

> **한 줄 요약**: 사용자가 명령을 입력하는 동안(LIBERO 평균 39%의 interaction time, 52.24 WPM 기준 ~12초) 정책이 idle 상태로 머무르는 *streaming-prefix 구간*을 활용하기 위해, frozen pi₀.₅ 위에 두 개의 2-layer MLP projection head (image · language)를 부착하여 **simulator segmentation mask로 supervise된 focus map**과 **단일 스칼라 readiness threshold τ**를 학습시키는 2.36M-parameter plug-in. LIBERO에서 wall-clock 34.0s → 29.4s (-13.6%) 단축에도 baseline 95.0% vs Premover 95.1%로 success를 유지하며, naive premoving (66.4%) 대비 +28.7pp를 회복. UNIST/Catholic University of Korea.

---

## 1. 배경 및 동기

### 빠른 inference vs 사용자 입력 지연

기존 VLA acceleration 연구(DeeR-VLA, VLA-Pruner, TinyVLA, FAST)는 모두 **명령이 완료된 후의 inference latency**를 줄이는 데 집중. 그러나 실제 배포에서는 사용자가 명령을 입력하는 시간이 평균 12초(LIBERO 명령 = 12 단어 × 52.24 WPM) 소요되며, 이는 LIBERO 4 suite 평균 **interaction 시간의 39%**, 가장 짧은 LIBERO-Spatial에서는 **57%** 를 차지한다.

📌 [Figure 1 삽입] — LIBERO 4 suite의 instruction(hatched) vs forward+action(solid) 시간 비율: Spatial 17.7s+13.4s, Object 12.3s+18.2s, Goal 8.3s+15.6s, LIBERO-10 15.2s+35.1s, 평균 39%/61%.

### 핵심 통찰
- 표준 평가 protocol은 이 입력 시간을 *완전히 제거*한 oracle setting에서 측정.
- "Pick up the ketchup…"처럼 일부만 입력되어도 시각 장면의 대부분은 이미 irrelevant.
- 즉 **streaming prefix**가 유의미한 visual grounding prior를 제공할 수 있음.

### 두 가지 coupled challenge
1. **어디를 볼 것인가 (Where)** — partial language에서 referent를 localize. Frozen VLA backbone은 action imitation으로만 학습되었기에 background/distractor에 attention mass가 분산 (Figure 4의 "Original Attn. Map" 시각화).
2. **언제 시작할 것인가 (When)** — premature commitment는 잘못된 target으로의 motion 유발 (naive premoving이 95.0% → 66.4%로 collapse하는 이유).

---

## 2. 방법론 심층 분석

### 2.1 전반 구조

📌 [Figure 3 삽입] — Premover overview: 두 projection head(Vision Focus Head + Lang Focus Head)가 frozen pi₀.₅의 intermediate layer를 공유 latent space로 mapping → focus map (3.1) → Action Readiness Gate (3.2).

### 2.2 Focus Map 학습

**Projection heads** (2-layer MLP + GELU, L2-normalize):

$$z^{img}_i = \tilde{f}_{img}(H^{img}_i), \quad z^{lang}_j = \tilde{f}_{lang}(H^{lang}_j)$$

**Cosine similarity** in shared space, max over prefix tokens, sigmoid:

$$S_{i,j} = \langle z^{img}_i, z^{lang}_j \rangle, \quad p_i = \sigma\left(s \cdot \max_{j} S_{i,j}\right) \in [0,1]$$

- $s = 6.0$ (fixed logit scale).
- $N$: image patches (across 2 cameras: agent + wrist), $L$: streaming prefix tokens.

**Supervision** — Class-balanced BCE against simulator segmentation mask $m^\star$:

$$\mathcal{L}_{focus} = -\frac{\sum_i \beta_i [m^\star_i \log p_i + (1-m^\star_i)\log(1-p_i)]}{\sum_i \beta_i}$$

여기서 $\beta_i = N_-/\max(N_+,1)$ for positive patches (class imbalance 보정).

> ❓ **예상 질문**: 왜 max-over-token이지 mean-over-token이 아닌가?
> **답변**: 일반적으로 한 prefix token만이 target object를 명시하고 나머지는 verb/article. max는 가장 informative한 token signal을 그대로 살리는 반면, mean은 informative token이 dilute됨.

### 2.3 Focus Map Injection (input reweighting)

**핵심 디자인 결정**: focus map을 *같은 step*의 input에 적용하려면 forward pass가 2번 필요 (focus map은 그 step의 hidden state에서 계산되므로). 저자는 인접 timestep의 attention pattern이 유사하다는 사실 [24]에 근거하여 **step t의 focus map을 step t+1의 input reweighting**으로 사용.

$$w_{t,i} = \alpha + (1-\alpha) p_{t,i}, \quad \hat{e}^{img}_{t+1, i} = w_{t,i} \cdot e^{img}_{t+1,i}$$

- $\alpha = 0.2$ (calibration set에서 선택).
- $\alpha = 0$: non-target patch 완전 mute → obstacle avoidance/gripper alignment 같은 peripheral behavior 손상.
- $\alpha = 1$: injection 비활성화 → readiness-only로 reduce.
- 0.2가 *target amplification* 과 *scene context preservation* 의 sweet spot.

### 2.4 Action Readiness Gate

**Readiness score**:

$$r_t = \frac{1}{K}\sum_{i \in T_K(p_t)} p_{t,i} - \frac{1}{N}\sum_{i=1}^{N} p_{t,i}$$

- Top-K mean - Global mean: 활성화 *집중도* 측정.
- $K = 10$ (sweep으로 결정).
- 단순 top-K mean은 background noise로 inflate되므로 global mean을 차감.

**Threshold** — Learnable scalar $\tau$:

$$\text{execute action at } t \iff r_t \geq \tau$$

**Supervision** — Differentiable BCE with temperature $T = 0.10$:

$$\mathcal{L}_{ready} = \text{BCE}\left(\frac{r_t - \tau}{T},\, y\right)$$

여기서 $y \in \{0,1\}$ = target object가 prefix에 등장했는지 여부.

> ❓ **예상 질문**: target이 아직 prefix에 없을 때 focus loss는 어떻게?
> **답변**: 그 case에서는 readiness loss만 적용 (focus map은 *too early* signal에 해당하므로 noisy oracle mask 주입을 피함).

### 2.5 Streaming Prefix-Readiness Joint Loss

$$\mathcal{L} = \lambda_{focus} \mathcal{L}_{focus} + \lambda_{ready} \mathcal{L}_{ready}$$

$\lambda_{focus} = \lambda_{ready} = 1.0$. **학습 파라미터는 < 1%** (projection heads + scalar τ만, backbone은 frozen).

---

## 3. 실험 설계

### 3.1 Backbone 및 Benchmarks
- **Backbone**: pi₀.₅ (frozen, bfloat16).
- **Benchmarks**: LIBERO (Spatial/Object/Goal/L-10) + VLA-arena Level-1 (Extrapolation/Distractor/Safe/Long-horizon).
- 두 simulator 모두 per-instance segmentation mask 제공 → focus map supervision 가능.

### 3.2 데이터 split
- LIBERO: episode 0-9 (training projection heads), 10-14 (α calibration), **15-49 (evaluation, 35 rollout per task = 350 per suite)**.
- VLA-arena Level-1: episode 0-7 (train), 10-14 (calibration), 15-49 (evaluation).

### 3.3 Streaming protocol
- 52.24 WPM (Dhakal et al. [5] 평균).
- Gemma tokenizer ~4 char/token, policy loop ~13 Hz → **12 simulator step / token**.
- Wall-clock 측정은 first revealed token부터 terminal state까지.

### 3.4 Hardware
8 NVIDIA H200 (141 GB). Head set 학습은 GPU 1대.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO 4 suite (Table 1, 350 rollouts per suite)

| Metric | Setting | Spatial | Object | Goal | L-10 | **Mean** |
|--------|---------|---------|--------|------|------|----------|
| Success ↑ | Full-prompt | 99.4 | 97.4 | 94.9 | 88.3 | **95.0** |
| | Naive Premoving | 68.3 | 64.6 | 56.6 | 76.0 | 66.4 |
| | **Premover** | 98.6 | **99.1** | 93.7 | 88.9 | **95.1** |
| Wall (All) ↓ | Full-prompt | 31.0s | 30.7s | 23.8s | 50.8s | 34.0s |
| | Naive Premoving | 27.4s | 32.3s | 34.8s | 43.6s | 34.5s |
| | **Premover** | **22.7s** | 24.4s | **21.9s** | 48.6s | **29.4s (86.4%)** |
| Wall (Succ) ↓ | Full-prompt | 30.8s | 29.7s | 21.5s | 45.2s | 31.8s |
| | Naive Premoving | 15.5s | 19.6s | 16.8s | 32.2s | 21.6s |
| | **Premover** | 21.6s | 24.1s | 19.2s | 44.5s | **27.3s (86.0%)** |

**핵심 관찰**:
- Naive premoving은 wall (All) 34.5s로 *full-prompt보다 느림* (101.5%) — 실패 episode가 timeout까지 wander.
- Premover의 wall (All) 86.4% < Wall (Succ) 86.0% — 즉 성공 case뿐 아니라 실패 case도 빠르게 종료.
- Object suite에서 Premover (99.1%) > full-prompt (97.4%) → focus map이 distractor disambiguation에 *추가로* 도움. supervision 자체가 backbone의 implicit grounding을 *교정*하는 효과 시사.

### 4.2 VLA-arena Level-1 (Table 2)

| Metric | Setting | Extr. | Distr. | Safe | LongH | **Mean** |
|--------|---------|-------|--------|------|-------|----------|
| Success ↑ | Full-prompt | 25.1 | 39.4 | 41.8 | 0.0 | 33.0 |
| | Naive Premoving | 8.6 | 32.9 | 41.0 | 0.0 | 27.0 |
| | **Premover** | 25.9 | 41.4 | 35.9 | 0.0 | **30.9** |
| Wall (All) ↓ | Full-prompt | 99.9s | 67.7s | 68.4s | 162.3s | 85.4s |
| | Naive Premoving | 92.3s | 59.8s | 53.5s | 147.9s | 73.8s |
| | **Premover** | 87.0s | 59.9s | 62.7s | 148.4s | **76.6s (89.7%)** |

- VLA-arena가 더 어려운 환경 (Long Horizon은 모두 0% success).
- Premover는 wall -10.3% 단축 + 2.1pp 성공률 gap (33.0 → 30.9), naive (6pp gap)보다 우월.
- **LIBERO만의 효과가 아님**을 검증하는 critical experiment.

### 4.3 Component Ablation (Table 3)

| Stream | Focus | Ready | Spat | Obj | Goal | L-10 | Avg |
|--------|-------|-------|------|-----|------|------|-----|
| ✓ | | | 68.5 | 64.8 | 57.0 | 75.8 | 66.5 (naive) |
| ✓ | ✓ | | 69.8 | 68.5 | 60.0 | 79.5 | 69.5 (+3.0) |
| ✓ | | ✓ | 84.0 | 94.5 | 91.5 | 83.8 | 88.4 (+21.9) |
| ✓ | ✓ | ✓ | **98.6** | **99.1** | **93.7** | **88.9** | **95.1** (+28.6) |

**관찰**:
- Readiness gate alone → +21.9pp (가장 큰 단일 contributor — *when*이 더 중요).
- Focus map alone (+ stream) → +3.0pp 한정.
- **두 컴포넌트 결합 → 추가 +6.7pp**. 즉 focus map은 readiness gate가 commit한 *후*에 input reweighting으로 backbone attention을 sharpen하는 보조 역할.
- 결론: "**When the policy acts**" matters most; "**what it sees**" matters second.

### 4.4 Floor scale α sweep (Figure 5)

α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} → 모두 ~90% 이상 (broad plateau). 양 극단(α=0 또는 1)이 동일하게 ~90%인 이유:
- α=0: 모든 non-target patch mute → scene context 손실 → readiness gate만 작동.
- α=1: focus map injection 비활성 → readiness-only.

저자 default α=0.2가 calibration set 기준 best — 다만 평가 set에서는 plateau 안에 안전하게 위치.

### 4.5 Top-K sweep (Appendix Table 4)

| K | Success | Wall (s/ep) |
|---|---------|-------------|
| 10 | **88.0** | 43.8 |
| 30 | 84.0 | **43.4** |
| 60 | 86.0 | 43.9 |
| 120 | **88.0** | 44.3 |
| 150 | 74.0 | 49.8 |
| 256 | 0.0 | 78.4 |

K=10이 sweet spot. K>150에서 *급격한 collapse* — top-K mean이 global mean에 가까워져 readiness signal이 소실.

### 4.6 Compute overhead (Appendix F)

| Benchmark | Focus head time | Backbone forward | Overhead per ep |
|-----------|----------------|------------------|-----------------|
| LIBERO | 0.232 ms | 65.84 ms | 0.06s (0.18%) |
| VLA-arena | 0.239 ms | 65.98 ms | 0.11s (0.13%) |

추가 compute < 0.4% — 사실상 free.

---

## 5. 한계 및 미해결 문제

1. **Simulator segmentation mask 의존**. 실제 robot 환경에서는 ground-truth mask가 없으므로 Grounding DINO/SAM 등 weak supervision으로 대체 필요 — 저자는 floor scale ablation의 *plateau*가 mask 품질 변동에 대한 robustness를 시사한다고 주장하나 실험적 검증 부재.

2. **Per-benchmark training**. Projection heads는 benchmark별로 따로 학습 — *single set*로 cross-benchmark zero-shot transfer는 미실험. 즉 새 도메인마다 segmentation-enabled demonstration 수집이 필요.

3. **pi₀.₅ 단독 평가**. OpenVLA, GR00T, RDT 등 다른 backbone에서는 검증되지 않음. Focus map injection이 specific backbone의 attention 구조에 의존하지 않을지 의문.

4. **Linear typing window 가정**. 실제 사용자는 *수정, 일시정지, multi-turn correction* 등 non-monotone behavior를 보임 — 이런 prefix dynamics에서 readiness gate의 동작이 어떨지 미검증.

5. **LongHorizon에서 0% (VLA-arena)**. 4 family 중 1개가 success 0%이라 mean이 inflate된 측면. 진정한 long-horizon에서 premoving이 의미가 있는지는 미해결.

6. **VLA-arena에서 2.1pp success drop**. LIBERO에서는 거의 free lunch (+0.1pp)지만 VLA-arena에서는 trade-off 존재 (33.0 → 30.9). 더 어려운 환경일수록 readiness threshold 결정이 어려워지는 신호.

7. **52.24 WPM 고정 가정**. 사용자별 typing speed variance를 무시 — voice 입력 등 다른 modality는 별도 calibration 필요.

---

## 6. 관련 연구 비교

| Method | Optimization 축 | Backbone | LIBERO Wall (rel.) | LIBERO Success |
|--------|----------------|----------|-------------------|----------------|
| DeeR-VLA | Dynamic early exit | OpenVLA | post-input | - |
| VLA-Pruner | Token pruning | pi₀ | post-input | - |
| TinyVLA | Distillation | smaller | post-input | trade-off |
| FAST | Action tokenization | pi₀ | post-input | - |
| **Premover** | **Pre-input streaming prefix** | **Frozen pi₀.₅** | **86.4%** | **95.1% (≈ baseline 95.0)** |

### 핵심 차별점
- 기존 acceleration 연구가 모두 *post-input inference latency*를 줄이는 반면, Premover는 **pre-input idle window**라는 완전히 다른 축을 탐구.
- 두 축은 *orthogonal* — Premover와 VLA-Pruner를 결합하면 추가 latency 단축 가능 (저자가 명시적으로 future work으로 언급하지는 않음).
- *Backbone freeze + lightweight head*라는 점에서 PVI [Zhang et al.], VAP [Lee et al.]와 유사한 plug-in 철학.

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Streaming prefix를 acceleration 자원으로 활용하는 관점은 새로움. 기존 VLA acceleration의 *blind spot* 명확하게 지목 |
| **Technical depth** | ★★★★☆ — Focus map + Readiness gate가 깔끔하게 분해되어 component ablation으로 each contribution 명확 |
| **Experimental rigor** | ★★★★☆ — 2개 simulator (LIBERO + VLA-arena), 350 rollout × 4 suite, top-K + α sweep까지 충실 |
| **Practical impact** | ★★★★☆ — 13.6% wall-clock 단축, 2.36M parameter, < 0.4% compute overhead — 사실상 free lunch |
| **Writing quality** | ★★★★★ — Figure 1·2가 motivation을 직관적으로 전달, 수식과 ablation이 명확 |

**강점**: VLA latency를 *정책 성능 손실 없이* 줄이는 시스템적 관점 — 모델 자체보다 *상호작용 타이밍*을 재설계. Backbone frozen + 2.36M parameter lightweight 모듈이라 다양한 VLA에 쉽게 부착 가능. Naive premoving (66.4%) vs Premover (95.1%) 비교가 "readiness 학습"의 본질적 기여를 명확히 보여줌. LIBERO와 VLA-arena 두 benchmark에서 일관된 결과 → generalizable.

**약점**: Simulator segmentation 의존이라 real-robot transfer 미검증. Projection head가 benchmark별로 따로 학습되어야 함. Linear typing window 가정 — 실제 사용자의 typing pause/correction은 미고려.

**위상**: VLA 실시간 배포 측면에서 실용적 가치가 명확한 *blind spot fill-in* 연구. 후속으로 (1) weak supervision (Grounding DINO/SAM)으로 mask 의존 제거, (2) cross-benchmark zero-shot transfer, (3) voice/multi-turn modality 확장 등 자연스러운 follow-up이 예상됨.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Naive premoving 66.4%는 어떻게 측정되었나? | 모든 token이 도착할 때까지 기다리지 않고 *매 step* action을 출력. Readiness gate 없이 streaming prefix로 즉시 commit → 잘못된 target에 motion 시작 → 회복 불가. 즉 baseline of how *not* to do premoving |
| 2 | Object suite에서 Premover 99.1% > full-prompt 97.4%인 이유는? | Focus map supervision이 backbone의 dispersed attention을 *명시적으로 sharpen*. Full-prompt에서 backbone이 이미 충분하다고 가정되지만 Figure 4에서 보듯 distractor object에 mass가 일부 leak. Premover의 explicit grounding이 이를 교정 |
| 3 | LongHorizon이 0%인 VLA-arena에서 wall time 측정은 의미가 있나? | LongH는 timeout까지 모두 실패. Wall (All)이 의미 있는 비교는 timeout이 동일 적용되기 때문 (148.4s vs 162.3s). 단 success-conditional wall time(Wall Succ)은 "–"로 보고. 즉 LongH는 wall-clock benefit 측면에서만 의미 있고 success 측면에서는 information 없음 |
| 4 | Focus map을 step t에서 step t+1 input으로 옮기는 디자인은 정합한가? | 인접 timestep attention 유사성 가정 [24]에 근거. 빠르게 변화하는 dynamic scene에서는 lagging이 문제될 수 있으나, LIBERO/VLA-arena 같은 quasi-static manipulation에서는 충분. high-frequency dexterous control에서는 검증 필요 |
| 5 | α = 0.2 default는 LIBERO calibration이고, VLA-arena에 그대로 reuse — 정당한가? | 저자는 별도 tuning 없이 LIBERO α=0.2를 VLA-arena에 reuse. Figure 5의 plateau가 broad하기 때문에 robust. 하지만 새로운 task family에 대해서는 calibration 권장 |
| 6 | K=10과 K=120이 둘 다 88.0% top success — 어떻게 선택? | Wall time이 차이 (43.8 vs 44.3) → K=10이 약간 빠르고 일관성 있음. K=30이 wall은 43.4로 가장 짧으나 success 84.0%로 4pp 하락. K=10이 success-speed sweet spot |
| 7 | Image patches N과 top-K=10 ratio가 너무 작지 않은가? | LIBERO 두 카메라(agent + wrist), 각각 224x224 input ViT patch 14x14 = 256 patches × 2 = 512 patches. K=10/512 = 약 2%만 top-K. 즉 target object가 차지하는 patch 비율과 정합 |
| 8 | Real-robot에서 segmentation mask가 없을 때 어떻게? | 저자가 limitation에서 명시: Grounding DINO나 SAM-style mask predictor로 대체 가능 — Figure 5의 α plateau가 mask 품질 변동에 대한 finite tolerance를 시사한다고 주장. 다만 정량 검증 부재 |
| 9 | 다른 VLA backbone (OpenVLA, GR00T)으로 plug-and-play 가능한가? | pi₀.₅의 hierarchical subtask 예측 구조에 의존하지 않는 design — projection head는 어떤 transformer hidden state에도 부착 가능. 다만 실제 검증은 pi₀.₅ 단독. OpenVLA의 action token 기반 디자인에서 입력 reweighting이 동일하게 작동할지는 미검증 |
| 10 | 학습 비용은 얼마인가? | 8 H200 중 1대로 benchmark별 head set 학습. Demonstration replay (segmentation enabled) 위에서 epoch 한정 — 보고된 wall time 부재이나 backbone frozen + 2.36M parameter라 수 시간 이내 추정 |
| 11 | Cross-benchmark transfer는 시도되지 않았는데, 진정한 plug-and-play로 보기 어렵지 않나? | 맞는 비판. 저자도 future work으로 명시: "single set of heads trained on diverse pool"로 zero-shot transfer 가능 여부는 open question. 현 시점에서는 *backbone* plug-and-play일 뿐, *deployment domain* plug-and-play는 아님 |
| 12 | Voice input 등 다른 modality에서는? | 명시적 미검증. Voice는 typing보다 빠르지만(150 WPM 수준), pause/disfluency가 prefix dynamics를 non-monotone하게 만듦. 저자가 "revisions and pauses introduce non-monotone prefix dynamics; we expect typing pauses themselves to provide an additional commitment signal"로 hint |

<!-- VERIFIED: pdf -->
