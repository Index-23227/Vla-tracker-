# OneStepVLA (Let It Be Simple): One-Step Action Generation for Vision-Language-Action Models

> **한 줄 요약**: VLA의 action 생성은 image generation과 condition-target 구조가 다르기 때문에(rich condition + 저차원 compact target), consistency distillation / shortcut / mean-flow / teacher 같은 복잡한 few-step diffusion machinery 없이도 **표준 flow-matching velocity 학습 + 학습-시간 분포를 high-noise로 단순 편향(Beta(1,1.5) 또는 shifted u/(1+(alpha-1)(1-u)))** 만으로 1-NFE 정책이 가능하다. 1.4B VLM + 30M action head 풀-인코더 모델이 LIBERO-Long에서 **1-step 95.6%**(10-step 95.0%와 동등 이상)를 달성하고, LIBERO-Plus 18개 비교 레시피에서 평균 +5.4점, 실로봇 YAM RSS bimanual에서 1-step이 10-step을 매치/상회.

---

## 1. 배경 및 동기

- 기존 diffusion-based VLA(π0, π0.5, Diffusion Policy, Octo 등)는 image-generation의 iterative denoising 관점을 그대로 차용해 **10-step Euler decoding을 inference의 기본값**으로 삼았다.
- 그러나 image generation은 약한 조건(class label, text)에 고차원 target(이미지)을 매핑하는 반면, VLA action 생성은 **rich condition(이미지+language+proprioceptive state)에 저차원 compact target(action_dim × horizon, 수십~수백 scalar)** 을 매핑한다.
- 저자들의 주장: 이 비대칭 때문에 image generation용 one-step 기법(consistency models, distillation matching distillation, shortcut models, mean-flow, flow-map learning)이 **VLA에서는 불필요**하다. 표준 velocity prediction에 시간 분포만 high-noise로 편향해도 single-step이 강해진다.
- 동기를 isolate하기 위해 **MNIST 4×4 grid-to-sequence**라는 controlled probe를 먼저 설정(rich-condition + compact-target 구조), 거기서 high-noise schedule이 1-step exact match를 크게 개선함을 보인 뒤 LIBERO 계열로 확장.

---

## 2. 방법론 심층 분석

### 2.1 Conditional Flow Matching 기반

표준 flow matching convention:
- linear interpolation `x_t = t·x_1 + (1-t)·x_0`, `t ∈ [0,1]` (x_1 = data, x_0 ~ N(0,I))
- target velocity `v_t = x_1 - x_0` (constant)
- loss `L_CFM = E_{t,x_0,x_1,c} || v_θ(x_t, t, c) - (x_1 - x_0) ||²`
- population optimum: conditional mean velocity `v*(x,t,c) = E[x_1 - x_0 | x_t = x, t, c]`

저자 핵심 관찰: **one-step decoding은 이 conditional target distribution이 concentrated/simple할 때만 신뢰 가능**. 그렇지 않으면 incompatible action들을 평균낸 출력이 나옴.

### 2.2 High-Noise Time Shift (메인 레시피)

Base `u ∈ [0,1]`을 sample한 후 noise shift 적용:

```
t = u / (1 + (alpha - 1)(1 - u))
```

- `alpha > 1` → 샘플을 `t → 0`(노이즈 쪽)으로 이동
- 기본 base distribution: `Beta(1, 1.5)` (이미 약하게 high-noise biased)
- 메인 ablation에서 alpha ∈ {3, 4, 8} 비교

### 2.3 Pure-noise Probe

스트레스 테스트로 `x_t`를 independent Gaussian noise로 대체하고 condition에서 clean action을 직접 예측. "conditional action target이 충분히 단순한가"를 묻는 진단(default recipe는 아님).

### 2.4 VLA Architecture (SimVLA-like)

- **VLM encoder strong, action head small** (SimVLA[31]와 동일 철학)
- OpenPI[π0] backbone: SigLIP(visual) + PaliGemma(multimodal fusion)
- Action decoder: 입력은 VLM tokens + robot state + time embedding + noised action tokens → velocity 출력
- Action interface는 OpenPI 호환 32-dim (LIBERO는 7 physical dim만 supervise; padding은 supervise 안함 = mask7)

두 스케일 평가:
- **Tiny**: Gemma-tiny(4 layers, width 512), SigLIP 4-layer slice → 빠른 ablation용
- **Full encoder**: Gemma-2b(4 used layers, width 2048), 전체 SigLIP-So400m 27 layers → 메인 결과용. 이게 **abstract의 "1.4B VLM + 30M action head"**.

### 2.5 Action Head 세부 (Appendix D, Table 8)

- SimVLA mode, depth 4, width 768, 12 heads, mlp dim 3072
- action dim 32 / loss dim 7 / horizon 10
- pre-LayerNorm Transformer + MLP, condition·state·action-time token concat 후 self-attention

---

## 3. 데이터 전략

- **메인 simulation 학습/평가**: LIBERO 4 suites(Spatial / Object / Goal / Long), 각 suite 10 tasks, eval 50 episodes/task.
- **분포 변화 평가**:
  - LIBERO-Plus[14]: tiny-model로 직접 학습, 2000 episodes/suite로 평가.
  - LIBERO-Pro[15]: full-encoder standard-LIBERO checkpoint(LIBERO-Pro로 추가 학습/선택 없음)를 그대로 perturbation에 적용. 500 episodes/cell.
- **Real robot**: YAM bimanual setup, RSS 2026 OpenPI-baseline expert-data splits[16]에서 π0.5 base checkpoint를 task별로 fine-tune. 5 trials/task.
- **Toy probe**: MNIST 4×4 grid → 16-digit 순서 sequence (continuous diffusion for language modeling[12] 기반).
- **CIFAR-10 class-to-image**: 비교 대조군. condition은 weak(class label), target은 high-dim이라 high-noise pure-noise probe가 효과 없음을 보임(Fig. 2c, Fig. 4d/h).

---

## 4. 시스템 / 학습 세부사항 (Appendix D, Table 9)

| Parameter | Tiny ablation | Full encoder |
|---|---|---|
| Optimizer | AdamW | AdamW |
| β1, β2, ε | 0.9, 0.95, 1e-8 | 0.9, 0.95, 1e-8 |
| Weight decay / grad clip | 1e-10 / 1.0 | 1e-10 / 1.0 |
| LR schedule | cosine decay | cosine decay |
| Warmup / decay steps | 1k / 50k | 1k / 150k |
| Peak / final LR | 1e-4 / 1e-5 | 2e-4 / 2e-5 |
| Backbone LR multiplier | 0.1 | 0.1 |
| EMA decay | 0.99 | 0.99 |
| Batch size / steps | 64 / 50k | 256 / 150k |

- Base time distribution: `Beta(1, 1.5)` (구현 내부 좌표에서는 `top ~ Beta(1.5, 1)` = "1=noise" 코어디네이션에서의 high-noise biased)
- Hardware: Shanghai Innovation Institute GPU 시간(구체 GPU 모델 명시 없음)
- Replanning interval = trained action horizon이 기본; flow steps와 분리 보고(Appendix E에서 r-sensitivity 별도 분석)

---

## 5. 실험 설계 및 평가 프로토콜

- **One-step vs ten-step Euler**: 학습된 동일 checkpoint에서 inference flow step 수만 변경(1 vs 10) → "1-step이 실제로 10-step과 동등한가"라는 controlled comparison.
- **Schedule sweep**: Uniform / α=3 / α=4 / α=8 / Pure-noise 5 가지. Pure-noise는 multi-step Euler 정의 안되므로 1-step만.
- **Action horizon sweep**: H10/H20/H30/H40 (LIBERO-Long 기준).
- **Condition ablation**: image / prompt / wrist / state 한 channel씩 제거.
- **Action loss masking**: mask7 (7 physical dim만 loss) vs full32 (32 dim 전부 loss).
- **Replanning interval r-sensitivity**: closed-loop success에 큰 영향, flow steps와 독립적으로 보고.
- **Velocity-field diagnostics**: 학습된 velocity field의 MSE와 cosine error를 noise level τ에 따라 측정 (τ=1이 noise endpoint).

---

## 6. 실험 결과 심층 분석

### 6.1 표준 LIBERO Tiny-model Schedule Sweep (Table 1, H10)

| Schedule | Steps | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| Uniform | 1 | 88.8 | 92.8 | 90.2 | 70.2 |
| Uniform | 10 | 96.6 | 96.2 | 93.2 | 80.8 |
| α=3 | 1 | 95.8 | 97.0 | 94.4 | 78.0 |
| α=3 | 10 | 91.6 | 96.2 | 82.2 | 52.4 |
| α=4 | 1 | **96.4** | **99.6** | **96.8** | **85.2** |
| α=4 | 10 | 93.4 | 96.8 | 91.4 | 63.4 |
| α=8 | 1 | 95.6 | 99.0 | 94.0 | 78.0 |
| α=8 | 10 | 78.4 | 40.0 | 56.6 | 22.6 |
| Pure-noise | 1 | 96.0 | 97.8 | 96.2 | 72.8 |

**핵심 관찰**:
- α=4 1-step이 Uniform 10-step(96.6/96.2/93.2/80.8)을 모든 suite에서 상회.
- α=8은 너무 공격적: 1-step은 OK지만 10-step이 무너짐 (특히 Object 40.0%, Long 22.6%).
- Pure-noise 1-step도 strong, 진단 도구로만 유지.

### 6.2 Full-encoder 스케일 (Table 4)

| Target | Steps | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| mask7 | 1 | 97.4 | 98.4 | 97.8 | 92.8 |
| mask7 | 10 | 95.8 | 99.2 | 98.4 | 87.6 |
| **full32** | **1** | **98.4** | **100.0** | **97.0** | **95.6** |
| full32 | 10 | 98.8 | 99.0 | 98.0 | 95.0 |

**Abstract의 95.6% LIBERO-Long 1-step**이 여기서 나옴 (full32, full-encoder, 1.4B VLM + 30M head). full32 1-step이 10-step과 통계적으로 동등(95.6 vs 95.0).

### 6.3 LIBERO-Plus Distribution Shift (Fig. 5)

- 모든 비교 가능한(non-pure) 18개 recipe가 1-step vs 10-step diagonal **위 또는 위쪽**에 위치.
- **1-step 평균 마진 +5.4점** vs 10-step.
- Horizon별 (H10/H20/H30/H40): 1-step success가 81 / 74 / 61 / 57 (Uniform), 86 / 83 / 84 / 84 (α=4), 89 / 81 / 84 / 75 (α=8) → α=4가 모든 horizon에서 안정적.

### 6.4 LIBERO-Pro Robustness Probe (Table 5)

Full-encoder standard-LIBERO checkpoint(LIBERO-Long 92.8%)를 LIBERO-Pro에 zero-shot으로:
- 1-step mean 44.2%, 10-step mean 43.5% (16개 cell 중 14개가 ±5점 이내)
- 즉 표준 LIBERO에서 학습된 sampler-step trend가 harder probe에서도 유지

### 6.5 Real Robot YAM RSS (Table 6)

π0.5 base checkpoint를 task별로 fine-tune 후 inference flow step만 변경:

| Task | 1-step | 10-step |
|---|---|---|
| Insert mouse battery | 80% | 80% |
| Seal water bottle cap | **60%** | 35% |
| Tower of Hanoi game | **100%** | 50% |

5 trials small-sample이지만, **모든 task에서 1-step ≥ 10-step**. 다른 architecture(π0.5는 SimVLA-like가 아닌 cross-attention action expert)에서도 sampler trend 일관.

### 6.6 Velocity-Field Diagnostics (Fig. 4)

- LIBERO tiny/full 및 π0.5 RSS 모두 **noise endpoint(τ=1)로 갈수록 MSE와 cosine error 감소**.
- 대조군 CIFAR-10 class-to-image flow: error가 interpolation의 **중간**에서 최저.
- 즉 VLA의 학습된 velocity field가 noise 근처에서 가장 잘 calibrated → one-step inference가 잘 동작하는 메커니즘적 근거.

---

## 7. Ablation 분석

### 7.1 Action Horizon (Table 2, LIBERO-Long)

| Horizon | Schedule | 1-step | 10-step |
|---|---|---|---|
| H20 | Uniform | 64.4 | 77.4 |
| H20 | α=4 | 76.8 | 26.6 |
| H20 | α=8 | 75.0 | 32.8 |
| H30 | Uniform | 57.2 | 66.0 |
| H30 | α=4 | 61.8 | 43.0 |
| H30 | α=8 | 65.8 | 25.0 |
| H40 | Uniform | 31.6 | 53.4 |
| H40 | α=4 | 44.6 | 31.8 |
| H40 | α=8 | 45.2 | 18.4 |

- High-noise schedule이 H20/H30에서 1-step 손실을 크게 회복.
- H40에서는 gap이 다시 벌어짐 → 너무 긴 chunk는 single endpoint prediction으로 collapse 어려움.
- 즉 high-noise shifting은 **universal solution이 아니라 boundary evidence**.

### 7.2 Condition Ablation (Table 3, H10 α=4 1-step)

| Removed input | Object | Spatial | Goal | Long |
|---|---|---|---|---|
| No image | 82.4 | 62.8 | 53.4 | 31.6 |
| No prompt | 96.8 | 81.2 | 11.0 | 56.6 |
| No wrist | 95.2 | 68.4 | 78.8 | 52.6 |
| **No state** | **0.0** | 0.2 | 0.4 | **0.0** |

→ Proprioceptive state 제거가 가장 치명적. 거의 정책 붕괴. 다른 조건도 1-step success를 크게 떨어뜨려 "rich condition" 전제를 직접 검증.

### 7.3 Action Loss Masking (Table 4 + Table 10)

| Recipe | mask7 fs=1 | full32 fs=1 | mask7 fs=10 | full32 fs=10 |
|---|---|---|---|---|
| Uniform | 70.2 | 59.8 | 80.8 | 70.0 |
| α=3 | 78.0 | 58.8 | 52.4 | 55.8 |

Tiny LIBERO-Long에서 mask7 vs full32 차이가 크지만, **full-encoder scale에서는 차이가 사라짐**(Table 4 full32 1-step이 오히려 95.6%로 최고). 저자는 tiny ablation에서는 conceptual cleanness를 위해 mask7 유지, 메인 claim으로는 사용하지 않음.

### 7.4 Direct Learned Action-Start (Table 11)

Noise 대신 학습 가능한 token을 시작점으로 사용 (regression objective):
- Zero init: 92.8 / 99.0 / 59.4 / 68.2 (Spatial / Object / Goal / Long)
- Normal init (std=1): 95.4 / 98.6 / 59.8 / 69.4

→ Viable하지만 high-noise flow matching(Table 1 α=4: 96.4 / 99.6 / 96.8 / 85.2)보다는 분명히 낮음. **Noised action token이 단순한 1-step regression보다 더 좋은 training structure 제공**한다는 증거.

### 7.5 Toy Diagnostics (MNIST grid-to-sequence)

- **CNN encoder scaling**: 강한 visual feature가 ceiling을 올림.
- **Weak CNN + larger decoder**: decoder scale로 weak condition을 보상 불가.
- **CIFAR-10 class-to-image**: pure-noise training 효과 매우 약함 → "condition이 weak하고 target이 high-dim이면 one-step이 어렵다" 명시적 대조.
- AR baseline: 4-layer d=256은 25% exact-match, d=512 8-layer로 키워야 diffusion model과 비슷.

---

## 8. 관련 연구 비교

| 방향 | 예시 | OneStepVLA의 차별점 |
|---|---|---|
| Consistency models / DMD | Song'23, Yin'24 | Teacher / distillation stage / auxiliary loss 필요. OneStepVLA는 **표준 FM loss만** 사용 |
| Shortcut / mean-flow / flow-map | Frans'25, Geng'25, Boffi'25 | 새로운 objective와 training stage 추가. OneStepVLA는 추가 stage 0 |
| Consistency distillation for robotic policies | Song'23류를 diffusion policy에 적용한 [37, 38] | Teacher 정책 필요. OneStepVLA는 teacher 불필요 |
| SimVLA [31] | Lightweight action head with multi-step diffusion 유지 | **같은 lightweight head 직관 공유 + 1-step decoding으로 확장** |
| 동시기 SnapFlow (2604.05656) | Progressive FM/consistency mixing + zero-init target-time embedding | OneStepVLA는 **distillation 자체를 우회**하고 학습-시간 분포 시프트만 사용 — 훨씬 가벼움 |
| π0 / π0.5 / Octo (flow/diffusion VLA) | 10-step Euler를 기본값으로 |  동일 architecture(π0.5)에 fine-tune만 다르게 해도 1-step이 통함을 실로봇으로 보임 |
| Rectified flow / SD3 high-noise shift [39,40,41] | Image generation에서 time distribution 조정 사용 | 이미지에서는 one-step 충분 안함. **VLA에서는 condition-target 구조 덕에 충분** |

**핵심 메시지**: "complex few-step diffusion machinery를 도입하기 전에 VLA의 condition-target 구조를 먼저 활용하라."

---

## 9. 한계 및 미해결 문제

저자 명시 한계(Sec. 5) + 리뷰어 관점:

1. **이론적 설명이 intuitive 수준**: CIFAR-10에서는 noise endpoint에서 conditional variance가 큰데 VLA diagnostic은 noise endpoint로 갈수록 error가 줄어듦. 정확한 정량적 이유는 future work.
2. **Optimal alpha 선택 규칙 부재**: horizon, condition set, replanning interval에 따라 최적 alpha가 다르지만 a priori 선택 방법이 없음. H10에서는 α=4가 좋고 H40에서는 gap이 다시 벌어짐.
3. **새로운 method 이름 없음**: 논문이 명시적 method/model 이름을 제시하지 않음(레시피·관찰 페이퍼). 본 트래커에서는 편의상 "OneStepVLA"로 명명.
4. **Code release 정보 없음**: paper_url만 있고 code_url 미공개 (open_source: false).
5. **Real-robot evaluation 샘플 크기 작음**: 5 trials/task. 단 cross-architecture(π0.5)에서 sampler trend 일관성은 의미있는 신호.
6. **Larger backbone(>1.4B) / 더 다양한 benchmark(CALVIN, SimplerEnv, RoboCasa, RoboTwin, RLBench) 미평가**: VLA-Tracker 다른 모델들과 직접 비교 가능한 score는 LIBERO에 한정됨.
7. **mask7 vs full32 일관성 부족**: tiny에서는 mask7이 명확히 우세, full-encoder에서는 full32가 더 좋음. 저자도 명확히 설명 못함.
8. **α=8 ten-step collapse**: 동일 schedule이 1-step에는 도움이 되지만 10-step에 매우 해로움. 한 모델로 multi-step과 one-step을 동시에 잘하기 어려움 → deploy mode를 학습 시점에 결정해야 함.

---

## 10. 총평

| 항목 | 평가 |
|---|---|
| **Novelty (개념)** | ★★★★☆ — Method 자체는 매우 단순(time distribution shift)하지만 "VLA는 condition-target asymmetric이므로 image-gen one-step 기법이 불필요"라는 framing이 새롭고 reproducible |
| **Novelty (기술)** | ★★☆☆☆ — High-noise shift 자체는 SD3 등에서 이미 사용. 진짜 기여는 controlled study + velocity-field diagnostic |
| **Empirical rigor** | ★★★★★ — Toy probe → tiny ablation → full-encoder validation → distribution-shift(LIBERO-Plus) → robustness(LIBERO-Pro) → real robot(YAM RSS) → velocity-field diagnostic까지 6단계 검증. 통제변수 정리가 깔끔 |
| **Practical impact** | ★★★★★ — 10× inference 속도 향상을 teacher / distillation / architecture change 0개로 달성. π0.5 같은 기존 checkpoint에 fine-tune 단계에서 schedule만 바꾸면 됨 |
| **Reproducibility** | ★★★☆☆ — Hyperparameter 표(Appendix D) 상세하지만 code release 없음 |

**핵심 메시지**: "VLA action generation의 어려움은 image generation의 어려움이 아니다. Condition이 rich하고 target이 compact한 정책에서는 표준 flow matching에 high-noise time bias만 줘도 single-step이 strong해진다. 복잡한 few-step diffusion machinery를 들고 오기 전에 먼저 이 condition-target 구조를 인정하라."

특히 Table 4 full-encoder full32에서 1-step 95.6% vs 10-step 95.0%(LIBERO-Long), 그리고 LIBERO-Plus에서 18 비교 recipe 평균 +5.4점 마진은 "1-step이 단순히 cheaper alternative가 아니라 standard recipe 안에서 이미 우위"임을 명확히 보여준다.

---

## 11. 예상 날카로운 질문

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | High-noise schedule이 1-step을 돕고 10-step을 망치는 이유는? | α가 클수록 학습 분포가 noise endpoint로 쏠림 → velocity field가 t≈1 근처에서 잘 calibrated, t 중간 영역은 under-trained. Multi-step Euler는 중간 영역을 거치므로 미세 오차 누적(Table 1 α=8 10-step Object 40%, Long 22.6%). 이는 deploy mode를 학습 시점에 commit해야 한다는 의미. |
| 2 | 왜 image generation에서는 high-noise shift만으로 one-step이 안되는데 VLA에서는 되는가? | Condition-target asymmetry. Class label → 1024×1024 RGB는 conditional distribution이 multimodal/high-dim이라 single mean prediction이 평균화 collapse. VLA는 (image+prompt+state) → 70-dim action chunk라 conditional distribution이 narrow/unimodal. Fig. 2c CIFAR-10 pure-noise probe와 Fig. 4d/h CIFAR-10 vs LIBERO velocity-field 대비가 이를 직접 증명. |
| 3 | Pure-noise probe가 메인 레시피가 아닌 이유? | Pure-noise는 multi-step Euler decoding 자체가 정의 안됨(interpolation trajectory 없음). 1-step diagnostic으로만 유효. 또한 Table 11의 learned action-start regression이 보여주듯 noised action token이 단순 regression보다 더 좋은 training structure 제공. |
| 4 | LIBERO-Long full32 1-step 95.6%가 정말 SOTA인가? | SnapFlow가 동일 task에서 1-step 97.0% 보고했고 MPCoT는 98.9%(10-step). OneStepVLA는 absolute SOTA가 아니라 **distillation 없이도 1-step이 강함**을 보이는 controlled study. 가치는 SOTA가 아니라 "method 단순성 vs 성능" Pareto에서. |
| 5 | Action horizon H40에서 high-noise가 다시 무너지는 이유? | H40은 28~40-dim target chunk. Compact-target 전제가 약해짐. Conditional action distribution이 multimodal해질 가능성 ↑. Boundary evidence이지 universal claim 아님. |
| 6 | No state ablation이 0%인 이유? Image+language로는 action 생성 불가능한가? | LIBERO setup에서 proprioceptive state가 절대 좌표/현재 grasp state를 제공. 없으면 어디서부터 어디로 움직일지 ambiguous. 이건 "condition이 진짜 rich해야 한다"는 메인 가설을 직접 검증하는 데이터. |
| 7 | mask7 vs full32 차이가 tiny와 full-encoder에서 뒤집히는 이유? | 추측: tiny model은 capacity 부족해서 padded coordinate에 capacity 낭비하면 7-dim 정확도 손해. Full-encoder는 capacity 여유가 있어 padded coordinate 학습이 오히려 regularization 역할. 저자도 명확한 답 없이 mask7을 tiny에서만 사용. |
| 8 | YAM RSS real-robot에서 1-step이 10-step을 큰 차이로 이기는 이유(60% vs 35%, 100% vs 50%)? | 5 trials small-sample 통계 변동성 고려해야 함. 그러나 모든 task에서 1-step ≥ 10-step인 것은 sampler trend가 일관됨을 시사. π0.5는 cross-attention action expert(SimVLA-like와 다른 architecture)인데도 condition-target view가 holding. |
| 9 | High-noise shift는 SD3/rectified flow에서 이미 사용했는데 contribution 있나? | 그쪽에서는 high-noise shift가 one-step image generation을 만들지 못함. 본 논문의 contribution은 "동일한 light intervention이 VLA에서는 충분하다"를 controlled study로 증명한 것. 즉 **technique 발명이 아니라 applicability claim**. |
| 10 | OpenPI 32-dim action interface를 그대로 두고 mask7만 하는 게 fair한가? | full32 supervision(Table 4)이 오히려 더 좋은 결과 → 32-dim interface 자체가 손해는 아님. 다만 cross-robot compatibility 위한 padding이지 모델 inductive bias는 아님. |

---

## 12. 결론 및 향후 연구 방향

- **이 논문의 본질적 기여**: VLA action generation을 "rich-condition / compact-target" 관점에서 재정의함으로써, image-generation 분야에서 개발된 무거운 few-step diffusion machinery(consistency, distillation, shortcut, mean-flow, flow-map)가 VLA에는 **불필요**할 수 있음을 controlled study + real-robot로 입증.
- **즉시 실용 가치**: 기존 π0 / π0.5 / SimVLA / OpenPI 류 checkpoint를 fine-tune할 때 time sampling을 `Beta(1, 1.5)` + `t = u / (1 + (α-1)(1-u))` with α=3 또는 4로 바꾸면 추가 비용 0으로 inference 10×. 코드 변경 minimal.
- **향후 연구**:
  - Optimal α를 horizon × condition richness × replanning interval로 예측하는 scaling law
  - Long-horizon(H40+)에서도 1-step이 무너지지 않는 architecture (예: hierarchical action chunking)
  - Multi-modal action distribution을 정량화하고 그 multimodality에 condition-encoder capacity를 매핑하는 framework
  - CALVIN / SimplerEnv / RoboCasa / RoboTwin / RLBench로의 transfer (현재는 LIBERO 계열 + YAM RSS만)
  - Velocity-field diagnostic을 학습 중에 monitor해서 자동으로 schedule 조정하는 adaptive recipe

**가장 인상적인 결과**: Full-encoder + full32 + 1-step Euler에서 LIBERO-Long 95.6%(10-step 95.0%와 동등)와, LIBERO-Plus 18개 비교 레시피 평균 +5.4점 1-step 마진. 이는 "1-step이 10-step의 빠른 대체재가 아니라 표준 recipe 안에서 이미 우위"임을 보여주는 강한 증거다.

<!-- VERIFIED: pdf -->
