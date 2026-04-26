# DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA

> **한 줄 요약**: VLM(System-2)이 **자신의 ViT feature space 안에서** H-step 뒤의 visual foresight를 예측하고, flow-matching policy(System-1)가 latent inverse dynamics로 이를 풀어 action을 생성하는 differentiable bottleneck 구조. RoboCasa GR1 Tabletop 24-task 평균 **70.2%** (full data)로 FLARE 55.0%, GR00T-N1.6 47.6% 대비 큰 격차로 SOTA; few-shot(10%) 58.3%로 full-data FLARE보다도 우수.

---

## 1. 배경 및 동기

- 기존 end-to-end VLA는 VLM을 주로 "대형 multimodal encoder"로 사용하여 feature → action으로 직결, 훈련이 불안정하고 semantic representation이 붕괴하기 쉬움(Fig. 2 middle).
- Hierarchical planner는 text/pixel로 중간 표현을 생성하지만 non-differentiable, 고지연, **action gradient가 VLM을 refine하지 못함**.
- FLARE, SEER 같은 world modeling objectives가 제안되었으나 "latent foresight를 policy가 참고해도 되고 안 해도 되는" loose coupling — shortcut learning 여지가 남음.
- DIAL의 질문: "VLM의 intent에 정책을 **구조적으로** 묶으면서 end-to-end 학습이 가능한 아키텍처는?"

---

## 2. 방법론

### System-2: Latent World Modeling (Sec. 3.2)
Pretrained Qwen2.5-VL-3B 사용. 입력 sequence 끝에 N개의 learnable query token을 추가하고, LLM 출력을 MLP projection으로 latent intent x_t ∈ R^(N×d)로 변환. x_t는 **같은 ViT encoder**로 추출한 미래 관찰 o_{t+H}의 feature와 MSE로 정렬(Eq. 2): `L_world = ||x_t - Enc_ViT(o_{t+H})||^2`. H=16. N은 단일 observation의 patch 개수에 맞춤.

### System-1: Flow-Matching Latent Inverse Dynamics (Sec. 3.3)
4-layer self-attention으로 현재 visual feature Enc_ViT(o_t)와 latent foresight x_t를 융합 → 16-layer DiT에 cross-attention conditioning. Proprioceptive state q_t는 MLP projection 후 DiT input sequence에 concat. Flow matching loss(Eq. 3): `L_fm = E[||V_θ(A_t^τ | x_t, Enc_ViT(o_t), q_t, τ) - (A_t - ε)||^2]`.

### 2-Stage Training (Sec. 3.4)
- **Stage 1 (Decoupled Warmup)**: System-2는 L_world만, System-1은 ground-truth future feature Enc_ViT(o_{t+H})를 주고 L_fm만. 두 모듈이 서로 간섭 없이 각자의 역할을 습득.
- **Stage 2 (End-to-End)**: System-1의 condition을 x_t(System-2 예측)로 교체. 총 손실 `L_total = L_world + L_fm`이며 action gradient가 x_t를 통해 VLM까지 역전파.
- ViT encoder와 text embedding은 두 stage 모두 동결; 나머지 LLM 블록/쿼리/MLP는 학습.

---

## 3. 실험 결과

### RoboCasa GR1 Tabletop — Full Data (Fig. 7, 24 tasks × 50 rollouts, 1000 traj/task)

| Method | Pick & Place | Articulated | 24-GR Avg |
|---|---|---|---|
| Diffusion Policy | 30.1 | - | ~30 |
| UWM | 58.2 | 40.3 | 40.9 |
| FLARE | 55.4 | 50.4 | 55.0 |
| GR00T-N1.6 | 47.0 | 38.4 | 47.6 |
| GR00T-Qwen3 | 44.4 | 50.1 | 47.8 |
| π-Qwen3 | 40.4 | 51.3 | 43.9 |
| OFT-Qwen3 | 50.3 | 42.3 | 48.8 |
| FAST-Qwen3 | 35.0 | 43.7 | 39.0 |
| **DIAL** | **68.9** | **74.3** | **70.2** |

YAML의 robocasa_gr1_avg: 70.2와 일치(Figure 7).

### Few-shot (Fig. 8, 100 traj/task)
DIAL **58.3%** (24-task avg), full-data FLARE(55.0%)보다도 높음. 10× 데이터 절감.
Ablation: +SEER 49.6, +SEER-EV 47.2, +FLARE 51.9, DIAL-DINO 47.2 → 모두 DIAL 58.3 미달. **Extra-vision path(SEER-EV)가 오히려 성능 저하** → latent foresight를 구조적으로 강제하지 않으면 policy가 무시한다는 증거.

### Human Data Scaling (Fig. 9)
EgoDex basic_pick_place(27k traj) 추가 시 in-distribution Pick&Place 56.0 → 60.8, OOD 평균 46.2 → 51.2. Articulated task는 EgoDex 범위 밖이라 62.0 → 65.3로 개선 없음(오히려 미세 감소).

### Real-world IRON-R01-1.11 (Fig. 10, 11)
In-distribution avg **77.5%** (GR00T-Qwen2.5 50, FLARE 57.5). OOD avg **58.3%** (GR00T-Qwen2.5 6.7, DIAL w/o warmup 30.0). Decoupled warmup 제거 시 77.5 → 57.5, OOD 58.3 → 30.0으로 **급락 — warmup이 training stability의 핵심**.

---

## 4. 한계 및 미해결 문제

1. **Horizon H=16 고정**: 더 긴 horizon task나 multi-stage subgoal에서 단일 future feature로 충분한지 불명. RoboCasa tabletop 규모에서만 검증.
2. **ViT feature space에 종속**: DIAL-DINO 실험(58.3 → 47.2, Sec. 5.2)은 reasoning과 control이 동일 latent manifold를 공유해야 함을 보였지만, VLM의 ViT가 feature로 충분히 풍부하지 않은 도메인에서는 제약.
3. **Articulated task에서 human data 효과 부재**: EgoDex가 pure pick-and-place만 포함하기 때문이지만, 저자들의 "human priors로 scale한다"는 주장에 한계.
4. **Warmup ratio 스케줄**: 저자들은 50/50으로 두었지만 task별 최적 비율에 대한 연구는 부재.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "future ViT feature를 그 자체로 latent intent bottleneck으로 쓴다"는 설계는 FLARE/SEER보다 엄격한 구조적 constraint를 제공 |
| **Practical impact** | ★★★★★ — 10× data efficiency, 실제 humanoid에서 OOD 58.3%, SEER-EV 실패 분석으로 후속 연구 방향 명확 |

DIAL의 진짜 통찰은 "VLM을 encoder가 아니라 decision maker로 쓰려면 System-1이 System-2의 예측을 **무시할 수 없어야** 한다"는 점이다. FLARE처럼 foresight를 auxiliary loss로 달거나 SEER처럼 concat하면 policy가 shortcut으로 빠지는 것을 반복 ablation으로 입증했다. Decoupled warmup은 단순하지만 실전에서 training 붕괴를 막는 결정적 스케줄이며, real-robot OOD에서의 30% → 58.3% 차이가 이를 웅변한다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Latent inverse dynamics"가 왜 단순 concat(SEER)보다 강제력이 더 강한가? | System-1의 DiT가 x_t와 Enc_ViT(o_t)를 **양쪽 다** 받아야 action을 생성할 수 있게 설계 — concat은 intent를 무시해도 되지만 DIAL은 "현재 vs 미래"의 차이를 해결해야만 action이 나옴. Fig. 8: SEER-EV 47.2 < DIAL 58.3. |
| 2 | Warmup stage에서 System-1에 ground-truth future feature를 줘도 되는가(cheating)? | 이는 "완벽한 intent가 주어졌을 때 motor control 학습"이 목적 — inference 시에는 여전히 x_t를 사용. Stage 2에서 x_t ≈ Enc_ViT(o_{t+H})로 수렴해 seamless한 전환이 가능. Warmup 제거 시 실전 OOD 58.3 → 30.0. |
| 3 | ViT를 동결하는데도 VLM이 action-aware representation을 획득할 수 있는가? | ViT는 동결하지만 **LLM blocks, queries, MLP**는 학습 가능. Action gradient가 x_t → MLP → LLM을 통해 backprop되어 LLM이 "action에 도움이 되는 방향으로 미래를 상상"하게 학습. PCA 시각화(Fig. 12)가 이를 뒷받침. |

<!-- VERIFIED: pdf -->
