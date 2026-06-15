# VLGA: Vision-Language-Geometry-Action Models for Autonomous Driving

> **한 줄 요약**: Uber AV Labs + UVA의 자율주행 VLA. 기존 3-expert MoT(UniDriveVLA, U/P/A)에 **dedicated geometry expert(G)**를 네 번째 modality로 추가하고, DVGT-2 backbone에서 나온 per-patch geometry token에 대해 **dense per-pixel pointmap 재구성 손실(L_pmap, LiDAR ground truth)**을 부여. Pi3-style confidence-weighted L1로 ego-frame 3D point를 예측하도록 학습. nuScenes(no-ego) L2-ST-P3 평균 **0.50m**, 3초 collision **0.18%** (VLA 최저), Bench2Drive closed-loop Driving Score **79.08 (SOTA)**, UniDriveVLA(78.37) 대비 **+0.71**.

---

## 1. 배경 및 동기

- 자율주행 VLA의 핵심 한계: language reasoning은 강하나 **dense 3D world에 대한 spatial grounding이 약함**. Trajectory planning은 본질적으로 spatial task인데, 언어 추론만으로는 안전 주행에 필요한 연속 공간 정밀도를 제공하지 못함(Sec. 1).
- 기존 3D-aware 패러다임(Fig. 1) 세 가지 모두 결함:
  - **(a) Sparse perception VLA** (UniDriveVLA, Orion, OmniDrive): 3D box / lane / occupancy 같은 **discrete query 출력**만 사용 → dense spatial signal 부재.
  - **(b) Injection-based VLA** (VGGDrive): 3D foundation model feature를 LLM hidden state에 cross-attention / BEV token / 3D Q-Former로 주입 → **same LLM parameter가 language와 3D를 모두 처리**, geometry용 전용 capacity 없음, 주입된 geometry가 실제로 쓰이도록 강제하는 objective도 없음.
  - **(c) Geometry-only policy** (Vision-Geometry-Action, DVGT-2): per-pixel pointmap reconstruction은 하지만 **language stream을 제거**.
- 세 능력(language reasoning / dense spatial grounding / dedicated geometry capacity)을 모두 갖춘 정책이 없다는 것이 문제 정의(Fig. 1d). VLGA는 이를 모두 만족하는 첫 모델로 제안됨.

---

## 2. 핵심 기여

1. **Vision-Language-Geometry-Action(VLGA)** 프레임워크 제안: MoT(Mixture-of-Transformers)에 **dedicated, parameter-isolated geometry expert G**를 네 번째 modality로 추가하여 U/P/G/A 4-expert 아키텍처를 구성(Fig. 2).
2. **Dense per-pixel pointmap 재구성 손실 L_pmap** 도입: LiDAR sweep을 누적·투영한 ground-truth pointmap에 대해 ego-frame에서 5-layer transformer decoder D가 (x̂_p, c_p)를 회귀, Pi3-style confidence-weighted L1으로 학습.
3. **Two-stage training schedule**: (i) geometry warm-up — action expert/모든 inherited stream freeze, L_pmap unit weight로 G·projector·D만 학습; (ii) joint stage — action expert unfreeze, L_act + 0.1·L_pmap.
4. **추론 시 LiDAR 불필요**: pointmap decoder D는 학습 신호용으로만 쓰이고 inference에서는 discard(Sec. 3.4).
5. **실증 결과**: nuScenes no-ego에서 16개 planning metric 중 15개 SOTA(VLA 기준), Bench2Drive에서 Driving Score 79.08로 종합 SOTA. Safety-critical metric(collision, Give Way, Emergency Brake)에 gain이 집중됨을 정성·정량적으로 입증.

---

## 3. 방법론

### 3.1 문제 정의 (Sec. 3.1)
- 입력: 6-camera multi-view I ∈ R^(6×H×W×3), ego-status s(linear velocity, acceleration, one-hot driving command), high-level navigation instruction ℓ.
- 출력: 3-second 미래 ego trajectory T = {(x_t, y_t)}_{t=1}^{T_f} via π(I, s, ℓ).
- π는 MoT 기반 VLA로 구현, 본 논문은 여기에 G(geometry) expert를 추가.

### 3.2 Four-Expert MoT 아키텍처 (Sec. 3.2, Fig. 2)
- 구성요소:
  - **Understanding expert (U)**: Qwen3-VL 언어/장면 의미 처리 (frozen, UniDriveVLA init).
  - **Perception expert (P)**: agent/map/occupancy query 처리 (frozen).
  - **Geometry expert (G)**: dense spatial structure 전용 (trainable).
  - **Action expert (A)**: flow-matching trajectory 생성 (trainable, U/P/G에 attend + ego-status 조건화).
- **Masked joint attention**: 각 expert가 자신의 Q/K/V projection을 갖고 visibility mask M 아래서 concatenate해 attend. UniDriveVLA의 U/P/A 패턴 유지 + G token은 U/P에 attend, A는 G에도 추가로 attend.
- Ego-status 조건화: P의 self-prediction head가 s를 예측. 추론 시 `s = s_gt`(with-ego)와 `s = ŝ`(without-ego) 두 protocol을 동일 가중치로 평가.

### 3.3 Geometry Expert (Sec. 3.3)
- DVGT-2[59]의 geometry backbone을 그대로 사용(frozen). 6-camera 960×544 입력에서 카메라당 60×34=2,040 patch feature → 총 **6×2,040 = 12,240 geometry token**.
- Pooling 없이 per-patch grid 유지 (dense reconstruction에 per-pixel resolution 필요).
- Per-patch projector f_proj: R^d_g → R^d_MoT가 각 geometry token을 MoT token space로 매핑: g_i = f_proj(f_i^geo).
- 이 token들이 G expert의 입력으로 들어가 masked joint attention에 참여.

### 3.4 Dense Pointmap Supervision (Sec. 3.4)
- Lightweight 5-layer transformer decoder D({g_i})_p → (x̂_p, c_p): patch p마다 ego-frame 3D point x̂_p ∈ R^3 + uncertainty logit c_p.
- 손실(Pi3-style confidence-weighted, Eq. 4):
  - L_pmap = (1/|P|) · Σ_{p∈P} [ ‖x̂_p − x_p^gt‖_1 / b_p + log b_p ], b_p = softplus(c_p).
- Valid set P: ground-truth depth가 [0.5, 80.0] m 범위인 patch만 포함.
- Ground-truth pointmap: LiDAR sweep을 누적 후 각 카메라에 투영해 생성.
- **추론 시 D는 사용하지 않음** → deployment에서 LiDAR 불필요.

### 3.5 Two-Stage Training Schedule (Sec. 3.5)
- **Geometry stage** (warm-up): action expert + inherited stream freeze, geometry components(G expert, projector, D)만 학습. L_geom = L_pmap.
- **Joint stage**: action expert unfreeze, L_joint = L_act + λ_pmap · L_pmap, **λ_pmap = 0.1**.
- Random-init geometry component가 action expert의 학습 궤적과 간섭하지 않도록 분리 설계.

### 3.6 구현 세부 (Sec. 4.1)
- **변형 두 종류**:
  - VLGA-Base: Qwen3-VL-2B
  - VLGA-Large: Qwen3-VL-8B
- Geometry backbone: DVGT-2.
- 입력 해상도: 960×544, 6 camera.
- **Frozen**: vision-language backbone (UniDriveVLA init), perception expert, DVGT-2.
- **Trainable**: G expert, per-patch projector, A expert, pointmap decoder D.
- Optimizer: AdamW, LR 5e-5, effective batch size 128, EMA (momentum 2e-4, warmup 2000), 8×H100.
- Epoch: 10/3 (geometry stage on nuScenes/Bench2Drive), 30/7 (joint stage).

---

## 4. 실험 결과

### 4.1 nuScenes Open-Loop Planning — No Ego-Status (Table 1)

저자들은 **with-ego 프로토콜은 kinematic extrapolation 만으로도 강한 L2가 나오는 leakage 문제(Li et al. 2024)**가 있다고 명시. 핵심 평가는 no-ego setting.

| Method | L2 ST-P3 Avg ↓ | Coll. ST-P3 Avg ↓ | L2 UniAD Avg ↓ | Coll. UniAD Avg ↓ | LLM |
|---|---|---|---|---|---|
| UniAD | 0.73 | 0.61 | 1.03 | 0.77 | — |
| FSDrive | 0.53 | 0.17 | 0.96 | 0.40 | Qwen2-VL-3B |
| SparseDrive‡ | 0.55 | 0.08 | 0.99 | 0.21 | — |
| UniDriveVLA-Large | 0.51 | 0.11 | 0.90 | 0.27 | Qwen3-VL-8B |
| **VLGA-Base** | 0.52 | 0.14 | 0.95 | 0.35 | Qwen3-VL-2B |
| **VLGA-Large** | **0.50** | **0.09** | 0.90 | **0.22** | Qwen3-VL-8B |

- VLGA-Large가 **ST-P3 L2 avg 0.50m로 전체 최저**, **3초 collision 0.18%**(no-ego, ST-P3)로 모든 VLA 중 가장 낮음.
- 16개 metric 중 **15개에서 VLA SOTA**(UniAD L2 3s에서만 1.52 vs 1.50으로 근소 차).
- VLGA-Base도 UniDriveVLA-Base(2B) 대비 동일 scale에서 collision 0.41→0.35로 개선 → **gain이 scale이 아니라 geometry stream에서 옴**을 시사.
- DVGT-2(0.41 L2, with-ego)와 VGGDrive(0.31 L2, with-ego)가 mean L2는 더 낮으나 **3초 collision rate은 VLGA가 2.5–3× 낮음** — safety-critical 측면에서 우위.

### 4.2 Bench2Drive Closed-Loop (Table 2)

| Method | Avg L2 ↓ | Driving Score ↑ | Success Rate ↑ | Efficiency ↑ | Comfortness ↑ |
|---|---|---|---|---|---|
| Orion | 0.68 | 77.74 | 54.62 | 151.48 | 17.38 |
| UniDriveVLA | 0.72 | 78.37 | 51.82 | 198.86 | 11.78 |
| **VLGA** | **0.69** | **79.08** | 52.73 | 194.63 | **13.06** |

- Driving Score **79.08**로 모든 베이스라인 대비 신규 SOTA, UniDriveVLA 대비 **+0.71**.
- Success Rate 51.82→52.73, Comfortness 11.78→13.06로 동시 개선, Efficiency는 198.86→194.63로 거의 동등.

### 4.3 Per-Skill Closed-Loop SR (Table 3)

| Method | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|---|---|---|---|---|---|---|
| Orion | 25.00 | **71.11** | **78.33** | 33.00 | **69.15** | **54.72** |
| UniDriveVLA | **38.75** | **80.00** | 50.00 | 30.00 | 58.95 | 51.53 |
| **VLGA** | **38.75** | 77.78 | 55.00 | **40.00** | 53.68 | 53.04 |

- VLGA가 가장 크게 개선한 항목: **Give Way (30→40)**, **Emergency Brake (50→55)** — 둘 다 "오프셋·제동 거리를 정밀하게 맞춰야" 하는 dense-spatial 시나리오. 저자들의 가설("geometry stream이 spatial-precision 시나리오에서 도움")과 일치.
- Overtaking(80→77.78), Traffic Sign(58.95→53.68)은 약간 후퇴 — reactive control과 시각 의미가 병목인 항목.

### 4.4 Ablation (Table 4, nuScenes ST-P3, no-ego)

| Configuration | L2 avg ↓ | Coll. avg ↓ |
|---|---|---|
| Baseline (no geometry stream) | 0.539 | 0.169% |
| + Geometry expert | 0.529 | 0.149% |
| + Pointmap aux supervision | **0.524** | **0.136%** |

- Geometry expert 추가만으로는 modest gain (0.169→0.149%), **dense pointmap 감독을 더하면 추가로 collision 8.7% 상대 감소**.
- 핵심 메시지: "geometry stream만으론 부족, **dense reconstruction objective가 stream을 task-relevant하게 만든다**".

### 4.5 Qualitative (Fig. 3)
- nuScenes val에서 front camera에 GT(green) + pred(yellow) 3초 trajectory를 투영. UniDriveVLA는 회전/근접 차량 주변에서 lateral drift, VLGA는 GT에 더 밀착.
- 두 정책의 차이는 supervised geometry stream 뿐 → spatial grounding 개선의 직접 증거.

---

## 5. 한계 및 미해결 문제

1. **추론 비용**: 대형 VLM backbone(2B/8B) + DVGT-2 + 4-expert MoT → edge 배포에 부담. 저자도 "distillation, quantization이 자연스러운 방향"으로 명시.
2. **Per-frame supervision**: 현재 L_pmap은 프레임별. **multi-frame temporal consistency** 확장이 long-horizon geometric reasoning에 유익할 수 있다고 limitation에서 인정.
3. **Training-time LiDAR 의존**: 추론에는 불필요하지만 **학습에는 LiDAR-equipped fleet 필요**. LiDAR 없는 데이터셋(예: Waymo Open Motion w/o LiDAR, camera-only fleet)에는 그대로 적용 불가.
4. **Bench2Drive 의존**: closed-loop 평가가 CARLA simulator의 220 route에 국한, 실제 도로 검증 부재.
5. **Per-skill 일부 후퇴**: Overtaking, Traffic Sign에서 UniDriveVLA 대비 약간 감소 — geometry stream이 모든 시나리오에서 universally 이득은 아님.
6. **코드 공개 상태 불명확**: project page(https://yaojin17.github.io/VLGA)만 제공, GitHub 미공개로 보임 (open_source: false로 기재).

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA + dense geometry supervision을 "parameter-isolated 4번째 expert + auxiliary pointmap loss"로 깔끔하게 통합한 첫 사례. Injection(VGGDrive)도 geometry-only(VGA/DVGT-2)도 아닌 hybrid가 핵심. |
| **Practical impact** | ★★★★☆ — Uber AV Labs가 직접 발표, Bench2Drive SOTA + nuScenes no-ego SOTA + safety-critical metric 집중 개선. 추론 시 LiDAR 불필요는 deployment 친화적. |
| **Reproducibility** | ★★★☆☆ — 모든 하이퍼파라미터(epoch, LR, batch, EMA)와 frozen/trainable 구조가 명시되어 있어 재구현 가능. 단 weight/code 미공개. |

핵심 메시지: **"VLA driving policy에 dense 3D geometry를 넣는 올바른 방법은 (i) 전용 parameter budget을 분리해 주고, (ii) action loss에만 의존하지 말고 dense per-pixel pointmap reconstruction으로 직접 감독하는 것이다."** Geometry expert만으로는 한계가 있고(Table 4: 0.169→0.149%), **dense supervision까지 결합해야 safety-critical metric에서 의미 있는 gain**(0.136%)이 나타남.

---

## 7. 핵심 수식

**Pi3-style confidence-weighted per-patch L1 (Eq. 4):**

L_pmap = (1/|P|) · Σ_{p∈P} [ ‖x̂_p − x_p^gt‖_1 / b_p + log b_p ],  b_p = softplus(c_p)

- x̂_p: ego-frame 3D 예측 (R^3).
- c_p: per-patch uncertainty logit. softplus로 양수 scale b_p 변환.
- log b_p 항이 over-confident regularizer 역할 (Kendall & Gal, 2017).
- P: depth ∈ [0.5, 80.0] m 인 patch만.

**Two-stage objective (Eq. 5, 6):**

- Geometry stage: L_geom = L_pmap
- Joint stage: L_joint = L_act + λ_pmap · L_pmap, λ_pmap = 0.1

---

## 8. 다른 패러다임과의 비교

| 차원 | (a) Sparse perception VLA | (b) Injection VLA | (c) Geometry-only | **(d) VLGA** |
|---|---|---|---|---|
| Language reasoning | O | O | X | **O** |
| Dense spatial grounding | X | O (feature 주입) | O | **O** |
| Dedicated geometry capacity | X | X (LLM이 겸용) | O | **O** |
| Geometry용 dense supervision | X (sparse box/map만) | X (action loss 의존) | O (pointmap) | **O (pointmap)** |
| 대표 모델 | UniDriveVLA, Orion, OmniDrive | VGGDrive | VGA, DVGT-2 | **VLGA** |

VLGA의 본질적 차이: **(c)의 dense pointmap supervision을 (a)/(b)의 language reasoning과 결합**하되 LLM 파라미터를 오염시키지 않는 MoT 분리 구조.

---

## 9. UniDriveVLA와의 직접 비교

VLGA는 UniDriveVLA를 base 정책으로 그대로 사용(U/P expert + VLM 모두 frozen, UniDriveVLA init).

| 항목 | UniDriveVLA-Large | VLGA-Large | 차이 |
|---|---|---|---|
| Experts | U/P/A (3) | U/P/G/A (4) | G expert 추가 |
| Geometry signal | sparse (det/map/occ) | sparse + dense pointmap | dense layer 추가 |
| Bench2Drive DS | 78.37 | **79.08** | +0.71 |
| Bench2Drive Comfortness | 11.78 | **13.06** | +1.28 |
| nuScenes no-ego ST-P3 L2 avg | 0.51 | **0.50** | −0.01 |
| nuScenes no-ego ST-P3 Coll. 3s | 0.21 | **0.18** | −0.03 (rel. ~14%) |
| nuScenes no-ego UniAD Coll. avg | 0.27 | **0.22** | −0.05 (rel. ~19%) |
| Give Way SR | 30.00 | **40.00** | +10.0 |

→ 모든 변동의 원인은 **VLGA가 추가한 geometry stream 단 하나**이므로 ablation로서 가장 강한 근거가 됨.

---

## 10. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Geometry expert를 별도로 두지 않고 DVGT-2 feature를 곧장 LLM에 cross-attention(injection) 하면 안 되는가? | (b) injection 패러다임의 한계. VGGDrive 같은 방식은 LLM parameter가 language와 3D를 동시 처리해야 하므로 geometry용 전용 capacity가 없고, "feature가 실제 쓰이도록 강제하는 objective"도 없음. Table 4 baseline(geometry stream 없음) 대비 G expert만 추가해도 collision 0.169→0.149%로 개선되는 게 분리 capacity의 가치. |
| 2 | Dense pointmap supervision이 자동운전 stage에서 정말 필요한가? Box/lane/occupancy 같은 sparse 3D supervision은 이미 있는데? | Table 4가 직접 답: geometry expert만 추가해도 sparse perception은 그대로 있는 상태. 그럼에도 dense pointmap을 더하면 collision이 추가로 0.149→0.136%(8.7% 상대 감소). Sparse box list로는 "주차차량과 lateral clearance 유지", "마주오는 차의 swept volume 예측" 같은 continuous geometric 추론이 불가능하다는 게 저자 주장(Sec. 2). |
| 3 | 추론에 LiDAR이 필요 없다면, dense supervision의 효과가 실제로 어디에 남아 있는가? | Geometry expert의 weight + per-patch projector가 학습 후에도 유지됨. Pointmap decoder D만 discard. 즉, "DVGT-2 feature를 ego-frame 3D coordinate로 정렬·정제하는 능력"이 G expert 파라미터에 internalize된 채 inference 때 trajectory 예측에 활용됨. Fig. 3의 lateral drift 감소가 그 정성적 증거. |
| 4 | Two-stage가 정말 필요한가? Joint training만 하면 안 되나? | 저자 주장(Sec. 3.5): random-init geometry component가 inherited action expert의 optimization trajectory를 교란할 수 있어, 먼저 L_pmap만으로 warm-up 후 action loss와 결합. λ_pmap=0.1로 joint stage에서 약화하는 것도 같은 맥락(action 최적화를 우선시). 단, 단일-stage ablation은 논문에 명시되지 않아 정량 비교는 부재. |
| 5 | Comfortness가 UniDriveVLA 11.78→VLGA 13.06으로 개선됐는데, 왜? | dense geometry로 obstacle clearance/lateral offset이 정확해지면 급조향·급제동 빈도가 감소. Per-skill에서 Give Way(+10), Emergency Brake(+5)가 같이 개선된 패턴이 정합적. 그러나 여전히 Orion(17.38)에는 못 미침. |
| 6 | nuScenes no-ego L2 avg 0.50m는 VGGDrive 0.31m(with-ego)나 OpenDriveVLA 0.33m(with-ego)보다 높지 않은가? | 비교 setting이 다름. with-ego는 ego-status leakage(Li et al., 2024)로 kinematic extrapolation만으로 강한 L2가 나오므로, 저자는 no-ego만 공정한 비교라고 명시. VLGA의 with-ego L2(0.41 avg)도 경쟁권. 더 중요한 것은 collision rate: VLGA-Large의 3s collision 0.18%는 VGGDrive(0.55%) 대비 ~3× 낮음. |

---

## 11. 향후 연구 방향

1. **Temporal consistency for L_pmap**: 현재 per-frame. 다중 frame window에 걸쳐 pointmap을 일관되게 예측하도록 cross-frame consistency loss를 추가하면 long-horizon planning에 도움.
2. **Camera-only training**: LiDAR-free fleet 데이터에 적용하려면 self-supervised depth/pointmap(monodepth, MASt3R 등) pseudo-label로 L_pmap을 대체하는 연구가 필요.
3. **Distillation/quantization**: 8B + 4-expert MoT의 edge 배포를 위해 expert pruning, sparse routing, weight quantization 탐색.
4. **G expert size scaling**: 본 논문은 G expert 크기를 변화시키는 ablation 부재. Parameter budget이 어디까지 saturate되는지 흥미로운 질문.
5. **Open-world/safety-critical 일반화**: nuScenes/Bench2Drive 외에 OOD 도시, 악천후, dense pedestrian crowds 같은 시나리오에서의 dense geometry stream 효과 검증.
6. **Action head 변형**: 현재 flow matching. Diffusion / autoregressive action head와의 호환성과 trade-off는 미탐.

---

## 12. 결론

VLGA는 자율주행 VLA에 **dense 3D geometry를 들여오는 "올바른 방식"**을 제안한다. 핵심은 두 가지: (1) **parameter-isolated geometry expert**로 LLM 파라미터를 오염시키지 않으면서 dedicated capacity 확보, (2) **dense per-pixel pointmap 재구성 손실**로 geometry stream이 실제로 task-relevant 3D 정보를 인코딩하도록 강제. 결과는 Bench2Drive Driving Score 79.08(신규 SOTA), nuScenes no-ego 3s collision 0.18%(VLA 최저)로 명확하며, ablation은 두 컴포넌트가 monotonically 기여함을 보인다. Safety-critical metric(collision, Give Way, Emergency Brake)에 gain이 집중된다는 점은 "dense geometry는 mean accuracy가 아니라 tail risk를 줄이는 데 가장 큰 효과를 낸다"는 직관과도 일치한다. 추론 시 LiDAR 불필요라는 deployment 친화성까지 갖춘, **VLA 자율주행 라인의 차세대 baseline**으로 보아도 무방하다.

<!-- VERIFIED: pdf -->
