# ActionMap: Robot Policy Learning via Voxel Action Heatmap

> **한 줄 요약**: OpenVLA-OFT(7B)와 π0.5(flow matching) 두 backbone의 액션 디코더를, 7-DoF 델타 액션 공간 위에 정의된 voxel heatmap 분포(translation 48x48x24, rotation 24x24x24, gripper 2-bin)로 교체하고 Gaussian-blob cross-entropy + top-10 soft-argmax로 학습/추론한 결과, LIBERO 4-suite 평균 OpenVLA-OFT 대비 **+8.2%p (89.1→97.3)**, π0.5 대비 **+1.6%p (96.9→98.5)**를 동일 학습 step에서 달성한 voxel-heatmap action head 연구.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 모델은 backbone, training recipe, data scale 면에서 빠르게 발전했지만, **action decoder는 거의 변하지 않은 채 거의 모든 시스템에서 single-point predictor**로 남아 있음
- 세 가지 주요 디코더 패러다임 모두 single-point 출력: (i) autoregressive token bins(OpenVLA), (ii) L1 regression(OpenVLA-OFT), (iii) flow-matching denoising(π0, π0.5, GR00T)
- 결과: 학습 시 spatial structure 활용 불가, 이웃 액션에 대한 soft probabilistic supervision 없음, multimodal demonstration에 대한 differentiable handle 부재

### 핵심 질문
- **End-effector 액션 공간의 풍부한 기하적 구조(이웃 액션끼리 비슷한 물리적 의미)를 명시적 분포 출력으로 활용하면 large pretrained VLA에서도 성능이 향상되는가?**
- **Pose estimation에서 검증된 "regression → heatmap" 전환이 VLA 액션 예측에도 유효한가?**

📌 [Figure 1] — 기존 paradigm은 7-DoF 액션을 연속 공간의 단일 점으로 예측. ActionMap은 voxel heatmap에서 확률 질량을 추출해 액션 복원.

---

## 2. 방법론 심층 분석

### 2.1 Problem Formulation
- 7-D 액션 joint distribution을 단일 voxel grid로 표현하면 intractable → **세 개의 factored 분포로 분해**:
  - Translation: Pr(x, y, z) over 3-D 그리드
  - Rotation: Pr(φ, θ, ψ) over Euler 각도 3-D 그리드
  - Gripper: binary Pr(g)
- 좌표는 **delta-action(per-step velocity)** 기반으로 layout

### 2.2 Voxel Heatmap Action Head Architecture
- **순수 MLP + residual** trunk → 세 개의 독립 MLP 분기 출력
- Backbone의 마지막 hidden state에서 action token 위치의 표현을 읽어 logits 생성
  - Translation: Nx × Ny × Nz (default 48×48×24)
  - Rotation: Nφ × Nθ × Nψ (default 24×24×24)
  - Gripper: 2 bins
- **T-step chunk 지원**: T개 action-token 위치에서 동시에 T개 (translation, rotation, gripper) triplet을 한 번에 출력 — OFT의 parallel decoding 구조와 호환

> ❓ **예상 질문**: 7-D를 factorize하면 차원 간 상호의존성을 잃지 않는가?
> **답변**: 손실이 일부 있을 수 있으나 backbone hidden state h가 세 분기에 공통 입력으로 들어가므로 조건부 의존성은 backbone에서 흡수된다. 표현 가능한 joint는 product-of-marginals로 한정되지만 LIBERO/Franka에서는 실용적으로 큰 손실이 관찰되지 않음.

### 2.3 Training: Gaussian-Blob Cross-Entropy
- 각 GT 액션 component a*_c를 voxel grid 위 softmax-normalized Gaussian blob으로 변환:
  q_σ(b; a*_c) ∝ exp(−‖b − a*_c‖² / 2σ²)
- Loss: 세 분기에 대한 soft-label cross-entropy의 합, T chunk slot 모두에 걸쳐 합산
- **σ = 0.1** (normalized [-1, 1] 단위)
- Gripper는 degenerate case로 one-hot target

### 2.4 Inference: Top-k Soft-Argmax
- Hard argmax는 grid 점에 묶이고 인접 voxel 경쟁 시 불안정
- **Top-k soft-argmax (k=10, T=1.0)**가 default
  â = Σ_{b ∈ TopK(z)} softmax(z/T) · b
- 위 두 hyperparameter는 Section 4.3.1에서 11개 대안과 비교 ablation

> ❓ **예상 질문**: 그리드 해상도가 작으면 정밀도가 떨어지지 않나?
> **답변**: soft-argmax가 sub-voxel 정밀도를 제공한다. Pose estimation의 integral regression(Sun et al., ECCV 2018)에서 검증된 기법이며, Franka Pick에서 4.8 mm grasp 정확도로 OFT의 16.7 mm 대비 ~3.5배 정밀하다는 게 직접 증거.

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO**: 4 suite (Spatial, Object, Goal, Long), 10 tasks × full demos
- **Data efficiency 연구**: LIBERO-Spatial의 stratified 10/25/50% subset
- **Real-world Franka R3**: Pick (225 demos), Sweep (200), Insert (275) + 각 task의 50-episode partial 변형

### 데이터 사용 패턴
- OpenVLA-OFT recipe 그대로 사용, action head만 교체
- π0.5의 경우 공개 LIBERO recipe 그대로 + flow-matching expert만 교체
- Backbone, attention, parallel decoding 등 다른 컴포넌트는 모두 보존

> ❓ **예상 질문**: OXE같은 대규모 pre-training data 없이 fine-tuning만으로 97%를 달성하면 LIBERO에 과적합 아닌가?
> **답변**: LIBERO 자체 fine-tuning에서의 결과로 baseline OFT(같은 step) 89.1%, 풀 budget OFT(50-150K step) 97.1%와 비교한 수치. Real-world Franka에서도 일관된 향상(20/30 vs 7/30)을 보여 LIBERO-specific overfit이라 보기 어렵다. 다만 OXE 대규모 cross-embodiment generalization 검증은 없음.

---

## 4. 시스템/학습 세부사항

| 항목 | OpenVLA-OFT 설정 | π0.5 설정 |
|------|------------------|-----------|
| Backbone | Prismatic-7B VLM + parallel decoding | Flow-matching VLM |
| Fine-tuning | LoRA r=32, α=16, 모든 linear layers | 공개 LIBERO recipe |
| Optimizer | AdamW, lr 5×10⁻⁴ constant, no warmup, no WD | AdamW, cosine 5×10⁻⁵ → 3×10⁻⁵ |
| Batch | 효과 64 (per-GPU 8 × 4 grad accum) | 256 (LIBERO) / 64 (Franka) |
| Hardware | 2 × NVIDIA H200 DDP | 8 × H200 (LIBERO) / 4 × H200 (Franka) |
| Training steps | 10,000 (LIBERO), 4,000 (Franka) | 30,000 (LIBERO), 20,000 (Franka) |
| Translation grid | 48×48×24 (LIBERO), 48×48×48 (Franka) | 64×64×64 |
| Rotation grid | 24×24×24 (LIBERO), 16×16×16 (Franka) | 48×48×48 |
| Gaussian σ | 0.1 | 0.15 |
| Decoder | top-10 soft-argmax, T=1.0 | top-10 soft-argmax |

Franka 설정에서는 각 액션 axis의 1st/99th quantile로 [q01-2σ, q99+2σ] 범위를 padding한 뒤 [0,1] grid 좌표로 정규화. 학습 데이터 분포 적응.

---

## 5. 실험 설계 및 평가 프로토콜

평가는 두 축:
1. **LIBERO simulation** — OpenVLA-OFT, π0.5 두 backbone에 ActionMap 헤드를 drop-in. seed 7, 10 tasks × 50 trials = 500 episodes/suite
2. **Real-world Franka Research 3** — 3개 task (Pick, Sweep, Insert), 각 10 trial (5 seen + 5 unseen 위치), 두 데이터 크기 (50 episodes vs full)

각 backbone에 대해 matched-step 비교(OFT 10K, π0.5 30K)와 published full-budget(50K-150K step) 비교 모두 수행.

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO 메인 결과 (Figure 3)

#### OpenVLA-OFT backbone (10K step matched)
| Suite | OFT 10K (baseline) | OFT 50-150K (published) | **OFT + Ours 10K** | Δ vs 10K |
|-------|-------------------|------------------------|-------------------|----------|
| Spatial | 95.2 | 97.6 | **98.4** (best 98.8) | +3.2 |
| Object | 97.8 | 98.4 | **99.4** (best 99.6) | +1.6 |
| Goal | 96.0 | 97.9 | **96.0** (best 97.6) | +0.0 |
| Long | 67.2 | 94.5 | **93.8** (best 96.0) | **+26.6** |
| **Avg** | **89.1** | 97.1 | **97.1** (best 98.2) | **+8.0** |

- **LIBERO-Long에서 +26.6%p**: long-horizon에서 distributional supervision의 효과가 압도적
- Default 디코더로도 10K step만에 OFT의 50-150K step 결과를 거의 따라잡고, best 디코더면 +1.1%p 초과

#### π0.5 backbone (30K step)
| Suite | π0.5 30K (published) | **π0.5 + Ours 30K** | Δ |
|-------|----------------------|---------------------|---|
| Spatial | 98.0 | **98.6** | +0.6 |
| Object | 98.4 | **99.2** | +0.8 |
| Goal | 96.6 | **98.2** | +1.6 |
| Long | 90.0 | **94.8** | **+4.8** |
| **Avg** | **96.9** | **98.5** | **+1.6** |

- 10K step만 학습한 π0.5 + Ours가 이미 30K step published π0.5를 +1.1%p 능가 → **수렴 속도 우위**

### 6.2 Real-World Franka (Figure 4, 5)

#### Full data (Pick 225 / Sweep 200 / Insert 275 demos)
| Task | OFT | OFT + Ours | Δ |
|------|-----|-----------|---|
| Pick | 4/10 | **8/10** | +4 |
| Sweep | 2/10 | **7/10** | +5 |
| Insert | 1/10 | **5/10** | +4 |
| **Pooled** | **7/30** | **20/30** | **+13 (~3배)** |

#### Partial data (50 episodes per task)
- Pick: 1/10 → 4/10 (+3)
- Sweep: 3/10 → 10/10 (+7)
- Insert: 0/10 → 0/10 (Insert는 50 demos로는 양쪽 모두 불가능)
- **Pooled**: 4/30 → 14/30

#### Pick task의 grasp 정확도 (Figure 5)
| 데이터 | OFT (mm) | OFT + Ours (mm) | 향상 |
|--------|---------|-----------------|------|
| Partial (50 ep) | 35.5 ± 6.1 | **15.0 ± 5.5** | 2.4배 |
| Full (225 ep) | 16.7 ± 5.2 | **4.8 ± 5.8** | **3.5배** |

> ❓ **예상 질문**: Insert가 50-episode에서 양쪽 모두 0/10인 이유?
> **답변**: Insert는 "카드를 슬롯 위 정렬 → 수직 하강" 2단계 행동이라 multimodal demonstration 구조가 필요한데, 50 demos로는 양쪽 헤드 모두 학습 부족. 데이터 부족에 의한 fundamental limit이지 action head 문제는 아님(Appendix D).

### 6.3 Data Efficiency (Figure 6)

LIBERO-Spatial, OpenVLA-OFT backbone:

| Data | OFT | OFT + Ours | Δ |
|------|-----|-----------|---|
| 10% (43 demos) | 67.2 | **93.2** | **+26.0** |
| 25% | 82.8 | **93.0** | +10.2 |
| 50% | 82.4 | **97.2** | +14.8 |
| Full | 95.2 | **98.4** | +3.2 |

- **10% data에서 가장 큰 격차** → distributional supervision이 demonstration당 더 많은 신호 추출
- π0.5에서는 모두 92%+ 이상으로 격차가 작지만 일관된 우위(Figure 6b)

### 6.4 Convergence (Figure 7)
- 네 가지 설정 (OFT 100%/10%, π0.5 100%/10%) 중 **세 설정에서 ActionMap이 더 빠르게 plateau**, 나머지 하나(OFT 100%)는 비슷
- 가장 극적인 차이: OFT 10% data → L1 loss는 10K step 내내 감소, ActionMap loss는 ~2K step만에 saturate
- ActionMap의 loss curve가 OpenVLA-OFT 대비 훨씬 smooth → 안정적 학습

---

## 7. Ablation 분석

### 7.1 디코더 전략 (Section 4.3.1, Figure 8)

11개 디코더 후보를 같은 checkpoint에서 비교:
- soft-argmax (full grid, T=0.1/0.3/0.5/1.0), soft-argmax (top-100, top-1000), 균등 평균 (top-10/100/1000), hard argmax
- **4-suite 평균 스프레드: < 1%p**, suite별 최대 3.2%p (LIBERO-Long이 가장 sensitive)
- Default top-10, T=1.0이 안정적 선택

> ❓ **예상 질문**: hard argmax도 96.7%로 거의 차이 없는데 soft가 정말 필요한가?
> **답변**: average 차이는 작아도 LIBERO-Long(장기 horizon)에서 hard 93.8 vs top-10 soft 95.6의 차이는 의미 있고, Franka grasp 정확도(4.8mm)는 sub-voxel 보간이 핵심. Discrete grid의 한계를 soft-argmax가 보완.

### 7.2 Grid Resolution × Gaussian σ (Section 4.3.2, Figure 9)

3개 grid 해상도 × 4개 σ(0.05, 0.10, 0.15, 0.20) = 12 cell 전체에서 6.4%p 밴드 내에 분포:

| Trans/Rot grid | σ=0.05 | σ=0.10 | σ=0.15 | σ=0.20 |
|----------------|--------|--------|--------|--------|
| 32×32×16 / 16³ | 93.3 | 92.1 | 93.6 | 92.0 |
| **48×48×24 / 24³** (headline) | 96.2 | **97.3** | 95.9 | 94.7 |
| 64×64×48 / 48³ | 94.2 | 94.2 | 90.9 | (n/a) |

- Default 설정(중간 해상도, σ=0.10)이 **global optimum**
- 더 거친/세밀한 grid 모두 underperform → 그리드 폴리노미얼 스케일링의 implicit trade-off

### 7.3 Cross-Backbone 일반성

ActionMap은 **architecturally distinct한 두 backbone**에서 일관된 향상:
- OpenVLA-OFT (L1 regression, parallel decoding) → +8.2%p
- π0.5 (flow matching) → +1.6%p
- 즉 design이 특정 action paradigm에 매이지 않음 (논문 contribution 4)

---

## 8. 관련 연구 비교

| 방법 | Action 표현 | 적용 대상 | LIBERO Avg |
|------|------------|----------|-----------|
| OpenVLA | Discrete tokens (autoregressive) | OpenVLA backbone | ~76 |
| OpenVLA-OFT | L1 regression (continuous) | Prismatic-7B | 97.1 |
| π0/π0.5 | Flow matching | π0.5 | 96.9 |
| GR00T N1 | Flow matching | Humanoid | — |
| PerAct/RVT/RVT-2 | One-hot voxel / 2D heatmap | Small 전용 backbone (Perceiver-IO, multi-view CNN) | — (LIBERO 미평가) |
| BridgeVLA | 3D observation-side | VLM | — |
| **ActionMap** | **Voxel heatmap (factored, soft)** | **OpenVLA-OFT + π0.5** | **97.3 / 98.5** |

### 핵심 차이
- **Large pretrained VLA에 heatmap action head를 통합한 첫 사례** — PerAct/RVT는 specialized small backbone에 묶여 있었음
- Pose estimation의 "regression → heatmap" 패러다임 시프트를 robotic action에 적용한 점이 영감 원천(Sun et al. integral regression, ECCV 2018)
- **Drop-in replacement** — backbone/recipe/attention/parallel decoding 변경 없이 마지막 head만 교체

---

## 9. 한계 및 미해결 문제

### 저자가 인정한 한계
1. **Polynomial scaling**: voxel grid 파라미터가 per-axis 해상도의 다항식으로 증가 → 실용적 grid 세분화에 상한. **adaptive grid resolution**이 자연스러운 향후 방향.
2. **Delta-action 한정**: absolute coordinate grid는 미탐색
3. **Single Gaussian-blob width**: task/training stage별 adaptive σ 가능성 미탐색
4. **Per-frame argmax**: temporal/multimodal sampling이 distribution 출력의 잠재력을 완전히 활용 못함

### 추가 비판 포인트
1. **Cross-embodiment generalization 검증 없음**: LIBERO + Franka R3에 한정. OXE 같은 대규모 cross-embodiment 사전학습 시 ActionMap의 효과는 미검증
2. **Action factorization의 정합성**: translation/rotation/gripper 독립 분포로 분해 — multimodal demonstration에서 조건부 의존성을 잃을 가능성. 명시적 비교 없음.
3. **π0.5 에서의 향상폭(+1.6%)이 OpenVLA-OFT(+8.2%)보다 작음**: flow matching이 이미 implicit distributional 표현을 학습한다고 해석 가능. 그렇다면 ActionMap의 강점은 **L1 regression처럼 spatial structure가 약한 헤드에서만 두드러질** 가능성.
4. **Insert task fail (50 demos에서 양쪽 0/10)**: 진정한 multimodal trajectory 학습 능력은 미입증. ActionMap의 "soft probability mass" 주장은 단봉(unimodal) Gaussian-blob target에 묶여 있어 multimodal demonstration 자체를 capture하지 못함 — 이론적 약점.
5. **Hardware/cost**: 2x H200으로 10K step 학습은 reasonable하지만 grid 파라미터 폭증(64³=262K voxels) 시 메모리 cost가 상승. 정량 보고 부재.
6. **Insert 성공(5/10 vs 1/10) 대 grasp 정확도**: full data Pick에서 4.8mm grasp는 강력하나, Insert는 정확도 외 **2-stage planning**이 필요. ActionMap이 spatial precision 이상의 추론 향상을 제공한다는 증거는 부족.

### Attribution 의문
- 향상이 **action representation** 덕인지, **추가 MLP 파라미터** 덕인지 명확치 않음. "matched training step"은 controlled하지만 파라미터 수 균형은 정량 비교 없음 (heatmap head MLP는 L1 regression head보다 더 클 가능성).

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Large pretrained VLA에 voxel heatmap head를 통합한 첫 사례. Pose estimation의 잘 알려진 패러다임을 VLA로 옮긴 깔끔한 아이디어. 완전히 새로운 idea는 아니나 적용처가 신선. |
| **Technical depth** | ★★★★☆ — Factored grid 설계, Gaussian-blob target, top-k soft-argmax, σ-grid resolution interaction까지 ablation 체계적. |
| **Experimental rigor** | ★★★★☆ — 두 backbone × 두 평가 환경 (sim + real) × 데이터 효율성 × 디코더 ablation × resolution ablation. Drop-in 비교 통제 양호. |
| **Practical impact** | ★★★★☆ — Drop-in replacement, 단 10K step만에 50-150K full-budget을 능가 — 학습 비용 대폭 절감. Real-world grasp 정확도 3배 향상은 실제 시스템에 직접 영향. |
| **Writing quality** | ★★★★☆ — Pose estimation 유비를 명확히 정립, ablation 표/그림이 잘 정돈됨. |

**강점**: Action representation을 backbone/data와 독립된 design axis로 확립한 점. 두 architecturally distinct backbone에서 일관된 효과 → 이론적 주장 뒷받침. Real-world grasp localization 4.8 mm는 실용적 임팩트의 명확한 증거.

**약점**: Multimodal demonstration capture는 단봉 Gaussian-blob target으로 제한됨. Cross-embodiment generalization 미검증. π0.5에서의 향상폭이 작아 effect의 scope에 의문. Polynomial grid scaling이 본질적 병목.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Pose estimation에서 빌려온 아이디어인데 robot action에서의 진짜 novelty는? | Pose은 2-D/3-D 단일 joint, robot action은 7-D factored + chunked. T-step의 parallel decoding과의 통합, delta-action 위 layout, top-k soft-argmax 디코더 선택 등 적응 작업이 비자명. 또한 large pretrained VLA에 통합된 첫 사례. |
| 2 | π0.5에서 향상폭(+1.6%)이 작은데 ActionMap의 효과가 backbone 의존적인가? | Flow matching이 이미 implicit distributional 표현을 학습. L1 regression처럼 spatial structure 부재한 헤드에서 ActionMap의 효과가 두드러짐. Action representation의 "lever"가 backbone마다 다른 크기로 작용한다는 해석. |
| 3 | Insert 50-episode에서 0/10인 이유와 이를 multimodal demonstration 한계로 봐도 되는가? | Insert는 2-stage 행동(align→insert)을 50 demos로는 학습 불가. 양 헤드 모두 fail이라 ActionMap-specific 결함은 아니나, 단봉 Gaussian-blob이 multimodal trajectory 분포를 capture 못한다는 우려는 별개 문제로 남음. |
| 4 | Grid resolution을 더 키우면 (예: 128³)? | Polynomial scaling으로 head 파라미터 폭증. 논문에서 64×64×48 vs 48×48×24 비교 시 더 큰 grid가 underperform — saturation point 존재. Adaptive grid가 future work. |
| 5 | Translation/rotation 분리 factorization은 둘 사이 의존성을 잃지 않나? | 이론적 우려. 다만 backbone hidden state h가 공통 입력이라 조건부 의존성은 backbone에서 흡수. LIBERO/Franka에서 실용적 손실 관찰 안 됨. Multimodal task에서는 미검증. |
| 6 | LIBERO-Long에서 +26.6%p가 비정상적으로 크다 — 평가 protocol 차이 아닌가? | 같은 seed 7, 같은 10K step, 같은 backbone. 차이는 action head뿐. Long-horizon에서 정밀도가 오차 누적을 줄여 success rate에 비선형 영향 — 합리적. |
| 7 | OXE 같은 대규모 cross-embodiment 학습에서도 유효한가? | 미검증. 다른 embodiment에서는 action 공간의 scale/structure가 달라 grid layout 재조정 필요. 저자의 quantile-based padding이 부분 해결책. |
| 8 | Top-10 soft-argmax가 정말 top-1보다 의미 있는 향상인가? | 4-suite 평균 차이는 < 1%p지만 LIBERO-Long(95.6 vs 93.8) 및 Franka grasp 정확도(4.8 vs ~16mm 추정)에서 의미. Sub-voxel 보간이 핵심. |
| 9 | Drop-in이라지만 head MLP 파라미터 수는 backbone L1 head보다 클 텐데, fair comparison? | 정확한 파라미터 차이는 보고 안 됨. "Same backbone, same step" controlled이나 head capacity controlled는 아님. Attribution 약점. |
| 10 | Multimodal demonstration capture가 안 되는데 단봉 Gaussian-blob이라는 한계? | 인정되는 약점. 분포 자체는 multimodal을 표현할 수 있으나 training target이 단봉. Future work으로 mixture target 자연스러움. |

<!-- VERIFIED: pdf -->
