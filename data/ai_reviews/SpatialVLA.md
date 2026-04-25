# SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model

> **한 줄 요약**: Shanghai AI Lab + Tsinghua의 SpatialVLA는 PaliGemma2 위에 (1) **Ego3D Position Encoding** (ZoeDepth로 추정한 depth로 egocentric 3D 좌표를 SigLIP feature에 더함)과 (2) **Adaptive Action Grids** (Gaussian-fitted 분포로 polar coordinate를 비균등 grid로 양자화, action당 7 token → 3 token)을 추가하여, 1.1M episode pre-training으로 SimplerEnv에서 π0를 능가하는 zero-shot 성능과 LIBERO 평균 78.1% (rank #1)을 달성한 open-source 4B VLA.

---

## 1. 배경 및 동기 (Section I)

- 기존 VLA는 2D observation에 의존 → 3D 공간 이해 부족. 인간은 본능적으로 manipulation을 위한 3D mental representation을 구성 (Section I, Logie 1990 / Piaget 인용).
- 두 가지 도전 (Section I):
  1. **Observation heterogeneity**: 다양한 robot의 camera 위치·extrinsic이 달라 3D-aligned representation이 어려움.
  2. **Action heterogeneity**: 다양한 robot의 DoF·motion controller·workspace 차이 → cross-embodiment spatial action learning 어려움.
- 해결 방향: robot-camera calibration이 필요 없는 egocentric 3D representation + 데이터 분포 기반 adaptive action discretization.

---

## 2. 방법론 (Section III, Figure 2)

### Ego3D Position Encoding (Section III-A, Eq. 2)
- **ZoeDepth** (Bhat et al. 2023)로 monocular depth map D 추정.
- Camera intrinsic으로 back-projection π⁻¹: 각 pixel의 3D 좌표 p = (x, y, z) in egocentric frame.
- SigLIP visual feature X ∈ R^{d×h×w}와 sinusoidal+MLP encoded 3D position embedding P′를 더함:
  $$O_{3d} = X + P' = X + \text{MLP}(\gamma(P))$$
- Robot-camera calibration 불필요 → 임의 robot에 plug-and-play.

### Adaptive Action Grids (Section III-A, Figure 3, Eq. 4)
- 7-DoF action을 3 그룹으로 분해: a = {a_trans, a_rot, a_grip}.
- **Polar coordinate transform**: translation (x, y, z) → (φ, θ, r) — 방향(φ, θ)과 거리(r) 분리. 방향에 더 많은 grid 할당.
- **Gaussian fit**: 전체 데이터셋의 normalized action에 N(μ_a, Σ_a) fit.
- **Equal-probability grid**: 누적분포가 균등하도록 M개 interval로 분할:
  $$a_2, ..., a_M = \arg\min \int_{a_i}^{a_{i+1}} f(x)dx - 1/M$$
- 결과: M_trans = M_φ · M_θ · M_r grid + M_rot grid + 2 grip token. **Default V = 8194 token vocabulary**.
- 한 step action = **3 token만 사용** (vs RT-2/OpenVLA 7 token) → 2.3× inference speedup.

### Pre-training (Section III-B)
- 1.1M episode (OXE subset + RH20T) on **64 A100, 10 days, batch size 2048**.
- Standard next-token cross-entropy (Eq. 1).
- **Frozen text token embedding** (E_text frozen) — VLM의 world knowledge 보존, instruction following 향상.
- DROID는 마지막 1/3 단계에서 제외 (OpenVLA 관행).

### Post-training: Spatial Embedding Adaption (Section III-B, Eq. 6)
- 새 robot setup에서 새 Gaussian N(μ_new, Σ_new) fit → 새 grid G_new 구성.
- 새 grid embedding은 pre-trained grid의 인접 K개 grid에서 **trilinear interpolation으로 초기화** (Eq. 6).
- Grid 재이산화만으로 새 embodiment에 빠른 적응.

### 시스템 (Section IV)
- 4B parameter, 단일 third-person camera만 사용, 4-step future action chunk (12 spatial token).
- Inference: 8.5GB GPU memory, **20Hz on RTX 4090** (Section IV).

---

## 3. 실험 결과

### SimplerEnv Google Robot (Table I)
| Model | Visual Match Avg | Variant Agg Avg |
|-------|------------------|-----------------|
| RT-2-X (55B) | 60.7% | 64.3% |
| RoboVLM (FT) | 63.4% | 51.3% |
| π0* (BF16, open-pi-zero) | 70.1% | — |
| **SpatialVLA zero-shot** | **71.9%** | **68.8%** |
| **SpatialVLA fine-tuned** | **75.1%** | **70.7%** |

3.5B vs RT-2-X 55B로 **15× 작은 모델**이 우월. 특히 Open/Close Drawer variant aggregation에서 36.2% (vs RT-2-X 35.3%) — drawer task는 spatial precision이 핵심이라 Ego3D 효과가 두드러짐.

### SimplerEnv WidowX (Table II)
- SpatialVLA zero-shot 34.4%, fine-tuned 42.7% — RoboVLM (31.3%) 대비 +11.4%p.
- "Put Eggplant in Yellow Basket" fine-tuned **100% 성공** (grasp 100% + success 100%).

### Real-world WidowX (Section IV-A, Figure 5)
7 task suite × 11 trial. Motion distractor (#3, #4: 사람이 object를 동적으로 움직임)에서도 robust tracking & grasping. Instruction following (#5-7: "green cup on white plate, not pink") 정확.

### LIBERO Fine-tuning (Table III, Section IV-B)
| Suite | Diffusion Policy | Octo FT | OpenVLA FT | TraceVLA FT | **SpatialVLA FT** |
|-------|------------------|---------|------------|-------------|-------------------|
| Spatial | 78.3 | 78.9 | 84.7 | 84.6 | **88.2** (#1) |
| Object | 92.5 (#1) | 85.7 | 88.4 | 85.2 | 89.9 (#2) |
| Goal | 68.3 | **84.6** | 79.2 | 75.1 | 78.6 |
| Long | 50.5 | 51.1 | 53.7 | 54.1 | **55.5** (#1) |
| **Average** | 72.4 | 75.1 | 76.5 | 74.8 | **78.1** (#1) |

LIBERO-Spatial에서 **88.2%로 압도적 1위** — 명백히 Ego3D PE의 효과. Long-horizon에서도 1위지만 격차는 작음 (architecture상 long-horizon observation 처리 미설계라고 저자 명시).

### Franka 실세계 (Figure 6)
- Single-task 82% (vs Diffusion Policy 81%, OpenVLA 70%, Octo).
- Instruction following: SpatialVLA가 OpenVLA 대비 **+12%p** 향상.
- Multi-task 57% — 가장 어려운 setup에서 generalist 우위.

### Spatial Understanding 전용 평가 (Figure 7, Section IV-C)
4 task에서 Ego3D PE의 효과를 직접 측정:
- Franka task #1 ("place plush toy closest to robot on car"): 73% — spatial prompt 이해 필요.
- WidowX OOD #2-#4 (cup/pot **height change**): SpatialVLA가 큰 폭 우위.
- Octo·DP·OpenVLA 모두 50% 이하 — depth 정보 부재가 결정적.

### Ablation (Table IV)
| Setting | Pick Coke (VA) | Move Near (VA) |
|---------|----------------|----------------|
| Full SpatialVLA | 81.6% | 79.2% |
| Linear 256 bin | 40.7% (−40.9) | 47.1% (−32.1) |
| Uniform Gaussian | 77.9% | 64.2% |
| Resolution 1026 | 74.4% | 59.1% |
| − Ego3D | 68.9% (−12.7) | 66.7% (−12.5) |
| − Freeze LLM emb | 70.2% | 63.1% |

핵심 ablation:
- **Linear 256-bin → adaptive grid**: +36.5%/+42.1% (Pick Coke VA/VM) — 가장 큰 단일 영향.
- **Resolution scaling 1026 → 8194**: Move Near +20.1%, Eggplant +33.3% — 고해상도가 fine-grained motion에 결정적.
- **Ego3D 제거**: variant aggregation 큰 폭 하락 (lighting/texture/camera 변화에 vulnerable).
- **Frozen LLM embedding**: 학습 가속 + instruction following 향상.

### Post-training Ablation (Table V)
- 대규모 dataset (Fractal/BridgeV2): Spatial embedding adaption 효과 marginal (+2.9% Move Near VM) — distribution이 이미 pre-training과 유사.
- 소규모 dataset (LIBERO LoRA): adaption 큰 효과 (+4.6%/+5.1%/+2.2%/+5.4%) — Figure 8 시각화에서 spatial grid feature alignment 확인.
- LoRA > full FT on 소규모.

---

## 4. 한계 (Section V)

- **Gaussian 가정 한계**: 단일축 motion이 dominant인 robot에서 grid clustering 발생 가능 (저자가 명시).
- **Autoregressive 3-token decode**: 21Hz로 빠르지만 π0/CogACT의 multi-action chunk diffusion보다 느림.
- **Long-horizon 약함**: history token만으로 처리 → LIBERO-Long 1위지만 격차 작음. 명시적 long-context memory 부재.
- **OXE quality variance**: 일부 noise가 학습 방해 — 고품질 subset distillation이 future work.
- **Monocular depth 의존**: ZoeDepth estimation error가 propagate 가능. Transparent/reflective object 평가 누락.

---

## 5. 총평

SpatialVLA의 핵심 가치는 "거대한 vision encoder 없이도 ZoeDepth + sinusoidal embedding 같은 가벼운 도구로 **3D inductive bias**를 주입하면 spatial-heavy benchmark에서 큰 차이를 만든다"는 것을 보인 것이다. Table IV ablation에서 Linear 256-bin → adaptive grid 단일 변경이 +36~42%p 격차를 만들어, RT-2/OpenVLA가 "256 bin uniform"이라는 잘못된 default에 묶여 있었음을 시사한다. 모델 weight·코드가 **공개**되어 (`SpatialVLA/SpatialVLA`) 재현 가능하다는 점에서 π0/π0.5 대비 학술 가치가 높다. 다만 (1) Ego3D PE vs Adaptive Grid contribution 분리 부족, (2) inference 21Hz는 real-world high-frequency control엔 제한적.

---

## 6. 예상 질문

- **Q1**: ZoeDepth 대신 GT depth는?
  - A: 논문은 monocular ZoeDepth만 평가. 핵심 주장은 "calibration-free egocentric 3D"이므로 estimation으로 충분. GT가 있으면 더 좋을 수 있으나 cross-embodiment 적용성 약화.
- **Q2**: 왜 polar coordinate?
  - A: Section III-A — 방향 (φ, θ)과 거리 r 분리. 방향에 더 많은 grid 할당 가능. Cartesian 균등 분할은 직선 motion에 비효율.
- **Q3**: 8194 vocabulary는 너무 크지 않나?
  - A: Table IV — 1026 → 8194에서 task별 +20~33%p. fine-grained motion 표현에 필요.
- **Q4**: π0와의 직접 비교?
  - A: Table I에서 π0* Visual Match Avg 70.1% vs SpatialVLA 71.9%(zero-shot)/75.1%(FT). 약간 우위. 단 π0는 50Hz vs SpatialVLA 20Hz로 inference frequency는 π0 우세.
- **Q5**: Spatial embedding adaption은 LoRA와 어떻게 다른가?
  - A: LoRA는 weight delta 학습, spatial adaption은 **새 grid partitioning + trilinear interpolation init**. 직교적이며 함께 사용 가능 (Table V #5).

<!-- VERIFIED: pdf -->
