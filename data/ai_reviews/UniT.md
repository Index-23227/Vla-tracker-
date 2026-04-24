# UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

> **한 줄 요약**: Human egocentric video와 humanoid action을 "visual consequence"를 공통 anchor로 삼아 **embodiment-agnostic discrete latent space**로 통합하는, tri-branch cross-reconstruction 기반의 통합 physical language tokenizer.

---

## 1. 배경 및 동기

- Humanoid foundation model의 scaling 병목 = robotic data 부족.
- 대규모 egocentric human video는 scalable 대안이지만, 인간과 로봇의 **kinematic mismatch**(cross-embodiment chasm)로 직접 전이 불가.
- **핵심 통찰**: 이질적 kinematics도 **물리적 결과(visual consequence)**는 보편적 → 시각을 anchor로 활용하자.

---

## 2. 방법론

### UniT = Unified Latent Action Tokenizer via Visual Anchoring
- 인간과 humanoid action 모두를 공유 discrete latent space의 token으로 맵핑.
- Token은 "embodiment-agnostic physical intent"를 표현.

### Tri-Branch Cross-Reconstruction
1. **Action → Vision** 분기: action이 미래 vision을 예측하게 하여 kinematics를 물리적 결과에 anchoring.
2. **Vision → Action** 분기: vision이 action을 reconstruct하게 하여 무관한 시각 confounder를 필터링.
3. **Fusion** 분기: 정제된 두 modality를 결합해 shared discrete latent space를 형성.

### Downstream: VLA-UniT (policy) / WM-UniT (world model) — **Hybrid action head**

PDF Sec 3.3 확인: Policy는 **2단계 hybrid** 구조.

1. **UniT Token Prediction (Eq. 4, parallel CE)**: VLM 시퀀스에 learnable queries q_t 추가, VLM이 unified discrete UniT 토큰을 **병렬 분류** (causal AR 아님).
   ```
   p̂_t = f_VLM(o_t, ℓ, q_t),   L_token = CE(p̂_t, c_t)
   ```
2. **Flow-Matching Action Generation (Eq. 5)**: 연속 embodiment-specific action을 flow matching velocity head V_θ로 생성.
   ```
   L_fm = E[ ||V_θ(A^τ_t | x_t, Enc(o_t), τ) − (A_t − ε)||² ]
   ```

총 loss: `L_VLA = L_token + λ_fm · L_fm`

WM-UniT: Cosmos Predict 2.5 기반, UniT action-branch features z_a를 control interface로 사용.

### Backbone 구성

- Visual: **frozen DINOv2** (E_v branch, IDM 역할)
- VLM: **Qwen2.5-VL** (GR00T n1.5 framework 내)
- Tokenizer: RQ-VAE shared codebook across 3 branches (E_v / E_a / E_m)

---

## 3. 실험 결과

### RoboCasa GR1 Tabletop 시뮬 (Fig. 9, full-data 24×1000 trajs)

| 방법 | Pick & Place | Articulated | **Overall** |
|------|-------------:|------------:|------------:|
| Diffusion Policy | 40.4 | 35.0 | 40.9 |
| GR00T N1.6 | 47.0 | 42.3 | 47.6 |
| GR00T (Qwen3-VL) | 50.4 | 50.1 | 47.8 |
| FAST (Qwen3-VL) | 51.3 | 50.3 | 48.8 |
| OFT (Qwen3-VL) | 44.4 | 43.7 | 43.9 |
| FLARE | 58.2 / 55.4 | 38.4 | 55.0 |
| **VLA-UniT (Qwen2.5-VL)** | **67.3** | **64.7** | **66.7** |

- 이전 최강 FLARE 대비 +11.7pp, GR00T baseline 대비 +18.9pp.

### 데이터 효율성 (Fig. 10 Left)

| 설정 | GR00T-Qwen2.5 | **VLA-UniT** |
|------|--------------:|-------------:|
| 24×100 trajs (10% 데이터) | 21.3% | **45.5%** |
| 24×1000 trajs (full) | 47.8% | **66.7%** |

**⟹ UniT는 10% 데이터로 GR00T full-data (47.8%)와 유사** → **~10배 sample efficiency**.

### IRON-R01-1.11 실물 휴머노이드 (Fig. 11)

| Task | GR00T | UniT w/o human | **UniT w/ human** |
|------|------:|---------------:|------------------:|
| Pick & Place | 30 | 70 | **78** |
| Pouring | 5 | 35 | **75** |

- Pouring: human 데이터 co-train 추가 시 **75%** (baseline 5% → +70pp).

### OOD 일반화 (Fig. 11 Right, IRON)

5개 카테고리 (Background / Target / Distractor / Geometric / Combinatorial) 모두에서 human co-train 효과 확인.

---

## 4. 한계 및 미해결 문제

1. **Visual confounder의 완전 제거 난이도**: Vision→Action 분기가 조명·배경 등 irrelevant signal을 완벽히 걸러낸다는 보장 없음.
2. **Discrete tokenization의 precision loss**: 미세한 humanoid motor 명령을 discrete token으로 양자화할 때 정밀도 손실 가능.
3. **Human-egocentric 데이터 편향**: 일상 인간 활동이 humanoid 작업 분포와 다를 때 transfer 효과 제한.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual consequence를 cross-embodiment anchor로 삼은 tri-branch 설계가 독창적 |
| **Practical impact** | ★★★★☆ — Humanoid data scaling 문제에 정면 대응 |

**강점**: "Kinematics는 다르나 visual outcome은 같다"는 철학이 단순·강력하고, tri-branch 구조가 이를 엔지니어링적으로 구현. **약점**: Discrete latent의 해상도와 confounder filtering의 실제 품질 검증 필요.

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LAPA와 차별점은? | Tri-branch 구조와 visual anchoring 철학, discrete token의 physical intent로서의 정의 |
| 2 | 왜 discrete latent인가 (continuous 대비)? | Shared RQ-VAE codebook이 human/humanoid/LLM token prediction 모두와 자연스럽게 호환; VLM의 CE loss 목적함수와 일치. |
| 3 | Vision이 action을 복원하는 분기의 필요성? | Action이 vision을 예측만 하면 spurious correlation 학습 → reverse branch가 regularizer 역할 |
| 4 | Kinematic mismatch가 크면 visual anchor도 달라지지 않나? | 사용 도구·객체의 상태 변화 등 결과 수준에서는 공유됨 (핵심 가정) |

<!-- VERIFIED: pdf -->
