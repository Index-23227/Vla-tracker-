# FALCON: From Spatial to Actions — Grounding VLA in Spatial Foundation Priors

> **한 줄 요약**: VGGT 기반 Embodied Spatial Model(ESM, 1.0B)이 RGB(또는 옵션의 depth/pose)에서 풍부한 3D 토큰을 추출하고, 이를 VLM(Kosmos-2 1.6B) 출력의 action token에 element-wise add로 주입하는 Spatial-Enhanced Action Head를 통해 vision-language 정렬을 보존하면서 3D 공간 추론을 강화한 ByteDance Seed의 2.9B VLA. CALVIN ABCD→D 4.53, ABC→D 4.40, SimplerEnv WidowX 56.3 / Google Robot 62.9 SOTA 달성.

---

## 1. 배경 및 동기

- 기존 VLA는 2D 인코더 위에 구축되어 **3D 공간 추론 결손** → 높이·크기 변화·기하 추론에서 일반화 한계.
- 3D 통합 시도 두 갈래:
  1) **Explicit 3D 입력**(PointVLA, GeoVLA): 전용 센서 필요 + Open-X 같은 RGB-only 데이터셋과 호환 불가.
  2) **Weak 3D cue**(SpatialVLA, 3D-VLA): pseudo-depth/learnable embedding을 LLM 토큰에 concat → vision-language 정렬 붕괴, 약한 기하 신호.
- 핵심 질문: VLM의 의미 표현을 깨지 않으면서 **풍부한 3D priors**를 정책에 주입할 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Figure 2)
세 컴포넌트:
1) **2D VLM** (Kosmos-2 1.6B): 멀티모달 의미 표현, action token t̂_act 생성.
2) **Embodied Spatial Model (ESM, 1.0B)**: VGGT 기반 spatial encoder. DINO로 visual token T_vis 생성 후 learnable camera token과 함께 N개 cross-/self-attention 블록을 통과 → spatial token T_spl.
3) **Spatial-Enhanced Action Head**: T_spl을 max-pool → MLP adapter D로 t̃_spl 생성 → **element-wise add** f_fused = t̂_act + t̃_spl → MLP 또는 LSTM action predictor.

### 2.2 ESM의 3D 조건 주입
- Depth D_t (validity mask 동봉), camera pose P를 stochastic하게 활성화: (T_spl, t̂_cam) = E_spl(T_vis + b_d·T_dpt, b_p·t_gt-cam + (1−b_p)·t_cam), b_d, b_p ∼ Bernoulli(p) (식 5).
- VGGT의 depth/pointmap/pose loss로 multi-task supervision → RGB-only/RGB-D/RGB-D-Pose 모드 전환 가능.

### 2.3 학습 전략
- **2-stage**: Stage 1 — pre-trained 컴포넌트 동결, adapter만 학습 (정렬). Stage 2 — VLM + adapter joint refine, 나머지 동결.
- Loss: MSE(6D pose) + λ·BCE(gripper) (식 2).
- 32× A100 학습.

### 2.4 핵심 디자인 결정 (Table 4 ablation, ABC→D Avg.Len.)
- VLM에 spatial token 주입(FALCON_VLM-tokens) 3.79 / Cross-Attention 3.68 / FiLM-Gated 3.76 / **Element-wise Add 3.91** — parameter-free 최고.

---

## 3. 실험 결과

### 3.1 CALVIN (Table 1)
| Method | Setting | Avg. Len. ↑ |
|--------|---------|------------|
| RoboVLM | ABCD→D | 4.49 |
| Seer-Large | ABC→D | 4.28 |
| **FALCON** | ABCD→D | **4.53** |
| **FALCON** | ABC→D | **4.40** |

### 3.2 SimplerEnv WidowX (Table 2)
- FALCON 평균 56.3% (SpatialVLA 42.7, RoboVLM 37.5). Put Spoon 62.5% (SpatialVLA 16.7), Put Eggplant 100%.

### 3.3 SimplerEnv Google Robot (Table 3)
- FALCON 62.9% (SpatialVLA 55.3, RT-2-X 55B 46.3).
- Drawer-Apple 41.7% — SpatialVLA·OpenVLA·Octo 모두 0%, RT-2-X도 3.7%.

### 3.4 실세계 (Section 4.2)
- 9개 base task 평균 70.0% (SpatialVLA 44.4%).
- Few-shot Unseen 평균에서 차순위 대비 +27%.
- Spatial Understanding (큰/작은 블록, 컵 높이 변화)에서 RoboVLM의 충돌·조기 release 실패를 회피.

### 3.5 Modality Transferability (Table 5)
- FALCON RGB-only 4.08 ≈ Kosmos-VLA RGB-D 4.05 → RGB만으로도 explicit 3D 활용 모델과 동등.
- Test-time depth 추가 시 ABC→D 3.91 → 3.95.
- ESM monocular depth (Table 6): RGB-only δ<1.25 90.91%, +depth condition 99.79%.

---

## 4. 한계

1. **Open-source 미공개**: 코드·가중치 비공개(YAML 명시). 재현 불가.
2. **테이블탑 매니퓰레이션 한정**: 7-DoF 그리퍼 단일 팔. 양손·humanoid·navigation 미평가.
3. **VGGT 의존성**: ESM이 VGGT 백본 — VGGT가 학습된 도메인 분포 외(예: 외관 극단)에서 깨질 가능성.
4. **Element-wise add의 정보 손실**: max-pool로 spatial token 다수를 단일 벡터로 압축 → 다중 객체 장면에서 fine-grained 정보 손실 가능.
5. **LIBERO·RoboCasa 미평가**: 본 트래커의 표준 벤치마크 부재.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — Spatial token을 action head로 우회한 분리 설계 |
| Technical depth | ★★★★★ — 2-stage 학습, stochastic 3D 조건, 풍부한 ablation |
| Experimental rigor | ★★★★★ — sim 3개 + 11개 real task |
| Practical impact | ★★★★☆ — 그러나 비공개로 영향 제약 |
| Writing | ★★★★☆ |

**강점**: 의미 표현 보존하면서 3D 강화, RGB↔RGB-D 자유 전환. **약점**: 비공개, 매니퓰레이션 한정.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | Element-wise add가 cross-attention보다 좋은 이유? | t̂_act가 이미 task-conditioned이고 t̃_spl이 max-pool된 단일 벡터이므로 추가 파라미터 없는 add가 over-parameterization을 피함. 더 큰 spatial token 시퀀스에서는 결론이 다를 수 있음. |
| 2 | VGGT vs DUSt3R vs MASt3R 선택 근거? | VGGT는 multi-view 처리가 가능하고 token 표현이 풍부. 다만 ESM이 VGGT loss로 학습되어 종속성 큼. |
| 3 | RGB-only만으로 RGB-D Kosmos-VLA를 따라잡는다면 ESM이 정말 3D priors를 학습했는가, 단지 더 좋은 visual encoder인가? | 둘 다. δ<1.25 정확도가 +depth 시 90.9→99.8로 점프 → 3D 신호 활용 증거. 단 분리해 검증한 ablation은 부족. |
| 4 | Stage 2에서 VLM unfreeze로 catastrophic forgetting 위험은? | Stage 1 adapter alignment로 완화. 그러나 VLM의 zero-shot reasoning 손실 정량화 부재. |
| 5 | 다중 카메라(wrist + side) 활용 방식? | Side cam만 ESM 입력. Wrist cam은 VLM에만. 비대칭으로, 양손/eye-in-hand 작업 확장 시 재설계 필요. |

<!-- VERIFIED: pdf -->
