# MVUCF: Multi-View Unified Camera Fields — Geometry-Shaped Action-Facing Representations for RGB-Only Multi-Camera VLA Policies

> **한 줄 요약**: GR00T-N1.6-3B의 action-facing hidden grid(layer 15)에 **coordinate-query depth 목적함수**와 **preprocessing-aware cross-view correspondence 목적함수**를 학습 단계에서만 주입하여 멀티카메라 공유 latent field를 만들고, 배포 시에는 depth/calibration/auxiliary head를 모두 제거해 **추가 inference FLOPs 0**으로 LIBERO 98.9%, LIBERO-Plus +22.4점, RoboTwin 6-task +23.3점을 달성한 training-only 프레임워크.

- **arXiv**: 2608.01826v1 (2026-08-03, cs.RO)
- **소속**: HKUST(Guangzhou), Zhejiang University, Simon Fraser University, AgiBot, Xi'an Jiaotong University
- **베이스 정책**: GR00T-N1.6-3B (NVIDIA), DiT action head

---

## 1. 배경 및 동기

### 멀티카메라 VLA의 두 가지 구조적 결함
Contact-rich manipulation은 end effector·물체·목표를 동시에, 그리고 가림(occlusion) 하에서 관찰해야 하므로 head + wrist 같은 멀티카메라 관측이 표준이 되었다. 그러나 대부분의 멀티카메라 정책(ACT, Octo, GR00T 계열)은 **view별 토큰을 단순 concat**하고 sparse한 action supervision만으로 뷰 간 물리적 관계를 암묵적으로 학습하길 기대한다. 저자들은 여기서 두 결함을 지적한다.

1. **Metric depth 결여** — VLA 백본은 2D semantic 데이터로 사전학습된 VLM에서 초기화되므로, hidden state에서 metric 거리를 복원하기 어렵다.
2. **Cross-view 불일치** — concat만으로는 "같은 물리적 점"에 대응하는 토큰이 카메라 간에 일관되게 표현되도록 강제되지 않는다. El Banani et al.(2024)이 보인 visual foundation model의 뷰 간 표면 표현 불일치가 그대로 남는다.

### 진단 결과 (Figure 2 / Figure 6)
| 프로브 | Native backbone | Geometry injection 후 |
|---|---|---|
| Depth MAE (단일 프레임, per-view) | 4.3 cm | 0.78 cm |
| Depth MAE (full-eval 평균) | 4.9 cm | 0.44 cm |
| 2cm 이내 예측 비율 | 44% | 97% |
| Cross-view retrieval Hit@1 | 0.4% (random 0.3%) | 64% |

즉 **native 상태는 cross-view 검색이 사실상 랜덤**이라는 것이 이 논문의 출발점이다.

📌 [Figure 2 삽입] — depth probe MAE와 cross-view retrieval Hit@1 진단

---

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Figure 3)
```
Stage 1 (Geometry injection):
  multi-camera RGB → GR00T VLM (layers 8-15 학습, encoder/하위 블록 frozen)
    → layer-15 hidden grid
        ├── Coordinate-Query Depth Head   (L_depth)
        └── Cross-View Correspondence Head (L_cv)
  * depth 관측 + camera intrinsics/extrinsics 사용

Stage 2 (Action training):
  auxiliary head 제거, depth/calibration 제거, VLM frozen
    → native GR00T DiT action head만 RGB로 학습

Deployment: 원래의 RGB-only inference graph (추가 파라미터/FLOPs 없음)
```

핵심 설계 철학은 "**geometry를 별도 branch에 저장하지 말고, action module이 실제로 소비하는 바로 그 grid를 재형성하라**"이다.

### 2.2 Cross-View 타깃 구성 — project-before-preprocess
소스 픽셀 `u_i^raw`와 depth `z*_i`, intrinsics `K_i`, camera-to-world `T_{i→w}`로부터

```
P_w    = T_{i→w} · π⁻¹(u_i^raw, z*_i, K_i)
u_j^raw = π(T_{w→j} P_w, K_j)
```

로 타깃 뷰에 재투영한 뒤, 다음 3가지 검증을 통과한 대응만 채택한다.
- 이미지 경계 내부
- 타깃 프레임 depth ∈ [0.05, 5.0] m
- z-buffer 일관성 `|Z_j − z*_j(u_j^raw)| < 0.10 m` (가림·불일치 제거)

그 다음 **GR00T의 실제 전처리 체인**(letterboxing → resize → crop → smart-resize → patchify → pixel-unshuffle)을 그대로 합성한 `T_{img→grid}`로 raw 픽셀을 토큰 중심 `q*_j`에 매핑한다. 저자들은 raw 픽셀을 feature grid 인덱스로 "그냥 스케일링"하면 좌표 불일치가 생긴다고 강조하며, 이 **투영 후 전처리(project-before-preprocess)** 순서를 방법론의 실질적 핵심 중 하나로 둔다.

### 2.3 Coordinate-Query Depth Head
Dense full-image 예측이 아니라 **연속 좌표 질의**로 metric depth를 예측한다: `ẑ_i(q) = g_φ(F_i^t, q)`.

- 뷰당 3072개 query (절반 uniform, 절반은 근거리 표면/depth 불연속 지점에 집중)
- Grid-preserving trunk: `LN_2048 – Linear_1024 – GELU – Conv3×3(1024) – GELU` 후 bilinear feature lookup
- Query 표현 3074차원 = 샘플된 1024-d feature + local x/y difference + sub-cell phase `φ(q)=q−⌊q⌋` (**absolute position 입력 없음**)
- 예측: `MLP 3074→1024→1024→1` + Softplus (metric depth), 병렬 `MLP 3074→1024→1` (log variance, uncertainty-aware)

```
L_depth = 0.5·L_silog + 1.25·L_inv + 1.0·L_grad + 0.1·L_seam + 0.05·L_unc
```
Inverse-depth 항(가중치 1.25로 가장 큼)이 근거리 manipulation workspace를 강조하고, `+x/+y` 이웃 query가 국소 표면 변화를 보존한다.

### 2.4 Cross-View Correspondence Head
공유 projector `LN_2048 – Linear_2048→128` + L2 정규화 후 로짓 `ℓ_mn = 10·(e^s_m)ᵀ e^t_n`.

- Soft target: `y_mn ∝ exp(−‖c_n − q*_j‖² / 2σ²)`, **σ = 0.75** — 단일 grid cell에 확률을 몰지 않고 투영점 주변 이웃을 감독
- Hard-negative InfoNCE: spatial-ring / in-batch / alternate-view negative
- Margin loss(γ = 0.2)로 투영 정답과 최강 non-positive를 분리
- 3×3 local branch로 sub-token 이웃 해상 (logit scale은 학습 중 annealing)

```
L_cv = L_softCE + 0.1·L_hnce + 0.1·L_margin
```
외곽 가중치는 학습 중 annealing되어 **초기에는 same-point 정렬을 강조하고 action 학습 전에는 제약을 완화**한다.

### 2.5 왜 layer 8–15인가
- VLM probing 결과 metric depth 정보가 **깊은 블록에 집중**
- 개발 중 layer 12에 cross-view 목적함수를 걸었더니 토큰 수준 분리도가 악화 → **layer 15**(action module이 직접 소비하는 최종 hidden spatial grid) 선택
- 연속 상위 절반(8–15)을 full-rank로 업데이트, visual encoder와 하위 블록은 frozen. LoRA/adapter 같은 PEFT 모듈을 **추가하지 않는다**

---

## 3. 데이터 전략

- **LIBERO / RoboTwin 모두 공식 오픈소스 데모만 사용**. RoboTwin은 task당 official `clean50` split(50 궤적) → 6 task 총 300 궤적.
- 동일 궤적을 **replay**하여 synchronized depth + intrinsics/extrinsics를 기록. 이 depth/calibration은 **geometry injection에서만** 쓰이고 action 학습은 원본 RGB-action 기록만 사용.
- 따라서 Base와 Ours는 **데모 커버리지가 완전히 동일**하고, 차이는 action 학습 이전에 적용된 목적함수뿐이다 (attribution 측면에서 강한 통제).
- 실물: task당 100회 teleoperated 데모 (AgiBot Expedition A2 휴머노이드).
- 실물 depth 처리: 유효 센서 측정은 보존하고 invalid hole에만 completion 적용, validity mask는 depth loss에서 계속 활성.

---

## 4. 시스템/학습 세부사항

| Stage / benchmark | Steps | Batch | Hardware |
|---|---|---|---|
| Geo / LIBERO (~18 h) | 50k | 128 | 2×H100 |
| Action / LIBERO | 60k | 128 | 2×H100 |
| Geo / RoboTwin (~18 h) | 50k | 128 | 4×H100 |
| Action / RoboTwin (joint) | 180k | 128 | 4×H100 |

- 모든 체크포인트는 **사전 지정된 terminal checkpoint**(50k / 60k / 180k), validation·downstream success 기반 선택 없음 → cherry-picking 방지
- Warmup은 초기 5% step, GR00T-N1.6 공식 레시피와 동일
- Optimizer 설정은 Base와 Ours가 동일
- Visibility gate 10 cm는 **정책 평가 전에** 타깃 구성 자체(positive yield vs. 오대응 균형)만으로 고정

---

## 5. 실험 설계 및 평가 프로토콜

저자들은 하나의 "evidence chain"으로 실험을 구성한다.
1. **Latent property 진단** (held-out depth probe, cross-view retrieval, Cohen's d)
2. **RGB-only 제어 실험**: LIBERO, LIBERO-Plus, RoboTwin 6-task
3. **Component ablation** (Depth only / Cross-view only / Full)
4. **실물 휴머노이드 파일럿**

- LIBERO: 서로 다른 3개 training seed 체크포인트, suite당 100 episode → mean ± SD
- LIBERO-Plus: **seed-0 terminal checkpoint를 zero-shot**으로, 재학습·적응 없이 7개 perturbation family 평가
- RoboTwin: task당 100 rollout × 2 round = 200 rollout, 총 1200 rollout/policy. 두 round는 **동일 체크포인트 재사용**이므로 training variance가 아니라 rollout 수준 binomial(Wilson) 불확실성만 보고
- 효과 크기: `d(A,B) = (μ_A − μ_B) / sqrt((σ²_A + σ²_B)/2)`

---

## 6. 실험 결과 심층 분석

### 6.1 표준 LIBERO (Table 1)
| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| Spatial Forcing | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| **GR00T-N1.6 (Base)** | 99.3±1.2 | 99.2±0.8 | 98.4±1.2 | 92.9±1.0 | 97.4±0.3 |
| **MVUCF (Ours)** | **100.0±0.0** | 99.2±0.3 | **99.5±0.5** | **97.0±1.5** | **98.9±0.4** |
| Δ | +0.7 | 0.0 | +1.1 | **+4.1** | +1.5 |

포화된 벤치마크에서 +1.5점이지만, **이득의 대부분이 long-horizon LIBERO-10(+4.1)에 집중**된다는 점이 서사와 정합적이다 — depth 모호성·가림·공간 오차가 서브골에 걸쳐 누적되는 구간.

### 6.2 LIBERO-Plus robustness (Table 2) — 진짜 하이라이트
| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| Evo-Depth† | 47.2 | 49.2 | 78.9 | 88.1 | 76.4 | 77.6 | 69.6 | 69.6 |
| GR00T-N1.6 (Base) | 20.9 | 40.2 | 35.0 | 65.4 | 76.3 | 27.8 | 51.3 | 42.8 |
| **MVUCF (Ours)** | 24.1 | 54.1 | 72.9 | 95.1 | 91.8 | 61.3 | 74.9 | **65.2** |
| Δ | +3.2 | +13.9 | **+37.9** | +29.7 | +15.5 | **+33.5** | +23.6 | **+22.4** |

- **7개 카테고리 전부 개선**, zero-shot 전이. Language(+37.9)와 Noise(+33.5) 이득이 특히 큰 것은 geometry injection이 단순 시각 강건성을 넘어 표현 전반을 정리했음을 시사한다.
- 다만 **Camera 항목은 +3.2로 가장 작다**. 저자들 스스로 "geometry injection이 학습 시점 카메라 파라미터를 암묵적으로 표현에 내재화하므로 viewpoint/FoV perturbation에서 이득이 제한된다"고 인정한다 — 정직하지만 동시에 방법의 본질적 취약점이다.
- 절대값으로는 OpenVLA-OFT(69.6)·Evo-Depth(69.6)에 여전히 못 미친다(65.2). 이 논문의 주장은 SOTA가 아니라 **matched-pair 개선**이다.

### 6.3 RoboTwin 6-task (Table 3 / Table 6)
| Family | Task | Base | Ours | Δ |
|---|---|---|---|---|
| Touch | click_alarmclock | 92.0 | 99.5 | +7.5 |
| Touch | click_bell | 85.5 | 100.0 | +14.5 |
| Move/Place | move_pillbottle_pad | 6.5 | 40.5 | +34.0 |
| Move/Place | place_phone_stand | 0.0 | 27.0 | +27.0 |
| Contact | press_stapler | 43.0 | 93.5 | +50.5 |
| Contact | turn_switch | 4.5 | 11.0 | +6.5 |
| **Overall** | | **38.6** | **61.9** | **+23.3** |

- Base 38.6% (95% Wilson CI 35.9–41.4) → Ours 61.9% (59.1–64.6). **CI가 겹치지 않아** 이 개선은 통계적으로 확실하다.
- Family별 이득: Touch +11.0, Move-and-Place +30.5, Contact +28.5 → **정밀한 공간 정렬이 필요한 계열에서 이득이 크다**는 예측과 일치.
- `place_phone_stand`는 Base가 0/200으로 완전 실패, Ours 54/200. `turn_switch`는 Ours도 11%로 여전히 어렵다.
- 6개 task는 **학습 전에 action primitive 분류(Touch/Move-and-Place/Contact)로 사전 지정**되었고, 관측된 성능으로 고르지 않았다고 명시 — 그럼에도 full-benchmark leaderboard가 아니라는 점은 남는다.

### 6.4 실물 휴머노이드 (Figure 8)
AgiBot Expedition A2, two-cake placement + cup nesting, policy/task당 30 trial.
- Ours 49/60 = **81.7%** (Wilson CI 70.1–89.4)
- Base 40/60 = 66.7% (54.1–77.3)
- 세부: 17/30→21/30, 23/30→28/30 (두 task 모두 개선)
- **CI가 겹치므로** 저자들도 "pilot-scale 물리적 타당성 증거"라고만 주장한다 — 과대 해석하지 않은 점은 신뢰할 만하다.

---

## 7. Ablation 분석

**LIBERO-10, 동일 action 데이터, 60k step, 3개 seed (Table 4)**

| Configuration | Depth | CV | R1 | R2 | R3 | Mean±SD |
|---|---|---|---|---|---|---|
| Native Base | – | – | 92.10 | 92.52 | 94.02 | 92.88±1.01 |
| Depth only | ✓ | – | 93.00 | 94.00 | 95.00 | 94.00±1.00 |
| Cross-view only | – | ✓ | 95.05 | 91.50 | 96.07 | 94.21±2.40 |
| **Full** | ✓ | ✓ | 97.00 | 98.50 | 95.50 | **97.00±1.50** |

- Depth only +1.12, Cross-view only +1.33, **Full +4.12** → 합(+2.45)보다 큰 초가법적(super-additive) 이득. 두 목적함수가 상호 보완적(뷰 내 metric 위치 vs. 뷰 간 물리점 동일성)이라는 주장을 뒷받침.
- 단 Cross-view only의 SD가 2.40으로 크다 — 단독으로는 불안정.
- 각 seed당 **1회 평가**이므로 rollout 노이즈가 SD에 섞여 있다.

**추가 설계 ablation (Appendix)**
- 최종 depth 구성은 dense full-image가 아니라 **coordinate-query supervision**
- Cross-view tap을 layer 12에 두면 토큰 분리도 악화 → layer 15
- VGGT 깊은 layer feature를 distillation target으로 쓰는 방식은 global attention mixing 이후 **token-level discriminability가 붕괴**(Figure 5) → 외부 geometry feature 증류가 아니라 직접 감독을 택한 근거

---

## 8. 관련 연구 비교

| 축 | 대표 연구 | MVUCF와의 차이 |
|---|---|---|
| 명시적 3D | PerAct(RGB-D voxel), SpatialVLA, camera-aware VLA | 추론 시 depth/calibration 필요, 시각 인터페이스 변경. MVUCF는 배포 시 RGB-only |
| 암묵적 3D 증류 | Spatial Forcing (VGGT feature alignment) | geometric prior는 옮기지만 **metric depth 복원성도, same-point 대응도 직접 강제하지 않음**. MVUCF는 둘 다 action-facing grid에서 직접 감독 |
| Cross-view 일관성 | Selfi (reprojection-aligned feature adapter) | reconstruction/NVS/pose용이며 action-facing 표현이 아님 |
| 멀티뷰 진단 | El Banani et al. 2024 | 불일치를 진단만 하고 fusion 메커니즘 미제시 |
| 동시대 | Evo-Depth (2026) | LIBERO-Plus 69.6으로 절대값 우위, 다만 경량 모델 + 다른 백본 |

MVUCF의 포지션은 "**두 전제조건(metric grounding + cross-view consistency)을 하나의 프레임워크로 통합**"이다. 기존 연구는 이를 분리해 다뤘고, 백본 shaping 메커니즘과 표현 타깃이 서로 달라 모듈식 결합이 어려웠다는 것이 저자들의 논거.

---

## 9. 한계 및 미해결 문제

1. **정확한 학습 시점 calibration 가정** — 저자들 스스로 명시. 손상되거나 노이즈가 큰 calibration에 대한 강건성은 미검증.
2. **Camera perturbation 이득이 미미(+3.2)** — 학습 시 카메라 구성이 표현에 내재화되어 viewpoint 변화에 오히려 종속될 수 있다. 멀티카메라 리그가 바뀌면 재주입이 필요할 가능성.
3. **RoboTwin은 6-task diagnostic suite** — full benchmark leaderboard가 아니며, 2 round가 동일 체크포인트라 training variance 추정이 불가.
4. **실물 실험 규모가 작다** — 2 task × 30 trial, CI 중첩. 방향성 증거 이상은 아니다.
5. **Geometry injection 비용** — 50k step, 약 18시간, 2–4×H100. "추론 비용 0"이지만 **학습 파이프라인 비용은 결코 0이 아니며**, replay로 depth/calibration을 재수집해야 한다.
6. **단일 백본(GR00T-N1.6) 검증** — `T_img→grid`가 front-end에 이식 가능하다고 논증하지만 π₀, OpenVLA-OFT 등에서의 실증은 없다.
7. **코드/가중치 미공개** (v1 기준) — 전처리 합성 매핑처럼 재현이 까다로운 부분이 많은데 공개 정보가 제한적.

---

## 10. 총평

**강점**
- 문제 정의가 진단(Figure 2/6)에서 출발해 방법·결과로 이어지는 **evidence chain이 매우 깔끔**하다. "native 상태의 cross-view retrieval은 0.4%로 랜덤 수준"이라는 수치는 그 자체로 인용 가치가 있다.
- Base와 Ours가 action 데이터·헤드·스케줄·체크포인트 규칙·추론 그래프를 전부 공유하는 **엄격한 matched-pair 통제**. 사전 지정 terminal checkpoint 사용도 신뢰도를 높인다.
- **배포 비용 증가 0**이라는 실용적 매력. 추가 센서·모듈·FLOPs 없이 RGB-only 그래프를 그대로 유지한다.
- Project-before-preprocess 좌표 매핑은 사소해 보이지만 실제로 라벨 오차를 좌우하는 엔지니어링 통찰.
- 한계를 스스로 명시(Camera perturbation, calibration 의존, pilot 규모)하는 정직한 서술.

**약점**
- LIBERO 절대 성능은 이미 포화 구간이고, LIBERO-Plus 절대값은 OpenVLA-OFT/Evo-Depth에 미달. 기여는 "SOTA"가 아니라 "matched 개선"으로 읽어야 한다.
- RoboTwin 결과가 가장 인상적(+23.3)이지만 6-task 부분집합이라 일반화 주장에 제한이 있다.
- Ablation이 seed당 1회 평가라 통계적 여유가 크지 않다.

**평점**: 방법론적 참신성보다 **통제 품질과 실용성**이 돋보이는 논문. "training-only geometry injection"은 앞으로 멀티카메라 VLA의 기본 레시피가 될 여지가 있다.

---

## 11. 🔥 예상 날카로운 질문 모음

> ❓ **Spatial Forcing(VGGT feature alignment)과 본질적으로 무엇이 다른가?**
> Spatial Forcing은 사전학습된 geometry feature에 VLA state를 정렬시킨다. 그러나 Figure 5가 보이듯 VGGT의 **깊은 layer는 global attention mixing 이후 token-level 판별력을 잃어**, contact localization에 필요한 patch identity를 보존하지 못한다. MVUCF는 외부 feature를 타깃으로 삼지 않고 metric depth 복원성과 same-point 대응을 **action-facing grid에 직접 감독**한다.

> ❓ **Geometry를 주입한 뒤 백본을 freeze하는데, action 학습 중 그 정보가 소실되지 않는가?**
> 오히려 freeze가 보존 장치다. Base와 Ours 모두 action 학습 시 VLM을 freeze하므로, 두 조건의 차이는 "frozen된 표현이 무엇인가"뿐이다. Figure 7의 action-to-image cross-attention이 조작 대상 주변으로 더 집중되는 것이 정보가 action pathway에 도달했다는 간접 증거.

> ❓ **왜 dense depth 예측이 아니라 coordinate query인가?**
> Appendix Table 8이 최종 구성이 dense full-image가 아님을 명시한다. 연속 좌표 질의는 (1) grid 이산화 오차 없이 **sub-token 위치**를 감독할 수 있고, (2) 뷰당 3072 query 중 절반을 근거리 표면·depth 불연속에 집중시켜 manipulation workspace에 예산을 몰아줄 수 있다. absolute position 입력을 빼고 sub-cell phase만 준 것도 grid 좌표 암기를 막기 위한 설계.

> ❓ **σ=0.75의 Gaussian soft target은 임의적이지 않은가?**
> 저자 논거는 sub-token projection 오차 흡수다. 재투영은 연속 좌표를 내놓지만 토큰은 이산 격자이므로, 단일 cell에 확률을 몰면 라벨 노이즈를 그대로 학습한다. 동시에 spatial-ring negative를 써서 "가깝지만 틀린" patch가 등가로 붕괴하는 것을 막는다. 다만 σ에 대한 민감도 분석은 제시되지 않았다.

> ❓ **LIBERO 98.9%는 이미 포화된 숫자인데 의미가 있나?**
> 평균만 보면 +1.5로 작지만, 이득이 long-horizon LIBERO-10에 +4.1로 집중된다. 진짜 주장은 LIBERO가 아니라 **LIBERO-Plus(+22.4)와 RoboTwin(+23.3)** — 즉 in-domain 포화 상태에서도 강건성과 난이도 높은 접촉 과제에서 큰 차이가 난다는 것.

> ❓ **Camera perturbation 이득이 +3.2뿐인 것은 방법의 실패 아닌가?**
> 부분적으로 그렇다. 저자 해석은 geometry injection이 **학습 시점 카메라 파라미터를 표현에 내재화**하기 때문이라는 것. 이는 곧 "카메라 리그가 고정된 배포 환경"에는 잘 맞지만 뷰포인트가 변하는 환경에는 재주입이 필요할 수 있다는 뜻이다. Calibration augmentation이 자연스러운 후속 연구 방향.

> ❓ **RoboTwin 6 task는 유리한 것만 고른 게 아닌가?**
> 저자들은 학습 **전에** action primitive(Touch/Move-and-Place/Contact)로 후보를 분류하고 각 family에서 명확한 2개씩 고정했으며, 관측된 gain으로 고르지 않았다고 명시한다. 그럼에도 `turn_switch`(11.0)처럼 여전히 낮은 task를 포함시킨 점은 선택 편향이 크지 않다는 방증.

> ❓ **layer 8–15 범위는 어떻게 정당화되는가?**
> VLM probing에서 metric depth 정보가 깊은 블록에 집중되었고, layer 12에 cross-view 목적을 걸면 학습 후 토큰 분리도가 악화되었다. layer 15는 action module이 소비하는 최종 hidden spatial grid이므로 supervision tap으로 선택했고, 신호가 action-facing 블록 전체에 전파되도록 연속 상위 절반(8–15)을 업데이트했다. **downstream 성능으로 고르지 않았다**는 점을 강조한다.

> ❓ **다른 VLA(π₀, OpenVLA-OFT)에도 이식 가능한가?**
> 원리적으로는 가능하다. `T_img→grid`는 front-end가 유도하는 결정론적 좌표 맵일 뿐이며, 표준 ViT라면 resize/crop + patch-index 변환으로 축약되고 pixel-unshuffle이 없으면 해당 인자를 항등원으로 두면 된다. 다만 **실증은 GR00T-N1.6 하나뿐**이라 이식성 주장은 아직 논증 수준.

> ❓ **"추가 inference FLOPs 0"은 과장이 아닌가?**
> 배포 그래프 기준으로는 사실이다. auxiliary head, depth 입력, calibration이 전부 제거되고 파라미터도 추가되지 않는다. 다만 **학습 측면에서는 50k step × ~18시간 × 2–4 H100의 별도 스테이지와 depth/calibration 재수집(replay)이 필요**하므로, 총소유비용 관점에서는 결코 공짜가 아니다.

---

## 12. 재현/후속 연구 체크리스트

**재현 시 반드시 확인할 것**
- [ ] `T_img→grid`가 letterboxing → resize → crop → smart-resize → patchify → pixel-unshuffle **전 과정을 정확히 합성**하는지 (여기서 어긋나면 라벨이 전부 틀어진다)
- [ ] 재투영 필터 3종: 이미지 경계, target depth ∈ [0.05, 5.0] m, z-buffer `|ΔZ| < 0.10 m`
- [ ] Query 샘플링 비율(uniform 50% / 근거리·불연속 50%, 뷰당 3072)
- [ ] Loss 가중치: depth `0.5/1.25/1.0/0.1/0.05`, cv `1/0.1/0.1`, σ=0.75, margin γ=0.2, 외곽 가중치 annealing
- [ ] Terminal checkpoint 고정(50k/60k/180k), success 기반 선택 금지
- [ ] 실물 depth는 유효 측정 보존 + invalid hole만 completion, validity mask 유지

**유망한 후속 방향**
1. **Calibration 강건화** — noisy/self-calibration 조건에서의 주입, 또는 calibration augmentation으로 camera perturbation 이득 확대
2. **백본 이식성 검증** — π₀, OpenVLA-OFT, RDT 등 서로 다른 front-end에서 `T_img→grid` 일반화
3. **Injection 비용 절감** — 50k step을 줄이거나 지속적/온라인 주입으로 전환
4. **3+ 카메라, 동적 리그** — 현재는 head+wrist 구성. 카메라 수/배치가 변할 때 field가 유지되는지
5. **시간 축 확장** — 현재 field는 프레임 단위 공간 구조. temporal correspondence(모션·물체 추적)까지 주입하면 contact-rich long-horizon에서 추가 이득 가능
6. **RoboTwin full benchmark** 및 더 큰 실물 평가로 일반화 주장 강화

---

<!-- VERIFIED: pdf -->
