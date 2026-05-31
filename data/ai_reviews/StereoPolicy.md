# StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception

> **한 줄 요약**: Stanford·Northwestern·Lambda 팀이 제안한 calibration-free, reconstruction-free stereo perception 모듈. 좌/우 영상을 사전학습된 2D 인코더로 독립 처리한 뒤 **2-layer / 8-head Stereo Transformer**(2D RoPE 적용)로 융합해 128-d 표현을 만들고, Diffusion Policy 및 VLA(pi_0.5, GR00T-N1.5)의 head에 plug-in 형태로 결합한다. 5개 real-world tabletop task에서 **평균 59.0%** 로 RGB(42.0), RGB-D(41.0), RGBD-3DDA(45.0), point cloud(14-27), multi-view(44.0) baseline을 모두 능가했다.

---

## 1. 배경 및 동기

- 최신 imitation learning은 monocular RGB만으로 강한 visuomotor policy를 달성했지만, **단안 입력은 본질적으로 depth·공간 인식이 약하다.**
- 대안으로 등장한 RGB-D, point cloud, multi-view 입력은 **calibration·3D 재구성·센서 비용**을 추가로 요구해 scalability에 약점이 있다.
- 인간 시각은 **두 눈**으로 거리·기하를 추론한다. 이 paper의 가설은 "stereo image pair만으로도 명시적 3D 재구성 없이 manipulation policy가 깊이·공간 추론을 implicit하게 학습할 수 있다"는 것이다.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처
- 입력: 동기화된 좌/우 RGB 한 쌍.
- 각 영상은 동일한 사전학습 2D encoder(예: ResNet18 / DINOv2 / OpenCLIP-B/16 / SigLIP-SO400M/14 / EdgeNeXt-S)를 통과.
- **Stereo Transformer**:
  - 2 layers · 8 heads · 128-d latent
  - 각 layer는 self-attention + cross-attention + MLP
  - query·key projection에 **2D RoPE** 적용해 cross-view correspondence 학습 중에도 공간 정보 보존
- 출력 128-d feature는 그대로 Diffusion Policy head(StereoPolicy-DP) 또는 VLA action head(StereoPolicy-VLA, e.g. GR00T-N1.5 / pi_0.5)에 전달.

### 2.2 핵심 설계 결정
- **Calibration-free**: 카메라 외부/내부 보정 불필요.
- **Reconstruction-free**: depth map, point cloud, disparity map을 explicit하게 만들지 않음 — 모두 fused token에 implicit하게 인코딩.
- **Plug-in**: 정책 아키텍처와 독립적인 perception 모듈로 결계 정의.

### 2.3 학습 세부
| 항목 | StereoPolicy-DP | StereoPolicy-VLA |
|---|---|---|
| Backbone | ResNet18 (기본) | GR00T-N1.5 / pi_0.5 |
| Batch | 64 | 128 (pi_0.5) |
| LR | 1e-4 | 1e-4 / 2.5e-5 |
| Epochs/Steps | 500-1000 epochs | 60K / 80K steps |
| GPU | - | 8x H100 |

---

## 3. 실험 결과

### 3.1 Real-World Tabletop (Table 1, 5 tasks · 20 trials each)

| Method | Banana PnP | Toast Insert | Plastic Cup | Steel Cup | Glass Cup | **Avg SR** |
|---|---|---|---|---|---|---|
| RGB | 12/20 | 7/20 | 12/20 | 10/20 | 1/20 | 42.0% |
| RGB-D | 14/20 | 8/20 | 11/20 | 8/20 | 0/20 | 41.0% |
| RGBD-3DDA | 13/20 | 9/20 | 13/20 | 10/20 | 0/20 | 45.0% |
| PCD-PointNet | 7/20 | 0/20 | 5/20 | 2/20 | 0/20 | 14.0% |
| PCD-DP3 | 11/20 | 3/20 | 8/20 | 5/20 | 0/20 | 27.0% |
| Multi-View | 13/20 | 8/20 | 13/20 | 9/20 | 1/20 | 44.0% |
| **StereoPolicy-DP** | **16/20** | **12/20** | **15/20** | **13/20** | **3/20** | **59.0%** |

- 모든 5개 task에서 1위 또는 공동 1위.
- Glass Cup(투명 객체)에서 다른 모든 입력 modality가 0-1/20인데 StereoPolicy만 3/20 — depth sensor의 약점(투명·반사)을 stereo가 보완.

### 3.2 Simulation (Table 2 발췌)

| Benchmark / Task | Demos | RGB | RGBD-3DDA | PCD-DP3 | **StereoPolicy-DP** |
|---|---|---|---|---|---|
| RoboCasa ToolHang | 100 | 53.2 | - | 40.2 | **94.1** |
| RoboCasa ToolHang | 200 | 26.2 | - | - | **57.1** |
| RoboMimic Square | 200 | 94.3 | 94.2 | - | **96.2** |
| OmniGibson Strawberry | 100 | 59.3 | 74.4 | - | **82.8** |

- 특히 RoboCasa ToolHang 100-demo에서 RGB 53.2% → 94.1% (약 +41pp) — low-data·정밀삽입 시나리오에서 stereo의 이득이 두드러짐.

### 3.3 StereoPolicy-VLA (pi_0.5 mobile manipulation)

| Demos | RGB | **StereoPolicy-VLA** |
|---|---|---|
| 30 | 48.71% | **51.72%** |
| 300 | 70.31% | **74.40%** |

- 대규모 VLA backbone(pi_0.5)에서도 stereo 모듈이 평균 ~+4pp 추가 향상 → plug-in 효과가 large VLA에서도 유지됨을 입증.

---

## 4. 어블레이션

### 4.1 Vision Encoder (Figure 10, ToolHang 100 demos)
- Stereo Transformer 제거: 0.94 → 0.85 (약 -9pp).
- DINOv2 concat: external view에는 도움, wrist view에는 손해.
- 큰 backbone(OpenCLIP, SigLIP): low-data regime에서 ResNet18 대비 큰 향상.
- Multi-scale(EdgeNeXt-S, ResNet18+FPN): stereo fusion 품질에 유리.

### 4.2 Stereo Baseline-to-Distance 비율 (Figure 9)
- 최적 비율 r ∈ [0.09, 0.13].
- 2 cm baseline + >0.9 m 거리(r<0.03): disparity가 너무 약함.
- 10 cm baseline + 0.6 m 거리(r>0.17): overlap 부족, 기하 일관성 손실.
- 즉 "사람 눈 간격(약 6 cm)에 가까운 비율"이 정책 학습에 최적.

---

## 5. 한계

| # | 한계 | 코멘트 |
|---|---|---|
| 1 | 학습 규모 | tabletop / 모바일 매니퓰레이션 위주 — 산업 스케일 검증 부재 |
| 2 | 투명·반사 객체 | StereoPolicy도 절대 성공률은 낮음 (Glass Cup 3/20=15%) — 향상이지 해결은 아님 |
| 3 | 조명 민감도 | 강한 반사·역광에서 불안정 |
| 4 | Parameter 미공개 | Stereo Transformer 외 전체 파라미터/FLOPs/latency 보고 없음 |
| 5 | 코드 | project page만 공개, GitHub 공식 release는 paper 시점 부재 |
| 6 | 확장 | Droid 등 large robotics dataset stereo pair에 대한 scaling 실험 미수행(future work으로 명시) |

---

## 6. 의의와 위치

- **"Stereo = 가장 저렴한 3D modality"** 명제를 정책 학습 관점에서 강하게 재확인. depth sensor·calibration 없이도 5~50pp 단위의 향상.
- Diffusion Policy와 VLA(pi_0.5, GR00T-N1.5) 모두에 통한다는 점 → 본 모듈은 **architecture-agnostic perception primitive**.
- 향후 large-scale stereo robot dataset(Droid 등)과 결합되면 "vision encoder pretraining"의 표준이 stereo로 옮겨갈 가능성이 있다.

---

## 7. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | 2-layer transformer만으로 disparity가 정말 학습되는가? | Stereo Transformer 제거시 ToolHang 0.94→0.85. 명시적 disparity head 없이도 cross-attention + 2D RoPE가 implicit하게 학습. |
| 2 | RGB-D보다 stereo가 좋은 이유? | RGB-D는 active depth가 투명/반사에 실패 + sparsity. Stereo는 좌우 RGB만 사용해 이 모달리티의 단점이 없음. Glass Cup에서 3 vs 0/20이 그 증거. |
| 3 | r=0.09-0.13 최적이라는 결과의 일반화는? | 0.6-1.0 m tabletop 기준. mobile manipulation 등 다른 working distance에서는 카메라 베이스라인 재설계 필요. |
| 4 | VLA backbone(pi_0.5)에서 RGB 대비 +4pp는 marginal하지 않은가? | Large VLA는 대규모 pretraining으로 RGB만으로도 강함. 그 위에서의 +4pp는 marginal이 아닌 추가 ceiling 깨기로 해석. 단 +4pp가 noise일 가능성도 — seed/variance 보고 한계. |
| 5 | 왜 explicit disparity supervision을 안 썼나? | Stereo GT가 있는 robot dataset이 희소. 본 paper의 contribution은 "GT 없이도 implicit 학습 가능" 자체. |
| 6 | calibration-free가 정말로 성립? 카메라 변경시? | 좌/우 영상 token이 동일 encoder에 통과되어 표현 공간이 공유됨. 단, baseline·focal length가 크게 바뀌면 fine-tuning이 필요할 가능성 — ablation 미수행. |

<!-- VERIFIED: pdf -->
