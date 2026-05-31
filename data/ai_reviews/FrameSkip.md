# FrameSkip: Learning from Fewer but More Informative Frames in VLA Training

> **한 줄 요약**: VLA architecture, action head, training objective, inference 어디에도 손대지 않고 오직 dataloader 안에서 4가지 점수(action variation / visual-action coherence / task-progress prior / gripper transition)로 약 20%의 frame만 골라 학습시키는데, LIBERO/SimplerEnv/RoboCasa-GR1 매크로 평균 success를 66.50% → 76.15%로 끌어올린 data-layer 프레임웍.

---

## 1. 배경 및 동기

### 기존 VLA 학습의 구조적 비효율
- 일반적인 robot demonstration은 30Hz 전후로 수집되며, 한 trajectory는 수백~수천 프레임에 달함.
- 이를 단순히 모든 프레임을 동등한 가중치로 학습시키면, **dwell / approach / 정지** 구간(전체의 70~80%)이 supervision을 압도하고, 실제로 task success를 가르는 **contact, gripper transition, fine alignment** 순간은 매우 적은 supervision만 받는다.
- 결과적으로 모델은 "평균 trajectory"를 학습하게 되고, decisive moment에서의 정밀도가 떨어진다.

### 핵심 질문
- **각 trajectory에서 어떤 프레임이 "정보적(informative)"인가?**
- **VLA 자체에 손을 대지 않고 frame-level supervision balance만 바꿔도 성능이 오르는가?**

📌 [Figure 1 삽입] — Full-frame 학습 vs FrameSkip 선택 프레임 비교

---

## 2. 방법론 심층 분석

### 2.1 위치 설정: "Dataloader-only" 디자인

FrameSkip의 가장 강한 디자인 결정은 **VLA architecture, action head, objective, inference에 손을 대지 않는다**는 점이다. 모든 처리는 오프라인에서 frame index 선택으로 수렴한다.

이는 두 가지 의미가 있다:
1. 어떤 기존 VLA(diffusion, flow matching, autoregressive)에도 그대로 plug-in 가능
2. 비교가 매우 깨끗함 — 동일 모델, 동일 hyperparameter, 단지 dataloader만 교체

### 2.2 네 가지 Frame Scoring Signal

각 프레임 $t$에 대해 네 가지 score를 결합하여 최종 ranking을 결정한다.

| Score | 정의 | 의도 |
|-------|------|------|
| Action Variation Importance | 인접 timestep action vector의 L2 norm 차이 | 가속/방향 전환 등 local dynamics가 큰 시점 우선 |
| Visual-Action Coherence | DINOv2 visual feature 변화 vs action 변화의 일관성 | 화면이 거의 그대로인데 action만 큰 (혹은 반대) 비정상 supervision을 보정 |
| Task-Progress Importance | 1-D Gaussian Mixture Model로 trajectory를 stage로 분해 | "이 trajectory의 결정적 phase는 어디인가" 의 stage-aware prior |
| Gripper-Transition Preservation | gripper 상태가 binary로 바뀌는 시점 explicit 보존 | manipulation의 핵심 순간(잡는 순간, 놓는 순간)을 강제 포함 |

> ❓ **예상 질문**: 네 가지 점수를 어떻게 결합하는가? Learned vs heuristic?
> **답변**: 학습 없이 weighted sum + threshold 기반 selection. 핵심은 "어떤 weight를 쓰든 gripper transition은 hard-include" 라는 제약. Hyperparameter sensitivity는 ablation에서 다룸.

> ❓ **예상 질문**: DINOv2를 visual encoder로 사용하면 추가 계산비용이 발생하는데?
> **답변**: 오프라인 사전계산. 학습 시 cost는 0, 한 번만 score 파일을 생성하면 이후 모든 epoch에 재사용 가능.

### 2.3 Frame Retention Ratio

- **20% retention**이 main configuration.
- 즉 평균 100 프레임 trajectory에서 20 프레임만 학습 신호로 사용.
- 학습 step 수는 baseline과 동일하게 맞추므로, **각 프레임당 weight가 5배**가 되는 효과.

> ❓ **예상 질문**: 20%가 magic number인가? 10%나 30%는?
> **답변**: 논문에서 retention sweep 진행. 너무 낮으면 trajectory continuity 손상, 너무 높으면 imbalance 해결이 부족. 20% 부근이 sweet spot.

---

## 3. 백본 및 실험 설정

### 3.1 Base VLA: StarVLA

- **Understanding expert**: Qwen3-4B-VL-Instruct
- **Action expert**: Diffusion Transformer(DiT) + flow matching objective
- 즉 FrameSkip은 **flow-matching VLA**에 적용된 실험이지만, 디자인상 backbone-agnostic.

### 3.2 학습 인프라

| 항목 | 값 |
|------|-----|
| Hardware | 8 × NVIDIA H100 |
| Distributed | DeepSpeed ZeRO-2 |
| Global batch size | 128 |
| Steps (LIBERO) | 30K |
| Steps (SimplerEnv) | 60K |
| Steps (RoboCasa-GR1) | 100K |
| Frame retention | 20% |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 3)

| Suite | Full-frame | **FrameSkip** | Δ |
|-------|-----------|--------------|----|
| Spatial | 97.8 | **98.6** | +0.8 |
| Object | 98.8 | **99.0** | +0.2 |
| Goal | 97.4 | **98.2** | +0.8 |
| Long | 92.0 | **93.8** | +1.8 |
| **Avg** | **96.5** | **97.4** | **+0.9** |

- **Long horizon에서 가장 큰 향상(+1.8)** — 길어질수록 frame imbalance가 누적되어, 이를 보정하는 FrameSkip 효과가 두드러진다.

### 4.2 SimplerEnv WidowX (Table 2)

| Task | **FrameSkip** |
|------|--------------|
| Put Spoon on Towel | 90.63 |
| Put Carrot on Plate | 54.17 |
| Stack Green Block on Yellow | 45.59 |
| Eggplant in Yellow Basket | 95.83 |
| **Average** | **71.55** (vs 55.2 baseline) |

- **Δ = +16.4** — 세 벤치마크 중 가장 큰 절대 향상.
- Stack(45.59) 처럼 contact 정밀도가 핵심인 task가 base보다 훨씬 더 큰 폭으로 좋아진 것이 시사적.

### 4.3 RoboCasa-GR1 (Table 1)

| Task (대표) | **FrameSkip** |
|------------|--------------|
| PnP Bottle | 74.0 |
| PnP Can | 80.0 |
| PnP Cup | 46.0 |
| PnP Milk | 60.0 |
| **24-task Macro Avg** | **59.5** (vs 47.8) |

- **Δ = +11.7** — 24개 task 평균이라는 점에서 일반화 신뢰도 높음.

### 4.4 매크로 평균

- Full-frame: **66.50%**
- FrameSkip: **76.15%**
- **Δ = +9.65%p**, 그러면서도 학습 데이터는 80% 줄어듦.

---

## 5. Ablation

### 5.1 네 가지 score 제거 실험

논문은 4가지 score 중 하나씩 제거하며 다음 경향을 보고:
- **Gripper-transition** 제거 시 성능 가장 큰 폭으로 떨어짐 → manipulation에서 가장 결정적인 신호.
- **Visual-action coherence** 제거 시 SimplerEnv처럼 visual variation이 큰 환경에서 큰 손해.
- **Task-progress prior** 제거 시 LIBERO-Long에서 손해 — long-horizon stage 인식이 사라지기 때문.

### 5.2 Retention Ratio Sweep

10% → 20% → 40% 의 곡선에서 20%가 plateau의 시작점. 그 이상은 marginal.

---

## 6. 관련 연구와의 위치

| 분류 | 대표 연구 | 차이점 |
|------|----------|--------|
| Curriculum learning | 다양 | trajectory 단위 vs frame 단위, FrameSkip은 frame-grained |
| Token pruning (VLA inference) | EfficientVLA 등 | inference 단계 효율화, FrameSkip은 학습 단계 |
| Active learning | Coreset 등 | trajectory 선택, FrameSkip은 trajectory 내부 선택 |
| Demonstration filtering | 일반 IL | 좋은 trajectory 선택, FrameSkip은 좋은 frame 선택 |

핵심 차별점: **"내부(intra-trajectory) frame importance"** 라는 새로운 축을 정립했다는 점.

---

## 7. 한계 및 미해결 문제

### 방법론적
1. **Score weight가 수동 hyperparameter**: 학습 가능한 score function이 아닌 hand-crafted. Task domain이 바뀌면 재조정 필요할 수 있음.
2. **DINOv2 의존**: visual-action coherence 계산에 DINOv2가 필요 — 모든 dataset에 동일한 visual encoder를 적용해도 좋은 신호인지 보장 없음.
3. **Flow-matching backbone 단일**: 실험은 StarVLA(flow-matching) 위에서만 진행 — diffusion policy, autoregressive(OpenVLA) 위에서도 동일한 게인이 나오는지 외삽 필요.

### 평가
1. **Real-world 실험 부재 또는 약함**: 모두 시뮬레이션 벤치마크.
2. **데이터 노이즈 강건성**: noisy demonstration에서 score 자체가 잘못 잡힐 가능성 — 예컨대 잡음 큰 action variation을 "informative"로 잘못 인식할 수 있음.
3. **Trajectory 길이 분포 영향**: 매우 짧은 trajectory(10 프레임 이내)에서는 20% retention이 2 프레임 — 의미 있는 frame 선택이 가능한가?

### 해석
- 매크로 평균 +9.65%p의 절대다수가 baseline이 약했던 RoboCasa(+11.7), SimplerEnv(+16.4)에서 발생. 이미 강한 LIBERO에서는 +0.9에 머묾.
- 즉 FrameSkip의 이득은 **"기존 base가 underfit인 영역"에서 가장 크다** — frame imbalance가 가장 심한 곳에서 효과가 크다는 자연스러운 해석.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — frame-level supervision balance라는 새로운 축. 단순하지만 영리함. |
| **Technical depth** | ★★★☆☆ — 4가지 score 모두 heuristic. 학습 가능한 buisness logic 부재. |
| **Experimental rigor** | ★★★★☆ — 3개 벤치마크, 24-task RoboCasa 포함, 일관된 게인. |
| **Practical impact** | ★★★★★ — dataloader만 교체하면 됨. 학습 데이터 80% 감소 + 성능 향상. ROI 최고. |
| **Writing/Clarity** | ★★★★☆ — 디자인 결정의 동기가 명확. |

**강점**: VLA 학습에서 "supervision balance"라는 개념을 정립. Plug-in 가능성과 실제 코드 공개(github.com/ZGC-EmbodyAI/FrameSkip)로 확산성 높음. **약점**: score function이 hand-crafted, backbone 다양성 실험 부족.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 같은 step 수로 5배 적은 unique frame을 5번씩 보게 되는데, 이는 overfitting 아닌가? | LIBERO처럼 작은 domain에서는 risk. 하지만 macro-avg가 모두 오른 것은 informative frame 반복이 noisy frame 반복보다 generalization에 더 도움됨을 시사. |
| 2 | Stack Block(45.59%) 처럼 어려운 task에서 baseline 대비 향상폭이 정확히 얼마인가? | 논문 본문에서는 task별 baseline 비교를 모든 task에 공개하지 않음 — average만 명확히 비교 가능. |
| 3 | DINOv2 score가 visual occlusion이 심한 시점에서 오작동할 가능성은? | 가능. Gripper-transition score가 hard-include 안전장치 역할. |
| 4 | OpenVLA(autoregressive) 위에서도 같은 게인이 나오는가? | 미실험. Backbone 일반화 주장은 design-level claim에 머무름. |
| 5 | Inference speed는? | 학습 only 영향, inference 동일 (디자인상 변화 없음). |
| 6 | 학습 시간은 baseline의 몇 %? | Step 수 동일이면 동일. 하지만 dataloader가 가벼워져 I/O는 줄어듦. |
| 7 | Score 계산의 one-time cost는? | DINOv2 pass + GMM fit, 한 번만 수행. dataset 크기에 비례하나 매우 저렴. |
| 8 | LIBERO-Long에서 +1.8 향상은 통계적으로 유의한가? | 표준편차나 seed 실험이 본문에 명시되지 않음 — 재현 시 확인 필요. |

<!-- VERIFIED: pdf -->
