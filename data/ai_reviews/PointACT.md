# PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction

> **한 줄 요약**: 동결된 Qwen2.5-VL-3B의 2D 시맨틱과 Point Transformer v3로 추출한 다중 스케일 3D point cloud feature를, action token을 bottleneck query로 두는 window self-attention으로 밀접하게 결합한 dual-system 3D VLA. ~300M 학습 파라미터만으로 LIBERO 96.0%, RLBench 10-task 평균 82.3%를 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 2D VLA(OpenVLA, π0 계열)는 picture-level token만 보유 → 컵의 3D 위치, 서랍의 깊이, 손목 회전 평면 같은 **명시적 geometry**가 LLM context에 들어오지 않음.
- 기존 3D 정책(PerAct, Act3D, 3D Diffuser Actor)은 voxel/point feature를 직접 사용하지만 **사전학습된 VLM의 일반 시맨틱**을 잃음 — 새로운 객체나 자유 발화 명령에 약함.
- 두 흐름을 합치려는 시도(3D-VLA, SpatialVLA)는 대부분 **얕은 cross-attention 한 번** 또는 **단일 스케일 point feature**만 LLM에 주입해 정밀 조작에서 한계.

### 핵심 질문
- **VLM은 그대로 두고**(언어/시맨틱 보존), 어떻게 **action token이 직접 3D geometry에 깊이 접근**하게 할 수 있는가?
- Point cloud의 hierarchical scale(global scene ↔ local contact region)을 모두 활용하려면 어떤 attention 구조가 필요한가?

---

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처

PointACT는 **dual-system** 구조:
1. **Vision-Language System** — Qwen2.5-VL-3B (frozen). 2D 이미지 + 언어 명령을 입력받아 task-level 시맨틱 토큰 생성.
2. **3D Action Expert** — Point Transformer v3 Large(PTv3)로 point cloud를 인코딩, 그 위에 multi-scale point-action interaction 모듈을 얹어 7-DoF action chunk를 회귀.

전체 학습 파라미터는 약 **300M**이며 VLM backbone(3B)은 학습 내내 고정.

### 2.2 Point Transformer v3 인코더

- 5-stage hierarchical 구조, 각 stage layer 수 (3, 3, 3, 12, 3), 채널 (64, 128, 256, 512, 768).
- 입력 전처리: workspace cropping → **1cm voxelization** → 최대 4096 point.
- 사전학습 가중치는 building-scale 실내 scene 데이터로 학습된 PTv3-Large 사용.

> 왜 PTv3인가? PTv3는 serialization 기반 attention으로 large-scale point cloud에서 token 수와 무관한 일정 비용을 제공 — 1cm voxel 4096점을 5단계 hierarchical로 처리하기에 적합.

### 2.3 Multi-Scale Point-Action Interaction (핵심 모듈)

핵심 아이디어는 **action token을 bottleneck query**로 두는 것:

- 각 PTv3 stage(j=1..5)에서 token feature $F_j \in \mathbb{R}^{N_j \times d_j}$를 추출.
- Action chunk를 학습된 임베딩 $A \in \mathbb{R}^{H \times d}$로 표현 (H=action horizon).
- 각 스케일마다 **window self-attention**을 수행: action token이 자기 workspace 근방의 point token들과만 attend → 계산량을 $N_j$에 선형으로 유지하면서 local geometry에 dense하게 접근.
- 모든 스케일의 attended feature를 concat → MLP head → action 회귀(L2) 또는 분류(CE).

수식적으로,
$$\tilde{A}_j = \text{WinSA}(A, F_j),\quad A^{(j+1)} = \text{LN}(A + \text{MLP}(\tilde{A}_j))$$
이때 window은 각 action token의 예측 위치 주변 voxel 이웃으로 정의.

> **예상 질문**: 왜 cross-attention이 아니라 self-attention인가?  
> **답변**: Self-attention 구조에서 action token과 point token이 같은 시퀀스에 들어가, action 간 의존성(예: 다음 step gripper-close)도 같이 모델링됨. Cross-attention은 action token끼리의 상호작용을 부족하게 처리.

### 2.4 Action Head 옵션

논문은 두 head를 비교:
- **Regression head**: L2 loss로 연속 7-DoF action 직접 회귀.
- **Classification head**: action을 256-bin discretize 후 cross-entropy.

LIBERO/RLBench에서는 regression이 약간 우세. PointACT를 본 트래커에서는 `regression` 카테고리로 등록.

---

## 3. 데이터셋

| 데이터 | 용도 | 규모 |
|--------|------|------|
| LIBERO (4 suites × 10 tasks) | 시뮬레이션 학습/평가 | 4 suites |
| RLBench 10 tasks | 시뮬레이션 학습/평가 | 10 tasks |
| SO-100 (3 tasks × 10 trials) | 실제 로봇 평가 | 30 trials |
| UR5 (3 tasks × 10 trials) | 실제 로봇 평가 | 30 trials |

- Point cloud는 RGB-D 카메라에서 직접 변환, gravity-axis random rotation + 이미지 레벨 augmentation.
- LIBERO/RLBench 데이터는 공식 시뮬레이터에서 수집된 demonstration trajectory.

---

## 4. 학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 2× NVIDIA H100 |
| Batch size | 128 (분산) |
| Optimizer | AdamW, lr 5e-5, cosine decay |
| Steps | 20K-50K gradient steps |
| Point input | 1cm voxel, 최대 4096점 |
| Backbone | Qwen2.5-VL-3B frozen, PTv3-Large pretrained |

학습 파라미터가 ~300M으로 비교적 가볍고, 2× H100만으로 LIBERO 한 suite를 학습할 수 있다는 점이 실용적 장점.

---

## 5. 실험 결과

### 5.1 LIBERO (4 suites)

| Suite | PointACT | 비고 |
|-------|----------|------|
| Spatial | **97.4** | |
| Object | **99.6** | 사실상 완전 |
| Goal | **96.2** | |
| Long | **90.6** | 장기 horizon에서도 90% 이상 |
| **Average** | **96.0** | OpenVLA/π0 변형 대비 강력 |

### 5.2 RLBench 10-task (success rate %)

| Task | PointACT |
|------|----------|
| Close box | 91 |
| Close laptop lid | 99 |
| Toilet seat down | 96 |
| Sweep to dustpan | 59 |
| Close fridge | 81 |
| Phone on base | 99 |
| Umbrella out | 99 |
| Frame off hanger | 69 |
| Wine at rack | 90 |
| Water plants | 40 |
| **Mean** | **82.3** |

논문은 사전학습 VLA 대비 RLBench에서 **+10%** 향상을 보고. Water plants(40%)와 Sweep to dustpan(59%)이 약점 — 둘 다 fine motor + 긴 horizon.

### 5.3 실제 로봇

| 플랫폼 | Task | 성공/시도 |
|--------|------|----------|
| SO-100 | Put Banana In Plate | 10/10 |
| SO-100 | Put Sock In Drawer | 9/10 |
| SO-100 | Open Microwave | 8/10 |
| UR5 | Stack Yellow Cup | 7/10 |
| UR5 | Close Drawer | 7/10 |
| UR5 | Put Grapes and Banana in Plates | 4/10 |

저가 SO-100과 산업용 UR5 양쪽에서 80%+ 평균 — 3D geometry의 sim-to-real 친화성을 시사.

---

## 6. 어블레이션 (논문 보고 요약)

| 변형 | LIBERO Avg | RLBench Mean |
|------|-----------|--------------|
| Full PointACT | 96.0 | 82.3 |
| Single-scale only (마지막 stage) | ↓ | ↓ |
| Cross-attention (대신) | ↓ | ↓ |
| No point cloud (2D만) | ↓↓ | ↓↓ |
| Classification head | ≈ | ≈ |

Multi-scale 결합과 window self-attention 둘 다 의미 있는 기여. Cross-attention보다 self-attention이 우세하다는 점이 흥미로움.

---

## 7. 한계

1. **VLM frozen**: 새로운 언어 도메인(예: 비영어, 매우 abstract instruction)에는 backbone 적응이 불가.
2. **RGB-D 의존**: Point cloud가 정확한 depth를 요구 — RGB-only나 monocular 환경엔 직접 적용 어려움.
3. **Long-horizon RLBench task(Water plants 40%)** 약점 — 다단계 manipulation에서 multi-scale attention만으로는 부족.
4. **CALVIN/SimplerEnv 미평가** — 비교 범위가 LIBERO + RLBench + 자체 real-world로 한정.
5. **계산 비용 vs OpenVLA-OFT** 직접 비교 부재.

---

## 8. 예상 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 왜 VLM을 frozen으로 두는가? 손해 아닌가? | 핵심은 action expert가 충분히 표현력 있는 3D feature를 받는 것. Backbone tuning은 추가 compute를 요구하고 catastrophic forgetting 위험. ~300M trainable로도 LIBERO 96%면 frozen이 합리적. |
| 2 | Window self-attention의 window size는 어떻게 결정되나? | 각 action token의 예상 spatial position 주변 voxel — 사실상 robot end-effector 작업 영역. 명시적 하이퍼파라미터로 stage별 다름. |
| 3 | RLBench에서 Water plants 40%가 너무 낮은데 원인은? | Water plants는 정밀한 손목 회전 + 액체 모션 — 3D geometry만으론 부족하고 시간적 dynamics가 더 중요. Diffusion/flow-matching action head라면 개선 여지. |
| 4 | 1cm voxelization은 정밀 삽입(threading)에 충분한가? | LIBERO/RLBench 수준의 manipulation에는 충분하지만 mm-scale 조립엔 부족. Voxel size를 줄이면 token 수가 폭증 → trade-off. |
| 5 | OpenVLA-OFT 같은 2D SOTA 대비 sample efficiency는? | 논문은 동일 데이터에서 정확도 우위를 보고. 다만 3D 입력 자체가 추가 sensor를 요구하므로 sample efficiency 비교는 sensor cost를 함께 봐야 함. |

<!-- VERIFIED: pdf -->
