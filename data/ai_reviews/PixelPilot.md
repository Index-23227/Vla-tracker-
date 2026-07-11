# PixelPilot: Scalable Vision-Language-Action Models for End-to-End Autonomous Driving

> **한 줄 요약**: 주행 VLA의 3D 궤적 직접 예측을 버리고, **센서 불변(sensor-agnostic) 2D 이미지 평면에서의 계획**과 **추론 시에만 수행하는 결정론적(deterministic) 2D→3D lifting**으로 분리. 이질적 데이터셋(nuScenes+Waymo) 통합 학습이 가능해지고 ego-status에 의존하는 trivial solution을 차단. GRPO 기반 dense intermediate reward로 perception→reasoning→meta-action→planning의 인과 사슬을 강제. nuScenes open-loop L2 평균 **0.30 m** (AutoVLA 0.40 대비 25% 감소), Bench2Drive closed-loop **Driving Score 79.14 / Success Rate 58.87%**로 SOTA.

---

## 1. 배경 및 동기

- 기존 주행 VLA(OmniDrive, Orion, OpenDriveVLA, AutoVLA 등)는 2D 사전학습 VLM을 fine-tuning하여 **2D 이미지에서 3D 궤적을 직접 예측**
- 문제 1 — **데이터 확장성 붕괴**: 2D→3D 매핑이 카메라 intrinsics/extrinsics에 종속. nuScenes와 Waymo를 합치면 동일한 픽셀 좌표가 전혀 다른 3D 위치에 대응 → 심각한 공간적 모호성(spatial ambiguity)
- 문제 2 — **Trivial solution 수렴**: 3D 공간 직접 최적화는 어렵기 때문에, 모델이 시각 이해 대신 **ego-status(속도/가속도) 외삽**에 의존. 실증: SOTA VLA에서 ego-status 제거 시 L2 1.98, 이미지 제거 시 0.36 — 이미지가 거의 무시되고 있음
- 인간 운전자 유추: 숙련된 운전자는 차종과 무관하게 2D 시각 관점에서 경로를 계획하고(센서 불변 계획), 차량 제원에 대한 익숙함으로 물리 제어로 변환(센서 특정 lifting)

📌 [Figure 1] — (a) 기존 VLA의 2D→3D 직접 예측 vs (b) PixelPilot의 decoupled planning & lifting

## 2. 방법론 심층 분석

### 2.1 Decoupled Planning and Lifting Paradigm

**Planning 단계 (학습+추론)**: 멀티뷰 이미지 I와 명령 C(ego-status 포함)로부터 이미지 평면에서 autoregressive하게 순차 생성:
$$(\hat{\mathcal{B}}_{2D}, \hat{\mathcal{R}}, \hat{A}, \hat{\mathcal{T}}_{2D} \mid \mathbf{I}, \mathbf{C})$$
즉 2D 객체 검출 → bounding box 기반 reasoning → meta-action → 2D 픽셀 waypoint (식 1의 조건부 확률 분해)

**Lifting 단계 (추론 시에만)**: 계획된 2D waypoint를 local plane 가정(높이 Z=−h) 하에 homography/ray-casting으로 3D ego 좌표계에 투영:
$$\hat{\mathcal{T}}_{3D} = \Psi(\hat{\mathcal{T}}_{2D}, K, h)$$
카메라 파라미터 K만 바꾸면 다른 캘리브레이션된 차량으로 일반화 가능

### 2.2 이론적 정당화
- **Trajectory Feasibility**: 3초 단기 계획에서 전방 도로면은 local plane으로 근사 가능. nuScenes 실측 3초 도로 높이 변화 평균 0.16 m — DepthAnything의 depth 오차 7.6 m 대비 무시 가능. Local plane 가정 하에 3D 도로면과 2D 투영은 전단사(bijective) projective transformation
- **Interaction Safety**: 이미지 평면을 단순화된 Configuration Space로 간주. 동적 에이전트의 2D bbox는 Image-Space Obstacle로서 가시적 3D 충돌 위험의 **보수적 상위집합(conservative superset)** — 3D 충돌은 반드시 2D 투영 겹침을 함의

### 2.3 Reasoning with Bounding Boxes
- 기존: perception을 feature projection의 proxy로만 쓰거나, 모호한 텍스트 기술 생성
- PixelPilot: `<think> The car at [363, 239, 402, 274] is directly ahead and appears close... </think>` — reasoning이 선행 perception 출력의 **구체적 bbox 좌표를 명시적으로 참조**
- 검증 가능한 시각적 앵커 → 공간 관계 추론 → 행동/계획의 해석 가능한 인과 사슬

### 2.4 Ego-Centric Consistency Preprocessing
- 전/후방 카메라를 두 행으로 단순 배치하면 차선 정렬 불일치 발생; 좌우 반전 스티칭은 궤적 불연속 유발
- 해결: **후방 뷰 180도 회전 스티칭** → 시각적으로 연속인 도로 평면과 일관된 멀티카메라 collage 상의 궤적

## 3. 학습 전략 (Knowledge-Instilled Policy Learning)

### Stage 1: Multi-Task SFT (2 epochs)
5개 태스크: (1) Perception (2D bbox, GT는 3D bbox의 2D 투영), (2) Meta-Action (lateral: TURN LEFT/RIGHT/FORWARD; longitudinal: ACC/DEC/KEEP/STOP), (3) Planning (3초 2D 궤적), (4) Integrated Control (perception-action-planning 3단계 시퀀스), (5) Reasoning with Bounding Boxes — SFT baseline의 실패 사례에 대해 Qwen-VL-Max로 인과 추론 사슬 합성 후, 추론 포함 시 실제 성능 개선(action 수정 + L2 감소)이 확인된 샘플만 엄선

### Stage 2: GRPO 기반 Holistic Policy Optimization
- 기존 RL(AutoVLA 등)의 **최종 궤적에 대한 sparse reward**와 달리, 검증 가능한 중간 출력 전체에 dense reward 부여:
  - $R_{fmt}$: `<perception><think><action><answer>` 구조/bbox 완결성/궤적 일관성
  - $R_{percep}$: Hungarian matching 후 mean IoU
  - $R_{action}$: meta-action F1-score
  - $R_{traj}$: 픽셀 거리 10 미만 시 L1 reward 1.0 + sigmoid-scaled L2 reward $\frac{2e^{-w}}{1+e^{-w}}$ + PDMS(안전/승차감/효율)
- 자유형 reasoning에는 semantic reward를 주지 않음 → 유연성 보존, shortcut learning 방지
- Long-horizon credit assignment 완화 + SFT에서 확립한 인과 사슬 강제
- GRPO completion 수 8, 80GB GPU 8장

## 4. 실험 설정

- **Open-loop**: nuScenes(28.1k) + Waymo(23.8k)로 SFT+RL 학습
- **Closed-loop**: Bench2Drive(274.5k)로 SFT만 수행(Orion 관행 따름), 2Hz로 3초 궤적 계획, PID 컨트롤러로 차량 제어
- 베이스 모델: Qwen2.5-VL-7B; 디코딩: temperature 1.0, top-p 0.5, top-k 20
- 지표: 2D detection mAP, meta-action F1, open-loop L2/Collision/Intersection, closed-loop Driving Score/Success Rate/Efficiency/Comfortness

## 5. 주요 결과

### Perception (Table 1, nuScenes 2D detection)
| 방법 | mAP |
|---|---|
| StreamPETR | 46.5 |
| MV2D | 52.3 |
| SimPB | 54.1 |
| **PixelPilot** | **54.2** |

전용 2D 검출기를 능가 — 검출만을 위해 최적화되지 않았음에도

### Meta-Action (Table 2, F1)
Qwen2.5VL-7B(nuScenes 학습) 대비 전 카테고리 우위: forward 94.46→**96.82**, left 63.00→**75.51**, right 67.01→**75.71**, dec. 77.10→**80.19**, stop 75.00→**81.80**

### Open-Loop Planning (Table 3, nuScenes)
| 방법 | L2 Avg (m) | Collision Avg (%) | Intersection Avg (%) |
|---|---|---|---|
| OmniDrive | 0.33 | 0.30 | 3.00 |
| Imprompt-VLA | 0.30 | – | – |
| AutoVLA | 0.40 | 0.20 | – |
| **PixelPilot** | **0.30** | 0.25 | **1.77** |

(FSDrive는 비공개 데이터, AutoDrive-R2는 ego-status 운동학으로 3D 궤적을 명시적으로 계산하므로 회색 처리)

### Closed-Loop Planning (Table 4, Bench2Drive/CARLA)
| 방법 | Driving Score | Success Rate (%) | Efficiency | Comfortness |
|---|---|---|---|---|
| Orion | 77.74 | 54.62 | 151.48 | 17.38 |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 |
| **PixelPilot** | **79.14** | **58.87** | **153.26** | 38.01 |

VAD 대비 Driving Score +36.79, Success Rate +43.87

## 6. Ablation 분석

- **2단계 학습** (Table 5): SFT only 0.46 < RL only 0.49 — RL 단독으로는 구조화된 다단계 추론의 방대한 탐색 공간을 헤매지 못함. SFT가 지식 기반 제공 후 RL이 정제
- **전처리/SFT** (Table 6): ego-centric consistency 제거 시 0.43 (front view만 쓰는 0.40보다도 나쁨!), multi-task 제거 시 0.45
- **RL reward** (Table 7): $R_{traj}$ 제거 0.44, $R_{percep}$ 제거 0.41, $R_{action}$ 제거 0.40 — 모든 검증 가능 reward가 기여
- **데이터 확장성**: Waymo 제외 시 L2 +0.06 — decoupled paradigm의 스케일링 효과 실증
- **Zero-shot 일반화** (Table 8): nuScenes 학습 → Waymo zero-shot 평가에서 L2 평균 **0.40 m** (fine-tuning 없이 EMMA 0.34에 근접)
- **Ego-status vs 이미지** (Table 9, 핵심): OmniDrive는 ego-status만으로 0.36 / 이미지만으로 1.98 — 이미지 무시. PixelPilot은 **이미지만으로 0.71 < ego-status만으로 0.90** — 유일하게 역전된 경향, 시각 기반 계획 입증
- **Local plane 가정** (Table 10): 3초 내 높이 변화 0.5/1/2 m 이상 샘플에서도 2D→3D VLA와 동등 (0.30/0.31/0.34)
- **CoT vs 지연** (Table 11): full CoT로 L2 0.35→0.30, 지연 0.31→0.33 s (~3 Hz, Bench2Drive 2 Hz 제어 주기 충족)

## 7. 기존 연구와의 비교

| 축 | 기존 주행 VLA | PixelPilot |
|---|---|---|
| 궤적 예측 공간 | 3D 직접 (센서 종속) | 2D 이미지 평면 (센서 불변) |
| 2D→3D 변환 | 네트워크가 학습 | 추론 시 결정론적 투영 |
| 데이터 통합 | 카메라 설정 충돌로 곤란 | nuScenes+Waymo 등 자유 통합 |
| Perception 역할 | feature projection의 proxy | 명시적 bbox 앵커로 reasoning에 조건화 |
| RL reward | 최종 궤적에 sparse | perception→planning 전 구간 dense |
| 차량 이전 | 재학습 필요 | 카메라 파라미터 교체만 |

## 8. 의의 및 임팩트

- 주행 VLA의 근본 병목이던 **센서 종속성**을 학습 목표에서 제거해 데이터 스케일링의 문을 엶 — VLM의 native 2D grounding 능력을 그대로 활용하는 우아한 설계
- Ego-status shortcut 문제(open-loop 지표의 고질적 허점)에 대한 구조적 해법 제시. Table 9의 역전된 ablation은 이 분야에서 드문 명확한 실증
- Manipulation VLA의 pixel-space action(예: image-space affordance) 흐름과 개념적으로 공명 — "행동을 관찰 공간에서 표현"하는 일반 원리의 주행 도메인 사례

## 9. 한계

- **Local plane 가정**: 다층 도로(고가/지하), 심한 오클루전은 여전히 어려운 케이스 (저자 인정)
- **캘리브레이션 필수**: lifting에 카메라 파라미터 필요 — 약캘리브레이션/무캘리브레이션 시나리오는 범위 밖
- 2D bbox/궤적 라벨을 3D GT의 투영으로 생성 — 학습 데이터 구축은 여전히 3D 라벨 자원에 의존 (배포 시엔 불필요하다고 주장)
- Bench2Drive는 SFT만 사용 — closed-loop에서 RL 단계의 효과는 미검증
- Comfortness는 AutoVLA(39.33)에 소폭 열세(38.01)

## 10. 예상 질문

> ❓ **2D 계획이 안전하다는 보장은?**
> **답변**: 2D bbox는 가시적 3D 충돌 위험의 보수적 상위집합 — 3D 충돌이면 반드시 2D 겹침. 단, 역은 성립하지 않아 과도하게 보수적일 수 있고, 비가시(오클루전) 위험은 커버 못함. Closed-loop Bench2Drive 결과가 실질적 안전성을 보완 검증.

> ❓ **Local plane 가정이 경사로에서 깨지지 않나?**
> **답변**: nuScenes 3초 도로 높이 변화 평균 0.16 m로 depth 추정 오차(7.6 m) 대비 무시 가능. Table 10에서 높이 변화 2 m 이상 샘플에서도 성능 유지. 다만 3초 단기 계획에 한정된 논리.

> ❓ **AutoDrive-R2(0.20)가 L2는 더 낮은데?**
> **답변**: AutoDrive-R2는 ego-status에서 운동학으로 3D 궤적을 명시적 계산 — 논문이 비판하는 ego-status 의존의 극단적 형태라 회색 처리. Open-loop L2는 ego-status 외삽에 유리하게 편향된 지표.

> ❓ **왜 reasoning에는 reward를 안 주나?**
> **답변**: 자유형 CoT에 semantic reward를 주면 shortcut learning(보상 해킹)과 표현 경직을 유발. 검증 가능한 양 끝단(perception/action/trajectory)만 조이면 중간 reasoning은 자연히 정렬됨.

## 11. 재현성 평가

- 베이스 모델(Qwen2.5-VL-7B), 데이터셋(nuScenes/Waymo/Bench2Drive), 학습 설정(epochs, GRPO completion 8, 디코딩 파라미터, 8x80GB GPU) 명시 — 양호
- reward 가중치 $\lambda$ 구체값, reasoning 데이터 생성 세부는 Supplementary 참조
- 코드 공개 여부 불명 (프로젝트 페이지만 존재: pixelpilotvla.github.io)

## 12. 종합 평가

**Score: 9.5/10.** 주행 VLA의 두 구조적 병목(센서 종속 3D 예측, ego-status shortcut)을 단일한 decoupling 원리로 동시에 해결한 개념적으로 명료한 논문. 이론적 정당화(bijective projection, C-space 유추), dense-reward GRPO 설계, 그리고 Table 9의 결정적 ablation까지 논증 구조가 탄탄하다. Open-loop/closed-loop 모두 SOTA이며 zero-shot 차량 이전이라는 실용적 가치도 크다. Local plane 가정의 적용 범위와 closed-loop RL 미적용이 남은 과제.

---
*리뷰 작성일: 2026-07-11 · arXiv 2607.04637 (ECCV 2026) PDF 전문 검증*

<!-- VERIFIED: pdf -->
