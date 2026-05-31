# OASIS: Observation-Action Space Alignment via SE(3) Trajectory Prediction for Robotic Manipulation

> **한 줄 요약**: Qwen2.5-0.5B + DINOv2 + SigLIP + Depth Anything 3로 3D-aware feature를 인코딩한 뒤, 4-block transformer로 카메라 좌표계 SE(3) end-effector trajectory를 명시적으로 예측하고 2-block decoder가 action chunk를 생성하는 1.73B-param VLA. LIBERO 평균 97.6%, CALVIN ABC->D 4.57 avg seq len, 실로봇 89.2%.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대부분의 VLA는 hidden representation에서 **바로 action token/chunk를 회귀** → "observation의 추상적 representation"과 "geometric action space" 사이에 명시적 alignment 부재
- 일부 연구는 observation-space에서 future-image prediction(VPP, DreamVLA, Seer 등)으로 representation을 augment → 그러나 image space는 action space와 직접 연결되지 않음
- 3D Diffuser Actor 같은 3D action 모델은 좋은 성능을 보이나 backbone이 VLM이 아니라 generalization 한계

### 핵심 질문
- **VLM의 representation을 "관찰 공간"이 아닌 "행동 공간"과 align할 수 있는가?**
- **SE(3) trajectory를 explicit한 intermediate target으로 두면 manipulation accuracy가 올라가는가?**

📌 [Figure 1 삽입] — OASIS pipeline: VLM + 3D depth → SE(3) trajectory predictor → action decoder

---

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처

| 모듈 | 역할 | 파라미터 |
|------|------|----------|
| Qwen2.5-0.5B VLM | Vision-language backbone (LoRA tuned) | 0.5B |
| DINOv2 + SigLIP | Dual visual encoder | ~1B (frozen) |
| Depth Anything 3 | Metric depth estimation | included in 1.73B total |
| SE(3) Trajectory Predictor | 4-block transformer, predicts pose sequence | ~50M |
| Action Decoder | 2-block transformer, generates action chunk | ~20M |

- **Total**: 1.73B / **Trainable**: 0.18B (LoRA + 70M trajectory+decoder)

### 2.2 3D-Aware Feature Encoder

- DINOv2 (semantic) + SigLIP (vision-language) → joint visual features
- Depth Anything 3로 **metric depth** 추출
- Visual feature + depth → 3D-aware token으로 융합

> ❓ **예상 질문**: 왜 metric depth를 명시적으로 입력하는가? Relative depth로는 안 되는가?
> **답변**: SE(3) trajectory는 metric scale을 가지므로 relative depth로는 alignment 불완전. Depth Anything 3가 monocular metric depth를 제공함으로써 trajectory predictor가 절대 좌표계 출력 가능.

### 2.3 SE(3) Trajectory Predictor

- 4-block transformer
- 입력: visual + language + state token
- 출력: H step 동안의 카메라 좌표계 end-effector pose sequence (translation + rotation, gripper)
- 학습 목표: ground-truth EE trajectory에 대한 supervision (smooth L1 + rotation geodesic loss)

> ❓ **예상 질문**: 왜 카메라 좌표계인가? Robot base frame이 더 자연스럽지 않나?
> **답변**: 카메라가 view-invariant feature와 직접 align됨. Camera-frame trajectory는 visual feature와 metric depth만으로 정의 가능 → calibration robustness 향상. 단점: extrinsics가 바뀌면 재학습 필요.

### 2.4 Action Decoder

- 2-block transformer
- 입력: pose-supervised hidden state + state token
- 출력: executable action chunk (joint angles or EE delta)
- Trajectory가 정답 정보 leak이 아닌 inductive bias 역할: trajectory를 명시적으로 학습한 representation 위에서 action 생성

> ❓ **예상 질문**: Trajectory prediction이 단순히 multi-task learning(co-training) 효과 아닌가?
> **답변**: 부분적으로 yes. 그러나 ablation에서 trajectory loss를 빼면 LIBERO Long 성능이 크게 떨어진다고 보고 → trajectory가 representation alignment에 기여.

---

## 3. 데이터 전략

### Training Data
- LIBERO 표준 expert demos (각 suite 별)
- CALVIN ABC->D split
- **No large-scale robotic pretraining** — 매우 가벼운 데이터 setting

### Real-world
- Franka Research 3 + Kinova Gen3 — 2개 로봇 platform 검증

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 4x NVIDIA A800 |
| Steps | 50,000 per suite |
| Batch size | 64 |
| Trainable | LoRA on VLM + trajectory predictor + decoder (0.18B) |

---

## 5. 실험 설계 및 평가 프로토콜

- **LIBERO 4 suite**: spatial / object / goal / long
- **CALVIN ABC->D**: long-horizon multi-task
- **Real-world**: Franka + Kinova 두 platform, goal / spatial / long subset
- **OOD**: 배경 변화, 카메라 위치 변경, 사람 간섭

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|--------|
| Baselines (range) | 88~98 | 89~99 | 85~98 | 78~95 | 86~96 |
| **OASIS** | **99.0** | **98.8** | **97.4** | **95.2** | **97.6** |

- LIBERO Long 95.2%는 최상위급 (대부분 모델이 90% 미만)
- LIBERO Spatial 99.0%는 거의 saturation

### CALVIN ABC->D (Table 2)

| Completed Tasks | OASIS |
|----------------|-------|
| t=1 | **98.1%** |
| t=2 | **94.9%** |
| t=3 | **91.7%** |
| t=4 | **88.9%** |
| t=5 | **83.3%** |
| **Avg seq len** | **4.57** |

- 4.57은 당시 GR-1(3.86), RoboFlamingo(2.48) 등을 압도하는 수치
- 5-task chain에서도 83.3% — long-horizon 강점 입증

### Real-World

| Task type | Success |
|-----------|---------|
| Goal | 98.6% |
| Spatial | 85.8% |
| Long | 83.3% |
| **Avg** | **89.2%** |

### OOD Robustness

| 설정 | 평균 성공 |
|------|---------|
| Unseen background | 90.8% |
| Altered camera | 90.8% (overall avg) |
| Human interference | included in 90.8% |

- 실로봇에서 OOD 90.8% 평균 → camera-frame SE(3) prediction이 view robustness에 기여

---

## 7. Ablation 분석

### Trajectory Loss의 기여

| 설정 | LIBERO Long | CALVIN avg len |
|------|------------|----------------|
| Full OASIS | 95.2 | 4.57 |
| -trajectory loss | ~89 | ~4.1 |
| -metric depth | ~92 | ~4.3 |

- Trajectory supervision이 long-horizon에서 큰 기여

### Depth ablation

- Metric depth 제거 시 spatial reasoning 약화
- Relative depth로 대체 시 -2~3%p

---

## 8. 관련 연구 비교

| 모델 | Backbone | 3D 정보 | Explicit trajectory | LIBERO Avg | CALVIN avg len |
|------|----------|--------|--------------------|-----------|----------------|
| pi-0 | PaliGemma | x | x | 94.2 | - |
| QDepth-VLA | depth-aware | △ | x | 95.0 | - |
| Unified-VLA | unified token | x | x | 95+ | ~4.0 |
| WorldVLA | world model | △ | x | - | 4.0+ |
| 3D Diffuser Actor | scratch | ✓ | △ (path) | - | - |
| **OASIS** | **Qwen2.5-0.5B** | **✓** | **✓ SE(3)** | **97.6** | **4.57** |

### 핵심 차이
- **SE(3) trajectory를 명시적 supervision target**으로 사용한 첫 VLA
- 1.73B의 비교적 small backbone으로 SOTA급

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Trajectory ground-truth 의존**: SE(3) supervision은 expert demo에 EE pose가 정확히 logging되어야 함 — 일부 dataset에서는 noisy
2. **Camera extrinsics 의존**: Camera-frame trajectory는 camera setup이 바뀌면 재학습 필요. Multi-camera 환경에서 generalization 미검증
3. **Open-source 여부 불확실**: Project page만 제공, 코드 공개 미명시
4. **Long-horizon limitation**: CALVIN 5-task 83.3%는 최상위지만, 더 긴 시퀀스에서의 deg-radation 미평가
5. **Depth Anything 3 dependency**: Monocular metric depth quality가 OASIS 전체 성능의 upper bound가 될 가능성

### Attribution 문제
- LIBERO 97.6%의 향상이 **SE(3) trajectory 때문인지, Depth Anything 3 metric depth 때문인지, Qwen2.5-0.5B + DINOv2 + SigLIP 강력한 encoder 때문인지** 완전 분리 어려움
- Ablation에서 trajectory 제거 시 -6~8%p지만, encoder 조합 ablation 부재

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — SE(3) explicit trajectory를 VLA에 통합한 첫 사례 |
| **Technical depth** | ★★★★☆ — 3개 encoder + 2-stage prediction 구조 |
| **Experimental rigor** | ★★★★★ — LIBERO + CALVIN + Real-world (2 robots) + OOD |
| **Practical impact** | ★★★★☆ — 1.73B small, large-scale pretraining 불필요 |
| **Writing quality** | ★★★★☆ — 비교적 명확 |

**강점**: VLA representation을 명시적으로 action space와 align하는 깔끔한 inductive bias. **LIBERO + CALVIN + Real-world 전 영역에서 SOTA급**. Pretraining 없이 강력한 성능 → reproducibility 우수. **약점**: Camera extrinsics 의존, code 미공개, depth backbone에 대한 의존성.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO 97.6은 saturation 영역인데 실제 의미가 있나? | LIBERO Long 95.2가 핵심 — 대부분 모델이 78-90 범위. Long-horizon에서의 +5%p는 의미 큼 |
| 2 | Camera-frame이 robot-base-frame보다 나은 이유? | View-invariant visual feature와 직접 align. 단, multi-camera 환경에서는 disadvantage |
| 3 | Depth Anything 3 없이 성능은? | Ablation에서 metric depth 제거 시 ~2-3%p 하락. Quality dependency 존재 |
| 4 | 4.57 avg seq len은 reproducible한가? | CALVIN ABC->D는 evaluation seed에 민감. 다중 seed average 보고 부재 |
| 5 | Trajectory loss vs action loss의 weight balance? | Paper에서 명확히 명시 없음. Critical hyperparameter 가능성 |
| 6 | Real-world 2개 robot에서의 cross-embodiment transfer? | Franka + Kinova 별도 학습으로 보임. Cross-embodiment zero-shot 평가 부재 |
| 7 | 0.18B trainable로 어떻게 0.5B VLM을 LoRA tuning하나? | LoRA rank가 작아 trainable param이 줄어듦. 다만 transfer expressivity 한계 |
| 8 | 왜 0.5B Qwen, 더 큰 모델은? | Compute efficiency 강조. 7B/13B로 scaling 시 robustness가 어떻게 변하는지 미실험 |

<!-- VERIFIED: pdf -->
