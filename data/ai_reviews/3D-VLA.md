# 3D-VLA: A 3D Vision-Language-Action Generative World Model

> **한 줄 요약**: BLIP2-FlanT5XL 기반 3D-LLM에 interaction token(물체·위치·행동)과 embodied diffusion model(RGB-D/point cloud 목표 생성)을 결합하여, 2M 3D-language-action 데이터 쌍으로 학습한 최초의 3D 기반 embodied generative world model.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA 모델(RT-2, PaLM-E 등)은 **2D 이미지 입력**에 의존 → 3D 공간 추론(깊이, 물체 간 거리, 기하학적 관계)에 본질적 한계
- Embodied AI에서 행동(action)은 3D 물리 공간에서 발생하지만, 모델은 2D projection을 통해 간접적으로만 3D 정보에 접근
- 기존 world model은 주로 2D 이미지 생성(video prediction)에 국한 → 3D 구조(point cloud)를 명시적으로 예측하는 모델 부재

### 핵심 질문
- **3D 이해(추론, 로컬라이제이션)와 3D 생성(goal image, goal point cloud)을 하나의 프레임워크에서 통합할 수 있는가?**
- **3D world model이 로봇 action planning에 실질적으로 기여하는가?**

📌 [Figure 1 삽입] — 3D-VLA 아키텍처: 3D-LLM + Interaction Tokens + Embodied Diffusion Models

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

3D-VLA는 **BLIP2-FlanT5XL**을 backbone으로 사용하며, 3D-LLM 프레임워크를 확장:
- **Q-Former**: Multi-view feature를 LLM이 처리 가능한 토큰으로 변환 (학습 시 unfrozen)
- **Interaction Tokens**: 3D 환경과의 상호작용을 위한 특수 토큰 vocabulary
- **Embodied Diffusion Models**: Goal image(RGB-D) 및 goal point cloud 생성

### 2.2 Interaction Token 설계

기존 LLM 토큰 외에 다섯 가지 특수 토큰 유형 추가:

| Token Type | 형식 | 역할 |
|-----------|------|------|
| Object tokens | `<obj>...</obj>` | 조작 대상 물체 지정 |
| Location tokens | `<loc0>`~`<loc255>` | 3D bounding box (AABB) 좌표 |
| Action tokens | `<aloc0-255>`, `<arot0-255>`, `<gripper0/1>` | 7-DoF 로봇 action |
| Scene tokens | `<scene>...</scene>` | 정적 장면 embedding |
| Modality tokens | `<image>`, `<pcd>` | Goal 생성 modality 지정 |

> ❓ **예상 질문**: Action tokenization에서 256 bin은 precision이 충분한가?
> **답변**: `<aloc0-255>`와 `<arot0-255>`로 위치와 회전을 각각 256 단계로 이산화. 이는 RT-2와 동일한 방식이며, ±1m 범위에서 ~7.8mm resolution. 정밀 삽입 등 고정밀 태스크에서는 bottleneck이 될 수 있음.

### 2.3 Embodied Diffusion Models

**두 가지** 생성 모델을 별도로 학습:

1. **RGB-D Goal Generation**: Stable Diffusion V1.4 기반, RGB와 depth latent를 concatenate하여 조건부 생성
2. **Point Cloud Goal Generation**: Point-E 기반, point cloud conditioning으로 목표 point cloud 생성

**Alignment**: Transformer 기반 projector가 LLM의 hidden features를 diffusion model 공간으로 매핑. LoRA를 diffusion model에 적용하여 fine-tuning.

> ❓ **예상 질문**: 왜 하나의 diffusion model이 아니라 두 개를 별도로 학습하는가?
> **답변**: RGB-D (2D+depth)와 point cloud (3D)는 data structure가 근본적으로 다름. Stable Diffusion은 grid-based latent에 특화, Point-E는 unordered point set에 특화. 통합 모델은 두 modality의 충돌로 품질 저하 가능.

> ❓ **예상 질문**: Diffusion model의 inference latency는 action planning에 적합한가?
> **답변**: 논문에서 latency를 명시적으로 보고하지 않음. Stable Diffusion 기반이므로 수백 ms~수초 소요 예상. Real-time closed-loop control에는 부적합하며, **open-loop planning**에 적합한 구조.

---

## 3. 데이터 전략

### 3D Embodied Instruction Dataset

**규모**: **2M 3D-language-action 데이터 쌍**

| 데이터 소스 | 유형 | 규모 |
|-----------|------|------|
| Open X-Embodiment (12 datasets) | Robotics | ~305K episodes |
| Dobb-E, RH20T | Depth-rich robotics | 포함 |
| RLBench, CALVIN | 시뮬레이션 | 포함 |
| Epic-Kitchens, HOI4D | Human-Object Interaction | ~11K episodes |

### 데이터 전처리 핵심 이슈

- **95%의 비디오 데이터에 3D 정보 없음** → **ZoeDepth**로 monocular depth estimation
- **RAFT**로 optical flow 추정하여 depth 일관성 보정
- **Grounded-SAM**으로 2D object mask 추출
- **spaCy**로 instruction parsing → 조작 대상 물체 식별
- **GPT-3.5-turbo-0125**으로 annotation 다양화 (2-3 few-shot 제공)

> ❓ **예상 질문**: 95%의 데이터가 estimated depth를 사용한다면, depth 품질이 전체 성능을 제한하지 않는가?
> **답변**: 핵심 우려. ZoeDepth의 metric scale 부정확성이 3D localization과 point cloud generation 품질에 직접 영향. 논문에서도 이를 인정하나, 대규모 데이터의 양적 이점이 개별 depth 오류를 보상한다고 주장.

---

## 4. 시스템/학습 세부사항

| 단계 | Epochs | Hardware | Batch Size | LR |
|------|--------|----------|-----------|-----|
| Pre-training | 30 | 6×32 V100 GPUs | 4/node | 1e-8→1e-5 (warmup 1K steps), cosine→1e-6 |
| Alignment | 20 (max) | 6×64 V100 GPUs | 2/node | - |

- Optimizer: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.05)
- Q-Former: unfrozen during training

---

## 5. 실험 설계 및 평가 프로토콜

평가 영역이 **다섯 가지**로 광범위:
1. 3D Reasoning (Embodied QA, Task Caption, What-if QA, Dense Caption)
2. 3D Localization (IoU, Acc@25, Acc@50)
3. RGB-D Goal Generation (PSNR, CLIP Sim, SSIM, FID)
4. Point Cloud Goal Generation (P-FID, Chamfer-L₁)
5. Action Planning (RLBench, CALVIN)

---

## 6. 실험 결과 심층 분석

### 3D Reasoning (Table 1)

| 태스크 | 메트릭 | BLIP2 Baseline | **3D-VLA** | 향상 |
|-------|--------|---------------|-----------|------|
| Embodied QA | BLEU-4 | 15.48 | **26.80** | +73% |
| Task Caption | BLEU-4 | 3.16 | **34.88** | +1004% |
| What-if QA | BLEU-4 | 0.06 | **29.38** | 극적 |
| Dense Caption | BLEU-4 | 13.96 | **34.62** | +148% |

### 3D Localization (Table 2)

| 모델 | IoU | Acc@25 | Acc@50 |
|------|-----|--------|--------|
| Kosmos-2 (GT Depth) | - | 12.73 | - |
| CoVLM (GT Depth) | - | 25.39 | - |
| **3D-VLA** | **29.33** | **42.26** | **27.09** |

### Goal Generation (Tables 3-4)

| 모델 | PSNR ↑ | CLIP Sim ↑ | SSIM ↑ | FID ↓ |
|------|--------|-----------|--------|-------|
| InstructPix2Pix (robotics FT) | 16.67 | **0.941** | 0.628 | 0.178 |
| SuSIE | 15.20 | 0.898 | 0.549 | 0.182 |
| **3D-VLA** | **17.21** | 0.920 | **0.636** | **0.177** |

| Point Cloud | P-FID ↓ | Chamfer-L₁ ↓ |
|------------|---------|-------------|
| Point-E | 5.241 | 0.159 |
| **3D-VLA** | **4.796** | **0.139** |

### ⚠️ Action Planning — RLBench (Table 5)

| 태스크 | LanCon-Learn | **3D-VLA** |
|-------|-------------|-----------|
| Put Knife | 28.8% | **68%** |
| Take Umbrella | 45.6% | **52%** |
| Pick up Cup | 44.2% | 40% |
| Pick up Cup (unseen) | - | 24% |

- **태스크별 편차가 매우 큼** (24%~68%)
- Pick up Cup에서는 baseline보다 약간 열위

### ⚠️ Action Planning — CALVIN (Table 6)

| Completed Tasks | MCIL | **3D-VLA** |
|----------------|------|-----------|
| 1 task | 28.2% | **44.7%** |
| 2 tasks | 2.5% | **16.3%** |
| 3 tasks | 0.3% | **8.1%** |

- **CALVIN 성능이 매우 낮음** — 3 tasks 8.1%는 당시 SOTA (HULC 2.64 avg len) 대비 크게 뒤처짐
- 이는 3D-VLA가 **manipulation보다 3D reasoning/generation에 초점**을 맞췄음을 보여줌

---

## 7. Ablation 분석

### Bounding Box 예측의 영향

| 설정 | Image PSNR | Image SSIM | PC Chamfer-L₁ |
|------|-----------|-----------|--------------|
| With predicted BBox | **17.21** | **0.636** | **0.139** |
| Without BBox | 17.02 | 0.632 | 0.143 |

- BBox 예측의 기여가 미미 (PSNR +0.19, Chamfer +0.004)
- 이는 "interaction token의 필요성"에 의문을 제기

---

## 8. 관련 연구 비교

| 모델 | 입력 | 3D 이해 | Goal Generation | Action Planning |
|------|------|---------|----------------|----------------|
| RT-2 | 2D image | ✗ | ✗ | ✓ (strong) |
| PaLM-E | 2D + state | Implicit | ✗ | ✓ (language plan) |
| 3D-LLM | 3D | ✓ | ✗ | ✗ |
| SuSIE | 2D | ✗ | ✓ (image) | ✓ (via goal) |
| **3D-VLA** | **2D + depth** | **✓** | **✓ (RGB-D + PC)** | **△ (weak)** |

### 핵심 차이
- **3D 이해 + 3D 생성**을 동시에 수행하는 유일한 모델
- 그러나 **action planning 성능이 약함** — 3D reasoning과 manipulation의 bridging이 불완전

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Action planning 성능이 핵심 약점**: CALVIN 3-task 8.1%는 당시 SOTA(HULC avg 2.64)의 절반 수준도 안 됨. 3D reasoning이 뛰어나도 action으로의 전이가 약함
2. **Open-loop 평가**: RLBench에서 **open-loop control** 사용 — 환경 피드백 없이 전체 trajectory를 한번에 생성. Closed-loop baseline과의 불공정한 비교 가능성
3. **95% estimated depth**: 데이터의 절대 다수가 monocular depth estimation에 의존 → 3D 정보의 정확도에 구조적 한계
4. **Computational cost 미보고**: 모델 파라미터 수, 추론 latency, FLOPs 등 핵심 효율성 지표 전무. 192대 V100로 학습하는 규모인데, 추론 비용도 상당할 것
5. **Real-world 실험**: 실제 로봇에서 "사람 감독 하에" 시연만 제시 — 정량적 real-world 결과 없음, 충돌 위험으로 사람 감독 필수

### Attribution 문제
- 3D reasoning에서의 큰 향상이 **3D 표현** 덕분인지, **2M 데이터 규모** 덕분인지, **BLIP2의 강력한 backbone** 덕분인지 분리 불가
- BBox ablation에서의 미미한 차이(+0.19 PSNR)는 "interaction token"의 실제 가치에 의문을 제기
- Action planning에서의 낮은 성능은 "3D가 action에 도움된다"는 핵심 주장을 약화

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 3D reasoning + 3D generation + action을 통합하는 최초 시도 |
| **Technical depth** | ★★★★☆ — 다양한 모듈 통합이 체계적 |
| **Experimental rigor** | ★★★☆☆ — 5가지 평가 축은 좋으나 action planning이 약하고, open-loop 평가의 공정성 의문 |
| **Practical impact** | ★★☆☆☆ — 192 V100 학습, latency 미보고, action 성능 미달 |
| **Writing quality** | ★★★★☆ — 광범위한 실험 포함 |

**강점**: 3D embodied AI의 비전을 제시하는 선구적 연구. 2M 데이터 구축과 다섯 가지 평가 축의 포괄성이 인상적. **약점**: 논문의 핵심 동기("3D 이해가 로봇 action을 향상시킨다")에 대한 실험적 근거가 가장 약한 부분(action planning)에서 설득력이 부족. CALVIN 3-task 8.1%는 당시에도 경쟁력이 없는 수치.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | CALVIN avg len으로 환산하면 얼마인가? | 1-task 44.7%, 2-task 16.3%, 3-task 8.1% → avg len ≈ 0.69. HULC(2.64), GR-1(3.86)에 비해 극히 낮음 |
| 2 | Open-loop 평가가 공정한가? | RLBench에서 baseline(LanCon-Learn)은 observation history 사용. 3D-VLA는 open-loop. 직접 비교에 주의 필요 |
| 3 | BBox ablation에서 +0.19 PSNR이면 interaction token이 정말 필요한가? | 맞음, 기여가 marginal. Interaction token의 necessity에 대한 강력한 ablation 부재 |
| 4 | 95% estimated depth의 quality impact는? | GT depth vs estimated depth 비교 실험이 없음. Localization에서 Kosmos-2(GT depth)보다 3D-VLA가 나은 것은 depth quality보다 모델 능력 차이일 수 있음 |
| 5 | 192 V100으로 학습하는데 추론 latency는? | 미보고. Stable Diffusion + Point-E + LLM → 수초 이상 소요 예상. Real-time 불가 |
| 6 | 3D reasoning 성능 향상이 action planning으로 전이되지 않는 이유는? | 3D understanding과 motor control 사이의 gap. LLM이 3D를 "이해"해도 정밀한 7-DoF action 생성은 별개 문제. 이 disconnection이 논문의 근본적 한계 |
| 7 | Point cloud goal generation의 downstream 활용은? | 논문에서 생성된 point cloud를 action planning에 직접 활용하는 실험 부재. 생성 자체의 품질만 평가 |
| 8 | 왜 PerAct, Act3D 등 3D manipulation 전문 모델과 비교하지 않았는가? | PerAct의 RLBench 49.4%와 비교하면 3D-VLA의 특정 태스크 성능(40-68%)이 competitive하나, 전체 18-task 비교가 없음 |

<!-- VERIFIED: pdf -->
