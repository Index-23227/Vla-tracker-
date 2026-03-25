# CrossFormer: Scaling Cross-Embodied Learning — One Policy for Manipulation, Navigation, Locomotion and Aviation

> **한 줄 요약**: 130M 파라미터의 경량 transformer로 20+ 이종 로봇의 900K trajectory를 학습하여, 단일 팔·양팔·바퀴 로봇·쿼드콥터·사족보행 모두를 하나의 정책으로 제어하고 **specialist와 동등(73% vs 67%)**, 쿼드콥터 zero-shot 82% 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 cross-embodiment 학습(RT-X, Octo)은 **manipulation에 편향**
- 서로 다른 로봇의 observation/action space를 수동으로 alignment해야 함 (Yang et al.)
- "단일 정책"이 specialist를 능가하지 못하면 실용적 의미 부재

### 핵심 질문
- **수동 alignment 없이 4개 도메인(manipulation, navigation, locomotion, aviation)을 하나의 정책으로 학습할 수 있는가?**
- **Cross-embodiment 학습이 specialist 대비 positive transfer를 보이는가?**

---

## 2. 방법론 심층 분석

### 2.1 Architecture

| 항목 | 값 |
|------|-----|
| Total params | **130M** |
| Transformer | 12 layers, 8 heads |
| MLP dim | 2048 |
| Token embedding | 512 |
| Context window | **2135 tokens** (5 timesteps) |
| Image encoder | ResNet-26 (ImageNet init) |

### 2.2 Modality-based Tokenization

관측을 modality별로 토큰화 (로봇별 adapter 불필요):
- **Images**: ResNet-26 → flattened spatial features → projection to 512
- **Proprioception**: Direct projection to 512
- 4가지 이미지 encoder variant: workspace, egocentric, wrist (2종)
- **같은 카메라 유형이면 encoder weight 공유**

### 2.3 Action Prediction Heads (4 Types)

| Head | Dimensions | Chunk Size | Freq |
|------|-----------|-----------|------|
| Single-arm Cartesian | 7D (EE delta) | 4 | 5-15Hz |
| Navigation waypoints | 2D (xy delta) | 4 | 4Hz |
| Bimanual joint | 14D | **100** | 20Hz |
| Quadruped joint | 12D | 1 | 20Hz |

- **L1 regression loss** (diffusion/classification 아닌 — bimanual high-freq에 적합)

> ❓ **예상 질문**: L1 regression은 multimodal action에 취약하지 않은가?
> **답변**: 맞음. 이론적으로 diffusion이 multimodal에 강하지만, 저자들은 "prior work on high-frequency bimanual manipulation에서의 성공"을 근거로 L1 선택. Bimanual 100-step chunk에서 diffusion의 latency가 더 큰 문제.

---

## 3. 데이터 전략

### 900K Trajectories 구성

| 소스 | 비율 | Domain |
|------|------|--------|
| Fractal (OXE) | 17% | Single-arm manipulation |
| Bridge | 17% | Single-arm manipulation |
| GNM navigation | 17% | Navigation |
| ALOHA multi-task | 17% | Bimanual |
| Go1 quadruped | 8.5% | Locomotion |
| Franka tabletop | 8.5% | Single-arm manipulation |
| Others (DROID 등) | 나머지 | Mixed |

### 학습 설정

| 항목 | 값 |
|------|-----|
| Steps | 300K |
| Batch size | 512 |
| Hardware | **TPU V5e-256 pod** |
| Duration | **47 hours** |
| Optimizer | AdamW, LR 3e-4, WD 0.1 |
| Goal relabeling | Hindsight (random future obs) |

---

## 4. 실험 결과 심층 분석

### Cross-Embodiment Results (Table 3)

| Platform | Tasks | Single-Robot | Best Prior | **CrossFormer** |
|---------|-------|-------------|-----------|----------------|
| WidowX | 4 | 0.42 | 0.34 (Octo) | **0.50** |
| Franka | 2 | 0.62 | 0.60 (OpenVLA) | **0.62** |
| ALOHA | 2 | 0.50 | 0.50 (ACT) | **0.70** |
| LoCoBot | 3 | 0.92 | 0.48 (ViNT) | **0.93** |
| Tello Quadcopter | 1 | 0.68 | 0.68 (ViNT) | **0.82** |
| Go1 Walking | 1 | 1.0 | N/A | **1.0** |

**Overall**: CrossFormer **73%** vs Single-Robot 67% vs Best Prior 51%

### 핵심 발견

1. **ALOHA에서 +20%p** (0.50→0.70): Cross-embodiment data가 bimanual에 가장 큰 positive transfer
2. **Tello zero-shot 82%**: 학습 데이터에 쿼드콥터 없이 navigation head로 zero-shot 성공
3. **Franka에서 동등** (0.62 vs 0.62): OpenVLA와 비슷하지만 모델이 53x 작음 (130M vs 7B)

### Trial 수

| Platform | Trials |
|---------|--------|
| WidowX | 48 (12×4 tasks) |
| Franka | 39 |
| ALOHA | 20 (10×2 tasks) |
| LoCoBot | 6 (1×6 locations) |
| Tello | 3 |
| Go1 | 25-min normalized |

> ⚠️ **Trial 수가 적음**: 특히 Tello (3), LoCoBot (6)은 통계적 유의성 우려

---

## 5. 관련 연구 비교

| 모델 | Params | Embodiments | Domains | Manual Align | Overall SR |
|------|--------|------------|---------|-------------|-----------|
| RT-1-X | Large | ~7 (manip.) | 1 | ✗ | - |
| Octo | 93M | ~9 (manip.) | 1 | ✗ | 0.34 (WidowX) |
| Yang et al. | ~100M | Multi | 2 | **✓** | 0 (WidowX) |
| **CrossFormer** | **130M** | **20+** | **4** | **✗** | **0.73** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점 (저자 명시 포함)
1. **Positive transfer 미미**: 저자 직접 인정 — "Our results do not yet show significant positive transfer across embodiments." CrossFormer가 specialist와 비슷하지만 명확히 넘지는 못함
2. **Hand-picked sampling weights**: 데이터 비율을 수동으로 조정. "Ideal scaling would eliminate need" — 자동 data mixture 전략 부재
3. **Inference speed**: 130M이지만 2135 token context → 고주파 제어(>20Hz)에서 한계. 저자 명시
4. **Trial 수 부족**: Tello 3회, LoCoBot 6회로는 통계적 신뢰 불충분
5. **Long-horizon 미평가**: 대부분 short-horizon. Multi-step task에서의 cross-embodiment 성능 미검증
6. **Manipulation 편향**: 데이터의 ~50%가 manipulation → navigation/locomotion은 상대적으로 under-represented

### Attribution 문제
- 73% overall이 **cross-embodiment transfer** 덕분인지, **단순히 더 많은 데이터 (900K)** 덕분인지 불명확
- 동일 데이터 양에서 in-domain data만 사용한 비교가 더 공정

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 4개 도메인 통합이 야심적이고 실현됨 |
| **Technical depth** | ★★★☆☆ — 아키텍처는 간결, 깊은 분석 부족 |
| **Experimental rigor** | ★★★☆☆ — 다양한 embodiment이나 trial 수 부족 |
| **Practical impact** | ★★★★☆ — 130M으로 multi-domain이 가능하다는 증거 |
| **Writing quality** | ★★★★☆ — 명확하고 읽기 쉬움 |

**강점**: 4개 도메인을 130M 모델로 통합, manual alignment 불필요, Tello zero-shot 성공. **약점**: 저자 스스로 인정한 "significant positive transfer 부재", 적은 trial 수, data sampling weight 수동 설정.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Positive transfer가 없다"면 왜 cross-embodiment 학습을 하는가? | Transfer는 미미하나 **negative transfer도 거의 없음** (specialist와 동등). 하나의 모델로 여러 로봇을 운용하는 **실용적 가치** |
| 2 | ALOHA +20%p가 진짜 transfer인가? | 가장 설득력 있는 positive transfer 사례. Manipulation data가 bimanual coordination에 도움 |
| 3 | Tello 3 trials로 82%를 주장하는 것이 타당한가? | 통계적으로 불충분. 95% CI가 매우 넓을 것. 더 많은 trial 필요 |
| 4 | 130M이면 VLM이 아닌데, language understanding은? | 텍스트 지시 대신 **goal image conditioning** 사용. Language 이해는 제한적 |
| 5 | 900K를 더 키우면 (9M?) transfer가 emergence하는가? | Scaling 분석 부재. Octo→RT-X에서도 규모가 커지면 transfer가 더 명확해짐을 시사 |

<!-- VERIFIED: pdf -->
