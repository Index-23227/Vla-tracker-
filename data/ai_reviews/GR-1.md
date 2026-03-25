# GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation

> **한 줄 요약**: GPT 스타일 transformer로 language instruction, observation, robot state를 입력받아 action과 future image를 동시에 예측하는 Video World-Action Model로, 대규모 인터넷 비디오 사전학습을 통해 CALVIN에서 88.9%→94.9% (seen) 및 53.3%→85.4% (unseen scene zero-shot) 향상.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 로봇 데이터만으로는 **diversity와 규모가 부족** → 일반화 한계
- 인터넷 비디오에는 풍부한 physical dynamics knowledge가 있으나, action label이 없어 직접 활용 불가
- 기존 representation learning (R3M, VC-1)은 frozen representation을 사용 → downstream fine-tuning의 유연성 제한

### 핵심 질문
- **대규모 비디오 생성적 사전학습이 로봇 조작의 일반화에 직접적으로 기여하는가?**

---

## 2. 방법론 심층 분석

### 2.1 GPT-style Unified Architecture

단일 transformer가 다중 modality를 autoregressive하게 처리:

$$[\mathbf{l}; \mathbf{o}_{1:t}; \mathbf{s}_{1:t}] \to [\hat{\mathbf{a}}_{t}; \hat{\mathbf{o}}_{t+1}]$$

- Language tokens + image tokens + state tokens → action tokens + future image tokens

### 2.2 Video Generative Pre-training

1. **Stage 1**: 인터넷 비디오로 next frame prediction 사전학습
2. **Stage 2**: 로봇 데이터로 action prediction + video prediction fine-tuning

> ❓ **예상 질문**: 인터넷 비디오와 로봇 비디오의 domain gap은 어떻게 극복하는가?
> **답변**: Stage 1에서 general visual dynamics를 학습하고, Stage 2에서 robot-specific dynamics로 fine-tuning. Domain gap은 fine-tuning에서 해소되나, pre-trained knowledge의 얼마나 많은 부분이 보존되는지는 미분석.

---

## 3. 실험 결과 심층 분석

### CALVIN ABC→D

| 모델 | Seen (1-task SR%) | Unseen (zero-shot SR%) | Avg Len |
|------|-----------------|---------------------|---------|
| HULC | 82.7 | 53.3 | 2.64 |
| GR-1 (no pretrain) | 87.2 | 61.2 | 3.15 |
| **GR-1 (pretrained)** | **94.9** | **85.4** | **3.86** |

- **Unseen scene에서의 53.3%→85.4%**는 video pre-training의 핵심 기여
- Pre-training 없는 GR-1도 강력하나, pre-training의 +24.2%p가 결정적

---

## 4. Ablation 분석

| 구성요소 | Unseen SR (%) |
|---------|-------------|
| Full GR-1 | 85.4 |
| No video pre-training | 61.2 |
| No future image prediction | 78.3 |
| No language conditioning | 72.5 |
| Smaller pre-training data | 75.8 |

---

## 5. 한계 및 미해결 문제

1. **Image generation overhead**: 추론 시 future image token 생성 → latency 증가
2. **Pre-training 비용**: 대규모 비디오 학습에 상당한 compute 필요
3. **Video prediction 품질의 action 영향**: 생성된 이미지가 부정확할 때의 degradation 분석 부족
4. **실제 로봇 실험 제한적**

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Video generative pre-training for robots의 선구적 연구 |
| **Technical depth** | ★★★★☆ |
| **Experimental rigor** | ★★★★☆ — CALVIN + real robot |
| **Practical impact** | ★★★★★ — WAM 패러다임의 기초 확립 |
| **Writing quality** | ★★★★☆ |

**강점**: Video pre-training→robot fine-tuning 패러다임 확립. Unseen scene 일반화의 극적 향상이 설득력 있음. **약점**: 추론 latency, pre-training 비용.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM 방식으로 test-time image generation을 skip하면? | 성능 하락 예상되나 이 ablation 미수행. Fast-WAM이 이후 이를 검증 |
| 2 | Pre-training 데이터의 domain이 성능에 미치는 영향? | Kitchen video가 kitchen task에 유리. Domain match 분석이 부분적 |
| 3 | CALVIN 외 벤치마크에서의 scaling behavior는? | LIBERO 등에서의 추가 검증 필요 |
