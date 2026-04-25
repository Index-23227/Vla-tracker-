# InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation

> **한 줄 요약**: Eagle2-2B VLM을 backbone으로 사용해 latent action queries + MoE LoRA + flow matching action expert를 결합, 650K VLA-IT corpus와 multimodal data로 공동 학습하여 SimplerEnv에서 SpatialVLA 대비 +33% (61.2 vs 45.9 평균), SimplerEnv-Instruct에서 OpenVLA(FT&GPT) 대비 +29% (46.9 vs 35.6)를 달성한 ICLR 2026 spotlight 후보.

---

## 1. 배경 및 동기

기존 VLA의 세 가지 한계(§1): (1) **task interference / catastrophic forgetting** — 액션 학습 시 VLM의 multimodal reasoning이 손실됨 (Table 1에서 OpenVLA의 MMMU=26.0, MMStar=28.2로 붕괴 확인), (2) **data scarcity** — multimodal supervision이 풍부한 manipulation 데이터 부족, (3) **methodological gap** — multimodal reasoning을 action으로 변환하는 메커니즘 부재. "어떻게 VLM의 multimodal reasoning을 침식하지 않으면서 manipulation skill을 학습할 수 있는가?"가 핵심 질문이다.

---

## 2. 방법론 심층 분석 (§3, Fig. 2)

### 2.1 아키텍처

- **Embodied VLM**: Eagle2-2B (Li et al. 2025c). N개의 learnable action query Q ∈ R^(N×D)가 VLM hidden state에 attend하여 latent action C 추출.
- **MoE Adaptation**: LoRA expert + scalar gating head. h = W0·x + Σ Bi·Ai·x·αi·λi (§3.1). Language LoRA + Action LoRA로 reasoning/manipulation을 동적 전환.
- **Action Expert**: DINOv2 vision encoder + FiLM modulation + 12-layer transformer + flow matching (§3.1). 비인과 attention(입력 내) + 인과 attention(입력 간) 결합.

### 2.2 Two-Stage Training (§3.2)

- **Stage 1 (Action Pre-training)**: language motion 텍스트 + action을 동시 예측. 650M params(latent action embedding + action LoRA)만 학습 → "Expert" 모델.
- **Stage 2 (VLA-IT)**: language LoRA + scalar head 추가, 220M params만 학습 → "Generalist" 모델. 1:7 multimodal:manipulation 비율로 interleaved 훈련.

### 2.3 VLA-IT Dataset (§4.1, Fig. 3)

650K samples, 4가지 annotation 유형: scenario captioning / QA / command rewriting / context creation. GPT-4o가 episode당 3 frame과 GT instruction으로 annotation.

### 2.4 SimplerEnv-Instruct (§4.2)

80개 zero-shot task, 1.1K trial. Task aggregation(50) + Situated reasoning(30). OOD object/instruction.

---

## 3. 실험 결과 심층 분석

### 3.1 Multimodal Understanding (Table 1)

InstructVLA-Generalist(1.5B)는 MMMU=44.2, MMStar=56.2, MM-Vet=51.7로 base Eagle2 (43.1/56.4/53.8) 대비 거의 동등 — VLM 능력 보존 성공. OpenVLA-FT는 MMMU=26.0, MMStar=28.2로 catastrophic forgetting 심각.

### 3.2 Robotic Manipulation (Table 2, 3 random seeds)

| Method | Google+WidowX Avg | Task Aggregation | Situated Reasoning |
|---|---|---|---|
| OpenVLA-7B | 27.2 | 14.8 | 13.6 |
| SpatialVLA-3B | 45.9 | 23.6 | 9.8 |
| π0-3B(S.) | 41.7 | 12.1 | 11.8 |
| OpenVLA(FT&GPT) | — | 38.8 | 32.4 |
| **InstructVLA-Expert(S.)** | **61.2** | 20.9±0.3 | 20.5±1.0 |
| **InstructVLA-Generalist(S.)** | 54.9 | **48.2±1.3** | **45.6±0.5** |

Expert는 SimplerEnv에서 SpatialVLA 대비 **+33.3%** (61.2 vs 45.9). Generalist는 SimplerEnv-Instruct에서 OpenVLA(FT&GPT) 대비 **+31.7%** (46.9 vs 35.6).

### 3.3 Ablation (Table 3, Fig. 6-7)

- DINOv2 제거 → 평균 SR -50.0% (52.9 → 23.0)
- FiLM 제거 → -15.3% (52.9 → 45.9)
- Language motion 제거 → -9.3% (52.9 → 48.4)
- VLA-IT QA & Captioning 제거(Table 4) → 46.2 → 41.7 (-10.8%)
- Thinking 활성화 시 generalist가 +36.1% 향상 (Fig. 7b)

---

## 4. 한계

1. **LIBERO 본문 미보고**: §A.3 부록에서만 다뤄짐. YAML의 `libero_spatial=96.2 / object=97.0 / goal=92.4 / long=76.8`은 부록 출처로 추정되며 본 paper의 main result는 SimplerEnv 중심.
2. **CALVIN 평가 없음**: YAML의 `calvin_abc_d_avg_len=4.4`는 본 논문에서 확인되지 않음(YAML과 paper 평가 불일치).
3. **Eagle2-2B 의존**: Backbone 변경 시 성능 보존 미검증.
4. **GPT-4o 의존 annotation**: VLA-IT corpus 품질이 GPT-4o의 embodied reasoning 한계에 의존(§4.1에서 본인들도 인정).
5. **Real-world 평가 제한**: WidowX-250(zero-shot) + Franka FR3(few-shot)에 한정(§5.2).

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★★ — VLA-IT 패러다임 + MoE 전환 |
| Technical depth | ★★★★★ — 2-stage training 합리적 분해 |
| Experimental rigor | ★★★★☆ — 3 seed 평균, 풍부한 ablation |
| Practical impact | ★★★★☆ — 1.5B로 7B 모델 능가 |

**강점**: VLM 능력 보존 + manipulation 향상 동시 달성, 광범위한 ablation. **약점**: LIBERO/CALVIN 평가가 본문에 부재, GPT-4o annotation 의존성.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | Magma도 multimodal+VLA 공동학습인데 왜 InstructVLA가 우수? | Magma는 단순 co-training, InstructVLA는 MoE로 task-specific routing + 2-stage(Fig. 6b). +12.5% gap. |
| 2 | SimplerEnv-Instruct는 본 저자들이 만든 self-serving 벤치인가? | 80 task, 3 annotator cross-validation, OOD object. 그러나 self-bias 가능성은 존재(§4.2). |
| 3 | Thinking이 manipulation을 +36.1% 향상하는 메커니즘은? | Latent action이 textual reasoning에 condition됨. 그러나 reasoning quality에 대한 정량 분석 부재. |

<!-- VERIFIED: pdf -->
