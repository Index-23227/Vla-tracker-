# SuSIE: Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models

> **한 줄 요약**: Image-editing diffusion model (InstructPix2Pix)을 high-level planner로 활용하여 현재 관측에서 중간 subgoal 이미지를 생성하고, low-level policy가 이를 달성하는 2-stage 구조로 CALVIN에서 기존 language-conditioned policy 대비 큰 일반화 달성.

---

## 1. 배경 및 동기

- Language instruction → action의 직접 매핑이 어려운 이유: semantic gap
- **Image subgoal**: "다음에 어떤 모습이어야 하는가"를 이미지로 표현하면 policy에 더 직접적

---

## 2. 방법론

### 2-Stage Architecture

**Stage 1 (High-level)**: Image-editing diffusion model
$$\hat{\mathbf{o}}_{t+k} = \text{InstructPix2Pix}(\mathbf{o}_t, \text{"pick up the red cup"})$$

**Stage 2 (Low-level)**: Goal-conditioned policy
$$\mathbf{a}_t = \pi(\mathbf{o}_t, \hat{\mathbf{o}}_{t+k})$$

> ❓ **예상 질문**: Image editing model의 출력 품질이 low-level policy에 얼마나 중요한가?
> **답변**: Pixel-perfect일 필요 없음. 대략적 spatial layout과 물체 위치만 정확하면 됨. 그러나 심각한 hallucination은 policy를 혼란시킴.

---

## 3. 실험 결과

| 모델 | CALVIN Avg Len |
|------|---------------|
| HULC | 2.64 |
| RT-2 style | 2.85 |
| **SuSIE** | **3.42** |

---

## 4. 한계 및 미해결 문제

1. **2-stage latency**: Diffusion-based image generation이 매우 느림
2. **Subgoal timing**: k step 후의 subgoal을 생성해야 하는데, 최적 k 결정이 어려움
3. **Image editing 품질**: 로봇 환경 특화 fine-tuning 필요할 수 있음

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Pre-trained image editing as planner |
| **Practical impact** | ★★★★☆ — Zero-shot generalization |

**강점**: Pre-trained vision model의 knowledge를 planning에 직접 활용하는 창의적 접근. **약점**: Latency, subgoal timing.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | CoT-VLA와의 차이는? | CoT-VLA는 single model에서 visual CoT, SuSIE는 별도 diffusion model. SuSIE가 더 high-quality 이미지 생성 but 더 느림 |
| 2 | Subgoal image를 latent space에서 생성하면? | FLARE의 접근과 유사. Latency 대폭 감소 가능 |

<!-- VERIFIED: abstract-only -->
