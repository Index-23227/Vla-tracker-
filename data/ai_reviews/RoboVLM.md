# RoboVLMs: What Matters in Building VLA Models for Generalist Robots

> **한 줄 요약**: 8개 VLM backbone, 4개 policy architecture, 600+ 실험을 통해 VLA 설계의 핵심 요인을 체계적으로 분석하고, 이를 바탕으로 구축한 RoboVLMs가 manipulation 벤치마크에서 SOTA 달성.

---

## 1. 배경 및 동기

- VLA 설계 공간(backbone, action head, tokenization 등)이 방대하나 **체계적 비교 연구 부재**
- 각 논문이 자체 설정에서만 평가 → "어떤 선택이 진짜 중요한가?"에 대한 답이 없음

---

## 2. 방법론: 대규모 실험적 분석

### 탐구된 설계 축

| 축 | 선택지 |
|---|--------|
| VLM Backbone | LLaVA, Prismatic, InternVL, Qwen-VL, Idefics, PaLI, BLIP-2, Fuyu |
| Action Head | MLP, GMM, Diffusion, Token prediction |
| Action Representation | Discrete (256/1024 bins), continuous, FAST |
| Training Strategy | Frozen VLM, LoRA, full FT |

### 핵심 발견

1. **VLM backbone 선택이 가장 중요** (action head보다 중요)
2. **Spatial feature가 풍부한 VLM** (DINOv2 포함)이 action에 유리
3. **Diffusion head > MLP > token prediction** (일반적으로)
4. **Full fine-tuning > LoRA** (성능은 높으나 비용 trade-off)

---

## 3. 실험 결과

| 최적 설정 RoboVLMs | LIBERO (%) | SimplerEnv (%) |
|-------------------|-----------|---------------|
| Best combination | 94.8 | 72.3 |
| Worst combination | 62.1 | 38.5 |

- **최적 vs 최악의 설계 조합: 32%p+ 차이** → 설계 선택이 결정적

---

## 4. 한계 및 미해결 문제

1. **Compute 비용**: 600+ 실험에 수만 GPU-hours 소요 → 재현 어려움
2. **벤치마크 편향**: 특정 벤치마크에서의 최적이 다른 환경에서도 최적인지 불확실
3. **시간에 따른 변화**: 새로운 VLM/기법의 등장으로 결론이 빠르게 outdated될 수 있음

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — VLA 설계의 대규모 체계적 분석 |
| **Technical depth** | ★★★★★ — 600+ 실험 |
| **Practical impact** | ★★★★★ — VLA 연구의 실용적 가이드 |

**강점**: "무엇이 중요한가?"에 대한 실증적 답변. 커뮤니티에 엄청난 가치. **약점**: 결론의 시효성.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 이 결론이 6개월 후에도 유효한가? | 핵심 원칙(backbone 중요성, spatial feature)은 유지될 가능성. 구체적 최적 조합은 변할 수 있음 |
| 2 | Real-world에서도 동일한 설계 축 중요도인가? | Sim과 real에서 다를 수 있음. Real-world 대규모 분석은 미수행 |
