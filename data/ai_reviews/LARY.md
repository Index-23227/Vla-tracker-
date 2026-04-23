# LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment

> **한 줄 요약**: 대규모 human action video (1M+, 151 categories)로부터 학습된 **latent action** 표현의 품질을 체계적으로 평가하는 벤치마크로, 일반 비주얼 foundation model이 특화된 embodied model보다 latent-action encoder로 더 우수함을 보인 연구.

---

## 1. 배경 및 동기

- VLA 모델은 명시적 action label 데이터가 부족 → human action video가 scalable alternative.
- 그러나 human video는 ontology-independent한 표현(= latent action)으로 변환해야 로봇에 전이 가능.
- **핵심 질문**: 어떤 visual encoder가 좋은 latent action 표현을 산출하는가? → 이를 측정할 표준 벤치마크가 없었음.

---

## 2. 방법론

### Latent Action Representation Yielding (LARY) 벤치마크
- 1M+ human action videos, 151 action categories를 커버하는 대규모 평가 프로토콜.
- 두 축으로 평가: (a) **semantic action understanding** (e.g., action classification), (b) **generalizable vision-to-action alignment** (downstream 로봇 제어).

### Latent vs Pixel Space
- Pixel-based video prediction보다 latent-based visual space가 physical action space와 구조적으로 더 잘 정렬됨을 실증.
- 일반 visual foundation model (DINO/CLIP-류) vs specialized embodied encoder를 비교.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- 1M+ human action videos, 151 categories 규모의 평가.
- 일반 visual foundation model이 specialized embodied model을 **상회**하여 더 좋은 latent action encoder로 기능.
- Latent-based space가 pixel-based space 대비 action space와 더 잘 aligned.

---

## 4. 한계 및 미해결 문제

1. **Human→Robot embodiment gap**: Latent action이 ontology-independent하다 해도 최종 robot motor command로의 decoding에서 오차가 발생 가능.
2. **Semantic action category의 대표성**: 151 category가 실제 로봇 작업의 long-tail을 얼마나 커버하는가?
3. **Benchmark로서의 지속성**: 일반 visual FM이 빠르게 진화 → 평가 기준이 빠르게 구식화될 리스크.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Latent action 평가를 체계화한 benchmark contribution |
| **Practical impact** | ★★★★☆ — VLA pretraining encoder 선택에 직접적 가이드 제공 |

**강점**: Latent action이라는 다소 모호했던 개념을 측정 가능한 axis로 내림. **약점**: 자체 새로운 모델이 아닌 evaluation framework이므로 다운스트림 검증 깊이에 의존.

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 general visual FM이 embodied model보다 좋았는가? | 데이터 규모·다양성 우위, embodied pretraining이 과적합되는 경향 |
| 2 | Latent action은 어떻게 정의되는가? | Ontology-independent, embodiment-agnostic한 action의 분포적 표현 |
| 3 | 이 결과가 LAPA, UniPi 같은 latent-action VLA에 어떤 시사? | Encoder 백본을 embodied 전용이 아닌 general FM으로 대체해볼 가치 |
| 4 | Pixel-based video prediction(GR-1, UniPi)은 낡은 접근인가? | Latent space의 alignment 우위를 보였으므로, latent-diffusion 계열이 더 유망할 수 있다 |

<!-- VERIFIED: abstract-only -->
