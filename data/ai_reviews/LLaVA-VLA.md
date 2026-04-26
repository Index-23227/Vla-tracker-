# LLaVA-VLA: Rethinking the Practicality of VLA — A Comprehensive Benchmark and Improved Baseline

> **한 줄 요약**: HKUST(GZ)/HUST/Westlake가 제안한 **lightweight VLA + CEBench 벤치마크**. LLaVA-OneVision-0.5B 백본 + Qwen-2 0.5B + chunk-5 discrete action tokenizer. CALVIN avg 3.65, 작은 모델로 7B+ 모델에 근접. Open-source.

---

## 1. 배경 및 동기

- 기존 VLA 모델은 **costly pre-training + 다양한 시나리오 적용성 제한**으로 실용성에 의문.
- 단일/양팔/모바일 manipulation을 모두 다루는 **포괄적 cross-embodiment benchmark가 부재**.
- 3가지 핵심 질문(Q1–Q3):
  - Q1: 모델 크기 vs 성능, small model이 large model 따라잡는 기법?
  - Q2: small model에 pre-training이 필수인가?
  - Q3: 고정-base + mobile cross-embodiment hybrid action space 정의 방법?

---

## 2. 방법론

### CEBench (Cross-Embodiment Benchmark) — 새 벤치마크
- Sim 14.4k demos 수집:
  - **Single-arm**: CALVIN (4 environments A/B/C/D + DR)
  - **Bimanual**: RoboTwin 8 tasks (lift pot, place bottles, click bell 등) + DR (unseen settings)
  - **Mobile manipulation**: real-world
- "포괄적 + 실용적" 평가 프로토콜.

### LLaVA-VLA (improved baseline)
- **Backbone**: LLaVA-OneVision-0.5B (small VLM 시리즈)
- **LLM**: Qwen-2 0.5B (LLaVA-OneVision 내부)
- **Action head**: discrete action tokenizer + **chunk size 5**
- **Training**: 2-stage post-training (multi-task data → robot-specific 파인튜닝)
- **Hybrid action space**: 고정-base + mobile 통합 (Q3 답변)
- **Total**: ~0.5B params, consumer-grade GPU 배포 가능.

---

## 3. 실험 결과

### Table I — CEBench (CALVIN ABC→D 5-task chain)

| 모델 | Param | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg.Len** |
|------|------:|----:|----:|----:|----:|----:|----:|
| **LLaVA-VLA (ours)** | **0.5B** | **96.2** | **84.8** | **72.6** | **60.8** | **50.6** | **3.65** |

→ 0.5B로 7B+ 모델들과 경쟁력 있는 chain length 달성.

### RoboTwin (8 bimanual tasks)
- Seen settings + DR(unseen) 모두 평가.
- LLaVA-VLA가 small-scale에도 baseline 대비 우수.

### Real-world Mobile Manipulation
- 단일 small VLA로 mobile + manipulation 통합 demo (LLaVA-VLA가 최초).

---

## 4. 한계 및 미해결 문제

1. **0.5B 모델 한계**: 매우 long-horizon (10+ subtasks) 시나리오에서 degradation 가능성. 5/5 chain 50.6%는 여전히 절반 수준.
2. **Pre-training scope**: small model에는 large-scale pre-training이 비효율적임을 보였으나, 어디까지 일반화될지 불명.
3. **Action space hybrid 정의**: 본문은 정의 제안하나 더 다양한 robot morphology (humanoid, dexterous)으로의 전이 미검증.
4. **Comparison fairness**: CEBench 자체를 저자가 제안 → 외부 베이스라인이 동일 evaluation에 fine-tune되지 않은 경우 비교 의미 제한.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — small-scale practicality에 집중한 contribution 명확. CEBench는 도구로서 가치 있음. |
| **Practical impact** | ★★★★☆ — 0.5B로 consumer GPU에서 multi-embodiment 가능 → 실배포 적합. Open-source. |

**강점**: open-source (github.com/OpenHelix-Team/LLaVA-VLA), small + mobile 가능, 새 벤치마크 contribution.
**약점**: small model 한계로 long-horizon 5/5에서 50%대, CEBench 자체 외 외부 검증 데이터 부족.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 0.5B로 정말 7B를 따라잡나? | CALVIN 1/5에서 96.2 → 5/5에서 50.6으로 long horizon에서 격차 발생. 1-2 subtask까지는 경쟁력, 5-step chain은 large 모델 우위. |
| 2 | Discrete action tokenizer + chunk-5의 의미? | OpenVLA의 single-step discrete tokenizer를 chunk으로 확장. Action smoothness + inference 효율 trade-off. |
| 3 | CEBench의 차별점? | CALVIN/RoboTwin/real-world를 cross-embodiment 통합 + DR 설정 포함. 단 외부 모델이 동일 환경에서 evaluate되지 않으면 비교 한계. |
| 4 | TinyVLA와 차이? | TinyVLA는 ~93M, LLaVA-VLA는 0.5B. LLaVA-OneVision의 multi-image 강점 활용 + chunk action. TinyVLA는 단일 이미지 + diffusion head. |

<!-- VERIFIED: pdf -->
