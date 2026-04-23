# ViVa: A Video-Generative Value Model for Robot Reinforcement Learning

> **한 줄 요약**: Pretrained video generator를 value function으로 재활용, 현재 관측과 proprioception으로부터 미래 proprioception과 scalar value를 공동 예측하여 long-horizon 로봇 RL의 value 추정을 개선.

---

## 1. 배경 및 동기

- VLA는 large-scale pretraining으로 발전했지만 실세계 배포는 partial observability와 delayed feedback으로 어려움
- RL의 value function이 task progress를 평가하지만, VLM 기반 value 모델은 temporal dynamics 포착에 한계
- Video generator는 spatiotemporal prior가 이미 풍부 → value estimation에 "foresight"를 주입할 수 있는 자연스런 자원

---

## 2. 방법론

### Video-Generative Value Head
Pretrained video generator를 재목적화하여 (observation + proprioception) → (future proprioception, scalar value)를 jointly 예측. Future proprioception 예측이 auxiliary supervision으로 작용하여 value가 "embodiment dynamics에 grounded"되도록 함.

### RECAP Integration
기존 RL framework인 RECAP의 value component를 ViVa로 교체. Real-world box assembly task에서 substantial improvement를 보고.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- 3개 task에서 qualitative analysis 수행 (real-world box assembly 포함)
- Abstract 주장: 더 reliable한 value signal, novel object로의 generalization

---

## 4. 한계 및 미해결 문제

1. **추론 비용**: Video generator inference는 frame당 수백 ms ~ 수 초 → policy update step당 value query 비용이 realtime control에 부담.
2. **Scalar value 형성 방식**: Video latent로부터 단일 scalar를 어떻게 뽑는지(pooling/regression head) abstract에서 불명확.
3. **Benchmark 다양성**: Real-world box assembly 외 simulation benchmark(LIBERO/SimplerEnv 등) 수치가 공개되지 않아 comparative positioning 어려움.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Video generator를 value model로 재활용하는 관점은 최근 "world model as critic" trend에서 differentiated |
| **Practical impact** | ★★★☆☆ — Long-horizon real-world task에서 유망, 추론 비용 때문에 on-robot loop보다는 offline RL/eval에서 가치 |

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Video generator를 value로 쓰면 단순 VLM critic 대비 이득은? | Spatiotemporal prior 덕분에 "미래 상태"를 implicit simulate → partial observability/delayed feedback 환경에서 task progress 추정이 안정. |
| 2 | Future proprioception prediction이 왜 필요한가? | Value head가 단순 reward regression에 overfit되지 않도록 dynamics-grounded auxiliary signal 역할. |
| 3 | Inference 비용을 어떻게 감당? | 실시간 policy에는 과부하 → value를 낮은 빈도(episode/chunk 단위)로 query하거나 distilled critic을 학습하는 방향이 현실적. |

<!-- VERIFIED: abstract-only -->
