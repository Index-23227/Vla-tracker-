# UniPi: Learning Universal Policies via Text-Guided Video Generation

> **한 줄 요약**: 순차적 의사결정을 text-conditioned video generation 문제로 재정의하여, 생성된 비디오에서 inverse dynamics로 action을 추출하는 **video-as-policy** 패러다임의 선구적 연구.

---

## 1. 배경 및 동기

- 대규모 text-to-video 모델의 등장 → 이를 정책으로 활용 가능?
- Video = future plan → 비디오를 생성하면 implicit한 plan이 됨

---

## 2. 방법론

### Video-as-Policy

$$\text{Text instruction} \to \text{Video generation model} \to \text{Future frames} \to \text{Inverse dynamics} \to \text{Actions}$$

1. Text-conditioned diffusion model이 미래 비디오 생성
2. 생성된 비디오의 연속 프레임에서 inverse dynamics model로 action 추출

> ❓ **예상 질문**: Video generation의 latency가 real-time control에 적합한가?
> **답변**: 가장 큰 약점. 비디오 생성에 수 초 소요 → 실시간 제어 불가. Hierarchical (저주파 plan + 고주파 execution) 구조로 완화 가능.

---

## 3. 실험 결과

- 시뮬레이션 환경에서 combinatorial generalization 시연
- 학습에 없던 (goal, environment) 조합에서 zero-shot 성능

---

## 4. 한계 및 미해결 문제

1. **극도로 높은 latency**: Video generation 수 초
2. **Inverse dynamics의 정확도**: 생성된 비디오에서 action 추출의 오류
3. **Video quality → action quality**: 잘못된 비디오 = 잘못된 plan

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Video-as-policy의 개념 자체가 revolutionary |
| **Practical impact** | ★★★☆☆ — Latency가 실용성 제한 |

**강점**: 패러다임 정의적 연구. GR-1, Motus 등 모든 WAM 연구의 출발점. **약점**: Latency.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Latent video (pixel 아닌)로 전환하면? | FLARE, Fast-WAM의 접근. Latency 대폭 감소 |

<!-- VERIFIED: abstract-only -->
