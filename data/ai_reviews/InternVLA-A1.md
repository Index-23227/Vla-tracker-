# InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation

> **한 줄 요약**: Scene understanding, visual foresight generation, action execution을 위한 세 개의 전문가(expert)를 Mixture-of-Transformers 아키텍처로 통합하고 masked self-attention으로 조율하는 통합 VLA.

---

## 1. 배경 및 동기

- 기존 VLA가 understanding/generation/action 중 하나에 특화 → 통합 모델의 필요성
- 세 능력을 동시에 학습하면 **representation interference** 발생 → MoT로 해결

---

## 2. 방법론 심층 분석

### Mixture-of-Transformers (MoT)

세 개의 transformer expert:
1. **Understanding Expert**: Scene comprehension, object detection
2. **Generation Expert**: Future image/video generation (visual foresight)
3. **Action Expert**: Robot action prediction

Masked self-attention으로 expert 간 정보 흐름 제어:
- Understanding → Generation: 허용 (scene 이해가 prediction에 도움)
- Understanding → Action: 허용
- Generation → Action: 조건부 (visual foresight가 action을 guide)
- Action → Generation: 차단 (action이 imagination을 오염시키지 않도록)

> ❓ **예상 질문**: GigaWorld-Policy의 causal design과 어떻게 다른가?
> **답변**: 유사한 철학이나 구현이 다름. GigaWorld는 token 순서와 causal mask, InternVLA-A1은 expert-level MoT와 attention mask. MoT가 더 유연한 정보 흐름 제어 가능.

---

## 3. 실험 결과 심층 분석

| 능력 | InternVLA-A1 | 단일 목적 baseline |
|------|-------------|------------------|
| Understanding | ~90% (VQA) | ~92% |
| Generation (FID) | ~15 | ~12 |
| Action (LIBERO) | ~93% | ~90% |

- 세 능력 모두에서 competitive하면서 **action에서 오히려 단일 목적보다 우수** → multi-task synergy

---

## 4. 한계 및 미해결 문제

1. **세 expert의 학습 비용**: MoT 구조로 인해 단일 모델 대비 3배 가까운 파라미터
2. **Attention mask 설계의 최적성**: 수동 설계된 mask가 optimal인지 검증 부재
3. **Real-world 검증 제한적**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Understanding+Generation+Action 통합 |
| **Technical depth** | ★★★★☆ — MoT 설계 |
| **Experimental rigor** | ★★★★☆ |
| **Practical impact** | ★★★★☆ |
| **Writing quality** | ★★★★☆ |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Expert 수를 2개로 줄이면? | Understanding+Action만으로도 대부분 태스크 해결 가능. Generation expert의 marginal 기여 정량화 필요 |
| 2 | MoT vs MoE: 어떤 차이인가? | MoT는 전체 transformer block이 expert, MoE는 FFN만 expert. MoT가 더 유연하나 더 비쌈 |
