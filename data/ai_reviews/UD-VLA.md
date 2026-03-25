# UD-VLA: Unified Diffusion VLA via Joint Discrete Denoising Diffusion Process

> **한 줄 요약**: 다중 modality(language reasoning + action)를 단일 이산 denoising diffusion trajectory에서 jointly 생성하는 통합 프레임워크로, CALVIN·LIBERO·SimplerEnv에서 SOTA 달성과 autoregressive 대비 4x 빠른 추론.

---

## 1. 배경 및 동기

- Language reasoning (text tokens)과 action prediction (action tokens)이 별도 디코딩 → 비효율
- **Joint diffusion**: 두 modality를 하나의 denoising process에서 동시 생성

---

## 2. 방법론

### Joint Discrete Denoising

$$p(\mathbf{r}_0, \mathbf{a}_0 | \mathbf{o}) = \int p(\mathbf{r}_t, \mathbf{a}_t \to \mathbf{r}_0, \mathbf{a}_0 | \mathbf{o}) d(\mathbf{r}_t, \mathbf{a}_t)$$

Reasoning tokens과 action tokens이 동일 noise schedule에서 jointly denoised.

> ❓ **예상 질문**: Reasoning과 action이 같은 noise level에서 denoised되어야 하는 이유는?
> **답변**: Joint denoising으로 두 modality 간 mutual conditioning 가능. Reasoning이 action을 가이드하고, action이 reasoning을 grounding.

---

## 3. 실험 결과

| 모델 | CALVIN | LIBERO | SimplerEnv | Speed |
|------|--------|--------|-----------|-------|
| OpenVLA | 3.45 | 76.5% | 48.7% | 1x |
| **UD-VLA** | **4.2+** | **95+%** | **70+%** | **4x** |

- **4x 빠른 추론**: Joint parallel denoising이 sequential autoregressive보다 효율적

---

## 4. 한계 및 미해결 문제

1. **Joint noise schedule**: 최적 schedule 탐색 필요
2. **Reasoning 품질**: Diffusion으로 text 생성의 품질이 autoregressive 대비 낮을 수 있음
3. **Standard text 생성과의 차이**: Diffusion text generation은 아직 mature하지 않음

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Joint diffusion for multi-modal VLA |
| **Practical impact** | ★★★★★ — 4x speedup |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Text diffusion의 품질이 AR 대비 열위면? | 현재 discrete diffusion LM의 text quality가 AR에 미치지 못할 수 있음. Reasoning 품질 검증 필요 |

<!-- VERIFIED: abstract-only -->
