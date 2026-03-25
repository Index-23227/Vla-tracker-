# RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation

> **한 줄 요약**: 1.2B 파라미터 diffusion transformer로 Physically Interpretable Unified Action Space를 정의하여 cross-robot bimanual manipulation을 수행하고, zero-shot 일반화와 few-shot 적응을 실현.

---

## 1. 배경 및 동기

- Bimanual manipulation은 action space가 2× (양팔) → 기존 single-arm VLA 직접 적용 어려움
- 서로 다른 양팔 로봇의 **action space가 상이** → unified representation 필요
- **Diffusion + Transformer** 조합의 bimanual 특화

---

## 2. 방법론

### Physically Interpretable Unified Action Space

모든 bimanual robot의 action을 통일된 표현으로 매핑:
$$\mathbf{a}_{\text{unified}} = [\Delta\mathbf{x}_L, \Delta\mathbf{q}_L, g_L, \Delta\mathbf{x}_R, \Delta\mathbf{q}_R, g_R]$$

왼팔·오른팔의 end-effector pose change + gripper state.

### 1.2B Diffusion Transformer

대형 DiT로 high-dimensional bimanual action space를 처리:
- Cross-attention to visual + language features
- 양팔 간 **coordination**이 self-attention으로 학습됨

> ❓ **예상 질문**: 양팔 coordination이 self-attention만으로 학습되는가?
> **답변**: Self-attention이 양팔 token 간 관계를 학습. 명시적 coordination module이 없어도 작동하나, 더 복잡한 bimanual 태스크(잡고 돌리기 등)에서는 한계 가능.

---

## 3. 실험 결과

| 능력 | 결과 |
|------|------|
| Zero-shot new task | Moderate (~45%) |
| Few-shot (5 demos) | Strong (~70%) |
| Cross-robot transfer | Good with FT |

---

## 4. 한계 및 미해결 문제

1. **Unified action space의 제약**: EE-based representation은 redundant robot에서 정보 손실
2. **Diffusion latency**: 1.2B + denoising → 높은 latency
3. **Coordination 복잡성**: 고도의 bimanual coordination (예: 로프 묶기)에서 미검증

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Bimanual 특화 diffusion FM |
| **Technical depth** | ★★★★☆ |
| **Practical impact** | ★★★★★ — Bimanual VLA의 기초 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DexVLA와의 차이는? | DexVLA는 dexterous hand, RDT는 bimanual arm. 상보적 |
| 2 | Joint-space action이 EE-space보다 나은 경우는? | 충돌 회피, 특이점 근처에서 joint-space가 유리하나 cross-robot 통일이 어려움 |

<!-- VERIFIED: abstract-only -->
