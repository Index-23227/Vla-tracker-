# RoboDual: Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation

> **한 줄 요약**: VLA-based generalist (System 2)와 diffusion-transformer specialist (System 1)를 20M 학습 가능 파라미터로 결합하여, real-world 26.7% 향상과 CALVIN 12% 향상을 3.8x 높은 제어 주파수에서 달성.

---

## 1. 배경 및 동기

- VLA generalist: 넓은 일반화 but 느린 추론
- Diffusion specialist: 빠르고 정밀 but 일반화 약함
- **두 시스템의 시너지**로 속도와 일반화를 동시에

---

## 2. 방법론

### Dual-System Integration

**System 2 (VLA, ~7B)**: High-level reasoning, task understanding → hidden state
**System 1 (DiT, ~200M)**: Low-level action generation, high-frequency control

System 2가 **간헐적으로** reasoning을 수행하고 (매 N step), System 1이 **매 step** action 생성:
$$\text{System 2: } \mathbf{h} = \text{VLA}(\mathbf{o}_t, \mathbf{l}) \quad (\text{every N steps})$$
$$\text{System 1: } \mathbf{a}_t = D_\theta(\mathbf{a}_t^k, k, \mathbf{h}) \quad (\text{every step})$$

> ❓ **예상 질문**: System 2의 실행 주기 N은 어떻게 결정하는가?
> **답변**: 태스크 복잡도에 따라 조절. 단순 이동: N=10 (100ms마다 VLA), 복잡 파지: N=3 (빈번한 VLA). Adaptive N은 미구현.

### 20M Trainable Parameters

VLA는 **frozen**, adapter + DiT만 학습:
- 메모리·연산 효율적
- VLA의 기존 language/vision 능력 보존

---

## 3. 실험 결과

| 모델 | Real-world SR (%) | CALVIN Avg Len | Control Hz |
|------|------------------|---------------|-----------|
| OpenVLA | 52.3 | 3.45 | 5 Hz |
| CogACT | - | - | 10 Hz |
| **RoboDual** | **79.0 (+26.7%)** | **3.86 (+12%)** | **19 Hz** |

- **3.8x 높은 제어 주파수**가 핵심 이점 → reactive manipulation 가능

---

## 4. 한계 및 미해결 문제

1. **System 2 지연 시**: VLA 추론이 지연되면 System 1이 stale context로 action 생성
2. **N의 최적화**: 고정 N이 suboptimal. Event-driven triggering이 바람직
3. **두 시스템 간 alignment**: System 2 hidden state가 System 1에 최적 conditioning인지 불확실

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Dual-system with 20M trainable params |
| **Practical impact** | ★★★★★ — 고주파 제어의 실용적 해결 |

**강점**: 속도와 일반화의 최적 절충. 20M trainable params의 효율성. **약점**: Stale context 위험, adaptive N 미구현.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | GR00T-N1의 System 1/2와 어떻게 다른가? | GR00T은 joint end-to-end 학습, RoboDual은 VLA frozen + adapter only |
| 2 | N=1 (매 step VLA)과 N=10의 성능 차이는? | N=1이 성능 최고이나 속도 최저. Trade-off 분석 포함 |

<!-- VERIFIED: abstract-only -->
