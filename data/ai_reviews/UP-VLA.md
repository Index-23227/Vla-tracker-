# UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent

> **한 줄 요약**: Multi-modal understanding과 future prediction 목적함수를 통합하여 VLM의 low-level spatial 정보 캡처 능력을 보강, CALVIN ABC-D에서 33% 향상과 real-world manipulation 개선.

---

## 1. 배경 및 동기

- VLM은 **high-level semantic**에 강하지만 **low-level spatial detail** (정확한 위치, 미세 형상)에 약함
- Future prediction (다음 상태 예측)이 low-level feature 학습을 촉진

---

## 2. 방법론

### Understanding + Prediction Integration

- **Understanding branch**: Scene description, object grounding
- **Prediction branch**: Next-frame/state prediction
- Joint training으로 understanding이 prediction을 돕고, prediction이 spatial detail 학습을 촉진

---

## 3. 실험 결과

| 모델 | CALVIN ABC-D Avg Len |
|------|---------------------|
| GR-1 | 3.86 |
| OpenVLA | 3.45 |
| **UP-VLA** | **4.58 (+33%)** |

- CALVIN에서 33% 향상은 상당한 수준

---

## 4. 한계

1. 두 branch의 학습 비용
2. Prediction quality → action quality의 인과관계 명확화 필요
3. Real-world 결과의 depth

---

## 5. 총평

| **Novelty** | ★★★★☆ | **Practical impact** | ★★★★☆ |

---

## 6. 🔥 질문 모음

| 질문 | 답변 |
|------|------|
| FLARE와의 차이? | FLARE는 latent prediction, UP-VLA는 pixel-level prediction. 더 explicit한 supervision |
