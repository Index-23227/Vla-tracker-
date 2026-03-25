# TacVLA: Contact-Aware Tactile Fusion for Robust VLA Manipulation

> **한 줄 요약**: 촉각(tactile) modality를 contact-aware gating mechanism으로 VLA에 통합하여, 접촉이 중요한 태스크(분해 20%↑, in-box picking 60%↑)와 시각 차단 상황(2.1x 향상)에서 큰 성능 향상 달성.

---

## 1. 배경 및 동기

- 기존 VLA는 **vision + language만 사용** → 접촉 정보 부재
- 접촉이 필수인 태스크(삽입, 분해, 재질 판단)에서 시각만으로 한계
- **촉각 센서**가 보편화되고 있으나 VLA 통합 사례 부재

---

## 2. 방법론

### Contact-Aware Gating

촉각 token을 항상 활성화하지 않고, **접촉이 감지될 때만 선택적으로 활성화**:

$$\mathbf{h} = g(\mathbf{f}_{\text{tac}}) \cdot \text{TacEncoder}(\mathbf{f}_{\text{tac}}) + (1 - g) \cdot \mathbf{0}$$

- $g$: Contact gating function (접촉 시 1, 비접촉 시 0에 가까움)
- 비접촉 phase에서 불필요한 tactile noise 차단

> ❓ **예상 질문**: Gating을 고정 threshold로 하는가, 학습하는가?
> **답변**: 학습된 gating. 접촉 데이터에서 자동으로 contact/non-contact 구분 학습. 다만 gating 오류(false positive/negative)의 영향 분석 부족.

---

## 3. 실험 결과

| 태스크 | Vision-only VLA | **TacVLA** |
|-------|----------------|-----------|
| 분해 (disassembly) | 55% | **75% (+20%)** |
| In-box picking | 25% | **85% (+60%)** |
| Visual occlusion | 30% | **63% (2.1x)** |

- **시각 차단에서 촉각의 가치가 극명하게 드러남**

---

## 4. 한계 및 미해결 문제

1. **촉각 센서 하드웨어 의존**: 특정 센서(GelSight 등)에 맞춤 → 범용성 제한
2. **촉각 데이터 수집**: 기존 VLA 데이터에 촉각 미포함 → 새로운 데이터 수집 필요
3. **Cross-embodiment**: 촉각 센서가 다른 로봇에서의 일반화 미검증

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Tactile + VLA 최초 통합 |
| **Practical impact** | ★★★★★ — Contact-rich task의 혁신적 향상 |

**강점**: 촉각의 VLA 통합이 선구적. Contact-aware gating이 효과적. **약점**: 하드웨어 의존, 데이터 수집 비용.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 어떤 촉각 센서에서도 작동하는가? | Sensor-specific encoder 필요. Universal tactile tokenizer는 미탐구 |
| 2 | Force/torque sensor로도 비슷한 효과인가? | F/T는 scalar로 fine-grained spatial 정보 부족. Vision-based tactile이 더 풍부 |

<!-- VERIFIED: abstract-only -->
