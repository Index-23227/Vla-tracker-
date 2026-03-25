# SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model

> **한 줄 요약**: Ego3D Position Encoding과 Adaptive Action Grids로 VLA에 공간적 이해를 부여하여, 1.1M 로봇 에피소드에서 사전학습하고 뛰어난 in-distribution 일반화와 out-of-distribution 적응 능력 달성.

---

## 1. 배경 및 동기

- 기존 VLA의 visual encoding이 **spatial understanding에 취약** → 물체 위치, 깊이, 3D 관계 파악 부족
- 2D image token에 3D spatial information을 주입하는 경량 방법 필요

---

## 2. 방법론

### Ego3D Position Encoding

카메라 ego-centric 좌표계에서 3D position을 visual token에 인코딩:
$$\text{token}_i' = \text{token}_i + \text{PE}_{3D}(x_i, y_i, z_i)$$

- Depth estimation으로 각 pixel의 3D 좌표 추정
- 3D PE를 visual token에 추가하여 spatial awareness 부여

### Adaptive Action Grids

Action space를 task에 따라 적응적으로 discretization:
- Coarse grid for exploration, fine grid for precision

---

## 3. 실험 결과

| 설정 | OpenVLA | SpatialVLA |
|------|---------|-----------|
| In-distribution | 76.5% | 88.3% |
| OOD (new layout) | 35.2% | 52.8% |
| OOD (new objects) | 41.5% | 58.3% |

- **OOD에서 특히 큰 향상** → spatial understanding이 일반화에 기여

---

## 4. 한계 및 미해결 문제

1. **Depth estimation 의존**: 부정확한 depth → 잘못된 3D PE
2. **Adaptive grid의 설계**: Task-specific grid 설정 필요
3. **1.1M episode 규모의 재현성**: 대규모 데이터 수집 비용

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Ego3D PE의 간결한 spatial injection |
| **Practical impact** | ★★★★★ — OOD 일반화 향상 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | GST-VLA와의 차이는? | GST-VLA는 Gaussian primitives로 tokenization 자체를 3D화, SpatialVLA는 기존 token에 3D PE 추가. 더 경량 |
| 2 | Ego3D PE 없이 depth를 추가 채널로 넣으면? | 이 비교가 핵심. PE가 더 효과적임을 ablation에서 보임 |
