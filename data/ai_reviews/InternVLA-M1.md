# InternVLA-M1: A Spatially Guided VLA Framework for Generalist Robot Policy

> **한 줄 요약**: Spatial grounding을 language instruction과 robot action을 연결하는 핵심 고리로 설정하고, 2-stage pipeline (spatial reasoning → action generation)으로 일반화된 instruction following 달성.

---

## 1. 배경 및 동기

- Language instruction "빨간 컵을 집어"에서 action으로의 직접 매핑은 **spatial grounding** (빨간 컵의 위치)이 implicit → 오류 진단 어려움
- Spatial grounding을 explicit 중간 표현으로 활용하면 해석가능성 + 정확도 향상

---

## 2. 방법론 심층 분석

### 2-Stage Pipeline

**Stage 1: Spatial Reasoning**
$$\text{VLM}(\text{image}, \text{instruction}) \to \text{spatial grounding} (\text{bbox}, \text{keypoint}, \text{direction})$$

**Stage 2: Spatially-Conditioned Action**
$$\text{Action Head}(\text{image}, \text{spatial grounding}) \to \text{action}$$

> ❓ **예상 질문**: 2-stage의 error cascading은?
> **답변**: Stage 1의 grounding 오류가 Stage 2에 직접 전파. 이를 완화하기 위해 end-to-end fine-tuning을 수행하나, cascading risk은 구조적으로 존재.

---

## 3. 실험 결과

| 설정 | Direct VLA | InternVLA-M1 |
|------|-----------|-------------|
| Seen objects | 82.3% | 88.7% |
| Unseen objects | 51.2% | 68.5% |
| Novel instructions | 45.8% | 62.3% |

- **Unseen objects와 novel instructions에서 큰 향상** → spatial grounding이 일반화에 기여

---

## 4. 한계 및 미해결 문제

1. **2-stage latency**: Spatial reasoning + action generation의 순차 실행
2. **Grounding 오류**: VLM의 spatial grounding 정확도에 전체 성능이 의존
3. **Contact-level precision**: Bbox/keypoint 수준 grounding이 mm 단위 정밀도에 부족
4. **Standard benchmark 비교 부족**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Spatial grounding as bridge |
| **Technical depth** | ★★★★☆ |
| **Experimental rigor** | ★★★☆☆ |
| **Practical impact** | ★★★★☆ — Interpretable VLA |
| **Writing quality** | ★★★★☆ |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Grounding 없이 직접 action을 예측하면 얼마나 차이나는가? | Ablation에서 ~10-15%p 하락. Grounding의 가치 입증 |
| 2 | 3D grounding (depth 포함)으로 확장하면? | GST-VLA와의 접점. 3D grounding이 더 유효할 가능성 |
| 3 | Stage 1과 Stage 2를 parallel로 실행할 수 있는가? | 구조적으로 불가. Stage 2가 Stage 1 출력에 의존 |
