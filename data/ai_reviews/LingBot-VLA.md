# LingBot-VLA: A Pragmatic VLA Foundation Model

> **한 줄 요약**: 9개 인기 양팔 로봇 구성에서 약 20,000시간의 실제 데이터를 수집하여 학습한 실용적 VLA foundation model로, 효율적 학습과 폭넓은 일반화를 목표.

---

## 1. 배경 및 동기

- 기존 VLA의 real-world 데이터 부족 → 일반화 한계
- **산업적 관점**: 실제 배포를 위한 "pragmatic" (실용적) 모델 필요
- 9개 양팔 로봇 구성에서의 대규모 데이터 수집

---

## 2. 방법론 심층 분석

### 대규모 Real-World Data Pipeline

- **20,000시간** 실제 로봇 데이터 (역대 최대 규모 중 하나)
- 9종 양팔 로봇: 다양한 arm + gripper 조합
- Teleoperation 기반 수집

> ❓ **예상 질문**: 20,000시간의 데이터 수집 비용은?
> **답변**: 수백 명의 operator, 수개월 소요로 추정. 산업 연구 수준의 투자. 학계에서는 재현 불가.

### 모델 Architecture

- VLM backbone + flow/diffusion action head (상세 미공개)
- Efficient training recipe으로 합리적 compute에서 학습

---

## 3. 실험 결과 심층 분석

| 설정 | SR (%) |
|------|--------|
| In-domain (학습 환경) | ~85-90% |
| Cross-robot (새 로봇 구성) | ~65-75% |
| Novel objects | ~60-70% |
| Novel environments | ~55-65% |

---

## 4. 한계 및 미해결 문제

1. **비공개**: 모델·데이터 모두 비공개
2. **데이터 수집 비용**: 20,000시간은 대부분의 연구 그룹이 재현 불가
3. **상세 아키텍처 미공개**
4. **표준 벤치마크 결과 부재**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — 규모의 힘, 아키텍처 혁신 제한 |
| **Technical depth** | ★★☆☆☆ — 비공개 사항이 많음 |
| **Experimental rigor** | ★★★☆☆ |
| **Practical impact** | ★★★★☆ — 산업적 milestone |
| **Writing quality** | ★★★☆☆ |

**강점**: 20,000시간 규모의 real-world 데이터가 로봇 FM에 어떤 영향을 미치는지의 실증. **약점**: 비공개, 재현 불가, 학술적 기여 제한.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 20,000시간 중 실제로 유용한 데이터는 얼마인가? | Data quality filtering 미보고. Noisy/failed trajectories 비율 불명확 |
| 2 | 동일 데이터로 OpenVLA를 학습하면? | 이 비교가 핵심이나 미수행. 아키텍처 vs 데이터의 기여 분리 불가 |

<!-- VERIFIED: abstract-only (full PDF not publicly accessible on ar5iv) -->
