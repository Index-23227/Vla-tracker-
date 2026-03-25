# TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation

> **한 줄 요약**: 사전학습 단계를 제거하고 빠른 추론과 데이터 효율성에 초점을 맞춘 compact VLA로, OpenVLA를 성능에서 match하면서 언어·물체·환경 변화에 대한 강건한 일반화 시현.

---

## 1. 배경 및 동기

- 대형 VLA의 사전학습이 **매우 비싸고 시간 소요** → 소규모 연구그룹에서 재현 불가
- **사전학습 없이** 목적 데이터만으로 competitive VLA 학습 가능한지 검증

---

## 2. 방법론

### No Pre-training

기존 VLA: Large-scale VLM pre-train → robot data fine-tune
TinyVLA: **직접 robot data로 from-scratch 또는 VLM initialization만**

- VLM의 pre-trained weight를 initialization으로만 활용 (추가 pre-training 없음)
- 소형 모델로 fast inference

---

## 3. 실험 결과

| 모델 | Params | SR (%) | Data Efficiency |
|------|--------|--------|-----------------|
| OpenVLA | 7B | 76.5 | 970K demos |
| **TinyVLA** | **~1B** | **~75** | **<100K demos** |

- OpenVLA와 비슷한 성능을 **1/10 데이터, 1/7 파라미터**로 달성
- 일반화 (언어, 물체, 위치, 환경) 모두에서 robust

---

## 4. 한계 및 미해결 문제

1. **Ceiling**: Pre-training 없이는 더 복잡한 태스크에서 한계 도달 예상
2. **Data efficiency의 범위**: 100K이면 여전히 상당한 규모. 1K에서는?
3. **복잡한 reasoning**: Pre-training의 language/reasoning 지식이 없어 CoT 등 어려움

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "Pre-training 불필요" 주장의 검증 |
| **Practical impact** | ★★★★★ — 접근성 극대화 |

**강점**: Pre-training 없이 competitive 성능. 연구 접근성 향상. **약점**: Complex task에서의 ceiling.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Pre-training을 추가하면 얼마나 좋아지는가? | OpenVLA-OFT처럼 pre-training + 좋은 recipe = 97%+. Pre-training의 가치가 분명히 존재 |
| 2 | 1K demo에서는? | 성능 크게 하락 예상. Few-shot regime에서는 pre-training이 필수적 |

<!-- VERIFIED: abstract-only -->
