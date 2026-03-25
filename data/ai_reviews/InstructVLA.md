# InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation

> **한 줄 요약**: Multimodal instruction tuning에 MoE adaptation을 적용하여 embodied reasoning과 action generation을 jointly 최적화하는 VLA-IT (VLA Instruction Tuning) 패러다임 제안.

---

## 1. 배경 및 동기

- 기존 VLA의 instruction following이 단순 template에 의존 → rich, diverse instruction 처리 부족
- VLM의 instruction tuning 성공을 로봇 도메인에 적용하려는 시도
- Understanding (scene 이해) → manipulation (행동)으로의 자연스러운 전이 필요

---

## 2. 방법론 심층 분석

### VLA Instruction Tuning (VLA-IT)

다양한 형태의 instruction을 단일 모델에서 처리:
- Scene description: "이 장면에 무엇이 있는가?"
- Action instruction: "빨간 컵을 왼쪽으로 옮겨라"
- Reasoning instruction: "어떤 순서로 정리해야 하는가?"

MoE adapter가 instruction 유형에 따라 적절한 expert를 routing.

> ❓ **예상 질문**: Understanding과 action의 joint training이 각각의 품질을 저하시키지 않는가?
> **답변**: MoE가 task-specific expert로 분리하여 interference를 최소화. 다만 shared layer에서의 간섭은 여전히 존재 가능.

---

## 3. 실험 결과 심층 분석

| 모델 | Understanding Score | Action SR (%) |
|------|-------------------|--------------|
| OpenVLA | 32.5 | 76.5 |
| LLaVA (understanding only) | 72.1 | N/A |
| **InstructVLA** | **65.3** | **83.7** |

- Understanding과 action 모두에서 competitive
- Understanding에서 순수 VLM (LLaVA) 대비는 약하나, action 통합 모델로서는 최고

---

## 4. 한계 및 미해결 문제

1. **Instruction diversity의 한계**: 미리 정의된 instruction template에 의존
2. **MoE routing의 최적성**: 어떤 expert가 어떤 instruction에 할당되는지 분석 부족
3. **Real-world 검증 제한적**
4. **Understanding 성능의 trade-off**: 순수 VLM 대비 하락

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA-IT 패러다임 |
| **Technical depth** | ★★★☆☆ |
| **Experimental rigor** | ★★★☆☆ |
| **Practical impact** | ★★★★☆ — Multi-capability VLA |
| **Writing quality** | ★★★☆☆ |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Understanding 성능을 VLM 수준으로 올릴 수 있는가? | VLM 크기 확대 + 더 많은 VL 데이터로 가능하나 action과의 trade-off |
| 2 | 사용자가 자유형 지시를 내리면 어떤 expert가 활성화되는가? | Router가 자동 결정하나, ambiguous 지시에서의 routing 정확도 미검증 |
