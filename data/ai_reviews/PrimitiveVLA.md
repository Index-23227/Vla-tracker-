# PrimitiveVLA: Learning Reusable Motion Primitives for Efficient and Generalizable Robotic Manipulation

> **한 줄 요약**: 텔레오퍼레이션 trajectory를 SAM+Cutie+Qwen3-VL+DeepSeek-V3 파이프라인으로 11개의 reusable motion primitive(Grasp/Place/Lift/Move/Push/Pull/Insert/Press/Twist/Tilt/Rotate)로 자동 분해한 뒤, 동일한 primitive 토큰으로 fine-tuning(disassemble)하고 inference 시 VLM planner + LLM-generated switching code로 재조립(assemble)하는 model-agnostic VLA 향상 프레임워크. OpenVLA-base에서 Libero-Long 30.50% → 80.25%, Libero-90-Novel 0-shot 6× 향상.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA(OpenVLA, pi-0.5)는 instruction → 전체 trajectory를 end-to-end로 학습 → **direct instruction-to-control mapping**
- 이 방식의 두 가지 핵심 문제:
  1. **Data inefficiency**: 비슷한 motion(grasp, place 등)이 모든 task에 등장하지만 trajectory level에서는 독립적으로 학습
  2. **Poor generalization**: 새로운 instruction이나 새로운 객체 조합에서 trajectory 전체를 다시 학습해야 함
- 인간은 task를 명시적으로 "이 동작 + 저 동작"으로 분해해 학습/일반화

### 핵심 질문
- **Trajectory를 자동으로 reusable primitive로 분해 가능한가?**
- **분해된 primitive로 fine-tune된 VLA가 0-shot 일반화에서 더 강한가?**
- **Inference 시 primitive 시퀀스를 어떻게 동적으로 선택/전환하는가?**

📌 [Figure 1 삽입] — Disassemble(SAM+Cutie+VLM+LLM) → MCR → Assemble(planner+switching code)

---

## 2. 방법론 심층 분석

### 2.1 11개 Motion Primitive 분류

| 카테고리 | Primitive |
|----------|-----------|
| Spatial Transport | Grasp, Place, Lift, Move |
| Contact & Interaction | Push, Pull, Insert, Press |
| Orientation | Twist, Tilt, Rotate |

→ 11개로 거의 모든 tabletop manipulation을 cover

### 2.2 Disassemble Pipeline

1. **SAM**: 매 frame에서 객체 segmentation
2. **Cutie**: 객체 mask를 trajectory 전체에 걸쳐 tracking
3. **Qwen2.5-VL-72B**: 객체-gripper 공간 관계 추론
4. **Qwen3-VL**: 어느 primitive에 해당하는지 분류 (primitive reasoning)
5. **DeepSeek-V3**: trajectory를 primitive 경계로 segmentation하는 Python code 자동 생성
6. 결과: trajectory가 [(Grasp_cup, frames 1-30), (Lift, 31-50), (Move, 51-100), (Place, 101-130)] 형태로 분해

> ❓ **예상 질문**: 자동 분해의 정확도는 어느 정도인가?
> **답변**: 논문은 분해 정확도를 명시적으로 보고하지 않음. 하지만 분해 결과로 fine-tune된 VLA가 baseline보다 6× 일반화하므로 간접 평가됨. False segmentation의 영향은 downstream task에서 noisy label로 작용.

### 2.3 Multimodal Canonical Representation (MCR)

- 분해된 (primitive, target_object, motion_params)를 **통일된 토큰 형식**으로 표현
- 모든 task에서 같은 primitive는 같은 token으로 표현 → reusability 보장
- VLA는 instruction → MCR sequence → action 으로 mapping

### 2.4 Inference: Assemble

- **VLM Planner (Qwen2.5-VL-72B)**: 새 task instruction을 보고 필요한 primitive 시퀀스 생성
- **LLM-generated Switching Code**: 한 primitive에서 다른 primitive로 전환하는 조건(예: "gripper closed AND object lifted")을 Python 코드로 생성
- 런타임에 base VLA가 primitive별 action을 생성, 조건 만족 시 다음 primitive로 전환

> ❓ **예상 질문**: 왜 switching을 학습 기반이 아닌 code로 하는가?
> **답변**: Code 기반 switching은 (a) 해석 가능, (b) 새 primitive 조합에 즉시 일반화, (c) 학습 데이터 불필요. 단점은 LLM이 생성한 code가 buggy할 위험.

### 2.5 Base VLA Independence

- OpenVLA (autoregressive), OpenVLA-OFT (parallel), pi-0.5 (flow matching) 모두에 적용
- Action head는 primitive별로 별도 학습하는 게 아니라, 같은 head가 MCR conditioning을 받아 다른 motion 생성
- 7-DoF action (6-DoF delta pose + gripper)

---

## 3. 데이터 전략

| 항목 | 값 |
|------|----|
| Source | LIBERO-90 demonstrations |
| Re-segmentation | SAM + Cutie + Qwen2.5-VL-72B + Qwen3-VL + DeepSeek-V3 자동 |
| Output | Primitive-labeled trajectories |
| Data efficiency | 50% data variant > 100% data OpenVLA |

> ❓ **예상 질문**: 자동 re-segmentation 비용은?
> **답변**: 대규모 LLM(Qwen 72B, DeepSeek-V3) 호출이 frame 단위로 필요 → 데이터 전처리 비용이 크나 1회성. Trained VLA의 inference 비용은 base VLA와 동일.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|----|
| Hardware | 명시되지 않음 |
| Optimizer | Base VLA inherit |
| Fine-tuning recipe | Base VLA inherit |
| Primitive 수 | 11 |
| Action dim | 7 (6-DoF + gripper) |

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO

| Setting | Baseline | **PrimitiveVLA** |
|---------|----------|-----------------|
| Libero-90 (OpenVLA base) | (baseline) | **+9.2%p** |
| Libero-90-Novel (0-shot) | ~7-8% | **45.50%** (6× baseline) |
| Libero-Long | 30.50% | **80.25%** |

- **Libero-Long의 30.50% → 80.25%** 약 2.6× 향상은 핵심 결과
- **Libero-90-Novel 0-shot 6×** 일반화는 primitive reusability의 강한 증거
- 50% data variant도 100% data OpenVLA baseline (80.30%) 이상 → data efficiency 입증

### 5.2 RLBench

| 지표 | **PrimitiveVLA** |
|------|-----------------|
| Avg improvement (10 tasks) | **+7.0%p** |

- 절대 baseline 수치는 본 review에서 미추출
- RLBench의 다양한 task에 primitive 일반화 입증

### 5.3 Real-World on UR5e

| Setting | Success |
|---------|---------|
| In-distribution | **90%** |
| Task generalization (OOD) | 57% |
| Compositional tasks | 65% |

- In-dist 90%는 강력
- OOD 57%는 절대 수치는 낮으나, baseline 대비 의미 있는 향상으로 추정 (baseline 미보고)

> ❓ **예상 질문**: Libero-Long 30.50% baseline은 너무 낮지 않은가?
> **답변**: OpenVLA의 Libero-Long은 보통 40-60% range. 30.50%는 특정 baseline (예: zero-shot 또는 weak fine-tune) 일 가능성. 정확한 baseline 정의 확인 필요.

> ❓ **예상 질문**: Libero-90-Novel 0-shot의 baseline 7-8%는 어떻게 가능한가?
> **답변**: Novel split은 학습에 보지 못한 새 task variation. OpenVLA는 fine-tune 분포 밖에서 성능이 급락. Primitive reusability가 이 gap을 6× 메움.

---

## 6. Ablation 분석

논문 ablation 핵심:
- **Disassemble 제거** (raw trajectory fine-tune): baseline 수준
- **MCR 제거** (primitive별 별도 token): cross-primitive transfer 손실
- **Switching code 제거 (학습 기반 switching)**: OOD generalization 큰 폭 하락
- **50% data**: 여전히 baseline 이상 — primitive efficiency 증명

→ Disassemble + MCR + Code-based switching 모두 essential

---

## 7. 관련 연구 비교

| Method | 분해 단위 | Switching | 0-shot 일반화 |
|--------|-----------|-----------|--------------|
| RT-2 | None (e2e) | None | 약 |
| OpenVLA | None (e2e) | None | 약 |
| CoT-VLA | Subtask | language | 중 |
| pi-0.5 | None (e2e) | None | 중 |
| **PrimitiveVLA** | **11 primitives** | **Code-based** | **강 (6×)** |

### 핵심 차이
- 명시적 motion primitive 분해 + reusability
- Code-based switching이 해석 가능 + 일반화
- Model-agnostic plug-in으로 3가지 base VLA에 모두 적용

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **자동 분해 정확도 미보고**: SAM/Cutie/Qwen/DeepSeek 파이프라인의 분해 정확도가 정량 평가되지 않음
2. **분해 비용 큼**: 72B VLM + DeepSeek-V3 호출이 frame당 → 데이터 전처리 비용이 base VLA fine-tune 비용을 초과할 수 있음
3. **Switching code의 robustness**: LLM이 생성한 Python code가 edge case에서 hang하거나 잘못 trigger할 위험
4. **11 primitive 분류의 외연**: tabletop manipulation에는 충분하나, articulated object, deformable, contact-rich 정밀 task는 cover 못할 가능성
5. **Hardware/training cost 미보고**: 재현성 평가 어려움
6. **Real-world OOD 57%, Compositional 65%**: 절대값이 낮음 — 일반화는 baseline 대비 향상이지 절대 수준은 미흡

### Attribution 문제
- Libero-Long 30.50% → 80.25%가 (a) primitive 분해, (b) MCR token, (c) switching code 중 어느 것의 기여인지 ablation 필요
- 6× 0-shot 향상이 (a) primitive reusability, (b) 단순한 fine-tune extra data (분해 과정에서 추가 supervision)인지 분리 불명확

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 명시적 motion primitive를 VLA fine-tuning에 도입한 정연한 프레임워크 |
| **Technical depth** | ★★★★☆ — 자동 분해 파이프라인 + MCR + code switching의 결합 |
| **Experimental rigor** | ★★★★☆ — LIBERO + RLBench + real-world 광범위 |
| **Practical impact** | ★★★★☆ — 3 VLA에 적용, data efficiency 50%, 0-shot 6× |
| **Writing quality** | ★★★★☆ — 명확한 framework 제시 |

**강점**: Direct mapping의 한계를 정확히 진단하고 명시적 primitive 분해로 해결. Libero-Long 80.25%와 Libero-90-Novel 0-shot 6× 향상은 매우 인상적. **약점**: 자동 분해의 정확도 정량평가 부재, 분해 비용이 큼, switching code의 robustness 우려, real-world OOD 절대값(57%)은 미흡.

---

## 10. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 11 primitive로 모든 task를 cover 가능한가? | Tabletop manipulation은 가능. Contact-rich(insertion 정밀도), deformable, articulated는 한계 |
| 2 | 자동 분해 정확도는? | 미보고. Downstream 성능으로 간접 평가. 실제 분해 quality는 case-by-case 검증 필요 |
| 3 | 분해 cost vs fine-tune cost? | 72B VLM + DeepSeek-V3 호출이 매 frame이라 매우 큼. 1회 비용이지만 base fine-tune보다 클 가능성 |
| 4 | Switching code가 buggy하면? | LLM이 생성한 Python의 syntax/logic 오류 가능. 논문은 robustness 측정 미보고 |
| 5 | Libero-Long 30.50% baseline은 정상? | OpenVLA의 보통 수치는 40-60%. 30.50%는 특정 weak baseline일 가능성, 정확한 정의 확인 필요 |
| 6 | pi-0.5 base에서도 80.25%인가? | 논문에 base별 결과 표 있음. pi-0.5는 더 강한 baseline이라 향상폭이 OpenVLA보다 작을 수 있음 |
| 7 | 0-shot 6× 향상이 단순 데이터 augmentation 효과 아닌가? | 분해 과정에서 supervised signal이 추가되긴 하나, MCR token이 primitive 간 transfer를 가능케 함 |
| 8 | CALVIN/RoboCasa로 확장 가능한가? | Long-horizon 시퀀스에는 적합. 그러나 CALVIN의 instruction 분포가 다르므로 추가 fine-tune 필요 |

<!-- VERIFIED: pdf -->
