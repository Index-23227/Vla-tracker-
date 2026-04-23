# DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models

> **한 줄 요약**: 순차 제어 환경에서 **temporal error accumulation**으로 붕괴하는 VLA PTQ 문제를, **Cross-Space Representation Compensation + Motion-Driven Mixed-Precision** 두 축으로 해결하는 **drift-aware PTQ** 프레임워크.

---

## 1. 배경 및 동기

- VLA 배포는 memory/compute 제약 로봇에서 병목 → Post-Training Quantization(PTQ)이 자연스러운 선택지.
- 그러나 VLA에 PTQ를 naive하게 적용하면 **순차 제어 도중 성능이 급격히 무너짐**.
- 원인: vision-language → action 인터페이스에서의 양자화 오차가 시간에 따라 **kinematic drift** 로 누적.

---

## 2. 방법론

### 2.1 Drift-aware 최적화 formulation
- PTQ를 한 step의 오차 최소화가 아닌, **sequential decision process 위의 drift-aware optimization** 으로 재정의.
- 시간 축 오차 누적을 목적함수에 직접 포함.

### 2.2 Cross-Space Representation Compensation
- Multimodal(비전/언어) representation과 action space 사이의 **구조적 왜곡(structured distortion)** 을 보정.
- Action consistency를 개선하여 step-to-step error variance를 감소.

### 2.3 Motion-Driven Mixed-Precision Allocation
- Per-layer / per-component bit-width를 **trajectory-level motion error** 기준으로 할당.
- 모션에 민감한 부위에 더 많은 비트를, 덜 민감한 부위엔 적은 비트를 배정 → 효율과 정확도의 trade-off 최적화.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- 저비트 설정에서 **full-precision과 비교 가능한 성능** 달성을 주장.
- kinematic drift 유의미한 감소 보고.
- 구체 bit-width, 벤치마크, 에너지/메모리 수치는 PDF 확인 필요.

---

## 4. 한계 및 미해결 문제

1. **모델/백본 일반성**: 특정 VLA 아키텍처(discrete-token vs flow-matching 등)에 따라 drift 양상이 다름 → 일반화 검증 필요.
2. **Calibration 데이터 의존**: PTQ는 calibration set 분포에 민감. Trajectory-level objective는 calibration의 covering 문제를 더 민감하게 만들 수 있음.
3. **Hardware 매핑**: Mixed-precision이 실제 edge accelerator에서 지원되는 bit 조합과 정합해야 실질 속도 이득.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA의 sequential 특성을 PTQ 목적함수에 직접 반영한 관점 전환은 신선. Mixed-precision을 motion error로 구동하는 것도 깔끔. |
| **Practical impact** | ★★★★☆ — 저메모리 로봇 배포에 직격. 성능 유지 + 저비트가 사실이면 즉시 파급 큼. |

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 QAT (Quantization-Aware Training)가 아니고 PTQ인가? | QAT은 재학습 비용이 크고 대형 VLA에서는 현실적이지 않음. PTQ는 calibration만으로 배포 가능 → DA-PTQ는 그 한계를 drift-aware로 극복. |
| 2 | "Kinematic drift"를 어떻게 측정하나? | Trajectory-level motion error(예: end-effector 경로 deviation)를 objective로 사용. 구체 metric은 PDF 확인. |
| 3 | 다른 PTQ(예: GPTQ, AWQ) 대비 이득은? | 일반 LLM PTQ는 static per-token loss 기반이라 **순차 control drift에 맹목**. DA-PTQ는 trajectory 관점을 명시적으로 포함. |

<!-- VERIFIED: abstract-only -->
