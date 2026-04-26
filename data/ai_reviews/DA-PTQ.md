# DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models

> **한 줄 요약**: 순차 제어에서 **temporal error accumulation**으로 붕괴하는 VLA PTQ를, **Cross-Space Representation Compensation (CSRC) + Drift-Aware Mixed-Precision Allocation (DA-MPA)** 두 축으로 해결. CogACT W4A8에서 42.5% 메모리 감소 + 54.8% 속도 향상을 달성하면서 FP 대비 성능 손실을 최소화.
>
> ⚠️ **용어 혼동 주의**: Abstract/Introduction에서는 "Motion-Driven Mixed-Precision Allocation"이라는 legacy 용어를 사용하지만, Sec 3.4 본문과 DA-**MPA** 약어는 **"Drift-Aware"** 를 일관되게 사용. 본 문서는 Sec 3.4의 canonical 용어를 따릅니다.

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

### 2.3 Drift-Aware Mixed-Precision Allocation (DA-MPA)
- Structural Jacobian으로 per-layer drift sensitivity 프로파일링 (Tikhonov damping λ=3e-4).
- Axis-dependent weights로 translation vs rotation 민감도 분리: **w_trans=1.8**, **w_rot=0.15** (translation 오차가 grasp 실패로 직결되어 높은 페널티).
- Top-k=30% drift-sensitive layers → **BF16**, 나머지 70% → **W4** (W4A8 전체).
- 최종 2개 DiT block은 continuous output에 근접하여 quantization 제외.

### 2.4 구현
- 구현: **PyTorch**, **NVIDIA RTX 5090**.
- 완전 post-training, 파인튜닝 없음.
- Calibration: **BridgeData V2 512 trajectories**, 6 spatial bins, 128 warmup steps.

---

## 3. 실험 결과

### Table 2 — WidowX SimplerEnv Visual Matching (기저 모델 CogACT, 대비 W4A8)

| Method | Put Spoon | Put Carrot | Stack | Put Eggplant | **Avg** |
|--------|----------:|-----------:|------:|-------------:|--------:|
| CogACT FP | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| VLA-Cache | 78.3 | 39.1 | 17.4 | 52.2 | 46.8 |
| QuantVLA | 47.8 | 39.1 | 17.4 | 69.6 | 43.5 |
| **DA-PTQ (Ours)** | 65.2 | **52.2** | **17.4** | 60.9 | **48.9** |

### Table 3 — Google Robot SimplerEnv

| Method | Pick Coke | Move Near | Open/Close Drawer | Drawer+Apple | **Avg (VM/VA)** |
|--------|----------:|----------:|------------------:|-------------:|----------------:|
| CogACT FP | 91.3 / 89.6 | 85.0 / 80.8 | 71.8 / 28.3 | 50.9 / 46.6 | **74.8 / 61.3** |
| **DA-PTQ** | 92.4 / 87.5 | 87.9 / 74.5 | 58.3 / 20.1 | 35.2 / 24.7 | **68.5 / 51.7** |

- Pick Coke / Move Near는 FP를 **능가**하기도 (양자화 noise의 regularization 효과)
- Drawer 계열 long-horizon 태스크에서 drop이 큼 → 저자도 인정한 한계

### Table 4 — Ablation on WidowX VM, 메모리/속도 포함

| 설정 | SR avg | **메모리 감소↑** | **속도↑** |
|------|-------:|-----------------:|----------:|
| + W4A8 + CSRC | 43.8 | 42.6% | 55.2% |
| + W4A8 + DA-MPA | 39.6 | 42.9% | 54.8% |
| **+ W4A8 + DA-PTQ (둘 다)** | **48.9** | **42.5%** | **54.8%** |

→ CSRC와 DA-MPA의 **시너지**: 단독보다 합쳤을 때 SR이 크게 뛰고 메모리/속도 이점은 유지.

### 벤치마크 종합

- 기저 모델: **CogACT** (diffusion VLA, 7-DoF 출력)
- 벤치마크: SimplerEnv (WidowX + Google Robot)
- 평가 대상: W4A8 (weights 4-bit, activations 8-bit)
- **42.5% 메모리 감소**, **54.8% 추론 속도 향상**
- **WidowX VM**: FP 51.3 → DA-PTQ 48.9 (-2.4pp)
- **Google Robot VM**: FP 74.8 → DA-PTQ 68.5 (-6.3pp; drawer 계열 민감)

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

<!-- VERIFIED: pdf -->
