# Ω-QVLA: Robust Quantization for Vision-Language-Action Models via Composite Rotation and Per-step Scaling

> **한 줄 요약**: π₀.₅와 GR00T N1.5를 LLM과 DiT action head **전체를 W4A4**로 양자화하면서도 LIBERO 성능을 FP16 대비 유지(π₀.₅ 97.1→98.0, GR00T 87.0→87.8)하는 PTQ 레시피. 핵심은 (1) 64-블록 단위의 composite SVD-Hadamard rotation으로 weight outlier 평탄화 + activation outlier 분산, (2) DiT의 8-step diffusion에 걸친 per-step·per-channel activation scaling. 정적 메모리 71.3% 절감, ARX R5 bimanual real에서 QuantVLA 25.0 대비 51.0의 진행률 점수 달성.

---

## 1. 배경 및 동기

### VLA on-device 배포의 양자화 병목
- π₀.₅, GR00T N1.5 같은 VLA는 5-10B 파라미터급 — edge GPU/embedded SoC에서 FP16으로는 메모리/지연 부담
- 일반 LLM 양자화(SmoothQuant, GPTQ, AWQ)는 W4A8까지만 안정적, **W4A4에서 큰 손실**
- VLA의 추가 도전: **DiT(diffusion action head)의 activation 통계가 denoising step마다 비정상(non-stationary)** → 단일 scale로 양자화 시 후반 step 활성 outlier가 망가짐

### 핵심 질문
- **LLM과 DiT 모두 W4A4**로 양자화하면서 task 성공률 손실 없이 가능한가?
- DiT의 step-별 activation drift를 어떻게 보정할 것인가?

📌 [Figure 1] — Composite SVD-Hadamard rotation + per-step activation scaling pipeline

---

## 2. 방법론 심층 분석

### 2.1 Composite SVD-Hadamard Rotation (CR)

기존 QuaRot/SpinQuant: 단일 random Hadamard rotation으로 outlier 분산
**Ω-QVLA의 CR**:
1. Weight matrix를 64-크기 블록으로 나눔
2. 각 블록 SVD: $W = U \Sigma V^\top$
3. $U$를 직교 conjugation으로 사용 + Hadamard rotation 합성
4. → **per-channel weight energy 평탄화** (SVD 효과) + **activation outlier 분산** (Hadamard 효과)

→ 4-bit weight quantization 후에도 동등 정보 보존

### 2.2 Per-step Per-channel Activation Scaling (PAS) — DiT 전용
- DiT는 보통 8 denoising step → step t에서의 activation 분포 $p_t(x)$가 t에 따라 변함
- 표준 PTQ: 모든 step에 동일한 scale s — 후반 step의 outlier를 cover 못함
- Ω-QVLA: 각 step t × 각 channel c별 개별 scale $s_{t,c}$ 학습
- 8 step × C channels의 scale을 작은 calibration set으로 추정

### 2.3 LLM은 GPTQ, DiT는 RTN
- LLM weight: GPTQ (second-order error correction) — 정확도 우선
- DiT weight: RTN (round-to-nearest) — 단순하지만 rotation+PAS와 결합 시 충분
- → 두 backbone의 특성에 맞는 hybrid quantization 전략

> ❓ **예상 질문**: 왜 DiT에 GPTQ를 안 쓰는가?
> **답변**: DiT의 8-step inference 동안 동일 weight가 반복 사용되므로 GPTQ의 input-aware calibration이 step-별 분포 다양성에 대응 못함. PAS가 step별 보정을 담당하므로 weight는 RTN으로 충분.

> ❓ **예상 질문**: 71.3% 메모리 절감의 분해는?
> **답변**: FP16 → INT4 weight는 4× 압축. Activation은 inference 시점에만 4-bit이므로 정적 메모리에는 영향 없음. 71.3%는 주로 weight 압축 + KV cache + 모델 metadata의 종합.

---

## 3. 데이터 전략
- **No fine-tuning** — 순수 PTQ
- Calibration: LIBERO 일부 demo (소량)
- 평가 데이터셋과 calibration이 겹치지 않게 split

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (4 suites)

**π₀.₅:**
| Suite | FP16 | Ω-QVLA W4A4 |
|-------|------|-------------|
| Spatial | 99.0 | 99.0 |
| Object | 97.5 | 97.0 |
| Goal | 98.5 | 100.0 |
| Long | 93.5 | 96.0 |
| **Avg** | **97.1** | **98.0** |

**GR00T N1.5:**
| Suite | FP16 | Ω-QVLA W4A4 |
|-------|------|-------------|
| Spatial | 92.0 | 86.0 |
| Object | 92.0 | 92.0 |
| Goal | 86.0 | 91.0 |
| Long | 76.0 | 82.0 |
| **Avg** | **87.0** | **87.8** |

- **양자화가 평균을 *유지하거나 약간 향상***시키는 흥미로운 결과
- 향상의 이유: quantization noise가 약한 regularizer로 작동했을 가능성 (특히 LIBERO-Long 93.5→96.0)
- 단 GR00T Spatial은 92.0 → 86.0으로 6%p 하락 — uniform W4A4가 특정 layer에서 손실 보임

### 4.2 Real-world ARX R5 Bimanual (5 tasks)
| 방법 | 평균 진행률 |
|------|------------|
| QuantVLA (기존 SOTA W4A4) | 25.0 |
| **Ω-QVLA W4A4** | **51.0** |

- **104% 상대 향상** — 같은 W4A4에서 진행률이 2배
- 5 tasks: Pick Cup, Put Blocks, Put Fruit, Put Flowers, Fold Towel — bimanual complexity 높음

### 4.3 메모리 절감
- 정적 메모리 **71.3% 감소** — 8GB 모델이 ~2.3GB로
- → Jetson AGX Orin, RTX 4080 등 mid-range edge에 배포 가능

---

## 5. Ablation 분석 (제공된 한도 내)
- Composite Rotation only vs full method 비교가 있을 것으로 추정 — 명시 부족
- PAS의 step 수(8) sensitivity 미보고

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **Task scope이 LIBERO + 5 real tasks로 제한**: SimplerEnv, RoboTwin, CALVIN 같은 다양한 benchmark에서의 검증 부재
2. **GR00T Spatial -6%p**: Uniform W4A4의 일부 layer 손실을 노출 — 혼합 정밀도(MP) 비교 부재
3. **Inference latency 미보고**: 4-bit 연산 가속 GPU(예: H100 INT4) 외 edge에서 실제 속도 측정 없음
4. **Calibration 크기 sensitivity 미보고**

### Attribution 문제
- LIBERO 향상이 *quantization noise regularization*인지 *방법의 정확성*인지 분리 어려움
- Composite rotation vs SVD only vs Hadamard only ablation이 보고 한도에서 부족

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — DiT-specific PAS는 새롭고 의미 있음 |
| **Technical depth** | ★★★★☆ — SVD+Hadamard composite + per-step scaling이 잘 설계됨 |
| **Experimental rigor** | ★★★☆☆ — LIBERO + 5 real로 한정 |
| **Practical impact** | ★★★★★ — On-device VLA 배포 가능성, code 공개 |
| **Writing quality** | ★★★★☆ |

**강점**: W4A4에서 LIBERO 손실 없이 71.3% 메모리 절감 + real-world 2× 성능 — VLA on-device 시대를 여는 의미. Code open-source. **약점**: 평가 범위가 좁고, 일부 카테고리(GR00T Spatial)에서 손실, latency 미보고. **카탈로그상으로는 architecture가 아니라 quantization 기법**이라는 점도 데이터셋 분류 시 유의해야 함.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 새 VLA 모델인가, quantization 레시피인가? | 후자. Pi 0.5와 GR00T N1.5를 양자화. |
| 2 | LIBERO 향상의 원인은? | Quantization noise regularization 가능성. 통제 실험으로 명확히 분리 필요 |
| 3 | GR00T Spatial -6%p의 원인은? | Uniform W4A4가 특정 layer(아마 attention)에서 손실. 혼합 정밀도가 답일 수 있음 |
| 4 | Real-world 추론 latency는? | 미보고 — 실제 edge 가속 효과 측정 부재 |
| 5 | SimplerEnv, RoboTwin에서는? | 평가 부재 — generalization 검증 약함 |
| 6 | Calibration data size sensitivity? | 미보고 |
| 7 | PAS의 8-step 의존성은? | 8-step diffusion에 fit. step 수 다른 모델로 일반화는 검증 부재 |
| 8 | Composite Rotation의 각 컴포넌트 기여? | 명시적 ablation 한도에서 부족 |
| 9 | INT4 kernel 없는 GPU에서 실제 이득은? | 메모리는 이득, latency는 dequant 오버헤드로 손실 가능 |

<!-- VERIFIED: pdf -->
