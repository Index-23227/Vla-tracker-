# ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models

> **한 줄 요약**: HSIC 기반 inter-tensor bit allocation과 Action-Mixed Fisher(AMF) intra-tensor scale optimization을 결합한 action-aware PTQ로, OpenVLA-OFT를 3.0 bpw로 압축해도 LIBERO 평균 95.0% (FP16 96.9% 대비 -1.9%p)를 유지하며 5.3x 크기 감소와 실제 UR3 로봇 배포까지 검증.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 모델은 LLM(7B+)을 backbone으로 사용하여 **메모리/연산 부담이 큼** → 실로봇 edge 배포에 부적합
- 기존 LLM quantization 기법(GPTQ, AWQ 등)은 **next-token prediction loss**를 기준으로 중요도를 평가 → action 생성 품질과의 mismatch
- VLA에서는 action token 한 비트의 오류가 manipulation 실패로 직결됨 → "perplexity-optimal" 양자화가 "action-optimal"이 아님

### 핵심 질문
- **Action 품질을 직접 보존하는 quantization criterion을 어떻게 설계할까?**
- **Sub-4-bit 영역(2.5~3.0 bpw)에서도 VLA가 실용적으로 작동하는가?**

📌 [Figure 1 삽입] — ActQuant pipeline: HSIC inter-tensor allocation → AMF intra-tensor scale → OmniModel.cpp 배포

---

## 2. 방법론 심층 분석

### 2.1 전체 구조: 2-stage PTQ

| 단계 | 목표 | 핵심 도구 |
|------|------|---------|
| Inter-Tensor Bit Allocation | 어느 weight matrix에 더 많은 bit를 줄 것인가 | HSIC (Hilbert-Schmidt Independence Criterion) |
| Intra-Tensor Scale Optimization | 각 matrix 내부에서 어떤 block에 fine scale을 줄 것인가 | Action-Mixed Fisher (AMF) |

### 2.2 HSIC 기반 Inter-Tensor Allocation

- 각 layer의 weight matrix W_l에 대해, **action prediction output과의 dependency를 HSIC로 측정**
- HSIC가 높은 layer = action에 결정적 → 더 많은 bit 할당
- 전체 budget B 하에서 정수 계획(integer program)으로 bit-width 분배

> ❓ **예상 질문**: 왜 HSIC인가? Fisher information이나 SNR도 가능하지 않나?
> **답변**: HSIC는 kernel 기반 nonlinear dependency를 잡으며, action output처럼 LLM logits + de-tokenization이 결합된 비선형 함수에 적합. Fisher는 second-order 정보지만 local linearization 가정이 강함.

### 2.3 Action-Mixed Fisher (AMF) Scale Optimization

- 각 matrix 내부 block scale을 결정할 때, **action head loss + LM loss의 mixed Fisher**를 사용
- 단순 LM loss Fisher (예: GPTQ의 Hessian)보다 action-critical parameter에 가중치 부여
- AMF = α · F_action + (1-α) · F_LM, α는 layer별 자동 조정

### 2.4 OmniModel.cpp: Agentic C/C++ 변환 파이프라인

- HuggingFace PyTorch 모델 → GGML 커널로 자동 변환하는 **agentic LLM 파이프라인**
- VLA의 multi-modal input(이미지+텍스트+상태) 처리를 위한 dispatch 코드를 LLM이 생성
- Edge 하드웨어(UR3 동봉 PC 등)에서 sub-4-bit GGML inference 가능

> ❓ **예상 질문**: OmniModel.cpp가 본 논문의 핵심인가, 부수적인가?
> **답변**: 부수적이지만 실용적 기여. Quantization 알고리즘만으로는 edge 배포가 어려운데, 변환 파이프라인을 함께 제공함으로써 "실로봇 검증"이 가능해짐.

---

## 3. 데이터 전략

### Calibration set
- **LIBERO fine-tuning split에서 60 episodes만 사용** — 매우 적은 양
- PTQ이므로 fine-tuning 없이 calibration data로 statistics 추정
- 추가 학습 없음

> ❓ **예상 질문**: 60 episodes로 7B 모델의 양자화 statistic을 안정적으로 추정 가능한가?
> **답변**: PTQ의 표준 calibration data 규모(GPTQ 128 samples 등)와 유사. Action token vocab이 작아 oversampling 부담이 적음. 다만 분포 외 task에서는 추가 calibration이 필요할 수 있음.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 단일 NVIDIA L40S GPU |
| Wall-clock | 모델당 ~2시간 (양자화) |
| Fine-tuning | 없음 (PTQ) |
| Compression | OpenVLA-OFT 기준 5.3x at 2.5 bpw, 4.5x at 3.0 bpw |

---

## 5. 실험 설계 및 평가 프로토콜

- **두 가지 baseline VLA**: OpenVLA-OFT (discrete action token) + pi-0.5 (flow-matching continuous)
- **LIBERO 4 suite**: spatial / object / goal / long
- **실로봇 검증**: UR3 manipulator, pi-0.5 3.0 bpw, 75% 평균 성공률

---

## 6. 실험 결과 심층 분석

### LIBERO — OpenVLA-OFT (Table 1)

| Bit-width | Spatial | Object | Goal | Long | Avg | 압축률 |
|-----------|---------|--------|------|------|-----|--------|
| FP16 baseline | 97.6 | 98.4 | 96.8 | 95.1 | **96.9** | 1.0x |
| ActQuant 3.0 bpw | 92.8 | 97.4 | 94.0 | **95.6** | **95.0** | 4.5x |
| ActQuant 2.5 bpw | 86.4 | 98.2 | 84.8 | 91.0 | **90.1** | 5.3x |

- 3.0 bpw에서 **평균 -1.9%p 손실**로 4.5x 압축 — 매우 인상적
- 2.5 bpw에서 **goal task가 84.8%로 급락** — task별 robustness 격차 큼
- Long task에서 오히려 baseline보다 향상되는 경우(95.6 vs 95.1)는 regularization effect 또는 측정 분산

### LIBERO — pi-0.5 (Table에 일부)

| 설정 | Avg |
|------|-----|
| FP16 baseline | 97.0 |
| ActQuant 3.0 bpw | **94.8** |

- Flow-matching head 보유 모델에도 적용 가능 — 방법론의 일반성 입증

### Real-world UR3

| Bit-width | 평균 성공률 |
|-----------|------------|
| pi-0.5 3.0 bpw | **75.0%** |

- 시뮬레이션 95% → 실로봇 75% 격차는 sim-to-real gap 자체로, 양자화 기여 아님
- 2.5x 메모리 감소 확인

---

## 7. Ablation 분석

### Inter-tensor만 vs. AMF만 vs. 결합

논문에서 두 단계의 기여도를 분리하는 ablation을 제공:
- HSIC inter-tensor allocation만: LIBERO ~93%
- AMF intra-tensor만: LIBERO ~92%
- 둘 다 결합 (Full ActQuant): **95.0%**

> ❓ **예상 질문**: 두 stage의 결합 효과가 단순 sum-of-gains를 초과하는가?
> **답변**: 약하게 그렇다. Bit allocation이 옳은 layer에 더 많은 bit를 줘야 AMF의 scale optimization이 의미를 가짐. 두 stage 모두 action-aware라는 점에서 일관된 inductive bias.

---

## 8. 관련 연구 비교

| 방법 | Criterion | Bit | LIBERO Avg | Hardware-deployable |
|------|-----------|-----|-----------|-------------------|
| GPTQ | LM Hessian | 4 | ~93 | LLM 위주 |
| AWQ | Activation magnitude | 4 | ~92 | LLM 위주 |
| SmoothQuant | Activation smoothing | 8 | ~95 | x |
| **ActQuant** | **Action-aware HSIC + AMF** | **3.0** | **95.0** | **UR3 검증** |

### 핵심 차이
- **Action-aware criterion 최초 적용**
- Sub-4-bit 영역에서 실로봇 검증까지 수행한 첫 사례

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **2.5 bpw에서 task별 격차**: spatial 86.4% vs object 98.2% — task에 따라 quantization sensitivity가 크게 다름. Task-adaptive bit allocation은 미연구
2. **2개 모델만 검증**: OpenVLA-OFT, pi-0.5만 평가. RT-2, OpenVLA(원본), Octo 등 다른 backbone에서의 일반화 미입증
3. **Calibration set이 LIBERO와 일치**: in-distribution calibration. Real-world distribution shift 하에서의 robustness 미평가
4. **OmniModel.cpp의 자동화 수준**: 논문에서 "agentic" 변환이라고 표현하나, 실제 LLM이 얼마나 자동화하는지/실패 빈도 미보고

### Attribution 문제
- 3.0 bpw에서의 -1.9%p 손실이 **HSIC criterion 때문인지, AMF 때문인지, 그냥 PTQ가 충분히 robust한 task이기 때문인지** 분리가 불완전
- LIBERO 자체가 비교적 짧고 quantization에 관대한 task일 가능성. 더 어려운 long-horizon task(SimplerEnv, CALVIN)에서의 검증 부재

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Action-aware quantization은 명확한 새 angle |
| **Technical depth** | ★★★★☆ — HSIC + AMF 결합, 2-stage 구조 체계적 |
| **Experimental rigor** | ★★★☆☆ — LIBERO만, 2개 모델만, 더 도전적 task 부재 |
| **Practical impact** | ★★★★★ — 실로봇 검증, 5.3x 압축, 1 GPU 2시간 |
| **Writing quality** | ★★★★☆ — 명확한 motivation, 깔끔한 ablation |

**강점**: VLA 양자화의 첫 action-aware PTQ. 실로봇 UR3 검증과 OmniModel.cpp 변환 파이프라인으로 **end-to-end edge 배포** 가능성을 보여줌. **약점**: LIBERO 외 평가 부재, 2.5 bpw에서의 task-level 변동성, calibration-distribution match에 대한 잠재적 의존성.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | HSIC 대신 단순 gradient-based importance를 쓰면? | Ablation 부재. HSIC가 kernel-based nonlinear 의존성을 잡지만, simple grad-norm이 충분할 가능성도 있음 |
| 2 | 2.5 bpw goal 84.8%는 왜 그리 떨어지나? | Goal-conditioning이 LLM의 instruction-following에 더 의존 → low-bit에서 representation collapse. AMF의 α를 task-conditional하게 해야 할 가능성 |
| 3 | OmniModel.cpp의 실제 자동화 정도는? | "agentic"이라는 표현이 모호. 명세서/template-driven인지 fully LLM-generated인지 불분명 |
| 4 | Real UR3 75%는 vanilla pi-0.5와 얼마나 차이? | FP16 pi-0.5의 UR3 성능 baseline이 보고되지 않음 → 양자화의 real-world 손실분 직접 측정 불가 |
| 5 | Action vocabulary가 작아서 양자화에 관대한가? | OpenVLA-OFT는 256-bin discrete action token. Token logit space가 크지 않아 quantization noise가 argmax를 잘 안 흔듦. Continuous head(pi-0.5)에서는 어떨지가 더 흥미로움 |
| 6 | 1.5 bpw, 1 bpw 영역으로 더 갈 수 있나? | 논문은 2.5 bpw가 lower bound. Binary/ternary는 미실험 |
| 7 | Calibration distribution이 다르면? | Cross-task calibration 실험 부재. Out-of-distribution에서 success rate 급락 가능성 |
| 8 | 학습 가능한 quantization-aware fine-tuning(QAT)과 비교하면? | QAT 비교 부재. PTQ의 simplicity 장점이 본 논문 주장이지만, 더 낮은 bit에서는 QAT가 필수일 수 있음 |

<!-- VERIFIED: pdf -->
