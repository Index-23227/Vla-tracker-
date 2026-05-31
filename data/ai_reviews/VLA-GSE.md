# VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts

> **한 줄 요약**: Qwen3-VL-4B 기반 OpenVLA-OFT 스타일 VLA에서 LoRA 대신 **SVD 분해 기반 두 전문가 체계**(항상 활성인 Generalized Experts + 라우팅되는 Specialized Experts)와 expert-wise gradient scale balancing + backbone weight adjustment를 결합하여, 단 2.51% trainable parameter로 full fine-tuning을 LIBERO-Plus에서 능가하고 multimodal understanding(MMMU/OCRBench/DocVQA 등)을 LoRA 수준으로 보존.

---

## 1. 배경 및 동기

### VLA PEFT의 두 가지 dilemma

- **Full fine-tuning**: LIBERO 등 robot benchmark에서는 우수하나 backbone의 multimodal understanding(이미지 이해, OCR, 문서 QA)을 망가뜨림 → **catastrophic forgetting of pretrained capabilities**
- **LoRA**: Multimodal understanding은 잘 보존하나 robot benchmark 성능, 특히 OOD(LIBERO-Plus) 강건성이 부족
- **MoE-LoRA, AdaLoRA 등**: 여러 expert를 라우팅하면 expressivity는 늘지만 어느 expert가 generic skill을 담당해야 하는지가 임의적이라 학습이 불안정

### 핵심 질문

- **Robot task에 필요한 변화량을 "모두에게 공통인 부분(generalized)"과 "task-specific 부분(specialized)"으로 원리적으로 분리할 수 있는가?**
- 그 분리가 SVD라는 closed-form decomposition으로 가능하다면 expert 역할의 임의성이 사라질까?
- 그 결과 robot 성능과 multimodal 이해를 모두 보존할 수 있는가?

📌 [Figure 1 삽입] — GSE 구조: backbone weight ΔW를 SVD로 분해 → 큰 특이값에 해당하는 component는 Generalized Experts(항상 활성), 작은 특이값은 Specialized Experts(라우팅됨)

---

## 2. 방법론 심층 분석

### 2.1 SVD 기반 expert 분해

기존 LoRA가 $\Delta W = BA$ (low-rank 두 행렬)를 학습한다면, GSE는:

$$\Delta W = \underbrace{U_g \Sigma_g V_g^\top}_{\text{Generalized Experts, top-k singular values}} + \underbrace{\sum_e g_e(x) \cdot U_e \Sigma_e V_e^\top}_{\text{Specialized Experts, residual singular values, routed}}$$

- **Initialization**: Pretrained $\Delta W$의 SVD로 $U, \Sigma, V$를 초기화 → 각 expert의 초기 역할이 데이터로부터 자연 도출
- **Generalized Experts**: 큰 특이값 ⟶ 모든 입력에 대해 활성, "공통 robot skill"을 담당
- **Specialized Experts**: 작은 특이값 ⟶ router $g_e(x)$가 입력 의존적으로 선택, task별 특이성 담당

> ❓ **예상 질문**: SVD를 초기화로만 쓰고 학습 중에는 자유롭게 두는가, 아니면 직교성 등을 강제하는가?
> **답변**: 초기화로 사용 후 학습 중에는 free update. 그러나 expert-wise gradient scale balancing이 큰 특이값/작은 특이값에 받는 gradient magnitude를 균형 잡아 초기 역할 분리를 유지하도록 유도.

### 2.2 Expert-wise Gradient Scale Balancing

SVD 분해 직후, $\Sigma$의 분포가 매우 sharp하면 큰 특이값에 해당하는 generalized expert만 빠르게 학습되고 specialized expert는 학습 신호가 약해짐. 이를 막기 위해 각 expert에 inverse-magnitude scaling을 적용 → gradient norm이 expert별로 비슷하게 유지.

### 2.3 Backbone Weight Adjustment

PEFT 본연의 목적(pretrained backbone 보존)을 유지하면서도, backbone weight를 **아주 작게** 조정 가능하게 함. 이는 LoRA처럼 완전히 frozen하지 않고, GSE update와 함께 backbone도 미세 조정 — 하지만 학습률을 backbone에 대해 매우 작게 설정해 multimodal 능력을 보존.

> ❓ **예상 질문**: Backbone을 건드리면 LoRA의 장점(pretrained 보존)이 약화되지 않는가?
> **답변**: 정확히 그것이 trade-off의 핵심. GSE는 backbone에 가해지는 update magnitude를 명시적으로 제어해 "거의 frozen" 상태로 유지하면서도 robot task에 미세 적응할 여지를 남김. 결과적으로 MMMU·MMStar 등에서 LoRA와 동등.

### 2.4 Trainable Parameters

총 trainable parameter 비율: **2.51%** (4B backbone 기준 ~100M). LoRA-rank 16 정도와 유사하나 expert routing으로 효과적 capacity는 더 큼.

---

## 3. Base Architecture

| 구성요소 | 선택 |
|---------|------|
| VLM backbone | Qwen3-VL-4B-Instruct |
| Action head | MLP, parallel decoding (OpenVLA-OFT 방식) |
| Action space | continuous |
| Action chunk | 명시 안 됨 (OpenVLA-OFT 기본값 추정) |

OpenVLA-OFT의 핵심(parallel decoding으로 throughput 향상)을 그대로 가져가며 backbone만 Qwen3-VL로 교체.

---

## 4. 학습 설정

| 항목 | 값 |
|------|-----|
| Optimization steps | 80,000 |
| Batch size | 128 (16/GPU × 8 GPUs) |
| Training data | LIBERO 4-suite demonstrations |
| Trainable params | ~100M (2.51%) |

> ❓ **예상 질문**: 80k step × 128 batch = 10M sample-step. LIBERO 데이터셋이 그만큼 많지 않으므로 여러 epoch을 도는데 overfitting risk는?
> **답변**: LIBERO-Plus가 held-out OOD evaluation이라 overfitting이 직접 측정됨. 81.2%라는 결과 자체가 in-distribution overfit 여부를 시사하는 신호. 다만 standard LIBERO 4-suite의 in-distribution score는 직접 보고되지 않아 비교가 어려움.

---

## 5. 실험 결과

### 5.1 LIBERO-Plus Zero-shot (Table 1 핵심)

| Suite | VLA-GSE |
|-------|---------|
| Spatial | 90.3% |
| Object | 86.2% |
| Goal | 74.2% |
| Long | 74.1% |
| **Avg** | **81.2%** |

LIBERO-Plus는 LIBERO의 **OOD 변형**(camera, robot init, language, lighting, background, sensor noise 등)으로 측정됨. Standard LIBERO 점수는 본문에 명시 표가 없음 — 사실상 GSE는 **OOD robustness**를 중심 성능 지표로 채택.

### 5.2 Multimodal Understanding 보존 (Table 비교)

| Benchmark | Full FT | LoRA | **GSE** |
|-----------|--------|------|---------|
| MMMU | (큰 하락) | 거의 유지 | **거의 유지** |
| MMStar | (큰 하락) | 거의 유지 | **거의 유지** |
| OCRBench | (큰 하락) | 거의 유지 | **거의 유지** |
| MMB | (큰 하락) | 거의 유지 | **거의 유지** |
| DocVQA | (큰 하락) | 거의 유지 | **거의 유지** |
| InfoVQA | (큰 하락) | 거의 유지 | **거의 유지** |
| AI2D | (큰 하락) | 거의 유지 | **거의 유지** |
| RealWorldQA | (큰 하락) | 거의 유지 | **거의 유지** |

핵심 주장: **GSE는 LoRA-수준의 multimodal 보존 + Full FT를 능가하는 robot 성능**을 달성하는 첫 PEFT 기법.

### 5.3 PEFT 방법 간 비교 (Table 2)

GSE > Full FT > LoRA > AdaLoRA > MoE-LoRA (LIBERO-Plus 평균 기준).

### 5.4 Real-world 평가

4개 manipulation task에 distribution shift를 가하여 평균 **82.5%** success.

> ❓ **예상 질문**: Real-world 4개 task 평균 82.5% — 어떤 baseline 대비인가?
> **답변**: 논문에서 baseline 수치 직접 비교 표가 보이지 않음 (LoRA vs GSE의 real-world 비교 부재). 절대 점수만 보고됨 → claim 강도가 약화. Real-world에서 LoRA가 60%인지 80%인지에 따라 GSE의 우위가 달라짐.

### 5.5 Ablation (Table 4)

LIBERO-Plus Long suite에서:
- SVD 초기화 없음 → 큰 폭 하락
- Gradient scale balancing 없음 → 중간 하락
- Backbone weight adjustment 없음 → 작은 하락

세 가지 구성요소가 모두 기여하며 SVD 초기화가 가장 critical.

---

## 6. 한계 및 미해결 문제

### 방법론적

1. **Standard LIBERO 4-suite 점수 부재**: in-distribution 성능 표가 본문에 명시되지 않음. OOD 81.2%만 보고하면 "단순히 작은 데이터에 안 맞춘 결과인가"의 의심을 제거하지 못함.
2. **LIBERO-Plus Spatial 90.3% vs Goal 74.2%**: suite 간 편차가 큼. Goal/Long이 추론 의존적인데 GSE가 단순 task에만 강한 적응을 보일 가능성.
3. **MoE 라우터의 안정성**: Router $g_e(x)$의 학습이 불안정하면 specialized expert가 collapse될 위험. Router ablation이 없음.
4. **Backbone weight adjustment magnitude 미명시**: "small adjustment"의 정량적 정의가 없음 — replicability 약화.
5. **Action chunk size 미명시**: OpenVLA-OFT 기본값 추정이나 명시 부재.

### 비교 fairness

- Full FT가 multimodal를 잃는 것은 자명한 결과 (overfitting). 더 흥미로운 비교는 **same parameter budget의 LoRA vs GSE** — Table 2가 이를 다루지만 LoRA의 rank가 명시되지 않으면 capacity-matched 비교인지 불명.
- LIBERO-Plus에서 평균 81.2%인데 OA-WAM(83.9%)이 더 높음. 다만 OA-WAM은 7B Chameleon backbone + slot-aware pretraining(600k step Stage 0) — GSE의 2.51% trainable과는 비교 척도가 다름.

### Real-world

- 4개 task만, 비교 baseline 부족.
- "Distribution shift"의 구체적 정의(조명? 배경? 물체 위치?) 미상세.

---

## 7. 관련 연구 비교

| 방법 | Trainable % | LIBERO | MM 보존 | Robot 강건성 |
|-----|------------|--------|---------|-------------|
| Full FT | 100% | high | **broken** | medium |
| LoRA | ~1% | medium | preserved | low |
| AdaLoRA | ~1% | medium | preserved | low |
| MoE-LoRA | ~3% | medium-high | preserved | medium |
| **VLA-GSE** | **2.51%** | **high** | **preserved** | **high** |

### 차이의 본질

- LoRA는 low-rank로 capacity를 제한 → robot task에 부족
- MoE-LoRA는 expert 역할이 임의적 → 학습 불안정
- **GSE는 SVD로 expert 역할을 데이터-주도적으로 결정** → 안정성 + capacity의 양립

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — SVD 기반 expert 분해 자체는 신선, 다만 LLM PEFT에서 유사 아이디어(MoSLoRA, PiSSA) 존재 |
| **Technical depth** | ★★★☆☆ — 세 가지 구성요소가 모두 단순하나 결합 효과 명확 |
| **Experimental rigor** | ★★★☆☆ — LIBERO-Plus 강조는 좋으나 standard LIBERO 부재, real-world baseline 부재 |
| **Practical impact** | ★★★★☆ — 2.51% trainable + GitHub 공개는 실용성 높음 |
| **Writing quality** | ★★★★☆ — 동기와 ablation이 명확 |

**강점**: VLA PEFT에서 "multimodal 보존 vs robot 성능"이라는 본질적 trade-off를 정조준한 첫 시도. SVD 초기화라는 closed-form trick으로 expert 역할의 임의성을 제거. Code 공개로 reproducibility 확보. **약점**: Standard LIBERO 점수 부재, real-world baseline 부재는 claim의 정량적 결정성을 약화. 절대 점수만 보면 OA-WAM 등 비-PEFT 방법에 못 미침 — GSE의 가치는 절대 성능이 아니라 "trainable 비율 대비 성능".

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Standard LIBERO 4-suite 점수는 왜 안 보고하는가? | 본문에 명시 표 부재. LIBERO-Plus만 보고. In-distribution 성능이 LoRA보다 낮다면 OOD 81.2%의 의미가 약화될 위험. |
| 2 | LIBERO-Plus Goal 74.2%, Long 74.1% — 추론·장기 task 성능이 약한 것은 아닌가? | 그렇게 보임. Suite 편차 90.3 → 74.1로 16%p 격차. 단순 spatial reasoning은 강하지만 multi-step reasoning은 LoRA-level 한계 의심. |
| 3 | 2.51% trainable의 capacity가 SVD trick으로 부풀려지는 거 아닌가? | 부분적으로 맞음. SVD로 expert 역할이 분리되면 effective rank가 LoRA보다 높음. 그러나 절대 trainable parameter 수는 동일하므로 memory footprint는 같음. |
| 4 | Backbone weight adjustment는 결국 LoRA의 frozen-backbone 원칙을 깨는 것 아닌가? | 일부 그렇다. 다만 학습률을 매우 작게 두어 "거의 frozen"으로 유지. MMMU 보존이 이를 검증. |
| 5 | OA-WAM(LIBERO-Plus 83.9%)보다 낮은데 GSE의 가치는? | 비교 기준이 다름. OA-WAM은 7B + Stage 0 600k step. GSE는 4B + 2.51% trainable. 비용/성능 trade-off로 보면 GSE가 매력적. |
| 6 | Router 안정성 ablation 없는데 specialized expert가 collapse할 위험은? | 위험 존재. Router의 entropy regularization이 적용되었는지 명시 부재. SVD 초기화의 sharp distribution이 router 학습 초기에 saturating gradient를 만들 가능성. |
| 7 | Real-world 82.5%인데 baseline 점수 없으면 claim 강도가 어떻게 평가되나? | 약화. "absolute 82.5%가 인상적인가"는 task 난이도에 따라 다름. LoRA real-world 점수가 60%인지 80%인지로 결론이 갈림. |
| 8 | Action chunk size 미명시는 reproducibility 문제 아닌가? | 그렇다. OpenVLA-OFT 기본값(보통 8)으로 추정되나 명시 필요. Inference 속도 비교 시 chunk size가 critical. |
| 9 | SVD 초기화가 매번 새로운 task에서 의미가 있나, 아니면 LIBERO에 특화된 것인가? | 일반화 주장하나 실험은 LIBERO/LIBERO-Plus에 국한. CALVIN, SimplerEnv 등 다른 robot benchmark 검증 부재 — 핵심 누락. |
| 10 | Expert-wise gradient scaling이 hyperparameter sensitivity를 만들지 않는가? | 가능성 있음. Scaling factor 자체가 SVD singular value의 함수라 자동 결정되나, 그 함수 형태(역수? 로그?)가 임의적. Ablation 부재. |

<!-- VERIFIED: pdf -->
