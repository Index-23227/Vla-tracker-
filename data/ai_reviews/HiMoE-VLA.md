# HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist VLA Policies

> **한 줄 요약**: 로봇 데이터의 이질성(action space, embodiment, sensor)을 처리하기 위해 boundary 층의 Action-Space MoE(AS-MoE)와 인접 층의 Heterogeneity-Balancing MoE(HB-MoE)를 dense Transformer block과 interleave한 계층 MoE 구조. PaliGemma backbone에 flow-matching action expert를 결합하여 LIBERO 평균 97.8 / CALVIN ABC→D avg-len 3.967 달성.

---

## 1. 배경 및 동기 (Sec. 1)

VLM 데이터가 (텍스트·이미지로) 비교적 균질한 반면, 로봇 demonstration은 (i) **action space**(joint angle vs end-effector), (ii) **embodiment**(single/dual arm, 7D/14D), (iii) **sensor configuration & viewpoint**, (iv) **control frequency**, (v) tele-op 스타일까지 모든 축이 이질적이다. OpenVLA, RDT-1B, HPT 등 기존 OXE 기반 prior pre-train + fine-tune 모델은 이 이질성을 architecture 차원에서 다루지 않아 transfer가 비효율적이다. 저자들은 이질성의 **층별 추상화 위계**를 가정 — 얕은 층에서 action-space discrepancy를 specialize하고 깊은 층으로 갈수록 broader heterogeneity를 shared 표현으로 흡수.

## 2. 아키텍처 (Sec. 3.2, Figs. 1–2)

### 2.1 Vision-Language Module
- **PaliGemma**(SigLIP + Gemma-2B), π₀와 동일 백본
- 단일 마지막 layer 아닌 **모든 layer KV**를 action expert로 cross-attention 공급(stronger conditioning, Appendix B)
- KV cache로 rollout 가속

### 2.2 Hierarchical MoE Action Expert
입력: proprioception $q_t$ + noised action $A_t^\tau$ → 통일된 vector(action space는 고정 위치에 할당) → MLP → HiMoE
구조(Fig. 2):
```
[AS-MoE] ─ [HB-MoE] ─ [Dense Transformer × N] ─ [HB-MoE] ─ [AS-MoE]
   ↑boundary↑                  shared                    ↑boundary↑
```
- **AS-MoE**(boundary, top-K routing): action-space별 specialization. AS-Reg(Eq. 4–5)로 contrastive 학습 — 같은 action-space token에 routed된 expert pair는 positive, 다른 것은 negative
- **HB-MoE**(adjacent, top-K routing): embodiment·sensor 등 broader heterogeneity. HB-Reg(Eqs. 6–7) $\mathcal{L}_{HB} = \sum_i f_i P_i$ — load balance 보장
- **중앙 dense Transformer**: heterogeneous signal을 shared knowledge로 통합

### 2.3 Flow-Matching Loss (Eqs. 1–3)
$A_t^\tau = \tau A_t + (1-\tau)\epsilon$, $\tau \sim \text{Beta}$ (π₀ 관행). 최종 loss: $\mathcal{L} = \mathcal{L}_{flow} + \lambda_{AS}\mathcal{L}_{AS} + \lambda_{HB}\mathcal{L}_{HB}$

### 2.4 Pre-training 데이터
OXE + open-source ALOHA — diverse embodiment·action space·state 표현·task 포괄. Output: 6D EEF, 7D joint, 14D bimanual joint를 한 모델이 모두 처리.

## 3. 핵심 결과

### 3.1 CALVIN ABC→D (Tab. 1, sequence length 1–5의 average completed count)
| Method | 1 | 2 | 3 | 4 | 5 | Sum |
|---|---|---|---|---|---|---|
| MDT | 0.937 | 0.845 | 0.741 | 0.644 | 0.556 | 3.723 |
| π₀ | 0.914 | 0.830 | 0.739 | 0.676 | 0.599 | 3.758 |
| **HiMoE-VLA** | **0.932** | **0.855** | **0.789** | **0.731** | **0.660** | **3.967** |

### 3.2 LIBERO (Tab. 2, success rate %)
| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| **HiMoE-VLA** | **98.2** | **99.4** | **98.6** | **94.8** | **97.8** |

### 3.3 Real-world (Tabs. 3–4)
- **xArm7 single-arm**(Fruit-to-Plate / Cup-in-Cup / Block-on-Block, 6 sub-stage): HiMoE-VLA 75.0% vs π₀ 62.5% vs CogACT 61.5% — π₀ 대비 +12.5 pp
- **ALOHA dual-arm**(Cup-Handover / Scoop / Fold-Shorts): π₀, OpenVLA, RDT 등 baseline 대비 일관된 우위(논문 Tab. 4 page 7)

## 4. Ablation (Tab. 6 등)

- 단일 vanilla MoE (AS/HB 분리 없음, action space mixing): 통합이 어려움 — Tab. 6(b)에 정량화
- AS-Reg 제거: action-space specialization 약화 → 동일 action space 데이터 간에도 expert가 공유되어 cross-action transfer 손실
- HB-Reg 제거: load imbalance로 expert under-utilization

## 5. 비판적 분석

핵심 통찰은 "이질성은 균일한 단일 source가 아니라 **위계적**이다"는 것 — action space는 가장 'hard'한 partition(맞지 않으면 직접 충돌), embodiment·sensor는 더 부드러운 abstraction이다. AS-MoE를 boundary에 두는 것은 입력 transform/출력 detransform 모두에서 action-space 별 분기를 보장하고, 중앙 dense layer가 universal representation을 학습하도록 설계되었다. HPT의 dataset-specific stem/head 대비 **transfer 가능**하다는 것이 차별점.

## 6. 한계 및 VLA-Tracker 데이터 정합성

- **YAML 미스매치 플래그**: `data/models/himoe-vla.yaml`의 `benchmarks: {}`는 비어있으나 논문 Table 1·2에 LIBERO Spatial 98.2 / Object 99.4 / Goal 98.6 / Long 94.8 (Sum 97.8), CALVIN ABC→D length 3.967이 명시. 또한 YAML `action_head_category: regression`은 부정확 — 본 논문은 명시적으로 **flow matching**(Eq. 3)을 사용 → `flow_matching`으로 정정 권장.
- **YAML `action_head: "Hierarchical MoE action module with Action-Space MoE at boundary layers"`**는 정확하지만 flow-matching 설명 누락
- 논문은 추론 latency·HZ를 명시하지 않음 — deployment 측면 평가 부족
- Action-Space MoE가 사전 정의된 action space partition에 의존 — completely novel action space에는 새 boundary expert 추가 필요

<!-- VERIFIED: pdf -->
