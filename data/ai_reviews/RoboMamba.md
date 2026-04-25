# RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation

> **한 줄 요약**: Peking University/AI2 Robotics가 제안한 NeurIPS 2024 efficient VLA로, CLIP ViT-L + Mamba-2.7B (selective SSM, linear complexity)에 7MB(~3.7M, 모델 파라미터의 0.1%) 단순 MLP policy head만 fine-tune하여 SAPIEN 시뮬에서 ManipLLM 대비 seen +7.0%p / unseen +2.0%p, A100에서 9.0Hz inference(OpenVLA 3.4Hz, ManipLLM 1.1Hz 대비 3~7×)를 달성한다.

---

## 1. 배경 및 동기 (§1)

기존 MLLM 기반 VLA의 두 가지 한계: (1) 사전학습 MLLM의 robotic-scene 추론 능력 부족, (2) attention-based LLM의 비싼 fine-tuning/inference 비용. Mamba(Gu & Dao 2023)의 selective SSM은 input-dependent A,B matrix로 선택적 정보 보존 + linear time complexity를 제공 — "강한 reasoning + 효율을 동시에 가지면서, 적은 비용으로 manipulation skill을 획득할 수 있는가?"가 본 paper의 question.

---

## 2. 방법론 심층 분석 (§3)

### 2.1 아키텍처 (§3.2, Fig. 2)

- **Vision encoder**: CLIP ViT-Large (DINO/ConvNeXt ensemble은 의도적 배제, 단순성 우선)
- **Cross-modal connector**: MLP — visual feature R^(B×N×1024) → Mamba embedding R^(B×N×2560)
- **LLM**: Mamba 2.7B(또는 1.4B), discrete SSM (Eq. 2-4): A=exp(ΔA), B=(ΔA)^-1(exp(ΔA)-I)·ΔB, h_t = A·h_{t-1} + B·x_t
- **Policy head**: 두 MLP (position a_pos ∈ R^3, direction a_dir ∈ R^(3×3)), 합쳐서 ~3.7M (0.1% of model)

### 2.2 Two-Stage Training (§3.3-3.4)

- **Stage 1.1 Alignment**: LLaVA-LCS 558K, projection layer만 학습.
- **Stage 1.2 Instruction co-training**: LLaVA 655K mixed instruction + 300K RoboVQA. CLIP frozen, projection+Mamba fine-tune. Co-training이 generalization과 일반 reasoning 모두 향상시킴(Appendix C).
- **Stage 2 Manipulation fine-tuning** (Fig. 2): RoboMamba 전체 frozen, policy head만 학습. Loss(Eq. 5-6): L_pos = (1/N)Σ|a_pos − a_pos_gt|, L_dir = (1/N)Σ arccos((Tr(a_dir^T·a_dir_gt)−1)/2).
- 데이터: SAPIEN PartNet-Mobility로 10K train(20 task) + 1.1K test(20 seen + 10 unseen).

---

## 3. 실험 결과 심층 분석

### 3.1 Reasoning Benchmarks (Table 1)

| Benchmark | LLaVA1.5-7B | TinyLLaVA-2.7B | **RoboMamba-2.7B (224)** |
|---|---|---|---|
| OKVQA | — | — | **63.3** |
| VQAv2 | 78.5 | 77.7 | 79.6 |
| GQA | 62.0 | 61.0 | 64.2 |
| POPE | 85.9 | 86.3 | 86.3 |
| MMB | 64.3 | 68.3 | 60.9 |
| RoboVQA BLEU-4 | — | 29.6 | **42.8** |
| RoboVQA BLEU-1 | — | 43.5 | **62.7** |

LLaMA-AdapterV2(7B) 대비 추론 속도 7×.

### 3.2 Manipulation (Table 2, SAPIEN)

| Method | Seen Avg | Unseen Avg | Tunable params |
|---|---|---|---|
| UMPNet | 0.36 | 0.26 | — |
| FlowBot3D | 0.37 | 0.30 | — |
| RoboFlamingo | 0.43 (seen) | 0.43 | 1.8B (35.5%) |
| ManipLLM | 0.56 | 0.51 | 41.3M (0.5%) |
| **RoboMamba** | **0.63** | **0.53** | **3.7M (0.1%)** |

ManipLLM 대비 +7.0%p seen, +2.0%p unseen, 그러나 학습 파라미터는 11× 적고 추론은 ~7× 빠름.

### 3.3 추론 속도 (Fig. 1)

A100에서 RoboMamba 9.0Hz vs OpenVLA 3.4Hz vs ManipLLM 1.1Hz — 양자화/가속 없이 측정. Mamba의 linear complexity 기여.

### 3.4 Ablation (Fig. 3)

Mamba-2.7B가 RWKV-3B/Mamba-1.4B 대비 모든 reasoning bench에서 우수(a); Co-training 제거(w/o C) 시 unseen 급락 — RoboVQA가 generalization에 critical(b).

---

## 4. 한계

1. **벤치마크 부재**: LIBERO/CALVIN/SimplerEnv/RoboCasa 등 표준 sim VLA 벤치마크는 평가하지 않음 — SAPIEN PartNet-Mobility만 사용. **YAML의 `benchmarks: {}`는 실제로 정확** (paper-grounded).
2. **Open-loop evaluation**: §4.3에서 명시 — "open-loop task completion accuracy exclusively in the simulator". Closed-loop dynamics 평가 부재.
3. **2.7B LLM 한계** (§5 Limitations): 7B/13B MLLM 대비 복잡 reasoning에서 불리.
4. **6-DoF만 예측**: end-effector pose만, gripper status는 grasping에 추가만. Full trajectory 생성 아님.
5. **Real-world은 정성적**: Fig. 4-5의 demo는 시각적 결과뿐, 정량적 SR 미보고.
6. **Mamba의 1D scan 한계**: 2D spatial reasoning에 이론적 약점(저자들도 향후 4D extension 언급, §5).

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★★ — SSM의 VLA 최초 적용, 0.1% fine-tune 발견 |
| Technical depth | ★★★★☆ — Mamba ↔ MLP head 결합 명확 |
| Experimental rigor | ★★★☆☆ — SAPIEN/RoboVQA 한정, 표준 sim 부재 |
| Practical impact | ★★★★★ — 7× 빠른 inference, 10× 적은 fine-tune param |

**강점**: Reasoning ↔ manipulation 상관관계 실증, 압도적 efficiency. **약점**: 표준 sim 벤치 부재, real-world 정량 평가 부재, 2D Mamba scan 미적용.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | Mamba의 1D causal scan이 2D image에 부적합하지 않은가? | 맞음. CLIP feature가 이미 2D spatial 정보를 압축한 후 Mamba에 들어가므로 일부 완화. 4D extension(§5)이 향후 방향. |
| 2 | RoboFlamingo/ManipLLM도 동일 fine-tune 전략을 쓰면? | Fig. 3b에서 OpenFlamingo/LLaMA-AdapterV2에 동일 head fine-tune 적용 — 모두 RoboMamba 대비 낮음. Reasoning 사전학습이 핵심. |
| 3 | Linear complexity의 실제 이점은 어디서 나오는가? | 9.0Hz vs OpenVLA 3.4Hz는 sequence length가 짧을 때도 SSM의 효율적 hardware utilization 덕분. Long history에선 격차 더 커질 것. |

<!-- VERIFIED: pdf -->
