# SyVLA: Scaling by Diversified Experience for Vision-Language-Action Models

> **한 줄 요약**: Qwen2.5-VL-3B + 0.69B Flow Matching Action Expert를 20개의 lightweight Feature Query Token으로 연결한 dual-system VLA에, (1) gradient L2-norm 기반의 annotation-free **Intention Decoupling**으로 CoT 추론과 control intention을 분리하고, (2) IL 데이터에서 유사 샘플을 retrieval해 PPO 업데이트를 안정화하는 **Similar-Sample Guided RL**을 결합하여, Pi0의 5% 미만 사전학습 데이터로 실세계 long-horizon dexterous task에서 In-Domain 0.73 / OoD 0.64, LIBERO 평균 81.2%를 달성한 ICML 2026 모델.

---

## 1. 배경 및 동기

### 기존 VLA의 두 가지 구조적 한계
- **High-level reasoning과 low-level control의 entanglement**: 액션 학습에 치중하면 VLM의 vision-language 추론이 catastrophic forgetting됨. ChatVLA 등 multi-modal 혼합 학습으로 완화하려 했지만 action competence와 VL capability 사이 균형 실패.
- **Policy optimization 불안정성**: IL의 학습 목표(action loss)와 실제 task success 사이 mismatch로 closed-loop에서 오차 누적 → OoD drift → 실패. RL은 본질적 해법이지만 billion-parameter VLA + 고차원 연속 액션 공간에서 policy drift / capability collapse가 빈발.

### 핵심 질문
- **명시적 reasoning(Think-Before-Act)을 유지하면서도 control intention에 추론 정보가 leak되지 않게 만들 수 있는가?**
- **실세계에서 안전하고 안정적인 VLA RL pipeline은 어떻게 구성할 수 있는가?**

📌 [Figure 1 삽입] — SyVLA가 "Fold the shirt" 명령에 대해 `<reason>I should fold it in half.</reason>` 형태의 CoT를 생성하고, Feature Query Token이 VLM과 Action Expert를 잇는 구조

---

## 2. 방법론 심층 분석

### 2.1 SyVLA 아키텍처

- **VLM**: Qwen2.5-VL-3B (Bai et al., 2025) — perception + language reasoning + control intention 생성
- **Action Expert**: Transformer 기반 Flow Matching 모델 (0.69B params) — 다단계 denoising으로 continuous action chunk 생성
- **Feature Query Token (n=20)**: 학습 가능한 fixed tensor 집합. VLM의 auto-regressive CoT 출력 뒤에 append 되어, 마지막 hidden state(Feature Query States)를 MLP adapter 통해 Flow Matching의 condition으로 전달
- **Pi0 KV cache 대비 장점**: lightweight → 낮은 inference latency, VLM ↔ Action Expert **비동기 추론** 지원

### 2.2 Intention Decoupling Algorithm (핵심 기여 1)

**문제**: CoT가 활성화되면 Feature Query Token의 last hidden state에 **고차원 reasoning 정보가 leak**되어 action precision이 떨어지고 hesitant behavior 발생.

**해법**: 학습 시 매 step마다
1. Action loss를 각 Feature Query Token의 last hidden state에 대해 미분
2. Gradient L2 norm이 가장 작은 token을 비율 τ만큼 선택
3. 해당 hidden state를 **zero로 mask**

→ Reasoning 표현과 남아있는 control representation 사이 **mutual information을 감소**시켜 control intention을 disentangle.

**Annotation-free**: 추가 label/태깅 없이 gradient만으로 자동 분리.

> ❓ **예상 질문**: 왜 gradient L2 norm이 작은 token을 masking하는가?
> **답변**: Action loss에 대한 영향이 작다는 것은 그 token이 control보다는 reasoning 정보를 담고 있을 가능성이 높다는 신호. 이를 제거하면 control representation이 더 sharp해진다. Table 4에서 τ=5–15% 구간이 stable하고 τ=0%면 0.43, τ=5%면 0.86으로 큰 차이.

### 2.3 Similar-Sample Guided RL Pipeline (핵심 기여 2)

**문제**: VLA scale에서 standard PPO + GAE는 sparse reward + long horizon에서 gradient explosion / policy collapse 빈발. ablation에서 `w/ Standard Advantage` 설정은 **0.00 success rate**.

**해법**:
1. IL dataset에서 현재 rollout과 **semantically similar한 expert sample**을 retrieval
2. Expert sample에는 **fixed advantage**를 부여, rollout sample에는 0-1 sparse reward로 PPO 업데이트
3. Similar sample이 policy update를 expert behavior 근방으로 묶어줌 → drift 억제

**효과**: Long-horizon sparse-reward task (Folding-Shirt)에서 IL-initialized policy 대비 **최대 +15%p absolute** 향상.

> ❓ **예상 질문**: Expert dataset만 추가해도 stabilize되는 것 아닌가?
> **답변**: Table 3에서 `w/o Expert Dataset` = 0.21로 매우 불안정. `w/o Similar Sample`(랜덤 expert sample) = 0.79로 stabilization은 되지만 0.86까지 못 감. Similar retrieval이 **추가 +7%p**를 만든다. 저자 해석: 유사 샘플 간 advantage 차이가 "2mm 허용 오차 vs 3mm 실패"의 fine-grained boundary를 모델에 가르친다는 것.

---

## 3. 데이터 전략

### Pretraining
- **대규모 robotic dataset** (Pi0의 5% 미만 규모)
- **<1%만 task-oriented CoT 주석**
- **~30% multi-modal data 혼합** → VLM의 일반 vision-language 능력 보존

### Task Fine-tuning
- 각 target task당 **수백 trajectory** 텔레오퍼레이션 수집

### RL Stage
- Cobot Magic 실로봇 플랫폼에서 on-environment rollout
- 0-1 sparse reward
- Value model: ~100M params (SigLIP encoder + transformer)

> ❓ **예상 질문**: Pi0의 5%만으로 어떻게 경쟁력 있는 성능이 나오는가?
> **답변**: Table 1에서 SyVLA는 Pi0(from scratch, 동일 데이터) 대비 In-Domain 0.73 vs 0.38, OoD 0.64 vs 0.26으로 압도. Pi0(pretrained, 10k+시간) 대비도 평균 0.73 vs 0.63으로 우위. 즉 **데이터 효율성**이 핵심 claim이며 Intention Decoupling + multi-modal mixing이 그 동력.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM Backbone | Qwen2.5-VL-3B |
| Action Expert | Transformer Flow Matching, 0.69B params |
| Feature Query Tokens | n=20 (학습 가능) |
| Mask threshold τ | 3–15% (stable range) |
| Multi-modal mix ratio | ~30% during pretraining |
| CoT annotation ratio | <1% of pretraining data |
| RL algorithm | PPO + Similar-Sample Guidance |
| Value model | ~100M (SigLIP + transformer) |
| Reward | 0-1 sparse |
| Hardware | 논문 명시 X |
| 실로봇 플랫폼 | Cobot Magic |

---

## 5. 실험 설계 및 평가 프로토콜

### 실세계 (Cobot Magic)
- **Task 1 — Folding Shirts** (14 trials): Polo 셔츠 접기. OoD = off-center 위치. Long-horizon dexterous.
- **Task 2 — Calculating and Wrapping** (28 trials): 임의 산술 명령 → 답 큐브 선택 → 타월로 wrap. OoD = unseen instruction.
- **Task 3 — Bagging Snacks** (14 trials): 모호한 user 명령 → snack 선택 → 천 가방에 담기. OoD = unseen vague instruction.

### Multi-modal 벤치마크
- DocVQA, AI2D, MMMU, MME, HallBench

### 시뮬레이션
- **LIBERO 4 suite** joint training (Pi0 protocol, Appendix C.1)

---

## 6. 실험 결과 심층 분석

### Real-world Tasks (Table 1)

| Method | InD T1 | T2 | T3 | **InD Avg** | OoD T1 | T2 | T3 | **OoD Avg** |
|--------|-------|----|----|----|--------|----|----|----|
| OpenVLA-oft | 0.71 | 0.29 | 0.36 | 0.45 | 0.55 | 0.11 | 0.07 | 0.24 |
| GR00T | 0.71 | 0.21 | 0.21 | 0.38 | 0.64 | 0.07 | 0.00 | 0.24 |
| Wall-Oss | 0.50 | 0.18 | 0.14 | 0.27 | 0.14 | 0.04 | 0.00 | 0.06 |
| Pi0 (pretrained, 10k+h) | 0.93 | 0.39 | 0.57 | 0.63 | 0.78 | 0.29 | 0.36 | 0.48 |
| Pi0 (from scratch) | 0.64 | 0.21 | 0.29 | 0.38 | 0.50 | 0.14 | 0.14 | 0.26 |
| ChatVLA | 0.21 | 0.29 | 0.14 | 0.21 | 0.00 | 0.21 | 0.07 | 0.09 |
| **SyVLA (ours)** | **0.86** | **0.68** | **0.64** | **0.73** | **0.78** | **0.57** | **0.57** | **0.64** |

- **OoD에서 가장 작은 성능 drop** (0.73 → 0.64, Δ=0.09). Pi0(pretrained)는 0.63 → 0.48, Δ=0.15.
- Task 2(arithmetic+manipulation), Task 3(vague instruction)에서 ChatVLA/GR00T 등 baseline을 큰 폭으로 추월 → **reasoning이 살아있다**는 증거.
- Task 1 InD에서 Pi0(pretrained) 0.93에 약간 못 미치지만, Pi0 사전학습 데이터의 5% 미만으로 달성.

### Multi-modal (Table 2)

| Method | DocVQA | AI2D | MMMU | MME | HallBench |
|--------|--------|------|------|-----|-----------|
| Wall-Oss | 63.62 | 58.60 | 37.11 | 1146.56 | 36.57 |
| ChatVLA | 83.30 | 67.36 | 37.40 | 1435 | 39.90 |
| **SyVLA** | 80.01 | **67.70** | 35.78 | **1795** | **42.53** |

- AI2D, MME, HallBench에서 SOTA. DocVQA/MMMU에서 ChatVLA에 약간 밀림.
- OpenVLA-oft, GR00T, Pi0는 **VQA 능력 자체가 없음** (catastrophic forgetting).

### LIBERO (Table 8, Appendix)

| Spatial | Object | Goal | Long | **Avg** |
|---------|--------|------|------|---------|
| 87.7 | 84.0 | 87.3 | 65.7 | **81.175** |

- LIBERO sim과 실세계 dynamics gap이 있어 deformable/fluid task에는 sim 평가 한계.
- Long suite 65.7로 다른 SOTA(95%+) 대비 낮음 — 본 논문은 LIBERO 자체를 주력 벤치마크로 삼지 않음.

---

## 7. Ablation 분석

### 핵심 컴포넌트 (Table 3, Folding-Shirt)

| Setting | Avg Success Rate |
|---------|-----------------|
| w/o CoT | 0.79 |
| w/o Intention Decoupling | **0.43** |
| w/o RL | 0.71 |
| w/o Expert Dataset | 0.21 |
| w/o Similar Sample | 0.79 |
| w/ Standard Advantage (no similar) | **0.00** |
| **SyVLA (all)** | **0.86** |

- **Intention Decoupling이 가장 critical** — 제거 시 0.43 (절반 가까이 하락).
- Standard PPO GAE 적용 시 0.00 → **gradient explosion**이 실제 일어남.
- Expert dataset 제거 시 0.21 → expert-grounding이 stabilizer 역할.
- Similar Sample retrieval은 random expert 대비 +7%p 추가 향상.

### Mask Threshold τ Sensitivity (Table 4)

| τ | 0% | 3% | 5% | 8% | 10% | 15% | 20% |
|---|----|----|----|----|----|----|----|
| Success | 0.43 | 0.73 | **0.86** | 0.77 | 0.86 | 0.82 | 0.64 |

- τ=0%(masking 없음)이면 reasoning leakage로 0.43.
- τ=3–15% 광범위 stable region.
- τ=20% 이상이면 too aggressive → control intention 손상.

### Feature Query Token 개수 N (Table 5)
- N이 작으면 information bottleneck → 성능 저하.
- N≥15에서 stable; N=20 이상은 marginal gain.

### Multi-modal Ablation (Table 9)

| Setting | DocVQA | AI2D | MMMU | MME | HallBench |
|---------|--------|------|------|-----|-----------|
| All (Ours) | 80.01 | 67.7 | 35.78 | 1795 | 42.53 |
| w/o Intention Decoupling | 63.62 | 71.76 | 38.67 | 1648 | 41.7 |
| w/o Mixed Data | 0.675 | 0.365 | 0.091 | 1255 | 22.25 |
| w/o Both | 0 | 0.0625 | 0.018 | 1238 | 18.71 |

- Multi-modal mix 제거 시 **catastrophic forgetting** (DocVQA 0%, MMMU 1.8%).
- Intention Decoupling 단독 제거는 multi-modal에 큰 영향 없음 → ID는 주로 action 측에 기여.

### RL 비교 (Table 11, Fold Shirt)

| Pi-RL | SimpleVLA-RL | **SyVLA** |
|-------|--------------|----------|
| 0.714 | 0.357 | **0.86** |

- SimpleVLA-RL은 KL constraint 제거로 collapse 빈발, flow matching 미지원.
- Pi-RL은 dual-MDP + PPO clipping 사용하나 실세계에서 table collision 빈발.

---

## 8. 관련 연구 비교

| 모델 | 아키텍처 | RL | CoT/Reasoning | Real-world dex | VL 보존 |
|------|---------|------|-------------|----------------|--------|
| OpenVLA-oft | Llama2-7B + parallel head | ✗ | ✗ | △ | ✗ (VQA 불가) |
| GR00T | VLM + Flow Matching | ✗ | ✗ | △ | ✗ |
| Pi0 (pretrained) | VLM + Flow Matching (KV cache) | RECAP (π0.6) | ✗ | ✓ (10k+h data) | ✗ |
| ChatVLA | MoE 분리 | ✗ | ✓ | ✗ | ✓ |
| Wall-Oss | — | ✗ | ✗ | ✗ | △ |
| **SyVLA** | **VLM + Flow Matching (FQT)** | **Similar-Sample Guided** | **✓ (decoupled)** | **✓** | **✓** |

### 핵심 차이
- **Pi0 KV cache → Feature Query Token (20개)** : lightweight + asynchronous inference.
- **ChatVLA MoE 분리 → Intention Decoupling**: 같은 backbone 안에서 gradient-based separation으로 추가 파라미터 없이 disentangle.
- **표준 PPO → Similar-Sample Guided PPO**: VLA scale에서 처음으로 안정적 실세계 RL 시연.

---

## 9. 한계 및 미해결 문제

### 저자가 인정한 한계
1. **Theoretical guarantee 부재**: Intention Decoupling의 효과는 empirical하며 rigorous한 수렴/disentanglement 증명 없음.
2. **LIBERO sim ↔ real gap**: 특히 deformable/fluid object에서 sim 평가 한계 명시.

### 추가로 보이는 약점
3. **Hardware 미보고**: GPU 종류/개수 없음 → 재현성 부족.
4. **Pretraining dataset 비공개 시점**: "open source 예정"이라 명시되어 있으나 현 시점 미공개.
5. **Cobot Magic 단일 플랫폼**: 다른 로봇(예: Franka, ALOHA)으로의 transfer 검증 없음.
6. **LIBERO Long 65.7로 약함**: 본 논문이 simulation을 주력으로 두지 않지만, 수치 자체는 SOTA 대비 낮음.
7. **τ를 task별로 튜닝해야 하는가?**: τ=5–15% 광범위 stable이라지만 task 종류별 best τ 분석 없음.

### Attribution 모호성
- 성능 향상이 (a) Intention Decoupling, (b) Similar-Sample RL, (c) multi-modal mixing, (d) Feature Query Token 자체 중 어디서 오는지 — ablation이 부분적으로 풀지만, **(d) FQT vs Pi0 KV cache의 직접 비교**는 부재.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Gradient-based intention masking은 단순하면서 효과적; Similar-Sample Guided RL은 VLA RL에서 신선한 형태. |
| **Technical depth** | ★★★★☆ — 3-stage pipeline, ablation 폭넓음. 이론적 보장은 약함. |
| **Experimental rigor** | ★★★★★ — 실세계 3 task × InD/OoD × 7 baseline 비교, multi-modal 5 bench, LIBERO sim, RL baseline 비교까지 포괄적. |
| **Practical impact** | ★★★★☆ — Pi0의 5% 데이터로 경쟁력 + 실세계 RL이 실제 작동. Open source 약속. |
| **Writing quality** | ★★★★☆ — 잘 구조화. 일부 hyperparameter 세부는 부록에 산재. |

**강점**: (1) 명시적 reasoning을 유지하면서 control degradation을 막는 **annotation-free** 알고리즘, (2) VLA scale에서 stable한 **real-world RL pipeline** 첫 데모, (3) Pi0의 5% 데이터로 OoD 성능에서 오히려 우위, (4) Multi-modal capability 보존.

**약점**: Theoretical guarantee, hardware 미보고, 단일 로봇 플랫폼, LIBERO long task에서의 상대적 약세.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Gradient L2 norm으로 reasoning vs control을 분리한다는 가정이 정당한가? | Empirical 검증만. Table 4의 τ=0 vs 5%(0.43→0.86), Table 9의 mixed-data 효과가 간접 증거. 이론적 lower bound 증명은 없음. |
| 2 | Intention Decoupling은 어떤 token이 masking되는지 학습마다 변하는가? | Yes, gradient는 batch마다 다름. 따라서 stochastic regularization 효과도 가질 수 있다. |
| 3 | Similar-Sample retrieval에 어떤 embedding 사용하나? | 본문은 "semantically similar"라고만 명시. Appendix에 더 자세히 있을 가능성. |
| 4 | Pi0(pretrained, 10k+h)을 InD Task1에서 못 이긴 점은? | 데이터 양의 절대 우위는 인정. 다만 (a) Task 2/3에서는 압도, (b) OoD에서 압도, (c) 5% 데이터 효율성이 본 논문의 claim. |
| 5 | Standard PPO가 0.00 success rate이라는 게 정말 PPO 알고리즘 자체의 문제인가? | VLA scale의 high-dim continuous action + sparse reward + 0.69B Action Expert 조합에서 GAE-based advantage 추정이 불안정. SimpleVLA-RL/Pi-RL 비교(Table 11)도 이를 뒷받침. |
| 6 | LIBERO Long 65.7로 다른 모델(95%+) 대비 낮은 이유는? | LIBERO joint training에 큰 hyperparameter tuning을 하지 않음. 본 논문 주력은 실세계. |
| 7 | Feature Query Token 20개가 Pi0 KV cache 대비 representational capacity가 부족하지 않나? | "lightweight하지만 hesitant behavior 발생" → Intention Decoupling으로 보완. 즉 capacity 부족이 오히려 decoupling을 강제하는 design choice. |
| 8 | Cobot Magic이 아닌 다른 로봇에서도 동작할까? | 검증 안 됨. 그러나 Action Expert가 standard flow matching이라 action space 호환만 되면 transfer 가능성 있음. |
| 9 | "<1% CoT annotation"이라는데 CoT 생성 능력이 이렇게 적은 데이터로 가능한가? | Qwen2.5-VL-3B의 사전 학습된 reasoning 능력을 보존 + few-shot 식으로 finetune. multi-modal mix가 catastrophic forgetting을 막아줌 (Table 9). |
| 10 | Real robot RL에서 안전 사고는 없었나? | Pi-RL은 "table collision" 발생. SyVLA는 expert dataset이 expert 행동 근방에 머물게 해서 명시적 안전 사고 보고 없음. |

<!-- VERIFIED: pdf -->
