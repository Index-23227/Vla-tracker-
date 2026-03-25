# Diffusion-VLA (DiVLA): Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning

> **한 줄 요약**: Qwen2-VL (2B/7B/72B)에 FiLM 기반 reasoning injection으로 diffusion action policy를 가이드하여, 39K trajectory 사전학습만으로 970K 기반 OpenVLA를 압도 (multi-task 83.6% vs 39.4%), 102개 미지 물체 zero-shot 63.7%, **82Hz (2B)** 추론 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA의 reasoning(ECoT 등)은 **사후 추가된 보조 과제** → action에의 직접적 기여 불명확
- Reasoning을 매 step의 입력-출력 cycle로 수행하면 **computational overhead** 큼
- 사람이 만든 CoT annotation은 비용이 높고 scalability 한계

### 핵심 질문
- **Reasoning을 FiLM으로 policy에 직접 embedding하면, iterative reasoning 없이도 action이 향상되는가?**
- **소량 데이터 (39K)로도 대량 (970K) 기반 모델을 능가할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Architecture

**VLM**: Qwen2-VL (2B / 7B / 72B) + SigLIP visual encoder
**Action module**: Diffusion policy, VLM→projection (2 MLP + LayerNorm)→diffusion model
**Reasoning injection**: **FiLM (Feature-wise Linear Modulation)** — VLM의 reasoning tokens이 projection layer를 scale/shift

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \alpha \mathcal{L}_{\text{ntp}}, \quad \alpha = 10$$

> ❓ **예상 질문**: α=10이면 NTP loss가 diffusion보다 10배 가중되는데, action 품질에 해가 없는가?
> **답변**: NTP loss가 reasoning quality를 보장하고, 이 reasoning이 FiLM으로 action에 간접 기여. α가 높으면 reasoning이 더 정확 → action도 개선. Sensitivity 분석은 미제공.

### 2.2 Self-Generated Reasoning

- **GPT-4o**로 합성 reasoning annotation 생성 (학습 데이터)
- 모델이 reasoning을 autoregressive하게 생성 → FiLM으로 action conditioning
- 예: 대상 물체 변경 시 "grabbing toy blue car" → "grabbing hex key"로 **동적 자기 수정**

### 2.3 Inference Speed

| Model | Control Frequency |
|-------|------------------|
| **DiVLA-2B** | **82 Hz** |
| **DiVLA-7B** | **42 Hz** |
| OpenVLA-7B | 5 Hz |

- 82Hz는 **action chunk**으로 달성: chunk당 VLM 1회 + diffusion 1회 → 여러 step 동시 출력
- VLM reasoning은 매 chunk마다가 아닌 주기적으로만 실행될 가능성 있음

---

## 3. 데이터 전략

| 모델 | Pre-training Data |
|------|------------------|
| DiVLA-2B/7B | **DROID dataset** |
| DiVLA-72B | DROID + OXE |
| Fine-tuning | 400-580 trajectories/task |

- **39K trajectories**만으로 pre-train (vs OpenVLA 970K) — 24x 적은 데이터

### Fine-tuning

| 항목 | 값 |
|------|-----|
| LR | 2e-5 (fixed) |
| Epochs | 20 |
| VLM | LoRA applied |
| Visual encoder | Frozen |
| GPU | A6000 |

---

## 4. 실험 결과 심층 분석

### Multi-Task (Table 1, 5 tasks)

| Model | Pre-train Data | In-Dist Avg | **Visual Gen Avg** |
|-------|---------------|-------------|-------------------|
| Diffusion Policy | - | 27.9% | 8.9% |
| TinyVLA | - | 45.5% | 28.9% |
| Octo | 970K | 24.3% | 17.8% |
| OpenVLA-7B | 970K | 39.4% | 26.7% |
| **DiVLA-2B** | **39K** | **83.6%** | **57.8%** |

- **39K로 970K 기반 OpenVLA를 2배 이상 능가** (83.6% vs 39.4%)

### Zero-Shot Bin Picking (102 Unseen Objects)

| Model | SR (%) |
|-------|--------|
| Diffusion Policy | 8.9 |
| Octo | 19.6 |
| TinyVLA | 23.5 |
| OpenVLA | 28.4 |
| **DiVLA-2B** | **63.7** |

### Bimanual Table Bussing (Table 2)

| Model | Seen Objects | Mixed Objects |
|-------|-------------|---------------|
| Diffusion Policy | 45.8% | 31.2% |
| OpenVLA | 0% | 0% |
| **DiVLA-2B** | **72.9%** | **70.8%** |

### View Shifting Generalization (Table 6)

| Model | SR (%) |
|-------|--------|
| Diffusion Policy | 0% |
| OpenVLA | 0% |
| **DiVLA-2B** | **60%** |

### Model Scaling (Table 10)

| Model | Factory Sorting | Zero-Shot Bin Picking |
|-------|-----------------|---------------------|
| DiVLA-2B | 66.2% | 63.7% |
| DiVLA-7B | 74.9% | 66.7% |
| **DiVLA-72B** | **82.4%** | **75.9%** |

---

## 5. Ablation — Reasoning Injection Module (Table 8)

| Task | With FiLM | Without FiLM |
|------|----------|-------------|
| Task 1 | 100% | 66.7% |
| Task 2 | 100% | 66.7% |
| Task 3 | 63.6% | 45.5% |
| Task 4 | 63.6% | 45.5% |
| Task 5 | 90.9% | 27.3% |
| **Average** | **83.6%** | **50.3% (−33.3%p)** |

- FiLM 제거 시 **-33.3%p** → reasoning injection이 핵심 기여

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **Custom 벤치마크 위주**: LIBERO, CALVIN 등 표준 벤치마크 결과 없음 → 직접 비교 어려움
2. **GPT-4o annotation 의존**: Reasoning annotation 생성에 GPT-4o 비용. Self-generated가 이를 완화하나 초기 annotation 필요
3. **82Hz의 실체**: Action chunk 기반이므로 **effective control frequency**와 다를 수 있음. VLM reasoning 주기 미보고
4. **α=10의 sensitivity**: NTP vs diffusion loss 비율의 sensitivity 분석 부재
5. **DROID 39K의 특수성**: DROID가 다양한 환경을 포함하여 "적은 양이지만 높은 diversity"일 수 있음. 다른 39K dataset에서도 재현되는지 불명확

### Attribution 문제
- 83.6%가 **FiLM reasoning** 때문인지, **Qwen2-VL backbone** 때문인지, **diffusion action** 때문인지 → FiLM ablation (−33.3%p)이 가장 큰 기여를 보이나, backbone 변경 ablation은 부재

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — FiLM reasoning injection이 간결하고 효과적 |
| **Technical depth** | ★★★★☆ — Scaling (2B→72B), ablation 포괄적 |
| **Experimental rigor** | ★★★☆☆ — Custom 벤치마크, 표준 비교 부족 |
| **Practical impact** | ★★★★★ — 82Hz, 39K만으로 competitive, 미지 물체 63.7% |
| **Writing quality** | ★★★★☆ |

**강점**: 소량 데이터(39K)로 대량(970K) 모델을 압도하는 data efficiency. FiLM reasoning의 결정적 기여(+33.3%p). 72B까지의 scaling 분석. **약점**: 표준 벤치마크 부재, 82Hz의 실체 불명확, custom 평가의 재현성 우려.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO에서 얼마인가? | 미보고. 표준 벤치마크 결과 없어 직접 비교 불가 |
| 2 | 82Hz에서 VLM은 매 step 실행되는가? | 아니오. Action chunk 단위로 VLM 실행. Effective reasoning frequency는 훨씬 낮을 것 |
| 3 | DROID 39K가 특별한 것 아닌가? (diversity가 높아서) | 가능. 동일 양의 단일 환경 데이터에서 재현되는지 검증 필요 |
| 4 | ECoT와의 차이는? | ECoT는 GPT-4V로 structured CoT annotation, DiVLA는 FiLM으로 implicit injection. DiVLA가 더 효율적 (iterative cycle 없음) |
| 5 | Reasoning을 freeze하고 action만 학습하면? | FiLM ablation(−33.3%p)이 이를 간접적으로 보임. Reasoning이 action에 필수적 |

<!-- VERIFIED: pdf -->
