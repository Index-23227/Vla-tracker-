# ProgVLA: Progress-Aware Robot Manipulation Skill Learning

> **한 줄 요약**: DUNE ViT-Small backbone, T5 텍스트 인코더, 두 단계 Perceiver resampling, flow-matching action expert를 결합한 **0.1B param 초소형 VLA**가, 행동의 "진행도(progress)"를 예측하는 보조 head로부터 얻은 advantage·success weight를 모방학습 손실에 곱해 학습되어 — LIBERO에서 **91.1% avg**로 OpenVLA-7B(76.5)와 π₀-3.3B(86.0)를 70x 작은 모델로 능가한 NAVER LABS의 compact VLA(arXiv 2605.28231, 2026.05).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- OpenVLA(7B), π₀(3.3B), SmolVLA(2.25B) 등 최근 VLA는 모두 **수억~수십억 파라미터** — 학습/배포 비용이 막대하고, edge robot에서의 closed-loop control이 어려움
- VLA를 **순수 imitation learning**(behavior cloning)으로 학습할 때, 모든 demonstration step이 동일 weight를 가져 long-horizon task에서 부정확한 중간 step이 후반 step을 오염
- Demo data만으로는 어느 transition이 "task 완수에 유의미한가"라는 신호가 부족 → offline RL의 value function 개념을 imitation에 융합할 수 있는가?

### 핵심 질문
- **0.1B 규모의 VLA가 7B 모델을 능가할 수 있는가?**
- **Progress(task 진행도)를 예측하는 보조 head가 imitation loss를 어떻게 향상시키는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

ProgVLA는 다음 모듈로 구성:
- **DUNE Backbone (ViT-Small)**: 여러 vision specialist(2D + 3D task)에서 distill된 universal encoder
- **Frozen T5**: text encoder (학습 중 frozen)
- **Two-stage Perceiver Resampler**: 
  - Stage 1: per-modality(vision/text/state) bottleneck
  - Stage 2: post-fusion bottleneck
- **Flow-matching Action Expert**: prediction horizon 16, execution horizon 8 (receding horizon control)
- **Progress Heads** (Q, V, S): 행동의 가치·진행도·성공확률 예측

총 파라미터: **109M** (LIBERO에서 74M trainable, Meta-World에서 100M/65M trainable).

### 2.2 Progress Heads 설계

세 head가 context token을 공유:
- **Q-head**: state-action critic, return-to-go에 대한 Huber regression
- **V-head**: shared trunk, expectile regression(ρ=0.8)
- **S-head**: binary cross-entropy로 성공/실패 분류

이들의 **detached prediction**을 imitation loss에 곱하는 weight로 사용:

$$w_{A,t} = \min\{\exp(A_t / \beta), C\}, \quad w_{S,t} = 0.5 + 0.5 \cdot p_{\text{succ}}(c_t)$$

여기서 $A_t = Q_t - V_t$는 advantage, $p_{\text{succ}}$는 success probability.

> ❓ **예상 질문**: Progress head를 별도 학습하면 main policy와 disentangle되어 신호가 노이즈가 되지 않는가?
> **답변**: ProgVLA는 head들을 **context token을 공유**하도록 설계 — head와 policy가 동일한 latent representation을 공유하므로 internally-coupled. 단, head update시 detach하여 policy gradient는 받지 않음.

### 2.3 Two-stage Perceiver Resampling

핵심 아이디어: multimodal sequence를 한 번에 fusion하면 vision token이 dominate하므로,
1. **Per-modality resampler**: vision/text/state를 각각 독립적으로 작은 token set으로 압축
2. **Post-fusion resampler**: concatenated multimodal token을 다시 한 번 bottleneck

Ablation Table 2: context resampler 제거 시 LIBERO avg 91.1 → **75.1** (특히 long suite 88.6 → 51.2). 즉 resampler가 long-horizon task에 결정적.

---

## 3. 데이터 전략 및 학습

| 단계 | 데이터 | 규모 |
|------|--------|------|
| LIBERO | 4 suites × 10 tasks × 50 demos | 2,000 trajectory |
| Meta-World | 49 tasks × 50 demos | 2,450 trajectory |

- 모든 task를 단일 dataset으로 merge하여 학습
- **대규모 robot pretraining 없음** — benchmark demo만으로 학습
- 500 epochs, batch 256, AdamW
- LIBERO 25h / Meta-World 32h on single H100

> ❓ **예상 질문**: OpenVLA-7B(OXE 1M trajectory pretrain)과 동등 비교가 공정한가?
> **답변**: 불공정한 측면 있음 — OpenVLA의 강점은 zero-shot generalization이고, ProgVLA는 task-specific fine-tuning. 그러나 논문의 주장은 "0.1B로도 LIBERO 91.1% 달성 가능"이지 "OpenVLA보다 우월"이 아님.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 1)

| 모델 | Params | Spatial | Object | Goal | Long | **Avg** |
|------|--------|---------|--------|------|------|---------|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 3.3B | 90 | 86 | 95 | 73 | 86.0 |
| SmolVLA | 2.25B | 93 | 94 | 91 | 77 | 88.75 |
| **ProgVLA** | **0.1B** | **87.6** | **96.0** | **92.0** | **88.6** | **91.1** |

- **Long suite에서 두드러진 우세**(88.6 vs SmolVLA 77, π₀ 73): progress head의 효과가 long-horizon에서 가장 큼
- Spatial은 SmolVLA(93)보다 낮으나, 다른 모든 지표에서 우월

### 4.2 Meta-World (Table 1)

| 모델 | Params | Easy | Medium | Hard | Very Hard | **Avg** |
|------|--------|------|--------|------|-----------|---------|
| SmolVLA | 2.25B | 87.14 | 51.82 | 70 | 64 | 68.24 |
| **ProgVLA** | **0.1B** | **84.9** | **72.7** | **77.0** | **79.6** | **78.5** |

- 22.5x 작은 ProgVLA가 +10.3 avg
- Medium/Hard/Very Hard에서 큰 격차 → progress signal이 난이도가 높을수록 효과적

### 4.3 Real-World (PiPER 6-DOF)

10 task × 10 trial = 100 trial: **68% 평균 success**.
- 2개 RealSense 카메라(D405 wrist + D435 agent)
- 학습 시 사용한 데모만으로 평가

### 4.4 Ablation (Table 2, LIBERO)

| 변형 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| **Full Model** | **87.6** | **96.0** | **92.0** | **88.6** | **91.1** |
| w/o progress objectives | 87.0 | 90.6 | 90.2 | 85.1 | 88.8 |
| w/o context resampler | 84.4 | 77.2 | 87.4 | 51.2 | 75.1 |
| DINOv3 backbone | 88.2 | 92.4 | 93.2 | 81.0 | 88.7 |
| frozen DUNE | 79.0 | 85.2 | 85.4 | 60.6 | 77.6 |

핵심 관찰:
- **Context resampler가 가장 큰 기여**(−16.0 avg, long −37.4)
- Progress objective는 +2.3 avg(주로 long 88.6 vs 85.1)
- DUNE > DINOv3 by 약 2.4 avg
- **DUNE을 freeze하면 −13.5 avg** → backbone fine-tuning 필수

---

## 5. 강점

| 항목 | 평가 |
|------|------|
| **Compactness** | ★★★★★ — 0.1B로 7B 능가, edge deployment 가능 |
| **Novelty** | ★★★★☆ — Progress head + dual Perceiver의 조합 |
| **Experimental rigor** | ★★★★☆ — 풍부한 ablation, 3개 benchmark + real robot |
| **Practical impact** | ★★★★☆ — Single H100 25h 학습, laptop inference 가능 |
| **Open access** | ★★☆☆☆ — Code 미공개("to be released") |

---

## 6. 한계 및 미해결 문제

### 6.1 방법론적
1. **Progress head의 contribution이 작음**(+2.3 avg) — Context resampler(+16)에 비해 marginal. 논문의 핵심 주장이 ablation으로 강하게 뒷받침되지 않음
2. **대규모 pretrain 부재**: zero-shot generalization 능력은 평가되지 않음 — OpenVLA처럼 OXE pretrain 후 LIBERO fine-tune이 아니라 LIBERO만으로 학습. 따라서 unseen embodiment/task에서의 generalization은 unknown
3. **Frozen text encoder**: T5가 frozen이라 language reasoning 능력이 제한적. Complex instruction에 대한 평가 부재

### 6.2 실험적
1. **Real-world 68%는 단일 embodiment**: PiPER 6-DOF만으로 평가 — bimanual, dexterous hand 등 다양한 embodiment에서의 일반화 결과 부재
2. **No closed-loop async control**: pi0.5와 같이 chunk execution 중 inference하는 async 평가 없음. Real-time control rate 미보고

### 6.3 Attribution
- Long suite 큰 향상이 progress head 덕인지 perceiver resampler 덕인지 ablation에서 명확히 분리되지 않음(둘 다 long에 큰 효과)

---

## 7. 관련 연구 비교

| 모델 | Params | LIBERO Avg | Progress / Value head | Backbone |
|------|--------|-----------|----------------------|----------|
| OpenVLA | 7B | 76.5 | ✗ | Llama 2 |
| π₀ | 3.3B | 86.0 | ✗ | PaliGemma |
| SmolVLA | 2.25B | 88.75 | ✗ | SmolVLM-2 |
| **ProgVLA** | **0.1B** | **91.1** | **Q/V/S** | **DUNE + T5** |

핵심 차이:
- **유일하게 progress head를 imitation loss weighting에 활용**
- 70x 작은 모델로 SOTA 달성

---

## 8. 예상 날카로운 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 0.1B로 어떻게 7B를 이기는가? | LIBERO는 task-specific fine-tuning. 모델 크기보다 inductive bias(perceiver resampler)와 training signal(progress) 설계가 중요 |
| 2 | Progress head 없이 87.0→ +2.3은 작은 차이 아닌가? | Long suite에서는 +3.5 — long horizon에서 더 두드러짐. 그러나 main contribution은 인정해도 marginal |
| 3 | OXE pretrain 없이 zero-shot 일반화는? | 평가 부재 — 논문은 task-specific fine-tuning 시나리오만 다룸 |
| 4 | Real-world 68%는 어떤 baseline 대비? | SmolVLA, π₀ 등의 real-world 비교 부재. Absolute number만 제시 |
| 5 | DUNE backbone의 universality는 검증되었는가? | Ablation에서 DINOv3 88.7 vs DUNE 91.1 — DUNE이 +2.4 우월. 그러나 manipulation에 특화된 backbone과의 비교는 없음 |
| 6 | Flow matching의 multi-step inference latency는? | "Laptop에서 real-time"이라고 하나 정확한 Hz 미보고 |
| 7 | Code 미공개 → reproducibility 우려? | "Upon publication" 약속, 그러나 현재 미공개 |
| 8 | Progress signal이 GT label에서 오는가? | Demonstration의 success label만 사용 → no human progress annotation. 자체 supervised |

<!-- VERIFIED: pdf -->
