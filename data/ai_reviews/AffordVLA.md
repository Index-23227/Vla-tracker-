# AffordVLA: Injecting Affordance Representations into Vision-Language-Action Models via Implicit Feature Alignment

> **한 줄 요약**: Zhejiang University Grasp Lab의 π0.5 기반 VLA로, frozen zero-shot affordance teacher(Qwen3-VL + SAM3)의 representation을 VLA의 12층 visual feature에 **cosine-similarity loss(λ=0.5)** 로 직접 정렬. Explicit mask injection 없이 affordance prior를 internalize하여 cascading perception error를 회피하고, **12.4 Hz inference**(explicit-affordance baseline의 2.6×) 달성. RoboTwin2.0 Easy 61.2% / Hard 28.8%, 8-task real-world에서 π₀, π₀.₅, π₀.₅+explicit-affordance 모두 능가.

> ⚠️ **혼동 주의**: 같은 시기 Fudan/KAUST의 별도 paper "Afford-VLA: Action-Aligned Visual Planning via Internalized Affordance" (arXiv 2605.24203)와 무관한 다른 팀의 연구. 본 review는 arXiv 2605.17517(Zhejiang Grasp Lab)을 다룸.

---

## 1. 배경 및 동기

### 기존 affordance-augmented VLA의 구조적 한계
- 기존 affordance-VLA 접근(예: π₀.₅+explicit-affordance)은 **affordance mask를 perception module에서 추출 → VLA 입력으로 concat**하는 cascade 구조
- 두 가지 문제:
  1. **Cascading perception error**: affordance mask가 noisy하면 그 오류가 VLA에 그대로 전파
  2. **Inference overhead**: SAM 류 모델 forward + VLA forward → 4.8 Hz로 latency 증가
- Explicit mask가 정말 필요한가, 아니면 affordance **knowledge**만 internalize하면 충분한가?

### 핵심 질문
- **Affordance teacher의 representation만 VLA의 intermediate feature와 align해도 explicit mask injection과 동등 이상의 성능을 얻을 수 있는가?**
- **이 방식이 inference latency를 얼마나 절감하는가?**

📌 [Figure 1 삽입] — Implicit Feature Alignment 구조: Affordance teacher (frozen) → MLP projector → cosine loss with VLA layer-12 features

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

**Base VLA**: π0.5 (PaliGemma-3B = SigLIP-So400m + Gemma-2B + 18-layer flow-matching action expert)
- Understanding expert: ~3.0B
- Action expert: ~0.3B (chunk size H=30)

**Affordance Teacher** (Frozen): 두 모듈로 구성
1. **Task parsing**: Qwen3-VL → language + RGB → part-level affordance concept prompts
2. **Open-vocabulary perception**: SAM3-based → pixel-level affordance prediction + visual representation Z_t^aff (d=256)

### 2.2 Implicit Feature Alignment

핵심 contribution. VLA의 **12층 visual feature**(d_v=2048)와 teacher의 affordance representation(d=256)을 정렬:

```
1. Normalize VLA features
2. Bilinear interpolation으로 spatial resolution 정렬
3. 2-layer MLP로 2048 → 256 projection
4. Cosine similarity loss:
   L_align = -1/N · Σ cos(x̂_t,i^V,(m), z̃_t,i^aff)
```

> ❓ **예상 질문**: 왜 12층인가? Earlier/later layer는?
> **답변**: Paper에서 "intermediate-deep layer가 semantic abstraction과 visual detail의 balance"라고 정당화. Earlier layer는 low-level texture에 가까워 affordance 같은 functional semantics를 표현하기 어려움. Later layer는 action-specific feature로 collapse되어 teacher와의 alignment가 부자연스러움. 12층 ablation의 sensitivity는 보고되지 않음.

> ❓ **예상 질문**: Cosine similarity가 KL/MSE보다 나은 이유는?
> **답변**: Magnitude를 무시하고 direction만 정렬 → teacher와 VLA가 서로 다른 표현 공간 scale을 사용해도 robust. 또한 VLA의 다른 downstream task(action generation)와 충돌이 적음 — magnitude 자유도를 보존.

### 2.3 Combined Loss

```
L_AffordVLA = L_action + λ · L_align,  λ = 0.5
```

L_action은 π0.5의 conditional flow-matching loss. λ=0.5는 두 loss의 균형.

> ❓ **예상 질문**: λ=0.5의 sensitivity는?
> **답변**: Ablation 부재. 그러나 representation-level alignment가 action loss와 너무 강하게 경쟁하면 action 정확도 저하 가능. 0.5는 heuristic equal weighting.

### 2.4 Inference Efficiency

| 모델 | Latency | Hz |
|------|---------|-----|
| π₀.₅ + explicit affordance | 206.9 ms | 4.8 |
| **AffordVLA** | **80.4 ms** | **12.4** |

→ **2.6× 빠름**. Teacher가 inference 시 호출되지 않음 — alignment는 학습 시에만 사용.

---

## 3. 데이터 전략

| Phase | Dataset |
|-------|---------|
| Simulation | RoboTwin2.0 (5 tasks, 100 trials/task) |
| Real-world | 50 disturbance-free demos per task, 8 tasks |
| Affordance eval | AGD20K (unseen split) |

> ❓ **예상 질문**: Real-world 50 demos/task로 충분한가?
> **답변**: 매우 적음. 그러나 base π0.5가 OXE 사전학습 → AffordVLA는 affordance prior로 sample-efficient한 fine-tuning을 달성한다는 주장. Sample efficiency ablation(Fig 10a)에서 alignment 적용 시 45% success를 ~5.2k iteration 일찍 도달.

---

## 4. 시스템/학습 세부사항

| 항목 | 설정 |
|------|------|
| Hardware | 1× NVIDIA A100-SXM4-80GB |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | linear warmup → 1e-4 (10k steps), cosine decay → 5e-5 |
| Batch size | 32 (global) |
| Steps | 30,000 (sim) / 50 epochs (real) |
| Precision | bfloat16 mixed |
| Teacher | frozen throughout |

> ❓ **예상 질문**: A100 한 장으로 학습한 모델이 production VLA를 능가한다?
> **답변**: π0.5의 강력한 pretraining 덕분에 single-GPU fine-tuning이 가능. AffordVLA는 fine-tuning method이지 from-scratch가 아님 — 본 paper의 contribution은 representation alignment design이지 scale이 아님.

---

## 5. 실험 결과 심층 분석

### 5.1 AGD20K Affordance Prediction (Unseen split)

| Metric | AffordVLA | 비고 |
|--------|-----------|------|
| KLD | **0.905** | 최저 (best) |
| SIM | 0.496 | Espresso-2D 0.503 다음 second-best |
| NSS | **1.906** | 최고 (best) |

→ VLA 내부의 12층 feature가 stand-alone affordance prediction에서도 경쟁력 — alignment가 정상 작동함을 증명.

### 5.2 RoboTwin2.0 Simulation (5 tasks, 100 trials)

| Setting | AffordVLA | 베스트 baseline | 차이 |
|---------|-----------|---------------|------|
| Easy avg | **61.2%** | DP3 ~46.4% | +14.8pp |
| Hard avg | **28.8%** | RDT ~10.2% | +18.6pp |

> ⚠️ **Hard 28.8%는 absolute 값으로 낮음** — RoboTwin2.0 Hard가 매우 어려운 setting이며, AffordVLA의 상대적 우위(+18.6pp)는 의미 있으나 절대 성능은 production-ready 수준 아님.

### 5.3 Real-World (UR5 + Robotiq 2F-85, 8 tasks × 15 trials)

8 tasks: pouring, hanging mug, cutting banana, striking block, sweeping, wiping, placing marker, cluttered sorting.

| Baseline | 결과 |
|---------|------|
| π₀ | inferior to AffordVLA across all 8 |
| π₀.₅ | inferior to AffordVLA across all 8 |
| π₀.₅ + explicit affordance | inferior + 2.6× slower |
| **AffordVLA** | **best across all 8 tasks** |

> ❓ **예상 질문**: Exact per-task 숫자는?
> **답변**: Paper에서 "best across all 8" 정성적 진술 — 본 review 데이터에서는 task-by-task 정량 숫자 분리 어려움. 정확한 수치는 paper Table 확인 필요.

### 5.4 Ablation

| Ablation | 결과 |
|----------|------|
| W/o alignment | RoboTwin2.0 Hard 성능 significant degradation (Fig 8) |
| Sample efficiency (Fig 10a) | With alignment: 45% success @ -5.2k iterations |
| Attention shift (Fig 9) | Dispersed global → task-relevant functional regions |
| Representation analysis (Fig 10b) | t-SNE: aligned VLA features adopt teacher distribution structure while preserving VLA-specific semantics |

핵심 발견:
- Alignment loss 제거 시 Hard setting에서 가장 큰 손실 — 어려운 task일수록 affordance prior가 중요
- Attention map이 더 "intentional"하게 변함 — interpretability 측면의 부수적 이득

---

## 6. 관련 연구 비교

| Model | Affordance integration | Inference Hz | Cascading error |
|-------|----------------------|--------------|----------------|
| π₀ | ✗ | ~12 | N/A |
| π₀.₅ | ✗ | ~12 | N/A |
| π₀.₅ + explicit affordance | mask concat | 4.8 | ✗ (vulnerable) |
| Afford-VLA (Fudan, 2605.24203) | learned <AFF> tokens → flow matching | - | partial |
| **AffordVLA (Zhejiang, 2605.17517)** | **implicit feature alignment** | **12.4** | **✓ (avoided)** |

### 핵심 차이
- Explicit mask 없이 representation level에서 affordance를 internalize
- Teacher가 inference 시 비활성 → π₀ 수준의 latency 유지
- 다른 affordance paper(Afford-VLA Fudan)는 <AFF> query token + decoder를 학습하는 반면, AffordVLA는 frozen teacher의 representation을 단순 align만 함 — 더 가벼움

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Layer choice (12층) sensitivity 부재**: 다른 layer에서의 alignment 결과 미보고
2. **λ=0.5 sensitivity 부재**: λ에 대한 sweep 없음 — heuristic 선택
3. **Teacher dependency**: Qwen3-VL + SAM3가 학습 시 필요 → reproduction cost 증가
4. **LIBERO/SimplerEnv 미평가**: RoboTwin2.0 + real-world만 — 다른 manipulation benchmark에서의 일반화 불명확
5. **Single A100 학습**: 30k steps만으로 full performance라면 RoboTwin2.0 task의 simplicity 의문 (5 tasks)
6. **Real-world task별 정량 수치 모호**: 8 tasks 전체에서 "best" 진술뿐, per-task 성공률 disclose 부족
7. **Code 미공개**: GitHub repo 없음 ("Report GitHub Issue" 링크만 존재)
8. **Hard setting 28.8% 절대값**: 상대 우위는 크나 production deployment 한참 미달

### Attribution 문제
- 성능 향상이 (a) affordance representation의 본질적 가치, (b) teacher의 강력함(Qwen3-VL + SAM3), (c) implicit vs explicit alignment의 design choice 중 어디에서 오는지 분리 불완전
- Teacher를 다른 zero-shot affordance 모델로 교체했을 때의 결과 부재
- AGD20K에서의 강한 성능이 base π0.5의 visual feature 덕인지 alignment training 덕인지 separation 미흡 — frozen π0.5 baseline 측정 필요

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Implicit feature alignment design은 explicit mask injection 대비 새로움 |
| **Technical depth** | ★★★★☆ — Cosine alignment + intermediate layer choice + harmonic teacher 구성 정교 |
| **Experimental rigor** | ★★★☆☆ — RoboTwin2.0 + real-world 8 tasks + AGD20K. LIBERO 부재, per-task real-world 수치 부족 |
| **Practical impact** | ★★★★☆ — 12.4 Hz inference로 explicit affordance baseline 2.6× 빠름. Real-world deployment-friendly |
| **Reproducibility** | ★★☆☆☆ — Code 미공개, teacher(SAM3 등) 의존성, layer/λ sensitivity 부재 |

**강점**: Implicit alignment의 design은 직관적이며 inference latency를 보존. AGD20K에서 standalone affordance prediction 경쟁력 입증. Real-world 8 tasks에서 explicit baseline 일관 능가. **약점**: Hyperparameter sensitivity(layer 12, λ=0.5) 분석 부재, code 미공개, LIBERO 부재. Hard setting 28.8% 절대값은 RoboTwin2.0 Hard의 본질적 어려움을 반영하나 실용 성능으로는 부족.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 같은 이름의 Fudan Afford-VLA(2605.24203)와 무엇이 다른가? | 완전히 다른 팀(Zhejiang vs Fudan/KAUST), 다른 방법론(implicit alignment vs <AFF> query token), 다른 benchmark(RoboTwin2.0 vs LIBERO/SimplerEnv). 공교롭게 동시기 출판 |
| 2 | 왜 12층인가? Layer ablation은? | "Semantic-detail balance"라는 정성적 정당화만. Sweep 없음. 11/13층 또는 마지막 layer alignment 결과 부재 |
| 3 | Teacher를 다른 모델(예: GroundingDINO+SAM2)로 바꾸면? | 미평가. Teacher quality에 대한 robustness 불명 |
| 4 | LIBERO에 적용하면? | 미평가. RoboTwin2.0만 — LIBERO는 affordance가 덜 중요한 task가 많아 alignment 효과가 약할 가능성 |
| 5 | Cosine similarity vs MSE/KL는? | Ablation 부재. Cosine은 magnitude-free라는 정성적 이유만 |
| 6 | Per-task real-world 수치는? | "Best across all 8" 진술뿐. Pouring/Cutting 등 specific task에서 baseline과의 차이 정량 미공개 |
| 7 | 50 demos/task로 학습한 real-world 결과의 신뢰성은? | 15 trials/task는 통계적 power 약함. 95% CI를 보면 baseline과 overlap 가능성 |
| 8 | Inference 12.4 Hz는 action expert만의 latency인가? | Pi0.5의 chunked action(H=30)을 한번에 생성하므로, 실효 control rate는 chunk_size × 12.4 Hz의 effective control |
| 9 | Teacher (SAM3)가 학습 시에 cost가 큰가? | Frozen이므로 forward만. SAM3 forward는 ~10초 가량이라 학습 throughput에 영향. Paper에서 학습 시간 disclose 없음 |
| 10 | AGD20K KLD 0.905, SIM 0.496이 implicit alignment의 직접 증거인가? | 그렇다고 주장. 그러나 frozen π0.5 visual feature 자체의 affordance correlation을 baseline으로 측정해야 alignment의 marginal 효과가 명확 |

<!-- VERIFIED: pdf -->
