# Coarse-to-Control: Action-Token Planning for Vision-Language-Action Models

> **한 줄 요약**: PaliGemma-3B 기반 VLA에 plan-execute 구조를 도입하되, **plan을 자연어/이미지가 아닌 "coarse action token"으로** 표현하는 방법. 학습된 joint-mode residual-VQ tokenizer가 단기 실행 토큰과 장기 계획 토큰을 동일 vocabulary로 공유시켜, plan이 곧 actionable한 control-manifold 신호가 되도록 한다. LIBERO 97.9%, SimplerEnv-WidowX 83.3%, 실세계 4-task 평균 62.5%를 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Direct-generation VLA** (OpenVLA, π0 등)는 관찰에서 곧장 모터 명령을 생성 → 장기 horizon task에서 초반 오차가 누적되어 실패
- **Textual CoT** (Embodied-CoT, ThinkAct)는 의미적 분해는 잘 하지만 low-level 모터에 대한 제약이 약함
- **Visual CoT** (CoT-VLA, DreamVLA, F1)는 공간적 직관은 주나, **실행 불가능한 긴 visual prefix**를 매번 생성해야 해 효율과 alignment 문제 발생

### 핵심 질문
- **언어/이미지가 아닌 "action 자체의 추상화"가 CoT의 매체가 될 수 있는가?**
- Plan 토큰과 execution 토큰이 **하나의 discrete action vocabulary**를 공유하면 planner-controller interface 자체가 사라지는가?

📌 [Figure 1 삽입] — 텍스트/비주얼/액션 CoT 비교: Coarse-to-Control은 plan을 coarse action token으로 표현

---

## 2. 방법론 심층 분석

### 2.1 전체 흐름 (plan-then-execute)

자기회귀 분해:
```
p(z^exec_t | obs_t, lang) = Σ_{z^plan_t} p(z^plan_t | obs_t, lang) · p(z^exec_t | z^plan_t, obs_t, lang)
```
- 입력: multimodal prefix (image tokens + language + proprioception)
- 출력 suffix 순서: **planning tokens → executable tokens**
- 추론 시 plan token은 internal prefix로만 사용되고, **executable token만** 연속 액션으로 디코딩됨

### 2.2 Dual-granularity Action Tokenizer (논문의 핵심)

**Joint-mode residual-VQ tokenizer**:
- 단일 codec이 두 mode를 가짐: execution mode(m=0), planning mode(m=1)
- **두 mode가 동일한 discrete action vocabulary 공유** ← 이게 핵심 차별점
- Execution mode: 짧은 horizon H_e 실행 액션을 인코딩
- Planning mode: 긴 H_p 소스 트랙젝토리를 chunking 후 motion은 chunk별 합, gripper는 마지막 값 → He-step **coarse plan representation**으로 sub-resolution화한 뒤 인코딩

토크나이저 사양:
- 3개 residual VQ codebook × 4096 entries
- Codec latent dim 64, action dim 7, mode-condition dim 64
- LIBERO에서 한 branch당 2(temporal patch) × 7(action dim) × 3(residual) = **42 action tokens**
- Loss = action L1 reconstruction + codebook loss + 0.25 × commitment

> ❓ **예상 질문**: planning과 execution이 정말 같은 vocab을 써야 하나? Separate가 더 자연스럽지 않은가?
> **답변**: Table 4에서 Faster-AR(95.40) → Separate(96.60) → Joint-mode(97.90). Separate도 plan을 두는 것만으로 +1.2%p 얻지만, Joint가 추가로 +1.3%p, 특히 Long suite에서 91.60→95.00으로 큰 격차. Separate는 execution이 plan vocab을 implicit하게 번역해야 해서 conditioning 신호가 약해진다는 해석.

### 2.3 VLA Policy (autoregressive)

- Backbone: **PaliGemma-3B 기반 local VLA**, SigLIP vision + PaliGemma projector
- 초기화: **π0-FAST checkpoint**로부터 시작
- 구조 사양: VLM stream hidden 2048, proprio/action stream hidden 1024, 18 layer, 8 attn head, 1 KV head, head dim 256, max position 8192
- Teacher forcing으로 plan-execute suffix 전체 학습
- 평가 시 EMA weight 로드

> ❓ **예상 질문**: π0-FAST에서 초기화하는데 결국 backbone이 같은데 향상은 plan 덕분인가 그냥 추가 토큰 덕분인가?
> **답변**: Table 3의 H_p=0 ablation이 정확히 그 질문에 답한다. 같은 joint tokenizer를 쓰되 planning horizon만 0으로 놓으면 96.45%, H_p=40 →97.55%, H_p=160 → 97.90%. Plan supervision 자체가 1.45%p 기여. 즉 향상은 추가 토큰이 아니라 **explicit한 future task structure 예측**에서 옴.

---

## 3. 데이터 전략

### Tokenizer pretraining (Table 8)
| 환경 | 데이터 | 가중치 |
|------|-------|-------|
| Simulation | LIBERO | 5.0 |
| Simulation | Bridge | 1.0 |
| Real world | Fractal | 1.0 |
| Real world | Kuka | 1.0 |
| Real world | Bridge | 1.0 |
| Real world | Droid (EEF) | 1.0 |
| Real world | LIBERO | 5.0 |

토크나이저는 8 × H200 GPU에서 별도 사전학습 후 정책 학습 동안 **고정**.

### Downstream policy 학습 (Table 9)
| 설정 | 데이터 | Horizon (exec/plan) | 학습 |
|------|-------|---------------------|------|
| LIBERO | LIBERO demos | 20 / 160 | batch 4, 60k steps |
| SimplerEnv-WidowX | Bridge | 10 / 80 | batch 16, 4 epochs |
| Real world | 50 demos/task × 4 task | 20 / 160 | batch 8, 30k steps |

> ❓ **예상 질문**: 50 demo/task는 너무 적지 않나?
> **답변**: 의도된 setup. π0, faster, π0-fast 모두 같은 50-demo budget으로 비교. 실세계 강건성은 적은 데이터에서 plan supervision의 inductive bias가 어떻게 작동하는지 보려는 것.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|----|
| Backbone | PaliGemma-3B VLA (SigLIP vision) |
| 초기화 | π0-FAST checkpoint |
| Action 표현 | 7-DoF EE, q99 normalization |
| Action horizon (LIBERO/real) | 20-step executable, 160-step plan source → K=20 coarse plan |
| Action horizon (SimplerEnv) | 10-step exec, 80-step plan → K=10 |
| Optimizer | AdamW, LR 2.5×10⁻⁵, 1k warmup, cosine decay |
| Weight decay | 1×10⁻¹⁰ |
| Tokenizer | Residual-VQ, 3 codebook × 4096, codec latent 64 |
| Codebook loss | L1 + codebook + 0.25 × commitment |
| Hardware (tokenizer) | 8 × H200 |
| 평가 | EMA weight, replan every 10 steps (SimplerEnv) |

---

## 5. 실험 설계 및 평가 프로토콜

세 갈래로 평가:
1. **LIBERO** — 4 suite × 50 trials/task, suite-level + overall success
2. **SimplerEnv-WidowX** — 4 WidowX manipulation task × 24 trials, real-to-sim generalization
3. **Real-world** — 4 physical task × 20 trials, 50 demos/task

비교 baseline은 CoT 패러다임별 분류:
- No CoT: π0-FAST, SmolVLA, GR00T-N1, π0, OpenVLA-OFT
- Textual CoT: ThinkAct, π0.5
- Visual CoT: CoT-VLA-7B, WorldVLA, DreamVLA, UniVLA, F1, UD-VLA
- Action CoT: MolmoAct-7B-D, **Coarse-to-Control (ours)**

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO (Table 1)

| Method | Spatial | Object | Goal | Long | **Overall** |
|--------|---------|--------|------|------|-------------|
| OpenVLA-OFT (no-CoT SOTA) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0.5 (textual CoT) | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| F1 (visual CoT) | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| UniVLA (visual CoT) | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| MolmoAct-7B-D (action CoT) | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| **Coarse-to-Control** | **98.8** | **100.0** | 97.8 | **95.0** | **97.9** |

- 모든 CoT 카테고리를 통틀어 **overall 최고** (97.9%)
- 특히 같은 Action CoT 계열인 MolmoAct-7B-D 대비 **+11.3%p**의 압도적 격차 — action token 정렬이 핵심임을 시사
- Object suite **100%** 달성

### 6.2 SimplerEnv-WidowX (Table 2)

| Method | Spoon | Carrot | Stack | Eggplant | **Overall** |
|--------|-------|--------|-------|----------|-------------|
| π0 | 29.1 | 0.0 | 16.7 | 62.5 | 40.1 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| F1 (visual CoT) | 50.0 | 70.8 | 50.0 | 66.7 | 59.4 |
| UD-VLA (visual CoT) | 58.3 | 62.5 | 54.1 | 75.0 | 62.5 |
| **Coarse-to-Control** | **100.0** | **95.8** | **79.2** | 58.3 | **83.3** |

- 이전 best(UD-VLA 62.5) 대비 **+20.8%p**의 큰 격차
- Eggplant 한 task에서만 baseline 대비 손실 (58.3 vs 75.0) — 단순 단일 stage task라 plan의 advantage가 약함

### 6.3 Real-world (Figure 3)

| Task | π0 | Faster | π0-FAST | **C2C** |
|------|----|----|----|----|
| Task 1 (Carrot) | **80** | 60 | 60 | 75 |
| Task 2 (Carrot+Button) | 50 | 35 | 50 | **70** |
| Task 3 (Plate→Basket) | 40 | 35 | 35 | **60** |
| Task 4 (Cleanup) | 30 | 20 | 25 | **45** |
| **Average** | 50 | 38 | 43 | **63** |

- 4 task 중 **3개에서 SOTA**, 평균 62.5%
- 단일 stage Carrot은 π0 우세 (80 vs 75) — plan의 이점이 단순 task에서는 marginal
- **multi-stage task로 갈수록 격차 확대** (Cleanup에서 +15~25%p)

> ❓ **예상 질문**: 단일 stage에서 π0보다 약간 떨어지는 이유?
> **답변**: 저자도 인정. 짧은 task는 plan token이 오히려 capacity를 잡아먹는 overhead가 됨. Plan-execute 구조의 가치는 stage transition(approach → grasp → transport → place)이 있는 경우에 빛난다.

---

## 7. Ablation 분석

### 7.1 Planning horizon (Table 3, joint tokenizer 고정)

| H_p | Spatial | Object | Goal | Long | **Overall** |
|-----|---------|--------|------|------|-------------|
| 0 (no plan) | 97.00 | 99.60 | 95.00 | 94.20 | 96.45 |
| 40 | 98.60 | 99.60 | 97.80 | 94.20 | 97.55 |
| 160 | 98.80 | 100.00 | 97.80 | 95.00 | **97.90** |

- Plan 자체로 +1.10%p, plan을 늘리면 추가 +0.35%p
- **Long suite에서 더 큰 horizon 효과** — stage transition을 인식하려면 충분히 긴 future context 필요

### 7.2 Tokenizer sharing (Table 4)

| Tokenizer | Spatial | Object | Goal | Long | **Overall** |
|-----------|---------|--------|------|------|-------------|
| Faster-AR (no plan) | 99.40 | 98.80 | 94.80 | 88.60 | 95.40 |
| Separate (2 indep VQ) | 97.40 | 99.60 | 97.80 | 91.60 | 96.60 |
| **Joint-mode (ours)** | 98.80 | 100.00 | 97.80 | **95.00** | **97.90** |

- Plan 추가만으로 +1.2%p (Separate), vocab 공유로 추가 +1.3%p (Joint), **Long에서 +3.4%p**
- **공유 vocabulary**가 단순한 design 선택이 아닌, 성능 자체에 본질적

### 7.3 정성 분석 (Figure 5)

- (a) Attention map: plan branch가 활성화되면 첫 frame의 attention이 **target object + gripper + target region**에 집중. No-plan은 분산.
- (b) Decoded coarse plan trajectory: executable action이 생성되기 전에 plan token이 이미 target을 향해 가는 방향을 가리킴

### 7.4 실세계 transfer 분석 (Table 7)

| Method | Carrot SR | Carrot+Button SR | Avg | **Success steps ↓** |
|--------|-----------|------------------|-----|---------------------|
| faster | 40.0 | 20.0 | 30.0 | 436 |
| **plan** | **70.0** | **35.0** | **52.5** | **239** |

- 성공한 rollout의 평균 control step이 436 → 239로 **거의 절반** — plan이 더 효율적인 trajectory 유도

---

## 8. 관련 연구 비교

| 모델 | CoT 형태 | 매체 | Planner-controller interface | LIBERO Avg |
|------|---------|------|------------------------------|-----------|
| ThinkAct | Textual CoT | 자연어 | 명시적 (text→action 번역 필요) | 84.4 |
| π0.5 | Textual CoT | 자연어 | 명시적 | 96.8 |
| CoT-VLA-7B | Visual CoT | 미래 이미지 | 명시적 (image→action) | 81.1 |
| DreamVLA | Visual CoT | World model | 명시적 | 92.6 |
| MolmoAct-7B-D | Action CoT | depth/3D action | 부분적 (수치 거친 표현) | 86.6 |
| **Coarse-to-Control** | **Action token CoT** | **공유 VQ vocab** | **없음 (자동 통합)** | **97.9** |

### 핵심 차이
- **유일하게 plan과 execution이 같은 vocabulary** — 따라서 plan은 actionable
- **추가 modality 불필요** — visual CoT의 긴 prefix 생성 부담 없음
- **π0-FAST initialization** 활용으로 large-scale pretrain 이점 흡수

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **단일 backbone (PaliGemma-3B)** 만 검증. 7B 이상 또는 다른 architecture에서 vocabulary alignment가 동일하게 효과적인지 불명
2. **Tokenizer 사전학습 의존도**: 토크나이저 품질이 낮으면 plan-execute alignment가 무너질 위험. Ablation 없음
3. **Real-world task 4개** — 모두 tabletop pick-and-place 류. Articulated/contact-rich/bimanual은 미검증
4. **Plan 검증 mechanism 부재**: plan이 잘못 생성되어도 execution이 그대로 따름. self-correction 없음
5. **Inference latency** 보고 없음 — plan token 디코딩의 추가 비용이 어느 정도인지 미공개
6. **Open source 미공개** — paper에 code URL 없음, 재현성 우려
7. **K(coarse plan steps)** 와 **chunk size** sensitivity 부분 미검증 — chunk size 8(160/20)과 2(40/20) 두 점만 비교

### Attribution 문제
- π0-FAST initialization 효과와 plan-execute 구조 효과의 분리가 명확하지 않음
- LIBERO의 Long suite 성과는 PaliGemma 자체 capability 때문일 가능성

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Plan과 execution을 같은 vocab에 두는 단순하지만 본질적인 아이디어 |
| **Technical depth** | ★★★★☆ — Joint-mode VQ tokenizer 수식과 ablation 체계적 |
| **Experimental rigor** | ★★★★☆ — LIBERO + SimplerEnv + real-world 3축 모두 검증 |
| **Practical impact** | ★★★★☆ — π0-FAST 호환 backbone, 실세계 multi-stage에서 큰 격차 |
| **Writing quality** | ★★★★☆ — CoT 분류 체계가 명확, 비교표 깔끔 |

**강점**: "Action token이 CoT 매체가 될 수 있다"는 주장을 joint vocabulary VQ tokenizer로 깔끔하게 구현. 같은 Action CoT 계열인 MolmoAct 대비 +11.3%p의 격차는 plan-execution alignment의 중요성을 강하게 시사. Real-world multi-stage task에서의 robustness가 인상적.
**약점**: Open source 미공개, latency 미공개, PaliGemma-3B 단일 backbone, contact-rich/bimanual real-world 미검증.

---

## 11. 후속 연구 방향

1. **Plan token interpretability**: VQ codebook entry를 정성 분석하여 plan token이 어떤 motor primitive를 인코딩하는지 탐구
2. **Larger backbones**: 7B/14B PaliGemma 또는 Qwen-VL backbone에서 scaling law
3. **Adaptive plan horizon**: task 복잡도에 따라 H_p를 동적으로 조절
4. **Plan verification**: plan token이 잘못 생성되면 reject/regenerate하는 mechanism
5. **Multi-level CoT**: action token CoT 위에 textual goal decomposition을 얹는 hybrid

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|-----------|
| 1 | "Action token CoT"가 MolmoAct에도 있는데 왜 그것은 86.6%, C2C는 97.9%? | MolmoAct는 depth/3D pose 같은 별도 매체. C2C는 진짜 **execution과 동일한 VQ vocab** 사용. Table 4의 Separate vs Joint가 이 차이를 +1.3%p로 정량화 |
| 2 | π0-FAST 초기화 효과를 어떻게 분리하나? | Table 3, 4의 baseline(Faster-AR, no plan)도 같은 초기화 사용. 95.40 → 97.90의 +2.5%p는 plan-execute 구조 자체의 기여 |
| 3 | Plan token이 정말 "의미 있는 미래"를 담는다는 증거는? | Figure 5b의 decoded coarse plan trajectory가 target object 방향을 가리킴. Figure 5a의 attention map이 plan 사용 시 target region에 더 집중 |
| 4 | Joint tokenizer 학습이 어려울 텐데 mode collapse 위험은? | 두 mode를 equal sampling으로 학습, mode-conditioning dim 64로 명시적 분리. Codebook entry 4096 × 3 residual의 대용량으로 collapse 방지 |
| 5 | Plan token이 wrong일 때 execution이 그대로 따르면 위험하지 않나? | 정확한 우려. 현재는 plan verification 없음. Limitation에서 저자도 "more adaptive action-space reasoning"이 future direction이라 명시 |
| 6 | 왜 K=20으로 고정? 더 많은 plan step이 도움 안 되나? | Table 3에서 H_p만 변경, K는 고정. K sensitivity 자체는 ablation 안 되어 있음. Plan을 20 step coarse로 보는 게 적절한지 논거 부족 |
| 7 | Inference latency가 얼마나 늘어나나? | **논문에 미공개**. Plan token 42개 + execution token 42개 = 84개 autoregressive step. π0-FAST 대비 약 2배 sequence length일 것 |
| 8 | LIBERO에서 Object suite 100%는 saturation 신호 아닌가? | 맞음. Spatial/Object는 이미 OpenVLA-OFT가 97%+ 도달. C2C의 진짜 가치는 Long suite (94.5→95.0)와 SimplerEnv (62.5→83.3), 그리고 real-world multi-stage에서 드러남 |
| 9 | 50 demo/task로 real-world에서 62.5%면 절대 성능이 낮지 않나? | 같은 50 demo budget에서 π0이 50%, faster 38%. **상대적 향상**이 핵심. Long-horizon에서 plan 가치를 보려는 의도된 setup |
| 10 | 코드/모델 공개 계획은? | 논문에 code URL 없음. 재현성 큰 우려. arXiv preprint 단계라 CoRL/RSS 등 publication 시점에 공개 가능성 |

<!-- VERIFIED: pdf -->
