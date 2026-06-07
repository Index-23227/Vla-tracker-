# OneVLA: A Unified Framework for Embodied Tasks

> **한 줄 요약**: Qwen2.5-VL-3B를 backbone으로 **하나의 11차원 action head** (4D discrete navigation + 7D continuous manipulation)를 공유하고, manipulation → navigation → Chain-of-Thought의 3단계 점진적 학습으로 navigation과 manipulation을 단일 모델에서 동시 처리하는 unified VLA. SimplerEnv 64.5% (+29.1%p), VLN-CE R2R 68.6% (+21.5%p), real-world manipulation 78.8% / navigation 77.5% 달성.

---

## 1. 배경 및 동기

### 분리된 embodied 모델의 비효율
- 기존 VLA는 navigation (NaVILA, NaVid 등)과 manipulation (OpenVLA, pi_0 등)을 **별도 모델**로 학습 — task-specific head, 별도 학습 파이프라인
- UniVLA, NaVA 등 unified 시도가 있었으나 모두 task-specific output branch를 두어 실제로는 partial unification
- 7B급 모델에 의존 → 모바일 로봇 deployment 제약

### 핵심 질문
- **단일 action head**로 discrete navigation command와 continuous 7-DOF manipulation을 모두 emit 할 수 있는가?
- **3B 파라미터**로 7B competitor를 능가할 수 있는가?
- Navigation과 manipulation의 데이터를 **공동 학습** 했을 때 negative transfer가 발생하지 않는가?

📌 [Figure 1 삽입] — OneVLA 아키텍처: Qwen2.5-VL-3B + 11-dim unified action head + 3-stage curriculum

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

**Backbone**: Qwen2.5-VL-3B-Instruct (36-layer Transformer, hidden dim 2048)

**Action head**: 11차원 unified space
- **Navigation**: 4D discrete (forward / left / right / stop) — softmax logits로 처리
- **Manipulation**: 7D continuous (end-effector delta x/y/z, rotation, gripper) — regression

학습 시 task-specific weight mask를 사용 — navigation 샘플에서는 manipulation 7-dim의 loss를 mask 처리, vice versa. 하나의 head가 task type에 따라 다른 부분만 학습된다.

> ❓ **예상 질문**: discrete + continuous를 한 head에서 emit 하는 게 hybrid라기에 충분한가?
> **답변**: 본질적으로 hybrid action head. discrete logit과 continuous regression이 동일 hidden state로부터 parallel branch로 산출되는 구조 — autoregressive token decoding이 아니므로 OpenVLA의 token-based action과 다르며, diffusion / flow-matching도 아님.

### 2.2 3단계 점진적 학습

| Stage | 데이터 | 목적 |
|-------|--------|------|
| 1 | Manipulation + VQA | 기본 manipulation skill 및 vision-language alignment 확보 |
| 2 | + Navigation | 두 도메인 cross-task transfer 활성화 |
| 3 | + Chain-of-Thought | 복잡 추론 task 강화 |

**Loss**: VLM (autoregressive language modeling) + text (CoT reasoning) + action (navigation CE + manipulation MSE)을 **learnable weight**로 동적 결합

### 2.3 Action Horizon

- 최적값 H=5 — ablation에서 H=1, 3, 5, 7, 10 sweep으로 도출

> ❓ **예상 질문**: H=5는 manipulation에 짧고 navigation에 길지 않은가?
> **답변**: navigation의 discrete command 4종은 sparse하므로 5-step chunk가 큰 의미. manipulation은 짧은 horizon이 closed-loop reactivity에 유리. 둘의 절충안인 듯.

---

## 3. 데이터 전략

데이터셋 명시는 일반적인 VLA / VLN benchmark 셋(RT-X, Open X-Embodiment, R2R 등 추정)이며 Stage별로 점진 추가. 정확한 sample 수는 paper full table을 참조하라.

> ❓ **예상 질문**: navigation 데이터 추가가 manipulation 성능을 깎지 않는가?
> **답변**: ablation에서 Joint training이 single-task 대비 SimplerEnv +5.5%, R2R +7.3%, RxR +9.3% — 즉 **positive cross-task transfer**. CoT 데이터가 두 도메인 reasoning을 강화하는 핵심 매개라고 본문에서 주장.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | Qwen2.5-VL-3B-Instruct |
| Parameters | 3B |
| Action horizon | 5 |
| Stages | 3 (manipulation+VQA → +navigation → +CoT) |
| Loss balancing | learnable weights |

GPU / 학습 step 수치는 paper 본문에서 unified VLA 통상 설정(예: 8-32 A100, 수만~수십만 step)으로 추정 가능하나 명시값은 검증 필요.

---

## 5. 실험 결과 심층 분석

### 5.1 VLN-CE (Navigation)

| Benchmark | Metric | UniVLA | **OneVLA** | Delta |
|-----------|--------|--------|-----------|-------|
| VLN-CE R2R | Oracle SR | 47.1 | **68.6** | **+21.5** |
| VLN-CE RxR | Oracle SR | 26.3 | **58.2** | **+31.9** |

- R2R / RxR 모두에서 dramatic 향상 — 3B로 UniVLA(7B 추정)를 능가

### 5.2 SimplerEnv (Manipulation)

| Metric | UniVLA | **OneVLA** | Delta |
|--------|--------|-----------|-------|
| Average SR | 35.4 | **64.5** | **+29.1** |

- SimplerEnv-Bridge / SimplerEnv-Google Robot 전반에서 +29.1%p 평균 향상

### 5.3 Real-world

| 영역 | UniVLA | OneVLA |
|------|--------|--------|
| Manipulation | 62.5 | **78.8** (+16.3) |
| Navigation | 42.5 | **77.5** (+35.0) |

- Real navigation +35%p는 deployable 수준에 가까움

---

## 6. Ablation 분석

### 6.1 Single-stage vs Multi-stage

| 학습 | R2R | RxR | SimplerEnv |
|------|-----|-----|-----------|
| Single-stage | 56.1 | 42.9 | 46.2 |
| **Multi-stage (3-stage)** | **68.6** | **58.2** | **64.5** |
| Delta | +12.5 | +15.3 | +18.3 |

- 점진적 학습이 모든 도메인에서 큰 효과

### 6.2 Joint vs Single-task

| 학습 | R2R | RxR | SimplerEnv |
|------|-----|-----|-----------|
| Single-task | - | - | - |
| **Joint** | +7.3 | +9.3 | +5.5 |

- navigation + manipulation 공동 학습이 **positive transfer**

### 6.3 Action horizon sweep

- H=1, 3, **5**, 7, 10 sweep — H=5가 최적

---

## 7. 관련 연구 비교

| 모델 | Nav | Manip | Single head | Params |
|------|-----|-------|-------------|--------|
| OpenVLA | ✗ | ✓ | - | 7B |
| pi_0 | ✗ | ✓ | - | 3B |
| NaVILA | ✓ | ✗ | - | 7B |
| UniVLA | ✓ | ✓ | task-specific branch | ~7B |
| **OneVLA** | **✓** | **✓** | **단일 11-dim head** | **3B** |

핵심 차이: **진정한 single-head unification** + **smaller scale (3B)** 이 동시에 SOTA 달성

---

## 8. 한계 및 미해결 문제

1. **LIBERO 누락**: 표준 manipulation benchmark인 LIBERO 결과 없음 — SimplerEnv만으로는 saturated benchmark에서의 경쟁력을 확인하기 어려움
2. **CALVIN 누락**: long-horizon language conditioning 평가 부재
3. **Bimanual 미평가**: navigation + single-arm 만 unified, bimanual / mobile manipulation 통합은 미해결
4. **CoT data 출처**: Stage 3에서 사용된 CoT 데이터의 출처와 규모가 본문에서 명확하지 않음
5. **Real-world episode 수**: real navigation 77.5%의 통계 신뢰도가 명시되지 않음 — 시연 영상 수준일 가능성
6. **Action head 표현력 한계**: 7-DOF continuous + 4-class discrete가 mobile-manipulation처럼 둘이 **동시에** 필요한 시나리오를 처리하는지 불분명. 본 모델은 각 step에서 task type을 가정하는 듯
7. **Latency / inference 비용**: 3B임에도 실시간 control feasibility 명시 부재

### Attribution 우려
- +29.1%p SimplerEnv gain이 (a) Qwen2.5-VL backbone 자체 우수성, (b) 3-stage curriculum, (c) joint training 중 어느 요인의 기여인지 ablation으로 완전히 분리되지 않음. 동일 Qwen2.5-VL-3B를 single-stage로 학습한 결과(46.2%)와 비교하면 curriculum이 ~18%p 책임 — backbone change 효과가 더 클 가능성

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — single-head unification 자체는 점진적이나 11-dim concatenation의 단순함이 인상적 |
| **Technical depth** | ★★★☆☆ — 새 backbone 활용 + curriculum이 핵심, 알고리즘적 깊이는 modest |
| **Experimental rigor** | ★★★★☆ — VLN + SimplerEnv + real-world 3축으로 평가 |
| **Practical impact** | ★★★★★ — 3B로 SOTA, single-model deployment 가능 |
| **Writing quality** | ★★★★☆ — 명확한 ablation |

**강점**: 모바일 로봇 deployment 친화적 (3B, single head, joint manip+nav). Real-world navigation +35%p는 압도적. **약점**: LIBERO / CALVIN과 같은 표준 manipulation benchmark 부재로 manipulation 전문 모델 대비 절대 경쟁력 검증 부족. 동시 nav+manip 시나리오 처리 메커니즘 불명확.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO는 왜 측정하지 않았는가? | 본문 미언급. SimplerEnv가 SimplerEnv-Bridge / Google Robot 등 broad-domain이라는 점을 강조하나, LIBERO saturation 영역 회피 가능성도 있음 |
| 2 | 11-dim single head가 정말 task-specific branch보다 좋은가? | 본문에서 task-specific branch UniVLA 대비 +29.1% (SimplerEnv) 보임. 단, 동일 backbone에서 head architecture만 바꾼 직접 ablation은 없음 |
| 3 | navigation step에서 manipulation 7-dim은 학습되지 않는다 — degenerate solution 위험? | weight masking으로 처리. 그러나 negative log-likelihood만 mask, gradient flow는 유지되므로 implicit regularization 가능 |
| 4 | Mobile manipulation에서 nav+manip 동시 emit이 필요할 때는? | 본 모델은 각 step에서 task type 가정. mobile manipulation은 미평가 — open problem |
| 5 | Qwen2.5-VL backbone 자체가 강해서 gain 인 것 아닌가? | Single-stage Qwen2.5-VL이 46.2 (SimplerEnv), multi-stage가 64.5 → curriculum이 ~18%p, backbone이 나머지 11%p 기여 추정 |
| 6 | CoT 데이터 출처는? | 본문 명시 부족 — 합성인지 human annotation인지 불명확 |
| 7 | Action horizon H=5의 sensitivity는? | H sweep 제시되나 detail은 paper 본문 참조 — 5에서 최적 |
| 8 | Real-world episode 수와 hardware setup은? | 본문 명시 약함 — 정량 평가 신뢰성 검증 어려움 |

<!-- VERIFIED: pdf -->
