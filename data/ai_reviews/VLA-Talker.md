# VLA-Talker (In-Context VLA): Endowing Vision-Language-Action Models with Language via In-Context Post-Training and Agentic Tool Use

> **한 줄 요약**: "VLA에 필요한 것은 언어를 *생성*하는 능력이 아니라 grounded 언어를 *소비*하는 능력"이라는 관점 전환 아래, 외부 agentic tool loop(GroundingDINO + DepthAnything + gripper projection + Qwen2.5-VL fallback)가 획득한 공간 evidence를 `<spatial>` 태그로 프롬프트에 주입하고 **action token에만 loss를 거는** in-context post-training + GRPO로 OpenVLA-OFT를 90.4 → LIBERO 97.4%, SimplerEnv 72.4%, RoboCasa-GR1 59.5%까지 끌어올린 연구.

- arXiv: 2608.05738 (2026-08-06) · AAAI 2027 형식 · 코드 미공개
- 저자: Jiarui Yang, Wen Huang, Jiale Zhang, Maowei Hu, Hang Guo (Nankai / Tsinghua / EPFL)

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 거의 모든 VLA는 **behavior cloning**으로 학습된다: 정적 관측 `o_t`와 고정 instruction `ℓ`가 주어지면 expert action chunk `a_{t:t+H}`를 모사(Eq. 1).
- 이로부터 두 가지 약점이 파생된다.
  1. **instruction이 불투명한 conditioning string**으로 취급됨 → paraphrase하거나 미학습 동의어로 물체를 지칭하면 성공률 급락.
  2. **관측이 단일 feed-forward로 소비**됨 → 정책이 부족한 정보를 능동적으로 획득할 수 없고, "한 번만 보면 알 수 있는" 위치도 추측해야 함.
- 자연스러운 처방인 textual chain-of-thought(CoT)는 저자들의 주장에 따르면 **오히려 해롭다**.

### 저자가 제시하는 generative CoT의 3대 실패 요인
| # | 실패 | 설명 |
|---|------|------|
| 1 | **Grounding gap** | rationale이 action head가 이미 보는 동일 feature에서 생성되므로 새 정보량이 0. 위치를 hallucinate하면 오히려 오도. |
| 2 | **Objective interference** | 하나의 autoregressive loss 안에서 language token이 action token보다 수십~수백 배 많아 gradient mass가 "그럴듯하게 말하기"에 지배됨 → 정책이 **actor가 아니라 narrator**가 됨. |
| 3 | **Latency & drift** | 결정마다 수백 토큰 생성 → closed-loop timing 붕괴, prefix의 초기 오류가 action suffix로 전파. |

### 핵심 질문
- **evidence 획득(acquiring)과 evidence 사용(using)을 분리하면 CoT의 이득만 취하고 비용은 피할 수 있는가?**
- **surface form을 다양화하면 정책이 spatial language를 암기 대신 해석하게 되는가?**

📌 [Figure 1 삽입] — CoT(N개 reasoning token 자기회귀 생성) vs ICL/tool-use(단일 pass 주입) 비교. SR은 0.97 vs 0.88, cost는 1.00 vs 0.22 수준.

---

## 2. 방법론 심층 분석

### 2.1 전체 구조
VLA-Talker는 **backbone과 action head를 전혀 바꾸지 않는다**. 추가되는 것은 두 가지뿐이다.

1. **Agentic tool-use loop** — keyframe마다 grounded 공간 evidence 획득
2. **In-context post-training** — 그 evidence를 read-only context로 주입하고 **action만 supervise**

학습 목표는 Eq.(1)의 관측을 evidence context `c_t`로 확장한 것에 불과하다:

```
L(θ) = - Σ_{k=0}^{H-1} log π_θ( a_{t+k} | o_t, s_t, ℓ, c_t, a_{t:t+k} )
```

`c_t`의 **어떤 토큰도 loss에 포함되지 않는다**. 이것이 논문 전체의 핵심 한 줄이다.

### 2.2 Agentic tool cascade (evidence 획득)
| 도구 | 구현 | 역할 |
|------|------|------|
| Open-vocab detector | GroundingDINO | instruction에서 파싱한 대상/목적지 물체의 픽셀 centroid `(u_o, v_o)` |
| Monocular depth | DepthAnything | `[0,1]` 정규화 depth map (1=near) → 상대 높이 순서 추론, metric calibration 불필요 |
| Gripper projection | 해석적(학습 없음) | proprioception `s_t[0:3]`를 카메라 intrinsics/extrinsics로 이미지 공간에 투영, renderer의 vertical flip 보정 → **정확한** gripper 픽셀 |
| VLM fallback | Qwen2.5-VL-7B | detector가 abstain/저confidence일 때 근사 좌표 또는 정성적 공간 서술 반환 |

출력은 evidence tuple `c_raw_t = ((u_g,v_g,d_g,grip_state), {(u_o,v_o,d_o)}, relations)`.

> ❓ **예상 질문**: gripper를 왜 detector로 찾지 않는가?
> **답변**: end-effector world 좌표가 proprioception으로 이미 알려져 있으므로 투영만 하면 sub-pixel 오차로 정확하다. 학습도, 추론 비용도 0. "굳이 배울 필요 없는 것은 배우지 않는다"는 설계 원칙.

### 2.3 Caption rendering — diversity가 본체
단일 템플릿(`gripper: (x=U, y=V, depth=D); mug: ...`)으로 학습하면 정책은 **surface string을 pattern-match**할 뿐이고 test 시 phrasing이 조금만 달라도 붕괴한다. VLA-Talker의 data engine은 같은 tuple을 reference modality × lexical paraphrase × depth verbalization 축으로 조합해 **tuple당 약 24가지 실현형**을 만든다.

```
# Realization A (좌표 중심, detector 성공)
<spatial>The gripper is at (128,96), depth 0.71; the open white mug is at (170,140),
depth 0.79. The mug is 42 px right and 44 px below the gripper, +0.08 depth difference.</spatial>

# Realization B (paraphrase, VLM fallback)
<spatial>My open gripper sits upper-center. The white mug lies lower-right, slightly
farther on the table, and below the gripper's current height.</spatial>
```

### 2.4 In-context post-training과 keyframe gating
- 학습 시퀀스: `[ℓ, <image>, c_t, a_{t:t+H}]`, supervision mask는 **action token(및 직전 separator)만** 활성.
- Keyframe gate: 초기 프레임 + gripper open/close 상태 변화 + 주기적 progress check에만 주입, 나머지는 `c_t = ∅` → 그 스텝에서는 평범하고 빠른 vanilla VLA로 동작. 추론에서도 동일 스케줄이라 amortized overhead가 작다.

### 2.5 Trajectory-level GRPO
- rollout `τ = {T_1, C_1, V_1, ..., T_n, A_n}` (결정 스텝 / tool call / 반환 evidence / 실행 action chunk).
- reward는 **sparse**: `R(τ) = α_s·I_success + α_f·I_format` (format term은 tool-call 문법 파싱 여부). dense reward 없음 → "언제 tool을 부를지"를 정책이 스스로 발견.
- group-relative advantage로 value function 제거, clipped surrogate + reference(in-context) 정책에 대한 KL anchor(β=0.01), group size M=16.
- **importance ratio에는 action token만 기여**하므로 RL이 모델을 rationale generator로 되돌리지 않는다.

---

## 3. 데이터 전략

| 항목 | LIBERO | RoboCasa-GR1 | SimplerEnv |
|------|--------|--------------|------------|
| Training episodes | 1,640 | 3,120 | 960 |
| Annotated keyframes | 9,830 | 27,960 | 5,230 |
| Detector success rate | 91.2% | 83.6% | 89.4% |
| VLM fallback rate | 8.8% | **16.4%** | 10.6% |
| Realizations / tuple | 24 | 24 | 24 |
| Injections / episode | 4.1 | 6.8 | 3.6 |

- RoboCasa-GR1의 fallback rate가 가장 높은 것은 주방 장면의 시각적 다양성/소형 물체/희귀 카테고리 때문이라는 설명이 자연스럽다.
- Injection이 episode당 한 자릿수에 머무는 이유는 gate가 wall-clock이 아니라 **state change**로 트리거되기 때문.

---

## 4. 시스템/학습 세부사항

- **Backbone**: 공개된 OpenVLA-OFT 가중치에서 초기화, parallel decoding + action chunking. 학습·추론 모두 **third-person RGB 1개 + 언어 instruction만** 사용(wrist camera 없음).
- **Optimizer**: AdamW, in-context stage batch 64 / LR 1e-5, RL stage batch 128 / LR 2e-6.
- **비용**(8×A100-80GB, 3개 벤치마크 평균): Gen-CoT SFT 19.5 h(156 GPU-h) vs VLA-Talker in-context SFT 17.1 h(137 GPU-h) + GRPO 8.7 h(70 GPU-h). in-context SFT가 오히려 약간 저렴한데, 저자 구현에서는 evidence token에 대해 LM head backward를 계산할 필요가 없기 때문.
- **추론 효율**(단일 A100, batch 1, LIBERO rollout 평균):

| 정책 | rationale tokens | latency | control |
|------|------------------|---------|---------|
| Action-only (reasoning 없음) | 0 | 73 ms | 13.6 Hz |
| Gen-CoT | ~256 | 359 ms | 2.8 Hz |
| **VLA-Talker** | **0** | **78 ms** | **12.8 Hz** |

→ Gen-CoT 대비 **4.6× 빠르고**, rationale 없는 순수 action policy 대비 7% 이내 오버헤드.

---

## 5. 실험 설계 및 평가 프로토콜

- **LIBERO**: Spatial / Object / Goal / Long 4개 suite, task당 50개 randomized initial condition.
- **RoboCasa-GR1**: GR1 humanoid 주방, 대표 pick-and-place 4개 task를 표로 보고하되 **평균은 24개 task 전체**에 대해 계산(부록 Table 13에 전체 분해).
- **SimplerEnv**: WidowX held-out 4개 task(spoon/carrot/stack/eggplant).
- **핵심 통제군 Gen-CoT**: *동일한* tool loop에서 나온 *동일한* evidence tuple을 텍스트로 생성·supervise하는 변형. VLA-Talker와의 차이는 오직 "생성 vs 주입" 및 "무엇을 supervise하는가"뿐 → 비교가 깔끔하다.
- 3 seed 재학습 + Welch's t-test(p < 0.01) 보고.

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO (Table 1)
| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| VLA-Thinker | 97.7 | 98.5 | 97.5 | **94.4** | 97.0 |
| Gen-CoT (matched) | 97.5 | 98.6 | 97.2 | 91.6 | 96.2 |
| **VLA-Talker** | 98.2 | **99.2** | **98.4** | 93.6 | **97.4** |

- Long suite에서는 VLA-Thinker(94.4)에 뒤진다. 즉 SOTA는 평균 기준이며 **long-horizon에서의 우위는 아니다**.

### 6.2 RoboCasa-GR1 (Table 2 / Table 13)
- 대표 4task: PnP Bottle 76.0 / Can 78.0 / Cup 48.0 / Milk 58.0.
- **24task 전체 평균 59.5%**로 1위 (ABot-M0 58.3, TwinBrainVLA 54.6, LangForce 52.6, PhysBrain 50.0, GR00T N1.5 48.2, Gen-CoT 46.5).
- 가족별로 보면 novel-placement에서 강점이 뚜렷: cuttingboard→* 64.1(1위), plate→* 69.8(1위, 특히 Plate→Plate 79.5). 반면 placemat→*(51.3)와 tray→*(50.5)에서는 ABot-M0(62.1 / 52.1)에 밀린다.

### 6.3 SimplerEnv (Table 5)
| Method | Spoon | Carrot | Stack | Eggplant | Avg |
|--------|-------|--------|-------|----------|-----|
| π₀.₅ / GR00T N1.5 | 49.3 / 64.5 | 64.7 / 65.5 | 44.7 / 5.5 | 69.7 / 93.0 | 57.1 |
| TwinBrainVLA | 87.5 | 58.3 | 33.3 | 79.1 | 64.5 |
| LangForce | 89.6 | 63.8 | 33.3 | 79.2 | 66.5 |
| Gen-CoT | 85.4 | 52.1 | 31.3 | 50.0 | 54.7 |
| **VLA-Talker** | **91.7** | 56.3 | **47.9** | **93.8** | **72.4** |

- Carrot(56.3)은 GR00T N1.5(65.5)·π₀.₅(64.7)보다 낮다. 평균 우위는 Stack/Eggplant에서의 큰 격차가 견인.

### 6.4 데이터 효율 (Table 6 / Fig. 5)
| demos/task | 5 | 10 | 25 | 50 |
|-----------|---|----|----|----|
| BC | 50.6 | 63.1 | 77.4 | 90.4 |
| Gen-CoT | 48.2 | 59.7 | 74.3 | 87.6 |
| **VLA-Talker** | **71.4** | **84.6** | **92.8** | **97.4** |

- **25 demo만으로 BC의 50 demo(90.4)를 상회**(92.8). Gen-CoT는 모든 예산에서 BC보다 낮다 — 생성 objective는 데이터를 더 줘도 도움이 되지 않는다는 강한 증거.

### 6.5 일반화 (Table 7) 및 phrasing 강건성 (Table 11)
| Method | Seen | Unseen obj. | +Distractors |
|--------|------|-------------|--------------|
| BC | 90.4 | 54.8 | 47.6 |
| Gen-CoT | 87.6 | 52.1 | 44.9 |
| **VLA-Talker** | **97.4** | **85.1** | **80.3** |

| Renderer | Original | Paraphrased |
|----------|----------|-------------|
| Single template | 95.8 | 77.2 |
| **Diverse (ours)** | **97.4** | **94.6** |

- 대상 identity를 open-vocab tool이 **지각 단계에서** 해결하므로 novel name/유사 distractor가 정책 혼동으로 이어지지 않는다.
- 템플릿 단일화 시 paraphrase에서 -18.6%p 붕괴 → diversity rendering이 장식이 아니라 핵심 기여임을 입증.

### 6.6 실물 로봇 (AgiBot G1, Table 8)
- JoyAI-RA-0.1 backbone, 8개 데스크탑 subtask, subtask당 20 trial, head + 좌우 wrist 3개 RGB(224×224), bimanual joint target.

| 설정 | Baseline S/M | +CoT S/M | **+In-Context (Ours) S/M** |
|------|--------------|----------|-----------------------------|
| 평균 | 41.9 / 28.1 | 41.9 / 29.4 | **58.1 / 45.0** |

- 단일 task +16.2%p, multi-task +16.9%p. **모든 subtask에서 +CoT를 앞선다.** 격차는 pen/eraser/correction fluid 같은 정밀 소형 물체에서 가장 크고, multi-task regime에서 더 벌어진다.

---

## 7. Ablation 분석

**(a) supervision scheme (Table 3, evidence 고정)**
| Scheme | LIBERO Avg | Rel. latency |
|--------|-----------|--------------|
| Generate + supervise text | 81.5 | 4.6× |
| Inject + supervise text | 89.7 | 1.0× |
| **Inject + action-only (ours)** | **97.4** | 1.0× |

→ 이득의 원천은 evidence 자체가 아니라 **소비와 생성의 분리**. (b)→(c)만으로 +7.7%p.

**(b) 학습 단계 (Table 4)**
| Variant | Spatial | Object | Goal | Long | Avg |
|---------|---------|--------|------|------|-----|
| OpenVLA-OFT backbone | 90.7 | 94.6 | 89.8 | 86.3 | 90.4 |
| SFT-only (in-context) | 97.0 | 98.2 | 96.8 | 90.4 | 95.6 |
| GRPO-only (no SFT) | 89.8 | 88.1 | 86.9 | 86.2 | **87.8** |
| Full (SFT+GRPO) | 98.2 | 99.2 | 98.4 | 93.6 | **97.4** |

→ in-context stage가 주 동력(+5.2%p), GRPO는 +1.8%p. **초기화 없는 GRPO는 backbone보다도 나쁘다**(sparse-reward RL의 전형적 취약성).

**(c) component ablation (Table 12)**: full 97.4 → depth 제거 93.2 / VLM fallback 제거 92.8 / 매 스텝 주입 95.1 / **tool loop 제거(자체 추측) 84.3**. 마지막 항목이 결정적 — 외부에서 획득한 grounded evidence가 없으면 BC 수준으로 회귀.

**(d) tool cascade 강건성 (Table 14)**: detector drop 0/30/60%에서 detector-only 96.4→87.3→70.6, VLM-only 93.5→93.0→92.4, cascade는 near-peak 유지. 신뢰성의 원천은 특정 지각 모듈이 아니라 **agentic routing**.

**(e) keyframe gating (Fig. 6)**: gripper-change 프레임 추가에서 대부분의 이득이 나오고 이후 포화. 선택된 gate는 곡선의 무릎.

**(f) GRPO 하이퍼파라미터 (Fig. 12)**: β ∈ [0.005, 0.02]에서 평탄, 양측에서 열화. M은 16까지 개선 후 포화, GPU-hour는 M에 선형 증가.

**(g) RL 동역학 (Fig. 10)**: success reward 0.80→0.94, **episode당 tool call 3.4→1.8로 감소** — RL이 "필요할 때만 agentic하게" 되는 법을 학습.

---

## 8. 관련 연구 비교

| 축 | 대표 연구 | VLA-Talker의 차별점 |
|----|-----------|---------------------|
| Visual CoT | CoT-VLA (subgoal 이미지 생성) | 중간 표현을 **생성하지 않음**, latency 0 |
| Textual/latent CoT | VLA-Thinker, ACoT, latent reasoning | reasoning을 latent로 옮기는 대신 **외부 도구로 옮김** |
| Tool-use agent | LLaVA-Plus, VisTA, SpaceTools, VLAs-as-Tools | 도구를 high-level planner가 아니라 **저수준 policy의 context 공급원**으로 사용 |
| Memory | MemoryVLA, RoboMemory | 시간적 메모리가 아니라 **프레임별 공간 evidence** |
| In-context learning | Sirko-Galouchenko et al. 2025 | in-context를 **post-training + action-only mask**로 정식화 |

가장 가까운 통제 비교는 논문 자체의 Gen-CoT이며, 이것이 이 논문의 최대 강점이다(동일 evidence, 동일 backbone, 동일 데이터).

---

## 9. 한계 및 미해결 문제

1. **시뮬레이터 특권 정보 의존**: gripper 투영에 "per-task camera intrinsics/extrinsics extracted from the simulator"를 사용한다. 실물에서는 calibration 품질이 곧 evidence 품질이 되며, 이는 논문이 정면으로 다루지 않는 배포 리스크다.
2. **코드 미공개**, α_s/α_f 값, M 외 rollout 세부, 데이터 엔진 전체 템플릿은 부록에 축약 제시.
3. **LIBERO-Long에서 VLA-Thinker에 뒤짐**(93.6 vs 94.4) — long-horizon에서는 evidence 주입만으로 부족.
4. **잔여 오류의 지배적 원인은 control precision**(Table 15). 접촉이 많은 정밀 삽입에서 실패(부록 Fig. 9의 pencil case 사례: grounding은 정확하나 shelf lip에 걸려 timeout). 즉 이 방법의 상한은 저수준 컨트롤러가 결정한다.
5. **파이프라인 복잡도**: detector + depth + VLM + 투영 + gate + 2단계 학습. Table 12 기준 어느 하나만 빠져도 3~4%p씩 손실이라 운영 부담이 실재한다.
6. **7B backbone 1종에서만 검증**: π₀ 계열 flow-matching head나 소형(2~3B) backbone에서도 동일 효과인지 미검증.
7. **RoboCasa 절대 성능 59.5%**는 여전히 배포 수준과 거리가 멀다.

---

## 10. 총평

**강점**: "무엇을 supervise하는가"라는 단일 변수를 통제한 Gen-CoT 비교 설계가 매우 깔끔하고, 결론(81.5 → 89.7 → 97.4)이 그 설계로부터 직접 도출된다. backbone/action head를 건드리지 않아 **어떤 VLA에도 얹을 수 있는 레시피**라는 점, 정확도를 올리면서 latency는 rationale-free policy의 7% 이내로 유지한 점, 25 demo로 BC의 50 demo를 이기는 데이터 효율, paraphrase 강건성 94.6%는 모두 설득력 있다. 실물 AgiBot G1 8task에서 +16%p는 시뮬레이션 결과의 신뢰도를 크게 높인다.

**약점**: 시뮬레이터 카메라 파라미터 의존, 코드 미공개, backbone 다양성 부재, 그리고 이득의 상당 부분이 **외부 지각 모듈의 성능**에서 온다는 점(tool loop 제거 시 84.3). 후자는 "VLA가 좋아진 것인가, 지각 파이프라인을 붙인 것인가"라는 공정성 논쟁을 부른다 — 다만 저자는 Gen-CoT에 동일 evidence를 주는 방식으로 이 반론을 상당 부분 선제 차단했다.

**한 문장 평가**: reasoning을 모델 안에서 생성하지 말고 밖에서 사올 것 — 그리고 사온 것을 loss에 넣지 말 것. 단순하지만 실증적으로 강력한 처방.

---

## 11. 재현/확장을 위한 체크리스트

- [ ] gripper 투영을 시뮬레이터 특권 파라미터 없이(hand-eye calibration만으로) 수행했을 때의 성능 저하 측정
- [ ] π₀/π₀.₅ 같은 flow-matching action head에 동일 in-context mask 적용
- [ ] evidence를 텍스트가 아닌 임베딩(좌표 토큰)으로 주입했을 때와의 비교 — 언어 형태가 정말 필요한가?
- [ ] detector/depth를 더 약한 모델로 교체한 cost-performance 곡선
- [ ] LIBERO-Long 격차를 메우기 위한 sub-goal 수준 evidence(현재는 프레임 단위 공간 관계만)

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | GroundingDINO + DepthAnything + Qwen2.5-VL-7B를 붙였는데, 이게 VLA의 개선인가 지각 파이프라인의 개선인가? | 저자의 방어는 Gen-CoT baseline — **동일한 evidence**를 받고도 96.2/54.7/46.5에 그친다. 즉 evidence 자체가 아니라 소비 방식이 차이. 다만 tool loop 제거 시 84.3이므로 외부 지각의 기여가 큰 것도 사실. |
| 2 | gripper 투영에 시뮬레이터 intrinsics/extrinsics를 쓰는 것은 privileged information 아닌가? | 실물 AgiBot G1에서도 동일 방식으로 작동(캘리브레이션 사용)했다는 것이 반론. 그러나 calibration drift에 대한 정량 분석은 없다 — 가장 아픈 질문. |
| 3 | 추론 시 detector/depth/VLM 호출 비용은 78 ms에 포함되는가? | Table 10은 per-decision latency이고 keyframe에만 tool이 돌아 amortize된다(episode당 3.6~6.8회). 그러나 VLM fallback(7B)이 호출되는 순간의 tail latency는 보고되지 않음. |
| 4 | 24가지 realization은 왜 24인가? | 포화 곡선에서 saturated accuracy의 1점 이내에 드는 최소 pool 크기(부록 I). 축(modality × lexical × depth verbalization) 조합의 산물. |
| 5 | GRPO가 +1.8%p뿐인데 70 GPU-h를 쓸 가치가 있나? | 정확도보다 **tool call 3.4→1.8 감소**(추론 비용 절감)가 실질 이득. 순수 정확도만 보면 SFT-only 95.6도 충분히 경쟁력 있음. |
| 6 | Gen-CoT가 BC보다도 낮은데(87.6 vs 90.4 @50 demos) baseline을 불리하게 세팅한 것 아닌가? | 저자 주장은 objective interference의 직접 증거. 다만 rationale loss weight를 낮춘 변형(예: 0.1×)에 대한 sweep이 없어 "약화된 CoT" 대조군이 빠져 있다. |
| 7 | `<spatial>` 토큰을 loss에서 빼면 모델이 그것을 무시하도록 학습되지 않는가? | 무시하면 action 예측이 나빠지므로 attention은 유지된다. 실제로 depth 채널 제거만으로 -4.2%p가 나온다는 것이 정책이 evidence를 실제로 읽고 있다는 증거. |
| 8 | LIBERO-Long에서 VLA-Thinker에 지는 이유는? | 주입 evidence가 **프레임 단위 공간 관계**라 sub-task 분해나 진행 상태를 담지 않는다. Long-horizon에는 시간축 evidence가 필요. |
| 9 | RoboCasa placemat/tray family에서 ABot-M0에 밀리는 것은 어떻게 설명하나? | placemat/tray는 source surface가 평면·저대비라 detector centroid 오차가 커지고 depth 대비가 작다. Table 16의 RoboCasa fallback rate 16.4%와 정합적. |
| 10 | 3 seed, Welch's t-test p<0.01은 충분한 통계인가? | 3 seed는 최소 수준이지만 suite별 error bar가 BC>Gen-CoT>VLA-Talker 순으로 줄어드는 일관된 패턴은 신뢰할 만하다. 실물 20 trial/subtask는 CI가 ±20%p 수준이라 subtask별 비교는 조심해야 함. |
| 11 | 이 레시피를 π₀ 같은 flow-matching 정책에 그대로 얹을 수 있나? | 원리상 가능 — mask는 language head에만 관여하고 action head는 불변. 미검증이며 가장 가치 있는 후속 실험. |
| 12 | 실물 multi-task 45.0%는 배포 가능한 수준인가? | 아니다. 다만 동일 backbone·데이터의 Baseline 28.1% 대비 상대 개선 60%는 방법론적 기여로 충분하다는 것이 저자 입장. |

<!-- VERIFIED: pdf -->
