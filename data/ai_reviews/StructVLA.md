# StructVLA: Beyond Dense Futures — World Models as Structured Planners for Robotic Manipulation

> **한 줄 요약**: Dense future prediction이 야기하는 plan drift와 semantic subgoal의 geometric grounding 부족 문제를 해결하기 위해, **gripper state transition + kinematic turning point** 기반의 sparse structured frame을 unified discrete token vocabulary로 예측·실행하는 두-단계 VLA. SimplerEnv-WidowX **75.0%** (Tab. 1) / LIBERO **94.8%** (Tab. 2) 달성.

---

## 1. 배경 및 동기

### 기존 generative VLA의 두 갈래와 그 한계
- **Dense future prediction** (DreamVLA, VideoVLA, FlowVLA, Cosmos Policy): step-wise pixel/latent rollout이 시각적 잉여(visual redundancy)와 error accumulation을 야기 → 장-horizon에서 plan drift (Fig. 1, 본문 §1).
- **Sparse semantic / latent foresight** (BagelVLA, ForeAct, LDP, Villa-X): 추상 subtask 또는 implicit latent로 horizon을 압축하나, **explicit kinematic grounding이 결여**되어 closed-loop 정밀 제어와의 정합성이 떨어진다.

### 핵심 질문
- "Dense는 잉여, semantic은 grounding 부족"이라는 딜레마를 우회하기 위해, **kinematic event에서 자동 추출된 sparse "structured frame"** 만으로 world model을 재구성할 수 있는가? 그리고 이 representation이 다음 단계에서 그대로 action policy로 전이될 수 있는가? (Fig. 1, 본문 §1, contributions)

---

## 2. 방법론 심층 분석 (§3)

### 2.1 Physics-Grounded Data Curation (§3.1)
오프라인 자동 파이프라인으로 두 종류의 structured frame을 추출한다.
- **Gripper State Transitions** (Eq. 1): flip 시점 `t_c` 이후 stabilization window `δ_settle`을 두고, EMA-smoothed end-effector 속도가 `τ_high`를 다시 넘는 직전의 quasi-static observation `o_{t*}`을 선택. 신호 jitter와 actuation 지연을 동시에 보정.
- **Kinematic Turning Points** (Eq. 2): 길이 `W`의 sliding window 내에서 EMA 속도가 `τ_low` 이하로 유지되는 시점을 추출. 횡 정렬 후 하강 직전 같은 motion-regime 전환을 포착.
- **Temporal Coverage / Gap Filling**: 길게 정적인 구간엔 ΔL2 action change가 가장 작은 frame을 추가 샘플링해 sparse 영역을 보강.
- 후처리로 **GPT-5.2 기반 offline VLM filtering**을 적용해 노이즈/중복 후보를 제거(고정 JSON schema로 출력 제한). **추론 시에는 VLM이 전혀 사용되지 않는다** (§3.1 결미).

### 2.2 Two-Stage Training with Unified Discrete Vocabulary (Fig. 2)
- **Stage 1 — Structured Planner**: Emu3 VQ tokenizer로 instruction·visual context·target frame을 동일 vocabulary에 사상 후, autoregressive cross-entropy로 다음 structured frame `T_s = {z_1,…,z_K}` 예측 (Eq. 3). 짧은 milestone supervision의 sparseness를 보완하기 위해 sliding-window augmentation으로 timestep을 살짝 이동시켜 학습.
- **Stage 2 — Plan-to-Act Transfer**: 동일 backbone의 예측 target만 action token(FAST tokenizer, vocab 1024)로 교체. instruction + interleaved (obs, action) history `H_t`로 conditioning하여 action token만 cross-entropy로 학습 (Eq. 4). vision/action/language가 **하나의 discrete vocabulary** 위에 살기 때문에 planning ↔ control 사이의 representation gap이 제거된다.

### 2.3 Implementation (§4.2)
- 8.5B autoregressive transformer, **UniVLA world model 가중치로 초기화**, Emu3 VQ encoder (spatial compression 8), action은 1·99 percentile 정규화 후 FAST 1024 token, language vocab 끝의 1024 ID에 매핑.
- LoRA rank=32로 PEFT.
- 학습 step: SimplerEnv (Stage 1: 30k / Stage 2: 20k @ batch 128), LIBERO (10k / 4k @ 196), real-world (2k / 2k @ 32). 시점별 view 구성도 도메인마다 다르다(LIBERO Stage 2는 third-person + wrist 추가).

> ❓ **예상 질문**: 가변 개수의 structured frame을 어떻게 다루는가?
> **답변(§3.2)**: target은 "다음 1개 structured frame"의 token sequence이며, sliding-window augmentation으로 supervision을 densify한다. 즉 모델은 가변 길이 sequence를 예측하지 않고 "다음 milestone frame" 단일 예측을 반복한다.

---

## 3. 실험 결과

### 3.1 SimplerEnv-WidowX (Tab. 1)
- StructVLA 평균 **75.0%** (Spoon 87.5 / Carrot 75.0 / Stack 45.8 / Eggplant 91.7).
- 직전 최강 UniVLA(69.8%) 대비 **+5.2%p**, mimic-video(56.3%)·VideoVLA(53.1%) 대비 큰 폭 우위. Stack Green-on-Yellow에서 VideoVLA(45.8%)와 동률로 가장 어려운 task 유지.

### 3.2 LIBERO (Tab. 2)
- 평균 **94.8%**: SPATIAL 95.4 / OBJECT 98.0 / GOAL 93.4 / **LONG 92.2**. 특히 LONG suite에서 UniVLA*(89.6%), DreamVLA(89.5%)를 뚜렷이 상회 — sparse milestone foresight가 long-horizon drift 억제에 기여한다는 본 논문의 가설을 직접 뒷받침.

### 3.3 Real-world (Fig. 5, §4.3)
- 4-task pick-and-place 평균 **87.5%** (UniVLA 대비 +20.0%p), Eggplant 10/10.
- Long-horizon Tidy-Up: **8/10**(StructVLA) vs UniVLA 4/10, SpatialVLA 2/10 (Fig. 5 우측).
- Zero-shot OOD pick-and-place(미경험 색·형상): 80.0% vs UniVLA 30.0% (Fig. 7). Human intervention 시 80.0% vs 40.0%.

### 3.4 분석 실험
- **Training efficiency (Tab. 3)**: 동일 batch 128에서 RoboVLMs 50k / UniVLA 12k / StructVLA **10k** step로 peak 도달 — structured plan이 search space를 축소.
- **Grasp vs Task (Fig. 6)**: 평균 grasp 89.6%로 최고이며, grasp→task 전환률에서 UniVLA보다 우위(예: Stack에서 grasp 95.8% 동률이지만 task 45.8% vs 33.3%).
- **History context (Tab. 4)**: history H=2→4로 확장 시 StructVLA는 70.8→**75.0%**, 반대로 UniVLA는 69.8→66.7%로 하락. 구조화된 planner가 긴 시간 맥락을 의미 있게 활용한다는 증거.
- **Mechanistic insight (Fig. 4a)**: kinematic transition 시 attention이 gripper-object 접촉 영역에 집중, 베이스라인은 분산.

---

## 4. 한계 및 미해결 문제

1. **실험 규모 한계**: 저자도 §5 Limitations에서 명시 — 다중 도메인 대규모 데이터셋 검증이 부족.
2. **Online RL 부재**: 본 모델은 imitation 기반이며, 저자 future work로 RL 통합 언급.
3. **Adaptive threshold(`τ_high`, `τ_low`, `δ_settle`, `W`)**의 cross-task 일반화는 정량 ablation 없음 — task마다 episode 분포로 자동 산출하나 robustness 분석 미공개.
4. **VLM filtering의 영향**: GPT-5.2 사후 필터가 frame 품질에 얼마나 기여하는지 ablation 미보고.
5. 액션 추론은 autoregressive하므로, FAST decoder hz는 명시되지 않음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Dense vs semantic 사이의 "physically grounded sparse milestone" 라는 중간 지점을 새 axis로 정립 |
| **Technical depth** | ★★★★☆ — 수식 명시(Eq.1–4), 두 단계 정렬, unified token vocabulary가 일관성 있음 |
| **Experimental rigor** | ★★★★☆ — LIBERO·SimplerEnv·real-world에 history/efficiency/grasp 세부 ablation까지 |
| **Practical impact** | ★★★★☆ — UniVLA 가중치 재사용 + LoRA로 비용 합리적, 실 환경 +20%p |

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | DreamVLA(dense world knowledge)와 본질적 차이? | DreamVLA는 dense forecasting을 guide로 활용; StructVLA는 kinematic event에 anchored sparse frame만 예측 → drift 누적 회피 (Tab. 2 LONG: 92.2 vs 89.5) |
| 2 | Gripper 동작이 거의 없는 push/wipe task는? | Kinematic turning point + gap-filling으로 커버 가능. 단, 본문 evaluation은 pick-place 중심이라 일반화는 미검증 |
| 3 | UniVLA 초기화의 기여도는? | §4.2에서 명시적 초기화. Tab. 3에서 동일 RoboVLMs 대비 빠른 수렴은 priors와 sparse target 공동 효과로 보임. 별도 from-scratch ablation은 없음 |
| 4 | Stage 2에서 Stage 1 representation이 실제로 보존되는지? | Fig. 4b OOD 시각화: Stage 1 외 scene/task에도 coherent 예측 → world-model prior 유지 |

<!-- VERIFIED: pdf -->
