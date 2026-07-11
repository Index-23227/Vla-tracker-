# Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents

> **한 줄 요약**: 사전학습된 VLA를 **일절 finetuning 하지 않고 얼린 채**, LLM 코딩 에이전트(Codex / Claude Code)가 고정된 소규모 primitive 라이브러리(analytic 6종 + 학습형 VLA_ACT 1종)를 JSON 호출로 조합(orchestration)하는 memory-augmented agentic harness. VLA는 contact-rich 국소 구간에서만 retryable primitive로 호출되고, 의미적 re-grounding·staging·transport·release는 planner가 담당. LIBERO-Pro에서 최강 기존 baseline(RATS 43.8%) 대비 **+38.6%p**(CC 82.4%), RoboCasa365에서 RLDX-1 대비 **+25.4%p**(Codex 55.4%), RoboTwin C2R zero-shot에서 **58.4%**(frozen LingBot-VLA 단독 50.4% 대비 +8.0%p) 달성. 표준 LIBERO도 96.0%로 base 정책(95.3%) 수준을 보존.

---

## 1. 배경 및 동기

### 두 패러다임의 비대칭적 실패 모드

- **End-to-end VLA** (RT-2, OpenVLA, π0/π0.5, GR00T 등): irregular grasping, 좁은 공차 placement, fixture actuation 같은 **국소 contact-rich visuomotor 제어**에는 강하지만, 학습 trajectory 분포 밖의 deployment perturbation — semantic retargeting(지시 대상 변경), goal re-binding, spatial-layout shift, 짧은 skill의 장기 조합 — 에서는 급격히 붕괴. 지시가 바뀌어도 **학습 때 하던 행동을 반복**하며(언어 채널이 사실상 vestigial), 단 한 번의 불안정한 접촉 실패가 monolithic rollout 전체를 탈선시킨다.
- **LLM 코딩 에이전트** (Code-as-Policies, ProgPrompt 등): 의미적·조합적 추론은 보완하지만, 순수 analytic primitive는 irregular grasping, constrained placement, articulated-object 조작에 부적합. 또한 스케일링이 곧 **skill library 확장**을 의미하게 되어, 새로 작성한 skill의 유효성/재사용성/안전성 판단 부담이 에이전트에 전가됨.

### 핵심 질문과 저자의 답

- "skill library를 늘리지 않고, **고정된 소규모 primitive의 operating range를 학습**하는 것만으로 frozen VLA를 분포 밖으로 확장할 수 있는가?"
- 답: **비대칭 계층 분해** — analytic primitive가 perturbation 공간을 횡단(traverse)하고, VLA는 학습 분포가 유효한 국소 contact-rich 영역 안에서만 호출. 여기에 Task Specific Memory(성공 trace)와 Global Memory(성공 규칙 + 실패 모델)로 "언제/어떻게 어떤 primitive를 쓸지"를 학습.

📌 [Figure 2 삽입] — perturbation으로 확장된 task 구성 공간에서 direct VLA rollout은 중도 실패; Harness VLA는 analytic primitive로 VLA-compatible 국소 영역 사이를 이동하고 VLA_ACT는 그 안에서만 호출.

---

## 2. 방법론 심층 분석

### 2.1 Agentic execution loop (Sec 2.1)

- 환경: MuJoCo/Robosuite 계열 rigid-body 물리엔진. 관측 $o_t = (I_t^{rgb}, I_t^d, q_t)$ — agent-view RGB, co-aligned metric depth, proprioception(EE pose + gripper).
- Task = 자연어 지시 ℓ + binary completion predicate G (episode 종료 시 sparse 성공 신호로만 노출).
- Planner Π가 매 턴 구조화된 **JSON primitive 호출** $c_t \in P$를 방출 → 물리엔진이 primitive 내부 post-condition 충족까지 실행 → 새 관측 반환. **VLA를 별도 계층으로 두지 않고**, frozen VLA $f_\theta$와 deterministic 컨트롤러를 모두 단일 primitive library P로 통합한 것이 구조적 특징.

### 2.2 REPL-style harness와 2단계 lifecycle (Sec 2.2)

- **Exploratory Bootstrapping**: task당 단일 reference seed(s0)에서 RESET primitive 허용 + 넉넉한 시간 예산으로 자율 탐색. staging 순서, pre-contact pose, VLA_ACT 호출 시점, early-return threshold를 반복 시행. 성공 시:
  - (i) 성공 primitive 시퀀스를 **JSONL trace**로 직렬화하되, 구체적 공간 좌표를 **symbolic perception query로 치환**(parameterization) → 다른 layout에서 재사용 가능한 구조적 prior로 Task Specific Memory에 저장.
  - (ii) 일반화된 heuristic(예: full task instruction을 활용한 prompting 전략)과 **failure model**(empty-grasp 감지, false success 감지)을 Global Memory에 축적.
- **Deployment Evaluation**: RESET 완전 비활성화, step budget 단축. JSONL trace를 retrieve하여 live RGB-D로 동적 grounding. 이 strict phase의 성능만 보고됨.
- 메모리는 rollout 후 일괄 기록이 아니라 **interaction 중 반복 구축**: primitive마다 outcome을 progress / recoverable failure / unrecoverable failure로 분류하고, 더 짧고 신뢰성 높은 해가 나오면 procedural trace를 교체.

### 2.3 고정 primitive vocabulary (Sec 2.3, Table 1/8)

| Primitive | 유형 | 역할 |
|---|---|---|
| MOVE_TO | Composite | world-frame Cartesian 목표로 EE 이동 (내장 solver) |
| MOVE_POSE | Composite | pitch 등 pose 변수 동시 조절 이동 |
| ROTATE_WRIST / ROTATE_PITCH | Atomic | yaw/pitch set-point |
| SET_GRIPPER / RELEASE | Atomic | gripper 개폐 / release post-condition |
| **VLA_ACT** | VLA | frozen VLA를 짧은 burst로 실행하는 유일한 학습형 primitive |
| NAVIGATE_TO / MOVE_BASE | (RoboCasa365 전용) | 주방 스케일 mobile-base staging |

- VLA_ACT 호출: `{"action": "vla_act", "prompt": <str>, "max_chunks": <int>, "stop": <predicate>}` — planner가 task-conditioned prompt와 early-return predicate τ를 공급, frozen VLA가 τ 충족 또는 chunk budget 소진까지 action chunk 방출.
- RoboTwin은 새 primitive 없이 `arm` 인자(left/right/bimanual binding)만 추가; handover는 VLA_ACT + analytic transport + RELEASE 조합으로 표현.
- **핵심 제약: primitive vocabulary는 평가 전 고정 — planner는 배포 시 새 primitive를 발명할 수 없다.**

> ❓ **예상 질문**: 이것은 skill-library 방식(ASPIRE 등)과 무엇이 다른가?
> **답변**: ASPIRE는 실패 진단→skill 합성→library 확장. Harness VLA는 정반대로 vocabulary를 고정하고 **조합 방법(operating range)만** memory로 학습 — 감사가능(auditable)한 인터페이스 유지가 설계 철학.

---

## 3. 데이터 전략

- **VLA 학습 데이터 없음** — 본 방법 자체는 training-free. 대신 벤치마크별 frozen backend를 그대로 채택:
  - LIBERO/LIBERO-Pro: RLinf의 `pi05_libero130_fullshot` π0.5-SFT 체크포인트 (LIBERO-130 SFT).
  - RoboCasa365: 공식 RLDX-1 체크포인트 (Qwen3-VL 8B + Multi-Stream Action Transformer, flow-matching DiT).
  - RoboTwin C2R: 자체 post-train한 **LingBot-VLA** (Qwen2.5-VL + MoT, L1 flow matching, chunk 50; 9개 embodiment 실기 teleop 데이터 사전학습 → RoboTwin SFT 후 동결. Table 14: AdamW lr 1e-4, vision encoder lr 1e-6, batch 256, 224×224, top+양팔 wrist 카메라, FSDP2, bf16).
- 에이전트가 소비하는 "데이터"는 자신의 **execution trace**: 성공 trace(JSONL) + 실패 관찰(negative evidence). Reference seed 1개만으로 bootstrapping한다는 점이 few-shot의 실체.

---

## 4. 실험 설계 및 평가 프로토콜

4개 벤치마크 패밀리 (Appendix C):

| 벤치마크 | Tasks | 시드/태스크 | 보고 rollouts | 프로토콜 |
|---|---|---|---|---|
| LIBERO (4 suites) | 40 | 10 (s1–s10) | 400 | few-shot; s0는 memory 구축 전용 |
| LIBERO-Pro (8 cells: Spatial/Object/Goal/L10 × T/S) | 80 | 10 | 800 | few-shot; T=instruction-redirection, S=position-swap |
| RoboCasa365 target50 (Atomic-Seen 18 / Comp-Seen 16 / Comp-Unseen 16) | 50 | 10/5/5 | 340 | few-shot, split별 시드 |
| RoboTwin C2R | 50 | 5 | 250 | **zero-shot clean-to-randomized**: clean 시드 1개에서 trace 획득 → 공식 demo randomized 시드에서 그대로 평가 |

- 성공 판정은 전부 **벤치마크 제공 binary completion predicate** (primitive post-condition은 제어 반환용일 뿐 성공 판정에 사용 금지).
- 두 planner 인스턴스: Harness VLA (Codex) vs Harness VLA (**CC** = Claude Code) — harness, memory, primitive, frozen VLA, 프로토콜 전부 동일하고 planner backbone만 다름.
- Zero-shot 분리 실험: LIBERO-Pro Goal에서 Task Specific Memory retrieval을 차단하고 순수 planner 추론만 평가.

---

## 5. 실험 결과 심층 분석

### 5.1 표준 LIBERO (Table 2) — "harness가 in-distribution 성능을 깎지 않는가"

- Harness VLA (CC) **96.0%** (384/400): Spatial 97.0 / Object **100.0** / Goal 94.0 / LIBERO-10 93.0.
- 내부 frozen RLinf 단독 95.3%, π0 94.2, AtomVLA 97.0. → harness 개입으로 표준 suite 성능이 보존됨 (Goal에서 97→94로 소폭 하락, Object에서 96→100 상승).

### 5.2 LIBERO-Pro (Table 3) — 논문의 headline

- End-to-end VLA의 붕괴가 극적: OpenVLA/NORA **0.0%**, π0 0.3%, MolmoAct 1.5%, X-VLA 3.8%, AtomVLA 6.3%, π0.5 11.0%.
- 기존 최강 baseline: RATS 43.8% (6 cell만 보고), Cap-X 18.2%.
- 직접 baseline RLinf 50.0% → **Harness VLA (Codex) 72.1% / (CC) 82.4%**. RATS 대비 +38.6%p가 headline. **같은 frozen checkpoint를 쓰는 RLinf가 50.0%라는 점이 결정적** — 이득이 VLA backbone이 아니라 harness에서 온다는 통제 비교.
- Cell별로 CC가 Spatial-T 94.0, Goal-T/S 87.0/87.0으로 균질하게 강함; L10-S 62.0이 최저.

### 5.3 RoboCasa365 (Table 4)

- RLDX-1 (직접 baseline): Atomic-Seen 60.0 / Comp-Seen 21.3 / Comp-Unseen 5.0, task-weighted 30.0%.
- Harness VLA (Codex) 91.6 / 56.3 / 13.8 (overall 55.4%, **+25.4%p**); (CC) 79.4 / 47.5 / 15.0 (48.6%). 여기서는 **Codex가 CC보다 우위** — planner backbone 간 순위가 벤치마크에 따라 뒤집힘.

### 5.4 Zero-shot 평가

- **LIBERO-Pro Goal zero-shot** (Table 5, memory retrieval 차단): CC가 Task-T **79.0%** vs Cap-X 16.8%, Pos-S 31.0% vs 25.6%. instruction-redirection은 memory 없이도 planner의 live 추론만으로 거의 해결되지만, position-swap은 31.0%로 크게 떨어짐 — **공간 재배치 대응이 memory에 훨씬 의존적**임을 시사.
- **RoboTwin C2R** (Table 6): frozen LingBot-VLA 단독 50.4% → Codex 58.0 / **CC 58.4%**. 외부 baseline: π0.5 47.9, GR00T-N1.7 20.7, StarVLA 10.6. 개선폭(+8.0%p)이 LIBERO-Pro보다 작은 이유는 bimanual contact-rich 비중이 커서 최종 predicate가 VLA 실행 품질 자체에 좌우되기 때문(Figure 6 attribution과 일관).

### 5.5 메커니즘 분석 (Sec 3.3, Key Findings)

1. **Planner-level semantic re-grounding** (Figure 3): RLinf는 지시가 바뀌어도 표준 행동을 반복(언어 조건화가 약함); harness는 planner가 지시를 파싱하고 live RGB-D로 대상을 re-bind.
2. **Planner-staged retryable invocation** (Figure 4/5): episode당 VLA 호출 수 상한을 sweep하면 처음 몇 회 호출만으로 frozen baseline을 초과하고 이후 포화 — VLA는 sparse하게 쓰이지만 **re-stage 후 재호출** 능력이 robustness의 핵심. 예: 우유팩이 바구니에 걸치면 EE를 re-stage하고 VLA_ACT 재시도.
3. **Analytic isolation** (Figure 6/7): 성공 rollout의 최종 predicate가 LIBERO-Pro 계열에서는 주로 analytic primitive(운반/release) 후 발화, RoboCasa/RoboTwin에서는 contact-rich 조작(VLA) 내부에서 발화 — 의도된 분업의 정량적 증거. MOVE_TO 중 grasp 실패(빈손) 감지 → 복귀 → VLA_ACT 재시도 사례가 대표적.

---

## 6. Ablation 분석

- 본 논문은 전통적 component-drop ablation 표 대신 **메커니즘 분석 3종**이 그 역할을 수행:
  - VLA 호출 수 capping curve (Figure 4) = "retryable invocation" 기여도의 dose-response.
  - Zero-shot(memory 제거) vs few-shot 비교 (Table 5) = Task Specific Memory의 기여 분리. Goal-S에서 few-shot 87.0 → zero-shot 31.0으로, **position-swap 강건성의 대부분이 memory에서 옴**.
  - 동일 harness에 planner만 교체 (Codex vs CC) = planner backbone 민감도. LIBERO-Pro에서는 CC +10.3%p, RoboCasa365에서는 Codex +6.8%p.
- **아쉬운 점**: Global Memory 단독 제거, 개별 analytic primitive 제거, depth 채널 제거 등의 세밀한 ablation은 없음. failure model이 실제로 몇 %p를 기여하는지 미분리.

---

## 7. 관련 연구 비교

| 축 | 기존 접근 | Harness VLA |
|---|---|---|
| VLA 강건화 | finetuning / co-training (π0.5), post-training (AtomVLA) | **완전 frozen**, 호출 방식만 학습 |
| 코딩 에이전트 | Code-as-Policies, ProgPrompt: 프로그램 합성; ASPIRE: skill library 확장 | **고정 vocabulary** + memory 기반 조합 학습 |
| 에이전트 harness | SWE-agent, OpenHands (SW 도메인), CodeAct | REPL-style file-mediated 인터페이스를 물리 조작에 이식 |
| Memory | Voyager (디지털 샌드박스) | VLA-backed contact primitive와 최초 결합; parameterized JSONL trace + 실패 모델 분리 저장 |
| VLA-Tracker 내 유사 계열 | CTRL-STEER (activation-level closed-loop steering), PaCo-VLA (passivity-shield 제안 인터페이스), Retrieve-then-Steer (success memory 기반 test-time guidance) | 셋과 같은 "frozen VLA wrapper" 계열이나, 개입 지점이 activation/제어 계약/sampler가 아닌 **task-level primitive 조합** |
| Agentic + VLA | Cap-X, RATS (LIBERO-Pro 선행 agentic baseline) | 동일 문제의식; primitive 고정 + 2-메모리 설계로 RATS 대비 +38.6%p |

---

## 8. 한계 및 미해결 문제

1. **저자 명시 한계**: (i) planner↔VLA 간 open feedback loop (VLA 실행 중 planner가 개입 불가), (ii) 환경 보상/인간 선호 기반 joint fine-tuning 부재 (GRPO 등 sample-efficient RL을 future work로 제시), (iii) fine-grained image captioning 부재로 고밀도 clutter·장기 task에서 구조적 추론 제약.
2. **시뮬레이션 한정** — 실기 검증 없음. RGB-D 기반 analytic staging은 실기에서 depth 노이즈·캘리브레이션 오차에 민감할 것.
3. **Bootstrapping 비용 미보고**: reference seed 탐색의 wall-clock/토큰 비용, 그리고 frontier LLM API 호출 비용이 정량화되지 않음. 실시간성(planner 턴당 지연)도 미보고.
4. **Per-task memory**: Task Specific Memory는 task 단위 — 완전히 새로운 task에는 bootstrapping을 다시 해야 함 (zero-shot 결과가 있긴 하나 Goal family 한정).
5. **Position-swap zero-shot 31.0%**: memory 없이는 공간 재배치 대응이 여전히 취약.
6. **평가 주체 문제**: LIBERO-Pro의 RLinf/자기 결과는 자체 프로토콜 평가라 RATS·Cap-X와 셀 구성이 완전 동일하지 않음(L10 미보고 baseline 존재).

---

## 9. 총평

frozen VLA의 "언어 채널이 vestigial하다"는 관찰을 정면으로 공략하여, **의미적 grounding을 VLA 밖으로 들어올리고 VLA를 contact specialist로 강등**시키는 분해가 깔끔하고, 동일 checkpoint 직접 baseline(RLinf 50.0 → 82.4)이라는 통제 비교 덕에 주장 설득력이 높다. skill library를 늘리지 않고 고정 vocabulary의 operating range를 memory로 학습한다는 프레이밍은 auditable robot agent 설계에 시사점이 크다. 다만 시뮬레이션 한정, bootstrapping 비용 미보고, 세밀한 memory ablation 부재로 "왜 되는가"의 정량 분해는 미완이다. Codex/CC 간 벤치마크별 순위 역전은 planner backbone이 무시할 수 없는 hyperparameter임을 보여준다.

---

## 10. Figure/Table 요약

| 참조 | 내용 |
|---|---|
| Figure 1 | 시스템 개요: planner + 2 memory + primitive library + rollout strip |
| Figure 2 | perturbation 공간을 analytic primitive로 횡단, VLA는 국소 호출 |
| Table 1/8 | primitive vocabulary와 벤치마크별 가용성 |
| Table 2 | 표준 LIBERO 96.0% (CC) |
| Table 3 | LIBERO-Pro 8-cell: CC 82.4 / Codex 72.1 / RLinf 50.0 / RATS 43.8 |
| Table 4 | RoboCasa365: Codex 55.4% overall vs RLDX-1 30.0% |
| Table 5 | zero-shot Goal: Task-T 79.0 vs Cap-X 16.8 |
| Table 6 | RoboTwin C2R: CC 58.4 vs LingBot-VLA 50.4 |
| Figure 4 | VLA 호출 수 capping — 소수 호출로 baseline 초과 후 포화 |
| Figure 6 | 최종 predicate 발화 primitive 분포 (analytic vs VLA) |
| Table 14 | LingBot-VLA post-training 설정 |

---

## 11. 예상 날카로운 세미나 질문

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | "few-shot이라지만 s0에서 RESET까지 허용한 탐색은 사실상 privileged practice run 아닌가?" | 맞음 — 다만 s0는 보고에서 제외되고 배포 시 RESET 금지. RATS/Cap-X 등 agentic baseline과의 프로토콜 공정성은 부분적으로만 논의됨. |
| 2 | "planner가 frontier LLM인데 호출 비용/지연은?" | 미보고. 턴당 LLM 호출 + primitive 실행이라 실시간 제어는 아님; 표준 LIBERO 96%는 지연이 성공률에 안 미치는 시뮬 환경 특성 덕. |
| 3 | "RoboTwin에서 +8.0%p로 이득이 작다. harness의 한계인가?" | Figure 6 참조 — bimanual C2R은 최종 predicate가 contact-rich 실행 내부에서 발화하는 비중이 높아 planner가 개입할 여지가 작음. |
| 4 | "Codex vs CC 순위가 벤치마크마다 뒤집히는 이유는?" | 논문은 분석하지 않음. planner의 공간 추론/코드 스타일 차이로 추정 — harness가 planner-agnostic하다는 주장에는 오히려 유리한 증거. |
| 5 | "empty-grasp 감지 같은 failure model은 어떻게 구현되나? 학습인가 규칙인가?" | Global Memory에 자연어/규칙 형태로 축적된 heuristic (예: gripper 닫힘 후 물체 미동 → empty grasp 판정, 시각적 근접만으로 종료 금지). 학습된 분류기는 아님. |
| 6 | "perception isolation을 주장하지만 depth map은 시뮬레이터 렌더링이다. 실기에서는?" | ground-truth pose는 차단하나 clean depth는 사용 — 실기 전이 시 가장 취약한 고리. 미검증. |
| 7 | "VLA_ACT의 stop predicate τ를 planner가 지정하는데, τ 오설정으로 인한 조기 종료/과실행은?" | bootstrapping에서 early-return threshold를 시행착오로 튜닝하고 trace에 기록. 배포에서 새 상황의 τ 일반화는 memory 품질에 의존. |
| 8 | "position-swap zero-shot 31%는 사실상 memory 의존성의 고백 아닌가?" | 그렇게 읽을 수 있음. 저자 주장은 'planner 추론만으로도 Cap-X는 이긴다'까지이고, swap 강건성은 few-shot memory가 담당. |
| 9 | "Table 3에서 Cap-X/RATS는 6 cell 평균, 본인들은 8 cell 평균 — 공정 비교인가?" | 논문이 각주로 명시. L10 두 cell이 CC의 최저 cell(71/62)이므로 8-cell 평균이 오히려 불리 — 비교 방향은 보수적. |
| 10 | "이 방식이 primitive가 부족한 dexterous hand나 deformable object로 확장되는가?" | vocabulary 고정 철학의 경계 조건. 저자도 ASPIRE식 skill discovery와의 결합을 future work로 제시 — 새 abstraction이 필요해지면 고정 vocabulary는 병목. |

---

## 12. 결론

Harness VLA는 frozen VLA를 재학습 없이 **retryable contact-rich primitive**로 재정의하고, 고정된 analytic primitive vocabulary + 2단 메모리(Task Specific / Global)로 그 호출법을 학습하는 agentic harness다. LIBERO-Pro +38.6%p, RoboCasa365 +25.4%p, RoboTwin C2R 58.4%라는 결과는 "VLA의 실패는 능력 부족이 아니라 책임 과부하"라는 진단을 강하게 뒷받침하며, 동일 frozen checkpoint 대비 통제 비교로 이득의 출처를 harness에 귀속시킨 점이 방법론적으로 우수하다. skill library 확장 대신 **operating range 학습**이라는 프레이밍, 그리고 CTRL-STEER(activation 개입)·PaCo-VLA(제어 계약 개입)·Retrieve-then-Steer(sampler 개입)와 구별되는 **task-level 조합 개입**이라는 위치가 이 논문의 정체성이다. 실기 검증, bootstrapping 비용 정량화, memory 구성요소별 ablation, closed feedback loop(VLA 실행 중 planner 개입)가 후속 과제로 남는다.

<!-- VERIFIED: pdf -->
