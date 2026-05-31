# PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models 세미나 리뷰

> **한 줄 요약**: 사전학습된 π₀.₅ 류 VLA의 prior가 full fine-tuning 중 좁은 학습 분포로 드리프트하는 문제를 해결하기 위해, **frozen Prior Expert + 학습 가능한 Adaptation Expert**를 두고 3종의 Expert Query(Scene/Motor/Action)로 prior를 명시적으로 추출/주입하는 PEFT 프레임워크. Full fine-tune의 **25% 파라미터**만 갱신하면서 LIBERO **99.1%**, RoboTwin 2.0-Hard 평균 **52.8%**, 실로봇 standard 81/57 ID/OOD, **10-shot OOD에서 π₀.₅ 대비 +22pp**.

---

## 1. 배경 및 동기

대규모 사전학습된 generalist VLA(π₀, π₀.₅, OpenVLA-OFT 등)는 광범위한 prior를 갖지만, 다운스트림 적용 시 일반적으로 **full fine-tuning**되며 이는 prior를 좁은 학습 분포로 끌어당기는 부작용을 낳는다. 결과적으로 ID(in-distribution) 성능은 올라가지만 OOD와 few-shot에서 일반화가 손상된다. PriorVLA의 명시적 목표는 "prior를 read-only 자원으로 분리하고, 별도 모듈만 학습한다"는 것이다.

---

## 2. 방법론 심층 분석

### 2.1 백본 — π₀.₅ 기반

- **VLM**: SigLIP-style vision encoder + **Gemma-2B** language backbone
- **Action expert**: flow-matching (continuous action chunks)
- **Adaptation 대상**: Prior Expert(동결) + Adaptation Expert(학습 가능, Prior Expert와 동일 weight로 초기화)
- **Trainable budget**: full fine-tuning이 갱신할 파라미터의 **~25%**

### 2.2 Dual-Expert 디자인

두 expert는 같은 weight로 초기화되며, 같은 noisy action chunk를 입력받지만:

| Expert | 학습? | 역할 | 출력 사용 |
|---|---|---|---|
| **Prior Expert** | ❌ frozen | 사전학습 행동/감각 prior 보존 (read-only) | Motor query에 의해 prior 추출됨 |
| **Adaptation Expert** | ✅ trainable | downstream 태스크 특화 | trajectory 업데이트에 직접 사용 |

이 비대칭 구조가 prior drift를 차단한다 — full fine-tune은 prior 자체를 수정하지만, PriorVLA는 prior를 보존한 채 그 *외부*에 적응 모듈을 둠.

### 2.3 Expert Queries — 세 종류의 학습 가능 토큰

1. **Scene Queries**: VLM 입력에 삽입되어 **task-relevant visual priors**를 사전학습 VLM의 출력으로부터 추출 (장면 grounding)
2. **Motor Queries**: 학습 가능 토큰. Prior Expert의 hidden state로부터 **motor priors**(어떻게 움직여야 하는가)를 추출
3. **Action Queries**: Adaptation Expert 내부에서 **scene + motor prior를 통합**해 최종 행동 생성에 conditioning

이 세 query는 prior 흐름을 **"VLM scene → Prior Expert motor → Adaptation Expert action"**으로 명시화한다는 점에서 단순 LoRA/Adapter와 다르다.

---

## 3. 데이터셋 및 평가 프로토콜

- **RoboTwin 2.0-Hard**: 50 demo / task, 13 OOD-difficult 태스크
- **LIBERO**: 4 suite, 50 demo / task
- **Real-world**: 8 태스크 × 2 embodiment
  - Standard: 태스크당 100-300 demo
  - Few-shot: 태스크당 10 demo
- **하드웨어**: 8 GPU (RoboTwin은 H20, LIBERO/real-world는 A100)
- **학습 step**: 30K, batch 32-256, LR 2.5e-5~5.0e-5

---

## 4. 실험 결과

### 4.1 LIBERO (Table 3)

| Suite | π₀.₅ | **PriorVLA** | Δ |
|---|---|---|---|
| Spatial | 98.8 | **99.4** | +0.6 |
| Object | 98.2 | **99.8** | +1.6 |
| Goal | 98.0 | **99.4** | +1.4 |
| Long | 92.4 | **97.6** | **+5.2** |
| **Avg** | 96.9 | **99.1** | **+2.2** |

LIBERO-Long에서 +5.2pp가 가장 큰 향상 — 사전학습 prior 보존이 long-horizon에서 특히 가치 있음을 시사.

### 4.2 RoboTwin 2.0-Hard (Table 1, 13 태스크 OOD)

| Task | PriorVLA SR (%) |
|---|---|
| Grab Roller | 93 |
| Handover Mic | 84 |
| Open Laptop | 83 |
| Stack Bowls Two | 73 |
| Lift Pot | 66 |
| Move Can Pot | 57 |
| Put Bottles Dustbin | 45 |
| Put Object Cabinet | 45 |
| Place Object Basket | 42 |
| Place Phone Stand | 35 |
| Pick Dual Bottles | 26 |
| Place Dual Shoes | 20 |
| Stack Blocks Two | 17 |
| **Mean (13 tasks)** | **52.8** |

π₀.₅ baseline 대비 **+11pp** 평균 향상. 어려운 dual-arm / 정밀 grasping 태스크(Place Dual Shoes 20, Stack Blocks Two 17)는 여전히 낮음.

### 4.3 Real-World Standard (Table 4)

100-300 demo / task 학습:

| Task | ID (%) | OOD (%) |
|---|---|---|
| Place Ring | 90 | 55 |
| Insert Peg | 75 | 30 |
| Pick Object | 90 | 60 |
| Stack Blocks | 90 | 65 |
| Stack Bowls | 95 | 85 |
| Sweep Blocks | 95 | 85 |
| Arrange Cups | 50 | 25 |
| Hang Towel | 65 | 50 |
| **Avg** | **81** | **57** |

ID-to-OOD 격차가 24pp로 비교적 큼 — Insert Peg 같은 정밀 태스크는 OOD에서 30%로 떨어짐.

### 4.4 Real-World Few-Shot (Table 5, 태스크당 10 demo)

| Task | ID (%) | OOD (%) |
|---|---|---|
| Place Ring | 75 | 40 |
| Insert Peg | 30 | 20 |
| Pick Object | 85 | 50 |
| Stack Blocks | 65 | 55 |
| Stack Bowls | 55 | 30 |
| Sweep Blocks | 35 | 30 |
| Arrange Cups | 15 | 10 |
| Hang Towel | 25 | 20 |
| **Avg** | **48** | **32** |

- ID 48% — full-data ID 81%의 약 60% 수준
- **OOD 32%** — π₀.₅ 대비 **+22pp** (10 demo만으로 OOD 일반화가 가능함을 입증)
- few-shot ID는 π₀.₅ 대비 **+24pp**

이 결과가 PriorVLA의 가장 강한 selling point — prior 보존이 데이터 부족 상황에서 generalization을 살린다는 직접적 증거.

---

## 5. Ablation 분석 (Table 6, RoboTwin 2.0 Easy/Hard %)

### 5.1 Prior Expert 변종

| 변종 | Easy | Hard |
|---|---|---|
| w/o PE | 75 | 42 |
| Random PE | 75 | 43 |
| Trainable PE | 73 | 44 |
| **Full PriorVLA (frozen PE)** | **77** | **49** |

- Random PE도 frozen PE만큼 안 됨 → PE의 사전학습 가중치 자체가 본질적인 prior 역할
- Trainable PE는 Hard에서 -5pp → 학습시키면 prior drift가 다시 일어남
- **frozen + initialized from pretraining**의 조합이 유일하게 가장 좋음

### 5.2 Expert Queries 변종

| 변종 | Easy | Hard |
|---|---|---|
| w/o SQ/MQ/AQ | 61 | 28 |
| w/o SQ | 75 | 42 |
| w/o MQ | 71 | 43 |
| w/o AQ | 75 | 42 |
| **Full** | **77** | **49** |

- 세 query 모두 제거 시 Hard 28% — **−21pp**, query 시스템이 핵심
- Motor Query 제거가 Easy에서 가장 큰 손실(-6pp) — motor prior 전달이 sketch-level 행동 형성에 결정적
- Scene Query 제거는 Hard에서 -7pp → 시각 grounding은 어려운 태스크에 더 중요

---

## 6. 한계 및 의의

**한계**:
- **π₀.₅에 강하게 결합** — 다른 backbone(OpenVLA, RT-2 등)에서도 prior preservation 효과가 성립하는지 미검증.
- **Trainable parameter의 정확한 개수와 위치** 미보고 — "full fine-tune의 25%"라는 비율만 제공.
- **코드/체크포인트 미공개** — 프로젝트 페이지(priorvla.github.io)만 존재. 재현성 평가가 어려움.
- RoboTwin 2.0-Hard에서 절대 점수가 여전히 낮은 태스크(Stack Blocks Two 17%, Place Dual Shoes 20%)들은 prior 보존만으로 해결되지 않음.
- few-shot에서 Arrange Cups 15%(ID) — 일부 태스크는 10 demo로는 본질적으로 학습 불가.

**의의**:
- "prior preservation"이라는 명시적 목표를 단순 PEFT(LoRA/Adapter)가 아닌 **dual-expert + query 분리**라는 구조적 디자인으로 구현한 작업.
- **10-shot OOD에서 +22pp**는 데이터 부족 실배포 시나리오에서 직접적 가치를 보여줌.
- Ablation이 충실 — Frozen PE / Random PE / Trainable PE / w/o Each Query 모두 비교되어 디자인 선택의 근거가 분명.
- 8 real-world 태스크 × 2 embodiment 검증 — sim-only 연구보다 신뢰도 높음.
- 25% 파라미터만 학습 → A100 5h(real-world), H20 5.6h(RoboTwin) 같이 학습 비용도 합리적.

---

## 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | "25% 파라미터"의 정확한 위치는? | Action expert 측에 집중된 것으로 보이나 layer-by-layer 분해는 미보고. future work |
| 2 | Random PE도 75%/43% 인데 왜 frozen PE가 본질적인가? | Random PE는 input-agnostic noise에 가까움 → MQ가 의미 있는 motor prior를 추출 못함. Frozen PE의 사전학습 가중치가 informative prior source |
| 3 | LIBERO 99.1%는 saturated benchmark — 진짜 차이는? | LIBERO-Long +5.2pp가 의미 있는 신호. Spatial/Object/Goal은 사실상 ceiling |
| 4 | RoboTwin Place Dual Shoes 20% — 왜 그렇게 어려운가? | 양손 정밀 좌우 대칭 조작 + 변형 가능 물체. Prior가 도와도 fine-motor 자체 한계 |
| 5 | 10-shot ID 48% vs full ID 81% — 데이터 효율성은? | 10 demo로 60% 수준 도달 — 실배포 관점에서 매우 매력적 |
| 6 | Expert Query 수는 어떻게 정했나? | 구체 hyper-parameter 미보고. SQ/MQ/AQ 개수 ablation 부재 |
| 7 | π₀.₅ baseline은 같은 25% 파라미터로 학습된 것인가? | 본문 비교는 π₀.₅ full fine-tune vs PriorVLA(25%). 매칭된 PEFT budget 비교는 없음 — 후속 검증 필요 |
| 8 | OpenVLA 등 다른 backbone에 plug-in 가능한가? | π₀.₅ specific (dual flow-matching expert가 전제). architectural extension 필요 |

<!-- VERIFIED: pdf -->
