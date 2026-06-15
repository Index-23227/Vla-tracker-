# CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy

> **한 줄 요약**: **하나의** 사전학습된 π0.5 VLA에 LoRA 파인튜닝만 적용해, **여러 대의 이종(異種) 모바일 매니퓰레이터**가 **추론 시 어떠한 통신도 없이** 각자 자기 관측만 보고 협력 조작을 수행하도록 만든 **완전 분산형(decentralized) 멀티-임바디먼트** VLA 정책. Stanford 팀(Ria Doshi, Tian Gao, Annie Chen, Chelsea Finn, Jeannette Bohg), 2026-06.

---

## 1. 배경 및 동기

- 멀티로봇 협력 조작에는 전통적으로 두 갈래의 접근이 있다.
  - **중앙집중형(Centralized)**: 팀 전체의 관측을 한 정책이 받아 모두의 action을 한 번에 출력 [RoCo, ALOHA 등]. → 팀 크기에 따라 context window / action space가 **선형 증가**, 추론 시 **로봇 간 통신 필수**.
  - **분산형(Decentralized) — 로봇별 정책**: 로봇마다 별도 policy [MIMIC-D, LatentToM 등]. → context는 일정하지만, 부분관측(partial observability)을 보완하려고 **teammate proprioception 공유, 공통 3인칭 카메라, online alignment** 같은 추론 시 가정이 필요. 또 각 로봇이 따로 학습 → 학습 비용↑, embodiment 간 표현 공유 안 됨.
- 한편 단일 로봇 도메인의 **사전학습 VLA**(OpenVLA, π0, π0.5 등)는 강한 visuomotor prior를 가짐. **양손(bimanual) 매니퓰레이션**은 본질적으로 멀티로봇의 단순화 형태(물리적으로 결합·중앙 제어)인데, VLA들은 이미 이를 잘 학습함.
- **연구 질문**: "VLA의 시각·운동 prior가 충분히 강하다면, **추론 시 alignment/통신 없이도** 분산 멀티로봇 협력을 가능하게 할 수 있지 않을까? 더 나아가 **단 하나의** 공유 가중치 정책으로 모든 로봇을 제어할 수 있지 않을까?"

---

## 2. 방법론 심층 분석

### 2.1 전체 그림
- 단일 정책 π_θ(A_r | o_r, c_r). 모든 로봇 r에 대해 **같은 파라미터**.
- 학습: 모든 로봇의 시점에서 본 single-robot tuple을 한 데이터셋 D로 모아 학습. **joint observation은 절대 보지 않음.**
- 추론: 각 로봇이 독립 인스턴스를 돌리며 자기 관측 o_r과 자기 식별 프롬프트 c_r만 입력.

### 2.2 백본 — π0.5 + LoRA
- π0.5 (PaliGemma 기반 VLM + flow-matching action expert)를 backbone으로 채택.
- LoRA rank: VLM에 16, action expert에 32.
- 32차원으로 **padded action vector**, **가변 image token** → 이종 임바디먼트의 DoF/카메라 수 차이를 아키텍처 변경 없이 흡수.

### 2.3 로봇 식별 프롬프트 c_r
- 매 timestep 모델 입력 앞에 prepend.
- 형식 예: `<YAM> Lift the measuring tape and hold for the Kinova`.
- 임바디먼트 이름(`<ARX>`, `<Kinova>`, `<YAM>`)을 명시해, "지금 내가 어느 로봇을 제어 중인가"를 관측에서 추론할 필요를 없앰.

### 2.4 Robot Sampler
- 매 배치는 **로봇별 single-robot tuple (o_r, A_r, c_r)** 을 독립적으로 sampling.
- 2-robot: 제어주파수 차이가 최대 2배 이내 → 균등 sampling.
- 3-robot (Kinova + 2×YAM): YAM 두 대가 합쳐서 Kinova의 ~4배 action을 만들기 때문에 Kinova tuple에 가중치 ×2 부여 → undertraining 방지.

### 2.5 Flow-matching Loss
- π0.5의 손실을 그대로 계승:
  - L(θ) = E_{(o_r,A_r,c_r)~D, τ~U(0,1), ε~N(0,I)} || v_θ(A_r^τ, τ | o_r, c_r) − (ε − A_r) ||²
  - A_r^τ = τ A_r + (1−τ) ε.
- 핵심: 손실은 **절대 joint (o, A)를 보지 않음** — single-robot tuple에 대해서만 정의됨.

### 2.6 분산 추론 + 비동기 chunk 실행
- 각 로봇 r은 매 timestep에 A_r^t ~ π_θ(·|o_r^t, c_r) 샘플 → 자신의 chunk를 독립 실행.
- chunk size를 제어주파수에 **비례**: YAM(30 Hz) chunk 40, Kinova(15 Hz) chunk 20 → 같은 plan horizon.
- **비동기**: Kinova가 chunk t의 액션을 실행 중일 때 ARX는 chunk t′(≠t)의 액션을 실행 가능. 중앙집중형은 매 query마다 가장 느린 링크를 기다려야 하지만, 분산은 minor latency gap을 자연스럽게 흡수.

### 2.7 Collaboration Strategy (데이터 수집 측 디자인)
- "분산 실행이 원천적으로 가능해야" 하므로, 데이터 수집 시 각 로봇의 카메라에 **상대 로봇과 작업물이 충분히 잡히도록** 동선을 설계.
- 즉, "이 task를 분산으로 풀 수 있게 만드는 책임"이 **데이터 큐레이션 단계**에 부분적으로 내려와 있음 — 한계의 중요한 단서(§9).

---

## 3. 데이터 전략

- 플랫폼: **TidyBot++** 오픈소스 holonomic mobile manipulator + **phone-based teleoperation**.
  - 2-robot: 한 명의 텔레오퍼레이터가 양손에 폰 하나씩.
  - 3-robot: 두 번째 오퍼레이터가 세 번째 로봇 추가 담당.
- 로봇 라인업: ARX(10-DoF), Kinova(11-DoF, 팔+베이스), YAM(10-DoF, 베이스 포함).
- Episode ξ = {(o_r^t, a_r^t)} for r ∈ [N], t ∈ [T] — **동기 로깅**된 trajectory.
- 추출 단위: 로봇별 tuple (o_r^t, A_r^t, c_r). A_r^t = (a_r^t, …, a_r^{t+H−1}) — chunk horizon H.
- Task별 데모 수:
  - Basket Lift: **43** demos (ARX + Kinova)
  - Tape Measure: **29** demos (YAM + Kinova)
  - Book Handover: **45** demos (YAM + Kinova)
  - 3-Robot Move: YAM 둘에는 **34**, Kinova에는 **18** (door-opening 일부 손상). **비대칭 demo 수가 분산 학습의 장점을 보여주는 사례** — centralized라면 18로 잘랐어야 함.

---

## 4. 시스템 / 학습 세부사항 (Appendix A·B)

| 항목 | 값 |
|---|---|
| Optimizer | AdamW |
| Batch size | 64 |
| Training steps | 20,000 |
| Training chunk size | 50 |
| Peak LR | 2.5 × 10⁻⁵ |
| Final LR | 2.5 × 10⁻⁶ |
| Decay steps | 30,000 |
| Warmup steps | 1,000 |
| LoRA rank (VLM) | 16 |
| LoRA rank (action expert) | 32 |
| Train compute | **1× NVIDIA H100** |
| Deploy compute | **1× NVIDIA RTX 5090** (여러 로봇 query batching 가능) |

**Deployment chunk**:
- Basket: 두 로봇 모두 동기, chunk 30.
- Tape/Handover: YAM chunk 40, Kinova chunk 20 (주파수 2:1 매칭).

**카메라 슬롯**: π0.5는 단일 모바일 매니퓰레이터의 front/back/base/wrist 4슬롯을 가정. CHORUS는 한 로봇의 top + wrist만 사용해 그 안에 맞춤 → pretraining 분포와 정합성 유지가 분산 정책의 이점(§6.3).

---

## 5. 실험 설계 및 평가 프로토콜

**4가지 핵심 질문 (Q1–Q4)**:
1. **Q1.** Pretrained backbone이 from-scratch 대비 의미 있는 이점을 주는가?
2. **Q2.** 가중치 공유가 teammate reactivity를 개선하는가?
3. **Q3.** 전체 관측을 보는 centralized 정책 대비 어떤가?
4. **Q4.** 3-robot으로 무수정 확장 가능한가?

**Baseline 4종 (Table 1)**:
| Baseline | 분산? | 가중치 공유? | N=2 파라미터 | N=2 context |
|---|:-:|:-:|:-:|:-:|
| **CHORUS (Ours)** | ✅ | ✅ | 3B | C |
| CHORUS (w/o WS) | ✅ | ❌ (로봇별 정책) | **6B** | C |
| VLA Centralized | ❌ | ✅ | 3B | **2C** |
| Decentralized Diffusion | ✅ | ❌ | n/a (from-scratch) | C |

**평가 셋업**:
- Task당 10–18 rollouts.
- 일부 rollout에 distractor 추가.
- 각 로봇은 task 전체에 대해 **하나의 prompt**만 받음 (subtask prompt 아님).
- 채점: 부분 성공에 half credit (한 로봇만 잡으면 0.5점 등).

---

## 6. 실험 결과 심층 분석

### 6.1 Q1 — Pretrained backbone 효과 (Figure 4)

- 두 VLA 기반 방법(CHORUS, CHORUS w/o WS) 모두 decentralized diffusion을 **압도적으로 능가**.
- **CHORUS의 평균 success rate는 decentralized diffusion 대비 +64pp**.
- Diffusion의 전형적 실패: **mismatch pattern** — 한 로봇이 자기 몫을 먼저 끝내고 진행해버려 (예: Kinova가 손잡이 잡고 출발해버려 ARX가 따라잡지 못함) → basket이 미끄러짐.
- Tape measure에서는 distractor가 들어가면 diffusion이 tape를 다른 물체로 혼동.
- 시사점: 협력 task가 pretraining 분포 밖이라도 VLA prior가 매우 유익.

### 6.2 Q2 — Weight sharing 효과 (Figure 4 + Figure 5)

- 일반(distribution-in) 시나리오에서는 CHORUS와 w/o WS가 **비슷한** 성공률.
- 그러나 **teammate reactivity 시험**(handover task에서 YAM을 scripted로 좌/우 perturb):

| | CHORUS (w/o WS) | **CHORUS** |
|---|:-:|:-:|
| Left perturb | 3/10 | **8/10** |
| Right perturb | 6/10 | **9/10** |
| Total (20회) | **9/20 (45%)** | **17/20 (85%)** |

- **CHORUS가 거의 2× 효과적**, +40pp.
- 실패 모드: w/o WS는 "in-distribution 위치에서 일찍 잡기" — YAM의 새 위치를 추적 못 함.
- 해석: 가중치 공유가 **양쪽 로봇 관점의 데이터를 동시에 학습**하게 만들어, 자연스럽게 **teammate behavior에 대한 표현**을 내부에 형성. 로봇별 정책은 자기 데이터만 보니 그런 동기가 없음.

### 6.3 Q3 — Centralized 대비 (Figure 6 + Table 3)

| Task | VLA Centralized | **CHORUS** |
|---|:-:|:-:|
| Basket Lift | ~ | ≈ 매칭/우위 |
| Tape Measure | ~ | 우위 |
| Book Handover | **0.611** | **0.833** |

- 이론상 centralized는 strictly more information을 받으므로 upper bound가 되어야 함. 그런데 실제로는 **CHORUS가 평균적으로 능가**.
- 저자 분석:
  1. **분포 이탈**: π0.5는 단일 모바일 매니퓰레이터 4-슬롯 카메라를 가정. Centralized는 그 슬롯에 "두 로봇의 top × 2, wrist × 2"를 강제로 채워야 함 → pretraining의 의미론적 대응이 깨짐.
  2. **차원 증가 → BC 성능 저하**(de Haan et al. 2019의 causal confusion 효과). Centralized는 팀 전체 state에 조건. 작은 desynchronization도 분포 이탈.
- CHORUS는 한 로봇 관측만 보니 pretraining 분포에 가까움 → prior 보존.

**Frequency reconciliation (Table 3, handover)**:
- VLA Centralized (downsample 30→15) = 0.611
- VLA Centralized (upsample 15→30) = 0.611
- → Resampling 방식과 무관하게 CHORUS(0.833) 우위.

### 6.4 Q4 — 3-robot 확장 (§4.4)

- Kinova가 문 열기, YAM 둘이 빨래 바구니 들고 문 통과해 침실로 배달.
- 어려움 2가지:
  1. YAM-1은 후방 카메라가 없어 Kinova가 문에서 비킨 시점을 직접 못 봄 → **YAM-2의 행동을 보고 추론**해야 함.
  2. YAM-2는 문틀 충돌 회피로 미세 조정 필요 → YAM-1은 그 동안 일시정지해야 바구니가 안 미끄러짐.
- 결과: **90% success rate**, 아키텍처 무수정 확장.

### 6.5 Summary 표 (실세계 SR)

| Task | Decentralized Diffusion | VLA Centralized | CHORUS w/o WS | **CHORUS** |
|---|:-:|:-:|:-:|:-:|
| Basket Lift | 낮음 | ~ | ~ | ≈ 100% |
| Tape Measure | 매우 낮음 (distractor 혼동) | ~ | ~ | ≈ 100% |
| Book Handover | 낮음 | 0.611 | ~ | **0.833** |
| Handover Reactivity (perturb) | – | – | 9/20 | **17/20** |
| 3-Robot Move | – | – | – | **0.90** |
| **Mean (2-robot)** | – | – | – | ≈ 94% |
| Mean gap vs decentralized diffusion | – | – | – | **+64 pp** |

---

## 7. Ablation 분석

1. **Without pretrained backbone (Decentralized Diffusion)**: 평균 −64pp → backbone이 핵심.
2. **Without weight sharing (CHORUS w/o WS)**: in-distribution에선 동등, but reactivity에서 −40pp (45% vs 85%). 또한 **파라미터가 N배** 증가(N=2일 때 3B → 6B; N=3이면 9B). 학습 비용도 N배.
3. **Centralized vs Decentralized**: centralized가 정보를 더 보는데도 성능이 낮음 → **추가 정보 ≠ 추가 성능** (분포 이탈 + 차원 증가 비용).
4. **Frequency reconciliation (Table 3)**: up/downsample 둘 다 같은 0.611 → centralized가 본질적으로 약함. 단순한 trick으로 회복 안 됨.

---

## 8. 관련 연구 비교 (저자 정리, §Appendix D)

| 방법 | 분산? | 추론 시 통신? | 카메라 공유? | Alignment? | 비고 |
|---|:-:|:-:|:-:|:-:|---|
| MIMIC-D | ✓ | proprio 공유 | – | – | 주로 시뮬, 이미지 conditioning X |
| LatentToM | ✓ | 3인칭 카메라 공유 | ✓ | online (sheaf Laplacian) | iterative update 필요 |
| GCo (flow-matching co-gen) | ✗ | central planner | – | – | non-prehensile pushing, 시뮬 |
| decPLM | (학습은 centralized) | – | – | – | 박스 픽업 특화, MLP+state |
| Bimanual (ALOHA 등) | ✗ | 단일 정책 양팔 | 3인칭 공유 | – | CHORUS의 centralized baseline 구조와 유사 |
| **CHORUS** | ✓ | **없음** | **없음** | **없음** | full decentralization 유지하며 VLA backbone 도입 |

→ "VLA backbone을 분산 멀티로봇에 적용한 최초"라는 포지셔닝이 합리적.

---

## 9. 한계 및 미해결 문제

1. **순간 동기화가 필요한 task는 불가**: 두 그리퍼를 정확히 같은 control step에 열어야 하는 등의 strict synchronization task는 분산으로는 원천적으로 어려움. 이런 영역은 centralized가 필수.
2. **데모가 분산-feasibility를 보장해야 함**: 데이터 수집 시 각 로봇의 view에서 teammate와 작업물이 충분히 보이도록 동선을 일부러 설계. 즉, "이 task를 분산으로 풀 수 있게" 사람이 사전에 결정해야 함. 일반 데이터셋을 그대로 가져다 쓸 수 없음.
3. **Latency gap이 커지면 desynchronize**: 비동기 chunk가 minor gap만 흡수. 큰 지연이 발생하면 데이터 분포에서 이탈.
4. **대규모 collaborative dataset 부재**: 기존 manipulation dataset (OXE, DROID, Bridge V2)은 압도적으로 single-robot. Collaborative data 수집 community effort가 필요.
5. **카메라 슬롯 정합성**: π0.5의 4-슬롯 의미론(front/back/base/wrist)에 묶여 있어, 카메라 구성이 크게 다른 로봇으로 옮기면 prior 활용도가 떨어질 수 있음.
6. **Quantitative evidence가 real-world 4개 task에 한정**: 시뮬 benchmark 비교 부재. Reproducibility 측면에서는 외부 검증이 어려울 수 있음 (코드 공개 여부도 불명).
7. **Prompt 의존성**: `<ARX>`, `<YAM>` 같은 임바디먼트 이름과 역할을 명시한 프롬프트가 학습 시 본 형식과 다르면 어떻게 일반화될지 미검증.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "VLA + 분산 멀티로봇"이라는 조합 자체가 새로움. Robot ID prompt + padded action만으로 가중치 공유가 동작한다는 깔끔한 결과. |
| **Technical depth** | ★★★☆☆ — π0.5 + LoRA는 표준. 진짜 기여는 데이터 큐레이션 + sampling + asynchronous chunk 등 **실제 시스템 설계**. |
| **Empirical rigor** | ★★★★☆ — 4개 task × 4 baseline × distractor 변형 + scripted reactivity 실험까지 잘 통제. Frequency reconciliation까지 점검(Table 3). 단, 시뮬 비교 없음. |
| **Practical impact** | ★★★★☆ — 다종 모바일 매니퓰레이터 팀을 별도 통신 인프라 없이 협력시킬 수 있다는 점이 산업적으로 매우 매력적. 비대칭 demo 활용도 ↑. |
| **Reproducibility** | ★★☆☆☆ — π0.5 backbone과 TidyBot++ 하드웨어가 필수. 코드/데이터 공개 정보가 paper 상 명확치 않음. |

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Centralized가 정보를 더 보는데 왜 더 나쁜가?" | (1) π0.5의 4-슬롯 카메라 의미론이 깨짐, (2) 입력 차원 증가에 따른 BC 성능 저하(causal confusion). 분포 이탈 비용 > 추가 정보의 이점. |
| 2 | "Robot ID prompt만으로 진짜 가중치 공유가 동작하나? 모델이 prompt를 무시할 수도 있는데." | Reactivity 실험에서 w/o WS는 자기 데이터만 보고 학습 → teammate 관점 표현이 약함. 같은 가중치가 두 로봇 시점을 모두 학습한 것이 +40pp의 reactivity 격차로 직접 입증. |
| 3 | "분산이면 teammate가 갑자기 멈추거나 다른 행동을 하면 어떻게 되나?" | 시각 기반 reactive coordination이 작동. 단, latency gap이 크거나 perturb이 학습 분포 밖이면 desynchronize. Strict-sync task는 원천적으로 불가. |
| 4 | "Demo 비대칭(YAM 34 vs Kinova 18)이 정말 이점인가?" | Centralized는 매 step마다 모든 로봇 데이터가 있어야 하므로 18에 맞춰야 함. CHORUS는 single-robot tuple로 학습하므로 각 로봇의 모든 demo 활용. → 분산 학습의 데이터 효율 장점. |
| 5 | "왜 LoRA만? Full fine-tuning이 더 낫지 않나?" | π0.5의 prior 보존이 핵심 가설. Full FT는 분포 이탈을 더 일으킬 위험. (AnySlot에서는 full FT가 필요했는데, 그건 sub-cm 정확도 task; 여기는 협력 운동 prior가 중요.) |
| 6 | "비동기 chunk가 desynchronize되면?" | Minor gap은 흡수하지만 large latency는 분포 이탈. 작은 chunk + 빠른 query 빈도가 안전. 실험은 RTX 5090 한 대로 batching → 실제 RTT 거의 동일. |
| 7 | "이걸 보고 못 보는 사각지대 robot도 협력시킬 수 있나?" | 3-robot 실험의 YAM-1이 사례. Kinova를 직접 못 보는 대신 YAM-2의 행동을 단서로 사용 → 정책이 indirect signal에서도 추론 가능함을 시사. 다만 데이터에서 그런 신호가 등장해야 함. |
| 8 | "팀 크기가 더 커지면(5~10대)?" | 파라미터/context는 일정. 학습 시 robot sampler의 가중치만 조정. 다만 visual coordination의 한계가 어디서 깨질지는 실험되지 않음 — open question. |
| 9 | "왜 시뮬 benchmark(LIBERO/CALVIN/RoboCasa) 비교가 없나?" | 멀티로봇 collaborative manipulation은 표준 시뮬 벤치마크가 사실상 부재. 새로운 collaborative benchmark의 필요성을 저자도 limitation에 명시. |
| 10 | "Robot ID prompt가 안 보이면(zero-shot 새 로봇)?" | 본 논문은 다루지 않음. Zero-shot embodiment 일반화는 후속 연구 영역. |

---

## 12. 종합 결론

CHORUS는 **"VLA prior가 충분히 강하다면 멀티로봇 협력은 추론 시 통신·alignment 없이 한 로봇 관점에서도 가능하다"** 는 가설을 깔끔한 실험 4종으로 입증한 작업이다. 핵심 기여를 압축하면:

1. **단일 π0.5 백본 + LoRA + robot ID prompt** 로 이종 모바일 매니퓰레이터 팀을 한 정책이 제어.
2. **추론 시 어떠한 inter-robot 통신·공유 카메라·proprio 공유도 없음** — 진짜 분산.
3. **From-scratch decentralized diffusion 대비 +64pp** (Q1), **centralized 대비도 우위** (Q3, handover 0.833 vs 0.611), **w/o weight-sharing 대비 reactivity 거의 2배** (Q2, 85% vs 45%), **3-robot 무수정 확장 90%** (Q4).
4. 파라미터/context 모두 팀 크기에 **상수**.

기술적 새로움이 백본·loss 자체에 있는 것은 아니지만 — 그것들은 π0.5 그대로다 — **"무엇을 빼면 분산 협력이 동작하는가"** 를 정밀하게 보여준 ablation-중심 시스템 페이퍼라는 점에서 의미가 크다. 향후 community-scale collaborative dataset이 모이면 이 recipe의 scaling 잠재력이 본격적으로 드러날 것으로 보인다.

<!-- VERIFIED: pdf -->
