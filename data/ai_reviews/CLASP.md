# CLASP: Language-Driven Robot Skill Selection and Composition using Task-Parameterized Learning

> **한 줄 요약**: 사전학습 VLM(Qwen3-VL-32B)을 fine-tuning 없이 task-parameterized KMP와 결합하여, 2-5회 kinesthetic demonstration만으로 자연어 지시 → skill 선택 / covariance-weighted trajectory 합성 / capability gap 자동 감지 후 데모 요청까지 수행하는 모듈형 프레임워크. 7-DoF 실로봇에서 73.3-100% 성공률, π0.5(50 demos) 대비 4 demos만으로 동등 이상 성능.

---

## 1. 배경 및 동기

- 산업 조립 환경 로봇 프로그래밍은 (i) 비전문가도 직관적으로 쓸 수 있어야 하고 (ii) 데이터 효율적이어야 한다.
- End-to-end VLA(π0.5 [3,4,5,32])는 자연어 인터페이스를 제공하지만 수만~수십만 demonstration과 H200 18시간 fine-tuning이 필요하다.
- 반대로 Task-Parameterized Imitation Learning(TP-IL, TP-GMM/TP-KMP [6,7,8])은 skill 당 2-5 demo로 충분하지만 **자연어 grounding이 없어** 운영자가 어떤 skill을 실행할지 코드로 직접 지정해야 한다.
- 본 논문은 TP-KMP의 데이터 효율성 + VLM의 자연어/시각 추론을 모듈형으로 결합하여 두 진영의 강점을 모두 잡는다는 동기.

---

## 2. 방법론 심층 분석

### 2.1 프레임워크 개요 (Sec. 3.1)
- VLM: **Qwen3-VL-32B-Instruct** (로컬 배포, API 없이 산업 데이터 프라이버시 충족). Pixtral, GPT 등으로 교체 가능한 모듈형 설계.
- 세 가지 컴포넌트: (1) Perception — 6D pose & semantic label, (2) VLM Reasoning — 스키마 생성 / skill 선택 / parameter binding, (3) Trajectory Generation — TP-KMP.
- 두 단계 동작: **Learning Phase**(스키마 생성 + TP-KMP 학습)와 **Execution Phase**(자연어 지시 처리).

### 2.2 Execution Decision Tree (Sec. 3.2)
1. **Skill Matching**: 사용자 명령 + workspace 이미지 → VLM이 모든 skill 스키마를 tool definition으로 받아 tool call로 skill 선택 및 detected object 중 P개를 reference frame으로 binding.
2. **Composition Feasibility**: 단일 skill 매치 실패 시 Eq. (1) 호환성 검사로 두 skill 조합 가능 여부 수학적으로 판정(VLM이 아님).
3. **Active Skill Acquisition**: 둘 다 실패 시 demo 요청 생성.

### 2.3 Learning Phase (Sec. 3.3, Algorithm 1)
- Skill 정의: (Θ, ϕ). Θ = {Θ^(p)}_{p=1}^P 는 P개 local KMP, ϕ는 VLM 생성 JSON 스키마.
- Demonstration: M=2-5회 kinesthetic, 100 Hz로 (s, ξ) 기록 (ξ = 6D EE pose + gripper).
- Perception 출력 E = {(b_i, A_i), d_i, ℓ_i}: 6D pose, bbox 크기, semantic label.
- TP-KMP 학습: P개 task-relevant object를 reference frame으로 선택 → demonstration을 각 frame local coordinate으로 투영 → frame당 KMP Θ^(p) = {s_n, μ_n, Σ_n} 학습. Frame-relative encoding으로 새 object 위치에 일반화.
- **Schema Generation**: VLM이 이미지+검출 객체만 보고 (trajectory 없이!) JSON tool definition을 생성. 필요한 객체 종류·개수·semantic label("grasp","pour") 자동 추출. Self-verification pass로 mislabel 보정.

### 2.4 Skill Composition (Sec. 3.4, Algorithm 2)
- VLM이 task semantics로 후보 skill A, B와 각 skill의 frame index p_A, p_B를 선택.
- 새 TP-KMP를 만들기 위해 (Θ^{p_A}, Θ^{p_B})를 **product-of-Gaussians**(Eq. 4)로 fusion.
- 핵심 제한: **Compatibility Constraint (Eq. 1)** — phase 변수 s가 P=2개의 non-overlapping region G_1, G_2로 분할되어 각 region에서 한 KMP가 dominant해야 함:
  ∀q ≠ π(j), ∀o: σ̃^{(q)}_{o,t} > σ̃^{(π(j))}_{o,t} + τ, with τ = 0.01.
  즉 한 시점에서 한 KMP의 uncertainty가 다른 것보다 명확히 낮아야 fusion이 coherent하다. 둘 다 high/low variance면 거부.

### 2.5 Active Skill Acquisition (Sec. 3.5)
- VLM이 capability gap을 인지하면 "I don't have a skill for inserting. Please demonstrate ..." 같은 자연어 demo 요청을 자동 생성.
- 사용자 확인 → Algorithm 1으로 즉시 학습 → S ← S ∪ {(Θ, ϕ)}.

---

## 3. 데이터 전략

- Motion skill: skill당 2-5회 kinesthetic demonstration이 전부. **No web-scale pretraining for actions.**
- Perception(6D pose estimator): 객체 class당 1회의 BlenderProc 합성 데이터 학습 필요. Demo 카운트에는 포함되지 않으나, "data-efficient"를 평가할 때 정직히 명시한 비용.
- VLM은 **frozen**. 어떤 task-specific fine-tuning도 없음. 모든 task 적응은 (i) JSON 스키마 생성과 (ii) tool calling으로만 달성.
- 실험에서는 5개 YCB object(grasp 학습용) + bearing ring / measurement station(산업용 high-precision 시나리오).

---

## 4. 시스템/학습 세부사항

- 로봇: 7-DoF torque-controlled DLR cobot + Robotiq gripper + 정적 측면 RGB-D 카메라.
- 제어: Cartesian impedance(K_p=750 N/m, 250 Nm/rad), K_d는 configuration-dependent로 동적 계산.
- TP-KMP 하이퍼파라미터: [8]을 따름. CPU 초 단위 학습.
- VLM tool-calling [25,26]로 skill 스키마들을 동시에 prompt에 전달. 출력은 skill id + frame binding.
- 합성된 skill도 표준 TP-KMP이므로 원리상 incremental chaining 가능(현재는 pairwise만 검증).

---

## 5. 실험 설계 및 평가 프로토콜

평가 차원 4개 (Table 1):

| Evaluation | Trials | 설명 |
|------------|--------|------|
| Object Generalization (4.2) | 44 | 5 YCB 객체에서 학습된 pick-and-place skill을 다른 객체로 전이 (5×4 pairwise + α) |
| Pose Generalization (4.2) | 29+29 | pick-and-pour, 15+ spatial config, vision pose vs manual pose |
| Skill Composition (4.3) | 16 | pick(4) × place(4) 조합, 두 base skill 합성 |
| Active Skill Acq. (4.3) | 20 + 15 | (i) capability gap 감지 20회, (ii) ring insertion 15 config |

추가로 동일 로봇 위에서 **π0.5 VLA** [32]를 5 / 50 demo로 fine-tuning하여 직접 비교(Table 2).

---

## 6. 실험 결과 심층 분석

### Object Generalization (Sec. 4.2)
- **90.9% (40/44)**. 5종 YCB에서 size/shape/material이 달라도 일반화. 실패 4회는 (a) 학습 object보다 작은 bearing ring에서 grasp offset 무효, (b) cracker box 변형 등 물리적 한계.

### Pose Generalization (Sec. 4.2)
- Vision pose: **79.3% (23/29)**, Manual pose: **100% (29/29)**.
- 실패 원인은 전부 카메라 occlusion에 의한 perception 오류. Skill 자체 일반화는 robust.

### Skill Composition (Sec. 4.3)
- **100% (16/16)** — grasp_apple→place_on_plate 와 grasp_potted_meat→pour_into_cracker_box 두 base skill에서 grasp phase + placement phase를 fusion. 4×4 조합 전부 성공.

### Active Skill Acquisition (Sec. 4.3)
- Capability Gap Detection: **95% (19/20)**. 1건은 pick-and-place skill로 잘못 시도한 false negative.
- Newly Acquired ring insertion: **73.3% (11/15)**. 실패 4건은 전부 180° 회전 config — 학습한 trajectory와 정반대 방향이 요구되어 발생.

### π0.5 비교 (Sec. 4.4, Table 2)
| | Ours (4 demos) | π0.5 (5 demos) | π0.5 (50 demos) |
|---|---|---|---|
| Object Gen | **90.9%** | 0% | 86.4% |
| Pose Gen (vision) | **79.3%** | 0% | 79.3% |
| Pose Gen (manual) | **100%** | — | — |

같은 5 demo 예산에서 π0.5는 0% autonomous success. 50 demo + 18h H200 fine-tuning 후에야 비슷한 수준 도달.

---

## 7. Ablation 분석

본 논문은 별도 ablation table을 두지 않고 대신:
- **Schema self-verification pass** 유무를 비교(Sec. 3.3): VLM이 trajectory를 보지 않고 scene만으로 label을 생성하므로 self-verification이 mislabel을 줄임.
- **Manual vs vision pose**(Sec. 4.2): perception을 제거하면 79.3 → 100%, perception이 유일한 병목임을 격리.
- **Compatibility constraint τ** = 0.01: Fig. 3에서 (a) 호환(보완 분산), (b)(c) 비호환(둘 다 high or low variance) 케이스를 보여 fusion 거부 동작 시각화.
- **VLM 교체 가능성**(Sec. 3.1): TP-KMP backend를 건드리지 않고 Pixtral, GPT 등으로 swap 가능하다고 명시(정량 비교는 없음 — 한계).

---

## 8. 관련 연구 비교

- **TP-IL 계열** [6,7,8,10]: 데이터 효율적이나 자연어 부재 → CLASP가 VLM tool-calling으로 채움.
- **End-to-end VLA** [3,4,5,14,32]: 자연어 가능하나 demo·compute 비용 과대 → 직접 비교 Table 2에서 4 vs 50 demos 격차 입증.
- **Symbolic skill grounding** [15,16,17]: 이산 primitive 시퀀싱만 가능 → CLASP는 trajectory-level fusion으로 새로운 연속 동작 생성.
- **Probabilistic skill blending** [18,19,20] (products of Gaussians): 자연어 grounding 부재 + composition refusal 기준 부재 → CLASP는 Eq. (1)로 형식적 거부 기준 제공.
- **저자들이 강조하는 gap**: "natural language grounding + trajectory-level composition + formal compatibility + 2-5 demo"를 동시에 충족하는 framework는 본 연구가 최초(Sec. 2 "Gap in Literature", Table 3).

---

## 9. 한계 및 미해결 문제

1. **Pairwise 합성에 국한**: 3개 이상 skill의 incremental chaining은 원리상 가능하다고만 언급, 검증은 미실시.
2. **Static perception 가정**: skill 시작 시점에 객체 포즈가 고정되어야 하며 occlusion에 취약(79.3 → 100% gap).
3. **6D pose estimator의 per-class 학습 비용**: BlenderProc 합성 데이터로 객체 class마다 학습 필요. "2-5 demo"라는 marketing 수치가 perception cost를 가린다는 점을 저자도 인정.
4. **180° 회전 등 motion이 반전되는 config 실패** (ring insertion 4/4 failures): TP-KMP가 학습된 방향성에서 벗어난 회전을 표현하지 못함.
5. **VLM swap 정량 비교 부재**: Qwen3-VL-32B 외 다른 VLM(Pixtral/GPT)을 실제로 평가하지 않음.
6. **단일 카메라 의존**: 다중 시점·active perception이 향후 개선 방향(Sec. A.12).
7. **Open-source 미공개**: code_url 없음 — 재현성에 한계.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLM tool-calling으로 skill 스키마를 자동 생성하고, product-of-Gaussians fusion에 **formal compatibility constraint(Eq. 1)** 라는 명시적 거부 기준을 추가한 부분이 깔끔 |
| **Practical impact** | ★★★★★ — 5 demo vs 50 demo, 초 단위 CPU 학습 vs 18h H200, 100% manual-pose 성공 — 산업 셀에 즉시 투입 가능한 ROI |
| **Rigor** | ★★★★☆ — π0.5와 **동일 로봇·동일 벤치**에서 비교, capability gap detection 95% 등 진솔한 수치 보고 |
| **Generality** | ★★★☆☆ — pairwise composition, single camera, per-class pose model 등 산업 셀 가정에 강하게 의존 |

CLASP는 "거대 VLA만이 자연어 로봇의 미래"라는 흐름에 정직한 반론을 제기한다. **frozen VLM + 고전 movement primitive + formal fusion 기준**이라는 보수적 조합이, fine-tuning 0회로 π0.5(50 demo)와 같은 성능을 내고 일부 시나리오(manual pose 100%, composition 100%)에선 능가한다. 특히 Eq. (1)의 compatibility constraint는 product-of-Gaussians fusion에 만연한 "averaged 무의미 trajectory" 문제를 형식적으로 차단한 점에서 이론적 기여도 있다.

---

## 11. 적용 가능성 및 향후 방향

- **즉시 가능한 산업 응용**: 알려진 객체 셋이 안정적인 조립 셀(부품 조립, 베어링 삽입, 키팅) — 객체는 적고 자주 안 바뀌지만 task는 자주 바뀌는 환경에 최적.
- **확장 방향**:
  1. 3개 이상 skill의 sequential composition + Eq. (1)을 chain-level로 확장.
  2. Active perception / multi-view로 79.3 → 100% 갭 해소.
  3. Single-image 3D reconstruction [34]으로 per-class pose model 비용 절감.
  4. VLM swap의 정량 비교(특히 더 작은 7B급 모델로 32B를 대체 가능한지).
  5. Composition 시 VLM이 선택한 frame index의 정당성(왜 p_A를 골랐는지)을 사용자에게 설명하는 explainability 모듈.

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "4 demo로 90.9%"는 perception을 BlenderProc로 클래스당 학습시킨 비용을 숨긴 것 아닌가? | 저자도 Sec. 5에서 인정. 단, 산업 환경은 "객체는 고정 / task만 변동"이므로 perception 비용은 long-term deployment에서 amortize된다고 주장. 단일 이미지 3D 재구성[34]으로 비용 절감 방향도 명시. |
| 2 | Eq. (1) compatibility constraint가 너무 엄격해서 실제로 거의 모든 composition을 거부하지 않는가? | Fig. 3 (a) 같은 complementary variance 케이스만 통과. 그러나 TP-KMP는 frame 별로 자연스럽게 phase의 일부 구간에서만 dominant하므로(예: grasp phase는 picking-frame, place phase는 placing-frame), 실험상 16/16 composition이 모두 통과한 결과(Sec. 4.3). 즉 phase가 잘 정의된 manipulation에서는 보수적이지 않다. |
| 3 | VLM이 trajectory를 보지 않고 schema label을 만들면 잘못된 의미 label(e.g. 실제는 push인데 grasp으로 라벨)이 합성을 오염시키지 않나? | 인정된 위험. 저자들은 (i) self-verification pass와 (ii) execution-time tool-calling이 multiple skill을 동시에 prompt에 노출하므로 mislabel skill은 자연어 매치 단계에서 걸러진다고 주장. 정량 평가는 부족. |
| 4 | π0.5 50 demo vs Ours 4 demo는 공정한가? π0.5는 cross-task transfer 가능성이 있는 반면, CLASP는 task별 demo가 필요하다 | 같은 robot 같은 task에서 4 vs 50 비교는 공정. 다만 π0.5는 새로운 객체 카테고리에 zero-shot으로 일반화 시도하는 반면, CLASP는 perception per-class 학습이 필요 — Sec. 5에서 이 trade-off를 솔직히 명시. |
| 5 | 180° rotated bearing ring에서 4/4 실패한 것은 fundamental limitation인가? | 그렇다. TP-KMP는 demonstrated trajectory의 방향성을 그대로 학습. 정반대 회전은 새로운 skill로 봐야 하며, 이때 active acquisition으로 추가 demo 1세트를 받으면 해결됨 — 즉 framework 차원의 fallback은 존재. |
| 6 | Qwen3-VL-32B는 로컬에서도 무거운데 실제 cycle time에 영향은? | Cycle time 측정은 본 논문 범위 밖. 다만 Skill matching/composition은 task 당 1회 VLM 호출이고 trajectory 실행은 TP-KMP가 담당하므로 closed-loop control은 VLM 부담이 없음. End-to-end VLA(매 step VLM forward) 대비 본질적 장점. |
| 7 | Composition을 chaining(>2 skills)으로 확장하면 Eq. (1)이 어떻게 일반화되는가? | 저자들은 Sec. 5에서 "composed TP-KMP는 표준 TP-KMP이므로 incremental chaining 가능"이라고만 언급. P=3에서는 phase domain을 3 region으로 partition + 모든 pairwise에 대해 Eq. (1)을 적용하는 자연스러운 확장이 가능하나, 실험 미실시. |
| 8 | Symbolic planner(예: SayCan)과 비교하면? | 본 논문은 symbolic composition은 "discrete primitive sequencing"으로 분류하며 trajectory-level continuous fusion이 본질적으로 다르다고 주장(Sec. 2). 새로운 trajectory profile을 만들 수 있다는 점이 차별점. |

<!-- VERIFIED: pdf -->
