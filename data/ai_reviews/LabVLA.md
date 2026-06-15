# LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories

> **한 줄 요약**: Qwen3-VL-4B + 18-layer DiT action expert를 (1) FAST action-token VLM 사전학습 → (2) knowledge insulation(stop-gradient) 하의 flow matching 사후학습 2단계로 훈련하고, RoboGenesis 시뮬레이션으로 합성한 LabEmbodied-Data(16개 로봇 플랫폼, atomic-skill 합성)로 미세조정한 과학 실험실 도메인 VLA. LabUtopia 6개 태스크 평균 성공률 ID 71.1% / OOD 70.0%로 π₀(63.3/63.2), GR00T N1.5(52.5/50.0), SmolVLA(52.2/53.1) 등 모든 4B 이하/3B 베이스라인을 능가.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **가정/탁상 위주 VLA**(OpenVLA, π₀, π₀.₅, SmolVLA, GR00T N1.5 등)는 Open-X, DROID, BridgeData V2 같은 household/tabletop 데이터로 학습 → 피펫, 원심분리기, 가열판, 투명 액체, 프로토콜 워크플로 같은 **실험실 인스트루먼트와 물리 상태**를 본 적 없음
- **자가구동 실험실(SDL)·LLM lab agents**(Boiko et al., Coscientist 등)는 디지털 추론은 잘하나, "비커를 들고 시약을 옮기고 버튼을 누르는" 물리 실행 단계는 여전히 사람에 의존
- **실험실 실데이터 수집**: 전용 인스트루먼트, 보정 하드웨어, 안전 절차 필요 → 수집 비용이 일반 robot data보다 압도적으로 큼

### 핵심 질문
- **시뮬레이션만으로 실험실 프로토콜 실행 VLA를 학습할 수 있는가?**
- **VLM-action loss 사이의 간섭(language/visual prior drift)을 막으면서 continuous control을 학습할 수 있는가?**
- **하나의 cross-embodiment 스키마로 single-arm, mobile-manipulator, dual-arm을 함께 다룰 수 있는가?**

📌 [Figure 1 삽입] — Qwen3-VL → DiT action expert, stop-gradient(KI), 좌측 외부 코퍼스(Robointer-VQA, AgiBot World Beta, OXE-AugE, Droid), 우측 RoboGenesis 3단계 파이프라인.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

LabVLA는 두 모듈의 조합:
- **VLM**: Qwen3-VL-4B-Instruct (d_vlm=2560). 입력은 최대 V개의 RGB 카메라 뷰 I_t^{1:V}, 언어 지시 ℓ, 로봇 상태 q_t^r
- **DiT Action Expert**: 18-layer, width 1024, 8 heads, head_dim 128. VLM prefix H_φ는 선형 투영 Π로 DiT width로 매핑되어 cross-attention 입력으로 들어감. 노이즈 액션 청크와 state는 별도 linear로 투영 후 DiT query로 concat.
- **출력**: K-step continuous action chunk A_t^r ∈ ℝ^{K×d_r}, d_r은 embodiment별 active action dim.

### 2.2 Stage 1 — FAST Token VLM Pretraining

Flow matching head를 generic VLM 위에 곧장 붙이면 prefix-action representation이 정렬되지 않아 학습이 불안정. 따라서:
- 연속 액션을 **FAST tokenizer**로 이산 토큰 z_{1:L_z}로 변환
- 시퀀스 X_pre = [v_t; c_t; y_t; z_{1:L_z}] (v_t=이미지 토큰, c_t=binning된 state 문자열+지시, y_t=annotation)
- masked next-token loss로 L_FAST 학습
- annotation target(VQA, subtask)이 있으면 가중치 λ_j로 L_CE 추가 → ℒ_VLM = ℒ_FAST + Σ λ_j ℒ_CE^{(j)}
- VQA/annotation-only 샘플은 FAST 블록 skip(전부 0인 액션으로 오염되지 않게)

### 2.3 Stage 2 — Flow Matching Posttraining

- Active GT 액션 청크 A_t^r를 d_max로 zero-pad → Ã_t^r, ε~𝒩(0,I) 샘플
- τ = 0.999·τ̃, τ̃~Beta(1.0, 1.5)
- X_τ = τ·Ã_t^r + (1-τ)·ε, U_τ = Ã_t^r − ε (target vector field)
- DiT는 V_θ = g_θ(X_τ, τ, q_t^r, Π(H_φ))를 예측
- Masked MSE 손실: 패딩 action dim과 annotation-only 샘플은 mask로 제외
- 추론 시 N=10 Euler step만으로 trajectory 도달 → diffusion policy의 수백 step보다 훨씬 빠름

### 2.4 Knowledge Insulation (KI) — 핵심 트릭

- 문제: flow matching gradient가 VLM에 직접 들어오면 언어/시각 prior가 drift → 라벨링/그라운딩 성능 저하
- 해결: VLM hidden state slice_p(f_φ(X_KI))에 **stop-gradient** 적용 → H̃_φ,p^{KI} = sg(H_φ,p^{KI})
- DiT는 분리된 prefix만 받음: V_θ^{KI} = g_θ(X_τ, τ, q_t^r, Π(H̃_φ,p^{KI}))
- FAST loss와 annotation CE loss는 **여전히 VLM을 업데이트** → token-level supervision은 유지, velocity-space gradient만 차단
- 결합 손실: ℒ_KI = α·ℒ_FM + ℒ_FAST + Σ λ_j ℒ_CE^{(j)}, **α=10**
- 추론 시 FAST/annotation head는 제거 (training-only)

> ❓ **예상 질문**: KI 없이 그냥 joint training하면 왜 안 되는가?
> **답변**: 저자가 §3.3에서 명시 — co-training the VLM directly with flow loss "made the prefix representations less reliable for downstream attention". 즉 instrument 이름 grounding이나 protocol step 식별 같은 cross-attention 의존 능력이 손상됨. KI는 prefix 표현은 token loss로만 학습되도록 보장.

---

## 3. 데이터 전략

### Pretraining 데이터
- **Robointer-VQA**: VQA + subtask annotation
- **AgiBot World Beta**: 대규모 real-robot 데이터
- **OXE-AugE** (LeRobot 포맷 subset만 사용, OXE 원본의 6개 소스 데이터셋 병합, ~572k trajectories)
- **Droid**: cross-embodiment manipulation

### Posttraining 데이터
- **OXE-AugE** (계속) + **LabEmbodied-Data** (RoboGenesis 합성)
- LabEmbodied-Data는 4개 task family 커버: single-arm primitives, multistep lab procedures, bimanual, mobile manipulator

### RoboGenesis 데이터 엔진 (핵심 기여)

3단계 파이프라인:
1. **Environment Building**: text-to-image + TRELLIS 2.0 reconstruction으로 3D asset library 구성, validated 실험실 scene 조립
2. **Agentic Workflow Generation**: 자연어 instruction → ordered atomic skill 시퀀스 → 10+ robot platforms × 6-axis domain randomization으로 instantiate
3. **Structured Export**: success filtering 후 15종 annotation stream(collision events, temporal segments, subgoals, quality scores, intervention flags 등)

Table 1 비교에서 RoboGenesis만 **9개 feature column 전부**(generative asset, agentic scene/task, randomization, success filter, structured annotation, long-horizon composition, lab protocol, 16 robot platforms) 충족. RoboTwin 2.0(5), RoboCasa 365(1), ManiSkill 3(23), RLBench(5), RoboGen(6)은 부분만.

> ❓ **예상 질문**: 시뮬레이션만으로 학습한 정책이 실 실험실에서 의미 있는 성능을 낼 수 있나?
> **답변**: 저자는 InternData-A1(Tian et al., 2025) 선행연구를 인용 — 거의 100% 합성 데이터로 학습한 정책이 실 로봇에서 π₀과 동등 성능. §5.2 Franka 실험에서 LabVLA 4-task 평균 ID-clean 86.5%, OOD-cluttered 74.0%로 sim-to-real transfer 확인.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM Backbone | Qwen3-VL-4B-Instruct, d_vlm=2560 |
| Action Expert | 18-layer DiT, width 1024, 8 heads, head_dim 128 |
| Action chunk | K-step continuous, padding to d_max로 cross-embodiment 배치 |
| FAST pretrain loss | masked next-token + annotation CE |
| FM posttrain loss | masked MSE on velocity field, α=10 |
| KI | stop-gradient on VLM prefix → DiT |
| Sampling | Euler N=10 step (diffusion 수백 step 대비 빠름) |
| Cameras | 최대 V개 view, missing slot은 dummy + zero attention mask |
| Pretrain action target | absolute action |
| LabUtopia finetune target | delta action |
| Hardware | 본문에 명시 X (Appendix A 참조) |

---

## 5. 실험 설계 및 평가 프로토콜

### LabUtopia (메인 벤치마크)
- 6개 lab operation: Pick Up, Press Button, Open Door, Pour Liquid, Heat Beaker, Transport Beaker
- ID/OOD 두 split (OOD는 object placement, appearance, scene configuration perturb)
- **120 episodes per task per setting**

### Baselines (3 families)
| Family | 모델 | 사이즈 |
|--------|------|-------|
| <1B | SmolVLA, X-VLA | <1B |
| 3B | GR00T N1.5, π₀, π₀.₅, π₀-FAST, InternVLA-A1 | 3B |
| 4B | Wall-oss-flow, **LabVLA** | 4B |

모든 baseline은 public checkpoint를 LabUtopia harness에 맞춰 action/state schema만 조정.

### Real-world (§5.2)
- Franka platform, 4 task (Shake Liquid, Pour Liquid, Magnetic Stir, Stopper Plug/Unplug)
- 각 task당 30-50 demonstrations, 5×5 cm randomization region
- 4 condition: {ID/OOD} × {clean/cluttered}, condition당 50 rollouts
- 비교: LabVLA vs DreamZero vs π₀.₅

---

## 6. 실험 결과 심층 분석

### LabUtopia ID (Table 2 핵심 숫자)

| Method | Size | Pick Up | Press Btn | Open Door | Pour Liq | Heat Beaker | Trans Beaker | **Avg** |
|--------|------|---------|----------|-----------|----------|-------------|--------------|---------|
| SmolVLA | <1B | 15.8 | 97.5 | 16.7 | 0.8 | 96.7 | 85.8 | 52.2 |
| X-VLA | <1B | 27.5 | 98.3 | 65.0 | 45.0 | 25.8 | 83.3 | 57.5 |
| GR00T N1.5 | 3B | 40.8 | 99.2 | 6.7 | 0 | **99.2** | 69.2 | 52.5 |
| π₀ | 3B | 21.7 | 92.5 | 51.6 | 37.5 | 90.0 | 86.7 | 63.3 |
| π₀.₅ | 3B | 38.3 | 60.0 | 55.8 | 29.2 | 40.8 | **90.0** | 52.4 |
| π₀-FAST | 3B | 16.7 | 37.5 | 17.5 | 5.8 | 3.3 | 20.8 | 16.9 |
| InternVLA-A1 | 3B | 25.8 | 93.3 | 38.3 | 2.5 | 82.5 | 67.5 | 51.7 |
| Wall-oss-flow | 4B | 11.7 | 54.2 | 0.83 | 0 | 0 | 29.2 | 16.0 |
| **LabVLA** | **4B** | **49.2** | **100** | **65.0** | **43.3** | 83.3 | 85.8 | **71.1** |

### LabUtopia OOD

| Method | Pick Up | Press Btn | Open Door | Pour Liq | Heat Beaker | Trans Beaker | **Avg** |
|--------|---------|----------|-----------|----------|-------------|--------------|---------|
| π₀ (2nd best) | 19.2 | 89.1 | 53.3 | 38.3 | 90.8 | 88.3 | 63.2 |
| **LabVLA** | **48.3** | 98.3 | **65.8** | **34.2** | 87.5 | 85.8 | **70.0** |

- LabVLA가 ID 평균 71.1%로 차상위 π₀(63.3%) 대비 **+7.8pp**, OOD 70.0%로 +6.8pp 우위
- **ID→OOD drop이 1.1pp에 불과** (71.1→70.0) — 도메인 랜덤화의 visual/spatial invariance 효과
- **Pour Liquid는 모두에게 hardest** — 어떤 baseline도 50% 미달, liquid surface tracking 미해결
- LabVLA는 모든 task에서 48%+ (Pour Liquid 34.2% 제외) — **balance가 강점**
- SmolVLA처럼 Heat Beaker 98.3%이면서 Pour Liquid 1.67%처럼 spike 패턴 없음

### LabEmbodied-Data 전이성 (Table 3, §5.1)

X-VLA를 LabEmbodied-Data로 finetune → 5개 non-saturated task 평균:
- ID: 49.3% → 64.3% (**+15.0pp**)
- OOD: 43.7% → 63.0% (**+19.3pp**)
- 최대 향상: Heat Beaker ID 25.8→68.3, Pour Liquid OOD 25.0→65.0
- → 데이터셋이 LabVLA 아키텍처 외에도 효과적임을 증명

### Real-world Franka (Table 4)

| Condition | LabVLA | DreamZero | π₀.₅ |
|-----------|--------|-----------|------|
| ID, clean | 86.5 | **87.0** | 85.0 |
| ID, cluttered | 80.0 | **81.0** | 76.5 |
| OOD, clean | **80.0** | 78.0 | 77.0 |
| OOD, cluttered | 74.0 | **75.5** | 71.5 |

- 4개 평균에서 LabVLA와 DreamZero가 within run-to-run variance
- LabVLA가 OOD-clean에서 단독 1위 (80.0)
- π₀.₅은 OOD cluttered에서 71.5%로 가장 큰 drop
- **Pour Liquid가 모든 정책에서 가장 민감** (LabVLA 86→72, DreamZero 88→70)

> ❓ **예상 질문**: 시뮬레이션 평균 71%, 실험실 평균 80%인데 시뮬레이션 점수가 더 낮은 이유?
> **답변**: LabUtopia 6태스크 중 Pour Liquid(34-43%)가 평균을 끌어내리는 outlier. 실 Franka 4태스크는 Pour Liquid가 단순 형태(pick→pour→place)고 평가도 50 rollout/condition으로 episode 수가 적어 variance 큼. 두 숫자는 직접 비교 불가.

---

## 7. Ablation 분석

논문 §4 메인에서 별도 ablation table은 두 개:

### Table 3 — LabEmbodied-Data 추가 효과 (X-VLA 위에서)
- baseline X-VLA: ID 49.3, OOD 43.7
- + LabEmbodied: ID 64.3 (+15.0), OOD 63.0 (+19.3)
- → 데이터 자체의 가치 분리 입증

### Table 4 — Real-world sim-to-real 검증
- DreamZero / π₀.₅ 대비 비등하거나 우위 → KI + FM 레시피가 실 하드웨어에 transfer

### 누락된 ablation (논문 한계)
- KI on/off (stop-gradient를 제거하면 얼마나 떨어지는가?)
- FAST pretraining stage skip 효과
- N=10 Euler step의 sensitivity (5/20/50)
- α=10 가중치의 sensitivity
- DiT 크기(18-layer × 1024)의 scaling

> ❓ **예상 질문**: KI가 정말 효과 있다는 직접 증거가 있나?
> **답변**: 본문에는 정성적 진술("prefix representations less reliable")만 있고 정량적 KI on/off ablation은 메인에 없음 — Appendix에 있을 가능성. **약점**.

---

## 8. 관련 연구 비교

### vs. Mainstream VLAs (π₀ 등)
- **공통점**: flow matching + DiT-style action expert (π₀, π₀.₅)
- **차이**: (1) Qwen3-VL backbone (vs PaliGemma/Llama), (2) FAST 사전학습으로 VLM이 action-aware 상태에서 flow matching 시작, (3) KI(stop-gradient)로 prefix 보호, (4) **데이터 분포가 실험실 도메인**

### vs. Synthetic-data VLAs (InternData-A1 등)
- 공통점: 시뮬레이션-주도 데이터 합성으로 real-collection 비용 회피
- 차이: RoboGenesis는 atomic skill composition을 통한 **long-horizon protocol 생성**과 **16개 embodiment** 지원 — 다른 엔진(RoboTwin 2.0 5개, ManiSkill 3 23개지만 자동 task generation 없음)보다 chemistry/biology protocol에 특화

### vs. Reasoning-augmented VLAs (CoT-VLA, ThinkAct 등)
- LabVLA는 **명시적 reasoning trace 없음** — instruction + visual prefix → flow matching이 trajectory를 직접 생성
- 대신 protocol-conditioning은 **데이터셋의 annotation stream**(subgoals, temporal segments, quality scores)에 의존

### 4-tier Capability Pyramid (§8 Discussion, Figure 6)
저자가 직접 제안:
- **Level 1 (Apprentice)**: single-step labware 상호작용
- **Level 2 (Technician)**: 다단계 written protocol 실행 (LabVLA의 현재 위치)
- **Level 3 (Specialist)**: 정밀 인스트루먼트(피펫, 원심분리기, thermal cycler) + 측정 로깅
- **Level 4 (Scientist)**: 관측 기반 protocol 수정, 분기, 종료 판단

LabVLA는 Level 2 — 고정 프로토콜 실행만 가능. Level 3-4에는 instrument competence, measurement awareness, scientific judgment 부족.

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **시뮬레이션 의존**: LabUtopia는 high-fidelity simulation이나 실 wet lab과의 격차는 여전히 큼. Real-world 실험은 4태스크 / Franka 1대 / 50 rollouts에 그침.
2. **Protocol following only**: 주어진 고정 절차 실행에 한정. 측정값 기반 분기, 절차 수정, 종료 판단 부재. Level 2 한계.
3. **Pour Liquid 미해결**: 모든 baseline이 50% 미달. Liquid surface feedback, 정밀 tilt angle control 부재가 원인. 화학 실험의 핵심 동작이라 critical.
4. **Hardware 미보고 (메인)**: VLM pretraining, KI posttraining, LabUtopia finetuning 각각의 GPU 수/시간 메인에 명시 X.
5. **Safety/contamination**: 본문에서 명시적으로 인정 — "hardware drift, reagent variability, safety constraints, contamination risks, unanticipated failure modes that no sanitized benchmark captures".
6. **KI ablation 부재**: stop-gradient의 정량적 효과가 메인에 없음.
7. **단일 backbone**: Qwen3-VL-4B에서만 검증. 다른 backbone에 plug-and-play 가능성 미검증.

### Attribution 문제
- LabUtopia +7.8pp 우위가 (a) Qwen3-VL backbone vs PaliGemma/Llama, (b) LabEmbodied-Data, (c) FAST pretraining, (d) KI, (e) DiT 구성 중 어디서 오는지 분리 불완전. Table 3가 (b)만 부분적으로 isolate.
- π₀-FAST가 LabUtopia에서 16.9%로 폭락한 것은 FAST tokenization이 실험실 도메인 OOD라는 의미 — LabVLA가 FAST pretraining 후 flow matching으로 넘어간 hybrid 전략의 정당화는 되지만, FAST-only 단독 평가가 좀 더 필요.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA를 과학 실험실 도메인으로 본격 확장한 첫 시도. 데이터 엔진(RoboGenesis)과 학습 레시피(FAST + KI flow matching) 양쪽에서 새 기여 |
| **Technical depth** | ★★★★☆ — KI 수식, FAST → FM 두 단계 분리, embodiment-agnostic batch 포맷 등 시스템 디자인이 견고 |
| **Experimental rigor** | ★★★☆☆ — LabUtopia 8개 baseline 비교는 강력하나 KI ablation 부재, 실 로봇 4태스크/50 rollouts로 통계 약함 |
| **Practical impact** | ★★★★☆ — 과학 자율화의 "마지막 1m" 문제에 직접 대응. RoboGenesis 자체가 재사용 가능한 인프라 기여 |
| **Writing quality** | ★★★★☆ — Capability pyramid, Table 1의 데이터 엔진 비교, stratified result 해석 등 구조적 |

**강점**: 실험실이라는 새 도메인 정의, 그에 맞는 데이터 엔진 + 학습 레시피의 **end-to-end 패키지**. KI는 VLM-action 간섭 문제에 깔끔한 해법. 모든 평가 baseline(SmolVLA부터 GR00T N1.5까지)을 LabUtopia 평균에서 능가하면서 OOD drop 1.1pp만으로 가장 generalize.
**약점**: 시뮬레이션 의존, Pour Liquid 미해결(34%), Level 2 한계, KI/FAST ablation 부족, code/model release 미명시.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | π₀이 이미 ID 63.3%인데 LabVLA +7.8pp가 데이터 덕인지 모델 덕인지? | Table 3에서 X-VLA + LabEmbodied가 +15pp 향상 — 데이터 기여만으로도 큼. 다만 Qwen3-VL backbone과 KI의 분리된 ablation은 부재 |
| 2 | KI(stop-gradient)가 정말 필요한가? joint training이면 안 되는가? | §3.3에서 정성적으로 "prefix가 덜 reliable해진다"고 명시. 다만 정량 ablation은 메인에 없음 — Appendix 또는 후속 작업 필요 |
| 3 | Pour Liquid 34%는 어떻게 풀 것인가? | 액체 표면 추적 feedback 부재가 근본 원인. force/torque sensor, 액체 시뮬레이션 fidelity, 또는 시각 surface segmentation 추가 필요 |
| 4 | 시뮬레이션→실 hardware transfer가 정말 보장되나? | Table 4에서 LabVLA 평균 80% — DreamZero/π₀.₅과 비등 또는 우위. 다만 4태스크 / 1 Franka / 50 rollouts라 통계 약함 |
| 5 | Wall-oss-flow 같은 4B 동급 모델이 LabUtopia에서 16%로 폭락한 이유? | Public checkpoint를 그대로 평가했기 때문. 실험실 도메인 OOD에서 일반 VLA가 망가짐을 보여주는 사례 — LabEmbodied-Data로 finetune했다면 결과 다를 것 |
| 6 | 16개 robot platform이라지만 evaluation은 LabUtopia 하나로 좁아진 것 아닌가? | 맞음. RoboGenesis는 16개 embodiment 합성을 지원하나, LabUtopia harness가 정의한 single Franka-style robot 평가에 한정. 다양한 embodiment 평가는 향후 과제 |
| 7 | FAST → flow matching 두 단계가 필요한 이유? End-to-end 한 번에 못 하나? | §3.1에서 명시 — flow matching head를 generic VLM에 곧장 붙이면 prefix와 액션 표현이 정렬되지 않아 unstable. FAST 토큰 next-token loss로 prefix를 먼저 action-aware로 만듦 |
| 8 | N=10 Euler step이 어떻게 가능한가? Diffusion policy는 수백 step 필요한데 | Flow matching의 deterministic vector field 특성 — Lipman et al. 2022. straight-path supervision으로 적은 step에서 수렴. Sampling sensitivity table은 본문 외 |
| 9 | Qwen3-VL-4B의 선택이 결정적이었나? | Backbone ablation 없음. d_vlm=2560이라는 단순 사실만 보고됨. 다른 VLM(PaliGemma 등)으로 동일 레시피를 적용한 비교는 약점 |
| 10 | Real lab safety/contamination 문제를 어떻게 풀 것인가? | 저자도 §Limitations에서 인정 — 본 논문은 "early study". 다음 단계는 reagent variability, safety constraints가 있는 working lab 배치 |
| 11 | Capability pyramid의 Level 2→3 점프는 어떻게 가능한가? | 정밀 인스트루먼트(피펫 등) 모델링, measurement logging, multi-modal sensor integration 필요. 현재 RoboGenesis는 atomic skill composition까지만 — Level 3은 시뮬레이션-한계 가능성 |
| 12 | Open source인가? | 본문 첫 페이지에 "Code, Model" 링크 placeholder 있으나 메인 텍스트에 명시적 release 약속 X. 보수적으로 closed source로 표기 |

<!-- VERIFIED: pdf -->
