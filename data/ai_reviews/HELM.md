# HELM: Harness-Enhanced Long-horizon Memory for VLA Manipulation

## 1. 배경 및 동기

- VLA 모델은 short-horizon에서 강하지만 long-horizon task에서 체계적으로 실패한다. OpenVLA는 LIBERO-SPATIAL(평균 2.3 subgoal)에서 91.2%를 내지만, LIBERO-LONG(평균 5.8 subgoal)에서는 58.4%로 32.8pp 하락한다(§1).
- 저자들은 context window를 H=8에서 H=32로 늘려도 TSR이 63.8%에 그친다는 것을 관찰하였고, 단순히 context 길이를 늘리는 것으로는 구조적 gap이 해소되지 않음을 보인다(Table 1).
- 200 LIBERO-LONG 롤아웃을 annotate(κ=0.81)한 결과 long-horizon 실패가 세 가지 구조적 mode로 분해됨을 확인한다: FM memory gap(41%), FV verification gap(33%), FR recovery gap(26%)(§3, Appendix A.1).
- 기존의 Reflexion, Inner Monologue, Code-as-Policies 등은 post-hoc reflection이나 hand-crafted detector에 의존하며, pre-execution 단계에서 memory를 활용해 실패를 예측하는 메커니즘은 없다.
- HELM은 frozen VLA backbone을 감싸는 model-agnostic한 실행 루프 wrapper로서 세 gap 각각에 전용 컴포넌트를 매핑한다.

## 2. 방법론

- **Episodic Memory Module (EMM)** (§4.2): CLIP ViT-B/32 (512-d, L2-normalized)로 인덱싱된 key-value store. (observation, 완료 subgoal, status, timestep, state delta)를 저장. Write trigger는 (a) subgoal 완료, (b) 실패 탐지, (c) 매 ∆c=20 step. Retrieval은 cosine similarity top-k (k=3, Eq. 1). |M|>Nmax=50이면 subgoal당 최근 checkpoint만 유지.
- **State Verifier (SV)** (§4.3): 3-layer MLP [1024→512→256→1] (ReLU, dropout 0.1). 입력은 memory-augmented observation ôt=[φ(ot); k_top-1]∈R^1024 (Eq. 2), 투영된 action Wa·at∈R^256, subgoal embedding φ_text(gt)∈R^256. Pre-execution failure probability p_fail 예측. 학습: 50K (ot, at, gt, yt) triple, yt=1 if failure within 5 steps, BCE with positive weight 4.0, Adam lr 1e-4, batch 256, ~2h on one A100.
- **Harness Controller (HC)** (§4.4): subgoal stack S와 completion detector(SV와 동일 구조)를 관리. p_fail>θv=0.65이면 recovery trigger. Rollback 전략(EMM의 최근 checkpoint로 복귀 후 "return to the state shown" prompt) 또는 forward recovery(HELM-Fwd) 변형. Rmax=3 attempts.
- **실행 루프** (Algorithm 1): retrieve M_t → π_θ가 a_prop 제안 → SV가 p_fail 계산 → if p_fail>θv then HC.recover() else execute → EMM.write().

## 3. 실험 결과

Table 2 (LIBERO-LONG / CALVIN ABC→D, 3 seed mean±std):

| Method | TSR (%) | SCR (%) | RSR (%) | CALVIN chains |
|---|---|---|---|---|
| OpenVLA (H=8) | 58.4±1.8 | 74.2±1.1 | 12.3±2.1 | 3.02±0.08 |
| OpenVLA H=32 | 63.8±1.7 | 78.1±1.2 | 14.7±2.0 | 3.24±0.09 |
| OpenVLA + Oracle Mem. | 72.4±1.5 | 83.6±1.0 | 28.5±2.6 | 3.41±0.08 |
| OpenVLA + Rule Verifier | 65.2±1.9 | 79.3±1.3 | 19.8±2.3 | 3.18±0.10 |
| OpenVLA + Ensemble (×5) | 67.9±1.8 | 80.8±1.2 | 22.3±2.5 | 3.29±0.09 |
| OpenVLA + LoRA (50K) | 69.3±1.9 | 81.4±1.3 | 18.2±2.4 | 3.31±0.09 |
| HELM-Fwd | 76.3±1.6 | 86.2±1.0 | 38.7±2.9 | 3.44±0.08 |
| **HELM (OpenVLA)** | **81.5±1.4** | **89.3±0.9** | **54.2±3.1** | **3.58±0.07** |
| Octo | 51.2±2.1 | 68.9±1.5 | 9.8±1.9 | 2.74±0.10 |
| HELM (Octo) | 72.8±1.7 | 83.1±1.1 | 46.3±2.9 | 3.21±0.09 |

- Ablation (Table 3): w/o EMM −11.2pp, w/o SV −8.4pp, w/o Rollback −6.3pp (RSR −31.1pp). 결정적으로 "SV w/o m_t"는 −2.3pp에 그쳐 EMM-SV coupling이 핵심임을 입증(AUROC 0.847→0.791).
- Retrieval ablation (Table 4): Random 64.3%, Recency 71.4%, CLIP 81.5%, Learned fine-tuned 82.1% — CLIP은 zero-cost로 fine-tuned retriever와 동급.
- Figure 2: HELM은 FM −76%, FV −61%, FR −82%로 각 failure mode를 해당 컴포넌트가 타겟팅함을 보인다.
- Per-task (Table 5): 5-subgoal task +18~+22pp, 6-subgoal task +24~+26pp — horizon이 길수록 gain이 커진다.

## 4. 한계 및 미해결 문제

- **Sim-to-real 미검증**(§6): LIBERO/CALVIN 시뮬레이션만 평가. 실제 로봇 배치 시 SV는 visual domain shift에 대한 일반화가 보장되지 않으며, CLIP embedding이 real sensor noise 하에서 discriminative power가 떨어질 수 있다. LIBERO-Recovery의 ±5cm perturbation은 현실의 full disturbance range를 반영하지 못한다.
- **Rollback 가정**(§6): Rollback은 reversible action을 전제로 하므로 irreversible 조작(예: 파손 가능 물체)에 적용 불가. HELM-Fwd는 이를 우회하지만 RSR이 54.2%→38.7%로 떨어진다.
- **환경당 데이터 의존**(§6): SV 학습에 환경당 50K rollout step이 필요하며, zero-shot 배포가 제한된다. Subgoal decomposition은 4.3% episode에서 실패하고 oracle 대비 −2.7pp의 bottleneck도 존재.

## 5. 총평

- **Novelty ★★★★☆** — Wrapper paradigm 자체는 새롭지 않지만, memory-conditioned pre-execution failure predictor라는 학습 문제 정의와 EMM-SV의 명시적 coupling은 독창적이다. SV를 reasoning MLP로 만들고 그 입력 안에 EMM의 검색 결과를 통합한 구조가 기여의 핵심이며, ablation(SV w/o m_t −2.3pp)은 이 coupling이 본질적임을 깔끔하게 입증한다.
- **Practical impact ★★★☆☆** — 12ms/step(15% overhead)로 실질적이며 OpenVLA/Octo 모두에서 일관된 gain을 보여 model-agnostic 한 framework이다. 하지만 환경당 50K rollout 수집 비용과 sim-to-real 미검증으로 인해 당장의 deployment는 한정적이다. 그럼에도 "execution-time augmentation"이라는 backbone scaling의 보완적 축을 제시했다는 점에서 실무적 영향력이 있다. LIBERO-Recovery protocol은 재사용 가능한 기여.

## 6. 예상 질문

- **Q1. "longer context로 해결되지 않는다"는 주장의 근거는 충분한가?** A. Table 1에서 H=8→16→32→64→Oracle H로 스윕해 각각 58.4/61.2/63.8/65.1/67.3%를 보고한다. H=64에서도 HELM(81.5%)과 16.4pp 차이가 남으며, 이는 context 확장만으로 닫히지 않는 구조적 gap임을 보여준다. 다만 H=32 이후로는 OpenVLA의 position encoding 한계 등 모델별 이슈일 수 있어 scaling law 증명은 아니다.
- **Q2. SV가 단순 rule verifier나 ensemble uncertainty 대비 정말 필요한가?** A. Table 2에서 rule verifier는 +6.8pp, ensemble(×5)은 +9.5pp(5× inference cost), SV는 +8.4pp(1× cost)로, cost-효율 측면에서 우위. 게다가 ablation의 "SV w/o m_t"가 memory context가 critical임을 보이므로 rule-based 접근이 접근할 수 없는 semantic failure에 학습된 SV가 접근 가능하다는 논거가 성립한다.
- **Q3. Rollback이 가능한 시뮬 환경에서만 통하는 trick 아닌가?** A. 저자들도 이를 인정하고 HELM-Fwd 변형을 제공한다. HELM-Fwd는 TSR 76.3%, RSR 38.7%로 rollback 대비 −5.2pp/−15.5pp이지만 여전히 OpenVLA(58.4%/12.3%)를 크게 상회한다. 즉 EMM+SV 부분은 forward-only 환경에서도 유의한 gain을 준다.
- **Q4. 왜 CALVIN gain(+0.56 chains)이 LIBERO-LONG gain(+23.1pp)보다 상대적으로 작아 보이는가?** A. CALVIN ABC→D는 이미 generalist policy들이 3.0 수준이며 HELM(OpenVLA) 3.58은 Oracle memory(3.41)까지 뛰어넘는다. Max 5 chains의 상단 근처에서의 marginal gain이라 해석해야 하며, 논문은 이것이 backbone-agnostic gain(Octo도 +0.47)임을 강조한다.

<!-- VERIFIED: pdf -->
