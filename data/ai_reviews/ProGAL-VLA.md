# ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models

## 1. 배경 및 동기

LIBERO-Plus가 드러낸 두 가지 VLA 실패 모드가 출발점이다(Sec. 1): (i) **language ignorance** — 정책이 시각 shortcut에 의존해 instruction 변화에 둔감함, (ii) **robotic instability** — 의미 추론이 시각-운동 결합을 교란할 때 제어가 무너짐. 저자는 대부분의 VLA가 shallow concatenation으로 언어/시각을 융합해 "추론된 symbolic goal이 실제 3D scene의 actionable entity와 매칭되는지" 검증하지 않는다는 구조적 결함을 지적한다. ProGAL-VLA는 **verification bottleneck** — 즉, 언어 의도가 3D entity에 명시적으로 bind되기 전에는 어떠한 action도 실행되지 않도록 강제하는 아키텍처적 제약 — 을 해결책으로 제시한다.

## 2. 방법론

아키텍처는 4개 컴포넌트로 구성된다(Sec. 3, Fig. 1).

- **Prospective Planner $\pi_\text{slow}$ (Sec. 3)**: Qwen-2.5-VL-Instruct-7B를 사용해 instruction $L$, 관측 $O_t$, 시간 메모리 $M_{t-1}$로부터 symbolic sub-goal $s_t$(예: "grasp green mug")를 생성. **에피소드당 1회만 비동기 호출**되므로 per-step 추론 stack은 여전히 OpenVLA-7B만 쓴다(공정 비교의 핵심).
- **Grounded State Module (GSM, Sec. 3)**: YOLO-World open-vocabulary detection을 3D로 lifting해 entity graph $G_t = \{e_1,\dots,e_n\}$, 시간 메모리 $M_t$, 결합 표현 $E_t = \text{Concat}(G_t, \text{Retrieve}(M_{t-1}, O_t))$를 구성(Eq. 4). GSM은 semantic reasoning을 하지 않고 object-centric 구조만 노출함.
- **State Alignment Cross Attention (SACA, Sec. 3, Eq. 6)**: $Q = \text{Embed}_\text{sym}(s_t)$, $K,V = \text{Embed}_\text{gnd}(E_t)$로 $g_t = \text{Softmax}(QK^\top/\sqrt{d})V$를 얻는다. attention entropy $H_t = -\sum_i \alpha_{t,i}\log\alpha_{t,i}$(Eq. 27)를 ambiguity signal로 활용해 selective prediction에 재활용.
- **Action Policy $\pi_\text{fast}$ (Sec. 3)**: OpenVLA-7B가 $g_t, O_t, q_t$만 조건으로 받아 $a_t$를 내놓는다. **원시 언어 feature는 $\pi_\text{fast}$에 직접 들어가지 않는다** — 이게 verification bottleneck의 구현.
- **Grounding Alignment Contrastive (GAC) Loss (Eq. 10–11)**: $(s_t, e^+)$ positive pair에 대해 InfoNCE 형식의 contrastive loss를 추가. 저자는 Theorem 1(Eq. 21)에서 $I(S; E) \ge \log N - \mathbb{E}[\mathcal{L}_\text{GAC}]$라는 entity-level mutual information lower bound를 증명한다.
- **Proposition 1 (Sec. 4, Eq. 19)**: Verification bottleneck($a_t \perp L \mid g_t, O_t, q_t$) 하에서 $I(L; a_t\mid O_t, q_t) = I(L; g_t\mid O_t, q_t) - I(L; g_t\mid a_t, O_t, q_t)$ — 즉 $g_t$가 언어에 민감하게 반응해야 action도 언어에 민감해진다.

## 3. 실험 결과

**Table 2 (LIBERO-Plus, 7 perturbation types)**: ProGAL-VLA가 총점 **85.5%**로 OpenVLA-OFT+(79.6), RIPT-VLA(69.3), OpenVLA(17.3)를 모두 능가. Robot perturbation에서 **30.3→71.5** (+41.2%p), Layout 77.6→86.7 (+9.1%p), Language 85.8→93.6 (+7.8%p)로 가장 큰 gain은 robot/layout. 8개 열 중 6개에서 1위.

**Table 3 (Custom Ambiguity Benchmark, CAB)**: AUROC 0.52(OpenVLA) → **0.81**(Ours), AUPR 0.49→**0.79**, ECE 14.3→**4.6**, Clar@Ambig 0.09→**0.81**, Unambig SR 0.74→**0.89**. GAC 제거 시 0.66, entropy-only 변형은 0.73으로, 세 요소(hierarchical planning + GSM + GAC)가 모두 기여함을 확인.

**Table 4 (Entity retrieval & language ignorance)**: Recall@1이 N=8에서 0.41→**0.71**, N=32에서도 0.15→**0.41**로 급등. Simple/Spatial/Relational instruction의 language ignorance도 (0.36, 0.49, 0.57) → (**0.08, 0.14, 0.19**)로 3~4× 감소. GAC 제거 시 (0.22, 0.31, 0.45)로, GAC 손실이 language sensitivity 회복의 핵심임이 Sec. 4의 이론적 예측(Theorem 1 → Eq. 21 → Recall@1 향상)과 일치.

**Table 5 (Latency)**: Detector 43.0 ms + GSM 15.8 + SACA 10.7 + $\pi_\text{fast}$ 26.9 = **96.4 ms (10.31 FPS)**. per-step overhead는 SACA 10.7 ms에 불과해 bottleneck이 되지 않음.

**Fig. 2–3**: 조건을 $L \to s_t \to g_t$로 바꿀수록 success rate 상승·language ignorance 감소. Risk-Coverage curve에서 ProGAL-VLA가 모든 coverage에서 하위 risk.

## 4. 한계 및 미해결 문제

- YOLO-World와 Qwen-2.5-VL-7B planner의 오류가 grounding failure로 전파될 수 있고, 저자들도 이를 명시적 한계로 인정(Sec. 6).
- 평가는 simulated LIBERO-Plus와 CAB뿐이며, occlusion/depth noise가 있는 실제 로봇에서의 GSM/SACA 이전은 미해결.
- CAB는 attribute collision(색/크기)만 다루며, pragmatic/temporal ambiguity는 future work.
- Planner가 에피소드당 1회만 호출되므로, 에피소드 중 장면이 크게 바뀌는 long-horizon 태스크에서는 $s_t$가 stale해질 위험.
- 7B controller + 7B planner + detector로 시스템 복잡도가 높고, 검증된 pretrained 컴포넌트에 의존.

## 5. 총평

- **Novelty**: ★★★★☆ — "verification bottleneck"의 아키텍처적 강제와 이를 정보이론적으로 뒷받침한 Proposition 1·Theorem 1의 결합은 명확한 기여. 단, 개별 컴포넌트(planner/GSM/contrastive)는 기존 아이디어의 합성.
- **Practical impact**: ★★★★☆ — LIBERO-Plus robot 카테고리에서 +41.2%p는 실용적으로 큰 의미이고, calibrated abstention(CLARIFY)은 안전 제약 로봇 배포에 직접 연결. 다만 YOLO-World + 7B slow planner의 시스템 복잡도가 엣지 배포의 걸림돌.

## 6. 예상 질문

**Q1. 왜 $\pi_\text{fast}$에 raw language $L$을 넣지 않는가?**
A. Assumption 1(Eq. 18): $a_t \perp L \mid (g_t, C_t)$. $L$을 직접 넣으면 verified embedding을 우회하는 shortcut이 생기고, $\pi_\text{slow}$가 확인하지 못한 의미가 policy에 스며들어 language ignorance가 재현된다. Table 4의 "ProGAL ($L \to \pi_\text{fast}$)" ablation이 바로 이 실패 모드를 정량화(language ignorance 0.27/0.33/0.42로 악화).

**Q2. Planner를 에피소드당 1회만 부르면 장면 변화에 어떻게 대응하나?**
A. GSM의 temporal memory $M_t$가 per-step 갱신되고, SACA가 매 step 재평가되므로 $s_t$ 자체는 stale해도 $g_t$는 현재 entity 집합에 재결합된다. 다만 근본 재계획이 필요한 경우는 Sec. 6에서 future work로 남겨둠.

**Q3. GAC loss를 제거했을 때 ambiguity detection은 왜 악화되는가?**
A. Theorem 1(Eq. 21)에서 $\mathcal{L}_\text{GAC}$ 최소화는 $I(S; E)$ lower bound를 올린다. 이 bound가 커져야 symbolic query가 3D entity와 선명히 구분되고, SACA attention 분포가 peaky해져 entropy가 ambiguity signal로서 기능한다. GAC 없으면 Table 3에서 AUROC 0.66으로 하락.

**Q4. OpenVLA-OFT+와 비교해 어떤 perturbation에서 강하고 어디서 약한가?**
A. Table 2: Camera(92.8 vs 93.2), Robot(30.3 vs **71.5**), Language(85.8 vs **93.6**), Layout(77.6 vs **86.7**)에서 ProGAL-VLA 승. 반면 Light(94.9 vs 86.8), Noise(89.3 vs 74.8)는 OpenVLA-OFT+가 근소 우세 — 저수준 센서 노이즈는 3D entity 그래프 구축 자체를 교란하기 때문으로 해석.

<!-- VERIFIED: pdf -->
