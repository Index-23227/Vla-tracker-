# GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization

> **한 줄 요약**: pi₀ flow-matching action expert의 attention head들을 **단일 학습기가 아닌 기능 모듈의 조립체**로 재해석하고, ControlNet 스타일 zero-init residual adapter로 세 가지 보조 신호 — **Object Head (Grounded-SAM mask), Skill Head (KL 기반 temporal sub-skill), Depth Head (frozen DepthAnything-v3 key/value attention)** — 를 head별로 명시 supervise하여, **LIBERO-Plus 평균 68.2 → 75.4 (+7.2pp)**, **RoboTwin 2.0 8-task 77.38 → 90.63 (+13.25pp)** 를 달성한 RSS 2026 채택 plug-and-play 프레임워크. 저자는 Fudan TEAI / Shanghai Key Lab Multimodal Embodied AI / SJTU / HKU OpenDriveLab 합작.

---

## 1. 배경 및 동기

### VLA의 task-relevant feature 학습 문제

현대 VLA(pi₀, OpenVLA, GR00T)는 강력한 VLM 위에 action을 또 하나의 modality로 정렬한다. 그러나 학습은 *end-to-end action imitation*에 의존하여, action decoder가 task-relevant feature를 **암묵적**으로만 학습한다.

명시적 inductive bias가 없으면:
- 시각적으로 두드러지지만 *instruction과 무관한* shortcut(예: 책상 가장자리, 조명)에 과적합.
- distractor object가 있는 OOD 시나리오에서 일반화 실패 (LIBERO-Plus에서 pi₀ baseline 68.2%로 크게 하락).
- 공간 기하 / 시간적 sub-skill 정보가 implicit attention pattern에 흩어져 디코더가 비효율적 학습.

### 핵심 질문
- *"Action decoder의 attention head를 사람이 정의한 task-relevant factor로 명시 supervise할 수 있는가?"*
- *"이런 specialization이 in-domain뿐 아니라 OOD에서도 일반화를 강화하는가?"*
- *"specialized head가 monolithic learner보다 우월한가?"*

📌 [Figure 1·2 삽입] — Plug-and-play attention specialization 개념: 세 종류 head(Object/Skill/Depth)가 각각 다른 auxiliary signal로 학습되어 decoupled high-quality feature 생성.

---

## 2. 방법론 심층 분석

### 2.1 Backbone 및 전반 구조

- **Base VLA**: pi₀ flow-matching VLA (~3B).
- **Vision encoder**: SigLIP.
- **Adapter**: ControlNet 스타일 zero-initialized residual projection — pretrained behavior를 *그대로 유지*하면서 점진적으로 factor-specific bias를 주입.
- 학습 가능 부분: head별 attention 마스킹 + adapter 파라미터. pi₀ backbone은 frozen에 가깝게 유지.

### 2.2 Object Head — Visual Grounding

**보조 신호**: Grounded-SAM으로 얻은 target object segmentation mask.

**손실**: Negative log object-mass loss — task-relevant object region에 attention probability mass가 집중되도록 강제.

$$\mathcal{L}_{obj} = -\log \sum_{i \in \text{mask}} \alpha_i^{(h)}$$

(여기서 $\alpha^{(h)}$는 지정된 head의 attention 분포)

**효과**: 단일-head 단독으로도 LIBERO-Plus 평균 73.4% (+5.2pp), 특히 **Object suite (74.1 → 82.5)** 와 **Long-horizon (60.1 → 64.0)** 에서 강함.

### 2.3 Skill Head — Temporal Skill Logic

**보조 신호**: Soft sub-skill label (task class + null/background class). 한 episode의 timestep을 task-level 카테고리 분포로 표현.

**손실**: KL-divergence between specified head's attention distribution and soft skill label.

$$\mathcal{L}_{skill} = \text{KL}(p_{\text{skill}}^{(h)}\,\|\,p_{\text{label}})$$

**효과**: 단독 73.4% → 72.5%, 특히 **Goal suite (61.4 → 68.9)** 에서 강함 — sub-task 전환 시점 인식이 goal-conditioned task에서 결정적.

### 2.4 Depth Head — 3D Geometric Awareness

**보조 신호**: Frozen DepthAnything-v3 (DA3) encoder의 depth feature.

**메커니즘**: 지정된 head의 **key/value를 DA3 feature로만 제한** (별도 depth loss 없음).
- 일반 RGB 처리 head는 그대로 두고, depth head는 *attention input 자체*를 depth-derived feature로 강제 → implicit geometric grounding.

**효과**: 단독 71.7%, 특히 **Spatial suite (77.7 → 81.4)** 에서 강함 — 공간 관계가 depth로 명확히 디코딩되기 때문.

> ❓ **예상 질문**: Depth head에 explicit depth loss를 두지 않은 이유는?
> **답변**: Attention 자체가 *어디를 보는가*를 결정하므로, depth feature를 key/value로만 제공하면 head는 자연스럽게 geometric reasoning을 internalize. 별도 reconstruction loss 부재로 학습 안정성도 향상.

### 2.5 Annotation Pipeline

특히 흥미로운 점: 보조 라벨이 비싸 보이지만 저자들은 **자동화**한다.

| Process | 시간 (50 episode) |
|---------|------------------|
| 수동 annotation | 43.5 분 |
| 자동 pipeline (Grounded-SAM + DA3 + task labels) | **4 분 (~92% automation)** |

Object mask는 Grounded-SAM, depth는 DA3, skill label은 task class 자동 매핑. 즉 supervision 비용이 자동화로 거의 해소된다.

---

## 3. 데이터 및 실험 설정

### 3.1 평가 환경
- **LIBERO-Plus**: 표준 LIBERO의 *harder 변종* — 더 강한 OOD generalization 요구. 표준 LIBERO 수치는 보고되지 않음.
- **RoboTwin 2.0**: 8 dual-arm task, Agilex Piper dual-arm.
- **Real robots**: ALOHA AgileX (Platform A) + PSI-Bot RealMan RM63 (Platform B) — cross-platform.

### 3.2 비교 baseline (LIBERO-Plus Table I)

OpenVLA, OpenVLA-OFT, NORA, WorldVLA, UniVLA, pi₀-Fast, RIPT-VLA, DreamVLA, AdaMoE, Spatial Forcing, VLA-Adapter.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO-Plus (Table I)

| Variant | Spatial | Object | Goal | Long | **Avg** |
|---------|---------|--------|------|------|---------|
| pi₀ baseline | 77.7 | 74.1 | 61.4 | 60.1 | 68.2 |
| + Object Head only | 80.6 | **82.5** | 67.1 | **64.0** | 73.4 |
| + Skill Head only | 79.8 | 78.9 | **68.9** | 62.7 | 72.5 |
| + Depth Head only | **81.4** | 79.0 | 65.4 | 61.8 | 71.7 |
| **Full (all heads)** | **84.0** | 80.9 | 70.8 | 66.2 | **75.4** |

**관찰**:
- 단일 head가 특정 suite에서 best — 즉 *complementary specialization*. Object는 Object/Long, Skill은 Goal, Depth는 Spatial에 가장 도움.
- Full model (75.4) > 모든 단일 head — synergistic 효과 명확.
- 가장 큰 향상은 **Goal (+9.4pp)** 과 **Object (+6.8pp)** — sub-task 인식과 grounding이 OOD 일반화의 핵심임을 시사.

### 4.2 RoboTwin 2.0 (8 tasks aggregate)

| Setting | Avg |
|---------|-----|
| pi₀ baseline | 77.38% |
| **GuidedVLA (Full)** | **90.63%** |

**+13.25pp** — bimanual 시뮬레이션에서 더 큰 향상. 저자 해석: dual-arm coordination이 sub-task decomposition (Skill Head)과 spatial geometry (Depth Head)에 특히 민감.

### 4.3 Real-world cross-platform (Table II)

| Condition | pi₀ baseline | **GuidedVLA** | Δ |
|-----------|-------------|---------------|---|
| In-domain | 55.8% | **75.8%** | **+20.0pp** |
| Scene generalization | 44.2% | **67.5%** | **+23.3pp** |
| Lighting generalization | 57.5% | **79.2%** | **+21.7pp** |

- Scene/lighting 변화에 대한 **+23pp / +22pp** 향상이 매우 큼. Frozen DA3 depth feature와 grounded object mask가 visual nuisance에 robust하다는 가설을 지지.
- ALOHA + PSI-Bot 2개 platform 모두에서 일관된 향상 — embodiment 일반화에서도 유효.

### 4.4 Factor Quality Ablation

저자들은 "factor 품질 ↔ task 성능"의 *양의 상관관계*를 정량적으로 검증.

| Ablation 축 | 변화 |
|-------------|------|
| Object attention mass 품질 | 61.3% → 74.6% |
| Skill recognition probe accuracy | 66.2% → 72.9% |
| Geometry perception (depth feature ratio↑) | 15.6% → 76.7% |

특히 *depth feature 비율*을 늘릴수록 spatial task가 극적으로 개선됨이 흥미롭다.

### 4.5 Specialization vs. Mixture

저자들의 비교: 세 보조 신호를 *모든 head에 함께* 주는 "mixture" supervision vs. *head별 specialization*. **Specialization 쪽이 우월** — 한 head가 한 factor에 집중하는 것이 multi-task head 사용보다 효과적.

---

## 5. Ablation 분석

각 head의 단독 기여와 combination 기여를 분해해 보면:

| Combination | Avg LIBERO-Plus |
|-------------|----------------|
| pi₀ baseline | 68.2 |
| Object Head | 73.4 (+5.2) |
| Skill Head | 72.5 (+4.3) |
| Depth Head | 71.7 (+3.5) |
| Sum if additive | 68.2 + 5.2 + 4.3 + 3.5 = 81.2 |
| Full model | 75.4 |

세 head의 효과는 **non-additive** (sum 81.2 vs actual 75.4) — 같은 attention 자원을 공유하므로 일부 trade-off 존재. 그럼에도 단일 head 최고치(73.4)보다 2.0pp 이상 향상 → 보완성 입증.

---

## 6. 한계 및 미해결 문제

1. **표준 LIBERO 수치 부재**. LIBERO-Plus는 harder 변종이라 기존 leaderboard와 직접 비교 불가. RotVLA·X-VLA(LIBERO 98+) 같은 SOTA와의 정량 비교가 어려움.
2. **Parameter count 미보고**. pi₀ ~3B 기준이라고 추정되나 adapter 추가 비용은 부재.
3. **세 factor는 사람이 정의**. Force, contact, affordance 등 다른 task-relevant factor에 대한 일반화 가능성은 enumerated하지만 실험되지 않음.
4. **Annotation pipeline 의존**. Grounded-SAM/DA3가 실패하는 도메인(투명 물체, 거울, 저조도)에서는 자동 annotation 품질 저하 가능.
5. **Pretrained model 의존**. DA3 depth encoder를 frozen으로 가정 — 새로운 sensor modality(thermal, force-torque 등)로의 확장 시 동등 품질의 frozen feature extractor가 필요.
6. **Real-world 1개 task family**. 5개 정도의 manipulation task만 cross-platform 평가 — 더 다양한 task 분포에서의 일반화는 미검증.
7. **Code 미공개** (project page만 존재).

---

## 7. 관련 연구 비교

| Approach | Backbone | Supervision 방식 | Trainable | LIBERO-Plus Avg |
|----------|----------|------------------|-----------|----------------|
| pi₀ baseline | pi₀ | End-to-end imitation | Full | 68.2 |
| Knowledge Insulation [Driess et al.] | pi₀ | Action-expert gradient block | Full + block | - |
| RoboGround | Separate VLM | Grounded mask conditioning | Full | - |
| ReconVLA | pi₀ | Gaze region reconstruction | Full | - |
| VAP | Frozen | Selective attention via det. | Adapter | - |
| PVI | Frozen | Visual injection (residual) | Adapter | - |
| **GuidedVLA** | **pi₀** | **Head-level explicit supervision** | **Adapter (zero-init)** | **75.4** |

### 핵심 차별점
- 다른 방법들이 *VLA 전체*나 *additional module*을 학습/주입하는 것과 달리, GuidedVLA는 **개별 attention head**를 supervise — 더 fine-grained.
- ControlNet 스타일 zero-init residual로 *pretrained behavior를 zero-shot에서 그대로 유지*하면서 점진적 specialization. 학습 안정성에 유리.
- Skill Head가 unique — 다른 어떤 baseline도 명시적 *temporal sub-skill recognition*을 attention level에서 supervise하지 않음.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Attention head-level supervision은 명확히 새로움 (head를 functional module로 보는 관점) |
| **Technical depth** | ★★★★☆ — 세 head 디자인이 서로 다른 supervision philosophy를 체계화 |
| **Experimental rigor** | ★★★★☆ — LIBERO-Plus + RoboTwin 2.0 + Real cross-platform, factor quality ablation까지 충실 |
| **Practical impact** | ★★★★☆ — RSS 2026 채택, real-world OOD에서 +23pp 일반화 향상 |
| **Writing quality** | ★★★★☆ — 핵심 idea가 명확하고 figure 정리도 좋음 |

**강점**: VLA decoder에 *명시적 inductive bias*를 부여하는 일반화 가능한 패러다임. 세 head는 초기 instantiation일 뿐이며 다른 factor (force, contact 등)로 확장 가능한 모듈 설계. Annotation 자동화 (4분/50ep, 92%) 덕에 supervision 비용이 비싸지 않음. Cross-platform real-world에서 +20pp 이상의 일반화 향상은 매우 인상적.

**약점**: 표준 LIBERO 수치 부재로 SOTA leaderboard 직접 비교 어려움. 세 factor가 사람이 미리 정의한 것이라, "interpretable factor가 미리 정의되지 않은 task"에서는 효과 보장 안 됨. 또한 모든 head를 supervise하지 않고 일부만 specialize하는 hybrid case에 대한 분석 부재.

**위상**: RSS 2026 채택. "VLA decoder의 attention head를 modular하게 supervise한다"는 관점은 향후 *interpretable / controllable VLA* 연구의 baseline으로 자리잡을 가능성이 있음.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 표준 LIBERO에서는 pi₀ 대비 얼마나 향상되는가? | 논문에 미보고. 모든 LIBERO 결과는 LIBERO-Plus(harder variant). 표준 LIBERO에서는 pi₀가 이미 94% 수준이라 marginal headroom — 저자가 의도적으로 LIBERO-Plus(68.2)를 선택해 향상 폭을 가시화 |
| 2 | 세 head의 효과가 non-additive (5.2+4.3+3.5=13pp vs 실제 7.2pp). 왜 같은 자원을 공유해서? | Attention head capacity는 fixed. 한 head를 object, 다른 head를 skill에 할당하면 일부 trade-off 존재. 그래도 단일 head 최고(73.4)보다 +2pp 향상 → complementary 가치 검증 |
| 3 | Depth Head에 explicit depth loss를 두지 않은 디자인은 합리적인가? | Attention 자체가 "어디를 보는가" 결정하므로 key/value를 DA3 feature로 제한하는 것만으로 geometric grounding이 internalize. 별도 reconstruction loss를 두면 overfitting + 학습 안정성 저하 가능 — 저자의 선택은 minimalist하나 ablation으로 정량 검증되지 않음 |
| 4 | Grounded-SAM이 실패하는 도메인에서 Object Head는 어떻게 학습되는가? | 명시적 fallback 부재. Annotation pipeline이 92% 자동이지만 8%는 수동 점검 — Grounded-SAM 실패 케이스가 그 안에 포함될 가능성. 투명 물체, 거울, 야간 시나리오에서는 fallback 전략이 필요할 것 |
| 5 | Skill Head의 soft label은 어디서 오는가? | Task 카테고리 + null/background class. Episode timestep의 sub-skill 진행 정보를 task class 분포로 표현. 즉 진정한 sub-task segmentation이 아니라 *task-level* temporal label — 더 정밀한 sub-skill (예: grasp → lift → place)으로 가지 않은 것은 ablation 미보고 |
| 6 | Pi₀ 외 다른 backbone (OpenVLA, GR00T)에 plug-and-play가 가능한가? | 저자가 "plug-and-play"라 명시하지만 실험은 pi₀ 단독. ControlNet 스타일 residual은 어떤 transformer backbone에도 부착 가능하나, head-level supervision이 backbone의 attention 구조에 의존 — OpenVLA(action token 기반)와 pi₀(flow-matching expert)는 다른 디자인이라 직접 transfer 보장 안 됨 |
| 7 | Cross-platform real-world에서 +20pp는 ALOHA + PSI-Bot 두 platform 평균인가, 각각인가? | 논문 Table II는 in-domain 75.8 vs 55.8 — platform별 분리 미보고. 즉 한 platform에서만 큰 향상이 있고 다른 platform은 작을 가능성도 존재. 추가 정보 필요 |
| 8 | LIBERO-Plus의 정확한 차이는 무엇인가? | 원문에 명시: scene perturbation, lighting variation, distractor density 증가 등 OOD generalization을 강조한 변종. 즉 *standard LIBERO success가 saturate한 SOTA models*도 LIBERO-Plus에서는 60-70%대로 떨어짐. 저자가 이 benchmark를 선택한 것은 합리적이지만 비교 baseline에 OOD 성능을 보고하지 않은 다른 모델(예: X-VLA)이 있어 head-to-head 어려움 |
| 9 | Factor quality와 success rate의 *양의 상관*은 인과인가 상관인가? | 저자는 "specialized factor의 품질이 task 성능과 양의 상관"이라 주장. 정량적 증거(Section 4.4)가 있으나 ablation 형식 — 인과성을 위해서는 factor quality를 *외생적으로* 조작한 controlled experiment 필요. 현 시점에서는 강한 상관 + 디자인 직관 수준의 증거 |
| 10 | Adapter parameter 수와 추론 overhead는? | 정량 보고 부재. ControlNet 스타일 zero-init projection이 backbone hidden dim의 fraction 수준일 것으로 추정되나, 정확한 수치와 inference latency 비교가 없음 — plug-and-play 주장의 실용성 검증에 필요한 정보 |
| 11 | RoboTwin 2.0 8 task의 task-level breakdown은? | 저자가 aggregate 90.63%만 보고. task-level breakdown은 main text 부재 — Open Microwave 같은 극저 성능 task에서도 향상이 있는지, 아니면 이미 높은 task만 더 올린 결과인지 분리 불가 |

<!-- VERIFIED: pdf -->
