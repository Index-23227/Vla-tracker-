# w2VLA: Decoupling the Declarative from the Procedural in Vision-Language-Action Models

> **한 줄 요약**: 두 개의 frozen two-tower VLM(MetaCLIP2) + VFM 백본 위에 FiLM 기반 conditioning-block transformer를 두고, proprioceptive hidden state를 Visual → Where(공간 heatmap) → What(skill embedding) 순으로 **순차 modulate**함으로써, 단일 (skill, object) demo 16개만으로도 **다른 객체로의 zero-shot skill transfer**를 91.7% 평균 성공률로 달성. OTTER(30.6%)와 π0.5(38.2%)는 실패. 55.17M 학습 파라미터.

---

## 1. 배경 및 동기

### 문제 정의 — 현대 VLA는 skill transfer를 못 한다
- OpenVLA·π0·π0.5·OTTER·VLA-0 등은 LIBERO·RoboTwin 등에서 SOTA이지만, 객체 배치·색상 같은 사소한 perturbation에 catastrophic 실패.
- 본 논문이 정조준하는 문제: **(skill, object) demo로 학습된 정책이 다른 object에 같은 skill을 zero-shot으로 옮기지 못함** — 예: "carrot을 90° 회전"으로 학습한 모델이 "banana를 90° 회전"을 실행하면 banana에 학습된 "5cm 뒤로 두기"를 답습.
- 이는 단순 scaling으로 해결 불가 — π0.5 같은 3B+ end-to-end VLA도 같은 함정에 빠짐.

### Key Insight — declarative와 procedural의 분리
- VLM은 객체 identity·공간 위치·시각 의미(declarative)에 강함 → frozen으로 활용해야 IL low-data regime에 의해 희석되지 않음.
- Skill을 실행하는 procedural knowledge는 demo로부터 학습되어야 하며 declarative와 entangle되면 안 됨.
- 인간 비전 시스템의 perception/action 분리(Goodale & Milner, 1992) 차용 — **where**(공간 grounding)과 **what**(motor intent)를 구조적으로 분리.

### 차별점
- OTTER처럼 VLM을 frozen하되, 토큰을 단순 concat하는 opaque action expert를 **거부**.
- CLIPort의 what/where stream 아이디어를 stateless discrete pick-and-place가 아닌 **고차원 연속 action policy**로 확장.

---

## 2. 방법론 심층 분석 — w2VLA 아키텍처

### 2.1 모델 입력 (§3.2)
- Timestep $t$에서 history length $T$의 proprioceptive states $P = \{p_{t-T+1}, ..., p_t\}$ + $C$ 카메라 시각 관측 $I$.
- 이미지는 $N$ patch로 분할, **두 인코더**:
  - **VFM**: dense spatial/semantic patch tokens $F \in \mathbb{R}^{T \times C \times N \times D}$.
  - **VLM (MetaCLIP2 two-tower)**: text-aligned patch embeddings $V \in \mathbb{R}^{T \times C \times N \times D}$.
- Language pre-processor가 instruction $l$을 **object 문자열 + skill 문자열**로 분해 → 각각 VLM text encoder에 통과 → $e_{obj}, e_{skill} \in \mathbb{R}^D$.

### 2.2 Robot state 인코딩
- MLP(Proprio Encoder)로 $P$를 $D$-차원 latent로 사상 + temporal positional embedding → hidden states $H = \{h_{t-T+1}, ..., h_t\}$.
- $H$가 action expert의 base sequence; conditioning block들이 순차적으로 modulate.

### 2.3 Visual Modulation
- VFM patch tokens $F$를 **Attentive Feature Aggregation (AFA)**로 압축: $Q$개 learnable query로 cross-attention → task-relevant summary tokens $F'$.
- Cross-Attention layer: $H$를 query, $F'$를 key/value로 통합 → hidden state에 시각 context 주입.
- 뒤이은 Causal Self-Attention block이 temporal reasoning + extrapolation.

### 2.4 Conditioning Block (FiLM + Pre-LayerNorm CSA)
- 공통 구조: 외부 conditioning vector $\sigma$를 MLP로 $\gamma(\sigma), \beta(\sigma)$로 projection → $h'_\tau = \gamma(\sigma) \odot h_\tau + \beta(\sigma)$ (RT-1 스타일 FiLM).
- 뒤에 Pre-LayerNorm CSA + residual: $h''_\tau = h'_\tau + \text{CSA}(\text{LayerNorm}(h'_\tau))$.
- **Pre-LayerNorm + residual**의 의도: FiLM이 injection한 $\gamma, \beta$가 다음 LayerNorm으로 표준화되어 무효화되는 것을 방지.

### 2.5 Spatial Conditioning (where) — §3.2의 핵심
- Similarity tensor: $S_t = V_t \cdot e_{obj}$ — 각 timestep·카메라별 visual patch와 object text embedding의 내적.
- Spatial probability distribution: $m_t = \text{Softmax}(S_t / \zeta)$, **temperature** $\zeta = 0.05$ (sharp localization).
- Heatmap $m_t$를 flatten → MLP로 transformer hidden dim에 project → $\sigma_\tau^{spatial}$.
- 첫 conditioning block에서 $\Sigma^{spatial}$로 hidden state를 modulate → "어디"를 grounding.

### 2.6 Skill Conditioning (what)
- Skill embedding $e_{skill}$을 MLP로 단일 vector로 project → $\sigma^{skill}$.
- Primitive 실행 동안 semantic goal은 불변하므로 **시간축으로 broadcast**.
- 두 번째 conditioning block에서 hidden state를 modulate → "무엇"을 procedural하게 지정.

### 2.7 Action Head
- 완전히 conditioned된 hidden state sequence를 4-layer MLP (hidden 512)가 처리 → action chunk $\hat{A} = \{a_{t+1}, ..., a_{t+L}\}$ (L=10).
- Action은 end-effector pose deltas로 parametrize.

### 2.8 Training Recipe (§3.3)
- **Backbone frozen**: VFM, VLM 모두 학습 전 과정 frozen → pre-trained representation 보존, IL low-data에 의한 희석 차단.
- **Random VFM patch masking** (q=0.5, MAE 차용): visual appearance bias 차단, where/what 모듈에 의존하도록 강제 → declarative/procedural decoupling 강화.
- Loss: 표준 BC L1 — $\mathcal{L} = \mathbb{E}_{(p, I, l, a) \sim D}[L_1(a, \pi_\theta(p, I, l))]$.

---

## 3. 실험 설정 — Skill Transfer Scenarios

### 3.1 문제 정의
- 두 객체 $o_1, o_2$ × 두 skill $s_a, s_b$ → 학습 demo는 $\langle s_a, o_1\rangle, \langle s_b, o_2\rangle$만.
- **Skill transfer 평가**: $\langle s_a, o_2\rangle, \langle s_b, o_1\rangle$ (swapped pair).
- Scoring: object 선택(1점) + skill 실행(1점) + task 완료(1점), 최대 3점.

### 3.2 네 가지 시나리오 (각 16 demos × 2 = 32 demos)
| Scenario | (skill A, obj A) | (skill B, obj B) |
|---|---|---|
| S1 | rotate 90°, carrot | place back 5cm, banana |
| S2 | place on plate, corn | place in bowl, eggplant |
| S3 | poke, sponge | nudge, toy car |
| S4 | drop on plate, toothpaste | place forward 5cm, cucumber |

### 3.3 Baselines
- **OTTER**: frozen two-tower VLM + text-aware AFA + causal transformer (concat 방식). CLIP → MetaCLIP2-ViT-L/16으로 업그레이드해 공정 비교. 67.11M trainable.
- **π0.5**: 3B PaliGemma + flow-matching action expert; knowledge insulation 전략으로 VLM frozen, action expert + continuous projection만 fine-tune (693.42M trainable).
- **w2VLA**: 55.17M (셋 중 최소).

---

## 4. 핵심 결과 (Table 1, Fig. 5)

### 4.1 In-domain (seen) vs Skill Transfer 성공률
| Policy | S1 Seen / Transfer | S2 | S3 | S4 | **Avg Seen** | **Avg Transfer** |
|---|---|---|---|---|---|---|
| OTTER | 97.2 / 33.3 ✘ | 91.7 / 25.0 ✘ | 94.4 / 30.6 ✘ | 91.7 / 33.3 ✘ | 93.8 | **30.6 ✘** |
| π0.5 | 94.4 / 41.7 ✘ | 97.2 / 27.8 ✘ | 97.2 / 44.5 ✘ | 94.4 / 38.9 ✘ | 95.8 | **38.2 ✘** |
| **w2VLA** | 94.4 / 91.7 ✔ | 94.4 / 94.4 ✔ | 97.2 / 91.7 ✔ | 94.4 / 88.9 ✔ | **95.1** | **91.7 ✔** |

→ **In-domain 성능은 동등** (95.1 vs 93.8/95.8) — 모듈러 설계가 기본 VLA 성능을 희생하지 않음을 입증.  
→ **Skill transfer에서 3× 격차** — w2VLA만이 swap된 (skill, object)을 안정적으로 처리.

### 4.2 Breakdown (Fig. 5a)
- OTTER와 π0.5는 transfer 시 "object 선택"은 잘함 (frozen VLM 덕분) — 하지만 학습된 **skill–object 상관**을 따라 잘못된 skill을 실행.
- 예: "rotate 90° banana"를 명령하면 OTTER/π0.5는 banana에 학습된 "place 5cm back"을 수행.
- w2VLA는 object와 skill을 모두 정확히 실행.

---

## 5. Robustness 실험 (§4.3, Fig. 5b)

### 5.1 Distractors (S1 + 3-5개 무작위 distractor 객체)
- Seen: 100% → 83.3% (-16.7%p), Transfer: 91.7% → 77.8% (-13.9%p).
- 실패 mode는 대부분 **wrong object가 아닌 imprecise interaction** (mid-air pick 등) — VLM localization heatmap은 distractor에도 robust(Fig. A5).
- Skill 실행은 두 setting 모두 거의 perfect → declarative/procedural decoupling이 robustness 하에서도 유지됨.

### 5.2 Unseen Objects (S2의 corn/eggplant → Coca Cola can, Fanta can, tomato, strawberry, lemon, potato 등 12개)
- Transfer success: 100% → 83.3% (-8.3%p)만 하락.
- Object 선택은 거의 항상 성공, skill 실행도 항상 성공 — VLM의 zero-shot localization 덕분.
- Task 완료 실패의 주 원인: 기하학적 차이(예: mug grasp vs bottle grasp)로 인한 trajectory 부적합.

---

## 6. Ablation (§A.4)

### 6.1 Visual Modulation + Patch Masking (Table A.2, S2 dataset)
| VM | Masking | Seen | ST |
|---|---|---|---|
| ✘ | ✘ | 66.7 | 52.8 |
| ✓ | ✘ | 100 | 58.3 |
| ✓ | ✓ | 100 | **94.4** |

→ Visual Modulation은 in-domain 성능에 필수.  
→ **Random patch masking이 skill transfer의 핵심** — masking 없이 VFM patch가 그대로 들어가면 spurious skill–object 상관을 학습.

### 6.2 Module Order (Table A.3)
| Sequence | Seen | ST |
|---|---|---|
| VM+Mask → where → what | 100 | **94.4** |
| VM+Mask → what → where | 100 | 55.6 |

→ **Where를 먼저** modulate해야 함. 인간 sensorimotor coordination(시각으로 객체 위치 파악 → 행동 계획)과 동일.

---

## 7. Implementation Details (§A.5)

### 7.1 학습 설정
- **OTTER & w2VLA**: 15,000 steps, batch 32, RTX 4090, AdamW (peak LR 1e-4, WD 1e-4, cosine to 1e-5, 200-step warmup), history 8, action horizon 10, 4-layer MLP head (hidden 512).
- **π0.5**: 10,000 steps, batch 8, RTX 4090, AdamW (peak LR 1.5e-5, β=(0.9,0.95), WD 0.01), grad clip 1.0, 1,000-step warmup, 224×224 입력, action chunk 50, max language len 200 tokens.

### 7.2 Parameter Breakdown (Table A.4)
| Module | Robot State Enc. | Visual Mod. | Where | What | Action Head | Total |
|---|---|---|---|---|---|---|
| Params | 0.27M | 14.18M | 15.22M | 21.52M | 3.99M | **55.17M** |

→ Where와 What 모듈이 전체의 ~67%; Action head는 4M에 불과.

### 7.3 Hardware
- SO-101 leader/follower, ZED2i 스테레오 카메라(왼쪽 렌즈만 사용, depth 미사용), 256×256 center-crop.

---

## 8. 강점

1. **Skill transfer라는 새 정의** — 단순 "unseen instruction"이 아니라 (skill, object) pair swap이라는 깨끗한 evaluation protocol.
2. **3× 격차의 실험적 증거** — π0.5(3B 파라미터)도 실패하는 문제를 55M w2VLA가 해결.
3. **Decoupling을 architectural prior로 인코딩** — scaling 의존이 아닌 정보 흐름의 재구성이 답임을 보임.
4. **In-domain 성능 미손실** — 모듈러 설계의 흔한 약점(전체 성능 trade-off)을 회피.
5. **Frozen backbone**: pre-trained representation의 가치를 최대화하면서 catastrophic forgetting 방지.
6. **Patch masking + Where-first ordering**의 ablation이 명확 — 단순 architectural choice가 아니라 정량적으로 justified.
7. **VLM heatmap의 robustness** 시각화(Fig. A5/A6) — distractor와 unseen object에 대해서도 sharp localization.

---

## 9. 약점 / 한계

1. **Primitive skill에 한정**: pick/rotate/poke/drop 등 기초 motion만 평가. 장기 horizon 복잡 task는 직접 다루지 않음.
2. **Language pre-processor 가정**: instruction을 "object"와 "skill" 문자열로 분해하는 외부 parser가 필요 — 자연어 처리가 단순한 시나리오에 한정됨.
3. **기하학적 유사 객체 가정**: skill transfer가 객체 간 geometric similarity에 의존 — mug handle grasp ↔ bottle grasp 같은 affordance shift는 실패 (저자도 명시).
4. **단일 robot embodiment**: SO-101 + ZED2i 단일 실험 — cross-embodiment 일반화 미검증.
5. **단일 instruction template**: skill–object 분리가 항상 가능한 instruction set만 다룸.
6. **데이터 양 매우 적음**: 각 (skill, object)당 16 demos × 4 scenarios. LIBERO/RoboTwin 같은 표준 벤치마크 미사용 → leaderboard 비교가 어려움.
7. **Open-source 아님**: 현재 project page만 있고 code release 없음.
8. **Single-axis skill 가정**: skill이 dominant motion mode 하나로 표현 가능해야 함 (저자 framework는 explicit하진 않지만 평가된 skill들이 모두 단순 motion).

---

## 10. 다른 연구와의 위치 — Related Work 정리

### 10.1 Fine-tuned VLM 계열 (scaling 의존)
- OpenVLA (Llama-2 + visual/language token mapping), π0/π0.5 (PaliGemma + flow matching), VLA-0 (text-as-action), SmolVLA, GR00T N1 — 모두 거대 모델 + 거대 데이터셋 의존.

### 10.2 Frozen VLM 계열 (선구자: OTTER, OpenVLA+)
- OTTER가 가장 가까운 baseline — 같은 two-tower frozen + AFA 사용. w2VLA의 차별점은 **token concat 대신 sequential modulation**.

### 10.3 Minimal Visual Cues / Compositional 계열
- CLIPort (what/where stream, discrete SE(2)), KITE (keypoint), PEEK (gripper path overlay), VoxPoser/F3RM/C2G (3D value map).
- w2VLA는 이 정신을 잇되 **continuous high-dimensional action policy**로 확장.

### 10.4 Skill Transfer 평가의 한계 노출
- RT-1·RT-2·OCTO·OpenVLA의 "generalization"은 massive data scaling에 의존한 visual interpolation에 가까움 — w2VLA의 실험은 그러한 모델이 **소규모 데이터에서는 skill transfer 능력이 없음**을 명백히 함.

---

## 11. 향후 연구 방향

### 11.1 저자가 명시한 방향
- **Hierarchical VLA 통합**: w2VLA를 low-level primitive executor로 사용, 상위 planner가 abstract command를 primitive로 분해.
- **"How" 모듈 추가**: AnyGrasp 같은 grasp pose 모델을 활용해 fine-grained geometric affordance에 conditioning.
- **Parameter-efficient 추가 학습**: 새 객체에 적응 시 few-shot demo로 trajectory adaptation.

### 11.2 자연스러운 확장
1. **3D spatial representation**: 2D heatmap을 3D point cloud / NeRF feature로 교체.
2. **Action head 교체**: 현재 MLP regression → flow matching / diffusion으로 swap (저자도 readily extensible하다고 언급).
3. **Multi-axis skill**: 현재 단일 dominant motion 가정을 multi-component motion으로 확장.
4. **Cross-embodiment**: 다른 robot arm/gripper에서 frozen backbone의 transfer 검증.
5. **Language pre-processor의 학습**: instruction → (object, skill) 분해도 학습 가능한 형태로.
6. **LIBERO/RoboTwin 등 표준 벤치마크**: skill transfer 평가가 표준 leaderboard에서 어떻게 보이는지 검증 필요.

---

## 12. 종합 평가

w2VLA는 "modern VLA가 skill transfer를 못 한다"는 진단을 깨끗한 실험으로 입증하고, **architectural prior(declarative/procedural decoupling via sequential FiLM modulation)**로 이를 해결한 매우 명료한 논문이다. 55M trainable 파라미터로 3B π0.5를 skill transfer에서 압도하는 것은 scaling-first paradigm에 대한 강력한 반박이며, frozen VLM + structured information flow의 가치를 입증한다.

핵심 통찰 셋:
- (1) Skill–object correlation은 **token concat의 부산물**이며, sequential modulation으로 끊을 수 있다.
- (2) **Random patch masking + Where-first ordering**이 단순 architectural trick이 아니라 정량적으로 결정적이다.
- (3) Frozen VLM의 dense feature는 **conditional probability distribution(heatmap)**으로 압축될 때 procedural module과 깨끗하게 분리된다.

한계는 실험 범위(SO-101 단일, primitive skill, 표준 벤치마크 부재)에 집중되어 있으며, 핵심 아이디어 자체는 매우 일반적이라 확장 여지가 크다. Compositional generalization을 architecture로 인코딩하는 방향성에서 **CLIPort의 정신적 계승작**이자, VLA 연구의 "scaling 외" 방향의 대표 사례로 자리매김할 가능성이 높다.

**평가 점수: 8.0/10** — clean diagnostic + minimal architectural fix + strong empirical result. LIBERO/RoboTwin 표준 벤치마크에서의 검증과 code release가 추가되면 9.0+로 평가 상향 가능.

<!-- VERIFIED: pdf -->
