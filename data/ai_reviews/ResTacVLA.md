# Feeling the Unexpected: ResTacVLA for Contact-Rich Manipulation via Residual Tactile Representation

> **한 줄 요약**: Predictive Coding에서 영감을 받아, 촉각을 raw 입력이 아니라 **"시각 prior가 예측하지 못한 잔차(residual)"** 로 재정식화. Cross-Modal Predictor(CMP)가 wrist 카메라로부터 촉각 latent를 예측하고, 실제 촉각과의 residual을 **VQ bottleneck으로 이산화한 Latent Contact Primitives**로 변환한 뒤, 예측 불확실성 σ_t 기반 **Surprise-Aware Gate(SAG)** 로 π0.5 flow-matching 정책에 적응적으로 주입. 5종 real-world contact-rich task 평균 성공률 **62.8%** — vision-only π0.5 (28.2%) 대비 **+34.6%p** (Tab. I).

---

## 1. 배경 및 동기 (§1, §2)

- 현행 VLA(π0, π0.5 등)는 vision-centric — 정밀 삽입, 나사 체결, 표면 닦기 같은 **contact-rich task**에서는 심한 occlusion과 미세 물리 동역학(마찰·변형)을 시각만으로 해석 불가.
- Naive한 촉각 융합(공유 feature space로 직접 projection)은 **'Modality Collapse'** 유발: 고대역폭·연속적인 시각 스트림이 이벤트성·희소한 촉각 신호를 압도해, 정책이 촉각을 무시.
- 인지신경과학의 **Predictive Coding**: 뇌는 예측 가능한 감각 입력을 감쇠시키고 '예상 밖(surprise)'에 집중 (자기 간지럼이 불가능한 이유, Wolpert [15]).
- 핵심 질문: "**VLA가 '예상 밖'을 느끼게 할 수 있는가?**" → 촉각을 시각 prior 대비 **surprise의 정량적 측정치**로 재정식화.

---

## 2. 방법론 심층 분석 (§3, Fig. 2)

### 2.1 문제 정식화
- 관측 `O_t = {V_base, V_side, V_wrist, I_tac, s_t}` (s_t ∈ R^7), 언어 지시 L → action chunk `A_t = {a_t, ..., a_{t+H-1}}`, π(A_t | O_t, L).
- π0.5 프레임워크 기반: PaliGemma VLM이 시각·언어·proprioception을 인코딩, conditional flow matching으로 action 생성.

### 2.2 Cross-Modal Predictor (CMP)와 Residual 추출 (§3-B.1)
- **Vision Encoder**: 학습 가능한 ResNet-18 + MLP head가 `V_wrist`에서 예측 평균 `μ_t ∈ R^{3×H'×W'}`와 **스칼라 표준편차 `σ_t ∈ R`** 출력 → 예측 촉각 latent `ẑ_t`의 Gaussian 분포.
- **Tactile Encoder**: UniT [21] 기반으로 실제 촉각 이미지 `I_tac`을 같은 latent 공간의 `z_t`로 projection.
- **Residual**: `r_t = z_t − ẑ_t` — 시각이 예상 못한 물리적 감각 성분만 분리.
- 예측기 손실 (Eq. 1): weighted NLL `L_pred = λ_σ log σ_t² + ‖z_t − μ_t‖² / σ_t²` — λ_σ가 uncertainty collapse 방지. **σ_t가 곧 cross-modal surprise의 원리적 측정치**.

### 2.3 Latent Contact Primitives via VQ (§3-B.2)
- Event encoder f_φ (conv residual blocks + global max pooling)가 `r_t → h_t ∈ R^D` 집약 → **VQ bottleneck** (codebook C, K entries)으로 이산 primitive `q_t` 획득.
- Codebook collapse 대응 4종: codebook 차원 축소, Euclidean → cosine similarity, EMA 업데이트, 비활성 entry 주기적 재초기화.
- 재구성 경로: 예측 prior `ẑ_t`를 `q_t`로 **FiLM conditioning** → `z̃_t` → 촉각 이미지 재구성 `Î_t`.
- 전체 손실 (Eq. 2): `L_CMP = L_rec + λ_p·L_pred + L_vq` (L_rec = MSE). 5개 task 데이터로 joint pre-train.

### 2.4 Surprise-Aware Tactile Policy (§3-C)
- Gate (Eq. 3): `g_t = Sigmoid(MLP(σ_t))` — σ_t 높음 = 시각이 촉각 상태를 예측 불가 = 촉각 정보 이득 높음.
- Gated interpolation (Eq. 4): `e_t = g_t·p_t + (1−g_t)·e_0` — `p_t`는 q_t의 linear projection, `e_0`는 학습 가능한 'no-contact' 토큰.
- `e_t`를 **noise token에 직접 concat**하여 action expert에 입력 (VLM 사전학습 표현 비침해). CMP frozen 상태로 전체 정책 end-to-end fine-tune (flow-matching objective).

---

## 3. 실험 설정 (§4-A, Fig. 3)

- **플랫폼**: Franka Research 3 + Robotiq 2F-85, 한쪽 fingertip에 **GelSight Mini**, RealSense D435 3대(정면/측면/wrist), RTX 4090 단일 GPU.
- **5개 task**: (a) Lightbulb Screwing — 나사산 체결·회전 저항 감지, (b) Plug Insertion — sub-mm 정밀도 + 심한 occlusion, (c) Peg Transfer — 순차적 미세 조작, (d) Plate Wiping — 지속 접촉 유지 (visual depth 오차로 hovering 방지), (e) Peg-in-Hole.
- Task당 ~100 expert demo. 평가: Screwing/Plug 25 trials, 나머지 15 trials. 3개 task는 **Alignment(A) / Interaction(I) 2단계 metric**으로 분해.
- **베이스라인 5종**: DP w/o T, DP w/ T-ResTac, π0.5 w/o T, π0.5 w/ T-ResNet (직접 인코딩), π0.5 w/ T-UniT (사전학습 SSL 표현).

---

## 4. 실험 결과 심층 분석 (§4-B, Tab. I)

| Method | Lightbulb-A | Lightbulb-I | Plug-A | Plug-I | Peg-A | Peg-I | Transfer | Wiping | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 (Vision Only) | 28.0 | 8.0 | 36.0 | 20.0 | 46.7 | 40.0 | 26.7 | 20.0 | 28.2 |
| DP w/o T | 20.0 | 0.0 | 28.0 | 16.0 | 40.0 | 26.7 | 13.3 | 6.7 | 18.8 |
| π0.5 w/ T-ResNet | 16.0 | 0.0 | 32.0 | 24.0 | 40.0 | 40.0 | 20.0 | 13.3 | 23.2 |
| π0.5 w/ T-UniT | 28.0 | 12.0 | 40.0 | 32.0 | 73.3 | 53.3 | 66.7 | 33.3 | 42.3 |
| DP w/ T-ResTac | 32.0 | 12.0 | 32.0 | 28.0 | 66.7 | 60.0 | 40.0 | 33.3 | 38.0 |
| **ResTacVLA** | **56.0** | **32.0** | **68.0** | **60.0** | **86.7** | **80.0** | 60.0 | **60.0** | **62.8** |

- **평균 62.8%**: π0.5 대비 +34.6%p, DP 대비 +44.0%p. 최고 단일 성공률 **86.7%** (Peg-A).
- **Phase 분석**: vision 정책은 Alignment는 되지만 Interaction에서 급락 (Plug: 36.0% → 20.0%). ResTacVLA는 68.0% → 60.0%로 일관 — 'Physical Gap'을 residual tactile로 메움.
- **Naive fusion의 실패**: T-ResNet은 vision-only보다 오히려 하락하는 경우도 (Lightbulb-A 28.0% → 16.0%) — 규제 없는 고차원 촉각 신호는 distractor. T-UniT은 개선되나 제한적 (42.3%).
- **Architecture-agnostic**: DP에 ResTac 적용 시에도 +19.2%p (18.8% → 38.0%).

---

## 5. Ablation 및 해석성 분석 (§4-C, §4-D, Tab. II)

### 5.1 Ablation (Plug Insertion + Plate Wiping 평균, Tab. II)
| Configuration | Avg. Success (%) | Δ |
|---|---|---|
| ResTacVLA (Full) | 60.0 | - |
| w/o VQ (연속 residual) | 33.3 | **-26.7** |
| w/o Gating (고정 fusion, g=1) | 46.7 | **-13.3** |
| π0.5 (Vision Only) | 20.0 | -40.0 |

- **VQ 제거 → -26.7%p**: 연속 residual은 산발적 jitter·grasp pose 민감성 유발. VQ bottleneck이 고주파 촉각 노이즈를 의미론적 contact event로 증류하는 **정보 필터** 역할.
- **Gating 제거(always-on) → -13.3%p**: 촉각 노이즈와 초기 grasp 오차로 trajectory drift. 적응적 gating이 필수.

### 5.2 해석성 (§4-C, Fig. 4-5)
- **t-SNE of Contact Primitives**: free-space motion 프레임은 5개 task 공통의 단일 compact cluster로 수렴(낮은 촉각 정보 이득), 물리적 상호작용 프레임은 collision·alignment success 등 **의미론적 contact event별 task-specific cluster**로 분화 — 2단계 구조 확인.
- **Gate 시계열 (Lightbulb Screwing)**: Approach 단계에서 g_t ≈ 0 (촉각 억제), 접촉·나사산 체결 시 급상승 — 인간 감각운동 제어와 유사한 phase-dependent modulation을 자연 학습.

---

## 6. Robustness 평가 (§4-E, Tab. III, Fig. 6)

| Method | Dynamic (Peg 3-5cm 이동) | Height +2cm | Height -2cm | Grasp (±5mm, ±10°) |
|---|---|---|---|---|
| π0.5 (Vision Only) | 26.7 | 33.3 | 0.0 | 8.0 |
| π0.5 w/ T-UniT | 40.0 | 46.7 | 13.3 | 20.0 |
| **ResTacVLA** | **66.7** | **53.3** | **40.0** | **52.0** |

- 특히 **Height -2cm에서 vision-only 0% vs ResTacVLA 40%**: 시각 depth 오차로 인한 접촉 손실을 힘 순응으로 회피.
- Grasp perturbation에서 8.0% → 52.0%: 시각으로 안 보이는 actuation misalignment를 촉각으로 보상.

---

## 7. 한계 및 미해결 문제

1. **σ_t가 스칼라**: 공간적으로 분해된 uncertainty가 아니라 프레임 단위 단일 값 — 국소적 surprise를 구분 못 함.
2. **CMP는 wrist 카메라에만 의존**: wrist view 자체가 가려지는 경우 예측 prior의 품질 저하 가능.
3. **단일 GelSight Mini(한쪽 fingertip)**: 양손·다지 촉각으로의 확장 미검증.
4. **표준 벤치마크 부재**: 자체 설계 5개 real task만 평가 — LIBERO 등 시뮬 벤치마크 비교 없음, 태스크당 15-25 trials로 통계적 검정력 제한.
5. **CMP 사전학습이 task 데이터에 의존**: 미학습 task/재질에서 codebook의 일반화는 open question.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | 왜 raw tactile 대신 residual인가? | §1, §2-B: 시각이 예측 가능한 성분은 정보 이득이 0에 가깝고 modality collapse만 유발. residual은 희소 촉각을 dense한 정보 이득으로 변환. |
| 2 | σ_t의 uncertainty collapse는 어떻게 막나? | Eq. 1: λ_σ log σ² 항이 분산 페널티로 작용해 σ→0 붕괴 방지 (Kendall & Gal NLL 정식화). |
| 3 | VQ가 왜 연속 residual보다 나은가? | Tab. II: -26.7%p. 이산화가 고주파 노이즈를 필터링해 semantic contact event로 증류, grasp 변동에 강건. |
| 4 | T-UniT(42.3%)과의 격차 원인은? | 같은 UniT 인코더를 쓰지만 ResTacVLA는 residual화 + VQ + surprise gating으로 시각 중복 성분을 제거 — 표현 자체보다 **통합 방식**이 관건. |
| 5 | Gate가 접촉 센서 thresholding과 뭐가 다른가? | TacCoRL식 binary taxel gate와 달리 **학습된 연속 gate**이며, 접촉 여부가 아닌 **시각 예측 불확실성**에 기반 — 접촉 중에도 시각이 신뢰되면 억제 가능. |
| 6 | DP에도 효과 있는 이유? | Tab. I: DP w/ T-ResTac +19.2%p — residual 표현이 백본 독립적 정보 이득을 제공함을 시사 (architecture-agnostic). |
| 7 | Peg Transfer에서 T-UniT(66.7%)에 밀린 이유는? | ResTacVLA 60.0%로 2위. 순차 파지·이동 중심 task라 surprise 이벤트가 적어 gating 이점이 축소된 것으로 해석 가능 (논문은 명시적 분석 없음). |
| 8 | 데이터 수집 시 operator에게 촉각 이미지를 보여준 이유? | §4-A: 접촉 의존 행동이 demo에 반영되도록 — 촉각 정보가 실제로 행동을 결정하는 궤적을 수집. |

---

## 9. 본 연구의 기여 정리

1. **ResTacVLA**: Predictive Coding 기반으로 modality collapse를 구조적으로 해결하는 residual tactile VLA 프레임워크.
2. **CMP + Latent Contact Primitives**: 시각→촉각 cross-modal 예측의 residual을 VQ로 이산화, 높은 정보 이득의 해석 가능한 contact vocabulary 학습.
3. **Surprise-Aware Gate**: aleatoric uncertainty σ_t를 gating 신호로 재활용 — 별도 접촉 감지기 불필요.
4. 5개 real contact-rich task에서 **최대 86.7%, 평균 +34.6%p** SOTA, 3종 perturbation에서 최고 robustness.

---

## 10. 후속 연구 방향

- **공간 분해 surprise**: σ_t를 spatial map으로 확장해 국소 접촉 이벤트별 gating.
- **양손·다지 촉각**: 복수 GelSight/taxel 배열로의 확장, TacCoRL식 sim-to-real RL과의 결합.
- **CMP의 대규모 사전학습**: 다양한 재질·객체의 vision-touch 쌍으로 codebook 일반화.
- **표준 벤치마크 정착**: contact-rich 시뮬 벤치마크(예: RoboTwin류)에서의 재현 가능한 비교.

---

## 11. 실용적 함의

- π0.5 등 기성 VLA에 **VLM 재학습 없이** 부착 가능 (tactile token을 noise token에 concat, CMP frozen) — 사전학습 표현 보존.
- 단일 RTX 4090에서 수집·평가 가능한 경량 파이프라인, task당 ~100 demo 수준의 데이터 요구량.
- Gate 시계열은 접촉 phase 감지기로도 활용 가능 — 디버깅·안전 모니터링에 유용한 부산물.

---

## 12. 결론

ResTacVLA는 "촉각을 얼마나 넣을 것인가"가 아니라 "**시각이 설명하지 못하는 것만 넣는다**"는 원리로 tactile-VLA 융합을 재구성했다. Residual 표현 + VQ 이산화 + surprise gating의 3단 구조는 각각 ablation으로 필요성이 입증되었고 (−26.7 / −13.3%p), vision-only π0.5 대비 평균 +34.6%p라는 큰 격차와 perturbation 강건성은 predictive coding 관점의 실효성을 보여준다. 스칼라 surprise·단일 센서·자체 태스크 평가라는 한계는 있으나, contact-rich manipulation에서 modality collapse를 정면으로 다룬 원리적이고 재사용 가능한 설계다.

<!-- VERIFIED: pdf -->
