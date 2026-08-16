# CofactVLA: Deconfounding Vision-Language-Action Models via Counterfactual Intervention

> **한 줄 요약**: VLA가 언어 지시를 무시하고 두드러진 시각 단서에 의존하는 **"vision-override" 현상**을 인과 혼입(causal confusion) 문제로 정식화하고, 단일 forward pass 안에 **language-masked counterfactual branch**를 만들어 (1) action 레벨의 **Orthogonal Projection Guidance (OPG)** 와 (2) feature 레벨의 **Counterfactual Covariance Reduction (CCR)** 로 이중 개입한 π0.5 기반 정책. LIBERO 평균 **96.9 → 98.5**, LIBERO-Plus zero-shot total **53.6(π0) → 69.1**, 실기계 OOD 평균 **23.5(π0.5) → 75.8 (+52.3pp)**. Tsinghua 자동화과, arXiv 2608.04396 (2026-08-05).

---

## 1. 배경 및 동기

VLA는 사전학습 VLM을 로봇 제어에 이식해 큰 진전을 이뤘지만, 개방 환경 배치에서 치명적 취약점이 드러난다. 저자들은 이를 **Vision-Override Phenomenon**이라 부른다.

- 로봇 데이터셋은 **모달리티 불균형**이 극심하다. 시각 스트림은 조밀(dense)하고 언어 지시는 희소(sparse)하다.
- 그 결과 정책은 언어 T를 인과 driver로 취급하지 않고, **가장 눈에 띄는 물체를 습관적으로 잡는** 식의 시각 shortcut에 과적합한다.
- 기존 대응은 두 갈래: (a) 언어 rephrasing·counterfactual data augmentation 등 **데이터 중심** 접근 — open-world 확장이 어렵고 내부 표현의 얽힘을 풀지 못함, (b) **Classifier-Free Guidance 류의 scalar 외삽** — 사후적 선형 외삽이 정렬되지 않은 노이즈를 증폭해 연속 궤적을 OOD로 밀어내고 물리적으로 위험한 실행을 유발.

핵심 질문: *"이 시각 장면에서 언어 지시가 완전히 사라진다면 VLA는 본능적으로 어떤 행동을 할 것인가?"* 이 counterfactual 질문의 답이 곧 순수 시각 혼입자(confounder)이며, 이를 명시적으로 뽑아내 제거하자는 것이 논문의 출발점이다.

---

## 2. 인과 정식화 — Dual-path Deconfounding Graph (DDG)

이상적으로 action A는 언어 의도 T와 관측 O에 인과적으로 의존해야 한다. 그러나 모달리티 불균형 때문에 모델은 **스퓨리어스 backdoor path `I ⇢ C → A`** 를 학습한다. 여기서 C는 잠재 시각 혼입자(두드러진 out-of-context 물체, 익숙한 배치)다.

DDG는 두 개의 do-operator를 배치한다.
- **빨간 do-operator (C 위)**: Vision Intervention = CCR. 잠재 표현에서 시각 혼입자를 억제.
- **파란 do-operator (T 위)**: Language Intervention = OPG. 순수 언어 의도를 기하학적으로 추출·증폭.

구현 관점에서는 하나의 forward pass 안에 **factual branch (O, T 조건)** 와 **counterfactual branch (O만, 언어 마스킹 φ)** 를 동시에 굴린다. counterfactual branch의 출력은 정의상 "언어 없이 시각 편향만으로 만들어진" 표현·속도장이다.

---

## 3. Action-Level Deconfounding: OPG

### 3.1 왜 action 레벨 개입이 필요한가

로봇 제어는 본질적으로 **multi-valued**다. 동일한 의미 의도 T도 여러 유효 궤적(파지 스타일, 접근 각도)으로 실행될 수 있다. action chunk 공간 A^H를 이산 모드 Z = h(A)로 분할했을 때, 전문가 시연이 nuisance-correlated mode selection을 보이면 I(Z; C | O) > 0 이고, data processing inequality로 I(A; C | O) > 0 이 따라온다. 이는 곧

```
H(A|O) − H(A|O,C) = I(A;C|O) > 0
```

즉 분포 변화 하에서 **엄격한 NLL gap**을 낳는다. 텍스트를 완벽히 인지해도 시각 혼입자가 모드 선택을 오염시킨다는 뜻이며, 생성 출력 공간에서 명시적 개입이 필요하다는 근거가 된다.

### 3.2 Flow matching에서의 score 등가성

Gaussian forward process `A_τ = τA + (1−τ)ω` 하에서 score는 속도장으로부터 선형 변환된다.

```
∇_{A_τ} log p_τ (A_τ | ·) = −(A_τ + τ v_θ(A_τ, τ | ·)) / (1 − τ)
```

이 **정확한 선형 등가성** 덕분에 속도장 조작이 곧 score function 합성(Product-of-Experts)에 대응한다.

### 3.3 직교 투영

기존 counterfactual action guidance는 `v_CFG = v_uncond + γ(v_cond − v_uncond)` 형태의 scalar 외삽이며, 이는 **직교 분리 가능성**을 암묵 가정한다. 이 가정이 깨지면 비정렬 노이즈를 지수적으로 증폭한다.

OPG는 대신 factual 속도장을 counterfactual 시각 본능 방향으로 **투영**한 뒤 그 성분만 제거한다.

```
v_proj = (⟨v_cond, v_uncond⟩ / (‖v_uncond‖² + ε)) · v_uncond
v_⊥    = v_cond − v_proj
v_causal = v_cond + γ · v_⊥      (γ > 1.0)
```

collinear한 시각 편향만 정확히 제거하므로, guidance가 **의미 방향의 causal mode odds만 재가중**하고 유효 action manifold를 이탈하지 않는다.

---

## 4. Feature-Level Deconfounding: CCR

OPG가 출력단을 교정한다면, vision-override의 **뿌리는 잠재 표현 공간**에 있다. 시각 혼입자가 VLM backbone의 attention을 지배해 KV cache 단계에서 이미 shortcut을 형성한다.

factual/counterfactual branch의 flatten된 attention feature `F_f, F_cf ∈ R^{N×d}`에 대해 중심화 공분산 `Σ_f, Σ_cf`를 구하고

```
ΔΣ = Σ_cf − Σ_f
```

두 가지 가정이 식별성을 보장한다.
- **Assumption 1 (Gain–bias decomposition)**: 잠재공간이 인과 관측 의도 부분공간 S_O 와 스퓨리어스 시각 혼입자 부분공간 S_C 로 직교 분해된다 (R^{d_r} = S_O ⊕ S_C).
- **Assumption 2 (Contrastive eigengap)**: `M := Σ_0^{-1/2} ΔΣ Σ_0^{-1/2}` 의 고유값이 `Σ_0^{1/2} S_C` 로 제한했을 때 `Σ_0^{1/2} S_O` 제한보다 **엄격히 크다**.

**Theorem 1**: ΔΣ의 양의 스펙트럼 주성분(λ > ε > 0 인 top-k 고유벡터)으로 이루어진 nuisance basis `U_bias ∈ R^{d×k}` 가 정확히 S_C 를 span한다. 따라서

```
F_causal = F − β (F U_bias) U_bias^T,   β ∈ (0, 1]
```

U_bias ⊥ S_O 이므로 언어 의미 표현은 **무손상**으로 보존된다. 실제로는 VLM 마지막 층들의 Key/Value 계산을 가로채 적용하며, 논문 설정은 intervention layers **[15, 16]**, β = 0.15.

---

## 5. 구현 및 학습 설정

| 항목 | 값 |
|---|---|
| Framework | HuggingFace lerobot |
| Backbone | 사전학습 **π0.5** 체크포인트에서 초기화 (Apache 2.0) |
| Compute | 4× NVIDIA H100 96GB |
| Batch size | 32 / GPU |
| Learning rate | 2.5e-5 |
| Training steps | 6K (warm-up 1K) |
| Optimizer | AdamW, betas [0.9, 0.95], weight decay 0.01 |
| Action chunk size | 50 |
| Freeze vision encoder / action expert | False / False |
| Gradient checkpointing | True |
| γ (causal scale) | 2.0 |
| β (intervention strength) | 0.15 |
| Intervention layers | [15, 16] |

중요한 점은 vision encoder와 action expert를 **모두 학습**한다는 것이다. 즉 CofactVLA는 frozen 정책 위의 추론 래퍼가 아니라, 인과 개입을 학습 목적에 통합한 **자체 fine-tuned 정책 산출물**이다.

데이터: 시뮬레이션은 **표준 LIBERO 데이터셋만** 사용(Spatial/Object/Goal/Long), 태스크당 10 에피소드 평가. LIBERO-Plus는 학습된 정책을 그대로 배치한 **zero-shot**(벤치마크 규모 때문에 태스크당 1 에피소드). 실기계는 태스크당 약 100개, 총 약 400개 전문가 궤적 수집, 태스크당 100회 독립 시행 평가.

---

## 6. 주요 결과 — 표준 LIBERO

| Model | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 (base) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| **CofactVLA** | **99.0** | **100.0** | 98.0 | 97.0 | **98.5** |

Object suite에서 **100.0%**, Spatial 99.0%. 베이스 π0.5 대비 +1.6pp, SOTA X-VLA 대비 +0.4pp. Long(97.0)에서 베이스 92.4 대비 +4.6pp로 장기 지평 과제의 개선폭이 가장 크다 — 언어 의도가 여러 단계에 걸쳐 유지되어야 하는 상황에서 deconfounding이 특히 유효함을 시사.

---

## 7. Zero-shot OOD — LIBERO-Plus

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | **65.1** | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT_w | 10.4 | 38.7 | 70.5 | 76.8 | **93.6** | 49.9 | 69.9 | 55.8 |
| OpenVLA-OFT_m | 55.6 | 21.7 | **81.0** | **92.7** | 91.0 | 78.6 | 68.7 | 67.9 |
| **CofactVLA** | 44.7 | **49.7** | 71.8 | 85.6 | 83.6 | 78.0 | **70.2** | **69.1** |

Total 69.1%로 1위. 개별 축에서 최고는 아니지만(Camera는 π0-Fast, Light/Language는 OFT_m, Background는 OFT_w가 우세) **전 축에 걸친 균형**이 압도적이다. 특히 다른 방법들이 대부분 붕괴하는 **Robot(새로운 embodiment) 49.7%** 와 **Layout 70.2%** 에서 최고 — 익숙한 시각 단서에 의존하지 않는다는 주장과 정확히 일치하는 패턴이다. π0 베이스라인(53.6) 대비 +15.5pp.

---

## 8. 실기계 실험

하드웨어: 6-DoF AgileX PiPer + 1-DoF 평행 그리퍼, Intel RealSense D435 2대(overhead 3인칭 + wrist), 10 Hz.

태스크 4종: I. 파란 접시에서 직육면체 치우기, II. 테니스공을 노란 접시 → 파란 접시, III. 사과를 노란 접시 위로, IV. 빨간 큐브를 노란 접시에.

| 조건 | π0.5 | CofactVLA |
|---|---|---|
| 표준 환경 평균 | 71.0 | **90.8** |
| 표준 Task IV (시각 distractor 다수) | 31.0 | **88.0** |
| OOD 장면 평균 | 23.5 | **75.8 (+52.3pp)** |

가장 극적인 지점: π0.5는 표준 환경 Task II에서 **100%** 를 달성하지만, 미지의 체크무늬 배경으로 옮기면 **0%** 로 완전 붕괴한다. 암기된 시각 shortcut 의존의 교과서적 증거다. CofactVLA는 같은 조건에서 견고하게 성공한다. OOD 설계는 LIBERO-Pro에서 영감을 받아 object perturbation(주변 distractor, 색/모양 변화), environment perturbation(잔디 텍스처, 파란 체크 테이블보, 밝은 노란 접시), instruction perturbation 세 축으로 구성.

---

## 9. Ablation

**모듈 분해 (LIBERO 4-suite)**

| Variant | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Baseline | 97.0 | 99.0 | 96.0 | 96.0 | 97.0 |
| w/ CCR | 99.0 | 99.0 | 98.0 | 94.0 | 97.5 |
| w/ OPG | 100.0 | 99.0 | 98.0 | 95.0 | 98.0 |
| **Full CofactVLA** | 99.0 | 100.0 | 98.0 | 97.0 | **98.5** |

CCR 단독 +0.5pp, OPG 단독 +1.0pp, 결합 +1.5pp로 **상가적 이상의 시너지**. 저자 해석은 CCR이 구조적으로 정화된 의미 의도를 제공해 OPG의 기하 투영이 더 정확해진다는 것. 흥미로운 세부: CCR 단독은 Long에서 오히려 94.0으로 하락(baseline 96.0)하지만 OPG와 결합하면 97.0으로 회복된다.

**Counterfactual 설계 비교**

| Design | Avg |
|---|---|
| Add | 94.0 |
| Sub | 96.0 |
| CAG | 97.5 |
| **OPG (Ours)** | **98.5** |

단순 덧셈은 오히려 베이스라인보다 나쁘고(94.0), scalar 뺄셈 96.0, 기존 CAG 97.5. 직교 투영이 비직교 노이즈 증폭 없이 편향만 제거한다는 주장을 뒷받침.

**민감도 분석**
- γ ∈ {0, 0.5, 1, 2, 3}: 증가할수록 복잡 태스크 개선(Spatial 99%, Long 97% 정점), Object/Goal은 98–100% 유지. γ=3에서는 소폭 하락 — 과도한 투영이 유효 action manifold를 왜곡. **γ=2 채택**.
- β ∈ {0, 0.05, 0.1, 0.15, 0.2}: **β=0.15에서 평균 98.5%로 명확한 정점**, Goal 98% / Long 97% 최대. β=0.2에서는 열화 — 과도한 공분산 페널티가 언어 정렬된 필수 시각 특징까지 씻어냄.

---

## 10. 강점

1. **문제 정식화가 정교하다.** vision-override를 막연한 "shortcut learning"이 아니라 backdoor path `I ⇢ C → A`로 명시하고, mode selection에 대한 상호정보 논증 `I(Z;C|O) > 0 ⇒ I(A;C|O) > 0 ⇒ NLL gap` 으로 개입 필요성을 유도한다.
2. **Flow matching score 등가성을 정확히 활용.** 속도장 조작이 score 합성과 등가임을 보인 뒤 그 위에서 기하 연산을 정의하므로, guidance 조작이 확률 밀도 수준에서 무엇을 하는지 해석 가능하다.
3. **CFG류 대비 명확한 개선 논리.** scalar 외삽의 직교 분리 가정을 지적하고, 실제로 Add/Sub/CAG 대비 일관된 우위를 보였다.
4. **단일 forward pass 설계.** counterfactual branch를 별도 모델이 아니라 언어 마스킹으로 동일 pass 안에서 얻어 오버헤드를 억제.
5. **OOD 증거가 강력하다.** 시뮬(LIBERO-Plus zero-shot)과 실기계 양쪽에서, 특히 π0.5가 100% → 0%로 붕괴하는 지점을 잡아낸 것은 설득력이 높다.

---

## 11. 한계 및 비판적 검토

**저자가 밝힌 한계**
- 베이스 VLM의 zero-shot grounding 능력에 본질적으로 병목된다 — 인과 개입은 학습되지 않은 개념을 합성하지 못한다.
- 심한 물리적 가림(팔이 카메라를 막는 경우)에 여전히 취약. 향후 dynamic multi-view fusion 계획.

**리뷰어 관점 추가 우려**
- **표준 LIBERO의 헤드룸이 거의 없다.** 98.5 vs X-VLA 98.1의 +0.4pp는 태스크당 10 에피소드 평가에서 통계적으로 유의하다고 보기 어렵다. 실질적 기여는 OOD 축에 있다고 읽는 것이 정확하다.
- **LIBERO-Plus 평가가 태스크당 1 에피소드.** 규모 때문이라 밝혔지만 분산이 큰 측정이다. 69.1 vs 67.9(OFT_m)의 격차도 이 노이즈 수준에서는 조심스럽게 봐야 한다.
- **추론 비용 보고 없음.** dual-branch forward + 매 스텝 공분산 고유분해는 비용이 있을 텐데, 지연시간·처리량(Hz) 수치가 제시되지 않았다. 실기계는 10 Hz로 운용된다고만 명시.
- **Assumption 1·2의 검증 부재.** 잠재공간의 직교 분해와 contrastive eigengap은 강한 가정인데, 실제 특징에서 이 조건이 성립하는지에 대한 경험적 진단(스펙트럼 플롯 등)이 없다. top-k의 k 선택 기준도 "λ > ε" 이상으로 구체화되지 않았다.
- **개입 층 [15, 16]의 선택 근거 부재.** 층 위치에 대한 ablation이 없어 하이퍼파라미터 민감도를 판단하기 어렵다.
- **실기계 OOD의 +52.3pp는 베이스라인이 0%로 붕괴한 사례에 크게 힘입는다.** 인상적이지만 baseline 붕괴 조건에서의 상대 개선은 과대 해석되기 쉽다.
- **코드/체크포인트 공개 언급 없음.** 재현성 확인 불가.

---

## 12. 종합 평가 및 위치

CofactVLA는 "VLA가 언어를 무시한다"는 널리 관찰된 실패를 **인과 그래프 위의 backdoor 문제로 환원**하고, 그 구조에 정확히 대응하는 두 개입(출력단 기하 투영 + 잠재단 공분산 감쇄)을 제안한 논문이다. 별도 데이터 수집이나 아키텍처 재설계 없이 π0.5 fine-tuning 파이프라인에 통합되며, vision encoder와 action expert를 모두 학습하는 **자체 정책 산출물**이라는 점에서 추론 시 래퍼류와는 구분된다.

- **CFG/CAG 계열과의 관계**: scalar 외삽을 기하 투영으로 대체 — 개념적으로 가장 직접적인 후속.
- **counterfactual data augmentation 계열과의 관계**: 데이터가 아니라 표현·출력에 개입하므로 open-world 확장성이 낫다.
- **X-VLA 등 SOTA 아키텍처와의 관계**: 경쟁이라기보다 **직교적**이다. OPG/CCR은 원리상 다른 flow-matching VLA에도 이식 가능해야 하는데, π0.5 외 backbone 이식 실험이 없는 것이 아쉽다.

표준 LIBERO 숫자보다 **LIBERO-Plus 69.1 total**과 **실기계 OOD 75.8 vs 23.5**가 이 논문의 진짜 결과다. VLA 일반화 연구에서 "인과 개입"이라는 축을 실용적으로 작동시킨 사례로 참조 가치가 높다.

---

**핵심 수치 요약**: LIBERO Avg **98.5** (Spatial 99.0 / Object 100.0 / Goal 98.0 / Long 97.0) · LIBERO-Plus zero-shot total **69.1** · 실기계 표준 **90.8** vs π0.5 71.0 · 실기계 OOD **75.8** vs π0.5 23.5 · γ=2.0, β=0.15, layers [15,16], 6K steps on 4×H100.

<!-- VERIFIED: pdf -->
