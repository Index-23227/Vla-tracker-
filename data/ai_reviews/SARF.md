# SARF: Structure-Aware Robust Fine-Tuning — Defending Vision-Language-Action Robots Against Physical Attention Hijacking

> **한 줄 요약**: VLA가 물리적 adversarial patch에 무너지는 이유는 표현(representation)이 흔들려서가 아니라 **action query가 어디를 보는지(where)** 가 패치로 납치되기 때문이라는 진단 위에, teacher-student 구조에서 **visual encoder만** 미세조정하여 action token의 cross-attention 분포를 JSD로 증류하는 zero-inference-overhead 방어 SARF를 제안한 연구. LIBERO에서 AGSD 공격 하 실패율 100% → 평균 28.6%, 실기 PiPER에서 성공률 23.0% → 65.0%.

- arXiv: 2608.03231 (2026-08-04) · IROS 2026 · 코드 미공개
- 저자: Jinquan Zhang, Dongfu Yin(교신), Run Yang, Yufeng Yan, Zhen Tian, F. Richard Yu
- 소속: Guangdong Laboratory of AI and Digital Economy (SZ) / 심천대학교 / Carleton University

---

## 1. 배경 및 동기

VLA 정책(RT-2, OpenVLA, Octo, pi0)은 이미지와 자연어 지시를 저수준 제어로 직접 매핑하며 perception-action loop를 end-to-end로 닫는다. 이 구조의 대가는 명확하다. **국소적인 시각 오류가 즉시 물리적 오동작으로 번역된다.**

여기서 가장 현실적이고 저렴한 위협이 **printable adversarial patch**다. EOT(Expectation-over-Transformation)로 최적화하면 시점·조명 변화에도 살아남는 인쇄 가능한 패턴이 만들어지고, 공격자는 그저 종이를 출력해 테이블에 놓기만 하면 된다. 디지털 노이즈처럼 픽셀 접근 권한이 필요하지 않다.

기존 방어의 대표격인 EDPA의 adversarial fine-tuning은 clean 관측과 patched 관측의 **전역(global) 표현을 정렬**하는 방식이다. 저자들의 문제 제기는 이 지점이다. 조작(manipulation)은 본질적으로 **희소한 task-critical evidence** — end-effector와 대상 물체 — 에 의존한다. 전역 표현 평균을 맞추는 것으로는 "정책이 정확히 그 두 지점을 계속 보고 있는가"를 보장하지 못한다.

> ❓ **예상 질문**: 왜 입력 정화(purification)나 PatchGuard 같은 certified defense를 쓰지 않는가?
> **답변**: 논문이 명시적으로 답한다. purification 계열은 test-time 모듈을 추가해 **추론 오버헤드**를 발생시키며, closed-loop 로봇 제어에서 지연은 곧 실패다. 또한 Athalye et al.의 obfuscated gradients 논의처럼 adaptive attack에 취약해 "거짓 안전감"을 줄 수 있다. SARF는 배포 시 아무것도 추가하지 않는다는 제약을 처음부터 설계 조건으로 못박았다.

---

## 2. 핵심 문제 정의: policy-critical action-to-vision attention hijacking

논문의 개념적 기여는 실패 메커니즘의 **이름 붙이기**다.

현대 VLA에서 action 출력은 소수의 **action-query token 집합 Q_act** 가 visual token으로 보내는 cross-attention이 구동한다. 패치는 강력한 **attention attractor** 가 되어 이 action query들의 attention mass를 흡수하고, task-relevant 영역의 attention을 억제한다. 결과적으로 long-horizon trajectory가 통째로 탈선한다.

핵심 명제는 이렇게 요약된다:

> 강건성은 정책이 **무엇을 표현하는가(what)** 가 아니라 **어디를 보는가(where)** 를 안정화하는 데 달려 있다.

이 진단은 검증 가능한 예측을 낳는다 — (a) attention만 조작해도 실패를 유도할 수 있어야 하고, (b) attention만 복원해도 방어가 되어야 한다. 논문은 두 가지 모두를 ablation으로 확인한다(§7).

---

## 3. 공격: AGSD (Attention-Guided Semantic Disruption)

방어를 평가하려면 먼저 충분히 강한 stress-test 공격이 필요하다. 저자들은 AGSD를 만든다.

### 3.1 위협 모델
패치 적용은 마스크 합성으로 정의된다:

```
x' = (1 - m) ⊙ x + m ⊙ δ
```

`m ∈ {0,1}^{H×W}`는 패치 위치 이진 마스크, `δ`는 패치. 물리 실현성을 위해 EOT로 최적화한다:

```
δ* = argmin_δ  E_{t~T} [ L_AGSD( t(x') ) ]
```

### 3.2 손실 설계
```
L_AGSD = λ_attn · L_attn − λ_disp · L_disp − λ_misalign · L_misalign
```
뒤 두 항의 **부호가 음수**라는 점이 중요하다. L_AGSD를 최소화한다는 것은 L_disp와 L_misalign을 **최대화**한다는 뜻이다.

| 항 | 형태 | 목적 |
|---|---|---|
| `L_attn` | `−(1/|Q_act||K_patch|) Σ_{q∈Q_act} Σ_{k∈K_patch} Ā_{q,k}` | action query → 패치 영역 key로의 평균 cross-attention을 **최대화**. `Ā`는 마지막 3개 cross-attention layer의 평균 |
| `L_disp` | InfoNCE, `−log[ exp(sim(z_adv,z_clean)/τ_nce) / Σ_j exp(sim(z_adv,z_clean^(j))/τ_nce) ]` | 최대화 시 `z_adv`와 `z_clean`의 유사도를 낮춰 feature space 교란 |
| `L_misalign` | `(1/B) Σ_i ‖ sim(z_adv^(i), z_text^(i)) − sim(z_clean^(i), z_text^(i)) ‖_1` | image-text 정합을 clean 대비 크게 이탈시킴 |

계수: `λ_attn = 0.8, λ_disp = 0.2, λ_misalign = 0.5`. attention guidance가 주 동력, 나머지는 보조적 semantic disruption이라는 설계 의도가 계수에 그대로 드러난다.

📌 [Figure 3 삽입] — LIBERO 4개 suite × (OpenVLA / OpenVLA-oft primary / OpenVLA-oft wrist)로 최적화된 인쇄 가능 패치 예시. 디지털 노이즈가 아닌 구체적·재현 가능한 패턴이라는 점이 물리적 배포 가능성의 근거.

---

## 4. 방법론 심층 분석: SARF

### 4.1 전체 목적함수
```
L_SARF = λ_feat · L_feat + λ_pcad · L_pcad + λ_geo · L_geo
```
계수: `λ_feat = 0.5, λ_pcad = 1.0, λ_geo = 0.3`.

구조는 teacher-student다. **Teacher** = clean 데이터로 사전학습된 원본 모델(고정), **Student** = patched 입력을 받는 모델. 학습되는 파라미터는 **student의 visual encoder E_v^S 뿐**이며, multimodal backbone `π`와 action head는 완전히 동결된다. 이것이 zero inference overhead의 원천이다 — 배포 시 아키텍처도, 인터페이스도, latency도 그대로다.

### 4.2 L_feat — Feature Anchor Loss
```
L_feat = 1 − (1/N) Σ_{i=1}^{N}  (z_S^(i) · z_T^(i)) / (‖z_S^(i)‖ ‖z_T^(i)‖)
```
`N`은 visual patch token 수. 방향적(directional) 코사인 일치를 강제해 fine-tuning 중 **catastrophic forgetting**을 막는다. 즉 "사전학습된 표현 공간에 닻을 내리는" 역할.

### 4.3 L_pcad — Policy-Critical Attention Distillation (핵심)
```
L_pcad = (1/(H·|Q_act|)) Σ_{h=1}^{H} Σ_{q∈Q_act}  D_JS( P_T^(h) ‖ P_S^(h) )
P^(h)(k|q) = Softmax( A_{q,k}^(h) / τ_attn )
```
`H`는 head 수. **모든 token이 아니라 action token에 대해서만** attention 분포를 증류한다는 것이 이름의 "policy-critical"이 뜻하는 바다. 비대칭 KL이 아닌 **대칭 Jensen-Shannon divergence**를 쓴 것은 teacher/student 어느 쪽으로도 붕괴하지 않는 안정적 정렬을 위해서다.

> ❓ **예상 질문**: 왜 attention 전체가 아니라 action token만 증류하는가?
> **답변**: 이것이 논문의 가설을 그대로 손실함수로 옮긴 것이다. 실패 메커니즘이 "action query의 attention 납치"라면, 방어도 정확히 그 경로만 교정하면 된다. 전체 attention을 증류하면 (a) 불필요한 제약으로 clean 성능을 해치고 (b) 패치와 무관한 attention 변동까지 억제해 적응성을 잃는다. Ablation(§7)에서 `L_pcad` 제거 시 AGSD 하 FR이 17.0 → 87.5로 붕괴하는 것이 이 항이 방어의 주축임을 보여준다.

### 4.4 L_geo — Language-Guided Geometric Consistency
텍스트가 지목하는 영역의 **기하 구조**만 보존하고 배경 노이즈는 무시하는 항이다.

1. Teacher의 text-to-vision attention으로 patch 중요도 계산: `m_i = max_{q ∈ Q_txt} A_{q,i}^T`
2. 날카롭게 만든 pairwise mask: `M_ij = (m_i · m_j)^2`
3. Gram 요소 `G_ij = (z^(i) · z^(j)) / (‖z^(i)‖‖z^(j)‖)` 의 teacher-student 차이를 마스크 가중 평균:

```
L_geo = [ Σ_{i,j} M_ij · (G_ij^S − G_ij^T)^2 ] / [ Σ_{i,j} M_ij + ε ]
```

설계상 가장 영리한 부분은 **마스크를 perturbed student가 아니라 clean teacher stream에서 뽑는다**는 점이다. student의 attention은 이미 패치에 오염되어 있으므로, 그로부터 마스크를 만들면 "패치 영역을 열심히 보존하라"는 자기모순적 신호가 된다. Teacher 기반 마스크는 공격과 무관한 **안정적 기준점**을 제공한다.

📌 [Figure 2 삽입] — (a) VLA 정책 구조, (b) AGSD의 attention hijacking + semantic disruption, (c) SARF teacher-student 학습(동결 부분 명시), (d) 배포 시 zero overhead.

---

## 5. 구현·학습 세부사항

| 항목 | 설정 |
|---|---|
| AGSD 계수 | λ_attn 0.8 / λ_disp 0.2 / λ_misalign 0.5 |
| SARF 계수 | λ_feat 0.5 / λ_pcad 1.0 / λ_geo 0.3 |
| LIBERO 패치 크기 | 이미지 면적의 **5%** |
| 실기 패치 크기 | 20×20, 15×15, 8×8, 5×5 cm 인쇄물 |
| EOT 변환 | 평면 내 회전 θ ~ U(−30°, 30°), 이동 최대 이미지 치수의 10%, 스케일 s ~ U(0.9, 1.1), 이미지 범위 내 균일 랜덤 배치, 완만한 perspective/조명 jitter |
| attention 집계 | 마지막 3개 cross-attention layer 평균 |
| 데이터 예산 | AF(EDPA 방어)와 **동일** |
| 학습 대상 | visual encoder만. 배포 아키텍처·추론 지연 불변 |

학습률, optimizer, epoch, GPU 사양 등은 논문에 보고되지 않았다(§11 참조).

---

## 6. 실험 설정

- **시뮬레이션**: LIBERO 4개 suite (Spatial, Object, Goal, Long). 지표는 **Failure Rate (FR, %)**.
- **실기**: PiPER 탁상 플랫폼, 3개 태스크(pick & place, open drawer, stack). 지표는 **Success Rate (SR, %)**. 조건당 **100회 물리 시행**, 초기 물체 자세·시행 순서 랜덤화, 카메라 시점/거리 변동.
- **공격 대상 모델**: OpenVLA, OpenVLA-oft, pi0.
- **공격 baseline**: Clean / Random(비최적화 인쇄 패치) / UADA / UPA / EDPA / AGSD.
- **방어 baseline**: Original(무방어) / AF(EDPA의 방어) / SARF.

지표 방향이 뒤집히므로 주의가 필요하다. 공격 평가에서는 **높은 FR이 강한 공격**, 방어 평가에서는 **낮은 FR이 강건**이다.

> ❓ **예상 질문**: 왜 방어를 OpenVLA에만 적용했는가?
> **답변**: 논문이 솔직하게 밝힌다 — OpenVLA는 **action-conditioned visual attention에 접근 가능**하기 때문이다. diffusion이나 flow-style decoder로 확장하려면 그에 대응하는 policy-critical visual pathway를 먼저 식별해야 한다. 이는 방법의 본질적 전제 조건이자 최대 한계다(§11).

---

## 7. 주요 결과

### 7.1 공격 효과 (Table I, FR↑)

| Suite | 모델 | Clean | Random | EDPA | **AGSD** |
|---|---|---|---|---|---|
| Spatial | OpenVLA | 14.2 | 35.8 | 100 | **100** |
| | OpenVLA-oft | 2.4 | 10.0 | 39.7 | **97.2** |
| | pi0 | 3.4 | 6.0 | 29.8 | **48.8** |
| Object | OpenVLA | 11.6 | 44.6 | 100 | **100** |
| | OpenVLA-oft | 2.6 | 20.4 | 52.3 | **93.6** |
| | pi0 | 2.0 | 5.2 | 39.5 | **50.4** |
| Goal | OpenVLA | 20.8 | 42.0 | 100 | **100** |
| | OpenVLA-oft | 3.0 | 16.2 | 80.8 | **100** |
| | pi0 | 10.4 | 16.8 | 44.3 | **70.8** |
| Long | OpenVLA | 46.2 | 75.6 | 100 | **100** |
| | OpenVLA-oft | 4.8 | 32.0 | 86.4 | **100** |
| | pi0 | 42.0 | 48.6 | 70.7 | **80.2** |

읽어낼 점 세 가지. 첫째, **비최적화 Random 패치만으로도** 실패율이 상승한다(국소 물리 교란 자체가 조작을 저해). 둘째, AGSD는 OpenVLA를 전 suite 100%로 몰아넣으며 Random 대비 24.4–64.2 포인트 개선. 셋째이자 가장 중요한 점, **cross-architecture 전이**다. OpenVLA 대상으로 만든 패치가 OpenVLA-oft에서 93.6–100%, 더 강한 pi0에서도 48.8–80.2%를 기록한다. EDPA는 OpenVLA에서는 비등하나 OpenVLA-oft/pi0에서 뚜렷이 약하다 — attention 경로를 직접 노린 것이 전이성의 원천이라는 방증.

### 7.2 방어 효과 (Table II, FR↓)

AGSD(적응형 재최적화) 하 OpenVLA:

| Suite | Original | AF | **SARF** | 절대 감소 |
|---|---|---|---|---|
| Spatial | 100 | 90.2 | **17.0** | −83.0 |
| Object | 100 | 99.8 | **14.2** | −85.8 |
| Goal | 100 | 97.2 | **26.5** | −73.5 |
| Long | 100 | 99.0 | **56.8** | −43.2 |
| **평균** | **100.0** | **96.6** | **28.6** | **−71.4** |

다른 공격들에 대한 suite 평균 FR: UADA 97.3 → 26.8, UPA 97.4 → 26.0, EDPA 100.0 → 27.4. AF 대비로는 AGSD 96.6 → 28.6 (추가 67.9 포인트), EDPA 65.8 → 27.4, UADA 78.3 → 26.8.

**Clean 성능 보존**: 14.2/11.6/20.8/46.2 (Original) → 14.4/11.8/21.0/46.6 (SARF). 평균 23.2% → 23.5%로 **0.3 포인트** 비용. 대조적으로 AF는 clean FR이 17.9/17.3/22.8/49.0으로 눈에 띄게 악화된다. 즉 SARF는 강건성-정확도 trade-off를 거의 지불하지 않는다.

**적응형 공격(adaptive attacker)**: SARF의 AGSD 수치는 고정된 패치를 재사용한 것이 아니라, 동결된 SARF-tuned 모델에 대해 **동일한 AGSD 목적함수와 EOT 설정으로 재최적화한** 패치로 얻은 것이다. 방어 논문에서 자주 생략되는 이 프로토콜을 지킨 점은 신뢰도를 크게 높인다.

### 7.3 실기 검증 (Table III, SR↑, 100회 시행/조건)

| Task | Clean(Original) | AGSD/Original | AGSD/AF | **AGSD/SARF** |
|---|---|---|---|---|
| pick & place | 79.0 | 32.0 | 51.0 | **74.0** |
| open drawer | 71.0 | 23.0 | 42.0 | **63.0** |
| stack | 66.0 | 14.0 | 29.0 | **58.0** |
| **평균** | **72.0** | **23.0** | **40.7** | **65.0** |

인쇄된 패치가 실제로 72.0 → 23.0으로 성능을 붕괴시킨다는 것 자체가 위협의 현실성을 입증한다. SARF는 공격 하에서 65.0을 회복해 clean 72.0과의 격차를 7 포인트로 좁힌다. pick & place는 74.0 vs 79.0으로 거의 복구되나, 정밀도가 요구되는 stack은 14.0 → 58.0(+44.0)의 큰 개선에도 clean 대비 8 포인트 격차가 남는다.

📌 [Figure 5 삽입] — rollout 시점 t = 0, T/3, 2T/3, T의 action-to-vision attention heatmap. AGSD 하 Original은 attention이 패치 영역으로 지속 collapse, SARF는 동일 패치에서 task-relevant 영역으로 재중심화. 정량 개선과 메커니즘 설명이 일치하는 증거.

📌 [Figure 6 삽입] — PiPER 실기 정성 비교(Start/Grasp/Approach/Outcome 4 키프레임).

---

## 8. Ablation 분석

### 8.1 AGSD 목적함수 ablation (Figure 4)
Random / Attn-only / Disp+Misalign / Full AGSD 비교. **Attn-only만으로도 FR이 상승**한다 — action-to-vision attention 조작이 그 자체로 실패를 유발한다는 §2 가설의 직접 증거. Disp+Misalign도 성능을 저하시키나, 둘을 결합한 Full AGSD가 전 suite 최강이다. 두 축이 상보적이라는 뜻.

### 8.2 SARF 구성요소 ablation (Table IV, LIBERO-Spatial, FR↓)

| Method | Clean | Random Patch | AGSD |
|---|---|---|---|
| only L_pcad | 34.7 | 41.1 | 54.4 |
| only L_geo | 37.9 | 59.8 | 81.6 |
| w/o L_feat | 31.6 | 34.9 | 37.2 |
| w/o L_pcad | 14.2 | 44.6 | **87.5** |
| w/o L_geo | 14.0 | 21.5 | 35.8 |
| **Full SARF** | **14.4** | **15.0** | **17.0** |

세 가지 해석이 깔끔하게 분리된다.
- **L_pcad 제거 → AGSD FR 17.0 → 87.5**: 방어의 **주 동력**. 다른 두 항으로는 attention hijacking을 막을 수 없다.
- **L_feat 제거 → clean FR 14.4 → 31.6**: 강건성은 어느 정도 유지되나(37.2) **clean 성능이 붕괴**. 사전학습 표현의 닻 역할이 실증됨.
- **L_geo 제거 → AGSD 35.8**: 중간 정도의 강건성 하락. 보조적이지만 유의미.
- **only-* 행들의 clean FR이 34.7/37.9로 나쁘다**는 점도 중요하다. 단일 항만으로는 clean 성능 자체를 지킬 수 없어, 세 항이 시너지적이며 대체 불가임을 보여준다.

---

## 9. 효율성

논문의 실용적 셀링 포인트는 **zero inference overhead**다. test-time 모듈이 없고 policy interface가 불변이며 배포 아키텍처와 추론 지연이 그대로다. 학습 데이터 예산도 AF와 동일하게 맞췄으므로 개선이 추가 데이터에서 온 것이 아니다. purification 계열이 프레임마다 생성 모델을 돌려야 하는 것과 대비하면, closed-loop 제어 주기를 지켜야 하는 로봇 도메인에서 이 제약은 협상 불가능한 요구사항에 가깝다. "drop-in defense"라는 표현은 과장이 아니다.

다만 **fine-tuning 비용 자체는 보고되지 않았다** — GPU 시간, step 수, 배치 크기 모두 부재.

---

## 10. 강점

1. **메커니즘 우선 접근**. "실패한다"가 아니라 "action query의 attention이 납치되어 실패한다"까지 내려가고, 그 진단을 손실함수 한 항(L_pcad)으로 정확히 번역했다. Ablation이 이 인과 사슬을 검증한다.
2. **적응형 공격 프로토콜 준수**. 방어 논문의 고전적 함정(고정 패치 재사용)을 피하고 SARF-tuned 모델에 재최적화한 패치로 평가했다.
3. **강건성-정확도 trade-off 거의 없음**. clean 평균 FR 23.2 → 23.5. AF가 clean까지 악화시키는 것과 대비된다.
4. **시뮬레이션-실기 일관성**. 인쇄 패치로 실제 PiPER에서 72.0 → 23.0 붕괴를 재현하고 65.0까지 복구. 조건당 100회 시행은 로봇 논문 기준으로 충실한 표본이다.
5. **Teacher 기반 마스크의 설계적 정합성**. 오염된 student가 아닌 clean teacher에서 language-guided mask를 뽑는 선택은 사소해 보이지만 자기모순을 피하는 핵심 결정이다.
6. **공격과 방어를 함께 제시**. AGSD가 EDPA보다 강한 전이성을 보이므로 방어 평가의 난이도가 실제로 올라가 있다.

---

## 11. 한계 및 비판적 검토

1. **아키텍처 일반성 부재**. SARF는 OpenVLA 단일 모델에만 적용됐다. 논문 스스로 인정하듯 diffusion/flow decoder(pi0 등)에는 "유사한 policy-critical visual pathway를 식별해야" 한다. 그런데 공격 평가는 pi0까지 포함했으므로, **위협 범위는 넓게 잡고 방어 범위는 좁게 검증한** 비대칭이 남는다. 실제로 최근 VLA의 주류가 flow matching 쪽으로 이동한 점을 고려하면 이 격차는 작지 않다.
2. **Long suite의 잔여 취약성**. AGSD 하 FR 56.8은 절반 이상 실패다. 저자는 "미세한 attention jitter가 long-horizon에서 누적된다"고 설명하나, 이는 방어가 오류를 제거한 것이 아니라 **감쇠시킨** 것임을 뜻한다. 장기 과제에서는 여전히 실용 수준이 아니다.
3. **학습 하이퍼파라미터 및 비용 미보고**. optimizer, lr, step 수, GPU 사양이 전혀 없다. λ 계수 선택 근거도 "설계 원칙을 반영한다"는 서술뿐 sweep 결과가 제시되지 않아 재현성과 민감도 평가가 어렵다.
4. **패치 크기 5% 단일 조건**. LIBERO 실험은 이미지 면적 5% 패치로 고정됐다. 실기에서는 20×20~5×5 cm를 언급하나 **크기별 분해 결과가 표로 제시되지 않는다**. 5 cm 패치에서도 65.0이 유지되는지는 알 수 없다.
5. **위협 모델의 협소함**. clutter, 자연 distractor, 다중 패치, 비패치형 물리 공격(조명·텍스처)은 다루지 않는다. 저자도 future work로 명시.
6. **AF baseline의 공정성**. AF는 EDPA 논문의 방어를 저자들이 재구현한 것으로 보이며, AGSD 하 96.6이라는 거의 무력한 수치가 원 논문 세팅에서도 그러한지 확인할 근거가 논문 내에 없다. 데이터 예산을 맞췄다는 서술은 있으나 튜닝 노력의 대등성은 검증되지 않는다.
7. **Clean 절대 성능의 낮음**. Original OpenVLA의 clean FR이 Long에서 46.2(성공률 53.8)로, 현재 SOTA VLA들이 LIBERO-Long에서 95% 이상을 내는 것과 큰 차이가 있다. 베이스 정책이 약한 상태에서의 강건성 결론이 강한 정책에도 이전되는지는 열린 질문이다.

---

## 12. 종합 평가 및 시사점

SARF는 "VLA 보안"이라는 비교적 신생 영역에서 **문제를 올바른 추상 수준에서 정의한** 드문 논문이다. 대부분의 adversarial robustness 연구가 표현 공간 정렬이라는 범용 처방을 로봇 도메인에 옮겨 심는 데 그치는 반면, 이 논문은 VLA의 구조적 특수성 — action token이 cross-attention으로 시각 증거를 소비한다는 사실 — 에서 출발해 공격(L_attn)과 방어(L_pcad)를 대칭적으로 설계했다. Ablation에서 L_pcad 제거 시 87.5로 붕괴하는 결과는 이 논리 사슬이 사후 해석이 아니라 실제 인과임을 뒷받침한다.

실용적 관점에서 zero inference overhead와 unchanged policy interface라는 제약을 지킨 것은 배포 가능성 측면에서 결정적이다. 100 → 28.6(시뮬), 23.0 → 65.0(실기)이라는 수치는, 적응형 공격 하에서 얻어졌다는 점을 감안하면 상당히 강한 결과다.

그럼에도 이 논문은 **완결된 방어가 아니라 방향 제시**로 읽는 것이 정확하다. OpenVLA 단일 아키텍처, Long suite 56.8%의 잔여 실패, 미보고된 학습 비용, 패치 크기 단일 조건이라는 제약이 결론의 일반성을 제한한다. 후속 연구에서 확인할 가치가 있는 지점은 (a) flow-matching/diffusion decoder에서 "policy-critical pathway"를 어떻게 정의할 것인가, (b) clean 성능이 이미 95%+인 최신 VLA에서도 동일한 방어 이득이 나오는가, (c) attention 재중심화 정도와 태스크 성공률의 정량적 상관, (d) 다중 패치·자연 clutter로 확장했을 때의 열화 곡선이다.

무엇보다 이 논문이 남기는 가장 가치 있는 문장은 방어 기법 자체가 아니라 그 전제다 — end-to-end visuomotor policy에서는 **모델이 무엇을 표현하는지를 광범위하게 정렬하는 것보다, 어디를 보는지를 안정화하는 것이 훨씬 효과적이다**. 이 명제는 adversarial robustness를 넘어 distribution shift, sim-to-real, long-horizon 안정성 전반에 시사점을 갖는다.

<!-- VERIFIED: pdf -->
