# HOIST: Humanoid Optimization with Imitation and Sample-efficient Tuning for Manipulating Suspended Loads

> **한 줄 요약**: GR00T N1.6를 modify한 high-level VLA가 PICO VR teleoperation 50개 데모로 finetune되고, **flow-matching initial-noise를 steering**하는 actor-critic이 30개 autonomous rollout만으로 RL refinement를 수행하는 humanoid hoisting 시스템. 실 humanoid에서 pure VLA rollouts 대비 **translational placement 19.9 cm 감소**, **raw angular 3.56° 감소**. Whole-body controller는 고정된 GR00T WBC를 그대로 사용.

---

## 1. 배경 및 동기

### 문제 정의: Humanoid Hoisting
- 건설/물류 현장에서 크레인이 들어올린 **외부에서 매달린(suspended) payload**를 작업자가 손으로 가이드/푸시 → struck-by, caught-between 사고 위험 (NIOSH/OSHA 통계)
- Humanoid robot이 이 작업을 대체할 수 있는가?
- **Underactuated**: 로봇이 payload를 직접 잡거나 액추에이트할 수 없고, **whole-body motion + intermittent contact**로만 영향을 줄 수 있음
- Payload는 pendulum-like dynamics → 접촉 후에도 계속 움직이고, overshoot/residual swing 발생

### 기존 접근의 한계
- **Imitation learning만**: 안전한 초기 동작은 학습되지만 final placement accuracy를 직접 최적화하지 않음, closed-loop에서 error 누적
- **RL from scratch on real humanoid**: unsafe + sample-inefficient — contact-rich 환경에서 hard contact / over-pushing / large swing 위험
- **Anti-sway crane control**: lifting mechanism을 직접 제어하는 모델 기반 방식 → humanoid처럼 외부 접촉으로만 영향을 주는 상황엔 직접 적용 불가

---

## 2. 방법론 심층 분석

### 2.1 3단계 파이프라인

| 단계 | 입력 | 출력 | 학습 여부 |
|---|---|---|---|
| (1) VR teleoperation | PICO 헤드셋 + 컨트롤러 | (o_t, u^demo_t, a^motor_t) 궤적 | 데이터 수집 |
| (2) Supervised VLA finetuning | (1)의 50개 데모 | High-level command increment chunk ΔU_t | VLM 최종 4 layer + action expert + 신규 modality encoder만 학습 |
| (3) Iterative batched RL | (2) policy의 자율 rollout 30개 + reward | Initial-noise steering vector | Actor-critic만 학습 (VLA 동결) |

### 2.2 Observation & Action 공간

**Observation**:
```
o_t = { I^ego,rgb_t , D^ego_t , I^side,rgb_t , x^r_t , ℓ , c^nav_{t-1} }
```
- I^ego,rgb / D^ego: humanoid onboard RGB + depth
- I^side,rgb: 외부 side-view (payload-target 관계 포착)
- x^r_t: joint state + base/IMU
- ℓ: language instruction
- c^nav_{t-1}: 직전 navigation command (modality로 명시 추가)
- **Contact-force sensing이나 payload state 직접 입력 X**

**Action chunk** (horizon H):
```
ΔU_t = πθ(o_t) = (Δu_t, Δu_{t+1}, ..., Δu_{t+H-1})
Δu_τ = ( Δy^head_τ , Δy^L_τ , Δy^R_τ , Δc^nav_τ , Δh^base_τ )
```
- Head / Left hand / Right hand pose 증분 + navigation command 증분 + base height 증분
- **Joint torque가 아닌 planner-level command**
- 실행: `u_t = u_{t-1} + Δu_t` 누적 후 `a^motor_t = π_wbc(x^r_t, u_t)` (GR00T WBC는 고정)

### 2.3 High-Level VLA Policy (GR00T N1.6 수정)

- **Backbone**: GR00T N1.6 (foundation policy)
- **추가 modality**: navigation command를 VLM에 추가 입력, depth + ego-velocity를 flow-matching action expert에 추가
- **Freeze 정책**: 대부분의 VLM layer 동결, **final 4 VLM layers + diffusion-transformer action expert + 신규 modality encoder만** finetune
- 사전학습된 visual-language / action prior를 보존하면서 humanoid hoisting에 적응

### 2.4 Flow-Matching-Steering RL Refinement (핵심 기여)

기존 RL이 정책 파라미터를 직접 업데이트하는 것과 달리, HOIST는:

1. VLA policy `πθ`는 **완전 동결**
2. Rollout 중 flow-matching action expert가 사용한 **initial noise vector**를 기록
3. **Actor-critic 모듈**이 주어진 state에서 어떤 initial noise를 써야 할지 **deterministically** 예측 (mean+variance를 sampling하지 않고 결정론적)
4. Reward = `weighted (translational placement error) + (raw angular error)` — 단순한 가중 합산, 학습된/shape된 reward model 사용 X
5. Offline actor-critic update → 갱신된 noise-steering module 재배포 → 다음 batch rollout → 반복 (iterative batched)
6. **VLM feature caching**: 동일한 visual-language observation의 반복 인코딩을 피하기 위해 frozen VLM의 feature를 저장/재사용

→ [15] Wagenmaker et al. (arXiv:2506.15799) "steering your diffusion policy with latent space RL" 기반

---

## 3. 데이터 / 시스템

| 항목 | 값 |
|---|---|
| Robot | Real humanoid (GR00T Whole-Body Control deployment stack) |
| Teleop interface | PICO VR headset + 컨트롤러 |
| Whole-body controller | Fixed decoupled WBC [22] (GR00T WBC) — 학습 X |
| Foundation VLA | GR00T N1.6 [23] modified |
| Demonstrations | VLA-50: 50개, VLA-80: 80개 |
| RL rollouts (HOIST) | 최대 30개 same-domain rollouts |
| Sensing | Onboard RGB + depth + external side-view RGB + proprioception + payload IMU (분석용) |
| 명시적 force sensing | 없음 (정책 입력에 contact force 포함 X) |

---

## 4. 실험 결과 (Paper Tables 1–3 직접 확인)

### 4.1 RL Progression (Table 1)

**Simulation**:

| Method | ∆x (cm) | ∆y (cm) | ∆ψ (deg) | |∆x|+|∆y| (cm) |
|---|---|---|---|---|
| Human Expert Teleop | 4.35 | 3.00 | 8.48 | 7.35 |
| VLA-50 | 21.56 | 0.88 | 3.85 | **22.44** |
| VLA-50 + 10 RL | 15.09 | 1.83 | 4.58 | 16.92 |
| VLA-50 + 20 RL | **1.72** | 0.82 | **0.29** | **2.54** |
| VLA-50 + 30 RL | 5.16 | 1.09 | 2.76 | 6.25 |

**Real platform**:

| Method | ∆x (cm) | ∆y (cm) | ∆ψ (deg) | |∆x|+|∆y| (cm) |
|---|---|---|---|---|
| VLA-50 | 1.60 | 7.68 | 14.5 | 9.28 |
| VLA-50 + 10 RL | 2.15 | 5.22 | 21.0 | 7.37 |
| VLA-50 + 20 RL | 4.20 | 2.49 | 12.1 | 6.69 |
| VLA-50 + 30 RL | 1.64 | 4.74 | 28.9 | **6.38** |

### 4.2 RL vs Additional Demos (Table 2)

| Domain | Method | Demos | RL | ∆x | ∆y | ∆ψ | Manhattan |
|---|---|---|---|---|---|---|---|
| Sim | VLA-50 | 50 | 0 | 21.56 | 0.88 | 3.85 | 22.44 |
| Sim | VLA-80 | 80 | 0 | 15.69 | 2.89 | 12.14 | 18.58 |
| Sim | **HOIST** | 50 | 30 | **5.16** | 1.09 | 2.76 | **6.25** |
| Real | VLA-50 | 50 | 0 | 1.60 | 7.68 | 14.5 | 9.28 |
| Real | VLA-80 | 80 | 0 | 0.54 | 8.03 | 27.0 | 8.57 |
| Real | **HOIST** | 50 | 30 | 1.64 | 4.74 | 28.9 | **6.38** |

→ Manhattan 기준 simulation **22.44 → 6.25** (HOIST), real **9.28 → 6.38**.
**Abstract 헤드라인**: vs pure VLA rollouts, **translational 19.9 cm 감소, raw angular 3.56° 감소**.
→ 같은 데모 수(50)에서 RL refinement가 데모 30개를 추가로 더 모으는 것보다 효과적.

### 4.3 Modality Ablation (Table 3, RL 적용 전)

| Variant | Depth | Nav cmd | ∆x | ∆y | ∆ψ | Manhattan |
|---|---|---|---|---|---|---|
| Base VLA | – | – | 1.54 | 10.52 | 26.6 | 12.06 |
| + Depth | ✓ | – | 0.78 | 8.78 | 14.8 | 9.57 |
| + Nav cmd | – | ✓ | 1.30 | 9.90 | 32.5 | 11.20 |
| + Nav + Depth | ✓ | ✓ | 1.60 | 7.68 | 14.5 | **9.28** |

→ Depth (contact geometry) + Nav command (현재 base motion 인지)의 결합이 최선. RL refinement 시작점으로 사용.

### 4.4 Residual Motion (Fig. 5, IMU 분석)

- HOIST vs imitation-only: accelerometer 상승 + **gyroscope 약 절반**으로 감소 → 더 활발한 corrective adjustment + 더 작은 각운동
- 두 학습된 정책 모두 human teleoperation보다 가속도/자이로 응답이 낮음 → 사람이 더 자주 closed-loop 보정을 수행

---

## 5. Ablation 핵심

- **VLA-50 → HOIST (sim)**: Manhattan 22.44 → 6.25 (3.6× 개선)
- **VLA-80 vs HOIST (sim, 동일 budget 관점)**: Manhattan 18.58 → 6.25 — 데모 +30개보다 RL rollout +30개가 우월
- **Depth 추가 단독**: Manhattan 12.06 → 9.57 (geometry 인지 향상)
- **Nav command 단독**: Manhattan 12.06 → 11.20 (단독 효과는 미미하지만 결합 시 시너지)
- **30 RL rollouts in sim**: 20 rollouts (Manhattan 2.54)에서 일시적 최적, 30에서 6.25로 약간 상승 — 과학습/탐색 발산 가능성 시사

---

## 6. Related Work 비교

| 접근 | 동작 |
|---|---|
| Anti-sway crane control [6,7,8,9] | Lifting mechanism 직접 제어 — humanoid의 외부 접촉 시나리오엔 부적용 |
| GR00T N1 [23] / SONIC [22] / Visual imitation HMC [21] | Foundation humanoid policy — hoisting / suspended payload 미겨냥 |
| FALCON [26] / ResMimic [27] / CHIP [28] | Loco-manipulation, residual / hindsight perturbation — rigid/quasi-static object 중심 |
| Diffusion-policy steering RL [15] | latent-space RL, HOIST가 base method로 차용 |
| **HOIST** | **VR teleop imitation + flow-matching noise steering RL + 고정 WBC = suspended payload positioning** |

---

## 7. Limitations (저자 명시)

1. **Low-level WBC가 hoisting에 fine-tune되지 않음** — 정책 인터페이스(nav cmd / head / base height / 양손)만 사용 가능, payload weight에 따른 force 적응 X
2. 결과적으로 **무거운 payload에서 충분한 힘 생성 불가** — heavy load 처리 한계

추가 (리뷰어 관점):
3. 실제 robot 평가가 한 가지 humanoid 플랫폼에 한정
4. RL이 **same-domain rollouts**만 사용 — cross-domain / cross-payload 일반화 미검증
5. Reward가 단순한 weighted sum (translational + angular error) — 더 풍부한 reward shaping 여지
6. 30 RL rollout이 sim에서 20보다 약간 나쁨 → hyperparameter 민감성

---

## 8. 종합 평가

| 항목 | 평점 (5점) |
|---|---|
| 혁신성 | 4.0 (humanoid hoisting을 새 problem으로 정식화, flow-matching noise steering의 hoisting 적용) |
| 재현성 | 2.5 (코드 미공개, GR00T N1.6 + GR00T WBC 의존) |
| 실험 폭 | 3.5 (sim + real, 3개 table, modality ablation, IMU residual analysis) |
| 이론적 깊이 | 3.0 (방법론 자체는 [15] 차용; humanoid hoisting 정식화가 본 기여) |
| 실용성 | 4.0 (real humanoid 검증, 19.9 cm / 3.56° 감소) |

**총평**: "VLA finetune + RL이 데모 추가보다 placement bias를 더 잘 교정한다"는 단순하지만 실용적인 메시지를 humanoid hoisting이라는 새 도메인에서 명료히 입증. 가장 인상적인 부분은 **VLA를 동결하고 flow-matching의 initial noise만 steering**하는 sample-efficient 설계 — 단 30 rollout으로 실 humanoid에서 의미 있는 개선을 얻음.

---

## 9. 예상 세미나 질문

> ❓ **VLA-50 + 20 RL이 sim Manhattan 2.54로 30 RL의 6.25보다 좋은데, 왜 HOIST를 30으로 보고하나?**
> Real platform에서 30이 가장 낮은 Manhattan (6.38)을 달성. Sim과 real의 최적 rollout 수가 다름 — domain-specific tuning이 필요함을 시사.

> ❓ **Initial noise를 deterministic하게 예측하면 exploration이 사라지지 않나?**
> 저자 설명: actor가 mean+variance를 출력하고 sampling하는 대신 steering vector를 결정론적으로 출력. Exploration은 새 batch rollout 수집 시 환경 stochasticity와 frozen flow-matching의 짧은 horizon noise로 확보됨.

> ❓ **GR00T WBC를 고정한 것이 heavy payload limitation의 원인 아닌가?**
> 정확히 그렇다. 저자가 Limitation에서 명시: 정책이 force-adaptive하게 WBC를 finetune 불가 → 무거운 payload에서 force 생성 부족.

> ❓ **Side-view 외부 카메라가 필요하다는 점은 실 배포에서 제약 아닌가?**
> Payload-target 관계를 포착하기 위한 보조 입력. Onboard ego camera만으로 처리하는 후속 연구 가능.

> ❓ **VLA-80이 real ∆x에서 0.54로 가장 작은데, 왜 HOIST가 더 낫다고 주장?**
> Manhattan = |∆x| + |∆y|로 평가. VLA-80은 ∆x 작지만 ∆y=8.03으로 큼 (Manhattan 8.57). HOIST는 ∆x 1.64 + ∆y 4.74 = 6.38로 종합 더 작음.

---

## 10. 코드 & 재현

- **Code**: 미공개
- **의존 모듈**: GR00T N1.6 (foundation policy) + GR00T Whole-Body Control [24] (NVlabs/GR00T-WholeBodyControl)
- **Teleop**: PICO VR headset
- **RL base**: Wagenmaker et al. arXiv:2506.15799 (diffusion-policy latent steering)

---

## 11. 데이터셋 / 후속 연구

- **데이터**: VR teleoperation으로 자체 수집한 humanoid hoisting demonstrations (50 또는 80 trajectory)
- **후속 방향** (저자 제시):
  - 다양한 payload / suspension 조건 확장
  - Explicit safety constraint 추가
  - Coordinated humanoid-hoist control (humanoid + crane operator 협력)

---

## 12. 결론

HOIST의 세 가지 핵심 기여:
1. **Humanoid hoisting** = whole-body loco-manipulation의 새 task로 정식화 (suspended, underactuated, oscillatory payload)
2. **VR teleop + GR00T N1.6 finetune + flow-matching noise-steering RL**의 3단계 파이프라인 — VLA + WBC 동결 + 작은 module만 RL 학습으로 sample efficiency 확보
3. 실 humanoid에서 pure VLA 대비 **translational 19.9 cm / angular 3.56° 감소** 입증, 동일 demo 예산에서 데모 추가보다 RL refinement 우월

"Foundation VLA + 고정 WBC + 작은 latent-space RL module"이라는 조합은 다른 contact-rich humanoid 작업으로 확장 가능한 일반적 레시피.

---

<!-- VERIFIED: pdf -->
