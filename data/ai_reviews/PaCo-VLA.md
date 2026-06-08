# PaCo-VLA: Passivity-Shielded Compliance Prior for Contact-Rich VLA Manipulation

> **한 줄 요약**: OpenVLA를 black-box proposal source로 두고, 250 Hz **proposal-independent passivity shield**가 (i) admittance 파라미터 box 사영 + (ii) sampled passivity margin + (iii) energy tank로 양수 M/K jump만 차감하는 contact-rich VLA 인터페이스. AUBO-i5 실 로봇 connector insertion 90%(vs 비-semantic 40%, energy-tank 60%), 0/1000 시뮬레이션 passivity violation, paired counterfactual 7개 ablation 전원 0% — VLA의 의미 가치를 인과적으로 분리해 입증.

---

## 1. 배경 및 동기

### 기존 VLA의 contact-rich 한계
- **VLA가 motor command를 직접 출력**하면 (1) 안전 보장 부재, (2) classical compliance 컨트롤러와의 호환성 부재, (3) 학습 분포 외 contact 상황에서 불안정
- **순수 classical admittance control**은 안전하지만 semantic 적응이 없어 다양한 작업 일반화 실패

### 핵심 아이디어
- VLA를 motor command 생성기가 아닌 **task-level compliance proposal source**로 재해석
- VLA 출력: (semantic binding, task stage, recovery cue, **diagonal admittance schedule M/D/K**)
- Proposal-independent **passivity shield**가 250 Hz로 모든 proposal을 동일 contract로 처리

---

## 2. 방법론 심층 분석

### 2.1 인터페이스 계층화

| 레이어 | Frequency | 역할 |
|---|---|---|
| Camera | 10 Hz | RGB-D observation |
| **OpenVLA proposal** | 50 Hz | (binding, stage, recovery, M/D/K, validity) tuple |
| **Passivity shield** | 250 Hz | box projection + margin + energy tank |
| Cartesian admittance port | 250 Hz | 실제 모터 명령 |

### 2.2 Passivity Shield 3단계

1. **Box projection**: M_i ∈ [0.1, 10], D_i ∈ [5, 500], K_i ∈ [0, 2000]
2. **Sampled passivity margin**: `2D_i − (M_i − M_{k-1,i})/Δt_k ≥ 2·d_margin` (d_margin=2)
3. **Energy tank**: 양수 M/K jump만 tank S_k에서 차감, bounded interpolation β_k ∈ [0,1], E_min=0.5, E_max=20, α=1.2

### 2.3 Sampled Storage Inequality

학습/classical/지연된/random/recovery proposal **모두 동일 shield**를 통과 → applied-command contract가 **provably proposal-independent**. 다음이 성립:

`S_k − S_{k-1} ≤ F_k^T v_k Δt_k + ε_k^num`

→ Paired counterfactual로 **semantic 가치의 인과 귀속(causal attribution)** 가능.

---

## 3. 데이터/시스템

| 항목 | 값 |
|---|---|
| Robot | AUBO-i5 6-DoF arm + AG-160-95 gripper |
| F/T sensor | KWR75B 6축 |
| Cameras | wrist RealSense D435i + external D415 |
| Target | 4.5 mm insertion depth, 4 N contact limit |
| Compute | 2× RTX 4090 (OpenVLA bfloat16 inference) |
| Backbone | OpenVLA 7B (black box) |

---

## 4. 실험 결과 (Paper Tables 1–6 직접 확인)

### 4.1 Real-Robot Connector Insertion (Table 4)

| Method | Task Success | Physical Contact | Lateral Err (mm) | F/T 위반 |
|---|---|---|---|---|
| **PaCo-VLA** | **90.0%** | **100%** | **0.181** | **0** |
| Energy-tank baseline | 60.0% | — | 0.617 | — |
| Nonsemantic baseline | 40.0% | — | 0.622 | — |

- VLA accept fraction: 96.8% (대부분 proposal 그대로 통과)
- Guarded recovery: 2.7%

### 4.2 EV Charging-Gun Transfer

- PaCo-VLA Task Success: **53.3%** (15 trials)

### 4.3 시뮬레이션 1000-trial Adversarial Runtime-Contract

- PaCo-VLA: **0/1000 violations**
- Without projection: 50.5% violation rate

### 4.4 시뮬레이션 Classical-Control 192-trial 매치 (Table 1)

| Controller | Task Success |
|---|---|
| **PaCo-VLA proposals** | **0.318** |
| Energy tank | 0.260 |
| Rule admittance | 0.255 |
| Passivity observer | 0.250 |
| Fixed admittance | 0.000 |
| Force scheduling | 0.000 |

### 4.5 Paired Counterfactual Semantic Trials (Table 2, 1584 paired)

| Condition | Task Success |
|---|---|
| **Oracle** | **0.993** |
| **Live L+V** | **0.979** |
| Shuffled language | 0.000 |
| Masked image | 0.000 |
| Wrong object | 0.000 |
| Contradictory language | 0.000 |
| Same geom, diff instr | 0.000 |
| Same instr, diff target | 0.000 |
| Stale context | 0.000 |
| Random proposal | 0.000 |

→ **7개 counterfactual 전원 0%** = semantic input이 가치의 source임을 인과적으로 입증.

### 4.6 Direct Action Interface (Table 3, N=144 매트릭스)

| Method (direct learned action under same shield) | Task Success |
|---|---|
| **PaCo-VLA** (proposal interface) | **0.7361** |
| OpenVLA + shield | 0.0000 |
| OpenVLA + projection | 0.0000 |
| Diffusion Policy | 0.0139 |
| ACT | 0.0000 |
| Octo | 0.0000 |

→ Direct action token 출력은 동일 shield 아래에서 거의 작동 안 함. **proposal interface가 핵심**.

---

## 5. Ablation 핵심

- **VLA proposal 제거 → nonsemantic baseline**: 90% → 40% (Table 4)
- **Energy tank만 단독 사용 → energy-tank baseline**: 90% → 60%
- **Passivity shield 제거 (1000-trial 적대적 sampling)**: 0% → 50.5% violation
- **7가지 semantic counterfactual 전원 0%** — semantic 입력이 제거되면 어떤 prior도 작동 안 함

---

## 6. Related Work 비교

| 접근 | 동작 |
|---|---|
| OpenVLA direct motor token | Contact-rich에서 적용 시 0% (Table 3) |
| Diffusion Policy / ACT / Octo | Direct interface, contact 안전 보장 X |
| Rule admittance / Fixed admittance | 안전, but semantic 적응 X (≤26% Table 1) |
| **PaCo-VLA (proposal + shield)** | **31.8% (sim) / 90% (real) + 0/1000 violations** |

---

## 7. Limitations

1. **Code 미공개** (paper 시점)
2. **Single arm, single gripper**: AUBO-i5만 검증 — bimanual 일반화 미검증
3. **OpenVLA 의존**: 다른 backbone에서의 proposal 품질 미확인
4. **Shield 하이퍼파라미터** (boxes, d_margin, E_min/max, α): task-specific tuning 가능성

---

## 8. 종합 평가

| 항목 | 평점 (5점) |
|---|---|
| 혁신성 | 4.5 (인터페이스 재설계 + 인과 검증 패러다임) |
| 재현성 | 3.0 (코드 미공개) |
| 실험 폭 | 5.0 (real + sim + 1584-trial counterfactual + adversarial) |
| 이론적 깊이 | 4.5 (sampled storage inequality, proposal-independent contract) |
| 실용성 | 4.0 (real connector insertion 90% + 0 F/T 위반) |

**총평**: VLA의 motor-command 인터페이스를 의문시한 데서 출발해 **proposal + shield**로 contact-rich manipulation에 깔끔한 해답을 제시. 7개 counterfactual 0%로 semantic 가치를 인과적으로 분리한 점이 가장 인상적.

---

## 9. 예상 세미나 질문

> ❓ **VLA를 black box로 쓰면 LLM 추론 비용이 50 Hz × OpenVLA 7B inference인데 latency 부담은?**
> Shield(250 Hz)와 VLA(50 Hz)가 분리되어 있어 stale-proposal threshold 0.30 s 내에서는 안전. OpenVLA inference timeout 0.20 s 내 fallback proposal로 처리.

> ❓ **Diagonal admittance M/D/K만 출력하면 일반 manipulation에는 부족하지 않나?**
> Connector insertion / EV charging gun처럼 1-DoF/2-DoF dominant contact에서는 diagonal로 충분. Full Cartesian / off-diagonal는 향후 과제.

> ❓ **Paired counterfactual에서 7개 ablation 전원 0%는 너무 깔끔하지 않은가?**
> 동일 shield 아래에서 의미 정보만 변형 → contact-rich 작업이 본질적으로 semantic guidance 없이 불가능한 task로 설계됨. Trial 수(144 paired/condition)도 충분.

> ❓ **Direct interface에서 OpenVLA + shield가 0%면 shield가 너무 보수적인가?**
> 그렇지 않음. Diagonal admittance schedule을 직접 출력하지 못하는 모델은 shield의 box 사영이 모든 명령을 안전 박스 내부로 강제 → 효과적으로 정지. proposal 자체가 admittance 형식이어야 함.

---

## 10. 코드 & 재현

- **Code**: 미공개 ("code will be available soon" per arXiv)
- **하드웨어**: AUBO-i5 + AG-160-95 gripper + RealSense D435i/D415 + KWR75B
- **Shield hyperparameters**: paper Table 5 (boxes, d_margin=2, E_min=0.5, E_max=20, α=1.2, β_k ∈ [0,1])

---

## 11. 데이터셋 / 후속 연구

- 별도 학습 데이터 없음 (학습 불필요, shield는 parameter-free)
- 후속: full Cartesian admittance, bimanual, 다른 VLA backbone 검증

---

## 12. 결론

PaCo-VLA는 (1) VLA를 proposal source로 재정의, (2) proposal-independent 250 Hz passivity shield로 안전 보장, (3) 1584-trial counterfactual로 semantic 가치를 인과 입증의 세 기여를 가진다. **contact-rich VLA 인터페이스의 새 표준**이 될 가능성 있음.

---

<!-- VERIFIED: pdf -->
