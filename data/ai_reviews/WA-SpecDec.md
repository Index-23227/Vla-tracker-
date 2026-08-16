# WA-SpecDec: World-Aware Speculative Decoding for Vision-Language-Action Models

> **한 줄 요약**: VLA speculative decoding의 relaxed acceptance가 "장면과 무관한(scene-agnostic)" 고정 토큰 거리 허용치를 쓴다는 점을 문제 삼고, **frozen Wan 2.2 VAE latent에서 유도한 World-Aware Bias(WAB)를 prefill 이전 visual patch embedding에 더해** draft와 target이 같은 물리 인지형 prefill hidden state를 공유하도록 학습(WAB + LoRA + next-frame latent 보조손실, 이후 draft distillation). acceptance rule은 그대로 두면서 LIBERO 4 suite 평균 65.4 → **71.0%**(AR target 72.2%), matched-success 기준 **1.5x** 가속, NCF 평균 **18.6%** 감소.

---

## 1. 배경 및 동기

- VLA 정책은 관측·지시로부터 action token을 **autoregressive**하게 디코딩하므로 closed-loop latency가 target model forward pass 반복 횟수에 지배된다.
- Speculative decoding(Leviathan et al., 2023)은 경량 draft가 여러 토큰을 제안하고 target이 병렬 검증해 target round 수를 줄인다. VLA에서는 exact token match가 과하게 엄격하다는 관찰에서 **relaxed acceptance**(Spec-VLA, KERV, HeiSD)가 등장했다: `Accept_rho(â_i) = 1[d(â_i, a*_i) <= rho]`.
- 핵심 문제 제기: 이 허용치 rho는 **전역(global)** 인데, 허용된 편차의 물리적 비용은 **상태 의존적**이다. 자유공간에서 무해한 15 cm 편차가 접촉 직전 1 cm 상황에서는 충돌·grasp 실패를 유발한다(Figure 1). 결국 성공률을 지키려면 rho를 보수적으로 유지해야 하고, 이는 accepted length와 speedup을 제한한다.
- 시각 백본(SigLIP, DINOv2)의 사전학습 목적은 object pose, gripper–object 거리, contact phase 같은 **제어 임계 물리량**을 보존하도록 설계되지 않았다는 표현적 병목(representational bottleneck)을 지적.

📌 [Figure 1 삽입] — 같은 draft–target 편차가 자유공간에서는 안전, 접촉 근처에서는 위험

---

## 2. 방법론 심층 분석: World-Aware Bias(WAB) 모듈

- frozen VAE latent에서 출발: `Z_t = E_vae(I_t) ∈ R^{B×H_z×W_z×C_vae}`. 구현상 **Wan 2.2 VAE encoder**를 latent compressor로만 사용(비디오 생성 파이프라인은 호출하지 않음).
- world token: `U_t = F_omega(ϕ_tok(Z_t)) ∈ R^{B×S×C}`, `S = H_z W_z`. ϕ_tok은 flatten + linear projection, F_omega는 **4-layer spatial VideoEncoder**.
- 공간 정렬 및 투영: `w_t = f_bias(SpatialAlign(U_t)) ∈ R^{B×N_v×H}` — latent grid를 2D로 되돌려 VLA visual patch 해상도로 보간 후 다시 flatten, 마지막에 VLA hidden width로 투영. 따라서 **토큰을 추가하지 않고** patch마다 위치별 물리 bias를 준다.
- 보조 목적: training-only VideoHead `g_eta`가 현재 프레임 world token으로부터 다음 프레임 VAE latent map을 예측, `L_video = ||g_eta(U_t) − E_vae(I_{t+1})||^2 / (H_z W_z C_vae)`. pooled가 아닌 **전체 latent grid**에 대한 dense loss라 공간 해상도를 가진 단기 변화 단서를 보존하도록 유도.
- 총 손실: `L = L_action + lambda * L_video` (구현에서 lambda = 0.1). VideoHead는 학습 후 폐기되어 추론에서는 현재 프레임만 사용.

---

## 3. 방법론 심층 분석: World-Aware Target과 공유 prefill

- 주입 위치가 핵심: `V^wa_t = V_t + w_t`(element-wise), `P^wa_t = [T_x; V^wa_t]`, `h^wa_t = Prefill_{theta+Δtheta}(P^wa_t)`. 즉 **prefill 계산 이전**에 더해져 transformer의 Q/K/V·FFN 투영과 KV cache에 물리 방향이 반영된다.
- 시퀀스 길이가 그대로이므로 target self-attention 추가 비용이 없다.
- acceptance rule은 손대지 않는다: `Accept^wa_rho(â_i) = 1[d(â_i, a^wa_i) <= rho]`로 **참조 분포만** 세계 인지형으로 바뀐다. 저자들은 vanilla VLA에 대한 분포 보존을 주장하지 않고 `p^wa_theta`에 대해서만 주장한다.
- Draft: world-aware target을 freeze한 뒤 `L_draft = E[KL(p^wa_theta || q_phi(·|h^wa_t, a_<i))]`로 distill. 제안과 검증이 **같은 h^wa_t**를 공유하는 것이 설계의 요체(draft만 강화하는 접근과 대비).

📌 [Figure 2 삽입] — WAB 경로, LoRA target, 보조 VideoHead(학습 전용), draft/verify 흐름

---

## 4. 학습·추론 절차 (Algorithm 1)

- 학습: 미니배치마다 Z_t → U_t → w_t → V^wa_t → h^wa_t 계산 후 L_action(demonstration action token NLL) + lambda·L_video로 ϕ_tok, F_omega, f_bias, g_eta, LoRA Δtheta 갱신. VAE와 base visual stack, base VLA θ는 frozen.
- 이후 target을 freeze하고 draft q_phi 학습.
- 추론: g_eta 폐기, 현재 (x, I_t)만으로 h^wa_t 1회 계산 → draft가 블록 길이 gamma만큼 제안 → target이 1회 병렬 검증 → 허용 prefix 실행, **첫 거부 토큰은 target 토큰으로 대체** 후 다음 라운드.

---

## 5. 실험 설정

- Target 정책: 주 실험은 fine-tuned **OpenVLA**, 추가로 **ActionCodec**(LIBERO), **UniVLA**(SIMPLER-Env).
- 베이스라인: **Standard SpecDec = Spec-VLA**(world-aware conditioning 없음), 동일 target·동일 디코딩 설정.
- Verifier 3종: rho 기반 토큰 거리, **KERV**(kinematic), **HeiSD**(token + kinematic hybrid).
- 벤치마크: LIBERO 4 suite(Spatial/Object/Goal/Long), 태스크당 50 rollout → suite당 500 에피소드; SIMPLER-Env 4 태스크(Carrot/Stack/Spoon/Eggplant).
- 지표: task success, accepted length, wall-clock speedup(draft 제안·target 검증·**WAB 오버헤드**·cache update 포함), 그리고 신규 지표 **NCF**(실패 에피소드 중 gripper fingertip–대상/수납부 최소거리가 delta=2 cm 미만으로 내려간 비율).

---

## 6. 주요 결과: LIBERO (OpenVLA target, Table 1)

| Verifier | Method | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| – | AR OpenVLA | 82.8 (1.00x) | 69.8 (1.00x) | 78.2 (1.00x) | 57.8 (1.00x) |
| rho | Standard SpecDec | 79.4 (1.26x) | 63.0 (1.13x) | 69.6 (1.23x) | 49.4 (1.18x) |
| rho | **WA-SpecDec** | **83.0 (1.54x)** | **68.8 (1.24x)** | **75.8 (1.32x)** | **56.2 (1.21x)** |
| KERV | Standard SpecDec | 80.2 (1.52x) | 63.4 (1.54x) | 70.8 (1.61x) | 48.6 (1.41x) |
| KERV | **WA-SpecDec** | **83.4 (1.59x)** | **68.4 (1.64x)** | **77.6 (1.67x)** | **52.8 (1.45x)** |
| HeiSD | Standard SpecDec | 77.4 (1.74x) | 68.0 (2.26x) | 70.8 (2.10x) | 50.4 (1.71x) |
| HeiSD | **WA-SpecDec** | **80.4 (1.77x)** | **70.6 (2.31x)** | **78.0 (2.12x)** | **57.0 (1.79x)** |

- 세 verifier 계열 모두에서 성공률·속도가 동시에 개선(속도를 성공률과 교환하지 않음). 예: rho에서 Object 63.0 → 68.8, Goal 69.6 → 75.8, Long 49.4 → 56.2.
- NCF: Table 1의 모든 짝 비교에서 평균 **18.6% 감소**(rho·Spatial 78.6 → 72.9, HeiSD·Goal 66.4 → 43.6 등).

---

## 7. 강한 target으로의 확장과 SIMPLER-Env 일반화

- **ActionCodec(LIBERO-Goal, Table 2)**: AR 95.8%. rho에서 88.6 → **95.2%**(1.18x → 1.25x), KERV에서 91.2 → **95.6%**(1.57x → 1.72x, AR과 0.2%p 차), HeiSD에서 88.0 → **94.2%**(2.41x → **2.63x**). NCF도 88.7 → 70.8 / 84.1 → 77.3 / 86.7 → 72.4로 감소.
- **SIMPLER-Env + OpenVLA(Table 3)**: 4태스크 평균 rho 14.6 → **24.0%**(1.25x → 1.46x), KERV 12.5 → **27.1%**(1.53x → 1.76x), HeiSD 19.8 → **30.2%**(2.06x → **2.64x**).
- **SIMPLER-Env + UniVLA(Table 4, 성공률이 분수로 표기)**: rho 0.386 → **0.573**, KERV 0.386 → **0.562**, HeiSD 0.490 → **0.625**(2.13x → 2.59x). 여러 조건에서 AR UniVLA(0.625/0.208/0.583/0.833)를 상회하거나 동률.

---

## 8. Robustness: 허용치 rho와 NCF 임계값

- **rho 스윕(Table 5, 4 suite 평균)**: Spec-VLA 67.7 → 65.4 → 61.2 → 55.1 (rho=5/10/15/20)로 급격히 붕괴하는 반면 WA-SpecDec은 71.3 → 71.0 → 67.0 → 63.5로 완만하게 저하. 동일 acceptance rule 아래 비교이므로 이득이 "규칙 변경"이 아니라 **참조 분포 개선**에서 온다는 근거.
- **NCF 임계값 스윕(Table 7, suite 평균)**: 1/2/3/4 cm 모두에서 WA-SpecDec이 Standard SpecDec보다 낮고, 평균 각각 12.54 / 11.77 / 10.87 / 8.03 %p 감소. 특히 접촉 기하가 가장 중요한 1–2 cm에서 개선폭이 크다. 기본 2 cm는 LIBERO gripper finger 폭에서 유래한 물리적 근거를 가진 값이며 결과 최적화를 위해 고른 값이 아님을 명시.

---

## 9. Ablation (Table 6, rho=10, 4 suite 평균)

| 변형 | Success | Accepted Len. | Speedup |
|---|---|---|---|
| AR target | 72.2% | – | 1.00x |
| WA target-only | 73.5% | – | 0.97x |
| Spec-VLA | 65.4% | 3.59 | 1.20x |
| WA w/o video loss | 69.4% | 4.09 | 1.31x |
| **WA-SpecDec (full)** | **71.0%** | **4.22** | **1.34x** |

- WAB만 붙인 AR 변형은 신뢰도는 소폭 올리지만 WAB 오버헤드로 **오히려 느려진다(0.97x)** — 가속은 speculative 경로에서만 나온다는 정직한 분해.
- video loss 제거 시 성공률·accepted length가 함께 하락 → 공유 world-aware state와 **예측적 latent 감독**이 모두 필요.

---

## 10. 강점

- **개입 위치 선택이 영리하다**: acceptance rule이나 post-hoc acceptor를 건드리지 않고 prefill 이전 visual embedding에만 bias를 더해, 기존 relaxed verifier(rho/KERV/HeiSD) 어디에나 끼워 넣을 수 있는 직교적 설계.
- 토큰을 추가하지 않으므로 target self-attention 비용이 늘지 않고, 보고된 speedup에 WAB 오버헤드가 이미 포함되어 있다.
- **NCF라는 실패 위치 지표**를 도입해 "성공률 회복이 접촉 근처에서 일어난다"는 인과 주장을 임계값 스윕까지 곁들여 뒷받침.
- 동일 rho 조건 비교(Table 5)로 "규칙을 느슨하게 만든 덕"이 아님을 통제.

---

## 11. 한계 및 논의

- **분포 보존을 포기**: 보존은 vanilla VLA가 아니라 world-aware target `p^wa_theta`에 대해서만 성립. 즉 이 방법은 순수 추론 가속이 아니라 정책 자체(WAB + LoRA)를 재학습하는 것이며, AR 대비 성능 차이는 여전히 존재(rho=10 71.0 vs 72.2).
- 절대 성능은 baseline 정책에 종속. SIMPLER-Env OpenVLA 성공률은 12–42% 범위로 낮고, UniVLA 표는 분수, OpenVLA 표는 백분율로 표기가 혼재해 비교 시 주의가 필요.
- **실기 로봇 실험이 없다**(윤리 진술에서도 시뮬레이션 전용임을 명시). 접촉 안전성이 논지의 핵심인데 검증은 시뮬레이터 상태 기반 NCF에 의존.
- 파라미터 수, 학습 시간, WAB 지연 시간의 절대값, gamma·rho 세부 설정 등 정량 구현 수치가 부족하며 코드도 accept 시 공개 예정.
- VAE latent가 정말 "물리"를 담는지에 대한 직접 증거(pose/contact probing 등)는 없고, next-frame latent 예측 손실이라는 대리 목표로만 정당화된다.

---

## 12. 종합 평가

- **기여의 성격**: 단순 런타임 트릭이 아니라 **학습되는 산출물(WAB 모듈 + LoRA target + distilled draft)** 을 가진 프레임워크. relaxed speculative decoding의 신뢰도 병목을 "acceptance rule"이 아니라 "공유 conditioning state" 문제로 재정의한 관점 전환이 가장 큰 기여.
- **실용성**: verifier 계열에 직교적이고 토큰 수를 늘리지 않아 기존 VLA 가속 스택에 얹기 쉽다. 다만 target 재학습이 필요하므로 "학습 없는 가속"을 원하는 사용자에게는 진입 비용이 있다.
- **재현성**: 500 에피소드/suite, 동일 splits·rollout budget 명시는 좋으나 코드 미공개와 하드웨어/시간 정보 부재가 감점 요소.
- **총평**: 접촉 민감 구간에서의 실패를 지표화하고 그 지표를 개선했다는 점, 그리고 rho 스윕으로 이득의 출처를 통제한 점에서 설득력 있는 논문. 실기 검증과 구현 세부 공개가 뒤따르면 VLA 가속 연구의 표준 비교군이 될 만하다.

<!-- VERIFIED: pdf -->
