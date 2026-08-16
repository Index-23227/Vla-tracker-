# RecoverFly: A Failure-Aware Reinforcement Learning Post-Training Framework for Aerial Vision-Language Navigation

> **한 줄 요약**: Behavior cloning으로 학습된 end-to-end UAV-VLA 정책(AerialVLA / OpenVLA-7B + LoRA)에 **token-level PPO + 동적 실패 리플레이 + 2단계 long-tail 씬 커리큘럼 + reference-policy KL**을 결합한 온라인 RL post-training을 적용해, TravelUAV 세 스플릿 전부에서 SOTA를 달성(Seen SR 47.96 → 56.33, Unseen Map 37.58 → 42.97, Unseen Object 56.60 → 59.72). 총 롤아웃 예산은 학습셋 크기의 약 30%, 8×A100으로 21시간.

---

## 1. 배경 및 동기

### 문제 정의
- UAV-VLN은 자연어 지시 + 온보드 시각 관측으로부터 전진량·고도·yaw·착륙 결정을 연속적으로 산출해야 하는 대규모 3D 문제.
- AerialVLA 같은 end-to-end UAV-VLA는 waypoint/외부 detector/별도 착륙 모듈 의존을 없앴지만, 학습 목적이 **behavior cloning(BC)** 이라 자기 오차로 유발된 상태에서의 교정 신호가 전무하다(Ross et al. 2011의 compounding error).
- 항공 데이터는 비싸고 씬 분포가 심하게 long-tailed → 표준 샘플링은 빈번한 씬에 집중되고 희귀 씬 전이는 약해진다.
- 두 문제는 결합되어 있다: **교정이 가장 필요한 희귀/난이도 높은 케이스가 가장 덜 재방문된다.**

### 왜 단순 RL로는 부족한가 (논문이 제시한 4가지)
1. 실패 이벤트는 순간 행동이 아니라 **이전 행동들**의 결과일 수 있다 → credit assignment.
2. 학습 가치가 큰 미해결 실패 케이스가 다음 배치에서 **금방 사라진다**.
3. 씬 불균형이 최적화를 빈번 환경에 집중시킨다.
4. 무제약 업데이트는 기존 내비게이션 능력을 **퇴화**시킨다.

RecoverFly는 이 넷을 개별 트릭이 아니라 하나의 post-training 파이프라인으로 묶는다는 것이 주장의 핵심이다.

---

## 2. 방법론 심층 분석

### 2.1 텍스트 액션 표현
- 정책 컨텍스트 h_t = (지시 x, 관측 o_t), 정책은 액션 토큰 시퀀스 z_t = (z_{t,1},...,z_{t,K_t})를 autoregressive하게 생성.
- AerialVLA의 액션 문법 계승: **수치 제어 토큰 3개 + 선택적 LAND 토큰**. 디코딩 결과는 3-DoF 제어 a_t = <Δx_t, Δz_t, Δψ_t>(전진, 수직, yaw).
- 각 차원은 **99개 bin**으로 균등 양자화, 범위는 각각 [0,5], [-5,5], [-π,π]. 결정론적 디코더가 유효 시퀀스를 연속 제어로 사상.
- 롤아웃 생성과 정책 최적화에 **동일한 문법 제약**을 적용하고, 파싱 실패 시 안전 no-op으로 매핑 + invalid-action 페널티.
- 마스크 m_{t,k} ∈ {0,1}로 패딩 후 유효 토큰 위치만 선택, M_t = max(1, Σ_k m_{t,k}).

### 2.2 이벤트 인지 보상 + GAE
- r_t = clip(κ_p (D_{t-1} - D_t), r_min^prog, r_max^prog) + Σ_{e∈E} R_e · I[e_t = e]
- D_t는 목표까지 유클리드 거리, 이벤트 집합 E = {success, collision, stuck, away, early-stop, timeout}.
- 즉 **조밀한 거리 진척 보상 + 희소한 이벤트 보상**의 결합.
- 장기 지평에서 성공/실패는 핵심 행동보다 여러 스텝 뒤에 관측되므로 GAE(λ)로 지연 리턴을 전파: Â_t = Σ_l (γλ)^l δ_{t+l}, δ_t = r_t + γ(1-d_t)V_φ(h_{t+1}) - V_φ(h_t). 종료 전이(d_t=1)는 부트스트랩 제거, 비종료 절단은 마지막 가치로 부트스트랩.
- 베이스 VLA 정책에 **value head V_φ(h_t)** 를 붙이고 표준 clipped PPO value loss로 학습.

### 2.3 Token-level PPO (백본)
- 연속 액션 PPO를 직접 못 쓰는 이유: VLA는 디코딩된 명령에 대한 명시적 밀도가 아니라 **토큰 확률**을 파라미터화한다.
- 시퀀스 레벨 대안(토큰 확률 곱으로 joint ratio)은 **시퀀스 전체에 단일 importance weight와 clipping 결정**을 적용해버린다.
- RecoverFly는 clipped surrogate를 **유효 액션 토큰마다 따로** 적용하되, 같은 액션을 인코딩하는 토큰들은 **공유된 action-level advantage Â_t** 를 받는다.
  - ρ_{t,k}(θ) = π_θ(z_{t,k}|h_t,z_{t,<k}) / π_old(z_{t,k}|h_t,z_{t,<k})
  - J_PPO^token(θ) = E_t[ (1/M_t) Σ_k m_{t,k} min(ρ_{t,k}Â_t, clip(ρ_{t,k},1-ε,1+ε)Â_t) ]
- 즉 **native autoregressive 액션 공간에서의 토큰 단위 업데이트**를 유지하면서 액션 스텝 단위 학습 신호를 보존.

### 2.4 Dynamic Failure Replay (핵심 기여)
- 문제: on-policy 샘플링은 어려운 실패를 빠르게 희석시킨다.
- 기존 대안: HER(사후 relabeling), Prioritized Level Replay(학습 가능성 높은 레벨 재생).
- RecoverFly의 차별점: **오래된 궤적이 아니라 미해결 태스크 초기화(task initialization)를 저장**하고, 매 롤아웃을 **현재 정책으로 재생성**한다 → PPO의 on-policy 성질을 깨지 않으면서 정보량 큰 실패를 반복 학습.
- 풀 M = {ξ_i}, ξ_i = (id_i, c_i, f_i, σ_i, n_i): 태스크 id, 씬, 실패 유형, 상태 σ ∈ {active, solved, dropped}, 실패 재시도 횟수 n_i.
- 매 리셋에서 replay ratio η에 따라 fresh/failed 인스턴스를 샘플. 실패한 fresh는 풀에 추가/재활성화, 성공한 replay는 solved, N_max회 재시도 후에도 실패면 dropped.

### 2.5 Two-Stage Long-Tail Scene Curriculum
- 실패 리플레이는 **씬 수준 불균형은 못 고친다**(리플레이가 샘플된 씬에 조건화되므로 경험 분포의 long tail을 그대로 증폭).
- Stage I: 경험적 빈도 비례 샘플링 P_1(c) = N_c / Σ_{c'} N_{c'} → 원 분포 하에서 기본 능력 확장.
- Stage II: 비례 샘플링을 **씬별 균등 쿼터**로 대체 → 희귀 씬 노출 증가, 특정 씬의 지배 방지.
- 커리큘럼이 작동하는 축은 궤적 난이도가 아니라 **씬 빈도**라는 점이 특징. (희귀 씬 분할과 Stage II 세부 전략은 Appendix A.1.)

### 2.6 Reference-Policy KL Regularization
- 단계별로 **frozen reference policy**를 둔다: Stage I은 초기 VLA 정책, Stage II는 Stage I 최종 정책.
- L_KL^(s) = E_t[ (1/M_t) Σ_k m_{t,k} D_KL(p_{θ,t,k} ‖ p_{ref,t,k}^(s)) ] — 유효 액션 토큰 분포에 대한 token-level KL.
- 최종 목적: L(θ,φ) = -J_PPO^token(θ) + c_v L_V(φ) + β L_KL^(s).
- 의도: 단계 간 과도한 drift 억제 + 적응/보존 트레이드오프 조절.

---

## 3. 실험 설정

- **데이터셋**: TravelUAV, AerialVLA가 채택한 **UAV-Need-Help** 태스크. 학습 7,922 궤적, 평가 Seen 1,418 / Unseen Map 958 / Unseen Object 629.
- **난이도 분할**: 250 m 미만 = easy, 나머지 = hard. Full/Easy/Hard 모두 보고.
- **지표**: NE(최종 유클리드 오차, m, 낮을수록 좋음), SR(목표 영역 내 성공 종료 — 올바른 LAND 출력 또는 목표 영역 내에서 10스텝 연속 near-zero 이동), OSR(궤적이 목표 영역에 진입한 적 있는지), SPL(경로 효율 가중 성공).
- **구현**: AerialVLA(OpenVLA-7B + LoRA)에서 초기화 + value head 추가. RL은 **LoRA 어댑터를 최적화**. RLinf 위에 AirSim + TravelUAV 환경 + AerialVLA 액션 토큰 파이프라인 통합. 8×A100(80GB), 약 21시간, 총 롤아웃 예산 ≈ 학습셋 크기의 30%.
- **베이스라인**: Random/Fixed Action, CMA, TravelUAV / TravelUAV-DA, NavFoM, LongFly, AerialVLA. 모두 동일 스플릿의 원 논문 수치 인용.

---

## 4. 주요 실험 결과

### 4.1 Test Seen (Table 1, 3 seed 평균)

| Method | Full NE↓ | Full SR↑ | Full OSR↑ | Full SPL↑ | Hard SR↑ |
|---|---|---|---|---|---|
| CMA | 135.73 | 8.37 | 18.72 | 7.90 | 4.57 |
| NavFoM | 93.05 | 29.17 | 49.24 | 25.03 | 23.58 |
| LongFly | 60.02 | 36.39 | **65.87** | 31.07 | 33.94 |
| AerialVLA | 65.88 | 47.96 | 57.69 | 38.54 | 46.30 |
| **RecoverFly** | **54.96** | **56.33** | 65.40 | **45.98** | **56.38** |
| Human (참고) | 14.15 | 94.51 | 94.51 | 77.84 | 93.37 |

- SR +8.37pp, Hard에서는 **+10.08pp**. 네 지표 모두 Easy보다 Hard에서 개선폭이 크다 → 장기 지평 오차 누적 상황에서 RL post-training이 가장 유효.
- SR과 SPL 상승폭이 거의 일치 → 추가 성공이 **경로 효율을 희생하지 않고** 얻어짐.
- 표준편차: full NE ±1.19, SR ±0.15, OSR ±0.80, SPL ±1.13 (SR 분산이 매우 작다).

### 4.2 Test Unseen Map (Table 2)

| Method | Full NE↓ | Full SR↑ | Full OSR↑ | Full SPL↑ |
|---|---|---|---|---|
| LongFly | 108.32 | 11.27 | 30.27 | 9.32 |
| AerialVLA | 67.42 | 37.58 | 52.92 | 28.22 |
| **RecoverFly** | **58.88** | **42.97** | **60.09** | **31.83** |

- SR +5.39pp. Easy NE는 거의 불변(44.99 → 44.02)인 반면 **Hard NE는 99.11 → 79.88로 19.23 m 감소**.
- OSR과 SR 동시 상승 → 목표 영역 도달성과 성공 종료가 함께 개선.

### 4.3 Test Unseen Object (Table 3)

| Method | Full NE↓ | Full SR↑ | Full OSR↑ | Full SPL↑ |
|---|---|---|---|---|
| LongFly | 66.74 | 43.87 | 64.56 | 38.39 |
| AerialVLA | 61.45 | 56.60 | 64.86 | 46.61 |
| **RecoverFly** | **53.42** | **59.72** | **68.31** | **51.34** |

- Full 네 지표 전부 최고. Hard에서는 LongFly가 NE 57.07 / OSR 74.16으로 앞서지만 RecoverFly가 SR 57.70 / SPL 51.52로 우위 → **근처 도달이 아니라 성공 종료로의 전환율**이 핵심 차이.
- RL post-training은 추가 객체 어노테이션이나 외부 detector를 도입하지 않으므로, 이득은 AerialVLA에서 물려받은 open-vocabulary 표현 → 접근/착륙 행동 사상의 강화로 해석된다.

---

## 5. Ablation 분석

### 5.1 컴포넌트 증분 (Table 4, seed 1, full-split SR %)

| ID | Replay | KL | Curriculum | Seen | Unseen Map | Unseen Object | Avg. |
|---|---|---|---|---|---|---|---|
| – | ✗ | ✗ | ✗ | 47.96 | 37.58 | 56.60 | 47.38 |
| 1 | ✗ | ✗ | ✗ | 48.17 | 42.90 | 56.44 | 49.17 |
| 2 | ✓ | ✗ | ✗ | 56.21 | 31.21 | 59.78 | 49.07 |
| 3 | ✓ | ✓ | ✗ | 55.01 | 37.47 | 62.32 | 51.60 |
| 4 | ✓ | ✓ | ✓ | 56.28 | 44.89 | 60.41 | **53.86** |

- **ID 1 (token-level PPO만)**: Seen 거의 불변(+0.21), Unseen Map +5.32, Unseen Object -0.16 → 평균 +1.79. 스플릿별 효과가 고르지 않은 완만한 이득.
- **ID 2 (실패 리플레이 추가)**: Seen +8.04, Unseen Object +3.34인데 **Unseen Map -11.69** → 평균은 오히려 -0.10. 리플레이 단독은 성능을 **재분배**할 뿐 균일 개선이 아니다. 리플레이가 샘플된 씬에 조건화되므로 경험적 분포 안에서 실패 중심 업데이트를 증폭할 뿐 long tail을 교정하지 못한다는 저자 해석.
- **ID 3 (KL 추가)**: Unseen Map +6.26, 평균 +2.53, 대신 Seen -1.2. 초기 정책에서의 과도한 이탈을 억제해 전이 가능한 행동을 보존한다는 의도와 일치.
- **ID 4 (커리큘럼 추가, 최종)**: Unseen Map +7.42, 평균 +2.26, Unseen Object만 -1.91.

### 5.2 씬 샘플링 전략 (Table 5)

| Strategy | Seen | Unseen Map | Unseen Object | Avg. |
|---|---|---|---|---|
| Uniform Sampling (1 stage) | 50.71 | 41.23 | 52.31 | 48.08 |
| Original Distribution (2 stage) | 56.13 | 40.92 | 59.14 | 52.06 |
| **Two-Stage Curriculum** | **56.28** | **44.89** | **60.41** | **53.86** |

- 중요한 통제: **Original Distribution은 RecoverFly와 학습 단계 수가 동일**하다 → 개선이 단순히 최적화 스텝 추가 때문이 아님을 보인다.
- 처음부터 균등 샘플링(Uniform)은 오히려 최악 → Stage I의 원 분포 학습이 선행되어야 함을 시사.

---

## 6. 강점

1. **문제 진단이 정확하고 해법이 그 진단에 1:1 대응**한다. 4가지 실패 모드 각각에 컴포넌트 하나씩(토큰 PPO ↔ credit assignment, 리플레이 ↔ 실패 희석, 커리큘럼 ↔ 씬 불균형, KL ↔ 능력 퇴화).
2. **"궤적이 아니라 태스크 초기화를 저장"** 이라는 리플레이 설계가 우아하다. off-policy stale gradient 문제를 피하면서 실패 재학습 효과를 얻는다.
3. **Token-level vs sequence-level PPO 구분**이 VLA RL에서 실질적으로 중요한 지점을 짚었다(공유 clipping 결정의 해악).
4. **정직한 ablation**: 리플레이 단독이 평균 SR을 -0.10 낮춘다는 부정적 결과를 그대로 보고하고 그 이유를 설명한다. 이런 보고는 드물다.
5. **비용 효율**: 롤아웃 예산이 학습셋 크기의 30%, 8×A100 21시간으로 BC 대비 +3.12~+8.37pp. RL post-training치고 매우 가볍다.
6. Hard 스플릿과 Unseen Map Hard NE(-19.23 m)에서 개선이 집중된다는 패턴은 "장기 지평 오차 교정"이라는 주장의 **기전적 증거**로 기능한다.

---

## 7. 약점 및 한계

1. **시뮬레이션 전용**. AirSim/TravelUAV 내 폐루프만 검증되었고 실기체 배치 결과가 없다. UAV 도메인의 sim-to-real 격차(공기역학, 지연, 센서 노이즈)는 매니퓰레이션보다 오히려 크다.
2. **단일 벤치마크**. TravelUAV UAV-Need-Help 하나. AerialVLN, OpenFly, UAV-ON 등 다른 항공 VLN 벤치마크로의 이전은 미검증.
3. **단일 초기 정책**. AerialVLA에서만 초기화했으므로 "프레임워크의 일반성"(다른 UAV-VLA 백본에도 적용 가능한가)이 실증되지 않았다.
4. **하이퍼파라미터 민감도 미공개**. replay ratio η, N_max, β, c_v, κ_p, 이벤트 보상 R_e 값들이 본문에 없고 Appendix로 밀렸다. 특히 η와 β는 Table 4가 보여주듯 스플릿 간 트레이드오프를 좌우하는 축인데 sweep이 없다.
5. **Ablation은 seed 1 단일**. 본 결과는 3 seed 평균인데 ablation은 단일 seed라 ±1~2pp 규모의 결론(예: ID 3의 Seen -1.2)은 노이즈와 구분하기 어렵다.
6. **Unseen Map의 취약성**이 완전히 해소되지 않았다. 최종 SR 42.97로 Seen(56.33)·Unseen Object(59.72) 대비 현저히 낮고, ID 2에서 -11.69pp까지 흔들렸다. 맵 일반화는 여전히 구조적 병목.
7. **인간 대비 격차**가 크다. Seen에서 인간 SR 94.51 vs RecoverFly 56.33.
8. **오픈소스 미공개**(코드/체크포인트 링크 없음). RLinf + AirSim + TravelUAV 통합은 재현 비용이 높은 스택이다.

---

## 8. 기존 연구와의 위치

- **BC 기반 UAV-VLA (AerialVLA, RT-2/OpenVLA 계열)**: 강력한 expert prior를 학습하지만 폐루프 자기 오차에 대한 교정 신호가 없다. RecoverFly는 이 체크포인트를 그대로 받아 **post-training 레이어**로 얹는다.
- **모듈형/계층형 UAV-VLN (TravelUAV, CityNavAgent, SkyVLN, NavFoM, LongFly)**: waypoint 예측기·외부 detector·계층 플래너로 장기 지평을 다루지만 추론/제어 인터페이스 사이에 오차가 누적된다.
- **UAV-VLN에서의 RL (SuReAL, HTNav, OpenVLN, FlightGPT)**: OpenVLN은 value 기반 waypoint 보상 + KL 정규화, FlightGPT는 SFT + GRPO 스타일. RecoverFly의 주장은 이들이 **autoregressive end-to-end VLA의 온라인 post-training**을 다루지 않으며, 희소 폐루프 피드백·long-tail 적응·정책 drift를 **동시에** 다루지 않는다는 것.
- **리플레이 계보**: HER(사후 relabeling), Prioritized Level Replay(학습 가능성 기반 레벨 재생)의 UAV-VLA판 변주이되, 저장 단위를 궤적이 아닌 태스크 초기화로 바꿔 on-policy를 유지한 점이 차별점.

---

## 9. 재현성 체크리스트

| 항목 | 상태 |
|---|---|
| 코드 공개 | ✗ (본문에 링크 없음) |
| 체크포인트 | ✗ |
| 초기 정책 | AerialVLA (arXiv:2603.14363) — 외부 의존 |
| RL 프레임워크 | RLinf (arXiv:2509.15965) — 공개 |
| 시뮬레이터 | AirSim + TravelUAV 환경 — 공개 |
| 하이퍼파라미터 | 부분 (Appendix A.2) |
| Seed | 본 결과 3 seed 평균, ablation 단일 seed |
| 컴퓨트 | 명시 (8×A100 80GB, ~21h) |

---

## 10. 세미나 토론 포인트

1. **Token-level vs sequence-level clipping**: 같은 액션의 토큰들에 공유 advantage를 주면서 clipping만 분리하는 것이 이론적으로 어떤 편향/분산 특성을 갖는가? (Appendix C에 비교가 있다고 언급되나 본문에는 수치 없음.)
2. **리플레이가 Unseen Map을 -11.69pp 망가뜨린 기전**이 정말 "씬 조건부 증폭"인가, 아니면 실패 케이스에 대한 과적합(memorization)인가? 두 가설을 구분할 실험은?
3. **태스크 초기화 저장 vs 궤적 저장**의 trade-off: 매 replay마다 롤아웃을 재생성하는 비용(시뮬레이터 스텝)이 off-policy 재사용 대비 정당한가? 30% 예산 중 리플레이가 차지하는 비율은?
4. **Stage II 균등 쿼터**가 Uniform Sampling(단일 단계 균등)보다 좋은 이유가 커리큘럼 효과인가, 단순히 Stage I의 warm start 효과인가?
5. **이벤트 보상 설계**: collision/stuck/away/early-stop/timeout에 서로 다른 R_e를 주는데, 보상 셰이핑이 특정 실패 회피 행동(예: 과도한 보수 비행)을 유도하지 않는가? SPL이 함께 오른 것이 반증이 되는가?
6. **KL 앵커의 단계 전환**: Stage II에서 앵커를 Stage I 최종 정책으로 옮기는 것은 drift 상한을 사실상 완화하는 것인데, 누적 drift는 어떻게 통제되는가?

---

## 11. 향후 연구 방향

1. **실기체 검증**: 실제 쿼드콥터에서의 폐루프 배치와 sim-to-real 격차 정량화(SuReAL이 물리 쿼드콥터 평가를 했던 선례가 있다).
2. **다중 벤치마크 이전**: OpenFly, AerialVLN, UAV-ON으로의 확장 및 cross-benchmark 일반화.
3. **백본 무관성 검증**: AerialVLA 외 다른 UAV-VLA(또는 매니퓰레이션 VLA)에 동일 프레임워크 적용.
4. **적응형 리플레이 비율**: η와 N_max를 학습 진행/실패 유형별로 동적 조절(예: 실패 유형별 우선순위 큐).
5. **실패 유형 인지 보상/커리큘럼**: 현재 f_i(실패 유형)는 풀에 저장만 되고 샘플링 우선순위에 쓰이지 않는 것으로 보인다. 유형별 가중 리플레이가 자연스러운 확장.
6. **드롭된 케이스의 활용**: N_max 후 dropped 되는 항목들이 정말 학습 불가능한지, 아니면 커리큘럼 후반에 재활성화할 가치가 있는지.
7. **GRPO/value-free 대안과의 비교**: FlightGPT류 GRPO 스타일과 value head PPO의 비용/성능 비교.

---

## 12. 종합 평가

RecoverFly는 "BC로 학습된 end-to-end UAV-VLA를 어떻게 저비용으로 폐루프 교정할 것인가"라는 실무적으로 중요한 질문에, **네 개의 결합된 실패 모드를 각각 겨냥한 네 개의 컴포넌트**로 답하는 잘 설계된 시스템 논문이다. 학습셋 크기의 30% 롤아웃 예산으로 SR을 3.12~8.37pp 끌어올리고, 특히 Hard 궤적과 Unseen Map Hard NE에서 개선이 집중된다는 점은 주장한 기전과 결과가 일관됨을 보여준다.

가장 인상적인 부분은 방법이 아니라 **정직한 ablation**이다. 실패 리플레이 단독이 평균 SR을 오히려 낮춘다(-0.10pp)는 결과를 숨기지 않고, 그것이 KL과 커리큘럼을 필요로 하는 이유임을 논증한 구조는 "컴포넌트 나열형" 논문들과 확실히 구분된다. Table 5에서 학습 단계 수를 통제한 Original Distribution 비교를 넣은 것도 방법론적 성실함의 증거다.

한계는 범위에 집중된다: 시뮬레이션 전용, 단일 벤치마크, 단일 초기 정책, 코드 미공개, ablation 단일 seed. 특히 Unseen Map SR이 42.97에 머물고 컴포넌트에 따라 ±10pp 넘게 흔들린다는 점은 맵 일반화가 이 프레임워크로 해결되지 않은 잔여 문제임을 보여준다. 또한 기여의 상당 부분이 RLinf·AerialVLA·AirSim이라는 기존 스택의 통합/조정에 해당하므로 알고리즘적 신규성은 중간 수준이다.

그럼에도 "VLA에 RL post-training을 붙일 때 무엇이 실제로 필요한가"에 대한 경험적 지침 — 토큰 단위 clipping, 궤적이 아닌 초기화의 리플레이, 원 분포 선행 후 균등화, 단계별 KL 앵커 — 는 항공 도메인을 넘어 일반 VLA post-training 연구에 이전 가능한 자산이다.

**평가 점수: 7.5/10** — 문제 진단·시스템 설계·ablation 정직성은 우수하나, 시뮬레이션 단일 벤치마크·단일 백본·미공개 코드가 영향력을 제한한다. 실기체 검증 또는 다중 벤치마크 이전이 추가되면 8.5+ 상향 가능.

<!-- VERIFIED: pdf -->
