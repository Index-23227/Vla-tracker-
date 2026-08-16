# DreamFly: Causal Memory and Receding-Horizon Diffusion Planning for Aerial Vision-Language Navigation

- **arXiv**: 2608.12308v1 (2026-08-12, cs.CV)
- **저자/소속**: Yan Deng, Fei Xu — Xi'an Technological University
- **백본**: Dream-VLA (discrete diffusion LLM, arXiv:2512.22615)
- **벤치마크**: OpenFly (AirSim / Unreal Engine 폐루프)

> **한 줄 요약**: Dream-VLA 위에 (1) read-before-write 규약의 causal 히스토리 메모리, (2) plan-K/execute-one receding-horizon discrete diffusion 플래닝, (3) all-mask 로짓에서 종료를 예측하는 LiteStop을 얹어 OpenFly test-seen/unseen에서 SR 32.04%/29.46%, SPL 28.22%/23.54%를 달성한 항공 VLN 정책.

---

## 1. 배경 및 동기

Aerial VLN은 UAV가 자연어 지시를 따라 대규모 3D 도시 환경을 비행하는 부분관측 폐루프 의사결정 문제다. 지상 VLN과 달리 수평 이동·수직 이동·시점 조정을 동시에 조율해야 하고, 고도 변화가 가시 영역의 범위·랜드마크 스케일·가림 정도를 모두 바꾼다. 저자들은 기존 VLA를 항공 VLN에 옮길 때 남는 세 가지 시간적 결함을 지적한다.

1. 현재 관측만으로 조건화된 정책은 이전에 본 랜드마크를 놓친다.
2. 단일 스텝 행동 예측은 lookahead가 없다.
3. 행동 생성에 종료를 암묵적으로 섞으면 정지 판단이 불안정하다.

## 2. 문제 정의

지시 I와 초기 자세 s0 = (p0, theta0)가 주어지고, 각 스텝 t에서 egocentric RGB 관측 Ot를 받아 이산 행동 공간 A(Stop, 서로 다른 보폭의 전진, 좌/우 회전, 상승/하강, 횡이동)에서 at를 고른다. Stop을 선택하거나 최대 스텝에 도달하면 에피소드가 끝나며, 최종 위치가 목표에서 20 m 이내면 성공이다.

## 3. Causally Aligned Historical Memory (Sec. 3.3)

- 정의: M_<t = F_mem(I, (O_tau)_{tau < t}). 즉 **현재 관측 Ot는 히스토리 분기에 절대 들어가지 않는다**. 여기서 "causal"은 변수 간 인과가 아니라 시간 순서상의 정보 접근 제한을 뜻하며, 저자들은 이것이 imitation learning의 covariate shift를 완화하지는 않는다고 명시한다.
- 후보 생성: **frozen CLIPSeg** dense router + **frozen OWLv2** region router. 라우터의 텍스트 컨텍스트 한계를 우회하려 지시문을 고정된 중첩 토큰 윈도우로 분할해 전체를 커버한다. 두 모델의 feature를 섞지 않고, OWLv2 region b를 CLIPSeg 시각 feature 공간으로 사영: f(b) = Norm(sum_g area(G_g ∩ b) · v_g).
- 장기 메모리: 후보를 최근 관측들에 걸쳐 active track에 연결해 증거를 누적. 반복적·안정적 지지를 받으면 persistent promotion, 그렇지 않아도 confidence/region-validity/score-separation/novelty 기준을 만족하면 single-observation promotion으로 진입한다. write utility로 랭킹해 **스텝당 최대 2개**만 기록.
- 슬롯 표현: **16개 슬롯**, 각 슬롯은 anchor feature와 선택적 prototype feature, prototype 존재 플래그, 마지막 갱신 이후 경과 스텝의 log(1+·)로 구성. 무효 슬롯은 zero-fill 후 slot-validity mask로 cross-attention key/value에서 제외.
- 융합: 현재 이미지 토큰 Z_t를 query, 슬롯 임베딩을 key/value로 하는 masked MHA로 컨텍스트 C_t를 얻고, 게이트 잔차로 주입한다. G_t = 1 + tanh(Z_t W_G + b_G), Z~_t = Z_t + M_img ⊙ G_t ⊙ (C_t W_O). **W_O, W_G, b_G는 zero-init**이므로 초기에는 항등 사상(identity)이며 게이트 초깃값은 1이다. 즉 메모리 어댑터가 학습 초기에 백본을 교란하지 않는다.
- 학습 시에는 전문가 궤적을 따라 만든 prefix memory를, 배포 시에는 롤아웃마다 새로 초기화한 online memory를 쓰지만 causal 경계는 동일하다.

## 4. Receding-Horizon Diffusion Planning (Sec. 3.4)

- 길이 K의 이산 행동 청크 a^_t = [a^0_t, ..., a^{K-1}_t]를 예측한다. 첫 슬롯이 실제 제어 결정이고 나머지는 **보조 예측 타깃**이다. 미래 관측 O_{t+1..t+K-1}은 절대 주어지지 않는다.
- 유효 접두부 감독: L_t = min(K, T-t), v_{t,h} = I[h < L_t]. 궤적 끝에서 고정 길이 템플릿을 유지하려 뒤쪽을 Stop으로 패딩하되, 패딩된 Stop은 손실에 기여하지 않는다(원래 궤적에 들어 있던 Stop은 유효 타깃으로 유지).
- 손실: 슬롯별 가중 CE. 가중치는 v_{t,h} · c_{t,h} · gamma^h이며, gamma ≤ 1이 근시 행동을 강조한다. c_{t,h}는 기하 커널(p_car = 0.1)로 계산한 CAR context 계수. CE는 Dream-VLA를 따라 **전체 vocabulary**에서 평가하고, 전용 action-token 집합 chi(A)으로의 제한은 생성 시에만 적용한다.
- 학습 시 K개 행동 위치를 모두 [MASK]로 두고 **단일 양방향 forward**로 동시에 예측한다. 즉 학습은 반복 denoising을 펼치지 않는다.
- 추론: monotonic origin sampler(Dream-VLA 계승), 스텝별 전이 확률 omega_s = 1 - t_{s+1}/t_s, 마지막 스텝에서 전량 해소. 해소된 슬롯은 remask하지 않고, 이미 확정된 토큰이 남은 슬롯의 컨텍스트가 된다. 샘플링 전 chi(A) 밖 로짓을 억제해 실행 가능한 행동만 나오게 한다.
- 실행: 청크 생성이 끝난 뒤 LiteStop을 평가하고, 종료되지 않으면 **첫 행동만** 실행한다. 나머지 예측은 캐시하지 않고 버린다.

## 5. LiteStop (Sec. 3.5)

- 초기 all-mask denoising forward에서 얻은 action-token 로짓 격자 H^(0)_t ∈ R^{K x |A|}를 vec → LayerNorm → MLP(W1, SiLU, W2)로 스칼라 stop 로짓에 매핑한다. **첫 위치의 Stop 로짓만 쓰지 않고 K x |A| 격자 전체**를 쓰는 것이 요점.
- 라벨은 y^stop_t = I[a*_t = a_stop]. 미래 청크 위치나 패딩의 Stop은 현재 라벨에 영향을 주지 않는다. 기하학적 성공 여부나 terminal metadata를 쓰지 않으므로, 독립적인 goal-reached 분류기가 아니라 **frozen 정책의 action-level Stop 성향을 보정**하는 역할이다.
- 손실은 positive weight w+ = 4.0의 가중 BCE. 학습 중 정책(시각·메모리·행동 플래닝 전부)은 완전히 동결된다.
- 최종 종료 결정은 d^term_t = d^stop_t OR I[a^0_t = a_stop]. 즉 LiteStop은 정책의 action-level Stop을 거부(veto)하지 않고 **추가 경로**만 제공한다. 또한 H^(0)_t를 캐시할 뿐 diffusion을 조기 종료하지 않으므로 early-exit 기법이 아니다.

## 6. 데이터 및 실험 설정

OpenFly 릴리스 데이터에 네 가지 표준화를 적용: (1) Forward 6m의 8-D action vector를 정본 인코딩으로 수정, (2) 약 **190,000 스텝**의 비표준 라벨을 재매핑(-1 → Go Up, -2 → Go Down), (3) **사전 패키징된 히스토리 keyframe 제거** 후 현재 RGB만 유지, (4) [dx, dy, dz, yaw]를 provenance 메타데이터로만 저장(정책 입력으로 사용하지 않음).

- 학습셋: 20개 subset, **85,785 궤적 / 1,356,622 decision step**. 학습 행동 분포는 전진 쪽으로 뚜렷하게 치우쳐 있다(Fig. 3a).
- 평가: 8개 AirSim/UE 환경, 총 **1,796 궤적**. test-seen 1,392개(UE BigCity + AirSim 도시 6종), test-unseen 404개(UE SmallCity).
- 지표: NE, SR, OSR, SPL. 성공 반경 20 m.
- 하이퍼파라미터: all-linear LoRA(r=32, alpha=16), 메모리 융합 어댑터 공동 학습, base projector 동결. **K = 4**, gamma = 0.7, p_car = 0.1. AdamW lr 1e-4, batch 8, 최대 10,000 step이나 **step 5,000 체크포인트**를 LiteStop 학습과 폐루프 평가에 사용. 추론 diffusion step **12**. 메모리 슬롯 16개 · 512차원, gated cross-attention 512-dim / 8-head.

## 7. LiteStop 임계값 파일럿 (Table 1)

64개 궤적(환경당 8개, 최종 평가 split과 분리) 위에서 step-500 LiteStop 체크포인트로 측정.

| tau_stop | SR ↑ | SPL ↑ | OSR-SR ↓ | NE ↓ | N_LS |
|---|---|---|---|---|---|
| **0.50 †** | 0.266 | 0.229 | 0.188 | 53.07 | 45/64 |
| 0.65 | 0.250 | 0.202 | 0.250 | 51.42 | 19/64 |
| 0.80 | 0.266 | 0.230 | 0.219 | 53.20 | 0/64 |

0.50을 고른 이유는 OSR-SR 격차가 가장 작고 NE가 0.80보다 약간 낮으면서 SR은 동일하기 때문. 0.80에서는 LiteStop 개입이 **0회**라 사실상 모듈이 죽어 있다.

## 8. 주요 결과 (Table 2)

| Method | seen NE↓ | seen SR↑ | seen OSR↑ | seen SPL↑ | unseen NE↓ | unseen SR↑ | unseen OSR↑ | unseen SPL↑ |
|---|---|---|---|---|---|---|---|---|
| Random | 65.67m | 13.51% | 18.75% | 9.72% | 59.99m | 15.35% | 23.76% | 11.31% |
| Action Sampling | 62.78m | 15.95% | 26.51% | 13.67% | 55.27m | 20.54% | 32.67% | 17.22% |
| Seq2Seq | 54.44m | 24.35% | 61.93% | 19.35% | 47.69m | 26.49% | 61.88% | 19.62% |
| CMA | 313.03m | 7.97% | 69.32% | 6.26% | 230.05m | 5.69% | 73.02% | 3.92% |
| AerialVLN | 176.29m | 16.52% | 65.66% | 14.63% | 161.19m | 9.65% | 68.07% | 7.93% |
| OpenFly-Agent | 122.89m | 22.63% | 52.73% | 20.42% | 163.87m | 14.11% | 62.38% | 12.49% |
| **DreamFly** | **44.87m** | **32.04%** | 46.77% | **28.22%** | **45.29m** | **29.46%** | 46.78% | **23.54%** |

DreamFly는 NE·SR·SPL 세 지표에서 모두 1위지만 **OSR은 오히려 최하위권**(46.77 / 46.78 vs CMA 69.32 / 73.02)이다. CMA·AerialVLN처럼 오래 배회하는 정책은 목표 근방을 한 번쯤 스치므로 OSR이 높지만 NE는 파국적(313m, 176m)이다. DreamFly는 반대로 짧고 결단력 있게 멈춘다. 저자들은 OSR-SR 격차를 종료 품질의 지표로 활용한다.

OpenFly-Agent는 공식 릴리스 체크포인트를 그대로 평가했고, 나머지 학습형 베이스라인은 저자들의 표준화 데이터로 동일 스텝 수만큼 재학습했다.

또한 저자들은 확률적 베이스라인의 non-zero SR을 **초기 조건 편향**으로 해부한다(Fig. 4). 궤적을 초기 목표 거리가 20 m 성공 반경 안/밖인지로 나누면 Random·Action Sampling의 성공률이 안쪽 그룹에서 극단적으로 높다. 즉 이들의 13~20% SR은 항법 능력이 아니라 유리한 초기 배치의 산물이다.

## 9. Ablation (Table 3)

| Experiment | NE↓ | SR↑ | OSR↑ | SPL↑ |
|---|---|---|---|---|
| Dream-VLA (baseline) | 67.82m | 21.55% | 42.32% | 16.09% |
| + Causal Memory | 48.93m | 24.11% | 48.22% | 19.85% |
| w/o Memory | 62.02m | 19.60% | 23.89% | 18.76% |
| w/o Chunk | 46.72m | 27.73% | 38.42% | 23.77% |
| w/o LiteStop | 50.69m | 26.61% | 55.18% | 22.29% |
| **DreamFly (Ours)** | **44.97m** | **31.46%** | 46.77% | **27.17%** |

- Causal Memory 단독 추가만으로 NE가 67.82 → 48.93 m로 크게 줄고 SR +2.56%p. 메모리 제거 시 OSR이 42.32 → 23.89로 붕괴하는 것이 인상적이다(목표 근방에 아예 도달하지 못함).
- w/o LiteStop은 OSR 55.18로 가장 높으면서 SR은 26.61에 그친다 — 정확히 "도달하지만 멈추지 못한다"는 실패 모드이며 LiteStop의 존재 이유를 뒷받침한다.
- **주의**: 이 표의 full 모델 행(SR 31.46 / NE 44.97 / SPL 27.17)은 Table 2의 헤드라인 수치(32.04 / 44.87 / 28.22)와 미세하게 다르다. 논문은 이 불일치를 설명하지 않는다. OSR만 46.77로 동일하다.
- 거리 구간별 분석(Fig. 5): LiteStop 이득은 최단 거리 그룹에서 최대, 메모리는 중거리에서, 장거리에서는 메모리와 청크 플래닝이 상보적으로 기여.

## 10. 정성 분석 (Fig. 6)

- 예시 1: DreamFly는 목표 오차를 78.9 m → 10.9 m로 줄이지만 메모리 제거 변형은 43.5 m에서 멈춘다.
- 예시 2: DreamFly 2.2 m 성공, 청크 예측 제거 시 초기에 이탈해 58.1 m 실패.
- 예시 3: LiteStop 없는 변형은 성공 영역에 진입한 뒤에도 계속 이동해 23.0 m로 이탈, DreamFly는 12.5 m에서 정상 종료.

## 11. 한계 및 비판적 검토

- **시뮬레이션 전용**. AirSim/UE 밖의 실기체 검증이 전무하며, 저자들도 sim-to-real을 향후 과제로 남긴다.
- **절대 성능이 낮다**. 최고 SR이 32%로, 3분의 2 이상의 에피소드가 여전히 실패한다.
- **OSR 하락**은 명시적 trade-off다. 종료를 공격적으로 만들면 SR/SPL은 오르지만 탐색 범위는 줄어든다. tau_stop = 0.50에서 파일럿 64 에피소드 중 45개를 LiteStop이 끊었다는 사실은 이 모듈이 상당히 공격적임을 시사한다.
- **임계값 선택 근거가 얇다**. 64 궤적, step-500 체크포인트, 3개 값 그리드에서 SR이 0.266/0.250/0.266으로 사실상 구분되지 않는데 SPL 기준으로는 0.80이 근소 우위(0.230)다. 선택 논리가 OSR-SR 격차에 크게 의존한다.
- **학습 예산이 작다**. LoRA r=32, batch 8, 5,000 step 체크포인트. 10,000 step까지 돌리고도 5,000을 쓴 이유(과적합? 폐루프 성능 역전?)가 설명되지 않는다.
- **코드·체크포인트 미공개**이며 파라미터 수도 명시되지 않아 재현이 어렵다. 메모리 승격 규칙(confidence/region-validity/score-separation/novelty)의 구체적 임계값도 본문에 없다.
- **베이스라인 구성이 다소 약하다**. Random·Action Sampling 두 자리를 확률적 베이스라인이 채우고, 최신 항공 VLA(WorldVLN, ImagineUAV, AerialVLA, FSD-VLN)는 related work에 인용만 되고 비교표에는 없다.
- LiteStop이 정책을 동결한 채 학습된다는 점은 안전하지만, 동시에 **행동 정책이 종료 신호로부터 학습할 기회를 차단**한다. 공동 미세조정과의 비교가 없다.

## 12. 총평 및 VLA-Tracker에서의 위치

DreamFly는 새 백본을 제안하는 논문이 아니라 **discrete diffusion VLA(Dream-VLA)를 항공 VLN에 적응시키는 방법론 논문**이다. 세 기여 모두 "시간"이라는 하나의 축으로 정렬돼 있다는 점이 깔끔하다: 과거(read-before-write causal memory), 미래(plan-K/execute-one), 그리고 종료 시점(LiteStop). 특히 zero-init gated residual로 메모리 어댑터를 항등 사상에서 출발시키는 설계와, action chunk를 실행 커밋이 아니라 보조 감독 신호로만 쓰는 해석은 다른 VLA에도 옮겨 적을 만하다.

Tracker 관점에서는 조작(manipulation) 중심의 LIBERO/CALVIN 계열이 아닌 **항공 VLN(OpenFly) 트랙의 discrete-diffusion 정책** 항목으로 분류된다. 자체 LoRA 미세조정 체크포인트와 별도 학습된 LiteStop 헤드를 산출하므로 frozen-policy 래퍼가 아니라 학습된 정책이다. 다만 벤치마크가 tracker의 표준 7종과 겹치지 않아 리더보드 평균에는 기여하지 않으며, 헤드라인 평균 지표가 논문에 없으므로 `_avg` 키도 두지 않았다.

---

<!-- VERIFIED: pdf -->
