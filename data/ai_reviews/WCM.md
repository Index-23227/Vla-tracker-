# WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning

- arXiv: 2607.29613 (2026-07-31)
- 기관: Tongji University / Shanghai Innovation Institute / Fudan University (OpenMOSS)
- 코드: https://github.com/sylvestf/WCM

## 1. 한 줄 요약

VLA 강화학습에서 critic이 단일 프레임만 보고 가치를 추정하는 구조적 한계를 POMDP 관점의 "상태 근사 문제"로 규정하고, 미래 latent 예측(world modeling)과 가치 추정을 하나의 end-to-end 아키텍처로 통합한 World Critic Model(WCM)을 제안한다.

## 2. 문제 정의

로봇 조작은 본질적으로 부분관측 MDP(POMDP)다. 한 장의 프레임은 물체 외형과 장면 배치는 보여주지만 운동, 접촉 진행 상태, 미래 전개 같은 동역학 정보를 담지 못한다. 그럼에도 기존 VLA-RL critic은 단일 프레임 관측 또는 단일 프레임 VLM latent에서 값을 회귀한다. 저자들은 단순히 히스토리를 입력으로 넣는 것만으로는 부족하다고 지적한다. 프레임 스태킹은 큰 관측 공간에서 비효율적이고, ViT + positional encoding 기반 temporal critic조차 성능이 개선되지 않는다. 원인은 스칼라 return 회귀가 시간축 동역학을 학습시키기에 지도 신호가 너무 약하기 때문이며, critic이 히스토리를 그저 더 큰 정적 특징 벡터로 취급해 버린다는 것이다.

## 3. 핵심 아이디어

"상상하고 평가한다(imagine-plus-evaluate)". 좋은 상태 표현이라면 자기 자신의 미래를 예측할 수 있어야 한다는 predictive state 관점을 차용해, world modeling을 분리된 보조 태스크가 아니라 critic 표현 학습의 주 목적함수 중 하나로 넣는다. 가치 회귀와 미래 latent 예측을 공동 최적화하면 critic 표현이 태스크 관련 미래 결과의 압축된 요약(predictive state)에 가까워진다는 주장이다.

## 4. 아키텍처

경량 LeJEPA를 기반으로 하며 네 구성요소로 이루어진다.

1. Observation encoder: 과거 K 프레임 각각을 독립적으로 latent z_{t-k}로 인코딩. 구현에 따라 ViT 또는 기반 VLA 정책의 VLM 백본을 그대로 사용한다.
2. Language conditioning: 지시문을 CLIP으로 인코딩하고 학습된 adapter로 WCM latent 공간에 매핑. 시각 히스토리가 이 instruction token에 cross-attention한다.
3. World predictor: causal Transformer trunk가 언어 조건화된 시퀀스를 처리해 hidden state h_t를 만든다.
4. 두 개의 head — value decoder는 스칼라 V̂_t를 출력하고, action-conditioned latent dynamics branch는 action encoder와 gated FiLM residual block으로 ẑ_{t+1} = D_world(h_t, a_t, z_t)를 residual 방식으로 예측한다.

## 5. 학습 목적함수

L = L_value + λ · L_pred + η · L_SIGReg.

- L_pred: teacher forcing으로 예측 latent와 실제 다음 latent의 L2 거리.
- L_SIGReg: Sketched-Isotropic Gaussian Regularization. latent를 임의 단위벡터에 사영한 1차원 분포의 경험적 특성함수가 표준정규 특성함수와 일치하도록 강제해 차원 붕괴와 mode degeneration을 막는다.
- L_value: 예측 가치와 return의 L2. 보상은 종료 시점 성공 0, 실패 -C_fail, 그 외 매 스텝 -1의 형태이고 return은 [-1, 1]로 min-max 정규화된다.

전체가 end-to-end로 학습된다.

## 6. 학습 파이프라인

- On-policy: AR 계열(OpenVLA-OFT)은 PPO, flow matching 계열(π0, π0.5)은 PPO 변형인 Flow-SDE를 쓰고, 두 경우 모두 critic 자리에 WCM을 넣어 GAE advantage를 계산한다. 이때 encoder는 정책의 VLM 백본을 재사용한다.
- Off-policy: 텔레오퍼레이션 SFT 데이터에 실패 롤아웃과 전형적 실패 사례를 합친 통합 버퍼로 WCM과 정책을 함께 갱신한다. AR은 AWR, flow matching은 π*_0.6의 RECAP을 사용한다. 실패 데이터를 섞는 것이 과도하게 낙관적인 가치 추정을 막는 역할을 한다.

## 7. 실험 설정

시뮬레이션은 네 벤치마크 149개 태스크다. RL4VLA 설정을 따른 ManiSkill(IND 및 vision/semantic/execution 세 축의 OOD), pick-and-place를 넘어서는 태스크를 위한 MetaWorld, 장기 과제 능력을 보는 CALVIN, 일반화를 보는 LIBERO-Plus. 모든 정책은 few-shot SFT에서 출발해 sparse 0/1 보상으로 학습한다. 실제 로봇은 WidowX-250S에서 7개 태스크(동적 파지, 장기 과제, 변형체 조작 2종, pick-and-place 3종)이며 태스크당 100 궤적 SFT 후 8회 RL 이터레이션, 이터레이션당 태스크별 50 롤아웃이다. 베이스라인은 시뮬레이션에서 Flow-Noise, Flow-SDE, π-stepNFT(π 계열)와 RLinf 구현의 PPO, GRPO(OpenVLA-OFT), 실기에서 AWR과 RECAP(Gemma 270M critic)이다.

## 8. 주요 결과

ManiSkill(Table 1, IND/OOD 평균):

| 백본 | 방법 | IND | OOD |
|---|---|---|---|
| π0 | SFT | 38.4 | 18.1 |
| π0 | + Flow-SDE | 78.8 | 39.3 |
| π0 | + π-stepNFT | 79.2 | 50.4 |
| π0 | + WCM | 84.4 | 51.5 |
| π0.5 | SFT | 47.0 | 26.4 |
| π0.5 | + Flow-SDE | 90.9 | 49.3 |
| π0.5 | + π-stepNFT | 85.4 | 59.5 |
| π0.5 | + WCM | 91.9 | 64.4 |
| OpenVLA-OFT | SFT | 28.1 | 18.3 |
| OpenVLA-OFT | + GRPO | 94.1 | 60.6 |
| OpenVLA-OFT | + PPO | 97.7 | 77.1 |
| OpenVLA-OFT | + WCM | 99.0 | 77.9 |

특히 ManiSkill 데이터를 전혀 보지 않은 zero-shot 초기값(0.8%)에서 시작해도 WCM으로 98.7%까지 올라간다(약 +97.9, 12,551% 상승).

MetaWorld 성공률(Figure 4): π0 기준 SFT 50.8 → Flow-SDE 78.1 → Flow-Noise 74.8 → WCM 83.4. π0.5 기준 43.8 → 70.7 → 66.1 → WCM 75.2.

CALVIN 평균 완료 길이(Figure 4): π0 기준 SFT 3.766 → 3.850 → 3.793 → WCM 3.918. π0.5 기준 3.838 → 4.717 → 4.652 → WCM 4.748.

LIBERO-Plus total(Table 2): π0는 One-SFT 39.1에서 WCM 72.8(Full-SFT 71.2 상회), π0.5는 38.0에서 73.7(Full-SFT 72.9 상회), OpenVLA-OFT는 29.3에서 74.0(Full-SFT 71.7 상회). 한 태스크당 데모 1개만 본 상태에서 약 250 RL 스텝만으로 2만 궤적 full SFT를 넘어선다는 점이 인상적이다.

실기(Table 3, 50회 중 성공 횟수). π0.5 기준 SFT → RECAP → WCM 순으로 당근 25/33/44, 바나나 31/37/38, 피망 34/40/43, 천 접기 21/32/38, 수건 접기 24/33/35, 가스레인지 청소 4/27/33, 회전 초밥 집기 13/18/24. OpenVLA-OFT 기준으로도 AWR 대비 7개 태스크 전부에서 우위다. 실기 critic은 107.2M 학습 파라미터 규모다.

## 9. 분석 및 어블레이션

World prediction 목적함수의 필요성을 세 조건으로 분리 검증한다. (a) 원래의 MLP value head에 히스토리만 늘리면 오히려 성능이 나빠질 수 있고, (b) 히스토리 기반 ViT(즉 λ=0인 WCM 특수 케이스)도 여전히 효과가 없으며, (c) world prediction 항을 넣어야 비로소 개선된다. 즉 이득의 출처는 히스토리 자체가 아니라 히스토리에 붙은 dense한 예측 지도 신호다.

히스토리 길이는 1~5를 훑었을 때 평균적으로 3이 최적이었다. 저자들의 직관적 설명은 두 프레임이 1차 동역학(속도), 세 프레임이 2차 동역학(가속도)을 암묵적으로 담고, 대상 태스크에는 그 이상이 필요 없다는 것이다. 즉 히스토리는 길수록 좋은 것이 아니다.

## 10. 강점

- 문제 진단이 명확하다. "히스토리 부족"이 아니라 "히스토리에 대한 지도 신호 부족"으로 원인을 재정의한 점이 이 논문의 핵심 기여다. λ=0 어블레이션이 이 주장을 직접 뒷받침한다.
- critic만 교체하는 최소 침습적 설계라 on-policy/off-policy, AR/flow matching, π0·π0.5·OpenVLA-OFT 전반에 그대로 꽂힌다.
- IND 개선보다 OOD 및 LIBERO-Plus 일반화 개선 폭이 크다. 가치 추정이 좋아지면 분포 이동 상황에서 더 크게 이득이라는 해석과 부합한다.
- 실기 검증이 사족이 아니다. 수백~수천 샘플, 1시간 미만 학습으로 반복 개선이 가능하다는 점은 실제 배치 관점에서 의미가 있다.

## 11. 약점과 의문점

- WCM은 정책이 아니라 critic이다. 리더보드 관점에서 보고되는 수치는 π0/π0.5/OpenVLA-OFT 정책의 성능이므로 백본과 분리해 읽으면 안 된다.
- 표준 LIBERO 4-suite 수치가 없다. LIBERO-Plus만 보고되어 기존 LIBERO 리더보드와 직접 비교가 불가능하다.
- CALVIN 결과의 split(ABC→D 등)이 본문에 명시되지 않는다. 평균 완료 길이만 제시된다.
- ManiSkill 이득 중 상당 부분은 SFT 초기값이 매우 약하다는 점(π0 38.4, OpenVLA-OFT 28.1)에 힘입는다. 12,551% 같은 상대 개선 수치는 분모가 0.78%라는 점을 감안해야 한다.
- Flow-SDE/PPO 대비 WCM의 IND 개선 폭은 OpenVLA-OFT에서 97.7 → 99.0으로 이미 포화 구간이며, 오차 막대와 겹치는 항목도 있다. 진짜 차별점은 OOD와 저데이터 영역이다.
- LeJEPA/SIGReg 선택의 필연성, λ와 η 민감도에 대한 정량 분석은 상대적으로 얇다.

## 12. 시사점

VLA-RL 연구가 지금까지 정책 쪽 알고리즘(PPO/GRPO/Flow-SDE 변형)에 집중해 온 데 비해, 이 논문은 critic 표현 학습이 남아 있는 병목임을 보인다. World model을 "정책이 미래를 상상하는 도구"가 아니라 "critic이 상태를 복원하는 지도 신호"로 쓰는 관점은 재사용 가치가 높다. 특히 실기 RL처럼 롤아웃이 비싸고 보상이 희소한 환경에서, 동일한 데이터로 더 많은 학습 신호를 뽑아내는 방향은 앞으로 표준 레시피가 될 가능성이 있다. 반대로 히스토리 길이 3에서 이득이 포화한다는 결과는 현재 조작 태스크의 동역학 복잡도가 아직 낮다는 신호이기도 하며, 더 긴 시간 지평의 태스크에서는 다른 결론이 나올 수 있다.

<!-- VERIFIED: pdf -->
