# TR-VLA: Reducing Temporal Redundancy for Efficient Vision-Language-Action Inference

## 1. 개요 (Overview)

TR-VLA는 flow 기반 Vision-Language-Action(VLA) 모델의 추론 지연(latency)을 시스템 수준에서 줄이는 가속 프레임워크다. 저자들은 기존 VLA 파이프라인의 두 가지 주요 시간적 중복(temporal redundancy)을 지적한다: (1) 연속 프레임 간 매우 유사한 시각 특징을 매번 재인코딩하는 낭비, (2) diffusion/flow 정책의 다단계 반복 샘플링. 이를 각각 perception 측의 "temporal token reuse"와 policy 측의 "2-step flow 압축"으로 해결하여, 백본 구조 변경 없이 2배 이상의 end-to-end 가속(시뮬레이션 2.6x, 실로봇 1.6x)을 달성한다. 참고로 논문 자체에는 고유 약칭이 없으며, 본 트래커에서는 편의상 TR-VLA(Temporal Redundancy)로 명명한다.

## 2. 문제 정의 (Problem Statement)

현대 VLA는 대형 ViT 백본과 diffusion/flow matching 같은 반복 추론에 의존해 지연이 크며, 이는 안전하고 반응적인 조작에 필요한 고주파 폐루프 제어와 충돌한다. 기존 가속 연구는 개별 모듈(주로 perception/VLM)만 최적화해 병목을 옮길 뿐 제거하지 못하고, 실제 지배적 비용인 action generation은 대체로 다루지 않았다.

## 3. 핵심 관찰 (Key Observations)

- Perception: 인접 프레임의 시각 토큰은 여러 ViT 레이어에 걸쳐 코사인 유사도가 대부분 0.98 이상으로 극도로 높으며, dissimilarity 맵은 공간적 희소성을 보여 소수 토큰만 실제로 변한다.
- Policy: flow matching 속도(velocity) 궤적의 특이값 스펙트럼이 급격히 감소해 에너지가 상위 2개 성분에 집중된다 → 속도 진화가 저차원(low-rank) 부분공간에 놓인다. 즉 다단계 diffusion 갱신은 서로 강하게 상관되어 있다.

## 4. 방법 - Temporal Token Reuse (방법 1)

단순한 odd/even 청킹 전략을 사용한다. Odd 청크에서는 full forward를 수행하고 key projection을 참조 표현으로 캐시한다. Even 청크에서는 현재 key와 캐시된 key의 코사인 유사도 sim = cos(K_t, K_ref)를 계산해 유사도가 가장 낮은(변화가 큰) top-rho*N 토큰만 갱신 대상(dynamic region)으로 선택한다. 인덱스 선택은 첫 번째 Transformer 레이어에서 한 번만 수행하고 이후 레이어에서 재사용한다(레이어 간 유사도 패턴 일관성 활용). 갱신 토큰만 attention을 재계산하고 나머지는 캐시된 KV를 sparse scatter로 상속하며, 이후 MLP는 전체 토큰에 적용해 안정적 학습을 보장한다.

## 5. 방법 - Efficient Flow Matching Policy / Step Compression (방법 2)

Flow matching은 ODE dx_t/dt = v_theta(x_t, t, o)를 Euler 이산화로 t=1(가우시안 노이즈)→t=0으로 적분하며 통상 N=8~10 스텝이 필요하다. 저자들은 저차원 가정 하에 각 속도 v_t를 소수 기저 방향의 선형결합으로 표현하고, 초기 전역 방향을 나타내는 v0와 중간 궤적 정련을 나타내는 v7을 앵커로 선택한다(v_t ≈ a_t·v0 + b_t·v7). 최종 상태는 x_T ≈ x_0 + alpha·v0 + beta·v7 + b로 근사된다. 계수 (alpha, beta, b)는 관측에 의존하므로 lightweight adaptor g_phi가 v0로부터 직접 예측하도록 하고, 압축된 2-step 궤적과 원본 full-step 궤적의 L2 불일치를 최소화하는 efficiency-oriented training으로 학습한다.

## 6. 실험 설정 (Experimental Setup)

시뮬레이션은 LIBERO와 RobotWin 벤치마크(224x224 RGB, 자연어 지시, 7-DoF joint-space, 성공률 지표)에서 평가한다. 실로봇은 Marvin Pro 양팔 로봇(각 7-DoF, RealSense D435 헤드 + SHW 5G 손목 카메라 2대, Meta Quest 3 원격조작 데이터)에서 Cartesian 제어로 검증한다. 시뮬레이션은 A100, 실로봇 배포는 RTX 4090, FP16/PyTorch.

## 7. 주요 결과 - LIBERO (Main Results)

Table I 기준, pi0.5 + Ours(2-step)는 Mean SR 93.8%를 기록한다: LIBERO-10(Long) 88.2, Goal 91.2, Spatial 97.8, Object 98.0. 이는 원본 pi0.5(94.4%) 및 X-VLA(94.1%)와 대등하면서 샘플링을 10→2 스텝으로 줄였다. 지연은 286.9ms→121.2ms, 제어 주파수 3.5→8.2 FPS로 개선. pi0 + Ours도 Mean SR 91.0%로 유사 경향을 보여 백본 간 일반화를 입증한다.

## 8. 주요 결과 - RoboTwin & 효율 분석

Table II(RoboTwin 2.0)에서 Ours는 TOP10 SR 81.5%로 baseline pi0(82.2%)에 근접하면서 Total 298.46ms→125.4ms(42%), 3.35→8.0 FPS. 추론 시간 분해 결과 병목은 Action Expert(212.6ms → 40.9ms, 19%로 감소)이며 ViT/LLM 지연은 거의 유지된다. 부록 Table VIII의 48개 RoboTwin 태스크 전체 평균은 Ours 46.10 vs Base 46.68로, 타 가속 기법(ToMe 36.50, ToFu 32.28, SparseVLM 36.82 등)을 크게 상회한다.

## 9. Ablation

Table III(LIBERO, pi0.5 baseline): Token Reuse 단독은 ViT 40.1→28.5ms, SR 94.4→93.9%로 거의 손실 없음. Efficient Policy 단독은 Action 212.6→41.5ms, Total 293.2→123.9ms(8.1 FPS), SR 94.1%. 둘 결합 시 111.6ms/9.0 FPS, SR 93.8%로 최저 지연 달성. 두 구성요소가 파이프라인의 서로 다른 부분을 겨냥해 상보적 이득을 제공함을 확인한다. Table VI는 스텝 수 감소(10→2)에서 2-step이 121.2ms/2.37x임을, Table VII은 토큰 갱신 비율 r=0.4에서 SR 93.8%로 정확도-효율 균형이 최적임을 보인다.

## 10. 실로봇 평가 (Real-World)

Marvin Pro 6개 태스크(Table IV)에서 2-step 정책은 전체 SR 95.4%로 원본 pi0.5(97.2%)에 근접하면서, 30초 내 성공률(SR@30s)을 77.1%→82.3%로 개선한다. 즉 정책 지연 감소가 실효 제어 주파수를 높여 상태 편차에 더 빠르게 반응하고 시간 예산 내 장기 태스크 완료를 돕는다. Rollout(Fig. 6)은 압축 정책이 진동이나 제어 정밀도 저하 없이 안정적 궤적을 유지함을 보인다.

## 11. 한계 및 향후 연구 (Limitations & Future Work)

압축 정책은 주로 단기(short-horizon) 조작 태스크에서 검증되었으며, 장기 계획이나 매우 동적인 환경에서의 효과는 추가 검증이 필요하다. 향후 adaptive step scheduling과 더 다양한 태스크로의 확장을 계획한다. 또한 v0/v7 앵커 선택은 경험적이며, 코드는 아직 공개 예정 상태(open_source=false)다.

## 12. 총평 (Assessment)

TR-VLA는 "perception + policy를 동시에" 시간적 중복 관점에서 공략한 실용적 시스템 가속 연구다. 특히 flow velocity의 low-rank 구조를 근거로 10-step을 2-step으로 압축하면서 정확도를 거의 유지한 점, 그리고 실제 병목이 VLM이 아닌 Action Expert임을 정량적으로 드러낸 분해 분석이 기여로 볼 만하다. 다만 고유 방법명 부재, 코드 미공개, 단기 태스크 위주 검증은 재현성과 일반성 측면의 약점이다. 백본 불변·추가 학습 오버헤드 최소라는 배포 친화성이 강점이다.

<!-- VERIFIED: pdf -->
