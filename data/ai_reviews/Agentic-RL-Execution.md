# Agentic-RL-Execution 논문 리뷰

> **한 줄 요약**: 저수준 VLA 정책을 **동결(frozen)**한 채, 실행 이력과 두 가지 실행 품질 지표를 관찰하는 고수준 **agentic 정책(PPO)**이 {EXECUTE, RETRY, REPAIR, RESET} 네 가지 실행 모드를 선택하여, 저수준 정책을 재학습하지 않고도 LIBERO에서 표준 조건 최대 +13.7%, 외란 조건 최대 +39.2%의 성공률 향상을 달성한 실행 수준(execution-level) 복구 프레임워크.

## 1. 개요 (Overview)
본 논문은 로봇 조작에서 장기 실행 중 누적되는 오차로 인한 "실행 수준 실패(execution-level failure)"를 다룬다. VLA는 강력한 일반화 능력을 갖지만 실행이 정상 궤적에서 벗어났을 때 이를 감지하고 복구하는 명시적 메커니즘이 없다. 저자들은 저수준 정책의 액션을 직접 학습하는 대신, 저수준 정책을 언제 어떻게 적용할지를 결정하는 고수준 "agentic 정책"을 강화학습으로 학습한다. HIT Shenzhen과 Northeastern University 연구진이 arXiv(2607.13818, 2026-07-15)에 공개했다.

## 2. 문제 정의 (Problem Statement)
동결된 저수준 정책(예: VLA)이 태스크를 수행하지만, 외란·접촉 불확실성·오차 누적으로 로봇이 점차 정상 거동에서 이탈한다. 이를 POMDP M = <S, A, P, R, γ>로 정식화하며, A는 이산 실행 모드 집합이다. 목표는 시각 관측 없이 최근 실행 이력만으로 실행을 조율(regulate)하여 강건성을 높이는 것이다. 저수준 정책은 수정·재학습하지 않는다.

## 3. 핵심 기여 (Key Contributions)
(1) 런타임 실행 품질을 평가하는 두 개의 상보적 지표(local/global execution quality) 제안. (2) 저수준 액션을 직접 생성하지 않고 고수준 의사결정으로 실행을 관리하는 agentic 강화학습 프레임워크 제안. (3) 세 가지 복구 메커니즘(RETRY/REPAIR/RESET) 설계. (4) LIBERO 4개 서브셋과 다양한 저수준 정책(OpenVLA, π0, π0.5, Diffusion Policy)에서 일관된 성능·강건성 향상 검증.

## 4. 방법론 (Methodology)
**Local execution quality (q_local)**: 슬라이딩 윈도우 W 내 end-effector 이동 거리 대비 커맨드 액션 크기 비율로 motion effectiveness E를 계산(정체·jamming 감지), 속도의 역변동계수로 smoothness S를 계산한 뒤 가중합→시그모이드→EMA 평활화. **Global execution quality (q_global)**: 윈도우를 압축 특징 z_t로 인코딩하고, 배포 전 수집한 N=50개 성공 궤적을 진행률(progress ratio) 기준 B=10 bin으로 나눈 stage-aware reference library와 kNN(k=5) 거리로 비교, exp(-α d_t)로 점수화. 두 지표를 q_agg = λ q_local + (1-λ) q_global로 통합한다. **Recovery**: RETRY는 최근 M=15 step 중 q_agg 최대 상태로 롤백(OSC로 복귀, 그리퍼 개방); REPAIR는 접촉 없는 상태(MuJoCo 접촉력 임계 τ=5N) 중 최근 N=30 step 내 최고 q_agg로 더 강한 롤백; RESET은 에피소드를 초기 상태로 재시작.

## 5. 아키텍처 (Architecture)
고수준 agentic 정책과 critic 모두 경량 MLP로 구현된다. 정책은 K=5 저수준 step마다 결정하며, 길이 L=20의 실행 이력(최근 proprioception, 저수준 액션, 두 실행 품질 점수)을 관찰해 {0,1,2,3}에 해당하는 4개 이산 실행 모드에 대한 categorical 분포를 출력한다. Critic은 배포 시 사용 불가한 특권(privileged) 시뮬레이터 전역 상태에 접근하는 비대칭 구조로 희소·지연 보상 하에서 가치 추정을 개선한다. 저수준 정책은 항상 동결.

## 6. 데이터 및 학습 (Data & Training)
태스크별로 각 저수준 정책에 대해 50개 성공 궤적을 수집해 reference library를 구성한다. Agentic 정책은 태스크마다 별도로 PPO로 학습(최대 1,000,000 고수준 결정 step). 보상: 성공 +1.0, 실패 -1.0, 매 step -0.02 time penalty, 복구 비용 RETRY -0.1 / REPAIR -0.3 / RESET -0.5(EXECUTE 0). 하이퍼파라미터: lr 1e-4, γ 0.99, PPO clip 0.2, value coef 0.1, entropy coef 0.01. 롤백 지평 M=15, N=30은 학습 대신 고정(학습 시 불안정).

## 7. 실험 결과 (Experiments)
LIBERO 4개 서브셋에서 표준(nominal)·외란(disturbance) 두 설정으로 평가. 표준 조건 w/Agentic 평균 이득: Spatial +5.1, Object +5.4, Goal +6.6, Long +13.7. 외란 조건(임의 timestep에 δ=3.0 노이즈 5 step 주입) 평균 이득: +25.7 / +27.4 / +28.3 / +39.2로 훨씬 크다. 예: 외란 하 OpenVLA-Long 33.4→67.6, π0-Long 33.8→79.5. RESET이 호출된 에피소드는 공정성을 위해 실패로 계산한다.

## 8. 비교 분석 (Comparison)
네 가지 대표 저수준 정책(OpenVLA, π0, π0.5, Diffusion Policy)에 모두 적용해 프레임워크의 정책-무관(policy-agnostic) 특성을 보인다. 이득은 태스크 복잡도가 높을수록, 그리고 저수준 정책이 취약할수록 커진다. π0.5처럼 이미 포화에 가까운 강한 baseline은 성능을 저하 없이 유지(예: Long 92.4→95.2)하고, OpenVLA·Diffusion Policy 같은 취약 정책에서 최대 이득이 관측된다. VLM 기반 재계획(replanning) 방식 대비 의미론적 추론·검증 없이 실행 이력만으로 저지연 결정을 내린다는 점이 차별점이다.

## 9. 절제 연구 (Ablation)
**결정 분포(Q3)**: 저하 유형별로 mild(transient noise, grasp) → RETRY, collision/obstruction → REPAIR, irrecoverable → RESET으로 심각도에 단조 대응(Fig. 6). **품질 공간(Q4)**: local-global 품질 공간에서 EXECUTE는 고품질 영역, RETRY는 중간, REPAIR/RESET은 저품질 영역에 집중(Fig. 7). **품질 변화(Table III)**: EXECUTE는 -0.01/-0.02, 복구 액션은 양의 변화(RESET Δq_global +0.22, P(Δ>0)=0.95). **비용(Q5, Table IV)**: 취약 정책일수록 개입 빈도·에피소드 길이 증가가 크다(Diffusion Policy +2.1회, +15%; π0.5 +0.9회, +5%).

## 10. 강점 (Strengths)
저수준 정책 재학습 없이 plug-in으로 강건성 향상. 외란 조건에서 특히 큰 이득(+39.2%). 경량 MLP + 이산 결정으로 저지연. Proprioception과 저수준 액션만 사용(원시 시각·특권 시뮬레이터 상태 불필요)하여 sim-to-real 이식성이 높음. 복구 결정이 저하 심각도에 따라 해석 가능하게 분화됨.

## 11. 약점 (Weaknesses)
(1) agentic 정책을 태스크별로 개별 학습해야 하며 태스크당 성공 궤적 50개 수집이 필요. (2) 심각한 저하·OOD 상황 복구 능력은 여전히 제한적(저자 스스로 주요 한계로 명시). (3) 실험이 LIBERO 시뮬레이션에 한정되어 실제 로봇 검증은 미래 과제. (4) RESET을 실패로 계산하지만 실제 배포에서 잦은 RESET은 시간 비용이 큼. (5) 롤백 지평 M, N을 학습하면 불안정해 고정값에 의존.

## 12. 총평 (Overall Assessment)
저수준 액션 생성과 실행 관리를 분리한다는 "agentic" 관점은 단순하지만 실용적이며, 동결 정책에 적용 가능하다는 점에서 재사용성이 높다. 특히 외란 강건성 향상 폭이 인상적이다. 다만 태스크별 학습·궤적 수집 부담과 시뮬레이션 한정 검증은 실전 적용을 위해 해소해야 할 과제다. VLA 자체의 성능을 겨루는 모델이라기보다, 임의의 저수준 정책 위에 얹는 실행 수준 복구 래퍼로 이해하는 것이 타당하다.

<!-- VERIFIED: pdf -->
