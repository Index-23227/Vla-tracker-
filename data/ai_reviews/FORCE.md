# FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation

> arXiv:2606.26006 (2026-06-24) · Beijing Academy of AI / Peking University / CASIA

## 1. 배경 및 동기

VLA(Vision-Language-Action) 모델은 대부분 모방 학습(Imitation Learning, SFT)으로 사전학습되며, 시연 데이터 품질에 성능이 묶이는 "imitation ceiling"에 직면한다. 이 한계는 단순한 경험적 관찰이 아니라, 모방 학습의 오류가 태스크 horizon에 따라 2차적으로 누적된다는 통계적 한계(Ross et al., 2011; Rajaraman et al., 2020)에서 기인한다.

강화학습(RL) 미세조정은 환경과의 온라인 상호작용을 통해 이 천장을 넘을 수 있으나, 실세계 물리 상호작용에서 **샘플 비효율성**이 치명적이다. 저자들은 offline-to-online RL의 두 핵심 난점을 지목한다: (1) **Catastrophic Initial Unlearning** — 보수적 오프라인 Q-함수가 온라인 데이터에 의해 "속아" 초기 성능 붕괴를 겪는 현상(Q-value scale mismatch), (2) **Inefficient Policy Updates** — 미숙한 정책이 생성하는 저품질 탐색 데이터로 인한 비효율적 업데이트. 기존 방법은 이를 비싼 Human-in-the-Loop(HiL) 개입으로 우회해 왔다.

## 2. 방법론 심층 분석

FORCE는 **3단계 intervention-free 파이프라인**이다.

### 2.1 Stage 1 — Offline Cal-QL 사전학습
정적 전문가 데이터 D_E에 대해 Calibrated Q-Learning(Cal-QL)으로 보수적 critic을 학습한다. TD 오차에 calibration regularizer(하이퍼파라미터 alpha)를 더해 OOD 행동의 과대추정을 억제한다(Eq.1). Actor는 BC 항(η)과 Q-guided policy gradient 항(λ)을 결합한 behavior-regularized objective로 학습한다(Eq.2-4).

### 2.2 Stage 2 — Distributional (Value-Calibrated) Warm-up
O2O 전이의 핵심 문제인 covariate shift를 완화한다. 소량의 on-policy 궤적 D_warm를 수집해 오프라인 데이터와 병합(D_mix)하고, 이 혼합 버퍼에 보수적 가치 제약(Eq.1)을 재적용한다. 이로써 Q-함수의 유효 support가 supp(π_β)에서 supp(π_β) ∪ supp(π_φ)로 확장되어, 정책이 실제 방문하는 manifold 위에서 Q-추정이 well-defined·lower-bounded가 된다. Actor는 성공률 정규화 인자 ρ를 가진 비대칭 objective로 학습된다(Eq.5-7).

### 2.3 Stage 3 — VGPD (Value-Guided Policy Self-Distillation)
온라인 단계는 인간 개입 없이 진행되며, 전문가 버퍼 D_E(오프라인+성공 온라인)와 정책 버퍼 D_π(전체 rollout) 두 개를 균등 샘플링한다. Critic은 두 버퍼 합집합에서 표준 TD loss를 최소화한다(Eq.8). Actor는 VGPD로 업데이트된다(Eq.9): 상태당 K개 후보 행동을 샘플링하고, 경험적 평균값 q_mean(s)를 baseline으로 한 **Dynamic Advantage Filter**(Positive Advantage Truncation)로 음의 advantage 샘플을 폐기한 뒤, exponential energy weighting(온도 τ)으로 고가치 행동만 distill한다(Eq.13-15). 전문가 데이터는 BC로 degenerate된다(Eq.12).

## 3. 데이터 전략

전문가 시연 + on-policy 온라인 rollout을 함께 사용한다. 시뮬레이션은 ManiSkill, 실세계는 단일 암 Franka Emika Panda(wrist + side RealSense 2-camera)이다. 핵심은 정적 데이터셋과 동적 물리 상호작용 사이의 간극을, 버퍼 분리(expert/policy)와 균등 샘플링으로 메우는 것이다.

## 4. 시스템/학습 세부사항

Actor 네트워크로 **consistency policy**(다단계 diffusion/flow denoiser를 1-step으로 distill)를 채택해, 원 정책의 표현력을 유지하면서 빠른 추론과 안정적인 Q-gradient 전파, BPTT 문제 회피를 달성한다. backbone-agnostic 설계로 Octo와 pi0를 host VLA로 검증했다.

## 5. 실험 결과 심층 분석

### 5.1 ManiSkill 시뮬레이션 (Table 1)
6개 태스크 3-seed 평균 success rate. FORCE(Octo) **82.3%**, FORCE(pi0) **86.9%** — 최강 baseline ConRFT(no HIL) 71.1% 대비 10%+ 우위. Cal-QL은 StackCube/PlaceSphere 등 정밀·장기 태스크에서 정책 붕괴(PlaceSphere 0%). PushCube에서 FORCE는 5,000 step 내 100% 수렴(ConRFT 대비 sample complexity 32% 감소).

### 5.2 실세계 Franka (Table 2)
6개 태스크 평균 success rate **45.0%(BC) → 98.3%(FORCE)**, 4/6 태스크 100% 달성. 실행 step 112.8 → 38.9로 단축(Stack Cube 170.4→46.2, 약 4배). 인간 개입 없이 달성.

## 6. Ablation 분석 (Table 3, Steps@80%)

- **w/o Pre-Cal**: 온라인 시작 즉시 성능 저하 → warm-up이 cold-start 붕괴 방지의 핵심임을 확인.
- **w/o VGPD**: 조기 plateau, 80% 도달에 더 많은 step 필요(PickCube 12k→20k) 또는 미도달.
- 종합적으로 FORCE는 평균 Steps@80%를 **32.5%** 감소시켜 두 컴포넌트 모두 필수임을 입증.

## 7. 관련 연구 비교

PPO/GRPO 기반 on-policy 방법(SimpleVLA-RL, VLA-RL)은 시뮬레이션에 국한되고 autoregressive backbone에 구조적으로 제한된다. ConRFT(Chen et al., 2025)는 BC + Q-learning hybrid objective를 공유하지만, FORCE는 여기에 value pre-calibration 단계와 VGPD self-distillation을 추가해 안정성과 샘플 효율을 극대화한다. RL-100(Lei et al., 2025)의 imitation re-learning에서 value pre-calibration 아이디어를 차용했다.

## 8. 한계 및 미해결 문제

저자 스스로 밝힌 한계: VGPD 단계에서 상태당 K개 후보를 샘플링해야 하므로 **추론 비용 오버헤드**가 크며, 대형 VLA backbone 미세조정 시 부담이 된다. 향후 action caching이나 이전 iteration의 off-policy 샘플 재활용으로 완화 가능. 또한 현재 task-specific critic을 사용하므로, open-ended·장기 태스크 확장을 위해 Generalist Reward Model로의 일반화가 필요하다.

## 9. 이론적 기여

VGPD는 KL 제약 하 expected return 최대화(Eq.10)의 regularized policy improvement 문제에 대한 근사 해로 정식화된다. 최적 closed-form은 energy-based 분포 π*(a|s) ∝ π_old(a|s)·exp(Q/τ)이며, 이를 파라미터 공간으로 사영(DKL(π*‖π_φ) 최소화)하면 weighted log-likelihood 최대화와 동치다(Eq.11). **Proposition 1 (Monotonic Value Improvement)**: critic이 consistent하면 target 분포 하 기대 Q-value가 V_ref(s)로 하한되어, distillation target이 항상 현 정책 rollout 평균 성능 이상의 단조 개선을 보장한다.

## 10. 적응형 커리큘럼 (Fig.6)

VGPD는 수동 튜닝 없이 자동 커리큘럼을 구현한다. 초기에는 critic이 미숙한 정책 제안에 낮은 값을 부여해 advantage filter가 전문가 행동을 무겁게 가중(BC 안정자 역할). 정책이 향상되면 filter를 통과하는 self-generated 행동 비율이 증가해, 후기(예: PushCube)에는 on-policy 데이터 위주로 distill하며 sub-optimal 시연을 넘어선 fine-grained 최적화가 진행된다 — 즉 보수적 모방에서 공격적 자기개선으로의 자연스러운 전환.

## 11. 실무적 시사점

intervention-free라는 점이 실세계 배포에서 결정적이다. 물리 환경의 calibration 오차·센서 노이즈가 OOD 상태를 유발해도 Value-Calibrated Warm-up이 "안전 탐색 corridor"를 유지해 erratic motion을 방지하며, 실행 step 단축은 모션 품질(hesitant/jittery 거동 감소)의 proxy로 작동한다.

## 12. 종합 평가

FORCE는 offline-to-online RL의 두 고질병(initial unlearning, 저품질 탐색)을 각각 Distributional Warm-up과 VGPD로 원리적으로 공략하고, 시뮬레이션(SOTA, ConRFT 대비 +10%)과 실세계(45%→98.3%, 인간 개입 0)에서 강한 일관성을 보였다. 이론적 보장(단조 개선)과 backbone-agnostic 호환성(Octo·pi0)을 갖춘 범용 VLA post-training 프레임워크로서 의미가 크다. 추론 비용과 task-specific critic 의존이 남은 과제다.

<!-- VERIFIED: pdf -->
