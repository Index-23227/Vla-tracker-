# Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models

> **한 줄 요약**: 냉동(frozen) VLA(π0) 위에 얹는 잔차(residual) RL이 각 서브태스크의 성공률은 올려도 **연결(composition)** 시에는 이득이 사라지는 실패 모드를 규명하고, 그 원인이 **핸드오프(terminal) 상태 품질의 방치**임을 지적. 각 서브태스크의 희소 성공 보상에 **오프라인 추정 foresight value**(다음 서브태스크 성공 확률)를 곱해 핸드오프 상태를 성형(shaping)하는 **Foresight Residual RL** 제안. Isaac Gym 렌치-너트 조임 3단계(Grasp/Move-Insert/Rotate) 조립 태스크에서 **풀태스크 성공률 85.6%** 달성 — 표준 잔차 RL 54.5%, end-to-end π0 54.9%, 체인 π0 41.4%를 크게 상회. Rutgers University, IROS 2026 (arXiv:2607.16506v1).

---

## 1. 배경 및 동기

- VLA(π0, OpenVLA, Octo 등)는 강력한 범용 조작 prior를 제공하지만, **타이트 톨러런스·접촉 풍부(contact-rich) 조립**에서는 장기 크레딧 할당과 서브태스크 결합(coupling) 때문에 자주 실패
- 표준 관행: 장기 태스크를 서브태스크로 분해 → 각각 정책 학습 → 실행 시 체이닝. 이는 학습 지평(horizon)을 관리 가능하게 줄이고, 사전학습 정책(VLA) 재사용과 서브태스크별 독립 개선을 가능하게 함
- 그러나 **많은 조립 문제는 독립 서브태스크로 분해되지 않음**: 렌치로 너트를 조일 때, 로봇은 렌치를 삽입한 상태를 회전 내내 능동적으로 유지해야 함 — 중력·기하로 수동 보존되지 않고 접촉력에 맞서 유지해야 하는 **인과적 결합(causal coupling)**
- 핵심 관찰: **현재 서브태스크의 종료(terminal) 상태 품질**(= 미래 성공 확률로 측정)은 기하학적으로 성공한 완료들 사이에서도 크게 다른데, 상수 보상 학습은 이 변이를 완전히 무시. 불안정한 그립은 삽입 성공 기준을 만족해도 삽입 단계에서 실패

## 2. 방법론 심층 분석

### 2.1 문제 정식화 — Sequential Semi-MDP (§III)
- 장기 태스크를 K개 순차 phase로 분해, 각 phase k는 성공 술어(predicate) β_k: S→{0,1}
- phase는 **인과적으로 결합** — phase k+1은 phase k가 확립한 성공 조건을 유지하면서 자기 목표를 추구
- 목표: 최종 phase 성공 보상 β_K의 기대 return 최대화 (식 1). 유일한 보상원은 β_K
- 표준 접근(식 2): 각 phase를 상수 종료 보상 r = β_k(s_{t+1})로 독립 학습 → local value V_k 최대화. 하지만 다음 서브태스크 성공 여부를 무시하므로 장기적으로 준최적

### 2.2 Foresight Value Function (§IV)
- **정의 IV.1**: phase k(<K)에서 미래 정책 π_{k+1:K}를 고정했을 때, phase k+1 시작 상태 s에서의 기대 할인 최종 phase 성공 (식 3). 종료 상태 s가 이후 서브태스크에 미치는 영향을 포착
- **Local foresight value function**(식 4): local value의 종료 보상 β_k에 V_{k+1:K}(s_{t+1})를 곱함 — 유일한 차이는 phase-k 성공 보상과 상태 의존 foresight value의 곱. 보정이 국소적이라 각 phase 내부 최적화 과정은 보존
- 이론적 배경: **MAXQ의 completion function**을 선형 태스크 그래프에 제한한 직접 인스턴스. 차이는 운영적 — 결합 계층에서 TD로 온라인 학습하는 대신, 고정 다운스트림 정책의 **Monte-Carlo 롤아웃으로 오프라인 추정** 후 이미지 기반 predictor로 distill (frozen 사전학습 base 정책 사용 가능, 비정상성 회피)

### 2.3 학습 파이프라인 (§V)
- **Residual policy**(식 5): a_t = π_k^0(o_t) + α·π_k^res(o_t), α=0.1(≪1). base VLA는 냉동, 잔차가 종료 상태를 higher foresight value 영역으로 이동
- **Foresight 값 추정 (Alg. 2)**: (1) base 정책 π_k^0 롤아웃으로 N개 성공 종료 상태 수집, (2) 각 상태에서 다운스트림 π_{k+1}을 K_rep회 롤아웃해 성공 카운트 κ_i로 라벨 y_i=κ_i/K_rep 생성, (3) **binomial NLL(식 8)**로 predictor p_φ 학습 — 라벨의 카운트 구조를 존중해 보정된(calibrated) 확률 산출. base 분포에서 수집하는 것은 의도적(잔차가 방문할 상태에서 정확해야 함)
- **One-step foresight 근사**: 전체 다운스트림 value를 즉시 다음 서브태스크 성공 확률로 근사 — k=K-1이거나 이후 성공률이 조건부 상수일 때 정확
- **Backward Foresight Induction (Alg. 1)**: k=K,…,1 역순. k=K는 표준 보상으로 π_K^res 학습(foresight 불필요, V_{K+1:K}≜1); k<K는 phase-k 종료 상태에서 π_{k+1:K}를 평가해 V_{k+1:K} 추정 후 foresight 보상 rt = β_k(s_{t+1})·p_φ(o_{t+1})(식 9)로 학습. 역순이 필수인 이유: 서브태스크 k 학습에 다운스트림 정책이 이미 고정 학습되어 있어야 foresight 추정이 정상적(stationary)

## 3. 실험 설정

- **태스크**: NVIDIA Isaac Gym(GPU 가속 PhysX)의 **렌치-너트 조립** — 7-DoF Kuka IIWA + 11-DoF Robotiq 3-finger 그리퍼가 홀더에서 렌치를 잡고, 육각 너트에 삽입하고, 삽입 유지하며 최소 60° 반시계 회전
- **3 phase**: GRASP(160 steps) / MOVE-INSERT(200) / ROTATE(200), 최대 560 steps, 20 Hz 제어
- **관측**: RGB 2뷰(3인칭+손목, 256×256) + 8D 자기수용감각(EEF pose + 이진 그리퍼). 힘 피드백·GT 물체 pose 없음. 액션: 7D OSC 델타(3D 위치, 3D axis-angle 회전, 1D 그리퍼)
- **난이도**: 5개 이산 렌치-너트 크기 변형, 렌치 헤드-너트 간극 <2mm(서브밀리미터 정밀 요구). 리셋 시 렌치 홀더·너트·볼트 포즈 랜덤화
- **base VLA**: 범용 π0를 각 서브태스크 데모(2,560개/서브태스크, state-machine 전문가 + 삽입은 사전학습 상태기반 RL 정책)로 개별 파인튜닝 → phase별 3개 base 정책. action chunking H=20
- **Residual/predictor 학습**: 잔차는 손목 뷰만 입력(DINOv2-S/14 16×16 패치 → CNN → 2-layer LSTM → MLP actor/critic, actor 출력 zero-init). PPO(rl_games) 128 병렬 환경, critic-only warmup 10 iter + joint 300 iter. Predictor는 두 뷰 CLS 토큰 self-attention 융합 + sigmoid, LoRA rank 8, 80/10/10 split 20 epoch(<5분)
- **평가**: 512 에피소드 × 4 시드. foresight predictor는 **학습 시 보상으로만** 사용, 평가 시 역할 없음. 서브태스크는 oracle init(전문가 성공 종료 상태에서 시작), consecutive/full은 특권 초기화 없이 end-to-end

## 4. 핵심 결과 — 풀태스크 성공률 (Table II)

| Method | Grasp | Insert | Rotate | Grasp+Insert | Insert+Rotate | Full Task |
|---|---|---|---|---|---|---|
| π0 (end-to-end) | — | — | — | — | — | 54.9 ± 5.3 |
| π0 (chained) | 87.1 | 45.7 | 93.4 | 37.1 | 61.5 | 41.4 ± 3.3 |
| π0 + Residual | **98.4** | **92.2** | 99.2 | 55.5 | 83.4 | 54.5 ± 3.6 |
| π0 + Residual + Foresight (ours) | 95.7 | 91.4 | **99.8** | **87.3** | **91.8** | **85.6 ± 3.9** |

- oracle init에서 상수 보상·foresight 잔차 **둘 다 서브태스크 성공률 >91%** 로 유사 — 즉 foresight는 개별 능력을 바꾸지 않음
- 그러나 **상수 보상 잔차 RL은 이 이득을 풀태스크로 옮기지 못함**: 54.5%로 end-to-end π0(54.9%)와 사실상 동일. Insert 서브태스크를 45.7→92.2%로 두 배 올렸음에도 무의미
- **Foresight 보정이 잃어버린 값을 회복** → 85.6%, 상수 보상 변형 대비 +31.1%p (논문 초록/기여 표기)

## 5. 이득이 집중되는 지점 — Grasp→Insert 경계

- consecutive 성공률 개선: Grasp+Insert **55.5 → 87.3% (+31.8%p)**, Insert+Rotate 83.4 → 91.8%
- 이는 predictor의 판별 마진과 일치: Grasp→Insert 경계 정확도 86.2%(다수 클래스 baseline 73.9% 대비 큰 격차), Insert→Rotate 93.9%(baseline 90.4%) (Table I)
- **분해가 오히려 해로울 수 있음**을 확인: 체인 π0(41.4%)가 end-to-end π0(54.9%)보다 낮음 — 통제되지 않은 각 핸드오프가 상태 불일치를 누적. Foresight 보정은 이 tradeoff를 역전시켜 분해 접근이 모놀리식을 능가하게 만듦

## 6. 핸드오프 상태 품질 분석 (Table III)

| Method | Grasp terminal 점수 | Insert terminal 점수 |
|---|---|---|
| π0 (base, no residual) | 0.64 | 0.97 |
| + Residual | 0.32 | 0.86 |
| + Residual + Foresight (ours) | **0.73** | **0.91** |

- **반직관적 발견**: 상수 보상 잔차 RL은 서브태스크 성공률을 올리면서 **핸드오프 상태 품질(predictor 점수)을 오히려 저하** — Grasp terminal 0.64→0.32. 성공 술어는 더 확실히 만족하지만 다운스트림 삽입에 나쁜 구성을 생성
- Foresight 보정은 이 저하를 역전 — Grasp terminal 최고 점수 0.73, Insert terminal 품질도 0.91로 base(0.97)에 더 근접 유지. 이것이 Table II의 풀태스크 격차를 직접 설명

## 7. 정성 분석 (Fig. 3)

- 두 정책은 동일 시작 상태에서 출발해 현재 서브태스크는 모두 성공(중간 열: 기하학적으로 유효한 그립/삽입) — Table II의 유사한 서브태스크 성공률과 일치. **발산은 즉시 다음 서브태스크에서 발생**
- Grasp→Insert: 상수 보상 정책은 성공 술어는 만족하나 운반 정렬이 나쁜 방향으로 렌치를 잡음 → INSERT 인수 시 접촉 하에 오정렬 누적, 삽입 실패. Foresight 잔차는 더 중심 잡힌·각도 정렬된 그립 생성 → 삽입 여유 확보
- Insert→Rotate: 상수 보상은 접촉은 이루나 안정성이 부족한 pose → 회전 토크 시 렌치 이탈. Foresight는 더 단단한 안착(seat) 유지 → 회전 내내 접촉 지속
- 일반 원리: 서브태스크 성공 술어는 넓은 종료 상태 등가류를 정의하지만 **일부만 다운스트림과 잘 결합**. 술어에 보이지 않는 작은 기하 차이가 후속 phase 접촉 역학에서 증폭됨

## 8. 강점

1. **실패 모드의 명료한 규명**: "서브태스크 성공률 향상 ≠ 조합 성능 향상"을 통제 실험(oracle init 서브태스크 >91% vs 풀태스크 54.5%)으로 깔끔히 분리 입증
2. **이론적 정합성**: foresight value를 MAXQ completion function의 선형 태스크 그래프 제한으로 위치시켜 원리적 근거 제공
3. **실용적 구현**: 오프라인 Monte-Carlo + binomial NLL predictor + backward induction으로 결합 계층 학습의 비정상성을 회피, **냉동 사전학습 VLA(π0)와 호환**되는 단일 패스 절차
4. **핸드오프 품질을 직접 계측**한 Table III의 반직관적 증거(상수 보상이 품질을 저하)가 메커니즘 주장을 강하게 뒷받침
5. 견고한 평가 프로토콜: 512 에피소드 × 4 시드, 서브태스크/consecutive/full 3단 granularity, 평가 시 predictor 미사용(순수 정책 성능)

## 9. 약점 / 한계

1. **시뮬레이션 전용**: 저자도 인정 — 실세계 배치의 주 장애물을 잔차 정책의 온-하드웨어 온라인 RL 비용으로 예상(접촉 역학 gap보다)
2. **단일 태스크·단일 분해**: 렌치-너트 조립 하나, 알려진 phase 분해에만 검증. 추가 태스크·분해·다른 VLA 백본은 향후 과제
3. **표준 리더보드 부재**(LIBERO/CALVIN/SimplerEnv 등) — 타 VLA와 직접 수평 비교 불가, 커스텀 Isaac Gym 태스크 기반
4. **one-step foresight 근사**: 깊은 다운스트림 의존 시 근사가 부정확할 수 있으며 롤아웃 체이닝으로만 확장 (K_rep=5, N=2560의 표본 비용)
5. **α=0.1 잔차 스케일 의존**: 분포 이동을 억제해 predictor 보간 범위 유지에 필요하나, 큰 보정이 필요한 태스크로의 일반성은 미검증
6. base 정책이 상수 보상 잔차 학습 하에서 오히려 핸드오프 품질을 **저하**시킨다는 점은 foresight 없이 잔차 RL을 순진하게 적용하면 위험함을 시사(방법의 강점이자 잔차 RL 일반의 경고)
7. 전문가 데모가 homogeneous(고정 전략, 복구 행동 없음)라 좁은 종료 분포 — 더 다양한 base 정책에서의 predictor 정확도는 미탐구

## 10. 다른 연구와의 위치

- **Residual RL 계보**: ResiP(냉동 diffusion policy 위 per-step 잔차), ResFiT(off-policy 잔차 파인튜닝), PLD(π0 포함 VLA에 probe-learn-distill 잔차 RL)의 형식을 계승하되, **크로스-phase(다운스트림 성공 반영) foresight 보상**을 도입한 점이 이들과 결정적으로 다름 — 기존은 모두 개별 태스크·표준 보상
- **Skill sequencing / initiation overlapping**: T-STAR(적대적 종료 상태 정규화), SCaR(이중 정규화), Sequential Dexterity(전이 타당성 함수), Bagaria et al.(초기화 집합 분류기)은 모두 **분포적 프록시(overlap)** 를 최적화. 본 논문은 이 프록시를 **다음 phase의 실제 성공 확률**로 대체 — overlap이 있어도 다운스트림 정책 편향으로 저성능일 수 있다는 지적
- **계층 RL value decomposition**: MAXQ completion function, options framework, option-critic, reward-respecting subtasks(Sutton et al.)의 이론적 토대를 계승. 차이는 온라인 TD 대신 **오프라인 Monte-Carlo + distill**로 냉동 base 정책과 residual RL에 호환되는 단일 패스 backward induction 실현

## 11. 향후 연구 방향

- **실세계 이전**: base VLA와 오프라인 foresight predictor는 표준 이미지 파이프라인으로 전이되므로, 주 과제는 잔차 정책의 실기 온라인 RL 비용 절감
- **다양한 태스크·분해·백본**: 다른 조립/도구 사용 태스크, 자동 phase 분해, 대체 VLA 백본으로 일반성 검증
- **깊은 foresight**: one-step 근사를 넘어 다중 phase 롤아웃 체이닝으로 더 먼 다운스트림 의존 포착
- **표준 벤치마크 평가**: LIBERO/CALVIN 등으로 커뮤니티 비교 가능성 확보
- predictor 분포 이동 강건화 — 잔차 개선에 따라 방문 상태가 변할 때의 재보정 전략

## 12. 종합 평가

Foresight Residual RL은 "서브태스크 성공률을 올리는 것과 서브태스크들을 잘 **조합**하는 것은 다른 문제"라는, 장기 조작에서 자주 방치되던 핵심을 통제 실험으로 예리하게 드러낸 논문이다. 핸드오프 상태 품질을 다음 서브태스크 성공 확률(foresight value)로 정량화하고, 이를 오프라인 Monte-Carlo로 추정해 냉동 VLA(π0) 위 residual RL 보상으로 사용하는 설계는 MAXQ completion function의 실용적 인스턴스로서 이론적·공학적으로 모두 정합적이다. 특히 상수 보상 잔차 RL이 서브태스크 Insert를 45.7→92.2%로 두 배 올리고도 풀태스크는 54.5%로 end-to-end와 동일한 반면, foresight 보정만으로 85.6%(+31.1%p)에 도달한다는 결과와, 그 원인이 핸드오프 품질 점수(0.32→0.73)로 직접 설명된다는 점이 설득력의 핵심이다. 시뮬레이션 단일 태스크·표준 리더보드 부재라는 한계로 타 VLA와의 수평 비교는 어렵지만, "성공 여부가 아니라 어떤 성공 상태를 만드는가를 최적화하라"는 메시지는 접촉 풍부 장기 조작에서 VLA + residual RL을 조합하려는 후속 연구에 실질적 설계 원리를 제공한다. 표준 벤치마크 점수는 없어 벤치마크 트랙에는 등재되지 않으며, 커스텀 Isaac Gym 렌치-너트 조립 태스크의 풀태스크 85.6% 결과로 기록된다.

<!-- VERIFIED: pdf -->
