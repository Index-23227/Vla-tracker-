# Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning

> **한 줄 요약**: GRPO 기반 VLA post-training에서 rollout 하나의 스칼라 advantage가 궤적 전체 액션에 그대로 broadcast되는 **trajectory-level credit aliasing** 문제를 지적하고, 탐지 가능한 **ordered task stage** 단위로 비교군과 credit 구간을 재구성. RoboTwin 2.0 macro avg **75.8%** (SimpleVLA-RL 68.8 대비 +7.0pp), LIBERO-Long **99.1%** 달성.

- arXiv: 2608.13026 (2026-08-13, cs.RO)
- 저자: Yao Zhou*, Hang Gao*, Fengge Wu, Changwen Zheng, Wenwen Qiang† (Institute of Software, CAS / UCAS)

---

## 1. 배경 및 동기

Outcome-driven RL은 value model 없이 "최종 성공/실패"만으로 사전학습 VLA를 post-train할 수 있어 확장성이 좋다. 대표적으로 GRPO 계열(SimpleVLA-RL, TGRPO)은 동일 초기 상태에서 G개의 완전한 rollout을 뽑고, 최종 outcome으로 group-relative advantage를 구해 그 rollout의 **모든** 액션에 동일하게 적용한다.

문제는 long-horizon 조작 태스크의 최종 outcome이 여러 stage outcome의 집계라는 점이다. 잡기까지 성공하고 마지막 놓기에서 실패한 rollout과, 애초에 그립조차 못 만든 rollout이 **같은 실패 보상 → 같은 음의 advantage**를 받는다. 그 음수가 실행된 액션 전체에 뿌려지므로, 이미 유효한 진행을 만들어낸 선행 행동까지 억제된다.

## 2. 문제 정식화 — Trajectory-Level Credit Aliasing

정책 π_θ(a_t | s_t) (또는 chunk a_{t:t+H-1}), rollout τ_i = (s_i,0, a_i,1, …, s_i,Ti), 최종 outcome R_i에 대해:

```
Â_i = (R_i - mean{R_j}) / (std{R_j} + ε)        ... (7)
Â_i,t = Â_i,  t = 1..|a_i|                       ... (10)
```

stage progress를 z_i = (z_i,1,…,z_i,K), z_i,k ∈ {0,1}로 두면, 논문은 다음 붕괴를 지적한다:

```
z_i ≠ z_j,  R_i = R_j  ⟹  Â_i = Â_j            ... (12)
```

즉 (a) 서로 다른 stage progress가 같은 trajectory advantage로 매핑되고, (b) 그 advantage가 모든 액션에 균일 배정되는 두 현상의 결합이 **credit aliasing**이다. 완전한 rollout이 "비교 단위"이자 "credit 배정의 시간 단위"를 동시에 맡고 있는 것이 근본 원인.

## 3. 방법론 — Temporal GRPO

### 3.1 Task-Conditioned Stage Generation
- `M̃ = F_sem(l, o_0)`: 동결된 **RynnBrain-4B** embodied planner가 지시문 l과 초기 관측 o_0로부터 후보 semantic stage를 제안 (예: Near → Grasp → Move → Place).
- `M = F_comp(M̃, l, o_0) = (m_1,…,m_K)`: **Stage Compiler**가 순서/선행조건 정규화, 추상 서술 → 탐지 가능한 완료조건 변환, 중복·모호 stage 정리를 수행. 결과는 선형 선행구조 m_1 → … → m_K이며, **최종 stage m_K는 원래 task-success 조건과 일치**시켜 local credit이 전역 목표와 어긋나지 않게 한다.
- 컴파일된 stage 시퀀스는 태스크 인스턴스마다 **한 번만** 생성되어 모든 rollout에 재사용.

### 3.2 Rollout-to-Stage Alignment
relation detector가 m_{k-1} 확정 후에만 m_k를 검증하며, 안정적으로 완료조건이 만족되는 최초 시점을 기록한다.

```
T_i,k = min{ t | m_k stably completed in rollout i }   ... (17)
B_i,k = (T_i,k-1, T_i,k],  T_i,0 = 0                   ... (18)
실패 시:  B_i,k = (T_i,k-1, T_i]                        ... (19)
```

stability 제약이 순간 접촉·일시적 공간 겹침·시각 흔들림을 완료로 오인하는 것을 막는다. 도달하지 못한 이후 stage는 구간도 비교군도 배정되지 않는다. **stage predicate는 privileged simulator state로 post-training 중에만 평가**되며 평가 시점에는 detector도 privileged state도 쓰이지 않는다.

### 3.3 Stage-Conditioned Advantage
```
V_i,k = 1 if (k=1 or rollout i completes m_{k-1}) else 0    ... (20)   # entered-stage gating
R_i,k = 1 if rollout i completes m_k else 0                  ... (21)
μ_k, σ_k = V로 가중된 유효 rollout에 대해서만 계산             ... (22)
Â_i,k = (R_i,k - μ_k) / (σ_k + ε),  V_i,k = 1                ... (23)
Â_i,t = Σ_k 1[t ∈ B_i,k] · Â_i,k                             ... (24)
```
핵심은 V_i,k=0인 rollout을 "그 stage의 실패"로 세지 않고 **비교군에서 제외**한다는 점. 유효 rollout이 없거나 outcome이 전부 동일한 stage는 랭킹 신호가 없으므로 해당 업데이트에서 skip된다. 구간이 순서적·비중첩이므로 각 액션(및 action chunk의 토큰들)은 최대 하나의 stage advantage를 상속한다.

최종 목적함수 (25)는 GRPO의 clipped ratio 구조를 그대로 두고 Â_i를 Â_i,t로 치환한 형태이며, 모든 stage 구간이 **동일 배치·단일 목적함수·단일 정책**으로 함께 최적화된다(stage별 독립 RL이 아님).

## 4. 실험 세팅

- **RoboTwin 2.0**: 전체 성능 + sample efficiency (task horizon별).
- **LIBERO-Long**: 통제된 credit assignment 분석 및 ablation. 4개 표준 LIBERO suite 전체 결과는 supplementary로 미룸.
- **Baselines**: π0, RDT-1B (일반 VLA), SimpleVLA-RL, TGRPO (VLA-RL), 그리고 통제 비교용 **Trajectory-GRPO**(최종 outcome advantage를 전체 시퀀스에 배정), **Stage-Reward GRPO**(동일한 stage 탐지를 스칼라 reward로만 변환, advantage 배정은 여전히 trajectory-level).
- 통제 RL 방법들은 모두 **동일한 task-specific OpenVLA-OFT SFT 체크포인트**에서 출발하고, 동일한 rollout/환경 상호작용/정책 업데이트 예산과 평가 프로토콜을 사용. 결과는 3 seed 평균±표준편차.

## 5. 주요 결과 — RoboTwin 2.0 (Table 1, 성공률 %)

| Method | Short | Medium | Long & Extra-Long | Macro Avg |
|---|---|---|---|---|
| π0 | 45.5 | 58.8 | 43.3 | 49.2 |
| RDT-1B | 24.5 | 47.8 | 27.8 | 33.3 |
| OpenVLA-OFT (SFT) | 21.3 | 47.1 | 46.5 | 38.3 |
| Trajectory-GRPO | 37.8±1.8 | 52.6±1.6 | 48.7±1.9 | 46.4±1.3 |
| TGRPO | 43.9±1.6 | 58.4±1.5 | 54.1±1.7 | 52.1±1.1 |
| Stage-Reward GRPO | 52.7±1.4 | 64.2±1.3 | 60.8±1.5 | 59.2±1.0 |
| SimpleVLA-RL | 64.9±1.2 | 72.5±1.0 | 69.0±1.3 | 68.8±0.9 |
| **Temporal GRPO** | **73.2±0.9** | **79.0±0.8** | **75.2±1.1** | **75.8±0.7** |

- 모든 horizon 그룹에서 1위. 최강 통제 baseline(SimpleVLA-RL) 대비 macro **+7.0pp**, horizon별 **+8.3 / +6.5 / +6.2pp**.
- 흥미로운 점: OpenVLA-OFT SFT는 Long & Extra-Long(46.5)이 Short(21.3)보다 높은 비대칭 프로파일인데, RL이 Short 구간에서 특히 큰 폭으로 끌어올린다.
- 표준편차도 Temporal GRPO가 가장 작다(0.7~1.1 vs Trajectory-GRPO 1.3~1.9) — stage-local 신호가 학습 분산까지 줄인다는 간접 증거.

**Sample efficiency (Fig. 2)**: 모든 방법이 step당 동일 rollout 수를 쓰므로 같은 training step = 같은 환경 상호작용 예산. Temporal GRPO는 전 구간에서 Trajectory-GRPO 위를 유지하며, 지연 보상이 더 심한 Long/Extra-Long subset에서도 격차가 유지된다.

## 6. Controlled Credit Assignment (Fig. 3)

자연 샘플링된 rollout 중, **같은 선행 stage를 모두 통과하고 stage m_d에서 처음 outcome이 갈리는** 매칭 그룹을 고른 뒤, stage에 진입한 rollout 조건부 완료확률 변화를 측정:

```
Δp_k = 100 · [ Pr_{τ~π_θ+}(R_k=1 | V_k=1) − Pr_{τ~π_θ}(R_k=1 | V_k=1) ]   ... (26)
```

- **Trajectory-GRPO**: 공유 선행 stage들에서 뚜렷한 **음의 Δp** → 음의 trajectory advantage가 이미 성공한 액션 구간까지 전파됨을 직접 관측.
- **Temporal GRPO**: 선행 stage의 Δp는 0 근처로 유지, m_d에서 **최대 양의 개선**. 즉 "습득한 선행 행동 보존 + 차이를 만든 stage에 업데이트 집중"이라는 설계 의도가 metric 수준에서 확인된다.

이 실험이 이 논문의 진짜 기여다. 성공률만으로는 "어느 stage가 좋아지고 나빠졌는지" 알 수 없는데, Δp_k는 aliasing 가설을 반증 가능한 형태로 측정한다.

## 7. Ablation (Table 2, LIBERO-Long, 3 seeds)

| Variant | Success Rate (%) |
|---|---|
| **Temporal GRPO** | **99.1±0.4** |
| w/o Stage Compiler | 96.8±0.7 |
| Stage-Reward GRPO | 94.7±0.9 |
| w/o entered-stage gating | 92.5±1.1 |
| w/o same-stage grouping | 90.6±1.3 |
| Trajectory-GRPO | 88.4±1.5 |

- **same-stage grouping 제거(-8.5pp)**가 가장 큰 손실 → 서로 다른 stage의 액션 구간은 신뢰할 만한 상대 비교군을 이루지 못한다.
- **entered-stage gating 제거(-6.6pp)**: 미진입 rollout을 실패로 세면 μ_k, σ_k가 오염되어 성능이 무너진다.
- **Stage Compiler 제거(-2.3pp)**: 순서화된 선행조건 + 탐지 가능한 완료조건이 alignment 신뢰도를 높인다.
- Stage-Reward GRPO(94.7) > Trajectory-GRPO(88.4)는 "stage 정보 자체가 유용"함을, Temporal GRPO(99.1) > Stage-Reward GRPO는 "**정보를 reward로 넣는 것보다 비교군과 배정구간을 바꾸는 것이 더 중요**"함을 분리해 보여준다. 설계가 깔끔하다.

## 8. 관련 연구와의 위치

- **HER / RUDDER / Align-RUDDER**: 목표 relabeling·return redistribution으로 지연 보상을 다루지만, 학습된 return 모델이나 궤적 정렬을 필요로 한다.
- **Segment feedback / stage-aware reward (SARM 등), Group-in-Group PO**: 감독 신호의 시간 해상도를 높이거나 유사 상태를 재그룹핑. 단, 연속 로봇 rollout은 동일한 시각 상태를 거의 재방문하지 않고 경로·속도·stage 길이가 크게 달라 raw-state 기반 cross-rollout grouping이 불안정하다.
- **Reward Machine / scene-graph 계열**: 태스크 명세·계획·보상 구성·궤적 분할에 주로 쓰였고 group-relative advantage 추정에 직접 결합된 적은 없다.

Temporal GRPO의 자리는 "검증 가능한 stage를 **advantage 추정의 비교 조건**으로 승격시킨" 지점이다. 아키텍처 변경도, 새 value model도 없다.

## 9. 한계 및 미해결 문제

- **신뢰할 수 있는 stage predicate 의존**: 시뮬레이터 privileged state로 평가된다. 실기(real robot)에서 동일 품질의 detector를 얻는 방법은 다루지 않았고, 실기 실험 자체가 없다.
- **선형 stage 순서 가정**: 분기(branching), 반복 stage, 이전 stage로의 recovery가 있는 태스크에서는 alignment와 credit 배정이 흔들린다(저자도 명시).
- **RynnBrain-4B 계획 품질에 대한 민감도 분석 부재**: stage 제안이 틀렸을 때 성능이 어떻게 저하되는지, 더 작은/다른 VLM으로 바꾸면 어떤지에 대한 수치가 본문에 없다.
- **LIBERO 4-suite 전체 결과가 supplementary**: 본문에는 LIBERO-Long 99.1만 있어 일반 LIBERO 리더보드와의 직접 비교가 제한된다.
- **stage 수 K에 대한 스케일링 분석 없음**: K가 커질수록 stage별 유효 rollout 수가 줄어 σ_k 추정이 불안정해질 텐데(skip 규칙으로 처리), 그 빈도 통계가 보고되지 않았다.

## 10. 재현성 체크

| 항목 | 상태 |
|---|---|
| 코드 공개 | 본문에 링크 없음 (미공개) |
| 베이스 체크포인트 | 공개된 task-specific OpenVLA-OFT SFT 체크포인트 (SimpleVLA-RL 배포) — 재현에 유리 |
| Stage Compiler 스키마/규칙 | supplementary |
| relation predicate, stability 기준 | supplementary |
| 하이퍼파라미터·평가 설정 | supplementary |
| seed | 3 seeds, 평균±표준편차 보고 (양호) |

## 11. 총평

**강점**: (1) 문제 정의가 정확하고 식 (10)/(12)로 깔끔하게 형식화됨. (2) Trajectory-GRPO / Stage-Reward GRPO라는 **직접 통제 baseline을 스스로 만들어** "stage 정보의 가치"와 "credit 배정 구조의 가치"를 분리 검증. (3) Δp_k 측정으로 메커니즘 주장을 성공률 밖에서 확인. (4) 아키텍처 불변, 단일 정책, 단일 목적함수 — 기존 VLA-RL 파이프라인에 얹기 쉬움.

**약점**: 시뮬레이터 privileged state 의존과 실기 검증 부재, 선형 stage 가정, planner 민감도 분석 부재.

**의의**: VLA-RL의 개선 축이 "reward를 더 촘촘히 설계"에서 "**비교 조건과 credit 구간을 재구성**"으로 이동할 수 있음을 보여준 사례. Stage-Reward GRPO를 4.4pp 앞선 결과가 그 주장을 가장 잘 뒷받침한다.

**추적 가치**: 높음. RoboTwin 2.0 SOTA급 수치(75.8)와 LIBERO-Long 99.1로 리더보드 상단 후보이며, 후속 연구(uncertainty-aware stage detection, dynamic stage graph)의 기준점이 될 가능성이 크다.

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 코멘트 |
|---|---|---|
| 1 | privileged simulator state 없이 실기에서 stage를 탐지할 수 있나? | 본문은 "post-training 중에만 사용, 평가 시 미사용"이라고 방어하지만 실기 학습에는 여전히 detector가 필요. 최대 약점 |
| 2 | stage 분할이 틀리면? | 오분할 시 성능 저하 곡선이 없음. w/o Stage Compiler(-2.3pp)가 유일한 간접 증거 |
| 3 | 유효 rollout이 적은 stage에서 σ_k 추정이 불안정하지 않나? | "유효 rollout 없음 또는 outcome 동일 → skip" 규칙으로 회피. skip 빈도 통계가 있어야 설득력이 생김 |
| 4 | RynnBrain-4B라는 추가 4B 모델의 비용은? | 태스크당 1회 생성 후 전 rollout 재사용이라 상각되지만, 공정 비교라면 baseline에도 동등 예산을 줘야 한다는 반론 가능 |
| 5 | SimpleVLA-RL 대비 +7.0pp가 credit assignment 때문인가, 하이퍼파라미터 때문인가? | SimpleVLA-RL/TGRPO는 "권장 설정"을 썼고 Trajectory-GRPO만 하이퍼파라미터를 공유. 가장 엄밀한 비교는 Trajectory-GRPO(46.4 → 75.8, +29.4pp) |
| 6 | LIBERO-Long 99.1은 포화 아닌가? | 88.4~99.1 범위라 ablation 변별력은 있으나, 상단이 천장에 가까워 다른 suite 결과가 필요 |
| 7 | 분기·재시도가 있는 태스크에서 선형 stage 가정이 깨지면? | 저자도 한계로 인정. dynamic stage graph를 future work로 제시 |
| 8 | action chunk가 stage 경계를 걸치면 어떻게 되나? | "각 chunk와 그 액션 토큰은 자신이 속한 stage 구간의 advantage를 상속"이라고만 기술 — 경계 chunk 처리 규칙이 모호 |
| 9 | 최종 stage m_K를 task-success와 일치시키는 것이 필수인가? | 이것이 local credit과 전역 목표의 정합성을 보장하는 장치. 어긋나면 reward hacking 여지가 생긴다 |
| 10 | Stage-Reward GRPO가 이미 59.2인데 왜 stage reward로 충분하지 않은가? | reward를 촘촘히 해도 advantage가 여전히 궤적 전체에 broadcast되므로 aliasing의 (b) 성분이 남는다는 것이 논문의 답 |

<!-- VERIFIED: pdf -->
