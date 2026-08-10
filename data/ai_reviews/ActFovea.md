# ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency

**arXiv**: [2607.29169](https://arxiv.org/abs/2607.29169) · **발표일**: 2026-07-31 · **소속**: 통지대학교(Tongji University) / MBZUAI / 상하이 인공지능연구원(Shanghai AI Laboratory) / 전자과기대(UESTC)
**저자**: Wenda Yu, Tianshi Wang, Fengling Li, Xin Li, Jingjing Li, Lei Zhu
**코드**: 미공개

> **한 줄 요약**: 재학습이나 파라미터 수정 없이 frozen VLA 정책의 **관측-행동 인터페이스** 위에서 동작하는 plug-and-play **런타임 안전장치(runtime safeguard)**. 로봇 kinematics·proprioception·최근 행동으로 **행동 조건부 foveation** 영역을 만들고, 시공간 visual-action 일관성으로 위험을 탐지하며, 복구 가능한 교란에는 후보 관측을 재구성해 **유도된 action chunk를 검증한 뒤에만** 실행하고, 복구가 불가능하면 **경계된 safe-failure**로 전환한다. π0 / LIBERO에서 visual overlay 성공률 49.3% → 90.3%.

---

## 1. 문제 정의 (Problem Statement)

VLA 정책은 대부분 **action chunk**를 예측한다 — 새로운 관측을 조건으로 삼기 전에 여러 제어 스텝을 미리 실행한다. 이 설계는 추론 비용을 분산시키고 시간적으로 일관된 동작을 만들지만, 동시에 신뢰할 수 있는 제어가 **시각 관측 · 로봇 proprioception · 실행 중인 행동 사이의 지속적 정렬**에 의존하게 만든다. 이 정렬이 깨지면 정책은 로봇의 현재 물리적 상태에 더 이상 대응하지 않는 시각 증거 위에서 행동한다.

런타임에서 이 정렬은 서로 다른 방식으로 깨진다: **국소적 시각 overlay**(공간 증거 오염), **시각 피드백 지연**(관측과 상태의 시간적 어긋남), **action-chunk drift**(행동 측 궤적 이탈), **frozen-observation replay**(신선한 시각 증거의 완전한 소실). 기존 대응은 교란 종류마다 별도의 탐지기를 두거나, 순수 모니터로서 문제를 **알리기만** 하고 행동을 바꾸지 않는다. 더 근본적으로, 어떤 교란은 복구가 정당하지만 어떤 교란은 **복구 시도 자체가 위험**하다는 구분이 없다.

## 2. 핵심 기여 (Key Contributions)

1. **단일 관측 가능 구조로의 통합**: 공간 오염 · 시간적 어긋남 · 행동 궤적 drift · 관측 replay를 교란 종류별 탐지기가 아니라 **시공간 visual-action 일관성**이라는 하나의 위반 구조로 정식화. 보상 신호, 시뮬레이터 물체 상태, segmentation 주석, 교란 종류에 대한 사전 지식을 전혀 요구하지 않는다.
2. **행동 조건부 foveation (action-conditioned foveation)**: 이미지 중심이 아니라 **예상되는 상호작용**을 따라가는 고해상도 영역. Forward kinematics · proprioception · 직전 행동으로부터 contact-critical 영역과 **예측 운동 회랑(motion corridor)**을 구성하고 나머지 배경을 편집 가능 영역으로 desensitize.
3. **행동 검증 기반 복구 (verify-then-execute)**: 순수 모니터와 달리 능동 개입한다. 교란 조건부 후보 관측(raw / foveated / 시간 안정화 / overlay deblend)을 만들고, 각 후보가 **유도하는 action chunk**를 고정 가중 점수로 검증한 뒤에만 scaled·shortened chunk를 실행.
4. **복구가 정당한지 자체를 판단**: 지속적 frozen-observation replay에서는 복구를 우회하고 **경계된 motion-suppression safe failure**를 latch. 평가된 어떤 baseline도 달성하지 못한 동작.

## 3. 방법론 (Methodology)

ActFovea는 frozen 정책 위의 4단계 루프다.

**(1) 행동 조건부 foveation**: Forward kinematics와 proprioception으로 end-effector를 이미지에 투영하여 **contact disk**를 만들고, 직전 행동으로부터 예측되는 운동 방향을 따라 **motion corridor**를 확장한 뒤 safety margin으로 dilate한다. 이 preserve mask 밖의 영역은 편집 가능한 배경으로 취급된다. 정성 결과(Figure 4)에서 dynamic fovea의 preserve 비율은 phase에 따라 13.5% → 11.1% → 12.3%로 **적응적으로 변하는** 반면, contact-only는 9.0%, static ROI는 27.3%로 고정된다 — 전자는 운동 회랑을 놓치고 후자는 상호작용 변화에 적응하지 못한다.

**(2) 시공간 일관성 모니터**: 기하학적 · 동역학적 · 시간적(timestamp health, lag, replay similarity) 증거와 action-proprioception 일치도를 융합해 risk score R_t를 산출. 결정론적 router가 교란 **종류(threat typing)**와 **복구 가능성(recoverable / borderline / unrecoverable)**을 추론한다.

**(3) 교란 조건부 후보 관측 뱅크**: raw, foveated, 시간 안정화, 그리고 추정된 pattern/alpha로 overlay를 deblend한 관측을 후보로 구성. Overlay 복구가 admission gate를 통과하면 raw와 deblended 관측을 비교하여 — 첫 3개 motion 행동의 방향과 RMS 편차가 일치하면 **raw를 유지**하고, 그렇지 않을 때만 재구성 증거가 강하고 검증 점수가 raw와 일관될 때 복구된 행동을 선택한다. **raw가 항상 보수적 fallback**이다.

**(4) 위험 적응 실행 + safe failure**: 2단계 arbitration. 먼저 모니터가 monitor scale λ_mon 과 horizon h_mon 을 산출하고(preserve / damp / shorten / hold), 다음으로 행동 검증이 normal · short-horizon · servo recovery · safe failure를 선택하며 선택적 cap λ_ver, h_ver 을 건다. 최종 실행은 â = λ_mon · λ_ver · a\*, h = min(h_mon, h_ver). Safe failure는 여기서 **보수적 motion suppression**을 의미하며, 저자들은 형식적 충돌 회피 보장은 본 메커니즘의 범위 밖이라고 명시한다.

## 4. 아키텍처 상세 (Architecture Details)

- **적용 대상**: frozen VLA 정책을 관측-행동 인터페이스를 통해 wrapping. 평가는 **π0의 LIBERO 체크포인트**(PaliGemma 기반 VLM + flow-matching action expert).
- **추가 학습 파라미터**: **0**. 모든 router 임계값과 verifier 가중치는 추론 시점 고정 상수이며, 학습 목적함수도 파라미터 최적화도 없다.
- **정책 수정**: 없음. 재학습·fine-tuning·gradient 접근 불필요.
- **비용 구조**: 추가 정책 호출은 **후보 확장이 정당화될 때만** 발생 — clean-like, 저위험 ambiguous, unrecoverable 증거에서는 억제된다.
- **입력 제한**: ActFovea는 결과 관측과 표준 런타임 입력만 받는다. **교란 라벨과 generator 파라미터는 제공되지 않는다.**

## 5. 실험 설정 (Experimental Setup)

- **정책/벤치마크**: frozen π0 LIBERO 체크포인트, 네 개의 10-task suite — **LIBERO-Spatial / Object / Goal / LIBERO-10**.
- **프로토콜**: 태스크당 50 에피소드 → **method-scenario cell 당 2,000 에피소드**. 모든 비교가 동일 체크포인트와 매칭된 태스크·실행 설정을 사용하며 ActFovea는 추론 시점에만 적용.
- **교란 4종**: Smooth Action-Chunk Drift(task-family별 phase 템플릿으로 chunk의 motion 부분을 phase 조건부 window에서 섭동), Multi-View Visual Feedback Delay(exterior·wrist 양 뷰를 proprioception보다 **3 프레임** 뒤로 유지), Persistent Localized Visual Overlay(짧은 clean warm-up 후 고정 checker 패턴을 양 카메라 스트림에 alpha-blend), Frozen-Observation Replay(trigger 프레임을 종료까지 양 뷰에서 재사용).
- **지표**: SR, Gain(percentage point), **NRR** = (S_{D+AF} − S_D) / (S_C − S_D) × 100%. Replay에서는 각 에피소드를 Task Success / Timely Safe Failure / Unprotected Failure로 분류.
- **베이스라인(training-free)**: Action Clip/Smoothing, Fixed Short Horizon, Timestamp-Only Hold.

## 6. 주요 결과 (Main Results)

**Table 1 — 네 LIBERO suite 평균 복구 결과**:

| 시나리오 | Base 무교란 | ActFovea 무교란 | Base 교란 | ActFovea 교란 | Gain | NRR |
|---|---|---|---|---|---|---|
| Action Drift | 92.7 | 93.0 | 83.1 | **90.1** | +7.0 | 73.1% |
| Visual Delay | 92.6 | 93.2 | 76.2 | **86.0** | +9.8 | 59.8% |
| Visual Overlay | 93.0 | 93.8 | 49.3 | **90.3** | **+41.0** | **93.7%** |

Visual overlay에서 49.3% → 90.3%로 clean 성능 대비 격차의 **93.7%를 회복**한다. 무교란 성능은 세 행 모두에서 Base 대비 **저하되지 않고 오히려 소폭 상승**(92.7→93.0, 92.6→93.2, 93.0→93.8)한다는 점이 중요하다 — 안전장치가 정상 동작을 희생시키지 않는다.

**Table 3 — frozen-observation replay 결과 (%)**:

| 방법 | Task Success | Timely Safe Failure | Unprotected Failure |
|---|---|---|---|
| Base VLA | 3.05 | 0.00 | **96.95** |
| Timestamp-Only Hold | 0.00 | 0.00 | **100.00** |
| w/o Hold/Safe-Fail | 0.65 | 0.00 | 99.35 |
| **ActFovea** | 0.00 | **100.00** | **0.00** |

ActFovea만이 **전 시행에서 timely safe failure**에 도달하고 unprotected failure가 0이다. 탐지 후 실행되는 경계 행동은 평균 **2.0 스텝**, 누적 action-space motion norm **0.326**, action-bound 위반 0건. Hold/Safe-Fail을 제거하면 탐지 후 **259.2 스텝**을 계속 실행하고 motion norm이 **241.98**까지 누적된다 — 즉 전체 executor가 탐지 후 행동 수를 **99.23%**, 누적 운동량을 **99.87%** 줄인다.

## 7. Training-Free Baseline 비교 (Table 2)

| 방법 | 무교란 | Action Drift | Visual Delay | Visual Overlay |
|---|---|---|---|---|
| Base VLA | 93.0 | 83.1 | 76.2 | 49.3 |
| Action Clip/Smoothing | 82.2 | 70.4 | 70.2 | 30.9 |
| Fixed Short Horizon | 91.7 | 89.9 | 70.7 | 32.4 |
| Timestamp-Only Hold | 93.1 | 84.9 | **0.0** | 48.5 |
| **ActFovea** | **93.8** | **90.1** | **86.0** | **90.3** |

핵심 관찰: **고정된 반응은 교란 종류를 넘어 전이되지 않는다.** Fixed Short Horizon은 action drift에서 89.9%로 ActFovea(90.1%)에 근접하지만 — 잦은 replanning이 이 교란에 잘 맞는다는 것을 확인해 준다 — visual delay 70.7%, overlay 32.4%로 무너진다. Action Clip/Smoothing은 무교란 성능 자체를 82.2%로 떨어뜨리고 세 교란 모두에서 Base보다 낮다. Timestamp-Only Hold는 무교란 동작을 보존하고 overlay에서 48.5%를 내지만, **연속적 시각 지연이 timestamp 규칙을 계속 활성화시켜 성공률이 0%가 된다**. ActFovea는 무교란 성능을 보존하면서 세 복구 가능 교란 **모두**를 개선한 유일한 런타임 방법이다.

## 8. Ablation 및 분석 (Table 4)

교란별 SR과 Gain (Gain은 Table 1의 교란 Base 대비):

| 변형 | Action Drift | Visual Delay | Visual Overlay |
|---|---|---|---|
| w/o Threat Typing | 87.5 (+4.4) | 80.5 (+4.3) | 41.7 (**−7.6**) |
| w/o Recovery Bank | 87.5 (+4.4) | 84.0 (+7.8) | 16.0 (**−33.3**) |
| w/o Candidate Expansion | 90.6 (+7.5) | 85.4 (+9.2) | 17.6 (**−31.7**) |
| w/o Action Verification | 81.9 (**−1.2**) | 78.5 (+2.3) | 92.1 (+42.8) |
| **Full ActFovea** | 90.1 (+7.0) | 86.0 (+9.8) | 90.3 (+41.0) |

**명확한 역할 분담(division of labor)**이 드러난다:
- **행동 검증**은 시간·행동 측 복구의 핵심 메커니즘이다. 제거하면 drift gain이 +7.0 → **−1.2**(Base보다 나빠짐), delay gain이 +9.8 → +2.3으로 붕괴한다.
- **관측 복구 계열**(recovery bank, candidate expansion, threat typing)은 공간 복구의 핵심이다. 제거하면 overlay gain이 +41.0에서 각각 −33.3 / −31.7 / −7.6으로 무너진다.
- 흥미롭게도 overlay에서는 행동 검증 없는 변형이 **+42.8**로 full보다 높다 — 행동 검증이 보수적 수용 게이트로서 공간 복구에서는 약간의 비용을 지불한다는 정직한 신호다.
- **Threat typing은 visual delay에서 특히 중요**: 모든 증거를 unknown 상태로 라우팅하면 gain이 9.8 → 4.3으로 떨어진다.

## 9. 강점 (Strengths)

1. **완전한 training-free / 정책 불가지론**: 추가 학습 파라미터 0, frozen 체크포인트 수정 없음. 관측-행동 인터페이스만 요구하므로 배포 파이프라인에 삽입 비용이 낮다.
2. **무교란 성능 무손실**: 안전장치의 통상적 대가인 clean 성능 저하가 없다(세 시나리오 모두 Base 대비 소폭 상승). Action Clip/Smoothing이 82.2%로 떨어지는 것과 대조적.
3. **큰 표본**: method-scenario cell 당 2,000 에피소드(40 태스크 × 50)로 SR 차이의 통계적 신뢰도가 높다.
4. **탐지에 그치지 않는 개입**: 후보 관측 재구성 → 유도 action chunk 검증 → scaled/shortened 실행이라는 verify-then-execute 구조가 순수 모니터와 실질적으로 구분된다.
5. **복구 가능성 판단의 독창성**: "복구할 수 있는가"가 아니라 "**복구를 시도하는 것이 정당한가**"를 묻고, unrecoverable에서는 경계된 safe failure로 latch. 99.23% / 99.87% 감소라는 정량적 근거가 붙는다.
6. **정직한 ablation**: overlay에서 행동 검증 제거가 오히려 낫다는 결과를 숨기지 않고 역할 분담의 증거로 제시한다.
7. **교란 정보 비노출**: 교란 라벨과 generator 파라미터를 받지 않으므로 oracle 누수가 구조적으로 차단된다.

## 10. 약점 및 한계 (Weaknesses & Limitations)

1. **단일 정책 · 단일 벤치마크**: 평가가 **π0 하나, LIBERO 시뮬레이션 하나**로 국한된다. 다른 아키텍처(OpenVLA, π0.5 등)로의 transfer 실험이 없어 "policy-agnostic"이라는 주장이 경험적으로 뒷받침되지 않는다. **실기(real-robot) 실험도 없다.**
2. **런타임 오버헤드 미보고**: 후보 확장이 조건부라고 서술하지만 **지연/처리량 수치나 확장 발동 빈도 통계가 논문에 제시되지 않는다**. 안전장치를 제어 루프에 넣는 방법론에서 이는 중요한 누락이다.
3. **합성 교란**: 네 교란 모두 저자가 정의한 파라미터로 주입된 합성 교란이다(3 프레임 지연, 고정 checker 패턴 등). 실제 배포에서 발생하는 센서 노이즈·부분 가림·조명 변화·네트워크 지터 등 자연 발생 교란에 대한 증거가 없다.
4. **Delay NRR 59.8%로 낮음**: visual delay는 세 교란 중 회복률이 가장 낮아, 시간적 어긋남에 대한 복구가 여전히 부분적임을 보여준다.
5. **안전 보장 부재**: 저자 스스로 명시하듯 safe failure는 "보수적 motion suppression"일 뿐 **형식적 충돌 회피 보장이 아니다**. 안전 critical 배포에는 불충분.
6. **하이퍼파라미터 다수 고정 상수**: router 임계값, verifier 가중치, safety margin 등이 "fixed implementation constants"로 서술되지만 값과 민감도 분석이 제시되지 않아 재현성과 새 환경으로의 이식성이 불확실하다.
7. **Per-suite breakdown 없음**: 모든 결과가 네 suite 평균이라 특정 suite에서의 실패 양상을 볼 수 없다.
8. **코드 미공개**: 규칙 기반 구성요소가 많은 시스템에서 코드 부재는 재현을 특히 어렵게 만든다.

## 11. 의의 및 향후 방향 (Significance & Future Work)

이 논문의 개념적 기여는 **서로 다른 런타임 실패들을 하나의 관측 가능한 위반 구조로 환원**한 데 있다. 공간 오염, 시간 지연, 행동 drift, 관측 정지는 표면적으로 다른 문제지만 모두 "시각 증거 · 로봇 상태 · 실행 행동의 시공간 정렬 붕괴"로 읽을 수 있다는 관점은, 교란마다 탐지기를 늘리는 접근보다 확장성이 좋다.

실무적으로 더 중요한 것은 **safe failure를 일급 결과로 승격**시킨 점이다. Timestamp-Only Hold가 replay 질의의 96.55%를 hold로 보내면서도 **끝내 terminal safe failure로 전환하지 못한다**는 관찰 — 즉 "지속적 무동작"과 "경계된 실패 프로토콜"은 같지 않다는 구분 — 은 VLA 안전 논의에 유용한 개념적 정리다.

향후 방향: (a) 다중 아키텍처 및 실기 검증, (b) 오버헤드/지연 프로파일링과 실시간 제어 주파수 하의 실현 가능성, (c) 자연 발생 교란으로의 확장, (d) 고정 임계값을 학습되거나 보정 가능한 형태로 대체, (e) safe failure에 형식적 안전 보장을 결합.

## 12. 총평 (Overall Assessment)

ActFovea는 **"모니터링"과 "개입" 사이의 간극을 메우는** 잘 설계된 런타임 안전장치다. 순수 탐지기에 머무르지 않고 후보 관측을 재구성하여 그것이 유도하는 action chunk를 검증한 뒤 실행하며, 복구가 정당하지 않을 때는 경계된 safe failure로 전환한다. Visual overlay 49.3% → 90.3%(NRR 93.7%)와 replay에서의 100% timely safe failure / 0% unprotected failure는 설득력 있는 결과이고, **무교란 성능을 전혀 희생하지 않았다**는 점이 실용성을 뒷받침한다. Ablation은 관측 복구와 행동 검증이 서로 다른 교란 축을 담당한다는 역할 분담을 깔끔하게 입증한다.

다만 평가 범위가 **π0 + LIBERO 시뮬레이션 단일 조합**이고, 실기 검증과 런타임 오버헤드 수치가 모두 부재하며, 교란이 전부 합성이라는 점에서 배포 준비도(deployment readiness) 주장은 아직 이르다. 새로운 VLA 정책이 아니라 **frozen 정책 위의 추론 시점 안전 개입 wrapper**로서, VLA 안전성 연구 라인의 의미 있는 진전이되 일반화 증거는 후속 연구를 기다려야 한다.

<!-- VERIFIED: pdf -->
