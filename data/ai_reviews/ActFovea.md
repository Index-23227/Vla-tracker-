# ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency

**arXiv**: [2607.29169](https://arxiv.org/abs/2607.29169) · **발표일**: 2026-07-31 · **소속**: 통지대학교(Tongji University) / MBZUAI / 상하이 인공지능연구원(Shanghai AI Laboratory) / 전자과기대(UESTC)
**저자**: Wenda Yu, Tianshi Wang, Fengling Li, Xin Li, Jingjing Li, Lei Zhu
**코드**: 미공개

> **한 줄 요약**: 재학습·파라미터 수정 없이 frozen VLA의 **관측-행동 인터페이스**에만 붙는 plug-and-play 런타임 안전장치. 로봇 kinematics·proprioception·직전 action으로 **행동 조건부 fovea(주시 영역)** 를 만들어 시공간 visual-action 일관성을 감시하고, 복구 가능하면 후보 관측을 재구성해 **유도된 action chunk를 검증한 뒤 실행**하며, 복구 불가능하면 **경계된 safe failure**로 전환한다. pi0 / LIBERO 4개 suite에서 visual overlay 성공률 49.3% → 90.3%(손실의 93.7% 회복), visual delay +9.8pt, action drift +7.0pt, clean 성능 보존.

---

## 1. 문제 정의 (Problem Statement)

현대 VLA 정책은 대부분 **action chunk**를 예측한다. 한 번의 관측으로 여러 제어를 연속 실행하므로 추론 비용이 분산되고 모션이 매끄러워지지만, 그 대가로 **관측·고유수용감각(proprioception)·실행된 행동 사이의 시간적 정렬**에 신뢰성이 통째로 의존하게 된다. 이 정렬이 깨지면 정책은 이미 로봇의 현재 물리 상태와 무관해진 시각 증거 위에서 행동한다.

논문은 이 붕괴를 네 가지 런타임 교란으로 구체화한다: **공간 손상**(국소 visual overlay), **시간 오정렬**(visual feedback delay), **행동 궤적 이탈**(smooth action-chunk drift), **관측 replay**(frozen-observation replay). 기존 런타임 방어는 각각 특정 교란 계열만 다루거나(CBF 레이어는 기하 위험, BYOVLA는 시각 distractor) 고정된 보수적 규칙을 적용한다. 그런데 실제 배포에서는 같은 태스크 안에서 세 종류가 동시에 나타날 수 있고, **일률적으로 보수적인 대응은 정상 행동까지 억제**하는 반면 **모든 이상을 복구 가능하다고 보면 지각적 근거가 사라진 뒤에도 제어를 연장**하게 된다. 즉 런타임 안전장치는 "닫힌 루프가 불일치인가"뿐 아니라 **"정당화된 복구가 아직 가능한가"** 까지 판정해야 한다.

## 2. 핵심 기여 (Key Contributions)

1. **통합 형식화**: 공간 손상 / 시간 오정렬 / 행동 이탈 / 관측 replay를 각각의 공격 계열이 아니라 **시공간 visual-action 일관성(spatiotemporal visual-action consistency)의 위반**이라는 하나의 관측 가능한 구조로 묶는다. 보상 신호, 시뮬레이터 객체 상태, segmentation 주석, 교란 종류 사전지식 없이 판정 가능하다는 점이 핵심 제약.
2. **plug-and-play 안전장치 루프**: 행동 조건부 foveation + 교란 조건부 관측 복구 + action chunk 검증 + 복구가능성 인지형 safe failure를 하나의 파이프라인으로 결합. 기저 VLA는 끝까지 frozen.
3. **통제된 closed-loop 평가**: pi0 + LIBERO 40 태스크(시나리오 셀당 2,000 에피소드)에서 세 축 복구, clean 성능 보존, frozen replay 하 전량 timely safe failure를 보이고 training-free 런타임 baseline 비교 및 구성요소 ablation으로 뒷받침.

## 3. 방법론 (Methodology)

**위협 모델**: frozen 정책 π는 관측 o_t(RGB 다중 뷰 + EE/관절 상태 + 관측 타임스탬프)와 지시 l로부터 chunk A_t = [a_{t,0}, ..., a_{t,H-1}]를 내놓고, 그중 길이 h_t ≤ H의 prefix만 커밋된다. ActFovea는 오직 관측-행동 인터페이스를 통해 개입하며, 실행 chunk를 **실행 전에** 관찰하고 시각/상태/행동의 짧은 history를 유지한다. 이 인터페이스 **이후**에 주입되는 교란은 위협 모델 밖.

**(1) 행동 조건부 Foveation**: 각 카메라 v에 대해 투영된 접촉 중심 c_t^v와 짧은 모션 corridor Γ_t^v를 추정한다. 관절 캘리브레이션이 있으면 현재/행동 외삽 configuration을 forward kinematics로 그리퍼 pinch center에 매핑해 카메라 모델로 투영하고, 엄밀한 투영이 불가하면 등속 Cartesian 외삽이나 직전 추적 중심 + 관절 변화·속도·직전 행동에서 유도한 이미지 평면 방향으로 대체한다. 보존 마스크는 반지름 r_c 디스크와 반지름 r_Γ corridor의 합집합을 안전 여유 r_m만큼 dilate한 것:

> M_t^v = Dilate(M_{c,t}^v ∨ M_{Γ,t}^v, r_m)

그 여집합이 편집 가능한 배경이며, EMA 평활·임계화로 얻은 B̄_t^v에 대해 필터 이미지는 Ĩ_t^v = (1 − α B̄_t^v) ⊙ I_t^v + α B̄_t^v ⊙ E(I_t^v) (E는 배경 정규화·Gaussian 평활·컬러-그레이 혼합). **핵심은 fovea가 이미지 중심이 아니라 예측된 상호작용을 따라 이동한다**는 점이다.

**(2) 일관성 모니터링**: 동일한 kinematic 기준이 감시에도 쓰인다. 템플릿 매칭·모션 centroid·외형 centroid로 관측 중심 ĉ_t^v를 추정해 투영 접촉 중심과의 거리로 기하 일관성을, corridor 지지로 태스크 일관성을, 예측/관측 이미지 변위의 방향·크기 일치로 동적 일관성을 잰다. 시간 증거는 타임스탬프 건강도, "있어야 하는데 없는" 국소 모션, 짧은 history 매칭으로 추정한 lag, 전역 replay 유사도를 결합한다. 카메라 가중 평균으로 r_t를 얻고 EMA r̄_t와 함께

> R_t = clip(β r̄_t + (1−β) r_t + p_t^cam + p_t^lag + p_t^cal, 0, 1)

로 최종 risk를 만든다. 결정적(deterministic) router가 시간적 지속성과 결합해 교란 유형을 추론한다: 직접적 lag → temporal delay, 동적·고유수용감각 불일치 → action drift, 강한 stale/replay 증거 + hold latch 활성 → **복구 불가능한 freeze 상태**. 증거가 약하거나 상충하면 consistent/unknown으로 라우팅.

**(3) 일관성 유도 복구 + 행동 검증**: 후보 bank는 raw 관측과 foveated 관측을 항상 유지하고, 유한 시각 lag에 대해서만 조건부로 **시간 안정화 후보**(observed-to-predicted 중심 bias로 이미지를 kinematic 기준 쪽으로 평행이동, lag 추정치가 클수록 gain 증가, wrist 카메라는 감쇠)를 추가한다. 확정된 국소 overlay에 대해서는 공간 복원 연산자가 작동한다 — 검출 박스 **바깥**의 median dense optical flow로 마지막 clean 참조를 현재 이미지에 정렬하고, overlay 영역 P_t^v, 패턴 추정 Q̂_t^v, blending 추정 α̂_t^v로부터

> X̂_t^v = clip((P_t^v − α̂_t^v Q̂_t^v) / (1 − α̂_t^v), 0, 255)

를 재구성한 뒤 Gaussian feathering으로 경계를 섞는다. **모든 검출 뷰가 재구성되어야만** repair가 인정되고, 최약 뷰 품질(검출 신뢰도·clipping 비율·경계 연속성)이 admission 임계를 넘어야 한다.

각 후보가 유도한 chunk는 그리퍼 차원을 제외하고 첫 행동 방향, 종점 방향, 모션 크기, 평활도, horizon, chunk drift(횡방향/전체 변위 비, 종점 방향 불일치, 경로 곡률)로 채점된다: V_k = clip(w^T u_k + b_k, 0, 1), w는 고정 비음수 가중치, b_k는 위협 조건부 보너스. 선택은 V_k·실행 모드·후보 우선순위를 결합한 효용을 최대화한다. **raw 행동이 언제나 보수적 fallback**이며, deblended 관측의 첫 3개 모션 행동이 raw와 방향·RMS에서 일치하면 raw를 유지한다.

**(4) 위험 적응 실행과 Safe Failure**: 2단계 중재. 모니터가 λ_t^mon, h_t^mon을(보존/댐핑/horizon 단축/hold), 검증기가 λ_t^ver, h_t^ver을(정상/단축 horizon/servo recovery/safe failure) 정하고

> â_{t,i}^mot = λ_t^mon λ_t^ver a_{t,i}^{*,mot} (i < h_t), h_t = min(h_t^mon, h_t^ver)

모니터의 hold와 검증기의 safe failure는 나머지 모드를 override하고 모션을 0으로 만든다. 유한 delay는 정렬 가능한 순서 history를 남기지만 **지속 replay는 그렇지 않으므로**, 이미지 평면 정렬 후에도 stale 내용이 지속되면 hold를 latch하고(즉각적 강한 replay 증거 또는 설정된 stale streak 후) 충분한 fresh 증거가 돌아오면 해제한다. hold 발동 시 clipped reverse action 하나를 앞에 붙인 뒤 나머지를 hold로 채울 수 있다. 저자들은 여기서의 safe failure가 **보수적 모션 억제**를 뜻하며 형식적 충돌 회피 보장은 범위 밖임을 명시한다.

## 4. 아키텍처 상세 (Architecture Details)

- **적용 대상**: frozen VLA 체크포인트. 주 평가는 **pi0**(PaliGemma 기반 VLM + flow-matching action expert)의 LIBERO 체크포인트.
- **추가 학습 파라미터**: **0**. 학습 목적함수도 파라미터 최적화도 없음. β, 구성요소 가중치, 검증기 가중치 w, admission 임계는 모두 고정 구현 상수.
- **입력**: 다중 뷰 RGB, EE/관절 상태, 관측 타임스탬프, 직전 action chunk. 교란 라벨과 생성기 파라미터는 제공되지 않음.
- **연산 특성**: clean 프레임은 감시 비용만 지불. 후보 확장(추가 정책 호출)은 clean-like·저위험 ambiguous·복구 불가 증거에서는 억제된다.

## 5. 실험 설정 (Experimental Setup)

- **벤치마크**: LIBERO 4개 10-태스크 suite(Spatial, Object, Goal, LIBERO-10) = 40 태스크. 태스크당 50 에피소드 → 방법·시나리오 셀당 **2,000 에피소드**. 모든 비교가 동일 체크포인트·동일 태스크/실행 설정을 사용하고 ActFovea는 추론 시점에만 적용.
- **교란**: (a) Smooth Action-Chunk Drift — 태스크 계열별 phase template으로 chunk의 모션 부분을 phase 조건부 window에서 섭동, (b) Multi-View Visual Feedback Delay — 외부/wrist 뷰를 현재 proprioception보다 3프레임 뒤로 고정, (c) Persistent Localized Visual Overlay — 짧은 clean warm-up 후 고정 checker 패턴을 양 카메라에 alpha-blend, (d) Frozen-Observation Replay — 트리거 프레임을 종료까지 양 뷰에 재사용.
- **지표**: 성공률 SR, 절대 방어 이득 Gain = S_{D+AF} − S_D, 정규화 회복률 **NRR = (S_{D+AF} − S_D) / (S_C − S_D) × 100%**. Frozen replay는 Task Success / Timely Safe Failure / Unprotected Failure 3분류(Timely = 탐지·행동 예산 내 terminal safe-failure 도달 + 모든 safe-failure 행동이 executor bound 내).

## 6. 주요 결과 (Main Results)

**Table 1 — 4개 suite 평균 복구 결과 (%, Gain은 pt)**:

| 시나리오 | Undisturbed Base | Undisturbed ActFovea | Disturbed Base | Disturbed ActFovea | Gain↑ | NRR↑ |
|---|---|---|---|---|---|---|
| Action Drift | 92.7 | 93.0 | 83.1 | **90.1** | +7.0 | 73.1 |
| Visual Delay | 92.6 | 93.2 | 76.2 | **86.0** | +9.8 | 59.8 |
| Visual Overlay | 93.0 | 93.8 | 49.3 | **90.3** | **+41.0** | **93.7** |

지속적 visual overlay가 가장 큰 통제된 손실을 만든다(93.0% → 49.3%). ActFovea는 90.3%로 되돌려 유도된 손실의 **93.7%** 를 회복한다. 세 시나리오 모두에서 **무교란 성능이 보존**된다는 점(93.0/93.2/93.8 ≥ base)이 중요하다 — 보수적 억제로 얻은 이득이 아니라는 증거다.

**Table 2 — training-free 런타임 방법 비교 (SR %)**:

| Method | Undisturbed | Action Drift | Visual Delay | Visual Overlay |
|---|---|---|---|---|
| Base VLA | 93.0 | 83.1 | 76.2 | 49.3 |
| Action Clip/Smoothing | 82.2 | 70.4 | 70.2 | 30.9 |
| Fixed Short Horizon | 91.7 | **89.9** | 70.7 | 32.4 |
| Timestamp-Only Hold | 93.1 | 84.9 | **0.0** | 48.5 |
| **ActFovea** | **93.8** | **90.1** | **86.0** | **90.3** |

Fixed Short Horizon은 action drift에서 89.9%로 ActFovea(90.1%)에 근접 — 잦은 replanning이 이 교란에는 잘 맞는다는 확인이다. 그러나 **전이되지 않는다**(delay 70.7%, overlay 32.4%). Action Clip/Smoothing은 무교란 성능 자체를 82.2%로 떨어뜨린다. Timestamp-Only Hold는 무교란을 보존하지만 **연속적 visual delay가 타임스탬프 규칙을 계속 활성화시켜 성공률 0%** 라는 극적인 실패를 보인다. ActFovea만이 무교란 성능을 지키면서 세 복구 가능 교란 모두를 개선한다.

**Table 3 — Frozen-Observation Replay 결과 (%)**:

| Method | Task Success | Timely Safe Failure | Unprotected Failure |
|---|---|---|---|
| Base VLA | 3.05 | 0.00 | 96.95 |
| Timestamp-Only Hold | 0.00 | 0.00 | 100.00 |
| w/o Hold/Safe-Fail | 0.65 | 0.00 | 99.35 |
| **ActFovea** | 0.00 | **100.00** | **0.00** |

2,000 replay 에피소드 **전량**이 timely safe failure에 도달, unprotected failure 0. 탐지 직후 실행되는 경계 행동은 평균 **2.0 스텝**, 누적 action-space 모션 norm **0.326**, action-bound 위반 0. Hold/Safe-Fail 없이는 탐지 후 259.2 행동이 이어지고 모션 norm이 241.98까지 누적된다 → **탐지 후 행동 수 99.23%, 누적 모션 99.87% 감소**. Timestamp-Only Hold는 replay 쿼리의 96.55%를 hold로 보내면서도 그것을 terminal safe failure로 전환하지 못한다. 저자들의 지적대로 **"지속적 zero motion ≠ 경계된 실패 프로토콜"**.

## 7. Ablation 및 메커니즘 분석 (Ablations & Mechanism)

**Table 4 — 구성요소 ablation (SR % / Gain pt, Gain은 Table 1의 disturbed Base 기준)**:

| Variant | Action Drift | Visual Delay | Visual Overlay |
|---|---|---|---|
| w/o Threat Typing | 87.5 / +4.4 | 80.5 / +4.3 | 41.7 / **−7.6** |
| w/o Recovery Bank | 87.5 / +4.4 | 84.0 / +7.8 | 16.0 / **−33.3** |
| w/o Candidate Expansion | 90.6 / +7.5 | 85.4 / +9.2 | 17.6 / **−31.7** |
| w/o Action Verification | 81.9 / **−1.2** | 78.5 / +2.3 | 92.1 / +42.8 |
| **Full ActFovea** | 90.1 / +7.0 | 86.0 / +9.8 | 90.3 / +41.0 |

**깔끔한 역할 분담**이 드러난다. Action verification을 빼면 action drift(−1.2)와 visual delay(+2.3) 이득이 붕괴한다 — 궤적 검증이 시간·행동 측 후보를 신뢰 가능한 복구로 바꾸는 주 메커니즘이다. 반대로 visual overlay에서는 verification 없이도 +42.8을 유지하지만, threat typing·recovery bank·candidate expansion을 제거하면 각각 −7.6 / −33.3 / −31.7로 무너진다 — **공간 복구는 손상 영역을 찾아 쓸 만한 관측 대안을 만드는 데 의존**하고, verification은 공유된 보수적 승인 게이트 역할을 한다. 하나의 파이프라인이 모든 개입을 균일하게 적용하지 않고도 세 축을 지원하는 이유가 여기 있다.

**공간 메커니즘 (Figure 4)**: approach/grasp/place 단계에서 dynamic fovea의 보존 비율은 13.5% → 11.1% → 12.3%로 상호작용에 맞춰 변한다. contact-only는 9.0% 고정(모션 corridor 누락), static ROI는 27.3% 고정(단계 적응 불가). **Figure 3**의 런타임 결정 trace는 유한 delay가 간헐적 freshness와 경계된 lag 추정을 유지해 적응적 replanning을 허용하는 반면, 지속 replay는 fresh 증거를 소진시켜 recovery → unrecoverable로 상태를 몰고 실행 horizon을 줄인 뒤 safe failure로 종결됨을 보여준다.

## 8. VLA-Tracker 등재 판정 근거 (Inclusion Rationale)

순수 탐지·모니터링 논문(GUARD/SAFECAST 계열)이라면 등재 대상이 아니다. ActFovea는 그 경계 반대편에 있다: (i) 관측을 실제로 **재구성·교체**하고(foveation, 시간 안정화 평행이동, overlay deblending), (ii) 정책이 낸 chunk를 **검증·선택·스케일링·단축**하며(λ^mon λ^ver, h_t), (iii) 그 결과가 **LIBERO 성공률 수치로 개선**된다(49.3 → 90.3, +9.8, +7.0). 즉 정책 행동을 바꾸는 안전 개입 wrapper이며, TrustVLA(추론 시점 백도어 방어 wrapper)·PaCo-VLA(passivity shield wrapper) 선례와 동일한 범주다.

## 9. 강점 (Strengths)

- **하나의 원리, 네 가지 교란**: 교란별 detector를 쌓는 대신 시공간 visual-action 일관성이라는 단일 관측 구조로 통합한 형식화가 설득력 있고, ablation이 그 통합이 공짜가 아님(각 구성요소가 서로 다른 축을 담당)을 정량적으로 보인다.
- **복구 가능성 판정이라는 차별점**: "고칠 수 있는가"까지 판정해 유한 delay는 복구, 지속 replay는 safe failure로 분기시킨 설계는 기존 런타임 안전장치가 다루지 않던 축이다. Timestamp-Only Hold와의 대비(96.55% hold vs 100% timely safe failure)가 이 구분의 실체를 잘 보여준다.
- **clean 성능 보존**: 안전장치 논문에서 가장 흔한 함정(보수성으로 인한 정상 성능 희생)을 정면으로 통과했다. Action Clip/Smoothing의 82.2%와 대조된다.
- **평가 규모와 통제**: 셀당 2,000 에피소드, 동일 체크포인트·동일 실행 설정, 교란 라벨 비공개. NRR이라는 정규화 지표로 교란 강도가 다른 시나리오를 공정 비교.
- **정직한 범위 명시**: safe failure가 보수적 모션 억제일 뿐 형식적 충돌 회피 보장이 아니라는 점, 인터페이스 이후 교란은 위협 모델 밖이라는 점을 명시.

## 10. 약점 및 한계 (Weaknesses & Limitations)

- **단일 정책·단일 벤치마크**: pi0 + LIBERO(시뮬레이션)만. OpenVLA 등 다른 action head 계열이나 실기 로봇으로의 전이가 검증되지 않았다. wrapper를 표방하는 만큼 cross-policy 결과가 아쉽다.
- **suite별 분해 부재**: 모든 표가 4개 suite 평균이다. LIBERO-10(long-horizon)처럼 chunk 의존이 큰 suite에서 복구가 더 어려운지 알 수 없다.
- **고정 상수의 캘리브레이션 부담**: β, 구성요소 가중치, 검증기 가중치 w, admission 임계, r_c/r_Γ/r_m, α 등이 모두 "고정 구현 상수"로만 기술되고 민감도 분석이 없다. 새 정책·새 카메라 구성에서 재튜닝 비용이 불투명하다.
- **런타임 오버헤드 미정량**: 후보 확장이 추가 정책 호출을 유발하는데 실제 지연/처리량 수치(Hz, 지연 ms)가 표로 제시되지 않는다. "runtime comparisons"라는 표현은 있으나 비용 측면 수치가 약하다.
- **교란이 비적응적(non-adaptive)**: 네 교란 모두 ActFovea의 존재를 모르는 고정 생성기다. 안전장치의 fovea 마스크나 router를 아는 적응형 공격자에 대한 평가가 없다.
- **overlay 복구의 강한 전제**: 마지막 clean 참조 버퍼, 박스 바깥 dense optical flow 정렬, 모든 검출 뷰의 재구성 성공을 요구한다. warm-up 없이 시작부터 오염되거나 overlay가 넓게 분포하면 성립하지 않는다.
- **frozen replay에서 Task Success 0%**: 설계상 의도된 trade-off지만, base VLA가 우연히 성공하던 3.05%도 함께 포기된다. 안전-성능 교환의 방향이 태스크마다 옳은지는 별도 논의가 필요하다.

## 11. 의의 및 향후 방향 (Significance & Future Work)

VLA 보안 문헌은 지금까지 공격 구성·벤치마킹(AttackVLA, SilentDrift, FreezeVLA, BadVLA)에 크게 기울어 있었고, 방어는 학습 시점(SafeVLA, Phantom Menace)이나 단일 위험 유형(VLSA의 CBF, BYOVLA의 distractor 편집)에 국한됐다. ActFovea는 **frozen 정책 위의 배포 시점 통합 방어**라는 비어 있던 칸을 채운다. 특히 "복구 가능성"을 1급 런타임 상태로 승격시킨 것은 안전 개입 설계에 재사용 가능한 개념이다.

향후 과제로는 (i) OpenVLA/pi0.5 등 다중 백본 전이, (ii) 실기 로봇에서의 지연 예산 하 검증, (iii) 적응형 공격자 대비, (iv) 고정 상수의 자동 캘리브레이션, (v) safe failure를 형식적 충돌 회피 보장(CBF/reachability)과 결합하는 하이브리드가 자연스럽다. 특히 (v)는 저자들이 스스로 범위 밖이라 밝힌 지점이라 후속 연구 여지가 크다.

## 12. 총평 (Overall Assessment)

ActFovea는 "런타임 안전장치"라는 이름 아래 흔히 나오는 수동적 모니터가 아니라, **관측을 재구성하고 action chunk를 검증·조절해 실제 성공률을 끌어올리는 능동적 개입 wrapper**다. visual overlay에서 49.3% → 90.3%(NRR 93.7%)라는 수치와 무교란 성능 보존을 동시에 달성한 점, 그리고 ablation이 "공간 복구는 관측 대안 구성이, 시간·행동 복구는 chunk 검증이 담당"이라는 명확한 역할 분담을 드러낸 점이 이 논문의 실질이다. Timestamp-Only Hold가 visual delay에서 0%로 무너지는 대조 실험은 단순 보수 규칙의 한계를 인상적으로 보여준다. 반면 단일 정책·단일 시뮬레이션 벤치마크, suite별 분해 부재, 런타임 비용 미정량, 비적응적 교란은 분명한 한계이며, 특히 wrapper를 주장하는 논문에서 cross-policy 증거의 부재는 다음 버전에서 반드시 메워야 할 부분이다. 그럼에도 VLA 배포 안전 파이프라인의 실용적 구성요소로서 가치가 뚜렷하다.

<!-- VERIFIED: pdf -->
