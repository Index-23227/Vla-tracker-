# TTE-Handover: Temporal Tactile Encoding and Compliance for Intent-Aware Robot-to-Human Bimanual Handover

> **한 줄 요약**: 휴머노이드 ergoCub이 양손으로 든 상자를 사람에게 건넬 때 "진짜로 가져가려는 의도"를 판단하기 위해, 약 0.9초 분량의 손끝 촉각 힘 이력을 autoencoder로 압축한 **Temporal Tactile Encoding(TTE)**을 GR00T-N1.5-3B에 state로 넣어 post-train하고, 저수준 **compliance 제어**와 결합. 10명 × 10 trial 사용자 실험에서 성공률 93%(촉각 없음 54%, compliance 없음 82%).

- **arXiv**: 2609.05282v1 (2026-09-04, cs.RO)
- **소속**: Istituto Italiano di Tecnologia (Humanoid Sensing and Perception), Università di Genova
- **정책 백본**: NVIDIA Isaac GR00T-N1.5-3B
- **코드**: 게재 승인 후 공개 예정
- (모델명 "TTE-Handover"는 트래커용 표기이며 논문은 별도 이름을 쓰지 않음; 논문 내 명칭은 P1 = compliance+TTE)

---

## 1. 배경 및 동기

로봇→사람 인계는 너무 이르면 떨어뜨리고, 너무 늦으면 사람이 로봇과 줄다리기를 하게 된다. 시각만으로는 의도적 grasp-and-pull과 우발적 접촉, 약한 파지, 반대 방향 힘, 일시적 상호작용을 구분하기 어렵다. 기존 촉각 인계 시스템은 규칙 기반 release 로직이나 전용 제어기에 의존했고, visuo-tactile 학습은 주로 로봇-물체 조작에 집중되었다. 또 compliance는 필요하지만, 로봇이 양보해서 생긴 움직임이 release와 상관되어 학습 정책을 혼란시킬 수 있다.

## 2. 문제 설정 및 플랫폼

- ergoCub 휴머노이드, 양손 상자 파지 후 인계, 프롬프트 "Get the box and pass it to human".
- 관측: 자기중심 RGB, 양손 Xela 손끝 촉각(엄지 제외 4손가락 × 7 자기 taxel), proprioception.
- 행동: 양손 Cartesian 자세 + 머리 방향 16-step chunk(10Hz, 1.6초), 앞 8개 실행.

## 3. 데이터 수집

Meta Quest 3 텔레오퍼레이션, LeRobot 10Hz 기록, Xela는 약 70Hz 별도 기록 후 정렬. 시연자 8명 × 50 데모(compliance 없음 25, 있음 25), 조건당 약 50k 프레임. 다섯 세트로 구성: 표준 인계, 접촉 없이 손 접근 후 당김, 잘못된 방향 힘 후 진짜 당김, 불연속 접촉·임의 방향 힘 후 당김, 무작위 조합. 참가자가 진짜 당김 시작을 말하면 조작자가 약 1초 뒤 release.

## 4. 방법 — 힘 추정과 TTE

- **힘 추정**: Omega.3 로봇에 구형 indenter를 달아 손가락별 자극 데이터를 모으고 손가락별 회귀 모델로 3축 힘 추정 → 24차원(4손가락×2손×3축).
- **TTE**: (64, 2, 12) 창(약 0.9초)을 flatten 후 FC encoder로 24차원 embedding. 정책과 별도 학습 후 **고정**.
- 손실: L_TTE = 0.25·L_rec + 2·L_dir + 1·L_Δdir. L_dir은 재구성 힘 벡터와 목표 간 cosine 거리, L_Δdir은 d=8 step 힘 변화의 방향 cosine 거리(0.2N 초과 벡터만).

## 5. 방법 — Compliance 제어와 정책

- **Compliance**: K_p(P_c − P) + K_d(Ṗ_c − Ṗ) = Σ F_i (K_p=50, K_d=40)로 외력 방향으로 손 기준을 양보시키고, 2차 IK 몸통·팔 제어기가 관절 기준을 계산. 머리는 몸통 움직임을 보상.
- **정책**: GR00T-N1.5-3B를 공개 체크포인트에서 post-train, RGB + (proprio/촉각) state 입력. 모든 ablation이 같은 backbone·레시피·행동 공간 공유.

## 6. 실험 설정

- **Stage 1 (오프라인)**: A1(순간 힘) vs A2(TTE), 동일 데이터·100k step. 흔들기·두드리기·반대 방향 밀기가 있는 별도 Challenging Test Set(10 에피소드, 3,776 프레임)에서 예측 손 간 거리(0.35m 임계)로 open-loop 평가.
- **Stage 2 (사용자 실험)**: P1(compliance+TTE), P2(compliance only), P3(TTE only; 별도 no-compliance 데이터로 학습, 프레임 수 매칭). 수집에 참여하지 않은 10명(여5·남5), 정책 순서 counterbalance, 정책당 R(release 기대) 5 + H(hold 기대, 30초 방해 구간) 5 trial.
- 지표: trial 성공률, T_r = μ+2σ ≈ 2.8초 이내 release 성공률, release 지연 중앙값, 최대 당김 힘, 10문항 Likert 설문, 선호 순위.

## 7. 주요 결과

**Stage 1**: 10 에피소드에서 임계 통과 횟수 A1 51회 vs A2 3회, hold 중 1.5초 이상 지속 오개방 A1 4회 vs A2 0회.

**Stage 2 (Table I)**

| 지표 | P1 comp+TTE | P2 comp | P3 TTE |
|---|---|---|---|
| 성공률 (%) | **93** | 54 | 82 |
| 성공률, T_r=2.8s (%) | **92** | 45 | 80 |
| Release 지연 중앙값 (s) | **1.70** | 2.35 | 2.00 |
| 최대 당김 힘 (N) | **4.07** | 4.67 | 4.74 |
| 만족도 (1–7) | **6.2** | 3.3 | 4.7 |
| 전체 선호 (/10) | **9** | 1 | 0 |
| 가장 안전 (/10) | **10** | 0 | 0 |

## 8. 분석

- P2(촉각 없음)는 hold 기대 trial에서 크게 실패: T4(당기지 않고 잡기) 10%, T7(일시적 당김) 30%. compliance로 생긴 로봇 자체 움직임이나 사람의 부수적 움직임을 인수 의도로 오인한다.
- P3(compliance 없음)는 판단은 비교적 정확하지만 힘이 더 들고 느리며, 사람이 손이 다 열리기 전에 상자를 빼내 **P3에서만 4% 낙하** 발생.
- P1은 10개 trial 유형 중 7개에서 100%. T9(충격적 짧은 당김)가 가장 어려움(P1 60%).
- Figure 4: 짧은 당김과 최종 당김의 최대 힘이 같아도(3.8N) 지속 시간(약 0.2s vs 1.5s)으로 구분 → 힘 크기 임계값이 아닌 지속 증거 기반 판단.

## 9. 강점

- 인계 의도 판단을 "시간적으로 누적된 촉각 증거" 문제로 정확히 규정하고 TTE로 간결하게 해결.
- 촉각(판단)과 compliance(물리적 상호작용)의 **상보적 역할**을 깔끔한 2×1 ablation으로 분리.
- 수집 참여자와 분리된 10명 사용자 실험, counterbalance, 객관+주관 지표 병행.
- 저수준 반사(compliance) + 고수준 학습 정책이라는 계층 해석이 설득력 있다.

## 10. 약점 및 한계

- 단일 과제(상자 인계), 단일 물체, 단일 로봇. 일반화는 미래 과제로 남김.
- 정책당 100 trial, 참여자 10명으로 통계 검정 부재.
- Stage 1은 "from scratch"로 학습했다고 기술되어 Stage 2의 post-train 설정과 관계가 모호.
- TTE autoencoder가 고정되어 정책과 end-to-end 최적화되지 않음; 창 길이 등 하이퍼파라미터 ablation 없음.
- 표준 벤치마크 결과 없음; 코드·데이터 미공개(승인 후 예정).

## 11. 재현 및 확장 아이디어

- TTE를 end-to-end로 미세조정하거나 창 길이/embedding 크기 sweep.
- 다양한 물체·파지·인계 구성, 사용자별 적응 release 전략(저자 제안).
- 사람→로봇 인계, 공동 운반 등 다른 물리적 HRI로 확장.
- compliance 제어를 명시적 reflex layer로 두고 GR00T System-1/2 구조와 결합.

## 12. 총평

VLA를 사람과의 물리적 상호작용에 쓰려면 시각만으로는 부족하고, 짧은 촉각 이력과 저수준 compliance가 함께 있어야 함을 깔끔한 사용자 실험으로 보여준 응용 논문. 규모는 작지만 결론은 명확하다.

**한 문장 요약**: 얼마나 세게가 아니라 얼마나 오래 당기는지를 읽게 하자, 로봇이 제때 손을 놓는다.

<!-- VERIFIED: pdf -->
