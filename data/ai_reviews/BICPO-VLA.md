# BICPO-VLA: Behavior-Identified Continuation Preference Optimization for Smooth Asynchronous Vision-Language-Action Control

> **한 줄 요약**: 비동기 action chunking에서 "새 chunk를 요청한 시점과 실제로 제어를 넘겨받는 시점의 불일치"를 (1) 지시 기반 행동 식별, (2) Haar 스캐폴드/잔차 2단계 flow 생성, (3) 알려진 출력 명령을 handoff 상태까지 굴린 뒤 같은 행동 후보끼리 Flow-DPO로 선호 학습하는 BICPO로 푼 VLA. CALVIN ABC→D Avg.Len 4.52, RoboTwin 2.0 Hard 10과제 65.8%, LIBERO 99.1%, 실물 6과제 69.3%.

- **arXiv**: 2608.13924v1 (2026-08-14, cs.RO)
- **소속**: Beihang, BIT, CASIA, BUPT, Hunan Univ., PKU, Tsinghua
- **코드**: 논문에 명시 없음
- **백본**: VLM + Mamba 기반 행동 인코더/메모리(BehaviorVLA 계열) + Haar 구조 flow-matching action expert

---

## 1. 배경 및 동기

비동기 실행에서는 로봇이 이전 chunk를 계속 실행하는 동안 다음 chunk를 추론한다. 새 chunk는 한 운동 맥락에서 요청되지만 다른 맥락에서 제어를 넘겨받으므로, 의미적으로 맞는 예측도 경계 점프나 운동 추세 단절을 일으킨다. 저자들은 하나의 "행동"이 여러 궤적 실현(**behavior-conditioned action fiber**)을 갖는다는 관찰에서 출발해, 행동 정체성·실현 구조·handoff 선택을 분리해야 한다고 주장한다. 일반적인 smoothness 손실은 움직임 자체를 줄여 과제 진행과 충돌한다.

## 2. 문제 정의

- 요청 시점 r에서 관측 o_r, 지시 ℓ, 이력 H_r → 행동 상태 z_r, 메모리 m_r, prior μ_r. 공유 조건 b_r = (ℓ, z_r, m_r, μ_r)가 fiber F_r을 정의.
- 추론에 k 제어 주기가 걸리는 동안 실행되는 명령 U_r^k = (u_r, …, u_{r+k−1})은 이미 알려져 있다.
- 목표: 행동 의미는 유지하면서, 실제 handoff 상태와 호환되는 실현 Â_{r+k}를 온라인 후보 탐색 없이 생성.

## 3. 방법

- **지시 기반 행동 식별**: 시각 토큰이 먼저 언어에 cross-attend(Ṽ_t = V_t + Attn(LN(V_t), L_b, L_b))한 뒤 인과 Mamba 이력 인코더가 z_r을 만들고, 장기 메모리 m_r과 결합해 prior μ_r을 디코딩. "언어를 지각 단계에 넣는" 순서가 핵심.
- **Rolled handoff context**: 동결된 action-history 모델에 알려진 출력 명령을 적용해 h_{r+k}까지 굴리고, 절대 상태와 변위 [LN(h̄); LN(Δh)]를 사영해 g_k를 얻는다.
- **Haar 실현 좌표**: 한 단계 직교 Haar 변환으로 chunk를 스캐폴드 C(쌍 평균)와 잔차 D(쌍 차이)로 나누고 역변환으로 정확히 복원. 공유 expert가 stage embedding으로 C를 먼저, sg(Ĉ)에 조건해 D를 생성(각 1 step backward Euler).
- **BICPO**: 같은 조건 χ에서 서로 다른 노이즈로 두 후보를 샘플 → 경계 점프 J_jump = ‖a_0 − u_{r+k−1}‖²_W와 추세 불일치 J_trend를 합한 J_cont로 선호 결정(margin 이상일 때만). Haar 2단계 에너지를 쓰는 reference-relative Flow-DPO: Δθ = [E_θ(A−) − E_θ(A+)] − [E_ref(A−) − E_ref(A+)], L = −q log σ(βΔθ) + λ₊E_θ(A+) + λ_rep L_replay.
- 의미 경로·rollout·Haar·reference expert는 동결하고 P_ψ, W_g, 일부 expert 파라미터만 업데이트.

## 4. 학습과 배포

학습은 모방 학습 → L_struct로 continuation warm-up → 오프라인 DPO 순서. 호스트 정책(π0.5, Legato, RTC)에 적용할 때는 DPO 목적만 사용하고 BICPO-VLA의 인코더·Haar 생성기·사영기는 붙이지 않는다. 배포 시 handoff 상태 하나를 굴리고 (Ĉ, D̂)를 생성해 복원할 뿐, 온라인 선호 점수화는 없다.

## 5. 설계상 주목할 점

- Haar는 손실 있는 거친 표현이 아니라 **정확히 가역인 좌표계**이므로 두 단계 모두 handoff 조건에 반응할 수 있다.
- BICPO는 온라인 RL이 아니라 "고정된 행동의 실현들 사이의 오프라인 쌍 비교"이며, 무행동이 이기지 않도록 reference 호환성이 크게 다른 쌍을 제거.

## 6. 실험 설정

- CALVIN ABC→D (11개 VLA 기준선), RoboTwin 2.0 Hard 10개 과제(RDT, ACT, DP, DP3, π0, π0.5, B-VLA 비교), LIBERO(handoff 비용 중심), 실물 6과제(OpenVLA-OFT, π0.5, BehaviorVLA 비교).
- 지표: 성공률, CALVIN 평균 체인 길이, J_jump·J_trend.

## 7. 주요 결과

**Table 1 – CALVIN ABC→D**

| Method | 1 | 2 | 3 | 4 | 5 | Avg.Len |
|---|---|---|---|---|---|---|
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| B-VLA | 96.0 | 92.0 | 87.3 | 82.9 | 77.3 | 4.36 |
| **BICPO-VLA** | **98.9** | **95.4** | **91.0** | **86.0** | **80.7** | **4.52** |

**Table 2 – RoboTwin 2.0 Hard 10과제**: Group I 평균 80.6, 전체 65.8 (B-VLA 60.4, π0.5 51.4). 10개 과제 모두 4–8pt 향상.

**Table 3 – LIBERO 성공률 / handoff 비용**

| Method | SR | J_jump | J_trend |
|---|---|---|---|
| BICPO-VLA | 99.1 | 2.37 | 6.40 |
| w/o DPO | 98.8 | 3.01 | 8.07 |
| π0.5-FM | 96.9 | 4.22 | 8.50 |
| π0.5-FM + chosen-only SFT | 49.1 | 3.13 | 6.90 |
| π0.5-FM + continuity DPO | 97.1 | 2.52 | 6.70 |

DPO로 BICPO-VLA의 jump −21.3%, trend −20.7%; 직접 SFT는 모든 호스트에서 SR 하락(π0.5는 96.9 → 49.1).

**Table 4 – 지연 강건성**: k=3/4/5/random에서 SR 99.1/98.9/98.8/98.8.

**Table 5/6 – 절제 (CALVIN Avg.Len)**: 전체 4.520, Haar 제거 4.490, BICPO 제거 4.480, 지시 grounding 제거 4.396; 지시 주입 위치는 visual query 4.520 > query fusion 4.476 > memory fusion 4.417.

**실물**: 6과제 평균 69.3% vs B-VLA 60.2, π0.5 47.3, OpenVLA-OFT 33.3.

## 8. Related Work 상의 위치

- RTC, REMAC, Legato, SEAM 등 비동기 실행 기법이 prefix 조건·guided sampling·경계 정규화를 쓰는 것과 달리, 출력 명령을 **handoff 상태로 굴려** 같은 행동 후보를 선호 순위화.
- FlowPRO 등 flow 기반 선호 최적화 위에, DPO 목적 자체가 다른 호스트로 이식 가능함을 보인다.
- BehaviorVLA/B-VLA의 행동 표현을 계승하고 CF-VLA류 coarse-to-fine 생성과 달리 정확히 가역인 Haar 좌표를 사용.

## 9. 강점

1. **문제 분해가 명확**: 행동 식별 / 실현 구조 / handoff 선택을 각기 다른 불변량에 대응시켜 무엇을 바꿔야 하는지 분명하다.
2. **이식성 실험**: DPO 목적을 π0.5, Legato, RTC에 붙여 이득이 BICPO-VLA 표현 전체가 아니라 선호 원리에서 온다는 것을 보여 줌.
3. **SFT 대조군**: chosen-only SFT가 SR을 크게 떨어뜨린다는 결과로 "상대적 선호 vs 무조건 smoothness 회귀" 주장을 뒷받침.
4. CALVIN, RoboTwin, LIBERO, 실물을 모두 다룸.

## 10. 약점 및 한계

1. **지표 해석**: LIBERO에서 DPO의 SR 기여는 +0.3pt로, 주 이득은 연속성 비용이다. 연속성 개선이 실제 하드웨어 품질(마모, 진동)에 어떻게 이어지는지는 측정되지 않았다.
2. RoboTwin은 Hard 설정 10개 과제만 사용해 공식 50과제 리더보드와 직접 비교가 어렵다.
3. 백본·파라미터 수·학습 데이터 양 등 구현 상세가 본문에 부족하고 보충 자료로 미룬다. 실물 결과는 그림 수치 위주.
4. 알려진 출력 명령과 "지연 < 남은 chunk" 가정에 의존 — 외란으로 실제 궤적이 명령과 달라지는 경우는 미해결(저자 명시).
5. CALVIN 절제 차이(4.520 vs 4.480 등)가 작고 시드 분산이 보고되지 않았다.

## 11. 재현 및 확장 아이디어

- 실제 측정 상태(proprio)로 handoff를 보정하는 폐루프 rollout으로 외란 상황 확장.
- 다단계(>1 level) Haar/웨이블릿 분해와 단계 수에 따른 속도-품질 트레이드오프.
- 연속성 선호를 온라인 RL(GRPO 등)의 보조 보상으로 쓰는 변형과 비교.
- 공식 RoboTwin 2.0 50과제 및 LIBERO suite별 결과 보고.

## 12. 총평

BICPO-VLA는 비동기 VLA 제어의 "handoff 부정합"을 선호 최적화 문제로 재정의하고, 행동 의미를 고정한 채 실현만 바꾸는 설계로 성공률 손실 없이 연속성을 개선했다. CALVIN 4.52 등 수치도 강하지만, 연속성 이득의 실질적 의미와 구현 세부의 투명성은 보강이 필요하다.

**한 문장 요약**: 알려진 출력 명령을 handoff 상태로 굴려 같은 행동 후보끼리 Flow-DPO를 적용한 비동기 VLA로, CALVIN ABC→D 4.52·RoboTwin 2.0 Hard 10과제 65.8%를 기록하고 DPO 목적이 π0.5·Legato·RTC로도 이식됨을 보였다.

<!-- VERIFIED: pdf -->
