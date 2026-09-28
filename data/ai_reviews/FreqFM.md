# FreqFM — Frequency-Conditioned Flow Matching for Vision-Language-Action Models

**arXiv**: 2609.10405 · **기관**: AGIBOT, Xi'an Jiaotong University · **날짜**: 2026-09-09

## 1. 한 줄 요약
액션 청크를 시간축 DCT 계수로 표현하고 훈련 행동의 주파수별 파워 스펙트럼으로 소스 분포·학습 가중치·CFG 잔차 예산을 모두 조건화하는 flow-matching 프레임워크로, π0.5 기준 LIBERO 97.9, LIBERO-Plus 75.8(+9.3), VLA-Arena 68.2(+4.2), AgiBot A2 실로봇 6과제 SR 67.8(π0.5 57.8)을 달성.

## 2. 문제 설정
- 로봇 행동은 시간 상관 궤적이며 주파수 성분별 에너지가 극도로 불균일하다: 실로봇(H=30) 데이터는 7.4 decade 범위, 에너지 99.9% 이상이 최저 3개 bin에 집중; LIBERO 3.8, VLA-Arena 2.6 decade.
- 기존 flow-matching 액션 전문가는 시간 좌표에서 생성하며 이 이질성을 명시적으로 활용하지 않는다.
- 질문: 행동 주파수를 암묵적 궤적 속성에서 flow matching 전 과정의 명시적 조건 차원으로 끌어올릴 수 있는가?

## 3. 핵심 아이디어
- **Spectrum-Matched Source**: z̃_0,k,d = √P_k,d · ε. 소스와 타깃이 같은 2차 원모멘트 스펙트럼을 공유해 선형 경로 전 구간에서 상대 스펙트럼이 보존((1−t)²+t²)P).
- **PSD-Normalized Adaptive Objective**: 주파수별 회귀 오차를 (P+ε)^ρ(ρ=0.25)로 정규화한 뒤 Kendall식 불확실성 가중(학습형 log-scale s_k,d)으로 재균형 → 저주파 지배 완화.
- **Spectral Transport-Budgeted CFG(STB-CFG)**: E[u²]=2P에서 얻은 기준 수송 스케일로 CFG 잔차를 주파수별 타원 공(반경 βR_k)에 방사 클리핑. 예산 내 잔차는 그대로, 초과분만 경계로 투영.

## 4. 아키텍처
- π0 및 π0.5(LeRobot 코드베이스). VLA 백본과 액션 전문가 구조는 변경 없음.
- 부드러운 행동 차원은 DCT 계수 공간에서 flow matching, ODE 적분 후 역 DCT로 복원.
- CFG를 위해 학습 시 조건 접두부를 확률 0.05로 학습형 null 임베딩으로 대체.

## 5. 학습 목표
- L_freq = Σ_k,d [½ exp(−s_k,d)·ℓ̄_k,d + ½ s_k,d], ℓ̄ = ℓ/(P+ε)^ρ.
- 최적 s에서 목적함수는 ½Σ log E[ℓ_k,d]가 되어 고정 가중에 불변 — 정규화의 실효는 zero-init·유한 학습의 조건화 효과에서 나온다고 저자들이 명시.

## 6. 학습 절차
- 각 FreqFM 모델과 짝 기준선(시간 좌표 FM)은 동일 구조·데이터·옵티마이저·예산: 20k step, 2×A800, 글로벌 배치 64.
- 주파수 통계는 훈련 행동에서만 추정, LIBERO-Plus는 LIBERO 체크포인트와 통계를 그대로 사용(제로샷).
- CFG 계수 λ_0를 검증으로 먼저 고정한 뒤 예산 β를 선택.

## 7. LIBERO / LIBERO-Plus (Table 1, 2)
| 방법 | Spatial | Object | Goal | Long | Avg | LIBERO-Plus Total |
|---|---|---|---|---|---|---|
| π0 (재현) | 93.2 | 98.4 | 96.4 | 93.6 | 95.4 | 62.7 |
| FreqFM (π0) | 95.8 | 97.8 | 95.6 | 95.2 | 96.1 | 68.2 |
| π0.5 (재현) | 95.2 | 99.6 | 97.2 | 94.6 | 96.7 | 66.5 |
| **FreqFM (π0.5)** | 98.6 | 99.4 | 97.6 | 95.8 | **97.9** | **75.8** |
- FreqFM(π0.5) LIBERO-Plus: Camera 58.8(+17.1), Robot 77.2(+14.8), Language 81.5, Light 97.0, Background 89.1, Noise 53.5(+15.7), Layout 84.4.
- 개선폭은 기준선이 가장 약한 섭동(Camera, Robot, Noise)에서 최대이며 기준선 성공률과 음의 상관(π0 −0.81, π0.5 −0.95).

## 8. VLA-Arena (Table 3)
- FreqFM(π0.5) Total 68.2 (π0.5 재현 64.0), FreqFM(π0) 57.3 (π0 재현 54.7).
- Safety·Distractor 범주에서 크게 개선(Dynamic Obstacles +16.0, Hazard Avoidance +11.4)되지만 Extrapolation(PC, TW, UO)은 두 백본 모두 하락 — 훈련 스펙트럼과 다른 행동 통계에 고정 사전이 불일치.
- π0.5가 FAST(DCT 토큰)로 사전학습되어 주파수 표현에 이미 노출된 덕에 π0보다 개선폭이 크다고 해석.

## 9. 실제 로봇 (Table 4, AgiBot Expedition A2, 과제당 30회)
| 과제 | π0.5 SR | FreqFM SR |
|---|---|---|
| Folder Filing | 53.3 | 66.7 |
| Toy Storage | 90.0 | 90.0 |
| Kitchen Tidy | 56.7 | 60.0 |
| Table Cleanup | 60.0 | 66.7 |
| Cup Stacking | 30.0 | 53.3 |
| Stamp & Handover | 56.7 | 70.0 |
| **평균** | 57.8 | **67.8** |
- Sub-SR 평균 70.6 → 79.0. 정밀 정렬이 필요한 과제(컵 쌓기 +23.3, 폴더 삽입 +13.4, 엽서 집기 +13.3)에서 이득 집중 — 저에너지 고주파 말단 보정을 raw MSE가 가장 약하게 가중하기 때문이라는 해석.

## 10. 어블레이션 (Table 5, π0.5)
| 변형 | LIBERO | LIBERO-Plus |
|---|---|---|
| π0.5 (시간 FM) | 96.7 | 66.5 |
| + 시간 영역 대응 구성요소 | 97.0 | 69.0 |
| **FreqFM** | 97.9 | 75.8 |
| w/o Matched Source | 97.2 | 71.6 |
| w/o Adaptive Objective | 97.9 | 72.2 |
| w/o Transport Budget | 97.8 | 73.9 |
- 같은 세 기법을 시간 영역에서 구현하면 효과가 훨씬 작다 → 주파수 좌표가 핵심.
- 예산 CFG는 성공률 영향은 작지만 LIBERO 실행 궤적 평균 절대 jerk를 0.442→0.193(−56%)로 낮춤.

## 11. 강점과 한계
**강점**
- 백본 수정 없이 액션 전문가의 생성 과정 세 단계를 일관된 주파수 통계로 묶은 원리적 설계.
- 두 백본·세 시뮬레이션 벤치마크·실로봇 6과제에서 짝 기준선 대비 일관된 개선.

**한계**
- 스펙트럼 통계가 훈련 집합 수준에서 고정되어 관측·지시에 따라 변하지 않음 → VLA-Arena Extrapolation 회귀.
- 부드러운 행동 차원을 체화별로 지정해야 함.
- LIBERO는 포화 상태라 개선폭(+0.7/+1.2)이 작고, 짝 기준선 재현값이 π0.5 공식 보고값(96.9)보다 낮다.

## 12. VLA-Tracker 관점 평가
π0.5(및 π0)를 주파수 조건 flow matching으로 직접 미세조정한 자체 학습 정책이므로 ACCEPTED, `action_head_category: flow_matching`. 헤드라인은 π0.5 변형으로 LIBERO 4개 스위트·보고 평균 97.9와 LIBERO-Plus 섭동별·Total 75.8을 `benchmarks.libero`에 넣었다. π0 변형, VLA-Arena, 짝 기준선, 어블레이션, 실로봇 SR/Sub-SR은 별도 블록으로 분리했다. 같은 저자진의 TFGCA와 짝 기준선 π0.5 재현 LIBERO 수치(95.2/99.6/97.2/94.6)가 동일하다는 점도 참고할 만하다.

<!-- VERIFIED: pdf -->
