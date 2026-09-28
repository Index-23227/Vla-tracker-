# TFGCA — Time-Frequency Geometric Cross-Attention for Chunked Vision-Language-Action Models

**arXiv**: 2609.09925 · **기관**: 논문 PDF에 소속 미기재 (저자 구성상 AGIBOT / Xi'an Jiaotong University로 추정) · **날짜**: 2026-09-09

## 1. 한 줄 요약
액션 청크를 행동 공간에서 차원별 학습형 정상 웨이블릿(SWT)으로 시간–주파수 토큰화하고, 내적과 wedge product 크기를 섞은 기하 교차 어텐션으로 정제하는 4.23M 파라미터 drop-in 모듈을 π0.5에 붙여 미세조정, LIBERO 98.2 / LIBERO-Plus 73.0 / RoboTwin 랜덤화 42.7(+28.5) / AgiBot A2 61.67%를 보고.

## 2. 문제 설정
- 최신 VLA는 T×D 액션 청크를 한 번에 예측하지만, 청크는 일반 hidden 토큰 시퀀스 + 선형 헤드로 표현된다.
- 두 구조가 누락된다: (1) **주파수** — 느린 이동 추세와 빠른 접촉 보정이 한 토큰에 얽힘, (2) **교차 위상 기하** — reach/contact/grasp/settle 등 위상별 움직임이 표현 공간에서 거의 직교하는데, 내적 어텐션은 직교 근처에서 가장 둔감.
- 저자들은 RoboTwin 2.0 양팔 데이터에서 동시각 직교는 거의 없고, 시간축을 따라 순차적으로 직교 부분공간을 점유하는 "Temporal Orthogonal Division-of-Labor(TO-DoL)"를 관찰.

## 3. 핵심 아이디어
- (C1) 행동 공간 투영 후 차원별 학습형 SWT(db2 초기화, DC 제거, 선택적 차분)로 길이 보존 다해상도 주파수 토큰 생성.
- (C2) 시간 토큰(Q)이 시간–주파수 토큰(K,V)을 조회하는 교차 어텐션에서 A = (1−β)·softmax(S_dot) + β·softmax(S_wedge), ||q∧k||² = ||q||²||k||² − (q·k)². β는 학습형 스칼라(초기 0.5).
- Proposition 1: 정규화 후 ŝ²+ŵ²=1, β가 임계값을 넘으면 내적만으로는 불가능한 "더 직교한 키 우선" 순서를 실현.

## 4. 아키텍처
- π0.5(lerobot/pi05_base) 트랜스포머 출력과 선형 액션 헤드 사이에 삽입, 학습과 매 디노이징 스텝 모두 동일 적용.
- 출력 투영 W_O만 0 초기화 → 부착 시 기저 모델과 정확히 동일, 값 경로는 비영이라 학습 가능.
- 추가 파라미터 4,234,299(교차 어텐션 4.20M, SWT 인코더 0.029M, action_proj 0.007M), π0.5 백본의 약 0.10%.

## 5. 학습 목표
- L = L_FM + λ·L_align, L_align = ||Z_act − sg(a)||²로 행동 공간 투영에 물리적 의미 부여(λ: LIBERO 0.01, RoboTwin 0.1).

## 6. 학습 절차
- 공통: AdamW lr 2.5e-5 → 코사인 2.5e-6(30k step), 워밍업 1k, 배치 32/GPU·글로벌 64, bf16, 추론 10 디노이징 스텝.
- LIBERO: 청크 10, SWT 1레벨, 차분 0. RoboTwin: 청크 50, SWT 2레벨, 2차 차분, 과제당 30k step, clean 데모만 사용.
- 2×A800에서 LIBERO 약 26시간, RoboTwin 약 28시간. π0.5 기준선은 동일 데이터·예산으로 재현.

## 7. LIBERO 결과 (Table 1)
| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 (보고값) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| π0.5 (재현) | 95.2 | 99.6 | 97.2 | 94.6 | 96.7 |
| **TFGCA (full)** | 98.5 | 99.4 | 97.7 | 97.0 | **98.2** |
| w/o SWT | 96.8 | 99.1 | 97.1 | 97.8 | 97.7 |
| w/o geometry | 97.2 | 99.7 | 97.6 | 96.5 | 97.8 |
| w/o alignment | 94.9 | 99.5 | 98.5 | 95.5 | 97.1 |

## 8. LIBERO-Plus (Table 2, 제로샷)
| 방법 | Camera | Robot | Lang | Light | BG | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| π0.5 재현 | 42.1 | 62.9 | 77.2 | 96.3 | 88.4 | 38.4 | 79.6 | 66.7 |
| **TFGCA** | 51.5 | 73.4 | 81.8 | 96.4 | 87.8 | 52.3 | 81.7 | **73.0** |
| w/o geometry | 41.1 | 67.9 | 81.0 | 94.6 | 87.1 | 49.8 | 83.4 | 69.7 |
- OOD에서는 wedge 채널 제거가 가장 큰 하락(−3.3), 분포 내에서는 alignment 제거가 가장 큰 하락 — 순서가 뒤바뀜.

## 9. RoboTwin 2.0 (Table 3, 6과제 부분집합, clean/randomized)
| 과제 | π0.5 | TFGCA |
|---|---|---|
| open_microwave | 93 / 27 | 88 / 36 |
| stamp_seal | 15 / 5 | 19 / 3 |
| handover_block | 57 / 13 | 72 / 11 |
| turn_switch | 41 / 28 | 54 / 61 |
| stack_bowls_three | 83 / 6 | 74 / 59 |
| click_bell | 79 / 6 | 83 / 86 |
| **평균** | 61.3 / 14.2 | **65.0 / 42.7** |
- 랜덤화는 학습에 쓰이지 않아 제로샷 일반화 측정. click_bell, stack_bowls_three에서 π0.5가 약 6%로 붕괴할 때 86%, 59% 유지.

## 10. 실제 로봇 (Table 4, AgiBot A2, 과제당 20회)
- Soap into box 6/20 vs 4/20, Plush toys into basket 19/20 vs 17/20, Pull a tissue 12/20 vs 9/20.
- 전체 37/60(61.67%) vs 30/60(50.00%), +11.67 pts.

## 11. 강점과 한계
**강점**
- 항등 초기화로 사전학습 VLA에 안전하게 부착 가능하며, 파라미터 증가가 미미.
- OOD(LIBERO-Plus, 도메인 랜덤화)에서 개선폭이 커서 목적과 결과가 일관적.
- 저자들이 "메커니즘은 가설, 인과 귀속은 향후 과제"라고 증거의 경계를 명확히 서술.

**한계**
- RoboTwin은 6과제 부분집합이며 표준 50과제 프로토콜이 아니다.
- LIBERO는 청크 10·SWT 1레벨이라 주파수 분해가 사실상 추세/세부 2분할에 불과하다고 저자 스스로 인정.
- wedge 채널이 실제로 교차 위상 토큰쌍에 주의를 옮기는지에 대한 Q/K 라우팅 분석이 없다.
- 실로봇 데이터·절차 비공개, 소속 미기재.

## 12. VLA-Tracker 관점 평가
π0.5에 모듈을 부착해 전체를 공동 미세조정한 자체 학습 정책이므로 ACCEPTED. 액션 생성은 π0.5 flow matching 그대로라 `action_head_category: flow_matching`. LIBERO 4개 스위트와 논문 보고 평균 98.2, LIBERO-Plus 7개 섭동·Total 73.0을 `benchmarks.libero`에 등록했다. RoboTwin은 6과제 부분집합이므로 순위 오염을 막기 위해 `robotwin_v2`가 아닌 `robotwin_v2_6task_clean` / `robotwin_v2_6task_randomized` 블록으로 분리했다. 조직명은 PDF에 소속이 없어 동일 저자진의 FreqFM(AGIBOT / XJTU) 기준 추정임을 명시했다.

<!-- VERIFIED: pdf -->
