# EGR — Sensing Which Modality Matters: Evidence-Gated Regularization for Robust VLA Policies

- arXiv: 2609.03142 (v1 2026-09-02)
- 소속: University of North Carolina at Chapel Hill, Mitsubishi Electric Research Laboratories (MERL) (Yue Yang, Diego Romeres, Chiori Hori, Gedas Bertasius, Daniel Szafir, Siddarth Jain)

## 1. 한 줄 요약

EGR은 π0.5를 LoRA로 미세조정할 때 "프레임별·센서별 과제 관련성 증거(evidence)"로 두 가지 일관성 손실(저증거 센서에 대한 불변성, 고증거 센서의 단일 센서 충분성)을 켜고 끄는 학습 목적 함수다. 구조 변경과 추론 오버헤드가 없고, BEHAVIOR-1K 기반 벤치마크 MAN 성공률을 12.5% → 16.4%(Full), 9.4% → 16.5%(NoUseless), 2.8% → 6.1%(UsefulOnly)로, 실세계 양팔 플랫폼의 물리적 방해물 조건을 30% → 85%로 끌어올린다.

## 2. 문제 설정

VLA는 여러 카메라·촉각 입력을 초기 융합하는데, 적고 균질한 로봇 시연으로 학습하면 과제와 무관한 센서 간 상관에 기대는 **modality entanglement**가 생긴다. 저자들은 이를 두 실패 모드로 정식화한다.
- **Nuisance sensitivity**: 현재 상태에서 과제 정보가 없는 센서(예: 주행 중 바닥만 보는 손목 카메라)에 방해물을 넣으면 정책이 멈춘다.
- **Single-modality insufficiency**: 유용한 센서가 여럿일 때 하나만 남기고 가려지면(예: 선반 판이 헤드 카메라를 가림) 남은 손목 카메라로 이어가지 못한다.
랜덤 modality dropout은 센서·시점을 균일하게 다뤄 상태 의존적 선택을 못 하고, 어텐션 기반 선택은 같은 편향된 데이터에서 배우므로 같은 상관을 물려받는다.

## 3. 핵심 아이디어

"어느 센서가 지금 과제 증거를 갖고 있는가"를 데모 통계로 배우지 않고 **과제 구조에서 직접 계산**해, 이 점수로 정규화 손실을 게이팅한다. 증거 계산(모달리티별)과 정규화(모달리티 무관)를 분리해 카메라와 GelSight 촉각에 같은 틀을 적용한다.

## 4. 아키텍처

- 기반 정책: π0.5 (PaliGemma 2B + 300M action expert, flow-matching). 구조는 그대로이며 LoRA 어댑터만 학습.
- 비전 증거: 인스턴스 분할로 focal 객체와 interaction 객체의 픽셀 면적 비율을 구하고, focal 객체가 보일 때만 interaction 면적을 더함(s = a_focal + α·g·a_inter, α=0.3). 카메라 간 최대값으로 정규화하고, 어느 카메라에도 증거가 없는 탐색 프레임은 전역 게이트로 정규화를 끈다. 실세계에선 Grounding DINO + SAM2로 오프라인 분할.
- 촉각 증거: 희소 접촉 과제는 접촉 여부(0/1), 지속 접촉 과제는 목표 촉각 특징(예: 표면 균열)과의 유사도.

## 5. 학습과 추론

- 힌지 가중치: E < τ_low(0.2)이면 불변성 손실 L_inv(해당 센서만 오염시킨 입력과 원본의 velocity 예측 차이), E > τ_high(0.5)이면 충분성 손실 L_suff(해당 센서만 남긴 입력과 원본의 차이). 중간 구간은 기본 모방 손실만.
- 총 손실 L = L_flow + 0.2·L_inv + 0.1·L_suff. 오염 연산자는 학습 시 랜덤 직사각형 지우기(면적 30–80%).
- 중요도 샘플링으로 모달리티 하나만 뽑아 가중치 합으로 보정하는 비편향 추정기를 써서 추가 순전파를 2|M|회에서 2회로 줄임(부록 D에 증명).
- 설정: OpenPI Comet의 π0.5 체크포인트에서 시작, 2×B200, 배치 288, 50k step, LR 2.5e-5. 추론 시 비용은 원래 π0.5와 같다.

## 6. 주요 결과 — BEHAVIOR-1K 롤아웃 (Table 2, 세그먼트당 20회)

| 방법 | NAV Full | NAV SingleUseful | MAN Full SR | MAN NoUseless SR | MAN UsefulOnly SR |
|---|---|---|---|---|---|
| vanilla π0.5 | 42.27 | 15.45 | 12.50 | 9.44 | 2.78 |
| ModDrop | 40.00 | 15.45 | 7.50 | 6.81 | 2.78 |
| **EGR** | 40.91 | **37.27** | **16.39** | **16.53** | **6.11** |

벤치마크는 BEHAVIOR-1K Challenge 12개 과제(사용 가능 11개)에서 증거 기반 필터로 뽑은 47개 세그먼트(NAV 11, MAN 36)다. NAV Full은 약간 떨어지지만(42.3 → 40.9) SingleUseful은 15.5 → 37.3으로 크게 오른다. 절대 성공률은 MAN에서 여전히 낮다(Full 16.4%).

## 7. 추론 전용 진단과 실세계 결과

- Suite 1(Table 1, 오염 시 행동 MSE 증가율, 낮을수록 좋음): NAV SingleUseful 31.4% → 15.8%, MAN NoUseless 12.7% → 5.1%, MAN UsefulOnly 158.5% → 121.8%(vanilla → EGR).
- 양팔 Kinova(RGB 3대, 4과제×10회, Table 12): Clean 70 → 80, NoUseless 52.5 → 90, UsefulOnly 25 → 72.5, RealDistractor 30 → 85. 세 오염 조건 모두 Fisher 검정 p < 0.001.
- MELFA ASSISTA + GelSight 2개(2과제×10회, Table 13): Clean 90 → 85, SingleUseful 40 → 90, RealDistractor 55 → 70(p = 0.51로 유의하지 않음).

## 8. 어블레이션 및 통계 분석

별도의 구성요소 어블레이션(L_inv만, L_suff만 등)은 본문에 없다. 대신 부록 C에서 모든 수치에 표준오차·Wilson/부트스트랩 CI·Wilcoxon/Fisher 검정을 제공한다. MAN Full SR 향상은 vanilla 대비 p = 0.065로 경계선이고, NAV는 세그먼트가 11개뿐이라 SingleUseful에서도 p = 0.0625가 하한이다. 반면 MAN NoUseless/UsefulOnly SR 향상은 두 베이스라인 모두에 대해 p < 0.01이다.

## 9. 설계 해석

증거 점수를 "정답"이 아닌 소프트 귀납 편향으로 쓰고, 불변성과 충분성을 서로 다른 증거 구간에 배치해 상충을 피한 점이 핵심이다. ModDrop이 오히려 Full 성능을 깎는 반면(MAN 12.5 → 7.5) EGR은 Full도 올린 것은, 상태 의존적으로 정규화 대상을 고르는 것이 균일한 드롭아웃보다 표현 학습에 덜 해롭다는 근거다.

## 10. 강점

- 구조 변경·추론 비용 없이 기존 VLA 학습 파이프라인에 붙는 손실 함수.
- 카메라와 촉각이라는 이질적 센서에 같은 정규화 틀을 적용하고 두 실로봇 플랫폼에서 검증.
- 중요도 샘플링으로 모달리티 수와 무관한 학습 비용, 비편향성 증명.
- 통계적 유의성 분석이 충실하다.

## 11. 한계

- 증거 계산에 학습 시 분할 마스크와 과제별 focal/interaction 객체 주석이 필요하다(실세계는 수동 프롬프트 Grounding DINO).
- 시뮬레이션 절대 성공률이 낮고(MAN Full 16.4%), NAV Full은 소폭 하락.
- 평가 벤치마크를 저자들이 직접 필터링해 구성했으며 BEHAVIOR-1K 50개 과제 중 12개만 포함.
- 실세계 시험이 조건당 10회로 작고, 촉각 RealDistractor 향상은 유의하지 않다. 손실 항별 어블레이션이 없다.

## 12. VLA-Tracker 관점 평가

π0.5를 새 손실로 LoRA 미세조정한 자체 학습 정책이므로 수록 대상이다. 표준 추적 벤치마크(LIBERO, CALVIN 등) 결과는 없어 순위에는 들어가지 않는다. BEHAVIOR-1K 결과는 오염 조건별로 별도 블록(behavior1k_man_full / nouseless / usefulonly, behavior1k_nav_*)에 나누고 베이스라인은 조건별 sibling 블록으로 분리했다. `real_world`에는 양팔 플랫폼 Clean 조건(80%)만 두고, 오염 조건과 촉각 플랫폼은 별도 블록에 기록했다. 다중 센서 VLA의 강건성을 학습 목적 함수 수준에서 다룬 드문 사례다.

<!-- VERIFIED: pdf -->
