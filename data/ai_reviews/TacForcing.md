# TacForcing — Streaming Action Generation with Execution-Time Tactile Feedback

- arXiv: 2608.25798 (2026-08-26)
- 소속: 상하이교통대학(SJTU) 컴퓨터과학대학 외 SCUT, ECUST, ECNU, PolyU, CCUT, UESTC, HUST
- 프로젝트: https://88runaway.github.io/tacforcing/

## 1. 한 줄 요약

TacForcing은 청크 기반 flow-matching VLA의 action expert를 **Streaming Action Expert**로 교체해, 한 청크를 블록 단위로 순차 완성·실행하면서 실행 도중 들어오는 촉각 피드백으로 남은 블록을 계속 정제한다. 여기에 최신 촉각 토큰이 "바로 다음에 실행될 블록"만 보게 하는 **Execution-Aware Tactile Attention(EATA)**을 더해 UniVTAC 6개 과제 평균 65%, 실세계 3개 과제 평균 69%를 기록한다.

## 2. 문제 설정

접촉이 많은 조작(삽입, 스포이트 짜기, 칠판 닦기)에서는 한 action horizon 안에서도 접촉 상태가 크게 변한다. 기존 청크 VLA는 실행 전 관측으로 청크 전체를 한꺼번에 생성하므로 촉각 조건이 실행 중 점점 낡는다. 저자들의 스포이트 에피소드 분석(Figure 2)에서 40-step horizon 중 35 step(1.17 s) 뒤 시각 표현의 코사인 거리는 약 0.005, 촉각은 약 0.55로, 시각은 거의 변하지 않지만 촉각은 크게 변한다. 기존 반응형 방법(RDP 등)은 별도 고주파 촉각 제어기를 두어 구조와 학습이 복잡해진다.

## 3. 핵심 아이디어

(1) 청크 전체가 하나의 flow time을 공유하는 대신 **블록별 flow time**을 두어 앞 블록부터 차례로 완성시키고(Diffusion Forcing 계열 스트리밍 생성), 완성된 블록은 곧바로 실행한다. (2) 실행 후 얻은 촉각으로 미완성 블록의 중간 상태를 이어서 정제한다. (3) 최신 촉각은 다음 실행 블록에만 직접 주입해 "촉각 획득 시점 vs 행동 실행 시점" 불일치를 줄인다.

## 4. 아키텍처

- VLM이 시각·언어·고유감각 문맥 C_t를 청크당 한 번 인코딩해 재사용한다. 시뮬레이션에서는 π0.5, 실세계에서는 GR00T N1.7에서 초기화한다.
- 손가락 끝 M개의 변형 맵(deformation map)을 공유 촉각 인코더 f_tac(FTP-1 사전학습 인코더로 초기화)로 각각 토큰화한다.
- 블록 크기 B=5. 시뮬레이션은 H=50, K=10 블록, 실세계는 H=40, K=8 블록이다. N=KS 샘플링 스텝에서 블록 k의 flow time은 λ_k^(n) = min(n/(kS), 1)이므로 블록 k는 스텝 kS에서 완성된다.
- EATA: 행동 쿼리 i와 촉각 키 m 사이에 가산 마스크를 두어 b(i)=k(다음 실행 블록)일 때만 0, 나머지는 −∞로 막는다. 뒤 블록은 촉각에 직접 접근하지 않고 flow 스케줄대로 진화하다가 자기 차례가 되면 그때의 촉각을 본다.

## 5. 학습

추론과 맞추기 위해 생성 진행도 p~U[0,1)를 샘플링하고, 식 (7)로 블록별 보간 상태를 만든 뒤 k*=⌊Kp⌋+1 단계에 해당하는 촉각 표현 Z^(k*−1)과 EATA 마스크를 사용한다. 손실은 아직 완성되지 않은 행동 위치에 대한 flow-matching 속도 회귀의 평균이다. 시뮬레이션은 과제당 50개 시연, 15k step, 배치 256, 최대 lr 5e-5, 실세계는 과제당 100개 시연, 30k step, 배치 256, 최대 lr 6e-5(모두 AdamW, cosine 감쇠, warm-up 2k)로 **VLA를 실제로 미세조정**한다.

## 6. 실험 설정

- 시뮬레이션: UniVTAC 6개 과제(Lift Bottle, Pull-out Key, Lift Can, Put Bottle in Shelf, Insert Hole, Insert Tube), 과제당 100 rollout.
- 실세계: RealMan RM75 7-DoF 양팔 + 22-DoF Sharpa Wave 손(손끝 촉각), 상단·손목 RealSense. 과제는 Stand Bottle, Transfer Liquid(투명 스포이트로 액체 옮기기), Wipe Board. 과제당 16회.
- 기준선: π0.5(촉각 없음), UniVTAC-ACT, FTP-1(공개 체크포인트), RDP(slow-fast 촉각 반응형). 실세계에서는 π0.5, FTP-1, GR00T N1.7.

## 7. 주요 결과 — UniVTAC (Table 1, 성공률 %)

| 방법 | Lift Bottle | Pull-out Key | Lift Can | Put Bottle in Shelf | Insert Hole | Insert Tube | 평균 |
|---|---|---|---|---|---|---|---|
| π0.5 | 88 | 43 | 46 | 43 | 39 | 48 | 51 |
| UniVTAC-ACT | 59 | 41 | 24 | 6 | 36 | 58 | 37 |
| RDP | 84 | 18 | 12 | 41 | 23 | 75 | 42 |
| FTP-1 | 89 | 35 | 66 | 23 | 62 | 76 | 59 |
| **TacForcing** | **90** | **48** | 63 | **43** | **69** | **79** | **65** |

6개 중 5개 과제에서 최고 또는 공동 최고이며, 유일한 예외는 Lift Can(63 vs FTP-1 66)이다. π0.5 대비 +14pp, RDP 대비 +23pp.

## 8. 실세계 결과

실세계 평균 69%로 FTP-1·GR00T N1.7·π0.5를 각각 17·27·42pp 앞선다(본문). 과제별 TacForcing 수치는 Table 2 기준 Stand Bottle 81, Transfer Liquid 50, Wipe Board 75다. Transfer Liquid에서 모든 기준선은 19% 이하였다. 기준선의 과제별 수치는 Figure 3에만 있어 YAML에는 넣지 않았다.

## 9. 어블레이션 (Table 2, 성공률 %)

| 구성 | Pull-out Key | Lift Can | Insert Hole | Stand Bottle | Transfer Liquid | Wipe Board | Sim 평균 | Real 평균 |
|---|---|---|---|---|---|---|---|---|
| Base(촉각 없음) | 43 | 46 | 39 | 56 | 19 | 50 | 43 | 42 |
| Fixed Tactile | 39 | 53 | 34 | 50 | 13 | 31 | 42 | 31 |
| w/o EATA | 39 | 55 | 58 | 69 | 31 | 44 | 51 | 48 |
| **TacForcing** | **48** | **63** | **69** | **81** | **50** | **75** | **60** | **69** |

초기 촉각 하나로 청크 전체를 조건화하면(Fixed Tactile) 오히려 실세계에서 42→31로 떨어진다. 블록마다 촉각을 갱신하면 51/48로 회복하고, EATA를 더하면 6개 과제 모두 개선되어 60/69가 된다. "촉각을 넣는 것"보다 "언제 어느 행동에 넣느냐"가 핵심이라는 주장을 잘 뒷받침한다.

## 10. 강점

- 별도 반응형 제어기 없이 단일 생성기 안에서 실행 중 피드백을 반영하는 깔끔한 설계이며, π0.5와 GR00T N1.7 두 백본 모두에 적용된다.
- Fixed Tactile vs 스트리밍 vs EATA의 단계적 어블레이션이 설계 결정 각각을 분리해 보여 준다.
- 학습 시에도 EATA 마스크와 단계별 촉각을 똑같이 써서 학습-추론 불일치를 없앴다.

## 11. 한계

- 시뮬레이션 과제당 100회, 실세계 16회의 단일 시드 결과이며 신뢰구간이 없다. 실세계 기준선 과제별 수치는 그림에만 있다.
- 실세계와 시뮬레이션의 백본이 달라(GR00T N1.7 vs π0.5) 백본 효과와 방법 효과가 섞일 수 있다.
- 추론 지연·제어 주파수, 블록 크기 B나 S에 대한 민감도 분석이 본문에 없다.
- 촉각 센서 종류는 손끝 변형 맵 하나뿐이라 다른 촉각 모달리티로의 일반화는 검증되지 않았다.

## 12. VLA-Tracker 관점 평가

π0.5/GR00T N1.7을 블록 단위 flow-matching 손실로 **실제로 미세조정**한 자체 정책이며 Table 1·2에 정량 결과가 있으므로 추적 대상에 포함한다. 행동 헤드는 블록별 flow time을 갖는 flow-matching expert이므로 `action_head_category: flow_matching`. UniVTAC는 추적 표준 벤치마크가 아니므로 별도 `univtac` 블록에 두고 논문이 보고한 평균 65를 `univtac_avg`로 기록했다. 실세계 과제별 수치는 Table 2의 TacForcing 행(평균 69)을 `real_world`에 넣었고, 기준선 4종과 두 어블레이션(시뮬레이션 3과제, 실세계 3과제)은 형제 블록으로 분리했다.

<!-- VERIFIED: pdf -->
