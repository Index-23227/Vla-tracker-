# GRAFT — Grounded and Efficient Online Reinforcement Adaptation for Fine-Grained Robot Manipulation

- arXiv: 2608.27079 (v1 2026-08-27, v2 2026-08-28)
- 소속: 중국과학기술대학(USTC) 쑤저우 고등연구원, USTC 생명의학공학부
- 참고: 트래커의 FrozenGraft와는 무관한 별개 논문이다.

## 1. 한 줄 요약

GRAFT(Grounded Reinforcement Adaptation for Fast Task Learning)는 사전학습 VLA를 실험실(바이오) 정밀 조작 과제에 **실로봇 온라인 RL로 적응**시키는 프레임워크다. 카메라별 visual anchor를 학습 시에만 다중 영역 마스크로 감독하고(배포 시 제안 불필요), 단일 스텝 consistency 행동 헤드와 frozen prefix KV 캐시 재사용으로 학습을 가속한다. 45분 적응 예산에서 4개 과제 최종 성공률 82.5%(33/40)를 달성한다.

## 2. 문제 설정

피펫 팁 장착, 페트리 접시 뚜껑 열기, 액체 이송, 원심분리 튜브 적재 같은 과제는 장면 인식보다 작은 국소 접촉 기하에 성패가 달려 있다. 시연과 과제 수준 스칼라 보상은 어떤 영역이 중요한지 알려주지 않으므로, 제한된 실로봇 상호작용만으로 과제 관련 grounding을 배우기 어렵다. 또한 VLA 추론과 replay 기반 업데이트가 비싸 고정된 벽시계 예산 안에서 경험을 충분히 활용하기 어렵다.

## 3. 핵심 아이디어

(1) 영역 수준 감독으로 **view-specific anchor**가 국소 단서에 집중하도록 가르치되, 여러 후보 영역을 하나의 union 마스크로 합치지 않고 anchor-영역 대응을 고정하지 않는 **identity-free multi-region(FreeDice)** 감독을 쓴다. (2) VLM prefix를 고정해 replay 샘플의 KV 상태를 캐시·재사용하고, 반복 flow-matching 대신 **단일 스텝 consistency 정책**을 써서 actor 지연과 learner 비용을 모두 줄인다.

## 4. 아키텍처

입력은 side·wrist RGB, 언어, 8차원 adapter 상태이며, H=10 행동 청크(7차원×10: 이동·회전·그리퍼)를 예측한다. 고정된 VLM prefix가 뷰별 토큰 X^v, pooled context c_t, KV 상태를 만든다. 각 뷰에 N_v=8개의 anchor 템플릿이 있고, context로 변조된 query가 자기 뷰 토큰에만 cross-attention해 anchor z^v_i를 만든다(투영은 뷰 간 공유). 전역 context 기반 softmax 게이트로 anchor를 재가중하고 centered-tanh adapter를 거쳐 행동 suffix에 결합한다. 행동 헤드는 노이즈 w 하나로 청크 전체를 한 번에 생성하는 consistency 정책이다(ConRFT, Consistency Policy 계열). 논문은 기반 VLA를 π0 계열 flow-matching 모델로 서술하지만 구체적 체크포인트명은 밝히지 않는다.

## 5. 학습 목표

Grounding: anchor attention A_i와 제안 마스크 M_k 사이의 soft Dice 비용 C_{i,k}에 대해, 각 제안이 어떤 anchor에 덮이도록(coverage) + 각 anchor가 유효 영역 중 하나를 보도록 soft-min 매칭을 매 프레임 재구성하는 FreeDice 손실과, anchor 간 중복 억제 손실 L_dup를 합친다. RL: twin critic(γ=0.98)이 Bellman 목표로 학습되고, actor는 L = 0.5·BC + 1.0·(−Q)로 학습된다. 반복당 critic 2회, actor 1회 업데이트, 10회마다 제안이 있는 시연 창으로 grounding 전용 actor 업데이트를 한다. 온라인 replay가 100창을 넘으면 시연/온라인 버퍼를 같은 비율로 샘플링하고, 사람 개입 창은 시연 버퍼에도 넣는다. anchor, 행동 헤드, critic 가중치가 실제로 갱신된다.

## 6. 실험 설정

JAKA 로봇팔, 4개 과제(Petri Dish De-lidding, Centrifuge Tube Loading, Precision Liquid Transfer, Pipette Tip Attachment). 과제당 원격조작 시연 10개, 방법당 온라인 적응 45분, 물체 위치 ±30 mm 무작위화. Pipette Tip은 3행동마다, 나머지는 10행동 전부 실행 후 재계획한다. 베이스라인: RL-FM(원래 다단계 flow matching), RL-CP(단일 스텝 consistency 헤드), GRAFT-Union(anchor 등 모두 동일하고 union 마스크 감독만 다름). 개입 에피소드는 자율 성공으로 세지 않는다.

## 7. 주요 결과 (Table I, 마지막 10 에피소드 rolling 성공)

| 정책 | PD | CT | PL | PT | 전체 |
|---|---|---|---|---|---|
| RL-FM | 2/10 | 0/10 | 0/10 | 2/10 | 4/40 (10.0%) |
| RL-CP | 3/10 | 3/10 | 9/10 | 5/10 | 20/40 (50.0%) |
| GRAFT-Union | 7/10 | 2/10 | 8/10 | 6/10 | 23/40 (57.5%) |
| **GRAFT** | **9/10** | **9/10** | 8/10 | **7/10** | **33/40 (82.5%)** |

GRAFT-Union 대비 +25pt(통제된 감독 방식 비교), RL-CP 대비 +32.5pt(초록의 수치). 가장 큰 차이는 튜브·그리퍼·적재 목표를 동시에 찾아야 하는 Centrifuge Tube Loading(2/10 → 9/10). Precision Liquid Transfer에서는 RL-CP가 더 빨리 수렴해 9/10으로 GRAFT(8/10)보다 높다. 개입률은 3개 과제에서 거의 0으로 수렴했다(그림 5).

## 8. Learner 효율 (Table II)

Centrifuge Tube Loading, 시연 10개 + replay 946청크, learner 1,000스텝 기준 steady throughput: RL-FM 1.99 → 6.30 steps/s(KV 캐시, 3.17×), RL-CP 2.21 → 21.96 steps/s(9.94×). 캐시 사용 시 RL-CP는 RL-FM보다 3.49× 빠르다. 단일 스텝 생성과 prefix 재사용이 상호보완적임을 보여준다.

## 9. Grounding 진단

그림 6에서 GRAFT의 anchor attention은 GRAFT-Union보다 피펫 팁, 튜브, 액체 이송 목표 주변에 더 집중된다. side-view anchor는 전역 접근·정렬, wrist-view anchor는 국소 접촉·삽입 영역을 본다. RL-FM/RL-CP는 action token→patch attention을 시각화해 비교 대상 양이 달라 정성적 비교에 그친다고 저자들이 명시한다. 제안 생성기와 마스크는 추론에서 완전히 제거된다.

## 10. 강점

- GRAFT와 GRAFT-Union이 감독 목적함수 하나만 다른 통제 비교라서 FreeDice의 효과가 깔끔하게 분리된다.
- 실제 벽시계 예산(45분)으로 비교해 실로봇 RL의 실용 지표에 맞췄다.
- KV 캐시 재사용은 단순하지만 learner 처리량을 최대 약 10배 올리는 실용적 기법이다.
- 배포 시 추가 모듈(제안기, 마스크, critic)이 필요 없다.

## 11. 한계

- 성공률이 별도 고정 정책 평가가 아니라 온라인 학습 궤적의 마지막 10 에피소드 rolling 값이라 분산이 크고 과제당 표본이 10개뿐이다.
- 기반 VLA 체크포인트, 파라미터 수, 제안 마스크 생성 방식 등의 세부가 부족하며 코드 공개 언급이 없다.
- 시뮬레이션 벤치마크가 없어 다른 VLA와 직접 비교할 수 없다.
- 정책이 반응형이라 과제 진행을 기억하지 못하며, 장기 과제로의 확장은 미검증이다(저자 인정).

## 12. VLA-Tracker 관점 평가

사전학습 VLA의 anchor·행동 헤드·critic을 실로봇 온라인 RL로 실제 갱신한 자체 정책이므로 수록 대상이다(RL post-training). 트래커에는 real_world 블록에 과제별 성공 횟수(x/10)와 논문 보고 전체 82.5%를 기록하고, GRAFT-Union·RL-CP·RL-FM을 별도 블록으로 두었다. 추적 벤치마크 점수는 없으므로 순위표보다는 실로봇 RL 적응(ConRFT, HIL-SERL, FORCE 계열)과 실험실 자동화 VLA 사례로서 의미가 있다.

<!-- VERIFIED: pdf -->
