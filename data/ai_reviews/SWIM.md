# SWIM — Vision-Language-Grounded Soft Whole-Body Interactive Manipulation

- arXiv: 2609.17035 (v1, 2026-09-15)
- 소속: Nanyang Technological University, National University of Singapore, MBZUAI, A*STAR Institute of Advanced Intelligence and Computing
- 학회: Not stated in the paper / 코드 공개: Not stated in the paper

## 1. 한 줄 요약

SWIM은 텐던 구동 나선형 연속체 소프트 로봇(SpiRob)이 언어·RGB로부터 **몸 전체의 변형과 접촉**으로 조작하도록, OpenVLA-OFT를 LoRA로 미세조정한 확산 헤드 정책 SWIM-VLA와 "시뮬레이션 가상 롤아웃 → 실로봇 open-loop 실행" 배포 방식을 결합한 프레임워크다. 시뮬 held-out에서 packing/reaching/grasping 100/96/88%, 실로봇에서 100/80/75%(같은 체크포인트 온라인 직접 배포 75/40/25%)를 달성한다.

## 2. 문제 설정

기존 VLA는 엔드이펙터 델타·그리퍼 같은 강체 로봇 행동 인터페이스를 가정한다. 소프트 로봇의 전신 조작에는 (1) 엔드이펙터 행동이 몸체 변형·접촉을 규정하지 못하는 행동-인터페이스 불일치, (2) 엔드이펙터 상태가 고차원 몸체 형상을 담지 못하는 상태 불일치, (3) sim-to-real 차이와 추론 지연에 따른 실행 불일치가 있다.

## 3. 핵심 아이디어

- 행동을 2채널 텐던 모터 토크 명령 청크로 두고, 동일 조건에서도 여러 유효 명령열이 존재하므로 확산 헤드로 조건부 분포를 모델링.
- Visual Soft Proprioception(VSP): 학습 시에만 쓰는 보조 헤드가 몸체 위 4개 순서 앵커의 이미지 좌표를 회귀해 공유 표현이 전신 기하를 보존하도록 유도.
- 배포 시 VLA 추론을 물리 루프 밖(가상 장면)으로 빼고, 생성된 전체 명령열을 open-loop로 실행하며 로봇 고유의 컴플라이언스(embodied mechanical intelligence)가 국소 접촉 적응을 담당.

## 4. 아키텍처

OpenVLA-OFT 백본(256×256 상단 RGB + 언어 + 현재 텐던 길이 벡터) → 공유 표현 h_t → (a) 확산 행동 헤드(학습 50 스텝 노이즈 예측, 추론 10 DDIM 스텝, H=15 명령 = 1.5 s, 실행 K=8), (b) VSP 헤드(512 은닉 2층 MLP, GELU; 학습 후 폐기). 명령률 10 Hz, 쿼리당 평균 약 700 ms. 파라미터 수는 논문에 명시되지 않음.

## 5. 학습과 추론

손실 L = L_act + 0.15·L_VSP. MuJoCo 시연(조건당 4개의 서로 다른 성공 궤적): packing 72, reaching 128, grasping 592 궤적. 세 과제를 하나의 정책으로 공동 미세조정 — LoRA rank 32, bf16, AdamW LR 3e-4 cosine, GPU당 배치 4, 20,000 스텝, A100 4장. 추론: 첫 실제 이미지에서 HSV 분할 + 평면 캘리브레이션으로 객체 위치를 추정해 가상 장면을 초기화하고, 시뮬에서 반복 롤아웃하며 실행된 명령을 이어 붙여 완전한 명령열 U를 만든 뒤 실로봇에서 open-loop 실행.

## 6. 주요 결과 — 시뮬레이션 (Table III)

| 방법 | Packing | Reaching | Grasping |
|---|---|---|---|
| OpenVLA-OFT (동일 행동공간, L1) | 90 | 72 | 52 |
| **SWIM-VLA** | **100** | **96** | **88** |
| w/o VSP | 98 | 90 | 78 |
| w/o diffusion head (L1 회귀) | 94 | 84 | 69 |

held-out 조건: packing 50, reaching 50, grasping 100, 조건당 1회 롤아웃. grasping 대상은 고정(anchored).

## 7. 주요 결과 — 실로봇 (Table IV, 과제당 20회)

| 배포 방식 | Packing | Reaching | Grasping |
|---|---|---|---|
| Direct SWIM-VLA (온라인 쿼리) | 15/20 | 8/20 | 5/20 |
| **SWIM (가상 롤아웃 + open-loop)** | **20/20** | **16/20** | **15/20** |

## 8. 어블레이션

VSP 제거 시 reaching/grasping이 90/78로 하락(packing 98) → 기하 감독이 원위부 배치와 접촉 형성에 특히 기여. 확산 헤드를 결정적 L1 회귀로 바꾸면 grasping이 88 → 69로 크게 떨어져, 다봉 명령 분포 모델링이 접촉이 많은 전신 조작에서 중요함을 보인다. 배포 방식 비교(7절)는 같은 체크포인트로 sim-to-real·지연 효과를 분리한다.

## 9. 분석

Fig. 4의 공간 분포에서 reaching/grasping 실패는 작업공간 주변부에 몰리며, VSP나 확산을 제거하거나 OpenVLA-OFT일 때 더 넓게 퍼진다. 실로봇에서 시뮬 대비 남는 격차는 장면 초기화의 객체 위치 오차와 재료·구조 특성 차이로 설명된다.

## 10. 강점

- VLA를 소프트/연속체 로봇의 전신 접촉 조작으로 확장한 드문 사례이며, 행동공간을 액추에이터 명령으로 직접 정의했다.
- VSP·확산 헤드 기여를 통제 어블레이션으로 분리했고, 동일 체크포인트로 배포 전략만 바꾼 비교가 깔끔하다.
- 로봇의 수동 컴플라이언스를 활용해 온라인 추론 지연 문제를 우회하는 설계가 실용적이다.

## 11. 한계

- 평면(2D) 조작, 2개 텐던, 고정된 리셋 자세, 고정(anchored) 파지 대상으로 제한된다(저자 명시).
- 배포가 사전 정의된 객체 모델·색상 기반 분할에 의존해 일반 장면으로의 확장이 어렵다.
- open-loop 실행이라 실행 중 외란·물체 이동에 대응할 수 없다.
- 표준 VLA 벤치마크가 없고 자체 과제만 평가해 다른 모델과 순위 비교가 불가능하다.

## 12. VLA-Tracker 관점 평가

OpenVLA-OFT를 LoRA로 미세조정하고 확산 헤드·VSP 헤드를 학습한 자체 정책(SWIM-VLA)과 정량 결과를 보고하므로 수록 대상이다. 시뮬 결과는 비표준 `soft_robot_sim_heldout` 블록, 실로봇은 `real_world`(전체 SWIM 프레임워크 기준)와 `real_world_direct_deployment` 별도 블록에 기록했다. 소프트 로봇·전신 조작이라는 새로운 embodiment 축의 대표 사례로 의미가 있다.

<!-- VERIFIED: pdf -->
