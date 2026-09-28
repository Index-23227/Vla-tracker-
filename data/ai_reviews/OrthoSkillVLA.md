# OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation

> **한 줄 요약**: 사전학습된 X-VLA(0.9B)에 스킬을 순차적으로 추가할 때, 누적 gradient로 추정한 스킬 부분공간의 직교 여공간에만 LoRA 업데이트를 넣되 VLM과 ActionHead에 서로 다른 에너지 예산(ε_VLM=0.99, ε_Head=0.9999)을 주고, 최종 velocity decoder는 스킬별 소형 expert + 학습 없는 사영 라우터(MoE)로 바꾼 replay-free 연속학습 프레임워크. LIBERO-100 기반 스킬 증분 설정 Final SR 83.50%(KeepLoRA 56.61%), 실물 4스킬 최종 평균 86.25%.

- **arXiv**: 2608.19589v1 (2026-08-20, cs.RO)
- **소속**: Southeast University
- **코드**: https://github.com/Jiaqi-Wangx/OrthoSkillVLA
- **백본**: X-VLA 사전학습 모델(0.9B, flow matching)

---

## 1. 배경 및 동기

사전학습 VLA를 여러 스킬에 순차 적응시키면, 새 스킬의 업데이트가 기존 스킬의 표현과 velocity 매핑을 교란해 catastrophic forgetting이 생긴다. 아키텍처 기반 방법(스킬별 경로 분리)은 스킬이 늘수록 추론 비용이 커지고, gradient projection 계열(O-LoRA, KeepLoRA)은 크기를 고정하지만 **모델 전체에 단일 제약**을 건다. 저자들은 VLA 특유의 두 문제를 짚는다: (1) VLM은 넓은 의미 표현을 가져 부분공간 용량이 빨리 소진되는 반면 ActionHead는 국소적 velocity 패턴이라 작은 교란에도 취약하다(모듈 이질성), (2) 최종 velocity decoder는 readout 층이라 동결하면 표현력 병목, 갱신하면 기존 매핑 덮어쓰기.

## 2. 문제 정의

- 스킬 시퀀스 S_1, …, S_K를 demonstration replay 없이 순차 학습.
- R_{i,j}: 스킬 i 학습 후 스킬 j의 성공률. 지표 — FWT = (1/K)Σ R_{k,k}, NBT(망각), AUC(과정 전체 안정성), Final SR.
- 목표: 추론 footprint를 거의 늘리지 않으면서 망각을 줄이고 새 스킬 습득력 유지.

## 3. 방법

- **Gradient-informed skill subspace**: 각 스킬 학습 중 누적 gradient로 입력 특징 부분공간을 추정하고, 에너지 임계값 ε까지의 주성분을 "점유된 방향"으로 저장. 다음 스킬의 LoRA 다운사영 A는 잔차 gradient의 주방향으로 초기화하고, 업데이트는 점유 방향의 직교 여공간으로 제한.
- **Module-aware subspace budgeting**: VLM과 ActionHead에 별도 임계값. ActionHead는 엄격(0.9999)해 velocity 패턴을 보호하고, VLM은 느슨(0.99)해 의미 용량을 이후 스킬에 남긴다.
- **Feature-aware MoE decoder**: 최종 velocity decoder는 직교 미세조정에서 제외하고 스킬마다 소형 expert를 할당. 라우터는 학습 없이, 입력 특징을 각 스킬의 gradient-informed 기저에 사영한 친화도로 expert를 선택.
- 적용 대상: 입·출력 차원이 모두 hidden size(1024) 이상인 VLM·ActionHead 선형층.

## 4. 구현 세부

- 시작 체크포인트: X-VLA에 LIBERO용 domain soft prompt 차원을 하나 추가(기존 도메인 평균으로 초기화)하고 backbone 동결 상태로 LIBERO-Goal/Spatial/Object에서 새 파라미터만 warm-up. 모든 방법이 같은 체크포인트에서 출발.
- LoRA rank 64, AdamW, peak lr 5e-5, 1000 warmup 후 cosine, batch 16, bf16, 스킬당 15 epoch(약 35–40k step).
- 행동 공간: 3(위치) + 6(6D 회전) + 1(그리퍼) = 10차원 (X-VLA 인터페이스의 나머지 10채널은 양팔용으로 미사용).
- 스킬당 추가 디코더 상태 d_h·d_a + d_h·n_basis ≈ 기반 0.9B 모델의 0.0046%.

## 5. 평가 프로토콜

- **시뮬레이션**: LIBERO-100에서 복합 과제를 걸러낸 82개 단기 과제를 OpenClose(8), PickPlace(72), Turn(2)으로 분류하고 균형을 위해 4/4/2개 선택. 3가지 스킬 순서, 과제당 50 rollout.
- **실물**: 7-DoF xArm + 6-DoF Inspire 손 + Orbbec 336 손목 카메라, 4개 스킬(Flip, Pick, Push, Press), 스킬당 50 demo, 20 trial.
- **기준선**: SeqLoRA, IncLoRA, EWC, O-LoRA, KeepLoRA(통일 임계값 + 동결 decoder).

## 6. 실험 설계의 요점

기존 LIBERO 과제 증분(Spatial/Object)은 물체 정체성·위치만 바뀌고 운동 패턴이 비슷해 gradient 충돌을 충분히 드러내지 못한다는 이유로, 행동 분포가 크게 다른 스킬 증분 설정을 새로 구성했다. 따라서 결과는 표준 LIBERO suite 성능과 직접 비교 대상이 아니다.

## 7. 주요 결과

**Table 1 – 스킬 증분 LIBERO (3개 순서 평균)**

| Method | FWT↑ | NBT↓ | AUC↑ | Final SR↑ |
|---|---|---|---|---|
| SeqLoRA | 0.92 | 0.89 | 0.58 | 32.44 |
| IncLoRA | 0.90 | 0.83 | 0.58 | 34.11 |
| EWC | 0.73 | 0.68 | 0.46 | 25.17 |
| O-LoRA | 0.88 | 0.85 | 0.54 | 30.50 |
| KeepLoRA | 0.90 | 0.46 | 0.72 | 56.61 |
| **OrthoSkillVLA** | **0.94** | **0.13** | **0.88** | **83.50** |

**Table 3 – 절제**: Unified/Frozen(=KeepLoRA) 56.61 → Separated/Frozen 60.44 → Unified/MoE 67.94 → Separated/MoE 83.50. MoE decoder의 단독 기여가 더 크고, 둘을 합쳤을 때 시너지가 크다.

**Table 4/5 – 임계값**: ε_VLM=0.99 고정 시 ε_Head 0.99/0.999/0.9999 → Final SR 59.83/80.67/84.83. ε_Head=0.9999 고정 시 ε_VLM을 0.999로 올리면 85.50(+0.67)이지만 VLM SOR이 53.73%로 급증.

**Table 2 – 실물 (20 trial 중 성공)**: 마지막 단계에서 OrthoSkillVLA Flip/Pick/Push/Press = 16/16/17/20, KeepLoRA 11/12/15/19. 최종 평균 86.25%로 KeepLoRA보다 15.0% 높다.

**라우팅**: 학습 없는 라우터의 스킬 식별 정확도 OpenClose/Turn 98.9%, PickPlace 91.5%.

**Table 9 – LoRA rank**: 32/64/96 → Final SR 79.50/83.50/74.06; rank가 너무 크면 저에너지 꼬리 방향이 망각을 키운다.

## 8. Related Work 상의 위치

- 어댑터 확장·스킬 라우팅·원자 스킬 분해(아키텍처 기반) 대신, O-LoRA/KeepLoRA 같은 **부분공간 제약** 계열을 VLA 구조에 맞게 모듈별로 세분화.
- 표준 PEFT(SeqLoRA/IncLoRA)와 정규화 기반(EWC) 대비 명확한 우위.
- 백본으로 X-VLA처럼 소형 flow-matching VLA를 택해 실시간 배포를 염두에 둔다.

## 9. 강점

1. **VLA 특화 진단**: SOR(Subspace Occupancy Ratio) 분석으로 VLM과 ActionHead의 용량 소비 차이를 정량적으로 보여 준 뒤 설계를 도출.
2. **깔끔한 2×2 절제**로 두 구성요소의 개별·결합 효과를 분리.
3. **3개 순서 평균 ± 표준편차** 보고로 순서 편향을 줄임.
4. 추론 비용 증가가 매우 작고(스킬당 0.0046%), replay 불필요, 코드 공개.

## 10. 약점 및 한계

1. **소규모 설정**: 스킬 3개(시뮬), 4개(실물)뿐이라 스킬이 수십 개로 늘 때 VLM 부분공간 고갈·라우터 혼동이 어떻게 변할지 미지수.
2. 커스텀 LIBERO 스킬 증분 프로토콜이라 외부 결과와 비교하기 어렵다.
3. 라우터가 PickPlace에서 91.5%로 떨어지며, 오라클 라우팅 대비 격차(그림 수치)가 존재 — 스킬 간 특징 유사도가 높으면 취약.
4. 스킬 경계가 학습 시 명시적으로 주어진다는 가정(task-free 연속학습은 다루지 않음).
5. 기준선이 모두 LoRA/정규화 계열이며, 아키텍처 기반 연속학습 VLA와의 직접 비교가 없다.

## 11. 재현 및 확장 아이디어

- 10개 이상 스킬, 긴 시퀀스에서 SOR 포화 곡선과 성능 추적.
- 라우터를 소량 학습(또는 불확실성 기반 soft routing)해 PickPlace류 혼동 완화.
- π0/π0.5 등 더 큰 flow-matching VLA와 diffusion head로 일반화.
- 스킬 경계 없이 데이터 스트림에서 부분공간을 자동 갱신하는 task-free 변형.

## 12. 총평

OrthoSkillVLA는 "VLA 내부 모듈은 망각에 대해 다르게 행동한다"는 관찰을 SOR 분석으로 뒷받침하고, 모듈별 부분공간 예산과 스킬별 MoE 디코더라는 간단한 조합으로 KeepLoRA 대비 Final SR을 27pt 가까이 끌어올렸다. 다만 스킬 수가 적은 소규모 실험이라 확장성 검증은 후속 과제로 남는다.

**한 문장 요약**: X-VLA에 모듈별 직교 LoRA 예산과 학습 없는 라우팅의 스킬별 MoE decoder를 결합해 replay 없이 스킬 증분 LIBERO Final SR 83.50%, 실물 4스킬 86.25%를 달성한 연속 스킬 학습 방법.

<!-- VERIFIED: pdf -->
