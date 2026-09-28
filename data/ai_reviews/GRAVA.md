# GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving

> **한 줄 요약**: 행동 관련 언어 지칭을 2D 박스와 자차 기준 물리 상태에 묶는 궤적 고정 타입 그래프(GRA)를 추론 텍스트로 직렬화하고, 곧바로 STOP/CRAWL/CURVE/CRUISE 프리미티브 기반 Executable Planner 행동을 한 autoregressive 스트림에서 생성하는 Qwen3-VL-8B 주행 VLA로, Active RL 후 NAVSIM navtest PDMS 90.48을 기록.

- **arXiv**: 2609.15169v1 (2026-09-14, cs.CV), IEEE TPAMI 투고
- **소속**: Beijing Institute of Technology (National Engineering Research Center of Electric Vehicles; Shenzhen Automotive Research Institute), Nanyang Technological University, Shenzhen Jiguangzhijie Technology Co., Ltd.
- **코드**: https://github.com/AhernResearch/grava

---

## 1. 배경 및 동기

추론을 먼저 생성하는 주행 VLA가 늘고 있지만 저자들은 두 가지 격차를 지적한다. (G1) **grounding gap**: 추론 속 객체가 이미지 영역과 계량적 상태로 해소되지 않아, 그럴듯한 설명이 실제 물리 증거와 분리된다. (G2) **reasoning-to-action fragmentation**: grounding이 보조 QA나 별도 모듈로만 학습되어, 그 증거가 이후 상호작용 추론·결정·행동 생성까지 이어지지 않는다.

## 2. 문제 정의

단일 VLM 정책 π_θ(a | x, r)이 입력 x(전방 카메라, 자차 상태, 명령, 과거 궤적)로부터 grounded reasoning r을 생성한 뒤 곧바로 compact 행동 a = (primitive p, gear g, 파라미터 φ)를 생성한다. a는 고정 기하 디코더로 4초 8×2 웨이포인트로 변환된다. 목표는 추론 내 증거가 결정과 행동으로 끊김 없이 이어지도록 표현과 학습을 설계하는 것.

## 3. 방법

- **GRA 표현**: 장면·객체·상호작용·결정·행동 앵커 노드로 이루어진 궤적 고정 타입 그래프. 객체 노드는 `<|box_start|>` 박스 토큰과 거리·속도 등 물리량을 가진다. 각 상호작용/객체 결정이 grounded 객체 증거에 연결되고, 객체 결정이 자차 결정과 행동 앵커에 도달해야 한다는 경로 제약(Eq. 10–11).
- **Executable Planner**: STOP(끝점), CRAWL(8점), CURVE(3개 제어점), CRUISE(진행거리·끝속도·횡오프셋)의 프리미티브별 파라미터 공간. 학습 가능한 궤적 헤드 없이 결정론적 디코딩.
- **Agentic 데이터 파이프라인**: 2D 검출, 3D 상태, BEV, OCR 등 도구를 쓰는 다중 에이전트가 forward scene grounding과 expert 궤적에서 거꾸로 추적하는 backward trajectory anchoring을 결합, 같은 그래프에서 인지 QA와 계획 추론 감독을 투영.
- **점진적 학습**: GRA 사전학습 → 경량 planner warm-up → 검증된 self-distillation → Active RL Loop(GRPO + DAPO식 비대칭 클리핑, PDMS 보상, 고보상 행동이 도달 가능하지만 아직 불안정한 "recoverable" 장면을 반복 선별).

## 4. 데이터

GR-NavSim(nuPlan/NAVSIM 기반): grounded 상태 라벨 장면 107K, grounded QA 2.2M, 전방 카메라 Executable Planner 예시 415K, GRA 추론 트레이스 70K. 저자들은 비템플릿 개방형 주행 QA 데이터셋 중 최대라고 주장. 행동 감독에는 가용 인간 주행 데모의 약 60%만 사용. 별도로 내부 long-tail 벤치마크(50K 클립, 700K 프레임; 경로 장애물, 차로 빌림, 도로 위험)를 구축.

## 5. 구현 세부

- 백본 Qwen3-VL-8B-Instruct, 전방 단일 카메라.
- 주요 결과는 장면당 greedy 1회 생성; 다중 샘플은 self-distillation, 분석, Active RL에만 사용.
- RL 샘플링: PDMS 0.9 미만 사례에 8개 롤아웃 등 필터(부록 Table S2).
- GPU 수, 학습 시간, 학습률 등 세부 하이퍼파라미터는 본문 범위에서 Not stated in the paper.

## 6. 실험 설정

- NAVSIM navtest v1 공식 full protocol, PDMS(NC, DAC, EP, TTC, Comfort).
- 내부 long-tail 벤치마크: 궤적을 롤아웃해 Collision/Non-Drivable-Space 게이트를 곱한 Closed-loop Driving Score CDS = Safety·(0.45 KOC + 0.35 Progress + 0.20 Comfort). KOC는 통과/정지 주석을 경로상의 기하 제약으로 변환해 측정.
- GR-NavSim 보유 평가셋 grounded QA(0–10 점수/결정 라벨 정확도).

## 7. 주요 결과

**Table III – NAVSIM navtest v1 (단일 샘플)**

| 방법 | 백본 | PDMS |
|---|---|---|
| DiffusionDrive | — (C+L) | 88.1 |
| ReCogDrive | InternVL2-8B | 89.6 |
| DriveVLA-W0 | Emu-3-8B | 90.2 |
| ExploreVLA | — | 90.4 |
| AutoVLA Post-RFT | Qwen2.5-VL-3B | 89.1 |
| Curious-VLA | Qwen2.5-VL-3B | 90.3 |
| AdaThinkDrive | InternVL3-8B | 90.3 |
| GRAVA (RL 전) | Qwen3-VL-8B | 82.1 |
| **GRAVA (RL 후)** | Qwen3-VL-8B | **90.5** (Table IV: 90.48; NC 98.8, DAC 97.6, EP 83.5, TTC 97.1, Comf 100.0) |

직접 autoregressive 행동 생성 VLA 중 최고이며, 추가 궤적 헤드/월드모델 계열(ExploreVLA 90.4, DriveVLA-W0 90.2)과 비슷한 수준.

**Table IV – NAVSIM 절제**: 사전학습 없음 79.80(−10.68), Executable Planner 대신 직접 웨이포인트 87.23(−3.25), SD·RL 없음 81.74, SD만 82.10, 전체 90.48.

**Table V – 내부 long-tail (CDS / KOC)**: 사전학습 없음 79.7/81.7, 비grounded 사전학습 82.2/84.2, action-only 74.8/77.4, coarse reasoning 77.9/80.6, grounded objects only 85.5/87.9, Active RL 없음 73.1/69.8, **Full GRA 90.1/92.3**.

**Table VI – Grounded QA Overall**: Qwen3-VL-8B 4.40, Kimi-K2.5 4.77, GPT-5.4 5.39, **GRAVA 6.86**.

**반사실 추론 개입(Fig. 6)**: 저보상 추론을 고보상 GRA 추론으로 바꾸면 정규화 PDMS 0.438 → 0.823, 승률 +38.5pp — 추론이 실제 행동을 좌우함을 보임.

## 8. Related Work 상의 위치

EMMA, DriveLM, DriveVLM, OpenDriveVLA, Alpamayo-R1(Chain-of-Causation), ReCogDrive, DriveAgent-R1 등 추론형 주행 VLA 계열에 속한다. 행동 인터페이스 측면에서는 별도 궤적 생성기를 두는 ReCogDrive/DriveVLA-W0와 달리 AutoVLA, Curious-VLA처럼 직접 행동 생성 패러다임이지만, 코드북 토큰 대신 프리미티브별 파라미터 스키마를 쓴다는 점이 다르다.

## 9. 강점

1. grounding → 상호작용 → 결정 → 행동을 한 스트림으로 잇는 표현 설계가 명확하고, 박스 토큰과 물리량으로 추론의 검증 가능성을 높였다.
2. 절제가 풍부하다: 사전학습 유무, 비grounded 사전학습, 추론 구조 단계별, 행동 표현, SD, RL을 모두 분리.
3. 반사실 추론 개입 실험으로 "추론이 행동에 인과적으로 영향을 준다"는 주장을 직접 검증.
4. 데이터의 60%만 행동 감독에 사용하고도 경쟁력 있는 PDMS.
5. 코드 저장소 링크 제공.

## 10. 약점 및 한계

1. NAVSIM 최종 성능의 대부분이 Active RL에서 나온다(82.10 → 90.48). RL 전 모델은 UniAD(83.4)보다도 낮아, 표현 자체의 기여와 RL 최적화의 기여를 분리하기 어렵다.
2. 경쟁 방법 대비 우위가 작고(90.48 vs 90.3~90.4) 다중 시드나 신뢰구간이 없다.
3. long-tail 핵심 결과가 비공개 내부 벤치마크에 의존하고, 그 벤치마크 지표(KOC, CDS 가중치)도 저자 설계다.
4. Grounded QA 평가는 자체 데이터 보유 분할과 0–10 점수 기반으로, 채점 방식 편향 가능성이 있다.
5. NAVSIM v2(EPDMS)나 폐루프 벤치마크(Bench2Drive 등) 결과가 없다.
6. 학습 연산량·하드웨어 세부가 부족하다.

## 11. 재현 및 확장 아이디어

- NAVSIM v2/EPDMS와 Bench2Drive 폐루프에서의 평가.
- 동일 RL 예산을 action-only 모델에 적용한 비교로 GRA 표현의 순수 기여 분리.
- 다중 카메라 입력과 추론 길이-지연 트레이드오프 분석.
- GR-NavSim 공개 시 다른 VLA의 grounded 사전학습 데이터로 활용.

## 12. 총평

GRAVA는 주행 VLA의 추론이 실제 장면 증거와 실행 가능한 행동에 연결되어야 한다는 문제의식을 타입 그래프 표현, 에이전트 기반 데이터 구축, 점진적 SFT+RL 학습으로 일관되게 구현한 대형 연구다. 절제와 반사실 개입 실험이 설득력 있지만, 최종 NAVSIM 수치의 상당 부분이 Active RL에 기인하고 핵심 long-tail 증거가 비공개 벤치마크에 있다는 점은 감안해야 한다.

**한 문장 요약**: 2D 박스·물리량으로 grounded된 추론 그래프를 직렬화해 프리미티브 기반 Executable Planner 행동과 한 스트림으로 생성하고 GRPO Active RL로 다듬어 NAVSIM PDMS 90.48을 달성한 Qwen3-VL-8B 주행 VLA.

<!-- VERIFIED: pdf -->
