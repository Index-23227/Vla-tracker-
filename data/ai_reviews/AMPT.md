# AMPT: Aligning Multi-Trajectory Supervision with Policy Optimization for VLA Driving

> arXiv: [2608.30122](https://arxiv.org/abs/2608.30122) · Wuhan University / Dongfeng Research & Development Institute · 2026-08-31
> 한 줄 요약: "점수가 높은 궤적"이 아니라 "현재 정책이 소화할 수 있는 궤적"으로 multi-trajectory 감독을 구성하고, feasibility-first Pareto GRPO와 반복 distillation으로 NAVSIM v1 91.4 PDMS / v2 89.1 EPDMS를 달성한 driving VLA 학습 프레임워크.

## 1. 배경 및 동기

최근 driving VLA의 표준 파이프라인은 (1) 로그 데모로 imitation learning(IL) → (2) planning reward 기반 GRPO 후처리이다. 하나의 로그 궤적은 가능한 여러 정답 중 하나뿐이므로, IL 단계에서 추가 궤적(multi-trajectory supervision)을 넣어 행동 분포를 넓히는 방식이 널리 쓰인다. 추가 궤적은 보통 aggregate 점수, 물리적 타당성, 기하학적 다양성으로 고른다.

저자들이 지적하는 문제는 **supervision–optimization mismatch**이다. IL 체크포인트를 좋게 만드는 궤적 집합이 뒤이은 GRPO의 초기화로는 오히려 나쁠 수 있다는 것이다. 실제로 Score / Pareto 기준으로 확장한 IL 정책은 87.2 / 87.5 PDMS까지 오르지만, GRPO 후에는 85.8 / 86.7로 **떨어진다**. GT-only(90.3)와 제안 방법(91.1)은 GRPO 후 계속 개선된다(Table 2, Fig. 1b).

또한 GRPO의 scalar reward 기반 advantage는 안전 위반과 feasible 영역 내부의 trade-off를 구분하지 못하며, 그룹 내 모든 rollout이 infeasible이면 상대 순위가 안전 방향을 제시하지 못한다.

## 2. 핵심 아이디어

AMPT(Aligned Multi-Trajectory Policy Training)는 세 단계로 구성된다.

1. **PC-MTS (Policy-Compatible Multi-Trajectory Supervision)**: scene별 Pareto front 위에 있고, **동결된 GT-only 정책 rollout과 가까우며**, 작은 교란에도 feasible한 후보만 남겨 fine-tuning → π_initial.
2. **FF-PGRPO (Feasibility-First Pareto GRPO)**: feasible이면서 reference 대비 퇴보하지 않고 그룹 Pareto front에 속한 rollout에만 양의 credit. unsafe rollout은 추가 페널티. 전부 infeasible인 그룹은 safe reference로 약한 diffusion 감독.
3. **APR (Adaptive Pareto-guided Refinement)**: 다중 소스 teacher pool을 매 라운드 최신 정책 기준으로 재평가하고, 보간으로 "검증된 국소 개선 목표"를 만들어 advantage-weighted distillation + policy retention으로 전달.

## 3. 방법론 심층 분석

### 3.1 모델 구조
- ReCogDrive의 VLA 백본 **InternVL3-2B**와 **conditional diffusion planner**를 그대로 재사용 → 공정한 비교를 위해 구조를 고정하고 학습 절차만 바꿈.
- 추론 시 **단일 궤적** 출력, test-time scorer나 후보 reranking 없음.

### 3.2 PC-MTS
- 1차 필터: 기본 safety/compliance 만족 + aggregate score 기준 + 구성요소(EP, TTC, comfort) Pareto front. NAVSIM v1에서는 NC, DAC가 유효해야 하고 DDC는 보호 지표.
- 2차 필터(호환성): 동결된 π_GT에서 여러 rollout을 샘플해 rollout bank를 만들고, 후보와 K-최근접 rollout 간 평균 waypoint 거리로 호환성 측정. 임계값은 held-out rollout으로 **navigation command별로** 보정. 정확한 diffusion likelihood 추정을 우회하는 sample 기반 테스트.
- 3차 필터(국소 feasibility): π_GT rollout의 정상 변동 수준으로 인접 궤적을 만들어 대부분 feasible해야 통과.
- 각 scene은 로그 궤적과 채택된 후보 중 하나를 타깃으로 샘플(후보 수와 무관하게 scene당 1개) → scene별 가중치 불균형 방지.

### 3.3 FF-PGRPO
Advantage 할당 (Eq. 4):
- τ_i ∈ P (허용 Pareto front): max(Â_i, 0)
- feasible이지만 P 밖: min(Â_i, 0)
- infeasible: −λ_unsafe + min(Â_i, 0)

"aggregate 품질은 업데이트 세기를, feasibility·reference·Pareto는 양의 업데이트 허용 여부를 결정"한다는 분리가 핵심. EP 하한(reference 상대)으로 TTC를 높이려고 단순히 감속하는 꼼수를 막는다. 전 그룹이 infeasible이면 양의 예시 없이 위반 심각도에 따른 비양수 credit + safe reference 감독을 주고 가중치를 낮춘다.

### 3.4 APR
- 현재 정책의 단일 예측을 baseline으로, feasible + aggregate 점수 margin 개선 + 보호 지표 비퇴보 조건을 만족하는 후보만 teacher.
- 선택된 teacher를 현재 예측과 여러 비율로 보간해 재평가 → 조건을 만족하는 가장 큰 국소 타깃 채택.
- safety / progress / structure 특화 branch를 따로 학습한 경우 constrained task-vector merge로 통합(세부는 보충자료).
- 매 라운드 teacher 집합을 재구성하여 낡은 감독 제거.

## 4. 데이터 및 학습 설정

- 데이터: NAVSIM 로그 데모 + 다중 소스 후보 궤적 pool(다른 seed 정책, 구조적 궤적 확장기, 특화 정책).
- GT-only 정책 및 multi-trajectory IL 변형: 200 epoch.
- GRPO: scene당 16 rollout, 10 epoch.
- APR: 3 라운드.
- 하드웨어: NVIDIA A800 × 8. 세부 하이퍼파라미터는 보충자료로 미룸.
- 평가 지표: NAVSIM v1 PDMS(NC, DAC, TTC, C, EP), NAVSIM v2 EPDMS(+DDC, TLC, LK, HC, EC).

## 5. 실험 결과

### 5.1 NAVSIM v1 (Table 1, 단일 궤적)

| Method | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---|---|---|
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| Curious-VLA | 98.4 | 96.9 | 97.9 | 98.1 | 88.5 | 90.3 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| DiffusionDriveV2 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| **AMPT** | 98.5 | 98.0 | 96.0 | 100 | 86.7 | **91.4** |

DiffusionDriveV2 대비 +0.2 PDMS. 차이는 작지만 NC/DAC/TTC가 모두 올라가고 EP는 약간 낮아, "더 공격적으로 달려서 얻은 점수"가 아님을 보여준다. ReCogDrive(같은 백본) 대비 +0.6.

### 5.2 NAVSIM v2 (Table 3)

| Method | EP | LK | EC | EPDMS |
|---|---|---|---|---|
| ReCogDrive | 87.1 | 96.6 | 86.5 | 83.6 |
| DiffusionDriveV2 | 88.9 | 96.0 | 91.0 | 85.5 |
| DriveVLA-W0 | 86.4 | 93.2 | 58.9 | 86.1 |
| Drive-JEPA | 88.4 | 97.6 | 84.8 | 87.8 |
| **AMPT** | **89.4** | 91.7 | 87.7 | **89.1** |

Drive-JEPA 대비 +1.3 EPDMS. EP·EC가 이득의 주원인. 다만 LK는 91.7로 비교군 대부분(93~97)보다 낮다.

### 5.3 Failure recovery
ReCogDrive IL 정책에서 5회 샘플 모두 충돌/주행영역 이탈(PDMS 0)한 658개 scene 중, 원래 scalar GRPO는 367개(55.8%), AMPT는 440개(66.9%) 회복 → +73 scene, +11.1%p.

## 6. Ablation 분석

### 6.1 궤적 집합 구성 (Table 2, 모두 동일 scalar GRPO 후속)

| Data | IL | GRPO | Δ |
|---|---|---|---|
| GT | 86.4 | 90.3 | +3.9 |
| Score | 87.2 | 85.8 | −1.4 |
| Pareto | 87.5 | 86.7 | −0.8 |
| PC-MTS | 86.9 | 91.1 | +4.2 |

가장 흥미로운 결과. IL 성능 순위와 GRPO 후 순위가 **역전**된다. 논문의 중심 주장을 가장 직접적으로 뒷받침한다.

### 6.2 단계별 기여 (Table 4)

| PC-MTS | FF-PGRPO | APR | DAC | TTC | EP | PDMS |
|---|---|---|---|---|---|---|
| – | – | – | 94.7 | 94.2 | 80.9 | 86.5 |
| ✓ | – | – | 95.2 | 94.5 | 81.3 | 86.9 |
| – | ✓ | – | 98.0 | 95.8 | 85.93 | 91.0 |
| ✓ | ✓ | – | 98.0 | 96.0 | 86.0 | 91.1 |
| ✓ | ✓ | ✓ | 98.0 | 96.0 | 86.7 | 91.4 |

FF-PGRPO가 대부분(+4.5)을 차지. PC-MTS 추가 효과는 +0.1, APR은 +0.3(EP만 상승, 안전 지표 불변).

### 6.3 APR 라운드 (Table 5)
91.10 → 91.21 → 91.37 → 91.45로 단조 증가(라운드당 +0.08~0.16).

## 7. 관련 연구 비교

- **Curious-VLA**: IL 분포가 좁으면 RL 탐색이 제한됨을 보이고 분포 확장에 집중. AMPT는 "어떤 추가 궤적이 하류 최적화에 유용한가"를 묻는다.
- **DiffusionDriveV2**: anchor 내/간 비교로 GRPO 구조화. AMPT는 anchor 대신 feasibility 계층 + Pareto 적격성으로 양의 credit을 제한.
- **HAD / EvaDrive**: 지표별 또는 다목적 피드백. AMPT는 safety/compliance를 가중합의 한 항이 아닌 **선결 조건**으로 취급.
- **CLOVER**: evaluator 필터 pseudo-expert + scorer 기반 Pareto 타깃, 추론 시 proposal ranking 유지. AMPT는 teacher 유효성을 정책 의존적으로 매 라운드 재평가하고 추론 시 scorer 없이 단일 궤적.
- **ReCogDrive**: 백본·planner 제공자이자 직접 baseline(90.8 → 91.4).

## 8. 강점

- IL 성능과 RL 후 성능의 역전이라는 **명확하고 재현 가능한 관찰**(Table 2)에서 출발한 문제 정의.
- 동일 백본·동일 예산 통제 비교로 학습 절차의 효과를 분리.
- Advantage 할당에서 "세기"와 "허용 여부"를 분리한 설계가 해석 가능하고, EP 하한으로 감속 꼼수를 막는 등 실전적 디테일이 있음.
- 단일 궤적 추론을 유지해 배포 비용이 늘지 않음.
- 658개 hard scene 회복 분석으로 aggregate 점수 외의 안전 측면 근거 제시.

## 9. 한계 및 약점

- **이득 폭이 작다**: v1에서 SOTA 대비 +0.2 PDMS, APR 3라운드 총 +0.35. seed 분산이나 신뢰구간 보고가 없어 유의성 판단이 어렵다.
- **Open-loop 평가만**: NAVSIM은 비반응형 pseudo-simulation. 저자도 closed-loop 확장을 future work로 명시.
- 호환성 테스트가 유한 샘플 기반이고, teacher 검증이 오프라인 planning evaluator(PDM 점수 체계)에 의존 → evaluator 자체에 과적합할 위험.
- Branch별 학습 후 task-vector merge 규칙 등 핵심 세부가 보충자료로 넘어가 본문만으로 재현이 어렵다.
- LK(91.7), HC(97.1) 등 일부 v2 지표는 비교군보다 낮다.
- 코드 공개 언급 없음.

## 10. 재현성 및 실무 관점

- ReCogDrive 코드베이스 위에서 IL/GRPO 루프를 교체하는 형태이므로 구현 진입 장벽은 비교적 낮다.
- 그러나 multi-source teacher pool(다른 seed 정책, 특화 정책, 궤적 확장기)을 만드는 비용과 매 라운드 rollout bank 재생성 비용이 상당하며 총 계산량은 보고되지 않았다.
- 실무적으로는 "RL 전 SFT 데이터 확장 시 현재 정책 rollout과의 거리로 필터링"이라는 PC-MTS 원칙만으로도 다른 VLA+GRPO 파이프라인에 이식 가능한 교훈이 있다.

## 11. 🔥 예상 날카로운 질문

| # | 질문 | 예상 답변 / 논점 |
|---|---|---|
| 1 | +0.2 PDMS가 seed 노이즈보다 큰가? | 논문에 분산 보고 없음. 약점으로 지적할 부분 |
| 2 | PC-MTS 단독 효과가 +0.4(IL) 수준인데 필요성은? | Table 2에서 GRPO 후 91.1 vs Score 85.8 — 가치는 "초기화 품질"에 있다는 게 저자 주장 |
| 3 | 호환성 임계값을 navigation command별로 보정한 이유는? | 좌/우회전·직진별 rollout 분산 차이가 커서 단일 임계값이 부적절 |
| 4 | FF-PGRPO에서 feasible 판정이 evaluator에 의존 → reward hacking은? | reference 상대 EP 하한, DDC guard로 일부 방지. closed-loop 검증 부재 |
| 5 | APR의 teacher pool에 특화 정책을 쓰는 것은 사실상 앙상블 distillation 아닌가? | 맞다고 볼 수 있으나 추론 시 단일 모델·단일 궤적 유지가 차별점 |
| 6 | 658 failure scene이 ReCogDrive IL 기준으로 선정 → 선택 편향? | 같은 초기 정책 계열이라 공정하지만 다른 모델에 일반화되는지는 미확인 |

## 12. 총평

AMPT는 driving VLA의 IL→GRPO 파이프라인에서 "더 좋은 SFT 데이터가 더 좋은 RL 초기화는 아니다"라는 교훈을 Table 2로 설득력 있게 보여준 논문이다. 제안된 세 모듈 중 실질적 성능 대부분은 feasibility-first Pareto GRPO에서 나오고, PC-MTS와 APR은 각각 초기화 안정성과 잔여 효율 회수 역할을 한다. NAVSIM v1 91.4 / v2 89.1이라는 수치 자체는 경쟁력 있지만 v1 기준 SOTA 대비 폭이 작고 분산 보고와 closed-loop 검증이 없어, 방법론적 교훈 쪽이 수치보다 더 가치 있는 기여로 평가된다.

<!-- VERIFIED: pdf -->
