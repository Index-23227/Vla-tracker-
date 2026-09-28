# DREAM: Deployment-Time Demonstration Generation via Real-to-Sim for Scalable Policy Adaptation

> arXiv: [2608.29078](https://arxiv.org/abs/2608.29078) · The University of Tokyo · 2026-08-29
> 한 줄 요약: 작업공간을 찍은 짧은 영상과 언어 지시 하나로 3DGS 디지털 트윈 + TAMP + MimicGen식 증강 + splat 렌더링을 거쳐 VLA fine-tuning 데이터를 자동 생성하고, 이 데이터만으로 fine-tune한 π0.5가 실제 xArm7에서 teleop 100개로 학습한 경우보다 높은 성공률(93.3% vs 86.7%)을 보인 real-to-sim 데이터 생성 프레임워크.

## 1. 배경 및 동기

사전학습 VLA는 새 작업공간(객체 인스턴스, 포즈, 잡동사니, 지지면, 도달성)에서 바로 쓰기에 성능이 부족한 경우가 많고, 이를 개선하려면 그 작업공간의 action-labeled 데이터가 필요하다. Teleoperation은 품질은 높지만 작업공간·배치·과제마다 운영자 시간, 로봇 점유, 리셋, 안전 감시 비용이 든다.

기존 데이터 생성 방법은 다음 중 하나 이상을 여전히 요구한다.
- MimicGen / SkillMimicGen / DemoGen 계열: **seed 데모** 필요
- RialTo, RoboSplat, R2R2R: real-to-sim이지만 seed 데모나 데모 영상 필요
- Scaling Up / OPTIMUS: 수작업 시뮬레이션 환경

DREAM이 겨냥하는 설정은 "사용자가 작업공간을 찍고, 언어로 과제를 말하면, 과제 시연 없이 fine-tuning 데이터가 나오는 것"이다(Table I에서 Teleop-free / Demo-free / Real-to-Sim / Language task / Long-horizon 모두 ✓인 유일한 방법으로 제시).

## 2. 핵심 아이디어

**재구성된 작업공간을 "실행 가능한 감독의 원천"으로 쓴다.** 핵심 설계 원칙은 시각적 사실성과 물리적 타당성의 분리이다.
- 재구성(Gaussian splat)은 **렌더링에만** 사용
- 물리·계획은 로봇 + 이동 객체 + 지지면만 있는 **간단한 기하 세계**에서 수행
- 따라서 watertight scene mesh 없이도 시각적으로 작업공간에 grounded되고 기하적으로 실행 가능한 image–action 데이터를 얻는다.

## 3. 방법론 심층 분석

### 3.1 디지털 트윈 재구성
- 핸드헬드 카메라 영상 → COLMAP SfM → gsplat으로 3DGS 최적화(보정된 intrinsics 사용, world-space normalization 비활성화).
- **로봇 기준 좌표계 정렬**: 마커 없이 로봇 자신의 외형으로 SIM(3) 추정. 촬영 시 관절각으로 FK 포즈한 로봇 모델 점군 vs SAM3 텍스트 분할 + multi-view voting으로 얻은 재구성 내 로봇 점군 → FPFH + RANSAC 조정합 → scaled ICP(Umeyama) 정밀화.
- **자산 분해**: 로봇 근처 Gaussian 제거로 로봇 없는 배경, 링크별 로봇 splat(FK로 재포즈 가능), 객체는 사진 몇 장에서 TRELLIS로 생성한 appearance splat + 충돌 메쉬(실측 크기 보정).
- IsaacLab 환경은 scene description 메타데이터만으로 생성(과제별 코드 없음). fixed/wrist 카메라는 배포와 동일 보정값 사용.

### 3.2 TAMP 기반 데모 합성
- **과제 명세**: 사전 정의된 predicate 어휘(이름, arity, 인자 타입, 포즈 기반 평가 함수)를 GPT-4o-mini가 **선택·인스턴스화만** 함. 예: "put the fruits on the plate" → On(banana, plate), On(peach, plate). 문법·arity·객체 존재 검사로 오류 시 fail-closed.
- 성공 기준도 13개 사전 정의 checker로 구성된 staged rubric(reach, grasp, transport, place)으로 생성.
- 명세 오류는 잘못된 action label이 아니라 "데모 누락"으로만 나타나게 설계된 점이 중요.
- **계획**: PDDL 연산자(MoveFree, MoveHolding, Pick, Place, Push) skeleton을 BFS로 열거 → cuTAMP 방식 GPU 병렬 최적화(1000 particle, GraspGen 파지 샘플러 + 휴리스틱 배치 샘플러) → cuRobo로 구간별 충돌 회피 경로.
- 시뮬레이터에서 open-loop 실행, 파지 실패 감지 시 중단, rubric 통과 에피소드만 기록. 모든 프레임에 symbolic 연산자 주석.

### 3.3 증강
- MimicGen 방식이지만 subtask 경계를 신호 휴리스틱이 아닌 **plan에서 직접** 읽음.
- 새 시도마다 객체 포즈 재샘플링 → subtask별 최근접 source 데모 선택 → end-effector 구간을 새 객체 좌표계로 강체 변환 → bridge motion 보간.
- rubric 종료 조건 + 오프라인 물리 타당성 필터(정적 객체 교란, 파지 전 충돌, 지지면 높이 끌기, 관절 과속) 통과 시에만 채택.

### 3.4 렌더링과 fine-tuning
- 기록된 상태를 splat 장면으로 재생해 배경 + FK 포즈 로봇 링크 splat + 객체 splat을 하나의 Gaussian 집합으로 합쳐 래스터화. 가림은 depth-sorted alpha blending으로 자동 처리.
- 640×480 RGB(fixed + wrist), 로봇 상태, action, 언어 지시를 LeRobot 포맷으로 저장.
- 이 데이터로 **π0.5와 SmolVLA를 사전학습 가중치에서 supervised imitation fine-tuning**. 배포 시 TAMP는 사용하지 않음.

## 4. 실험 설정

- 로봇: xArm7, Intel RealSense D435i 2대(고정 외부 + 손목), ArUco로 외부 보정.
- 과제: **BlockIntoBowl**(단일 pick-and-place), **FruitPacking**(바나나, 복숭아를 순서대로 접시에 — 다단계).
- 정책: SmolVLA, π0.5(fine-tune), Diffusion Policy(상관분석용, scratch 학습).
- 데이터 비교: human teleop(SO-101 leader → xArm7 follower, 10/50/100개) vs DREAM(100/500/1000개).
- 지표: 과제당 15 trial 성공률(held-out 객체 배치), 생성기 통계, 인간 노력 시간 vs 자동 생성 시간.
- 하드웨어: 정책 학습 NVIDIA GB200, 그 외 RTX A6000(48GB) × 3.

## 5. 실험 결과

### 5.1 Experiment 1 — Sim-to-Real 상관 (RQ1)
teleop 100개로 학습한 세 아키텍처 체크포인트를 재구성 환경과 실제 로봇에서 같은 조건으로 평가.
- BlockIntoBowl: Pearson r = 0.98, MMRV = 0.000
- FruitPacking: Pearson r = 0.99, MMRV = 0.000
- 시뮬레이션 성공률이 실제보다 7~14%p 낮음(비관적), 단 SmolVLA/FruitPacking은 낮은 구간에서 일치.
→ 절대값은 비관적이지만 **정책 간 순위는 완벽히 보존**.

### 5.2 Experiment 2 — 데이터 수집 시간 대비 스케일링 (RQ2)

| 과제 | 정책 | DREAM 1000 | Teleop 100 |
|---|---|---|---|
| BlockIntoBowl | π0.5 | **93.3** | 86.7 |
| BlockIntoBowl | SmolVLA | **80.0** | 73.3 |
| FruitPacking | π0.5 | **86.7** | 80.0 |
| FruitPacking | SmolVLA | 6.7 | **26.7** |

- π0.5는 두 과제 모두 DREAM 데이터만으로 teleop을 넘어섬.
- SmolVLA는 FruitPacking에서 6.7%에 정체, 500개 이상으로 늘려도 개선 없음.

### 5.3 비용 분석
- DREAM 고정 비용: 첫 데모 전 36분(BlockIntoBowl) / 95분(FruitPacking). 이후 데모당 수 초.
- 손익분기: 73개 / 168개 데모 이상부터 DREAM이 wall-clock 기준 더 저렴.
- 고정 비용 대부분은 TAMP: 50개 source 데모에 25.8분 / 84.6분. FruitPacking은 에피소드 2.7배 길고 planner 성공률 100% → 62%.
- 증강 채택률 33.0% / 32.6% — 비용은 시뮬레이션 속도보다 rejection이 지배.

## 6. 분석 및 해석

- **데모 1개의 가치는 teleop보다 낮다**: 같은 크기에서는 DREAM 데이터가 불리하다. 증강 궤적에 잦은 손목 회전과 무작위화에서 오는 약간의 중복 동작이 섞이기 때문. teleop 수를 크게 넘도록 늘려야 역전된다.
- 그럼에도 인간 기여는 "스캔 1회, 보정 1회, 지시 1개"로 데이터셋 크기와 무관하게 고정 → 정책이 충분히 좋아질 때까지 기계 시간만으로 밀어붙일 수 있다는 것이 확장성 주장의 핵심.
- SmolVLA의 FruitPacking 실패는 모델 용량/장기 과제 한계와 합성 데이터 품질의 상호작용으로 보이며, 저자는 원인을 깊게 분석하지 않는다.

## 7. 관련 연구 비교

| 방법 | 핵심 차이 |
|---|---|
| MimicGen / SkillMimicGen / DexMimicGen | seed 데모 필요. DREAM은 TAMP로 source를 스스로 생성 후 MimicGen식 증강 |
| OPTIMUS / Scaling Up | 계획 기반이지만 수작업 시뮬레이션 환경, 시각 sim-to-real gap 미해결 |
| RialTo | 디지털 트윈 + RL, seed 데모 필요 |
| R2R2R | 사람 데모 영상 1개에서 객체 궤적 추적, 동역학 시뮬 없음 |
| RoboSplat / RoboSimGS / SplatSim | 3DGS 렌더링 기반 데이터/정책 전이. DREAM은 여기에 언어→symbolic goal + TAMP를 결합 |
| GSWorld | 3DGS + 물리엔진 closed-loop 시뮬. 언어 과제 명세 없음 |
| SIMPLER | DREAM이 채택한 paired sim/real 평가 프로토콜(MMRV) 제공 |

## 8. 강점

- **완전한 demo-free / teleop-free 파이프라인**을 실제 로봇에서 end-to-end로 검증.
- LLM의 역할을 사전 정의된 predicate 선택으로 제한해 hallucination이 잘못된 라벨로 이어지지 않도록 한 안전한 설계.
- 로봇 외형 기반 자동 SIM(3) 정렬로 마커·수동 대응점 불필요.
- 렌더링/물리 분리로 watertight mesh 없이 사실적 관측 확보.
- 성능뿐 아니라 **인간 시간 vs 기계 시간 비용 모델**과 손익분기점을 정량 제시한 점이 실무적으로 유용.
- 재구성 환경을 평가 도구로도 쓸 수 있음을 MMRV 0으로 보여줌.

## 9. 한계 및 약점

- **과제가 2개, trial이 15회**뿐: 93.3 vs 86.7은 15회 중 1회 차이(14/15 vs 13/15)로 통계적 유의성이 약하다.
- 정적 작업공간, 강체 객체, 탁상 조작에 한정(저자 명시). 변형체·관절체·접촉 풍부 과제 미검증.
- 카메라 외부 파라미터 사전 보정과 촬영 시 관절각 기록이 필요 — "영상만으로"라는 표현보다 전제 조건이 있다.
- SmolVLA FruitPacking에서 teleop보다 크게 뒤짐(6.7 vs 26.7) → 모델에 따라 합성 데이터가 해로울 수 있음. 원인 분석 부족.
- teleop과 DREAM 데이터 혼합 학습, 또는 DREAM + 소량 teleop 조합 실험이 없다.
- 새 정책 구조를 제안하지 않으므로 기여는 데이터 파이프라인에 있다. 코드 공개 언급 없음.

## 10. 재현성 및 실무 관점

- 구성 요소(COLMAP, gsplat, SAM3, TRELLIS, IsaacLab, cuTAMP, GraspGen, cuRobo, LeRobot)가 대부분 공개 도구라 조립 재현은 가능하나, predicate 어휘·13개 checker·물리 타당성 필터 세부가 공개되지 않으면 재현 부담이 크다.
- FruitPacking에서 TAMP 고정비 95분은 과제 horizon이 길어질수록 급증할 것으로 보여, 긴 조립 과제로의 확장이 병목.
- 실무적으로는 "새 현장 배치 시 1시간 남짓 투자로 수백~수천 개 현장 맞춤 데모"라는 워크플로가 매력적이며, π0.5처럼 강한 사전학습 VLA와 결합할 때 효과가 분명하다.

## 11. 🔥 예상 날카로운 질문

| # | 질문 | 예상 답변 / 논점 |
|---|---|---|
| 1 | 15 trial에서 1회 차이를 "teleop 초과"라고 할 수 있나? | 통계적으로 약함. 논문의 더 강한 주장은 비용 곡선 쪽 |
| 2 | 왜 teleop은 100개까지만? teleop 1000개와 비교하면? | 비용 대비 곡선이 목적이라 동일 시간축 비교. 동일 개수 비교는 teleop이 우위임을 저자도 인정 |
| 3 | SmolVLA FruitPacking 정체 원인은? | 증강 궤적의 손목 회전/중복 동작 + 작은 모델 용량 가능성, 논문 미분석 |
| 4 | LLM이 predicate를 잘못 고르면? | fail-closed 설계로 데모 누락만 발생, 잘못된 라벨은 생기지 않음 |
| 5 | 시뮬레이션이 7~14%p 비관적인 이유는? | splat 렌더링과 실제 관측 간 차이, 단순화된 물리 세계 등이 추정 원인 |
| 6 | 동적 장면·변형체에는? | 현재 가정 밖. 저자 future work |

## 12. 총평

DREAM은 새로운 VLA 구조가 아니라 **배포 시점의 현장 맞춤 fine-tuning 데이터를 사람 시연 없이 만드는 시스템**이며, 그 데이터로 π0.5와 SmolVLA를 실제로 fine-tune하여 실로봇 성능을 보고한다. 3DGS 재구성, LLM 기반 symbolic 과제 명세, TAMP, MimicGen식 증강, splat 렌더링을 설계 원칙(렌더링과 물리 분리, LLM은 선택만)에 맞게 잘 엮었고, sim-to-real 순위 보존(MMRV 0)과 비용 손익분기 분석이 설득력 있다. 다만 과제 2개·15 trial이라는 소규모 평가와 SmolVLA에서의 역전 실패는 일반화 주장을 제한한다. "데모 1개의 가치는 낮지만 기계 시간으로 무한히 늘릴 수 있다"는 메시지가 이 논문의 핵심 교훈이다.

<!-- VERIFIED: pdf -->
