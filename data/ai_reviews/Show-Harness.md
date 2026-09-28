# Show-Harness: Just a VLM Agent Can Play Robots

> **한 줄 요약**: 로봇 제어를 "MV_FWD, GRASP, ROTATE_CW" 같은 이산 semantic action unit으로 재정의하고, embodiment별 interpreter가 이를 2~4 cm 단위 동작으로 결정적으로 grounding하는 embodied harness. Frontier VLM(Gemini-3.1 Pro)은 zero-shot으로, 소형 Qwen3.5-2B는 LoRA 2시간 fine-tune으로 로봇을 조작하며, 실로봇 cross-task 86.0%(FT)로 π0.5(39.0%)·GR00T(35.0%)를 크게 앞섬.

---

## 1. 배경 및 동기

- VLA는 VLM을 embodiment-specific 연속 action regression으로 끌어내려 semantic 지식을 불투명한 pixel→actuation 매핑으로 붕괴시키고, 태스크·환경·embodiment마다 재적응이 필요.
- 계층형/code-as-policy 시스템은 VLM이 subtask·프로그램을 출력하지만 실제 물리 제어는 하위 controller가 담당 → semantic intent와 물리 실행 사이 연결이 약화.
- 저자 주장: VLM이 이해할 수 있으면서도 직접 물리 제어가 가능할 만큼 세밀한 **action interface**가 필요.

## 2. 방법론 심층 분석

### 2.1 Semantic action unit
- 각 unit은 한 방향 단일 이동 step 또는 gripper 동작. VLM은 unit을 선택하고, embodiment-specific interpreter가 이를 작은 bounded motion으로 결정적 변환(안전 한계·step size 포함).
- 모든 semantic 결정이 즉시 물리 세계에 반영되고 매 unit 후 실행 feedback이 돌아옴.

### 2.2 Harness plugin (perceive–reason–act)
- Perception: Multi-View Guidance, Visual Prompt. Reasoning: Situated Planning, subtask 관리. Action: Adaptive Step(wrist view에 타깃이 보이면 2 cm, 아니면 4 cm), Failure Recovery(grasp 실패 감지 후 rollback).

### 2.3 두 가지 모드
- **ZS 모드**: frontier VLM을 fine-tuning 없이 controller로 사용.
- **FT 모드**: 소형 open VLM이 native vocabulary로 unit을 직접 예측, 토큰 cross-entropy 최소화. 최소 context(지시문, multi-view 관측, 짧은 action history)만 사용. 별도 action head·special token 없음.

### 2.4 GUMI
- 동일 semantic unit을 GUI 버튼/키보드에 매핑해 사람·computer-use agent·VLM이 같은 인터페이스로 데모를 수집. 전용 teleop 하드웨어 불필요, 원격 수집 가능. 각 rollout은 low-level 궤적도 함께 기록해 연속 제어 정책 학습에도 재사용.

## 3. 데이터 전략

- 실로봇 164 episodes(7.8K decision steps, 2 cm step): Franka 101 ep(5.0K), AgileX 단일팔 63 ep(2.8K); 19 tasks. 이 중 19개 Franka ep은 grasp recovery 전용.
- Sim-to-real용 230 simulated episodes(13.5K steps): ManiSkill 100, RoboLab 130(12 tasks).
- Teddy bear, chess piece는 fine-tuning 데이터에서 제외(OOD).
- 데이터셋은 HuggingFace(showlab/Show-Harness-Data)로 공개.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| FT backbone | Qwen3.5-2B (기본), 0.8B/9B 등 스케일 ablation |
| 적응 방식 | rank-64 LoRA, LM linear layer 전체, vision encoder·projector frozen (~3% 파라미터) |
| 학습 | 40 epochs, lr 1e-4 cosine, warmup 0.1, bf16, 256×256, batch 32 |
| 학습 비용 | 단일 H200에서 2시간 미만 (24GB급 GPU로 가능) |
| ZS 모델 | Gemini-3.1 Pro (medium thinking) |
| 서빙 | 로컬 모델 RTX 5090 1장 |

## 5. 실험 설계 및 평가 프로토콜

- 하드웨어: Franka Research 3(외부 D435 + wrist D405), AgileX 양팔(Orbbec 3대).
- 10개 태스크 = 5 objects(block, banana, tennis ball, teddy, chess) × 2 receptacles(plate, bowl). 태스크당 10 trials, 50 step 상한, timeout은 실패.
- 세 가지 일반화: cross-task, cross-environment(background/lighting/viewpoint/distractor/sim-to-real, 20 trials), cross-embodiment(Franka ↔ AgileX, 5 plate tasks × 10 trials).
- Baselines: VLA(π0.5, GR00T — 같은 데모를 연속 EE 궤적으로 변환해 fine-tune), VLA-centric agent(Harness VLA, Goal-VLA), code-as-policy(CaP-X, RATS).

## 6. 실험 결과 심층 분석 (PDF Table 2 직접 인용)

| Setting | π0.5 | GR00T | H-VLA | G-VLA | CaP-X | RATS | ZS | FT |
|---|---|---|---|---|---|---|---|---|
| Cross-Task Avg (%) | 39.0 | 35.0 | 50.0 | 13.0 | 44.0 | 57.0 | 89.0 | **86.0** |
| Cross-Env Avg (%) | 40.0 | 34.0 | 63.8 | 15.0 | 52.5 | 65.0 | 100.0 | **88.0** |
| Cross-Embodiment Avg (%) | 41.0 | 36.0 | 49.0 | 11.0 | 43.0 | 52.0 | 93.0 | **87.0** |

- FT cross-task: Block/Banana→Plate 10/10, Tennis→Plate 8/10, unseen Teddy→Plate 8/10, unseen Chess→Plate 9/10, Banana→Bowl 7/10.
- Sim-to-real: FT 13/20, π0.5·GR00T 0/20 — 시뮬레이션 데모만으로 학습해도 semantic 인터페이스 덕분에 전이됨.
- Cross-embodiment FT: Franka 45/50, AgileX 42/50.

## 7. Ablation 분석

- Test-time step size, 회전 unit 추가, 양팔 per-arm/joint control 등 physical adaptability 분석(Fig. 6~7).
- Backbone 스케일(Fig. 8): Qwen3.5(0.8B/2B/9B)와 InternVL 계열 여러 크기를 fine-tune해 비교하고, Qwen3.5 세 용량에 대해 정밀도(block stacking·peg insertion)·반응성(tennis ball)·지연의 trade-off를 분석.
- Plugin leave-one-out(ZS 모드): Multi-View Guidance가 소형 물체 정밀 정렬에서 특히 중요.
- (세부 수치는 주로 figure로 제시)

## 8. 관련 연구 비교

- VLA(π0.5, GR00T, OpenVLA): 연속 action regression/생성. Show-Harness는 action을 VLM vocabulary 내 이산 semantic token으로 표현.
- RT-2류 discretized action token과 달리 bin index가 아닌 의미 있는 방향·gripper 명령을 사용.
- Harness VLA/Goal-VLA: frozen VLA를 실행기로 둔 agentic layer. CaP-X/RATS: 코드로 API 호출. Show-Harness는 VLM이 직접 미세 물리 결정을 담당.

## 9. 한계 및 미해결 문제

- 2 cm 단위 이산 이동은 연속·동적·고속 조작(접촉 풍부, 힘 제어)에는 부적합할 수 있음; 50 step 상한 내 pick-and-place 위주 평가.
- 태스크당 10 trials로 통계적 신뢰도 제한, 표준 시뮬레이션 벤치 결과 없음.
- π0.5/GR00T는 소량(164 ep) 데이터로 fine-tune되어 baseline이 과소평가됐을 가능성.
- ZS 모드는 API 지연·비용 의존.

## 10. 총평

"적절한 인터페이스만 있으면 VLM이 곧 정책이다"라는 주장을 실로봇에서 설득력 있게 보여준 작업. 특히 2B 모델을 2시간 LoRA로 학습해 sim-to-real까지 달성한 점과 GUMI를 통한 저비용 데이터 수집 루프가 실용적이다. 다만 action 해상도가 거칠어 dexterous 영역으로의 확장성은 열려 있다.

## 11. 🔥 예상 날카로운 질문 모음

1. 2 cm step은 정밀 삽입·접촉 태스크에서 어디까지 유효한가? 적응적 step size를 모델이 직접 선택하게 하면?
2. 한 에피소드 수십 step × VLM 호출 지연 — 실시간성은?
3. π0.5/GR00T를 더 많은 데이터 또는 원래 연속 teleop 데이터로 학습했을 때도 격차가 유지되는가?
4. LIBERO 등 표준 벤치에서 semantic unit 정책의 성능은?
5. 이산 unit이 동역학(굴러가는 공 등)에 대한 반응성을 얼마나 제한하는가?

## 12. 재현성 및 후속 연구 제안

- 코드·모델·데이터셋 공개(프로젝트 페이지, HF 데이터셋). 학습 하이퍼파라미터와 프롬프트(Appendix 7.4) 상세.
- 후속: (1) 연속 파라미터를 가진 hybrid semantic unit, (2) RL로 FT 모드 강화, (3) dexterous hand·모바일 매니퓰레이션으로 확장, (4) 표준 벤치 평가.

<!-- VERIFIED: pdf -->
