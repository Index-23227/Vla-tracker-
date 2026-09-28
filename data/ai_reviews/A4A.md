# A4A: Cross-Embodiment Transfer of Action-Oriented 4D Affordances from Human Demonstrations

> **한 줄 요약**: 인간 시연에서 "어디를 잡을지"가 아니라 **상호작용 관련 3D 점들이 앞으로 어떻게 움직일지(4D 궤적)**를 추출해, VLA의 행동 생성 모듈이 먼저 이 점 궤적을 예측하도록 사전학습한 뒤 로봇 행동으로 fine-tune. Octo/OpenVLA/OpenVLA-OFT/π0/π0.5 전부에서 LIBERO-Object 개선(Octo +29.0pp), RLBench 4과제 평균 39.0→61.0%, 실제 로봇 평균 56.7→72.5%.

- **arXiv**: 2609.05892v1 (2026-09-05, cs.RO)
- **소속**: Shanghai Jiao Tong University, Rutgers University, NTU, HKUST(GZ), Shanghai AI Laboratory
- **정책 백본**: 5개 VLA 계열 + 자체 RLBench 정책(Qwen3.5-4B)
- **프로젝트**: https://ru-arcl.github.io/a4a/ (코드 공개 여부 명시 없음)

---

## 1. 배경 및 동기

인간 비디오는 풍부하지만, 어떤 신호가 로봇 제어로 이전 가능한지는 불분명하다. 인간 팔 궤적은 embodiment에 묶여 있고, 기존 affordance(2D mask, 3D region, contact point, actionability score)는 **where**만 알려준다. 서랍 손잡이 위치는 당기는 변위를 결정하지 않고, 컵 가장자리는 따르는 동작을 정의하지 않는다. 저자들은 전이 가능한 신호는 **공간적이 아니라 조작적(operational)**이어야 한다고 주장한다.

## 2. 핵심 전제: 기하적 대응

그리퍼·도구·쥔 물체에 강체로 붙은 점은 end-effector 자세 G_t에 의해 완전히 결정되며, 변위 A_t = G_{t+1}G_t^{-1} ∈ SE(3)가 곧 점들의 움직임을 만든다. 인간 영상 속 상호작용 관련 점들도 짧은 horizon에서는 (준)강체 SE(3) 변환으로 근사 가능하므로, **4D 점 움직임과 로봇 행동은 같은 조작 전이의 두 기하적 표현**이다.

## 3. 4D Affordance 데이터셋

- HOI4D, EPIC-KITCHENS + 자체 RealSense RGB-D 약 30K clip, 총 80K+ clip.
- 자체 데이터로 pour, cut, hang, sweep, lid removal 등 기존 데이터에 드문 동작 보강.
- 파이프라인: LingBot-Depth로 깊이 보정 → GroundingDINO로 조작 대상/도구/접촉 부위 검출 → SAM2 mask → CoTracker3로 점 추적 → 깊이·내참수로 3D 역투영. 무효 깊이, 장기 가림, 불안정 추적, 이상치 제거.
- 표현: S = (I_{0:H}, l, Q^{0:H}_int).
- 로봇 평가와 사람 데이터는 물체·장면을 공유하지 않는다(부록 A.4).

## 4. 방법: Affordance-to-Action 표현 전이

**Prediction-space substitution**: 각 base VLA의 vision-language stack과 행동 생성 내부 모듈 F_θ(LLM decoder, block-causal Transformer, action expert 등)를 그대로 쓴다.
- **Stage I**: 로봇 state 대신 query 점 좌표를 point projector P_Q로 넣고, 행동 대신 미래 점 궤적을 H_Q로 예측. 목표는 constant-velocity 외삽 대비 residual. **목적함수는 base 정책의 native 계열 그대로**(OpenVLA: 점 토큰 CE, OFT: L1/MSE, Octo: DDPM, π0/π0.5: flow matching).
- **Stage II**: P_Q→P_S(proprio), H_Q→H_A(행동)로 교체하고 공유 모듈을 초기화로 두고 로봇 데모로 jointly fine-tune.
- π0.5는 저수준 action expert만 전이, 고수준 semantic 경로는 고정. OFT는 rank-32 LoRA.
- RLBench 정책: Qwen3.5-4B causal Transformer(hidden 2048) + 2층 MLP projector/head, context 16.

## 5. 실험 설정

- **LIBERO-Object** 10과제(과제당 50 데모, 50 에피소드, 최대 280 step): 5개 VLA 계열 base vs A4A. 4D 데이터가 물체 상호작용 중심이라 이 suite만 선택.
- **RLBench** 4과제(meat off grill, sweep to dustpan, turn tap, slide block): 변이당 100 데모, 25 rollout. 3D/4D 입력 사용 방법(ManiGaussian, ARM4R 등)과 비교.
- **실제 로봇** AgileX Piper: 전자레인지 열기, 컵 올리기, 복숭아 자르기, 밥짓기 3하위 skill. skill당 100 데모, 10 rollout. A4A 초기화만 변수.

## 6. 주요 결과 — LIBERO-Object (Table 1)

| 정책 | Base | A4A | Δ |
|---|---|---|---|
| Octo (diffusion) | 28.2 | 57.2 | +29.0 |
| OpenVLA (AR) | 66.4 | 76.4 | +10.0 |
| OpenVLA-OFT (regression) | 98.0 | 98.6 | +0.6 |
| π0 (flow) | 77.8 | 87.8 | +10.0 |
| π0.5 (hier. flow) | 94.0 | 96.0 | +2.0 |

모든 행동 생성 패러다임에서 평균이 오른다. 기준선이 높은 OFT/π0.5는 headroom이 작다.

## 7. 주요 결과 — RLBench와 실제 로봇

**RLBench (Table 2)**: Ours w/ pretrain 88.0 / 60.0 / 68.0 / 28.0, **평균 61.0**(w/o pretrain 39.0, ManiGaussian 51.0, ARM4R 41.0). RGB만으로 3D 입력 방법을 앞선다.

**아키텍처 통제 비교 (Figure 4, 본문)**: 동일 ARM4R 구조에서 generic scene-wide 4D 사전학습 41.0% → 본 affordance 사전학습 60.0%, turn tap 28.0→60.0%.

**실제 로봇 (Table 3)**: Octo 36.7→60.0, OpenVLA 48.3→68.3, OFT 71.7→80.0, π0.5 70.0→81.7(%). 전체 평균 56.7→72.5. 회전이 큰 pour rice/water에서 개선이 두드러진다.

## 8. Ablation

- 사전학습 유무(Table 2 w/o vs w/ pretrain): +22.0pp.
- 사전학습 데이터 종류(Figure 4): interaction-centric 4D > scene-wide 4D, 동일 구조.
- 점 선택 범위, residual 파라미터화, 데이터 규모, 인하우스 데이터 기여 등에 대한 ablation은 없다.

## 9. 강점

- "어떻게 움직일지"를 표현하는 4D affordance라는 명료한 개념과 SE(3) 기반 정당화.
- 5개 서로 다른 행동 패러다임에 **native 목적함수를 유지**하며 적용 — 범용성이 설득력 있다.
- 동일 구조 통제 비교로 데이터/감독 신호의 효과를 분리.
- 실제 로봇에서 4개 정책 모두 일관된 개선.

## 10. 약점 및 한계

- LIBERO는 Object suite 하나, RLBench는 4과제로 **표준 벤치마크 전체 성능이 아니다**.
- LIBERO base 정책 일부는 공식 체크포인트/LeRobot 포트를 평가한 것이라 A4A 쪽과 학습 조건이 완전히 같은지 불분명.
- RLBench 비교 대상 수치는 대부분 선행 연구 인용.
- 실제 로봇 10 trial/skill로 통계적 신뢰도가 낮다.
- 큰 비강체 변형, 반복 재파지(병뚜껑 여러 번 돌리기), 양팔 협응에는 한계를 저자들이 인정.
- 코드·데이터 공개 여부 불명.

## 11. 재현 및 확장 아이디어

- LIBERO 4 suite 전체, RoboTwin 등 양팔 벤치로 확장.
- 로봇 궤적에서 만든 4D affordance로 추가 중간 단계 학습(저자 제안).
- 사전학습 데이터 규모 scaling 곡선, 인하우스 30K의 기여 분리.
- 4D 점 예측을 fine-tune 단계에서 보조 손실로 유지하는 co-training 변형.

## 12. 총평

인간 영상에서 embodiment 중립적인 "기하 전이"만 뽑아 VLA 행동 모듈의 사전학습 목표로 삼는 아이디어가 깔끔하고, 5개 패러다임·실로봇에서 일관된 이득을 보인다. 다만 평가가 부분 벤치마크(LIBERO-Object, RLBench 4과제)에 머물러 표준 리더보드 수준 주장은 어렵다.

**한 문장 요약**: 사람 손이 아니라 물체 점이 어떻게 움직이는지를 먼저 배우게 하면, 어떤 VLA든 로봇 행동을 더 잘 배운다.

<!-- VERIFIED: pdf -->
