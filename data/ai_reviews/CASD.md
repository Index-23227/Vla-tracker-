# CASD: Chunk-Aligned Semantic Distillation for Multi-Stage Robot Manipulation

> **한 줄 요약**: action chunk 하나가 여러 조작 단계에 걸치는데 첫 step 라벨은 현재 단계만 설명한다는 문제를, 오프라인 VLM이 나눈 단계 설명을 **chunk 내 점유율로 가중 평균한 semantic teacher**로 정의하고, 이를 현재 입력만으로 예측하는 generator를 증류·고정한 뒤 World Action Model(Fast-WAM, DreamZero)을 그 예측으로 조건화해 해결. Fast-WAM-IDM+CASD LIBERO 98.9%, Joint+CASD RoboTwin 2.0 93.0%, DreamZero+CASD MolmoSpaces 47.9%.

- **arXiv**: 2609.08638v1 (2026-09-08, cs.RO)
- **소속**: Ant Group
- **정책 백본**: Fast-WAM (Wan2.2-TI2V-5B; Uncond / Joint / IDM), DreamZero (Wan2.1-I2V-14B)
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

VLA와 World Action Model(WAM)은 대부분 action chunk를 예측한다. 다단계 과제(예: 그릇을 서랍에 넣고 닫기)에서는 하나의 32-step chunk 안에 "운반 → 놓기 → 손잡이로 이동"이 섞일 수 있다. 첫 step 기준 라벨(예: transport)은 곧 도달할 release를 누락하고, 다음 단계 라벨(close)은 끝나지 않은 상호작용을 건너뛴다. 겹치는 단계 목록만으로도 각 단계가 chunk를 얼마나 차지하는지는 알 수 없다. 저자들은 **semantic supervision을 action 예측과 같은 시간 척도(chunk horizon)에 정렬**해야 한다고 주장한다.

## 2. 문제 정의

시연 {(o_t, s_t, a_t)}와 지시 ℓ가 주어질 때, 정책은 x_t = (o_t, s_t, ℓ)로부터 H-step chunk를 예측한다. 단계 주석 {([b_k, e_k), u_k)}에서, 시점 t의 유효 chunk C_t 중 단계 k가 차지하는 비율 w_{t,k} = |{τ∈C_t : g_τ = k}| / |C_t|를 정의한다(합 1). 이 점유율은 시연의 미래를 사용하므로 배포 시에는 쓸 수 없는 **privileged information**이다.

## 3. 방법: Chunk-Aligned Semantic Teacher

- **오프라인 주석**: GPT-5.4에 12프레임 contact sheet, 지시문, gripper 개폐·속도 peak·정지 등 action 기반 hint를 주고, 순서 있는 비중첩 단계와 자유형 설명(5–12단어)을 받는다. 결정적 후처리가 경계를 gripper 이벤트에 스냅하고 Place–Release를 action 신호로 재구성한다.
- **Teacher**: 각 단계 설명을 frozen UMT5로 인코딩, 토큰열을 4개 연속 구간으로 나눠 mean-pool(N=4 slot), 중심화 후 seed-0 랜덤 직교 사영(128차원). T_t = R(Σ_k w_{t,k} h̄_{k,n} − μ) ∈ R^{4×128}. 전역 단계 vocabulary가 필요 없고, 단계 수와 무관하게 크기가 고정된다. 단, 단계 순서는 인코딩하지 않는다.

## 4. CASD Generator와 Stage A 증류

- 입력: frozen VAE 현재 프레임 특징(432차원) + 8차원 robot state를 MLP로 융합한 c_t, 그리고 UMT5 지시문 토큰.
- c_t에서 N개의 병렬 query를 만들고 지시문에 cross-attention, Q + A를 LayerNorm+Linear로 출력(hidden width 128).
- 손실: 정규화된 직접 회귀 L_dir + 같은 과제 내 hardest-negative teacher에 대한 ranking loss L_rank(λ=1, δ=0.05, ρ=0.5). 10,000 step, batch 128, lr 1e-4, cosine. generator만 학습.

## 5. Stage B: 정책 학습과 주입

generator를 고정하고, 정책 학습 시에도 teacher가 아니라 **generator 예측**을 조건으로 쓴다(학습·배포 조건 원천 일치, 예측 오차 포함). Fast-WAM의 video/action expert 각각이 예측 토큰을 affine+GELU로 1024차원에 사영하고, 30개 블록마다 rank-192 Q/K/V/O cross-attention adapter(24 head × 128)로 주입. O의 마지막 층은 0 초기화해 사전학습 함수를 보존한다. 목적함수는 원래의 L_dyn + L_act(flow matching) 그대로. 추론 시 generator는 query당 1회, adapter는 매 denoising step 실행되며 온라인 VLM 호출·텍스트 디코딩·subgoal 이미지 생성이 없다.

## 6. 실험 설정

- LIBERO 4 suite, LIBERO-Plus 7개 교란(카메라, 로봇 초기상태, 언어, 조명, 배경, 센서 노이즈, 레이아웃; 10,030 variant pooled), RoboTwin 2.0 50개 양팔 과제(Clean/Rand., replan_steps=28), MolmoSpaces 4개 조작 카테고리(Pick, P&P, Open, Close; 내비게이션 제외).
- Fast-WAM 3개 변형 모두에 CASD 적용, DreamZero에는 DROID로 CASD 학습.
- 비교는 대부분 **공개된 reference 수치**이며 학습 run을 통제하지 않았다고 저자들이 명시한다.

## 7. 주요 결과

**LIBERO (Table 2)**

| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Fast-WAM (Uncond)† | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Fast-WAM+CASD | 97.6 | 94.8 | 92.6 | 92.8 | 94.5 |
| Fast-WAM-Joint† | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-Joint+CASD | 99.4 | 99.0 | 98.6 | 98.0 | 98.8 |
| Fast-WAM-IDM† | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| **Fast-WAM-IDM+CASD** | 99.6 | 99.8 | 98.2 | 97.8 | **98.9** |

**LIBERO-Plus pooled (Table 3)**: Uncond 49.7→57.2, Joint 68.1→72.2, IDM 71.4→73.9 (reference는 RIFT가 평가한 체크포인트). IDM+CASD: 카메라 50.3, 로봇 77.0, 언어 90.6, 조명 92.8, 배경 67.1, 노이즈 62.8, 레이아웃 81.2.

**RoboTwin 2.0 (Table 4)**: Joint 90.6→**93.0**(Clean 92.7 / Rand. 93.3), IDM 91.3→92.4, Uncond 91.8→90.9. LingBot-VA 92.2.

**MolmoSpaces (Table 5)**: DreamZero 40.7 → DreamZero+CASD **47.9**(Pick 59.7, P&P 36.1, Open 35.1, Close 60.8), Psi-R2 46.5.

**Teacher matching (Table 1)**: Recall@1 Pure 0.430(chance 0.185), Mixed 0.765(0.174), Mixed-3+ 0.790(0.166). 경계를 넘는 chunk의 문맥을 현재 입력만으로 상당히 복원한다.

## 8. Ablation 및 분석

정식 ablation 표(단일 단계 라벨 vs 점유율 가중, routing 위치 등)는 **없다**. 대신 LIBERO-Long 한 과제의 paired rollout(Figure 2)을 제시: 같은 초기 관측·seed에서 Joint 기준선은 700 step timeout, Joint+CASD는 217 step에 성공. replan r16에서 CASD 토큰이 Move/place → Close 쪽으로 이동하며 gripper 명령 10개 전부가 불일치하는 지점과 일치한다. 저자들 스스로 matched 비교가 요인을 분리하지 못한다고 인정한다.

## 9. 강점

- chunk 시간 척도에 맞춘 semantic target이라는 명확하고 일반적인 아이디어. 전역 단계 사전이 필요 없다.
- generator를 고정하고 그 예측으로 정책을 학습해 train/deploy 조건 불일치를 제거.
- 추론 시 VLM 호출이 없어 비용이 작다(쿼리당 1회 소형 generator).
- 세 가지 Fast-WAM 변형과 DreamZero에 걸쳐 적용하고, **성능이 떨어진 경우(Uncond LIBERO −3.2pp)도 보고**하는 정직한 서술.
- 주석 파이프라인을 부록에 매우 상세히 공개.

## 10. 약점 및 한계

- 대부분의 비교가 **공개 reference 대비 cross-report**이며 동일 학습 run의 CASD 유무 비교가 아니다. LIBERO에서 +0.3/+0.9pp는 run 간 분산 수준일 수 있다.
- Uncond 변형은 LIBERO에서 하락, RoboTwin에서도 하락 — 백본 의존성이 크다.
- 요인 분리 ablation 부재(현재 단계 라벨, 균등 혼합, expert routing 등).
- teacher가 단계 순서를 버리며, 동일 설명·점유율이면 순서가 달라도 같은 target.
- teacher matching은 학습 에피소드에서만 측정; 실제 로봇 실험 없음.
- GPT-5.4 기반 주석은 비용과 재현성 문제가 있다.

## 11. 재현 및 확장 아이디어

- 동일 seed·동일 step에서 CASD on/off를 여러 seed로 반복해 순수 효과를 측정.
- horizon을 여러 구간으로 나눈 순서 보존 mixture, 여러 미래 가능성을 예측하는 multi-hypothesis generator.
- 정책 rollout을 재주석해 generator 오차를 실행 중 측정하고 teacher/generator를 적응.
- π0.5 등 VLA 계열 flow-matching 정책에도 동일 adapter로 이식.

## 12. 총평

"chunk가 무엇을 하게 될지"를 점유율 가중 semantic token으로 요약해 WAM에 공급하는 간결한 privileged distillation. RoboTwin 2.0(93.0%)과 MolmoSpaces(+7.2pp)에서 가장 뚜렷한 이득을 보이지만, LIBERO 개선폭이 작고 통제된 on/off 비교가 없어 효과 크기는 신중히 해석해야 한다.

**한 문장 요약**: 라벨을 행동의 시간 척도에 맞추자 — chunk 안의 단계 비율을 semantic token으로 증류하면 WAM이 단계 전환을 더 일찍 잡는다.

<!-- VERIFIED: pdf -->
