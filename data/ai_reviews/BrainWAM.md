# BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving

> **한 줄 요약**: 자율주행에서 VLA(semantic prior)와 WAM(predictive dynamics)을 합치려는 시도가 **raw token 공유 attention**에서는 오히려 실패한다는 점(Tri-MoT 87.8 PDMS < WAM-only 88.1 PDMS)을 attention-allocation mismatch로 진단하고, 좌/우뇌 분업과 뇌량·소뇌에서 영감을 얻어 두 경로를 **compact action token 수준에서만 조율**하는 CAB + CIF 구조를 제안. NAVSIM v1 89.5 PDMS, NAVSIM v2 89.6 EPDMS로 SOTA 달성.

- arXiv: 2608.12854 (2026-08-13, cs.RO)
- 저자: Bing Zhan*, Shuyao Shang*, Jiahao Gu*† 외 (CASIA NLPR / Li Auto Inc.)

---

## 1. 배경 및 동기

자율주행 planning은 두 종류의 근거를 동시에 요구한다.

- **Semantic constraint**: 교통 규칙, 경로 지시, 장면 의미, 고수준 주행 의도 → VLM prior를 쓰는 **VLA** 계열이 강함 (ORION, ReCogDrive, OpenDriveVLA, AutoVLA 등).
- **Predictive dynamics**: 미래 장면 전개, 상호작용 결과, 물리적 실현 가능성 → action-conditioned world modeling 기반 **WAM** 계열이 강함 (GAIA-1, DriveDreamer, DrivingGPT, LAW, WoTE, DriveLaW 등).

두 계열은 상호보완적이지만, 기존 연구는 대체로 한쪽만 강조한다. 논문의 중심 질문은 "VLA와 WAM을 **어떻게** 결합해야 상보적 잠재력이 실제로 발현되는가"이다.

## 2. 핵심 진단: Tri-MoT의 attention-allocation mismatch

가장 직관적인 결합은 VLM token, VGM(Video Generative Model) token, action token을 하나의 attention 공간에 넣는 **Tri-modal Joint Attention (Tri-MoT)**이다. 그러나 논문은 이 raw-token fusion이 **WAM 단독보다도 못하다**는 것을 관측한다.

Fig. 2의 layer별 attention ratio 분석에 따르면, action token은 대부분의 Transformer layer(특히 shallow layer)에서 VGM token보다 **VLM token에 훨씬 강하게 attend**한다. 저자들은 이를 multimodal 학습의 **modality competition** 현상으로 설명한다.

- VLM token: 대규모 vision-language pretraining에서 온 **깨끗하고 안정적인** semantic abstraction → 학습이 쉬움.
- VGM token: rectified-flow denoising 과정 중이라 **아직 noise가 섞인 저신호** feature → 학습이 어려움.

결과적으로 action token이 "VLM 지름길(semantic shortcut)"을 택하고, planning에 필요한 predictive dynamics는 과소이용된다. Appendix A는 두 가지 반증을 제시해 이 해석을 뒷받침한다. (1) VGM이 무용해서가 아니다 — video denoising을 끄면 PDMS가 79.3까지 폭락한다. (2) 정보 부족도 아니다 — Tri-MoT는 WAM-only보다 **정보를 더 많이** 갖고도 87.8 < 88.1로 뒤진다. 즉 문제는 신호 부족이 아니라 **경쟁**이다.

## 3. 방법론: 뇌 기능 분화에서 온 설계 원리

신경과학적 관찰 — 좌반구는 언어·기호·순차 처리, 우반구는 시공간·전체론적 이해를 담당하고, 둘은 **뇌량(corpus callosum)**으로 정보를 교환하며, **소뇌(cerebellum)**가 운동 의도를 조율·정련한다 — 로부터 계산 원리를 도출한다: *두 경로가 먼저 각자 behavior-relevant한 action 표현을 형성한 뒤, compact action-level 통신으로만 조율하라.*

- **좌반구 경로 (VLA branch)**: 교통 장면 semantics, route instruction, rule-aware decision prior를 증류.
- **우반구 경로 (WAM branch)**: 시공간 dynamics, 물리적 일관성, 미래 상호작용 단서를 증류.
- **CAB (Callosal Action Bridge)**: 두 action stream 간 양방향 통신.
- **CIF (Cerebellar Intent Fusion)**: 정련된 action intent를 융합해 실행 가능한 궤적으로 디코딩.

## 4. WAM branch

Video backbone은 **Wan2.2-TI2V-5B**, 여기에 경량 action expert를 부착한다. Video latent $x^v$와 action trajectory $x^a$를 **서로 독립적인 rectified-flow timestep** $t_v, t_a$로 교란한다 — 이 decoupled schedule이 후술할 asynchronous inference의 근거다. Dual-MoT 모듈이 shared self-attention으로 두 stream을 결합하되 modality-specific FFN으로 각자의 모델링 능력을 보존한다.

Rectified Flow / Flow Matching 정의를 따라 $x_t = (1-t)x_0 + t\epsilon$, 속도 타깃 $u = \epsilon - x_0$이며, 손실은 $\mathcal{L}_{WAM} = \mathcal{L}_{vid} + \lambda^a_{pred}\mathcal{L}^a_{pred}$ (둘 다 velocity field에 대한 L2).

## 5. VLA branch

VLM backbone은 **Qwen3-VL-4B**. 멀티뷰 이미지와 주행 지시를 semantic token $U$로, ego history를 state token $E$로 인코딩하고, action expert가 noisy trajectory를 semantic-grounded action token $A_{sem}$으로 변환한다. Dual-MoT가 semantic/state/action token을 shared self-attention으로 결합해 action denoising을 guide하며, 손실은 $\mathcal{L}^a_{sem}$ 단일 rectified-flow 항이다.

## 6. CAB와 CIF

**CAB**는 $A_{pred}$와 $A_{sem}$ 사이의 양방향 cross-attention 메시지를 계산하고 **gated residual**로 주입한다. gate는 $\alpha = \tanh(g)$, $g$는 zero-init이라 CAB는 초기에 identity mapping으로 시작해 Stage 3에서 점진적으로 cross-stream update를 학습한다(Flamingo/LLaMA-Adapter 계열 기법). 구현: action token stream 각각 $L=8$, hidden 1024, action expert의 **layer 9와 18**에 삽입, head 8개 × head-dim 128, bias 없음, 총 **약 16.8M 파라미터**.

**CIF**는 두 stream을 1024차원 공유 공간에 투영하고 learnable source embedding을 더한 뒤, **action-timestep 조건 AdaLN**을 쓰는 **2-layer Transformer(head 8)**로 처리하고 element-wise 평균으로 융합한다. 융합 표현은 $\hat{u}^a_{fuse}$로 디코딩되며 joint stage는 **오직 fused prediction만** 감독한다. 약 **49.3M 파라미터**.

## 7. 3단계 학습

| Stage | 학습 대상 | 손실 |
|-------|-----------|------|
| 1 | WAM branch (VGM backbone + action expert) | video reconstruction(velocity) + action loss |
| 2 | VLA branch (VLM backbone + action expert) | action loss |
| 3 | **CAB, CIF, action decoder만** (두 branch는 freeze) | fused action loss |

Stage 3에서 두 action expert는 **동일한 noisy trajectory와 동일한 action timestep**을 받고, WAM의 video stream만 자체 $t_v$를 유지해 predictive context를 공급한다.

## 8. 실험 설정

- 벤치마크: **NAVSIM v1** (PDMS)와 **NAVSIM v2** (EPDMS). OpenScene(nuPlan 재가공) 기반 실주행 로그, 프레임마다 **4초·2Hz·8 waypoint** 궤적을 예측해 short-horizon non-reactive simulation으로 평가.
- PDMS = NC × DAC × (5·EP + 5·TTC + 2·C)/12. EPDMS는 DDC/TLC penalty multiplier와 LK/HC/EC 가중 subscore를 추가.
- 학습: 각 stage 100K step, **8× NVIDIA H20**, per-GPU batch 6, AdamW, peak LR 5e-5, weight decay 0.01, cosine + 200 warmup, bf16, DeepSpeed ZeRO-2, 3K step마다 체크포인트.
- 추론: action stream **3-step rectified-flow sampling**.

## 9. 주요 결과

**NAVSIM v1 (Table 1)** — BrainWAM: NC 98.1 / DAC 97.5 / TTC 94.9 / C 100.0 / EP 83.8 → **PDMS 89.5**. 참고로 Human은 94.8 PDMS. 비교군: TransFuser 84.0, UniAD 83.4, DiffusionDrive 88.1, ReCogDrive 86.5, DynVLA 87.2, AutoVLA 89.1, DriveVLA-W0 87.2, WoTE 88.3, DriveLaW 89.1. 이득은 **DAC(97.5, 최고)와 EP(83.8, 최고)**에 집중되어 drivable-area 준수와 주행 진척이 개선됐음을 보여준다.

**NAVSIM v2 (Table 2)** — BrainWAM: NC 98.1 / DAC 97.5 / DDC 99.6 / TLC 99.9 / EP 88.2 / TTC 97.4 / LK 97.6 / HC 98.4 / EC 85.8 → **EPDMS 89.6**. 비교군: TransFuser 76.7, HydraMDP++ 81.4, DriveSuprim 83.1, ARTEMIS 83.1, DriveVLA-W0 86.1, DriveDreamer-Policy 88.7. 개선은 주로 EP와 EC에서 오며, 여러 rule-compliance 지표는 이미 포화 상태다.

## 10. Ablation 분석

- **Branch 상보성 (Table 3)**: VLA-only 86.1, WAM-only 88.1, Tri-MoT 87.8, **BrainWAM 89.5**. NAVSIM에서는 predictive prior가 semantic prior보다 강력하며, raw-token fusion은 WAM-only보다도 낮다.
- **CAB/CIF (Table 4)**: CAB만 88.7, CIF만 88.5, 둘 다 89.5. 개선이 DAC/EP에 집중되고 NC/TTC는 안정적.
- **CAB 개수 (Table 6, 10-step joint denoising 기준)**: 1개 88.9 → 2개 89.3 → 3/5/28개 89.2–89.3. 2회 상호작용에서 사실상 포화하므로 layer 9/18 두 개를 채택.
- **CIF fusion 방식 (Table 7)**: MLP 88.8 < Gate 89.1 < **Transformer 89.3**. 깊이(Table 8)는 1층 89.0 → 2층 89.3 → 3층 89.3으로 2층이면 충분.
- **Stage-3 업데이트 전략 (Table 9)**: full-model fine-tuning 88.8 vs **CAB/CIF/decoder만 89.5**. 이유로 branch별 수렴 속도 차이(VLA-only 54K step에 86.1, WAM-only 81K step에 88.1)를 제시하며, 동시 학습 시 두 경로의 업데이트 불균형이 CAB/CIF 입력 표현을 계속 흔들어 조율을 어렵게 한다고 설명.
- **Asynchronous video denoising (Table 5, H20 1장 기준)**: 0 step → 382ms / **79.3 PDMS / 75.8 EPDMS**(붕괴), 1 step → 475ms / 89.3 / 89.4, 2 step → 565ms / **89.5 / 89.6**, 3 step → 644ms / 89.4 / 89.6. **video denoising 단 1 step만으로 유용한 predictive context 대부분이 확보**된다는 것이 실용적으로 중요한 발견이다.

## 11. 정성적 분석

Fig. 5/6은 네 가지 대표 시나리오로 두 경로의 실패 양상이 실제로 다름을 보인다. **VLA-only가 우세한 경우**: navigation following(국소적으로 그럴듯하지만 잘못된 분기 대신 경로 지시를 따름), red-light response(선행 차량 브레이크등 + 적색 신호의 결합 해석). **WAM-only가 우세한 경우**: pedestrian interaction 같은 상호작용 협상, curve lane keeping 같은 궤적 실현 가능성. BrainWAM은 네 경우 모두 처리하며, 두 단일 branch가 모두 실패하는 사례에서도 합리적 궤적을 생성한다.

## 12. 종합 평가

**강점**

- "VLA + WAM은 좋을 것"이라는 막연한 기대를 **정면으로 반증**한 뒤(Tri-MoT < WAM-only), attention 시각화와 modality-competition 문헌으로 **원인을 특정**하고, 그 원인에 정확히 대응하는 구조를 설계한 논리 전개가 매우 깔끔하다. 문제 진단 → 설계 근거 → ablation의 인과 사슬이 끊기지 않는다.
- 동일 backbone·유사 파라미터 수의 Tri-MoT 대비 87.8 → 89.5 개선이므로, 이득이 **capacity가 아니라 coordination 메커니즘**에서 왔다는 주장이 방어된다.
- Zero-init tanh gate + branch freeze라는 보수적 설계 덕분에 Stage 3가 사전학습 표현을 파괴하지 않으며, 이는 Table 9로 실증된다.
- Video/action timestep을 분리한 asynchronous inference는 성능 손실 없이 latency를 크게 줄이는 실용적 기여다(2 step에서 최고 성능).

**한계 및 논의**

- 저자 스스로 밝히듯 **추론 비용**이 가장 큰 약점이다. 생성형 video backbone을 추론 시에도 유지하므로 단일 branch planner보다 연산·메모리 비용이 크고, 475–644ms latency는 **실차 실시간 요구를 만족하지 못한다**. Video branch 압축/증류, 중복 연산 제거, early-exit이 후속 과제로 제시된다.
- 평가가 **NAVSIM v1/v2 open-loop non-reactive simulation에 한정**된다. Bench2Drive 같은 closed-loop이나 실차 검증은 없어, 조율 구조가 반응형 환경에서도 이득을 유지하는지는 미확인이다.
- NC(98.1)와 TTC(94.9)는 DriveLaW(99.0/96.7)나 AutoVLA(98.4/98.0)에 뒤지며, 개선이 사실상 DAC/EP에 편중되어 있다. "더 멀리, 더 규정 안에서 간다"는 이득이지 "더 안전하다"는 이득은 아니라는 점은 주의해서 읽어야 한다.
- 뇌 비유(좌/우반구·뇌량·소뇌)는 설명 장치로는 매력적이나 구조 자체는 dual-branch + cross-attention bridge + fusion Transformer라는 익숙한 조합이며, 신경과학적 주장이 설계 선택을 강하게 제약하지는 않는다.
- 3-stage 파이프라인은 각 stage 100K step으로 총 학습 비용이 크고, 두 개의 대형 backbone(5B video + 4B VLM) 사전학습을 전제한다. 코드/가중치는 공개되지 않았다.

**분류 노트**: action head는 두 개의 rectified-flow action expert가 velocity field를 예측하고 3-step 샘플링으로 궤적을 생성하는 구조이므로 `flow_matching`으로 분류한다.

**Score: 8.5 / 10**

<!-- VERIFIED: pdf -->
