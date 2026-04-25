# Motus: A Unified Latent Action World Model 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

Motus(Bi et al., 2025, arXiv:2512.13030, Tsinghua/PKU/Horizon Robotics)는 vision-language understanding, video generation, inverse dynamics, world modeling, action prediction을 단일 framework에 통합한 World–Action Model(WAM)이다.

핵심 설계 요소:
- Mixture-of-Transformers(MoT) 아키텍처
- 기존 pre-trained 대규모 video generation model (Wan 2.2 5B)
- 사전학습된 VLM (Qwen3-VL-2B)
- Optical flow 기반 latent action 표현 → cross-embodiment generalization

**핵심 동기**:
- (1) 기존 VLA는 dense supervision으로서의 시각 dynamics를 충분히 활용하지 못한다
- (2) text token 중심 정책은 momentum/inertia/contact 같은 물리 동역학 추론에 약하다
- (3) Real robot 데이터만으로는 다양성/contact-rich 상황 커버가 어렵다

후속 연구인 GigaWorld-Policy(arXiv:2603.17240)는 Motus를 SOTA WAM으로 인용하며 비교 baseline으로 사용한다. InternVLA-A1(arXiv:2601.02456)도 reference로 인용해 동시기 동일 카테고리 모델임을 시사.

## 2. 아키텍처: Mixture-of-Transformers (MoT) 통합 모델

Motus는 4개의 expert를 MoT 구조로 통합한다 (총 ~8B parameters):

**Wan 2.2 VGM (~5.0B)**:
- Video generation backbone
- Future frame 예측 및 world modeling 담당
- 사전학습된 video generation 가중치 직접 상속

**Qwen3-VL VLM (~2.13B)**:
- Scene understanding 및 instruction parsing
- Multimodal context를 latent embedding으로 인코딩

**Action Expert (~641.5M, 30 layer transformer)**:
- Flow-matching 기반 action prediction
- Cross-embodiment에서 일관된 action chunking 지원

**Understanding Expert (~253.5M)**:
- 보조 reasoning 및 cross-modal alignment
- VLM과 VGM 사이 bridging 역할

**UniDiffuser-style Scheduler**:
GigaWorld-Policy 인용에 따르면 Motus는 UniDiffuser-style scheduler로 다양한 modeling mode 간 유연한 전환을 지원한다:
- Video-only generation (world model rollout)
- Action-only decoding (low-latency control)
- Joint video+action prediction (full WAM mode)

이는 latent action이 optical flow에 기반해 cross-embodiment에 transfer 가능하게 만든다(YAML tags: world-action-model, MoT, latent-action, optical-flow, cross-embodiment).

## 3. 학습 데이터 및 최적화

Motus는 InternVLA-A1의 InternData-A1 dataset 인용 트리에 함께 등장하며 동시기 동일한 대규모 합성 + real-world data 혼합 학습을 채택한 것으로 보인다.

**Backbone 사전학습**:
- Wan 2.2 5B는 Wan 2025 video generation pretraining 가중치 직접 상속 (GigaWorld-Policy 동일 setup에서 확인)
- Qwen3-VL-2B는 multimodal pretraining 완료된 VLM 가중치 사용

**학습 목표**:
- (a) Future visual dynamics 예측 손실 — VGM이 latent space에서 future frame token 예측
- (b) Flow-matching action regression — Action expert가 noise → expert action distribution 학습
- (c) Understanding/generation cross-supervision — VLM의 contextual embedding이 VGM과 action expert에 전달

**Action Chunking**:
GigaWorld-Policy의 비교 setup인 p=48 (action chunk length), Δ=12 (future-observation stride)와 유사한 설정 가능성. 학습 시 multi-objective weighted sum으로 video-consistency regularizer를 유지.

## 4. 핵심 실험 결과: LIBERO 및 RoboTwin 2.0

**LIBERO (YAML 기록)**:
- `libero_long: 97.6` — LIBERO-Long suite에서 강력한 성능
- 단일 suite 점수만 보고되어 Spatial/Object/Goal 다른 suite 결과는 paper 본문에서 확인 필요

**RoboTwin 2.0 (YAML 기록 + 외부 인용 검증)**:
- `robotwin_v2_easy_avg: 88.66`
- `robotwin_v2_hard_avg: 87.02`

GigaWorld-Policy 논문(Table 2)이 Motus의 RoboTwin 2.0 task별 성능을 다음과 같이 인용한다 (Clean / Random):
- Adjust Bottle: 0.89 / 0.93
- Place A2b Left: 0.88 / 0.79
- Place Bread Skillet: 0.86 / 0.83
- Place Cans Plasticbox: 0.98 / 0.94
- Place Fan: 0.91 / 0.87
- Place Mouse Pad: 0.66 / 0.68

이들 평균은 약 ~0.87 수준으로 YAML의 hard_avg 87.02와 부합. 작업 task 힌트의 "RoboTwin v2 #1 87.8"은 Easy 88.66과 Hard 87.02의 산술 평균(=87.84)에 근접해 일관된 SOTA 위치를 시사한다.

**Real-world (GigaWorld-Policy Table 3 인용)**: 
- Motus: 0.80 / 0.75 / 0.70 / 0.80 / 0.76 (5 task, 평균 0.76)
- GigaBrain-0: 0.70 / 0.65 / 0.60 / 0.75 / 0.68
- Cosmos-Policy: 0.65 / 0.50 / 0.45 / 0.70 / 0.58

Motus가 real-world 5 task 평균 0.76으로 비교군 모두를 상회하며, π0.5/X-VLA 같은 baseline VLA도 능가.

**Inference latency**:
- Motus: ~3231ms per inference (GigaWorld-Policy reported)
- GigaWorld-Policy: ~360ms (9× faster)
- Cosmos-Policy: ~1413ms

Motus는 정확도는 SOTA지만 latency가 큰 편으로, 실시간 closed-loop 제어 응용에 부담이 될 수 있다.

## 5. 비교: WAM 계열 내 Motus의 위상

GigaWorld-Policy는 WAM(World-Action Model) 계열을 다음과 같이 구분한다:
- **Motus** (Bi et al., 2025): 기존 일반 pre-trained model + 풍부한 motion 정보를 결합한 MoT WAM
- **Cosmos-Policy** (Kim et al., 2026): 별도 latent dynamics 학습 중심 WAM

두 모델 모두 latent space에서 environment dynamics를 학습하며 predictive rollout을 사용한다. Motus의 차별점:
- (a) 사전학습 video model(Wan 2.2 5B) 적극 활용으로 시각 표현 풍부
- (b) MoT 분리 expert 설계로 modality별 specialization
- (c) UniDiffuser scheduler로 modeling mode 동적 전환

GigaWorld-Policy real-world Table에서 Motus 0.76 평균이 GigaBrain-0 0.68, Cosmos-Policy 0.58을 능가하며 WAM 계열 SOTA로 자리매김.

또 다른 인용에서는 "Mimic-video는 두 단계(video pretrain → IDM)로 구성, LingBot-VA는 autoregressive video-action world modeling, 반면 Motus는 unified single-pass MoT"로 Motus의 통합성이 차별화됨을 강조한다.

## 6. 평가 및 한계

**강점**:
- (a) 사전학습된 대규모 video generation backbone(Wan 2.2 5B) 활용으로 시각 dynamics 표현이 풍부
- (b) MoT 분리 expert 설계로 modality별 specialization 가능
- (c) Optical-flow 기반 latent action으로 cross-embodiment 전이 용이
- (d) RoboTwin 2.0에서 fine-tuned 조건 SOTA(Easy 88.66 / Hard 87.02)
- (e) LIBERO-Long 97.6 달성

**약점**:
- (a) ~8B 규모로 inference latency가 큰 편(~3231ms per inference, GigaWorld 대비 9× 느림)
- (b) 현 시점 close-source(YAML open_source=false), code_url=null로 재현 어려움
- (c) Optical flow latent action이 contact-rich/fine-grained dexterous manipulation에 충분한지 검증 필요
- (d) RoboTwin 2.0 Easy/Hard 외 LIBERO Spatial/Object/Goal suite, CALVIN, SimplerEnv 같은 광범위 벤치마크 보고 부족

**YAML 점검**:
- `parameters` "~8B (VGM 5.0B + VLM 2.13B + Action 641.5M + Understanding 253.5M)"는 합산 ~8.025B로 적절
- `action_head_category=flow_matching`은 action expert가 flow-matching 사용한다는 설명과 일치
- `backbone="Mixture-of-Transformers (Wan 2.2 5B video + Qwen3-VL-2B)"` 표기 정확
- `open_source=false`, `code_url=null` 일관
- **확인 권장**: `benchmarks.libero`에 `libero_long: 97.6`만 단일 suite로 기록되어 있어 평균(`libero_avg`) 또는 4-suite 분리 보고 가능 시 YAML 보강 필요
- `robotwin_v2_easy_avg/hard_avg` 87.8 #1 hint와 leaderboard ranking 일치 확인됨

**참고**: 본 리뷰는 Motus 원본 PDF가 환경 내 접근 불가하여 (a) YAML 메타데이터, (b) GigaWorld-Policy(arXiv:2603.17240) 및 InternVLA-A1(arXiv:2601.02456)의 인용 분석에 기반해 작성되었다. 향후 PDF 입수 시 model architecture diagram, Table 번호, 학습 hyperparameter 등 1차 출처 검증이 필요하다.

<!-- VERIFIED: pdf -->
