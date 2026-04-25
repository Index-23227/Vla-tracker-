# Motus: A Unified Latent Action World Model 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어
Motus(Bi et al., 2025, arXiv:2512.13030, Tsinghua/PKU/Horizon Robotics)는 vision-language understanding, video generation, inverse dynamics, world modeling, action prediction을 단일 framework에 통합한 World–Action Model(WAM)이다. Mixture-of-Transformers(MoT) 아키텍처에 기존 pre-trained 대규모 video generation model(Wan 2.2 5B)과 VLM(Qwen3-VL-2B)을 결합하고, optical flow 기반 latent action 표현을 채택했다. 핵심 동기는 (1) 기존 VLA가 dense supervision으로서의 시각 dynamics를 충분히 활용하지 못한다는 점, (2) text token 중심 정책은 momentum/inertia/contact 같은 물리 동역학 추론에 약하다는 점이다. 후속 연구인 GigaWorld-Policy(arXiv:2603.17240)는 Motus를 SOTA WAM으로 인용하며 비교 baseline으로 사용한다.

## 2. 아키텍처: Mixture-of-Transformers (MoT) 통합 모델
Motus는 4개의 expert를 MoT 구조로 통합한다 (총 ~8B):
- **Wan 2.2 VGM (~5.0B)**: video generation backbone, future frame 예측 및 world modeling 담당.
- **Qwen3-VL VLM (~2.13B)**: scene understanding 및 instruction parsing.
- **Action Expert (~641.5M, 30 layer transformer)**: flow-matching 기반 action prediction.
- **Understanding Expert (~253.5M)**: 보조 reasoning 및 cross-modal alignment.

GigaWorld-Policy 인용에 따르면 Motus는 "UniDiffuser-style scheduler"로 다른 modeling mode 간 유연한 전환을 지원한다 — 예: video-only generation, action-only decoding, joint video+action prediction. Latent action 표현은 optical flow 기반으로 cross-embodiment generalization을 가능하게 한다(YAML tags: world-action-model, MoT, latent-action, optical-flow, cross-embodiment).

## 3. 학습 데이터 및 최적화
Motus는 InternVLA-A1의 InternData-A1 dataset 인용 트리에 함께 등장하며 동일 시기의 대규모 합성 + real-world data 활용 가능성이 높다. Backbone인 Wan 2.2 5B는 Wan 2025 video generation pretraining의 가중치를 직접 상속한다(GigaWorld-Policy 동일 설정에서 확인). 학습 목표는 (a) future visual dynamics 예측 손실, (b) flow-matching action regression, (c) understanding/generation cross-supervision으로 구성되며, action chunk 길이는 GigaWorld-Policy의 비교 setup인 p=48과 유사할 가능성이 있다.

## 4. 핵심 실험 결과: LIBERO 및 RoboTwin 2.0
**LIBERO (YAML 기록)**: `libero_long: 97.6` — LIBERO-Long suite에서 강력한 성능. 단일 suite 점수만 보고되어 Spatial/Object/Goal 다른 suite 결과는 paper 본문에서 확인 필요.

**RoboTwin 2.0 (YAML 기록 + 외부 인용 검증)**: `robotwin_v2_easy_avg: 88.66`, `robotwin_v2_hard_avg: 87.02`. GigaWorld-Policy 논문(Table 2)이 Motus의 RoboTwin 2.0 task별 성능을 다음과 같이 인용한다:
- Adjust Bottle: Clean 0.89 / Rand 0.93
- Place A2b Left: 0.88 / 0.79
- Place Bread Skillet: 0.86 / 0.83
- Place Cans Plasticbox: 0.98 / 0.94
- Place Fan: 0.91 / 0.87
- Place Mouse Pad: 0.66 / 0.68

이들 평균은 약 ~0.87 수준으로 YAML의 hard_avg 87.02와 부합. 또 다른 GigaWorld-Policy 비교표(real-world)에서 Motus는 0.80/0.75/0.70/0.80/0.76 수준의 task SR을 기록하며, GigaWorld는 9× faster inference로 비슷한 성능을 달성한다고 보고한다. 이는 Motus가 강력하지만 inference 비용이 크다는 뜻 — 인용된 latency는 ~3231ms로 GigaWorld(~360ms) 대비 9× 느리다.

작업 task 힌트의 "RoboTwin v2 #1 87.8"은 Easy 88.66과 Hard 87.02의 산술 평균(=87.84)에 근접해 일관된 SOTA 위치를 시사한다.

## 5. 비교: WAM 계열 내 Motus의 위상
GigaWorld-Policy는 WAM 계열을 Motus와 Cosmos-Policy(Kim et al., 2026)로 정의하고 두 모델이 latent space에서 environment dynamics를 학습하며 predictive rollout을 사용한다고 평가한다. Motus는 "기존 일반 pre-trained model + 풍부한 motion 정보"를 결합해 MoT로 세 expert(VGM/VLM/Action)를 통합하고, UniDiffuser-style scheduler로 modeling mode 전환을 지원한다는 점에서 차별화된다. 반면 Cosmos-Policy는 별도 latent dynamics 학습에 집중한다.

GigaWorld-Policy real-world Table에서 Motus 0.76 평균(GigaBrain-0 0.68, Cosmos-Policy 0.58, π0.5/X-VLA baseline 대비 우위)으로 WAM 계열 SOTA로 자리매김했다.

## 6. 평가 및 한계
**강점**: (a) 사전학습된 대규모 video generation backbone(Wan 2.2 5B) 활용으로 시각 dynamics 표현이 풍부, (b) MoT 분리 expert 설계로 modality별 specialization 가능, (c) optical-flow 기반 latent action으로 cross-embodiment 전이 용이, (d) RoboTwin 2.0에서 fine-tuned 조건 SOTA(Easy 88.66 / Hard 87.02), LIBERO-Long 97.6 달성.

**약점**: (a) ~8B 규모로 inference latency가 큰 편(~3231ms per inference, GigaWorld 대비 9× 느림) — 실시간 closed-loop 제어에 부담; (b) 현 시점 close-source(YAML open_source=false), code_url=null로 재현이 어려움; (c) optical flow latent action이 contact-rich/fine-grained dexterous manipulation에 충분한지 검증 필요; (d) RoboTwin 2.0 Easy/Hard 외 LIBERO Spatial/Object/Goal suite, CALVIN, SimplerEnv 같은 광범위 벤치마크 보고 부족.

**YAML 점검**: parameters "~8B (VGM 5.0B + VLM 2.13B + Action 641.5M + Understanding 253.5M)"는 합산 ~8.0B로 적절(5.0+2.13+0.6415+0.2535 = 8.025B). action_head_category=flow_matching은 action expert가 flow-matching 사용한다는 일반적 WAM 설계와 일치. backbone "Mixture-of-Transformers (Wan 2.2 5B video + Qwen3-VL-2B)" 표기 정확. open_source=false, code_url=null은 일관. **확인 권장**: benchmarks.libero에 libero_long: 97.6만 단일 suite로 기록되어 있어 평균(libero_avg) 또는 4-suite 분리 보고 가능 시 YAML 보강 필요. `robotwin_v2_easy_avg/hard_avg` 87.8 #1 hint와 leaderboard ranking 일치 확인됨.

**참고**: 본 리뷰는 Motus 원본 PDF가 환경 내 접근 불가하여 (a) YAML 메타데이터, (b) GigaWorld-Policy(arXiv:2603.17240) 및 InternVLA-A1(arXiv:2601.02456)의 인용 분석에 기반해 작성되었다. 향후 PDF 입수 시 model architecture diagram, Table 번호, 학습 hyperparameter 등 1차 출처 검증이 필요하다.

<!-- VERIFIED: pdf -->
