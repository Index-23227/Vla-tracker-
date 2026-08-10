# HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving

> **한 줄 요약**: Emu3 기반 autoregressive VLA에 pixel-level VQ 토큰 예측과 video-VAE latent 예측을 **동시에** 학습시키는 hybrid world model을 도입하고, co-fine-tuning 단계에서는 latent만 예측하여 action expert에 조건으로 주입함으로써 단안 전방 카메라만으로 NAVSIM v1 PDMS **90.59**, NAVSIM v2 EPDMS **89.71**, 우천·안개 노이즈 테스트셋 PDMS **86.87**을 달성한 자율주행 VLA.

- 저자: Quanfu Yu\*, Xian Wu\*, Hao Xu\*, Liulong Ma† (BYD Automotive New Technology Research Institute)
- arXiv: 2607.20988v1 [cs.CV], 2026-07-23
- 코드: 미공개 (논문 내 저장소 링크 없음)

---

## 1. 배경 및 동기

### 기존 world-model VLA의 이분법
자율주행 VLA에 world modeling을 붙이는 방식은 크게 두 계열로 갈린다.

| 계열 | 대표 연구 | 장점 | 약점 |
|------|-----------|------|------|
| **Pixel-based WM** | FSDrive, DriveVLA-W0, WoTE, PWM, DriveLaW | 미래 프레임 전체를 복원 → dense한 기하/물리 제약, occlusion·long-tail 일반화 강함 | 비·안개·조도 변화 등 **환경 노이즈에 극도로 취약** |
| **Latent-based WM** | LAW, World4Drive, Epona, DreamerAD, Latent-WAM | compact feature 공간에서만 미래를 예측 → 노이즈 강건 | pixel grounding 부재 → **representation degeneration**, 해석성 저하 |

### 핵심 질문
- **pixel-level grounding의 정밀함과 latent 예측의 노이즈 강건성을 하나의 프레임워크에서 동시에 얻을 수 있는가?**
- 두 supervision을 학습 단계별로 **비대칭적으로** 배치하면(사전학습=둘 다, 미세조정=latent만) 두 마리 토끼를 잡을 수 있는가?

📌 [Figure 1 삽입] — (a) pixel-based WM: action↔image 반복 예측, (b) latent-based WM: latent state 예측, (c) HyWorldVLA: 둘을 동시에 모델링

---

## 2. 방법론 심층 분석

### 2.1 문제 정의
시각 t에서 전방 카메라 시퀀스 V_{t-H:t}, 과거 waypoint W_{t-H:t} = (x, y, θ), 과거 내비게이션 명령 L_{t-H:t}(Go Straight, Turn Right 등)를 입력받아 미래 waypoint W_{t+1:t+T}를 출력. 논문 설정은 H = 1.0 s, T = 4.0 s.

### 2.2 전체 구조 (3 컴포넌트 / 3 학습 스테이지)

1. **Video Encoder** (텍스트 유도 video VAE) — 미래 프레임을 compact latent로 압축
2. **VLM Backbone** (Emu3) — latent와 visual token을 함께 예측하는 world model
3. **Action Expert** — 예측된 latent + 과거 motion + VLM hidden state로 궤적 생성

📌 [Figure 2 삽입] — VAE 학습 → pre-training(iterative prediction + latent query Q) → co-fine-tuning(joint attention → action expert)

### 2.3 Text-Guided Latent Feature Learning
- 기반: **VideoVAEPlus** (Xing et al., 2024), 4채널 pre-trained 가중치에서 초기화
- **Spatial encoder**: Stable Diffusion image VAE 위에 3D conv를 쌓아 8× 공간 다운샘플, 시간 길이는 유지 → z1 ∈ R^{c×T×H/8×W/8}
- **Temporal autoencoder**: 3D ResNet block으로 4× 시간 압축 → z2 ∈ R^{c'×T/4×H/8×W/8}
- **텍스트 cross-attention**: 각 블록 내부에 multi-layer cross-attention 삽입. visual patch를 query/value, **Flan-T5** 텍스트 임베딩을 key로 사용, residual로 더함 → motion ghosting·edge blurring·temporal flickering 억제
- 손실: `L_vae = L_rec + λ_GAN·L_GAN + λ_KL·L_KL` (λ_GAN = 0.5, λ_KL = 1e-6)

### 2.4 Pre-training: 모든 모달리티를 discrete token으로
- **Vision**: VQGAN 토크나이저로 각 chunk의 첫 프레임을 discrete visual token V_q^j로 양자화
- **Action**: 절대좌표 → ego-relative 변환 후 **FAST** action tokenizer로 이산화 → A_q^j
- **Language**: Emu3 토크나이저로 주행 명령 이산화 → L_q^j
- **Future latent prediction**: 학습 가능한 query token **Q**를 시퀀스에 삽입, 해당 위치 hidden state를 MLP에 통과시켜 미래 latent ẑ2 예측

입력 시퀀스:
`S = [L¹, V¹, A¹, L², V², Q, A², L³, …, V^N, A^N]` (causal masking으로 정보 누수 차단)

학습 목적함수 (식 6):
`L_world = Σ CE(L̂,L) + Σ CE(Â,A) + λ1·Σ CE(v̂,v) + λ2·‖ẑ2 − z2‖²`

즉 **언어 + 행동 + 비주얼 토큰(pixel grounding) + latent 회귀**의 4항 결합.

### 2.5 Co-fine-tuning: latent만 예측
- world model과 action expert를 **joint attention**(π₀ 방식, Black et al. 2024)으로 결합
- 입력: 과거 행동, 주행 명령, backbone이 예측한 ẑ2 → attention 후 action token 위치 hidden state를 MLP head로 디코딩
- 두 계열의 action model에 모두 이식하여 cross-paradigm 호환성 검증:
  - **Generative (flow matching)**: `L = |v̂ − v|² + λ3·|ẑ − z|²`
  - **Selection-based (Hydra-MDP 계열)**: `L = KL(p_pred ‖ p_gt) + λ3·‖ẑ − z‖²`, p_gt는 expert 궤적과 평가 metric을 종합해 산출한 offline trajectory vocabulary 점수

> **핵심 설계 논리**: pixel reconstruction은 **사전학습 단계의 structural regularizer**로만 쓰이고, 추론 시에는 latent 경로만 살아남는다. 정밀도는 사전학습에서 흡수하고, 노이즈 강건성은 추론 경로에서 확보하는 비대칭 배치.

---

## 3. 데이터 전략

| 스테이지 | 데이터 | 규모 / 설정 |
|----------|--------|-------------|
| VAE fine-tune | **NuPlan** | 세그먼트당 8프레임 균일 샘플, 216×216 리사이즈, Qwen3.6-plus로 짧은 장면 설명 자동 생성 |
| Pre-training | **OpenScene** | 120시간 이상 주행 영상, 20.0 s 클립 단위, chunk 1.0 s → N = 6 |
| Co-fine-tuning | **NAVSIM trainval** | GT ego 궤적 annotation 10만 프레임 이상 |
| 노이즈 평가 | OpenScene test에서 별도 수집 | 비·안개로 인한 **비균일 노이즈 655 케이스** |

캡션 생성에 LLM(Qwen3.6-plus)을 쓴 점이 특징 — 텍스트 supervision을 사람 라벨 없이 확보했다.

---

## 4. 시스템 / 학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | Emu3 (unified multimodal autoregressive) |
| VAE 학습 | 4 PPU (Alibaba Cloud), per-PPU batch 1, 100K step, GAN loss는 50K step 이후 활성화 |
| Pre-training | 2 노드 × 16 PPU, batch 4, LR 2.2e-4, 최대 4000 step, λ1 = 0.5 / λ2 = 0.1 |
| Co-fine-tuning | 4000 step, total batch 96, LR 5e-5, λ3 = 0.1 |
| 센서 | **1× 전방 카메라만** (경쟁 모델 다수는 3× Cam + LiDAR) |
| 파라미터 수 | 논문에 미보고 |
| 추론 속도 | 논문에 미보고 (Appendix 실패 사례에서 "camera count vs inference latency trade-off는 미해결"이라고만 언급) |

특이사항: NVIDIA GPU가 아니라 **Alibaba Cloud PPU**로 학습했다고 명시.

---

## 5. 실험 설계 및 평가 프로토콜

- **NAVSIM v1**: NC(No at-fault Collision), DAC(Drivable Area Compliance), TTC, C.(Comfort), EP(Ego Progress)를 곱셈형으로 결합한 **PDMS**
  `PDMS = NC × DAC × (5·EP + 5·TTC + 2·C.) / 12`
- **NAVSIM v2**: DDC(Driving Direction Compliance), TLC(Traffic Light Compliance), LK(Lane Keeping), HC(History Comfort), EC(Extended Comfort) 추가 → **EPDMS**
  `EPDMS = NC × DAC × DDC × TLC × (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC) / 16`
- **노이즈 강건성**: OpenScene test에서 비/안개 655 케이스를 수집해 별도 PDMS 측정 — 저자들이 "자율주행 world model 노이즈 강건성에 대한 최초의 종합적 정량·정성 분석"이라 주장

---

## 6. 실험 결과 심층 분석

### 6.1 NAVSIM v1 (Table 1, PDMS)

| 방법 | 계열 | 센서 | NC | DAC | TTC | C. | EP | **PDMS** |
|------|------|------|----|-----|-----|----|----|----------|
| Human | - | - | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| UniAD | E2E | 6× Cam | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| DiffusionDrive | E2E | 3× Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| AutoVLA | VLA | 3× Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive-8B | VLA | 3× Cam | 97.8 | 97.7 | 94.9 | 100.0 | 86.3 | 90.5 |
| DriveVLA-W0 | Pixel WM | 1× Cam | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| CoWorld-VLA | Pixel WM | 1× Cam | 99.2 | 96.8 | 96.6 | 100.0 | 83.6 | 89.8 |
| ResWorld | Latent WM | 4× Cam + L | 98.9 | 96.5 | 95.6 | 100.0 | 83.1 | 89.0 |
| DreamerAD | Latent WM | 1× Cam | 98.0 | 97.2 | 94.3 | 100.0 | 83.1 | 88.7 |
| **HyWorldVLA (Ours)** | **Hybrid WM** | **1× Cam** | 98.6 | 98.9 | 95.1 | 99.3 | **84.6** | **90.59** |

- 단안 카메라만으로 3× Cam + LiDAR 계열을 전부 상회. 인간(94.8)과는 여전히 4.2 포인트 격차.
- ReCogDrive-8B(90.5) 대비 이득은 **+0.09**에 불과 — v1은 이미 포화 구간이다.

### 6.2 NAVSIM v2 (Table 2, EPDMS)

| 방법 | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | **EPDMS** |
|------|----|-----|-----|-----|----|-----|----|----|----|-----------|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.4 | 98.3 | 87.2 | 76.7 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ReCogDrive | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | 99.5 | 86.5 | 83.6 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| ExploreVLA | 98.8 | 96.2 | 99.6 | 99.8 | 87.1 | 98.2 | 97.8 | 98.3 | 86.8 | 88.8 |
| Latent-WAM | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | 87.7 |
| **HyWorldVLA (Ours)** | 98.8 | 98.2 | 99.5 | 99.9 | 87.8 | 98.1 | 96.2 | 98.3 | 77.4 | **89.71** |

- v2에서는 격차가 벌어진다: ExploreVLA 대비 **+0.91**, Latent-WAM 대비 **+2.01**, DriveVLA-W0 대비 **+3.61**.
- 주목할 약점: **EC(Extended Comfort) 77.4**로 DiffusionDrive(87.7)·ExploreVLA(86.8)에 크게 밀린다. 곱셈이 아닌 가중합 항이라 총점 손실이 흡수됐을 뿐, 승차감 관점에서는 열위.

### 6.3 노이즈 강건성 (Table 6, 655 rain/fog 케이스)

| 방법/설정 | NC | DAC | TTC | C. | EP | **PDMS** |
|-----------|----|-----|-----|----|----|----------|
| WoTE | 98.9 | 76.0 | 64.4 | 100.0 | 64.4 | 60.65 |
| DriveVLA-W0 | 93.4 | 76.2 | 83.1 | 98.6 | 59.8 | 61.18 |
| DriveLaW | 99.4 | 75.7 | 82.9 | 100.0 | 69.5 | 67.49 |
| Pure WAM (ours 변형) | 99.1 | 75.7 | 94.3 | 99.8 | 67.8 | 69.95 |
| w/o latent supervision | 99.2 | 78.5 | 94.8 | 99.7 | 70.9 | 73.18 |
| **HyWorldVLA** | 99.7 | **91.1** | 96.9 | 99.2 | **84.4** | **86.87** |

**이 논문의 진짜 결과는 여기다.** 클린 셋에서 0.1~1점 다투던 격차가 노이즈 셋에서는 **+19.4 ~ +26.2 포인트**로 폭발한다. 특히 DAC(주행가능영역 준수)에서 76 → 91.1로, 경쟁 모델들이 노이즈 하에서 차선/도로 경계를 잃는 반면 HyWorldVLA는 유지한다. 자체 변형(Pure WAM 69.95, w/o latent supervision 73.18)과 비교해도 latent 설계가 강건성의 원인임이 분리된다.

### 6.4 정성 분석
- Fig. 3a: 맑음 → 흐림으로 전방 조도를 점진 변화시킬 때 DriveVLA-W0는 행동 분산이 커지지만 HyWorldVLA는 일관된 궤적 유지
- Fig. 3b: 비·안개 비균일 노이즈에서 DriveVLA-W0는 과도하게 보수적(효율 희생), HyWorldVLA는 인간 정렬 궤적 유지

---

## 7. Ablation 분석

### 7.1 컴포넌트 (Table 3, NAVSIM PDMS)

| Config | PDMS | Δ |
|--------|------|---|
| Pure LWM (pixel regression 제거) | 87.50 | **−3.09** |
| Pure WAM (latent 표현 제거) | 89.91 | −0.68 |
| w/o latent supervision in co-fine-tuning | 90.17 | −0.42 |
| w/o latent condition in action expert | 90.29 | −0.30 |
| w/o language guidance in latent | 90.35 | −0.24 |
| **Full** | **90.59** | — |

→ 클린 셋 성능의 지배적 기여자는 **pixel-level 시공간 모델링**(−3.09). latent 계열 요소들은 각각 0.2~0.7 수준의 기여에 그친다. 즉 **latent의 가치는 PDMS가 아니라 Table 6의 강건성에서 회수된다** — 두 표를 함께 봐야 논문의 주장이 성립한다.

### 7.2 사전학습 loss 가중치 (Table 4)

| λ1 (visual) | λ2 (latent) | PDMS |
|-----|-----|------|
| 0.1 | 0.1 | 90.48 |
| **0.5** | **0.1** | **90.59** |
| 1.0 | 0.1 | 89.83 |
| 0.5 | 0.2 | 90.47 |
| 0.5 | 0.5 | 90.34 |
| 0.5 | 1.0 | 90.16 |

visual token 가중치가 너무 크면(1.0) 급락(−0.76). latent 가중치는 0.1 초과 시 단조 감소.

### 7.3 Co-fine-tuning loss 가중치 (Table 5)

| λ3 | PDMS |
|----|------|
| 0.05 | 90.47 |
| **0.1** | **90.59** |
| 0.2 | 90.01 |
| 1.0 | 89.75 |

적정 supervision은 latent 공간의 semantic collapse를 막지만, 과한 supervision은 action 생성에 필요한 latent semantics의 자율 학습을 방해.

---

## 8. 관련 연구 비교

| 축 | DriveVLA-W0 | Latent-WAM / LAW / Epona | OneVL | **HyWorldVLA** |
|----|-------------|--------------------------|-------|----------------|
| 미래 표현 | pixel(미래 이미지) | latent only | compact latent token | **pretrain: pixel+latent / finetune: latent** |
| pixel grounding | 있음(추론까지) | 없음 | 없음 | 사전학습에만 |
| 노이즈 강건성 | 61.18 PDMS | (Pure WAM 대리 69.95) | 미보고 | **86.87** |
| 센서 | 1× Cam | 1~4× Cam (+L) | 1× Cam | 1× Cam |
| action head | AR / diffusion | 다양 | - | flow matching + selection-based 양쪽 |

포지셔닝은 명확하다. "pixel이냐 latent냐"의 이분법을 **학습 스테이지 분할**로 해소했고, 그 효과의 증거를 클린 벤치마크가 아니라 **자체 구축 노이즈 셋**에서 확보했다.

---

## 9. 한계 및 미해결 문제

1. **단안 전방 카메라 제약** — Appendix 실패 사례(Fig. 9, 10)에서 좌/우회전 시 시야 밖 목표 차선 정보 부재로 중앙 이중 황색선을 넘는 사례 보고. world model 예측에도 이중선이 나타나지 않음. 저자 스스로 multi-view 필요성을 인정하되 camera 수 vs latency trade-off는 미해결이라 명시.
2. **Extended Comfort 열위 (77.4)** — 승차감 지표에서 DiffusionDrive(87.7) 대비 10 포인트 이상 뒤진다. flow matching / selection-based 어느 쪽이 최종 수치인지도 본문에서 불명확.
3. **노이즈 테스트셋이 자체 구축** — 655 케이스 선별 기준·공개 여부 불명. 저자 방법에 유리한 분포일 가능성을 외부에서 검증 불가.
4. **비용·속도 미보고** — 파라미터 수, 추론 지연, PPU-시간 총량이 전혀 없다. 실차 배포 판단 근거가 부족.
5. **코드/가중치 미공개** — 재현 경로 없음.
6. **NAVSIM v1 이득 미미** — ReCogDrive-8B 대비 +0.09. v1만 보면 통계적 유의성을 주장하기 어렵다(seed·분산 미보고).
7. **로봇 조작 도메인 미검증** — LIBERO/CALVIN/SimplerEnv 등 manipulation 벤치마크 결과 없음. VLA-Tracker의 표준 벤치마크 축과 직접 비교 불가.

---

## 10. 총평

**강점**
- pixel/latent 논쟁을 "둘 중 하나"가 아니라 "학습 스테이지별 역할 분담"으로 재정의한 점이 깔끔하다. 사전학습의 pixel supervision을 latent 붕괴 방지용 regularizer로 격하시킨 해석이 설득력 있다.
- Table 6이 논문 전체를 지탱한다. 클린 벤치마크 0.1점 싸움이 아니라 노이즈 조건에서 20점 이상 격차를 보인 것은 실무적으로 의미가 크다.
- action expert를 flow matching과 selection-based 양쪽에 이식해 방법의 범용성을 보인 것도 좋다.
- 단안 카메라만으로 LiDAR 포함 baseline을 넘긴 센서 효율성.

**약점**
- 클린 성능 이득은 사실상 노이즈가 아닌 pixel 항에서 나오고(Table 3), latent 기여는 강건성에 국한된다. 제목의 "hybrid"가 성능 향상 서사와 정확히 맞물리지는 않는다.
- 자체 노이즈 셋 의존, 비용/속도 미보고, 코드 미공개의 3중 재현성 공백.
- EC 지표 열위는 실차에서 체감되는 문제인데 논의가 없다.

**평점 (주관)**: 아이디어 명료성 ★★★★☆ / 실험 설득력 ★★★★☆ / 재현성 ★★☆☆☆ / 실용성 ★★★☆☆

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 예상 답변 / 논문 근거 |
|---|------|----------------------|
| 1 | NAVSIM v1에서 ReCogDrive-8B 대비 +0.09인데 SOTA라 부를 수 있나? | v1은 포화 구간. 저자의 실질 주장은 v2(+0.91 vs ExploreVLA)와 노이즈 셋(+19.4 vs DriveLaW). 다만 seed 분산 미보고라 v1 우위는 노이즈 수준 |
| 2 | 추론 시 pixel 예측을 안 쓴다면 사전학습 pixel supervision이 정말 살아있나? | Table 3의 Pure LWM(pixel 제거) 87.50 vs full 90.59 = −3.09가 직접 증거. latent 임베딩 품질에 흔적이 남는다는 주장 |
| 3 | 노이즈 강건성이 latent 때문인지, 단순히 학습 데이터가 많아서인지? | 동일 학습 조건의 Pure WAM 69.95, w/o latent supervision 73.18과 비교 → 데이터가 아니라 latent 설계가 원인으로 분리됨 |
| 4 | 최종 90.59는 flow matching인가 selection-based인가? | 본문에 명시되지 않음. 두 계열 모두 이식 가능하다고만 서술 — **논문의 실질적 공백** |
| 5 | Extended Comfort 77.4는 왜 이렇게 낮은가? | 논문에 설명 없음. latent 조건화가 궤적 부드러움보다 진행/준수 지표를 우선하도록 편향시켰을 가능성 |
| 6 | 655 케이스 노이즈 셋을 공개하는가? | 언급 없음. "새로운 벤치마크를 확립했다"는 주장 대비 공개 계획 부재는 약점 |
| 7 | Flan-T5 텍스트 유도가 정말 필요한가? | w/o language guidance in latent = 90.35 (−0.24). 기여는 작지만 일관되게 양(+). VAE 재구성 품질(ghosting/flicker) 개선이 주 효과라고 서술 |
| 8 | λ1을 1.0으로 올리면 왜 89.83으로 급락하나? | visual token CE가 커지면 backbone이 픽셀 재구성에 과적합되어 행동 관련 semantics를 잃는 것으로 해석. λ1 = 0.5, λ2 = 0.1이 sweet spot |
| 9 | 실차 배포 가능한가? | 파라미터·지연 미보고. Emu3 기반 AR backbone + video VAE 조합은 경량이라 보기 어려움. 저자도 camera 수 vs latency를 미해결 과제로 명시 |
| 10 | 로봇 조작 VLA로 전이 가능한가? | 검증 없음. FAST tokenizer, π₀식 joint attention 등 manipulation VLA 구성요소를 그대로 차용했으므로 이식 가능성은 높지만 실험 부재 |

---

## 12. VLA-Tracker 등재 판정

- **판정: ACCEPTED** — 분석/probing/벤치마크/프레임워크 논문이 아니라, 실제로 학습되어 궤적(행동)을 출력하는 **VLA policy 모델**이다.
- **도메인**: 자율주행 (NAVSIM v1/v2). LIBERO·CALVIN·SimplerEnv 등 트래커 표준 manipulation 벤치마크 점수는 **없음** → YAML의 `benchmarks`는 `{}`로 두고, NAVSIM 수치는 `training`/`eval_conditions`에 기록 (LVDrive, UniDriveVLA 등 기존 주행 VLA 항목과 동일한 관례).
- **action_head_category**: `hybrid` — 사전학습은 FAST 토큰 기반 autoregressive 행동 예측, co-fine-tuning은 joint attention action expert(flow matching / selection-based)로 연속 궤적 생성.
- **주의**: 파라미터 수, 추론 Hz, 코드 URL은 논문에 없으므로 `null` 유지. 추후 공개 시 갱신 필요.

<!-- VERIFIED: pdf -->
