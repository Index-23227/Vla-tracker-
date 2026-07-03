# ZR-0: Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision

> **한 줄 요약**: Qwen3-VL-2B VLM(System 2)과 500M DiT flow-matching action expert(System 1)를 cross-attention으로 결합한 2.6B VLA로, ProcCorpus-60M(약 6,000만 프레임, 약 1,000시간, ECoT 커버리지 96.8%)의 dense Embodied CoT 감독으로 cross-embodiment 표현을 정렬하고, attention mask 덕분에 추론 시 ECoT 생성을 완전히 생략하면서 LIBERO 97.8%, RoboTwin 2.0 88.70/87.98%(Clean/Rand.), RoboCasa GR-1 Tabletop 69.3%를 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Cross-embodiment transfer의 근본 난점**: 로봇 플랫폼마다 kinematic 구성(6-DoF vs 7-DoF), 제어 인터페이스(joint position vs EE pose), base 타입, 센서 구성이 달라 state/action 공간이 이질적
- 기존 해법은 **format-level**에 머묾: zero-padding·per-embodiment normalization(π0, GR00T N1), 또는 고정 semantic role의 unified action space(RDT-1B 등). 그러나 같은 차원(예: joint 1)이라도 로봇마다 회전축·범위가 달라 **semantic alignment**은 미해결
- Textual/embodied CoT VLA(RT-2 계열, ECoT, CoT-VLA)는 추론 시 autoregressive 텍스트 생성 오버헤드가 큼

### 핵심 관찰
- 저수준 state/action은 embodiment-specific이지만, **조작의 고수준 인지 과정**(장면 인식, 태스크 진행 판단, 계획, 서브태스크 분해, 대상 객체 식별)은 embodiment 간 **공유**됨 → 이것이 cross-embodiment 사전학습이 포착해야 할 전이 가능 지식
- 컵을 집는 인지 궤적은 팔이 6-DoF든 7-DoF든 유사하다

📌 [Figure 1 삽입] — ZR-0 프레임워크: VLM(ECoT next-token prediction) + DiT action expert(denoising vector field prediction), self-attention 1 : cross-attention 3 반복 블록

---

## 2. 방법론 심층 분석

### 2.1 Dual-Stream 아키텍처 (System 1 / System 2)

- **System 2 (VLM)**: Qwen3-VL-2B-Instruct(2.1B)에서 초기화. 태스크 지시 l과 n개 카메라 뷰 관측 o_t를 받아 구조화된 ECoT reasoning 시퀀스 r_t를 생성하도록 학습. 마지막 레이어 hidden states f_t를 action expert에 전달. 입력 이미지는 224×224로 리사이즈
- **System 1 (DiT action expert, ~500M)**: state encoder + action encoder(MLP), DiT 블록 스택, action decoder(MLP)로 구성. VLM feature f_t와 robot state s_t를 조건으로 flow matching으로 H-step 연속 action chunk 예측

### 2.2 1:3 Self/Cross-Attention 비율과 추론 시 ECoT 생략

- DiT 블록은 **self-attention 1층 + cross-attention 3층** 반복 패턴. GR00T N1의 1:1 비율 대비 cross-modal 상호작용 비중을 높임
- **핵심 설계**: cross-attention에서 action expert가 VLM의 **입력 프롬프트(지시+이미지) feature에만** attend하도록 마스킹, ECoT 토큰 feature는 제외
- 그 결과 추론 시 VLM forward **1회**로 action expert에 필요한 모든 feature가 나옴 → autoregressive ECoT 디코딩 전면 생략, 성능 손실 없음 (저자 주장)

> ❓ **예상 질문**: ECoT feature를 action expert가 못 보는데 ECoT 감독이 왜 액션에 도움이 되나?
> **답변**: L_ntp는 VLM 파라미터만 업데이트하지만, 그 gradient가 VLM의 표현 자체를 embodiment-agnostic하게 정렬시킨다. Action expert는 그 정렬된 입력 프롬프트 feature(f_t)를 소비하므로 간접적으로 혜택을 받는다. Table 5 ablation(ECoT PT 제거 시 LIBERO 97.8→95.7)이 이를 뒷받침.

### 2.3 학습 목적함수

- **ECoT**: 표준 next-token prediction loss L_ntp (VLM 파라미터만 갱신)
- **액션**: flow matching. 노이즈 chunk A^τ_t = (1−τ)ε + τA_t 구성, denoising vector field (A_t − ε)를 L2로 회귀. τ ~ Beta(1.5, 1.0)으로 noisy 구간 강조 (π0 따름)
- 전체 loss: L = L_ntp + α·L_fm. L_fm은 action expert와 VLM 모두에 gradient 전파(f_t 경유)

### 2.4 추론

- A^0_t ~ N(0, I)에서 시작해 forward Euler로 N-step 적분
- **A6000 단일 GPU, bfloat16에서 action chunk 생성 약 90ms** (Conclusion에서는 H100 기준 약 100ms/chunk로 표기 — 본문과 수치 표기 상이함에 유의)

---

## 3. 데이터 전략

### ProcCorpus-60M (사전학습)
- **약 6,000만 프레임 / 약 1,000시간 / 40만+ trajectory**: DROID, Bridge, Fractal, RH20T, Open X-Embodiment 서브셋 등 집계
- **96.8% 프레임에 dense ECoT 주석** (자동 VLM 기반 파이프라인, ProcVLM [22] 논문의 데이터셋)
- ECoT 6개 구성요소와 역할:

| 구성요소 | 형식 | 역할 |
|----------|------|------|
| Scene Description | 텍스트 장면 묘사 | 객체 인식 강화 |
| Progress Assessment | 추론 + Yes/No 완료 지표 | 태스크 진행 인지 |
| Future Plan | 자유형 자연어 | temporal reasoning, long-horizon 계획 |
| To-Do Actions | Verb+Object 명령문 리스트 | **embodiment-agnostic 서브태스크 분해 = cross-embodiment 정렬의 핵심 기제** |
| Target Objects | JSON bounding box | 시각적 grounding, 시점/배치 일반화 |
| Discrete Actions | FAST tokenizer 토큰 | 고수준 추론↔저수준 제어 교량 |

### VL 데이터 co-training
- CapsFusion, Pixmo 등 일반 VL 데이터를 혼합 (VLM만 언어모델링으로 학습, 액션 예측 없음) → catastrophic forgetting 완화, 새로운 장면·다양한 지시 robust성 향상

### Post-training
- 각 벤치마크 공개 학습 데이터만 사용, **ECoT·VL 데이터 없이** 표준 프로토콜 — ECoT는 사전학습 전용

> ❓ **예상 질문**: 1,000시간이면 다른 파운데이션 VLA 대비 작지 않나?
> **답변**: 저자 스스로 인정 — π0(1만+ 시간), LingBot-VLA(약 2만 시간), Qwen-RobotManip(3만+ 시간) 대비 한 자릿수 적음. 그럼에도 RoboTwin 2.0에서 LingBot-VLA를 근소하게 상회(88.70 vs 88.56 Clean)한다는 점이 ECoT 감독의 데이터 효율 논거.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 2.6B (VLM 2.1B + action expert 500M) |
| VLM 초기화 | Qwen3-VL-2B-Instruct |
| 사전학습 chunk 길이 | H=32 |
| 사전학습 batch | 1,024 (global) |
| Loss weight | α=5 (사전학습), α=1 (post-training) |
| State/action 패딩 | 64차원 zero-pad + padded 차원 loss mask |
| 정규화 | 1/99 percentile min-max per-dim |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| LR | cosine, warm-up 5%, peak 3e-5 → 3e-6 |
| 정밀도 | bfloat16, gradient clipping 1.0 |
| 인프라 | DeepSpeed ZeRO, Flash-Attention 2, gradient checkpointing |
| Post-training | batch 64, H=10(LIBERO) / H=16(RoboTwin 2.0, RoboCasa, xArm) |

---

## 5. 실험 설계 및 평가 프로토콜

세 가지 embodiment의 시뮬레이션 + 실기 실험:
1. **LIBERO** (single-arm): 4 suite, 40 태스크 1,693 trajectory로 단일 모델 학습, 50 episodes/task
2. **RoboTwin 2.0** (bimanual, ALOHA): 50 태스크 전부, clean 50 + randomized 500 demos/task = 27,500 demos 병합 학습, 100 episodes/task, Clean/Randomized 양 조건 평가
3. **RoboCasa GR-1 Tabletop** (humanoid): 저자들이 도입한 평가 세팅, 24 태스크, 100 episodes/task
4. **실기 xArm**: 4 태스크(Push Blocks, Clean Table, Pick & Place, Hang Cups), 2,000+ teleop trajectory(5Hz, 50+ 객체), 10 trials/task, 0~100 progress score 루브릭

**모두 동일한 사전학습 체크포인트에서 fine-tune** — cross-embodiment 적응력 검증 설계.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1)

| 모델 | Spatial | Object | Goal | LIBERO-10 | Avg |
|------|---------|--------|------|-----------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.7 | 97.7 | 97.5 | 98.5 | 94.4 | 97.0 |
| MolmoAct2 | 97.8 | 100.0 | 97.8 | 93.2 | 97.2 |
| **ZR-0** | 97.4 | 99.4 | 98.0 | **96.4** | **97.8** |

- 차별화 포인트는 **LIBERO-10(long-horizon)**: 96.4%로 π0.5 대비 +4.0pt. 나머지 3개 suite는 최근 방법들 사이에서 포화 상태

### RoboCasa GR-1 Tabletop (Table 2)

| 모델 | GR00T-N1.6 | Qwen3-VLA | PI-M0(ABot) | VP-VLA | JoyAI-RA | **ZR-0** |
|------|-----------|-----------|-------------|--------|----------|----------|
| Avg | 47.6 | 43.9 | 53.8 | 58.3 | 63.2 | **69.3** |

- 차순위 JoyAI-RA 대비 **+6.1pt**. Pick-and-place 계열에서 큰 우위(CuttingboardToTieredbasket 80 vs 36, PlacematToPlate 88 vs 38, PlateToPan 89 vs 46)
- 그러나 **Close 계열 6개 태스크에서는 열세**(BottleToCabinetClose 39 vs 84, CanToDrawerClose 47 vs 90) — 사전학습 코퍼스에서 closing primitive가 희소한 탓으로 저자 해석

### RoboTwin 2.0 (Table 3/6, 50 태스크)

| 모델 | π0 | π0.5 | X-VLA | Motus | LingBot-VLA | **ZR-0** |
|------|-----|------|-------|-------|-------------|----------|
| Clean | 65.92 | 82.74 | 72.80 | 88.66 | 88.56 | **88.70** |
| Rand. | 58.40 | 76.76 | 72.84 | 87.02 | 86.68 | **87.98** |

- 약 1,000시간 사전학습으로 2만 시간의 LingBot-VLA를 근소 상회
- **Clean→Randomized 하락이 0.72pt에 불과** (Motus 1.64, π0.5 5.98) → 시각적 변동(clutter, 조명, 배경) robust성. VL co-training 효과로 해석
- Bimanual 협조 태스크 강세: HandoverMic 100/99, PickDualBottles 97/98. 일부 태스크는 Randomized가 Clean보다 높음(BlocksRankingSize 70→81, StackBowlsThree 79→88)

### 실기 xArm (Table 4, progress score)

| 방법 | Pick & Place | Hang Cups | Clean Table | Push Blocks | Avg |
|------|-------------|-----------|-------------|-------------|-----|
| π0.5 | 56.7 | **85.0** | 63.3 | 66.1 | 67.8 |
| **ZR-0** | **66.7** | 70.0 | **73.4** | **94.0** | **76.0** |

- Push Blocks(OCR 추론)에서 **+27.9pt** — VL co-training이 VLM의 텍스트 인식 능력 보존
- 단 **Hang Cups에서는 π0.5에 역전**(70.0 vs 85.0) — 고정밀 dexterous 제어는 고수준 추론보다 액션 감독 스케일에 의존한다는 시사점

---

## 7. Ablation 분석

### ECoT 사전학습 유무 (Table 5, LIBERO)

| 설정 | Spatial | Object | Goal | LIBERO-10 | Avg |
|------|---------|--------|------|-----------|-----|
| **ZR-0** | 97.4 | 99.4 | 98.0 | 96.4 | **97.8** |
| w/o ECoT PT | 96.8 | 98.6 | 94.8 | 92.6 | 95.7 |

- ECoT 사전학습 제거(Qwen3-VL base + 랜덤 초기화 action expert로 LIBERO 직접 fine-tune) 시 **−2.1pt**, 특히 Goal(−3.2)과 LIBERO-10(−3.8)에서 하락 → long-horizon·목표 지향 태스크에서 ECoT 표현 정렬 효과가 큼

> 📛 **주의**: ablation이 이 한 개뿐이다. ECoT 6개 구성요소 각각의 기여도, 1:3 attention 비율 vs 1:1, α 민감도, VL co-training 단독 효과 등의 분해 실험은 없음.

---

## 8. 관련 연구 비교

| 모델 | 아키텍처 | ECoT/추론 활용 | 추론 시 텍스트 생성 |
|------|---------|---------------|--------------------|
| RT-2 / OpenVLA / FAST | 이산 액션 토큰 autoregressive | ✗ | 액션 토큰 디코딩 |
| π0 | MoT + flow matching expert | ✗ | ✗ |
| π0.5 | π0 + 고수준 subtask planning | 추론 시 subtask 생성 | ✓ |
| GR00T N1 | cross-attention DiT (1:1 비율) | ✗ | ✗ |
| ECoT (Zawalski et al.) | autoregressive | 추론 시 ECoT 생성 | ✓ (느림) |
| **ZR-0** | cross-attention DiT (1:3 비율) | **학습 시에만 dense ECoT** | **✗ (마스크로 생략)** |

### 핵심 차이
- ECoT를 **inference-time 추론이 아니라 training-time 표현 정렬 신호**로 사용 — "reasoning의 대표성 이득은 취하고 latency 비용은 버리는" 설계
- GR00T N1과 같은 dual-stream 계보이나, ECoT 감독 + 1:3 attention 비율로 차별화
- Chen et al.의 training-strategies 연구[13]에서 확립된 "ECoT gradient가 표현을 개선한다"는 관찰을 대규모(60M 프레임) cross-embodiment로 확장

---

## 9. 한계 및 미해결 문제

### 저자가 인정한 한계 (Discussion)
1. **데이터 스케일**: 약 1,000시간은 π0(1만+), LingBot-VLA(약 2만), Qwen-RobotManip(3만+) 대비 한 자릿수 부족. 사전학습에 희소한 skill(closing 등)은 약함 — RoboCasa Close 태스크 열세가 증거
2. **정밀 제어**: Hang Cups 역전이 보여주듯, dexterous 정밀 조작은 ECoT 고수준 추론만으로 부족
3. **ECoT 주석 비용**: 프레임마다 VLM forward가 필요 — informative frame 선택 전략은 future work

### 리뷰어 관점의 추가 미비점
4. **Ablation 빈약**: ECoT PT on/off 단일 ablation. 6개 ECoT 구성요소별 기여, attention 비율, VL co-training 분리 실험 부재
5. **"성능 손실 없는 ECoT 생략" 주장의 직접 검증 부재**: 추론 시 ECoT를 생성해 attend하게 한 변형과의 비교 표가 없음
6. **cross-embodiment transfer의 직접 측정 부재**: 주장의 핵심인 "한 embodiment 지식이 다른 embodiment에 전이"를 leave-one-embodiment-out 등으로 직접 검증하지 않음 — 세 벤치마크 모두 벤치마크별 fine-tune 후 성능
7. 추론 latency 표기가 본문(A6000, ~90ms)과 결론(H100, ~100ms)에서 GPU·수치가 어긋남

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — dual-stream + flow matching 자체는 π0/GR00T 계보. "dense ECoT를 학습 전용 정렬 신호로 + 마스크로 추론 시 생략"의 조합이 기여 |
| **Technical depth** | ★★★★☆ — 학습 목적함수, 마스킹 설계, 데이터 파이프라인이 명료하고 재현 정보 풍부 |
| **Experimental rigor** | ★★★★☆ — 3개 embodiment 시뮬 + 실기, 강한 baseline 비교. 단 ablation은 1개로 빈약 |
| **Practical impact** | ★★★★☆ — 2.6B로 ECoT 없이 90ms/chunk 추론, 코드·체크포인트 공개, 1,000시간으로 2만 시간급과 경쟁 |
| **Writing quality** | ★★★★☆ — 관찰→설계→검증의 논리 전개가 깔끔, 실패 사례(Close 태스크, Hang Cups) 분석 정직 |

**강점**: ECoT의 표현 학습 이득과 추론 효율을 attention mask 하나로 분리한 실용적 설계. 동일 체크포인트에서 single-arm/bimanual/humanoid 3개 embodiment 모두 경쟁력 있는 결과(LIBERO 97.8 / RoboTwin 2.0 88.70/87.98 / RoboCasa 69.3). 데이터 효율(1,000h) 대비 성능이 인상적.
**약점**: ablation 부족으로 6개 ECoT 구성요소 중 무엇이 실제로 기여하는지 불명. cross-embodiment transfer 자체의 직접 실증 부재. 정밀 조작·희소 skill 약점.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | ECoT를 추론에서 생략해도 정말 손실이 없나? 직접 비교 실험은? | 논문에 해당 비교표는 없음. 설계상 action expert가 ECoT feature에 attend하지 않도록 학습되므로 분포 불일치는 없다는 논리. Table 5는 사전학습 유무 비교일 뿐 |
| 2 | 성능 이득이 ECoT 덕인가, 그냥 1,000시간 로봇 사전학습 덕인가? | Table 5의 w/o ECoT PT는 사전학습 전체를 제거한 것 — "액션만으로 동일 코퍼스 사전학습" 대조군이 없어 ECoT 감독 자체의 기여를 완전히 분리하지 못함. 논문의 가장 취약한 지점 |
| 3 | 1:3 cross-attention 비율의 근거는? | GR00T N1의 1:1 대비 cross-modal 상호작용 비중 증가라는 직관 제시. ablation 없음 |
| 4 | RoboCasa Close 태스크 열세의 원인은? | 사전학습 코퍼스에서 closing primitive 희소 → ECoT 정렬이 형성되지 못함. 데이터 커버리지가 ECoT 이득의 전제조건임을 보여줌 |
| 5 | Discrete Actions(FAST 토큰)를 ECoT에 넣는 이유는? action expert가 따로 있는데 중복 아닌가? | 고수준 embodiment-agnostic 추론과 저수준 제어 사이의 교량으로 VLM 표현에 액션 구조를 주입하는 역할. 실제 제어는 flow matching expert가 담당 |
| 6 | LingBot-VLA(2만h) 대비 우위가 0.14pt(Clean)인데 유의미한가? | 절대 우위보다 **20배 적은 데이터로 동급**이라는 데이터 효율이 논지. Randomized에서는 +1.30pt로 격차 약간 더 큼 |
| 7 | 실기에서 π0.5에 Hang Cups로 지는 건 ECoT 접근의 본질적 한계인가? | 저자 해석: 고정밀 모터 제어는 고수준 추론이 아니라 액션 감독 스케일 문제. ECoT는 인지 정렬 도구이지 정밀도 도구가 아님 |
| 8 | 사람 egocentric 비디오로 확장 가능하다는 주장의 근거는? | ECoT 구성요소가 행위 주체(로봇/사람)에 무관한 구조라는 점. Ego4D/EPIC-KITCHENS에 ECoT 주석을 달면 액션 라벨 없이 VLM 표현 학습 가능 — 단 아직 Discussion 수준의 제안 |

---

## 12. 레퍼런스 및 리소스

- **논문**: https://arxiv.org/abs/2606.30552 (v2, 2026-07-01)
- **코드/체크포인트**: https://github.com/RUCKBReasoning/ZR-0
- **사전학습 데이터**: ProcCorpus-60M — ProcVLM (Feng et al., arXiv:2605.08774)의 자동 ECoT 주석 파이프라인 산출물
- **소속**: Renmin University of China, Zhipu AI
- **주요 관련 연구**: π0 (arXiv:2410.24164), π0.5 (arXiv:2504.16054), GR00T N1 (arXiv:2503.14734), ECoT (Zawalski et al., CoRL 2024), FAST (arXiv:2501.09747), RoboTwin 2.0 (arXiv:2506.18088), RoboCasa (RSS 2024)

<!-- VERIFIED: pdf -->
