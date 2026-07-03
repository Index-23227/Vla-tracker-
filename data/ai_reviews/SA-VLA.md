# SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance

> **한 줄 요약**: Discrete action tokenizer가 각 코드를 고정된 continuous action prototype에 매핑하는 한계("같은 delta action이라도 관절 상태·접촉 조건에 따라 다른 물리적 결과를 낳는다")를 지적하고, VQ-VAE 기반 action tokenizer에 **로봇 proprioceptive state를 주입**하는 SA-VLA를 제안. 두 가지 주입 방식 — (A) state-action cross-attention, (B) 경량 adapter가 state로부터 action 차원별 scaling factor w를 예측해 `a_trans = a ÷ w`로 변조 후 양자화 — 중 **Method B**가 최고 성능. 유한 codebook의 각 token이 "state에 따라 달라지는 continuous action의 family"를 표현하게 되어 사실상 연속 action 공간으로 확장. Qwen2.5 기반 VLA에 통합해 RoboTwin 12개 태스크에서 평균 성공률 **0.29(VQ-BET) → 0.56**, zero-shot sim-to-real 3개 태스크에서 **0.15 → 0.33** 달성.

---

## 1. 배경 및 동기

- **VLA의 두 갈래**: (i) discrete action token을 생성해 tokenizer로 연속 action을 복원하는 계열(OpenVLA, VQ-VLA), (ii) 연속 action을 직접 생성하는 계열(π0). 본 논문은 (i) 계열에서 **discrete token → continuous action 복원 과정이 성능 병목**이라고 진단.
- **기존 tokenizer의 한계**:
  - OpenVLA의 uniform binning: 구현이 단순하나 tokenization granularity가 낮음.
  - FAST: DCT + BPE로 압축 성능은 좋으나, BPE 특성상 같은 길이의 action sequence가 서로 다른 길이의 token sequence로 매핑되어 학습이 어려움.
  - VQ-VLA의 residual VQ-VAE: data-driven이지만 압축 성능에 개선 여지.
- **핵심 관찰**: delta action을 쓸 때, 동일한 delta action이 관절 구성·물체 pose·접촉 조건에 따라 서로 다른 물리적 결과를 낳는다. 예: 병을 잡는 동일한 delta action도 grasp 위치에 따라 성공/실패가 갈리는데, 이 서로 다른 상황의 action이 모두 같은 token으로 매핑되면 성능이 저하됨. → **state 정보를 tokenizer에 주입해야 한다.**
- 기존 tokenizer들이 NLP/CV의 tokenization 기법을 차용할 뿐 VLA 고유 특성(state 의존성)을 무시한다는 문제의식.

---

## 2. 방법론 심층 분석

### 2.1 문제 설정
데이터셋 {(o_t, s_t, a_t)}과 언어 지시 L이 주어질 때, 정책 π_θ(a_{t:t+k} | o_t, s_t, L)로 향후 k-step action block을 예측. s_t는 관절 회전각 등 proprioceptive state.

### 2.2 Base tokenizer
CNN + Transformer encoder로 temporal action feature 추출 → codebook 양자화(q_i = argmin ||z_i − ẑ_i||²) → 대칭 구조 decoder로 복원하는 표준 VQ-VAE. Loss는 reconstruction + codebook(sg) + commitment의 가중합:
`L = ||â − a||² + λ1·||sg(x) − q(x)||² + λ2·||x − sg(q(x))||²`

### 2.3 Method A: State-Action Cross-Attention
π0에서 VLM token과 action expert의 상호작용에서 영감. Encoder/decoder의 2-layer Transformer에서 **key/value를 state 입력에서 유도**하여 cross-attention으로 state를 주입. Decoder에서도 raw state를 K/V로 사용.

### 2.4 Method B: 경량 State Adapter (핵심 기여)
"손이 컵에 다가가면 잡을 확률이 높다"는 인간 행동 습관에서 영감. MLP + sigmoid로 구성된 경량 adapter가 state로부터 **action 각 차원의 scaling factor w**를 예측:
1. 변조: `a_trans = a ÷ w` → 표준 encoder(이때 Transformer는 state 미사용)
2. 양자화 → decoder → `â_trans`
3. 복원: `â = â_trans × w` (대칭성 유지)

**해석**: VQ-VAE decoding은 codebook 용량에 갇혀 token↔action이 1:1 대응이지만, Method B는 discrete token에 state-의존 scale을 곱하므로 **같은 token이 state에 따라 여러 action을 생성** — token은 "action의 클래스", state network는 "그 클래스의 scale"을 담당. 이산 VQ-VAE를 regression 문제로 재정식화해 연속 action 공간으로 확장하면서 고정 codebook의 이점(효율, 호환성)은 유지.

### 2.5 VLA 통합
- 텍스트: base LLM native tokenizer / state: FAST 방식으로 차원별 256-bin 이산화 / 이미지: 224×224 resize 후 SigLIP-SO400M-patch14-224로 16×16 grid = 프레임당 256 continuous image token / action: SA-VLA tokenizer.
- 모달리티 경계용 special token: `t_bos/t_eos`, `s_bos/s_eos`, `i_bos/i_eos`, `a_bos/a_eos`로 단일 시퀀스 직렬화.
- **두 decoding 전략**: (i) Autoregressive — next-token NLL (Eq.1), (ii) Parallel Decoding — 고정 길이 token 덕분에 bidirectional attention + placeholder embedding으로 action block 전체를 1회 forward로 생성 (Eq.2). LLM 본체는 불변, 입력 인터페이스와 attention mask만 최소 수정.

---

## 3. 데이터 전략

- RoboTwin 시뮬레이터, **clean mode**, 12개 bimanual 태스크 × 태스크당 1,600 demo = **총 19,200 trajectory**.
- 플랫폼: Piper 매니퓰레이터 2대(간격 0.6m). 수집 설정: save frequency 15, random background/light True, cluttered table False, D435 head camera (Table A.1).
- 태스크 평균 길이 52~141 step (Handover Mic 134, Place Burger Fries 141이 최장; Table A.2).
- 실로봇 데이터 없음 — real-world는 순수 zero-shot sim-to-real.

---

## 4. 시스템/학습 세부사항

- **2-stage 학습** (Table A.3):
  - Stage 1 (tokenizer): batch 1024, 200 epoch, AdamW lr 5e-5 cosine, commitment coef 1.0, adversarial coef 0.1, random horizontal flip.
  - Stage 2 (VLA): batch 64, 10 epoch, AdamW lr 1e-4 cosine, commitment coef 1000.0, adversarial coef 1.0.
- Backbone: Qwen2.5 기반 VLM (정확한 파라미터 수 미명시).
- 평가: 단일 RTX 4090 GPU, RoboTwin 공식 평가 프로토콜.
- 실로봇: AgileX Cobot Magic 모바일 플랫폼, Aloha 구성 4-arm(각 AgileX Piper 6-DoF + 1-DoF parallel gripper), RealSense D435 (640×480, ~30Hz).

---

## 5. 실험 설계 및 평가 프로토콜

- **시뮬레이션**: 학습에 쓴 동일 12개 RoboTwin 태스크에서 태스크당 100 rollout, 성공률 측정. 비교군은 **동일 Qwen2.5 VLM + tokenizer만 교체**: (1) Binning(OpenVLA), (2) FAST, (3) VQ-BET(VQ-VLA), (4) Method A, (5) Method B — 모두 동일 하이퍼파라미터. tokenizer 효과를 깨끗하게 분리한 controlled comparison.
- **실세계**: RoboTwin sim RGB로만 학습한 VLA를 그대로 실로봇 3개 태스크(Click Bell, Place Container Plate, Pick Diverse Bottles)에 **zero-shot** 적용, 태스크당 20 trial.

---

## 6. 실험 결과 심층 분석

### 6.1 시뮬레이션 (Table 4.1, 12태스크 평균)

| Tokenizer | 평균 성공률 |
|---|---:|
| FAST | 0.17 |
| Binning | 0.24 |
| VQ-BET | 0.29 |
| Method A (AR / PD) | 0.55 / 0.52 |
| **Method B (AR / PD)** | **0.56 / 0.56** |

- Method B(AR) 0.56은 Binning 대비 +23%p, FAST 대비 +40%p, VQ-BET 대비 +28%p (abstract의 "0.29→0.56"은 최강 baseline VQ-BET 대비).
- 태스크별로 격차 극적: Move Can Pot 0.13→0.62, Place Burger Fries 0.15→0.59, Handover Mic(2번째 표) 0.59→0.99, Place Container Plate 0.54→0.86. 반면 Pick Diverse Bottles(0.22)나 Beat Block Hammer(0.16)는 여전히 낮음.
- **Parallel decoding이 AR과 동률(0.56)** — 고정 길이 token화 덕에 1-pass 생성으로 추론 효율을 크게 올리면서 성능 유지. FAST처럼 가변 길이 token을 만드는 방식은 PD가 원천적으로 어렵다는 점에서 구조적 이점.
- A < B인 이유(저자 해석): A는 여전히 codebook 용량에 갇힌 token↔action 1:1 대응이지만, B는 state별로 scale이 달라져 같은 token이 여러 action을 표현 → 연속 action 공간으로 확장.

### 6.2 Zero-shot Sim-to-Real (Table 4.3, 3태스크)

| Tokenizer | Click Bell | Place Container Plate | Pick Diverse Bottles | 평균 |
|---|---:|---:|---:|---:|
| Binning | 6/20 | 0/20 | 0/20 | 0.10 |
| FAST | 4/20 | 1/20 | 0/20 | 0.08 |
| VQ-BET | 7/20 | 2/20 | 0/20 | 0.15 |
| Ours (PD) | 8/20 | 5/20 | 3/20 | 0.27 |
| **Ours (AR)** | **10/20** | **7/20** | **3/20** | **0.33** |

- baseline들은 Pick Diverse Bottles에서 전멸(0/20)인데 SA-VLA만 3/20 성공 — state-aware decoding이 sim-to-real gap에서도 유효함을 시사. 다만 절대 성공률은 낮음.

---

## 7. Ablation 분석

- **State 유무 (Table 4.2)**: w/o state 0.43(PD)/0.51(AR) → Method A 0.52/0.55, Method B 0.56/0.56. state 주입만으로 PD +13%p, AR +5%p. state 없을 때 PD가 AR보다 크게 뒤지는데(0.43 vs 0.51), state 주입 후 그 격차가 사라짐 — state가 병렬 생성의 조건 신호로 특히 유효.
- **일반화 (Table C.1)**: unseen task — Shake Bottle Horizontally 0.95, Place Empty Cup 0.25 / unseen scene(random 환경) — Handover Mic 0.10, Place Container Plate 0.16. 학습 태스크의 변형(수평 shake)엔 강하나 random 환경 일반화는 크게 하락(clean mode 학습의 한계).
- **Tokenization granularity (Table C.2, Fig.5)**: Beat Block Hammer의 유사한 4개 action(cosine similarity 0.9989~0.99999 수준)이 state-agnostic tokenizer에선 동일 index로 붕괴되지만, SA-VLA는 **0.001 수준의 cosine similarity 차이까지 구분**해 서로 다른 token으로 매핑. reconstruction loss 감소와 codebook utilization 향상도 보고(단, 해당 Figure 참조가 본문에서 깨져 있음 — "Figure ??").

---

## 8. 관련 연구 비교

| 방법 | Tokenization | State 활용 | 한계 |
|---|---|---|---|
| OpenVLA (binning) | 차원별 uniform bin | ✗ | granularity 낮음 |
| FAST | DCT + BPE | ✗ | 가변 길이 token → 학습·병렬화 곤란 |
| VQ-VLA (VQ-BET/RVQ) | residual VQ-VAE | ✗ | codebook 유한, token↔action 1:1 |
| **SA-VLA** | VQ-VAE + state adapter | **✓ (decoding 조건화)** | 소규모 데이터 검증뿐 |

- π0 계열(continuous action expert)과 달리 discrete token 인터페이스를 유지하면서 연속성을 회복하려는 접근. X-Tokenizer가 tokenizer를 "semantic interface"로 재정의했다면, SA-VLA는 tokenizer를 "state-conditioned decoder"로 재정의 — 둘 다 tokenizer를 단순 압축기 이상으로 보는 2026년 흐름.
- FAST의 state 256-bin 이산화는 입력 인코딩에 차용하되, tokenizer 자체의 state 조건화는 본 논문의 고유 기여.

---

## 9. 한계 및 미해결 문제

- **(논문 명시) Scalability**: 12태스크 19.2k trajectory의 소규모 검증뿐. 대규모 데이터셋(OXE 등)에서의 scaling law 검증은 future work.
- **(논문 명시) 아키텍처**: VQ-VAE 고정 — diffusion 등 다른 생성 모델로의 대체 가능성 미탐구.
- **(논문 명시) Embodiment**: 로봇 팔 한정, dexterous hand 미검증.
- (리뷰어 관찰) 비교 대상이 tokenizer baseline뿐 — π0, RDT 등 continuous action 계열 SOTA와의 직접 비교 부재. RoboTwin 공식 리더보드 모델들과의 비교도 없음.
- (리뷰어 관찰) 학습 태스크에서 평가(in-distribution)라 12태스크 0.56이 일반화 성능은 아님. random 환경 일반화는 0.10~0.16으로 급락.
- (리뷰어 관찰) VLM 파라미터 수, codebook 크기/코드 수 R, action chunk 길이 k 등 핵심 하이퍼파라미터 다수 미명시. Table 4.1의 "Handover Mic"이 두 sub-table에 다른 값으로 중복 등장하고 "Place Burger Fries ↓ / Shake Bottle ↓" 화살표 표기가 모호하는 등 표 정리가 거침.
- (리뷰어 관찰) `a ÷ w` 변조에서 w→0 근방의 수치 안정성, sigmoid 출력 범위 처리 등 세부가 불명확.

---

## 10. 총평

기여의 본질은 단순하고 명확하다: **"discrete action token의 의미는 로봇 state에 의존한다"**는 관찰을 경량 adapter 하나로 구현해, 유한 codebook을 state-조건부 연속 action family로 확장한 것. controlled comparison(동일 VLM, tokenizer만 교체)으로 tokenizer 효과를 깨끗하게 분리했고, 시뮬레이션 +27%p와 zero-shot sim-to-real +18%p라는 큰 폭의 일관된 개선, 그리고 parallel decoding에서 성능 저하 없음(0.56)이라는 실용적 결과까지 확보했다. 다만 소규모 in-distribution 평가, continuous-action SOTA와의 비교 부재, 세부 하이퍼파라미터 미공개로 주장의 상한선은 아직 검증되지 않았다. VQ 계열 tokenizer를 쓰는 모든 autoregressive VLA에 거의 공짜로 얹을 수 있는 기법이라는 점에서 실용적 가치가 크다.

---

## 11. 🔥 예상 날카로운 질문 모음

1. `a ÷ w`에서 w가 0에 가까우면 a_trans가 발산하는데, sigmoid 출력의 하한 처리나 clipping은 어떻게 했는가?
2. Method B에서 scale w는 state만의 함수인데, 같은 state에서 서로 다른 action class(token)에 동일한 w가 곱해진다. token별로 다른 변조가 필요한 경우는 없는가 (w를 token-조건부로 만들면 더 좋아지는가)?
3. state adapter가 사실상 regression head라면, 그냥 VQ 없이 state-conditioned regression policy를 쓰는 것 대비 discrete token 인터페이스의 이점이 무엇인가 — LLM 호환성 외에 성능상 이점이 있는가?
4. 학습 태스크에서 평가한 0.56이 core 결과인데, RoboTwin 공식 unseen-task 프로토콜이나 randomized mode 50태스크 벤치마크에서는 어느 수준인가? Table C.1의 random 환경 0.10~0.16은 우려스럽다.
5. π0, RDT-1B 같은 continuous action expert 계열과 동일 데이터로 비교하면 어떤가? tokenizer 개선이 continuous 계열과의 격차를 얼마나 좁히는가?
6. codebook 크기, RVQ 여부, token 수 R, chunk 길이 k가 미명시다 — Method B의 이득이 codebook을 단순히 키우는 것으로도 얻어지는 것은 아닌가?
7. Table 4.2에서 state 없는 AR(0.51)이 이미 VQ-BET(0.29)보다 훨씬 높은데, baseline 대비 이득의 상당 부분이 state가 아니라 base tokenizer/입력 state token 등 다른 요인에서 오는 것 아닌가?
8. delta action 전제의 motivation인데, absolute action 표현에서도 state-aware decoding이 같은 폭으로 유효한가?
9. sim-to-real에서 Pick Diverse Bottles 3/20은 baseline 0/20보다는 낫지만 여전히 낮다 — 실패 모드는 perception gap인가 action decoding gap인가?
10. Stage 1의 adversarial coefficient(0.1/1.0)가 training recipe에 있는데 본문 loss에는 adversarial 항이 없다 — 어떤 loss인가?

---

## 12. 레퍼런스 및 리소스

- **논문**: https://arxiv.org/abs/2606.30113 (v1, 2026-06-29, cs.RO)
- **코드**: 미공개
- **저자**: Tengyue Jiang (ECUST), Chunpu Xu (HK PolyU), Jiayue Kang (Xidian), Yao Mu (SJTU)
- **주요 비교 대상**: OpenVLA (arXiv:2406.09246), FAST (arXiv:2501.09747), VQ-VLA (ICCV 2025), π0 (arXiv:2410.24164)
- **벤치마크**: RoboTwin 2.0 (arXiv:2506.18088), 평가 프로토콜 https://robotwin-platform.github.io/doc/index.html
- **실로봇 플랫폼**: AgileX Cobot Magic (Aloha 구성, Piper 6-DoF arms, RealSense D435)

<!-- VERIFIED: pdf -->
