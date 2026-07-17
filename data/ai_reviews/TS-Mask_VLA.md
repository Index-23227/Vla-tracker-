# TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging

> **한 줄 요약**: Qwen2.5-0.5B라는 초경량 백본 위에 Bridge Attention으로 다층 vision-language 특징을 주입하는 Discrete Diffusion Action Expert를 얹고, 연속 액션을 256-bin 이산 토큰으로 양자화해 (시간 x 액션차원) 2D 격자에 배치한 뒤 temporal-spatial 2D 마스킹으로 masked-denoising 학습하는 discrete VLA. LIBERO 평균 95.7%, CALVIN ABC->D avg-len 4.19를 단 0.5B로 달성해 7B급 모델들을 능가.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Autoregressive token VLA**(OpenVLA, RT-2 계열)는 vision-language-action을 하나의 토큰 시퀀스로 통합해 next-token 예측하지만, **표현 학습과 제어 정책 사이의 구조적 분리(decoupling)**가 없어 long-horizon에서 취약
- **Continuous diffusion VLA**(Diffusion Policy, pi0 계열)는 궤적을 통짜 실수 신호로 취급해 반복 denoising하지만, **장기 horizon에서 불안정**하고 액션의 시간-공간(inter-dimensional) 상관을 명시적으로 모델링하기 어려움
- Discrete diffusion은 텍스트/이미지에서 성공했지만 **로봇 액션 모델링에는 거의 적용되지 않음**

### 핵심 질문
- **Vision-language 표현을 전용 action expert로 "브리지"하여 조건부 분리를 명확히 하면 더 안정적인가?**
- **액션을 이산화하고 2D 구조(시간 x 차원)로 모델링하면 복잡 환경에서 일반화가 좋아지는가?**

📌 [Figure 1 삽입] — TS-Mask VLA 아키텍처: DINOv2+SigLIP 시각 인코딩 -> Qwen2.5-0.5B VLM (Action Query 토큰 포함) -> N-layer Discrete Diffusion Action Expert (Bridge Attention + FFN) -> 2D 마스킹된 액션 토큰의 반복 denoising -> detokenizer -> 최종 액션

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

세 가지 핵심 구성:
- **VLM 백본**: DINOv2 + SigLIP로 third-view/wrist 이미지 특징 추출, 언어 임베딩 공간으로 fused MLP 투영. Qwen2.5-0.5B가 멀티모달 표현 생성. 학습 가능한 **Action Query(AQ) 토큰**을 입력에 추가해 action-centric guidance 제공
- **Discrete Diffusion Action Expert**: 이산 액션 토큰을 생성하고 반복 denoising으로 정제
- **Temporal-Spatial 2D 마스킹 모듈**: 이산화된 액션 토큰에 2D 마스킹 적용

### 2.2 Bridge Attention Block

VLA-Adapter에서 영감을 받아, 한 번의 attention 계산 안에서 세 정보원을 융합:

| 스트림 | 출처 | 역할 |
|--------|------|------|
| Self tokens | 현재 액션 표현 | intra-action 시간 의존성 모델링 |
| AQ tokens | VLM hidden state의 Action Query 특징 | action-oriented 조건화 |
| Task tokens | VLM hidden state의 vision-language 특징 | 외부 task 제약 제공 |

공유 query Q에서 세 스트림별 독립 MLP로 key/value를 유도하고, attention logit을 key 차원으로 concat 후 공동 normalize. **task 브랜치에는 학습 가능한 gating scalar g에 tanh를 적용**해 초기 학습에서 과도한 task 조건화를 완화. 각 key 브랜치에 RoPE 독립 적용.

> ❓ **예상 질문**: 최종 layer만이 아니라 모든 L개 layer의 hidden state를 쓰는 이유는?
> **답변**: 저층은 fine-grained 공간/경계 단서, 고층은 semantic/instruction grounding을 제공. Expert를 백본과 layer-aligned로 정렬해 각 layer 특징을 대응 layer에 주입 -> 저수준 지각과 고수준 지시를 동시에 활용.

### 2.3 Action Head: Discrete Diffusion + 2D 토큰화

- **Uniform Quantized Tokenization**: 정규화된 스칼라 â ∈ [-1,1]을 V=256 균등 bin으로 이산화(257 bin edge, left-closed right-open). VQ-VAE는 과도한 압축으로 fine-grained 정보 손실 우려가 있어 **의도적으로 배제**
- **2D 구조**: 토큰 시퀀스 길이 M = T x D (T=시간 프레임, D=액션 차원)를 T x D 격자로 reshape
- **디코딩**: 토큰 -> bin center로 매핑해 연속 액션 복원

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO**: 4 suite (Spatial, Object, Goal, Long) — object-centric, goal-conditioned, spatial reasoning, long-horizon 커버
- **CALVIN ABC->D**: 표준 long-horizon 다단계 language-conditioned 프로토콜
- **Real-world**: UR5e 6-DoF + ROBOTIQ-85 gripper, D435i 카메라 2대로 자체 수집 (apple 놓기, tissue 뽑기, pan 뒤집기)

### 데이터 사용 패턴
- 백본은 **LoRA**로 parameter-efficient fine-tuning
- action chunk length = 8 (기본)

> ❓ **예상 질문**: 0.5B로 7B 모델을 이긴다는 게 데이터 규모 차이 때문 아닌가?
> **답변**: 오히려 반대다. pi0 등은 대규모 multi-task 데이터로 학습하는데, TS-Mask는 동일 벤치마크 데이터로 fine-tuning만 하고도 pi0를 +1.5%p(LIBERO) 앞선다. 핵심은 데이터량이 아니라 **이산 2D 구조 모델링이라는 inductive bias**라는 게 저자 주장. 다만 대규모 자원 하 성능은 미검증(Limitations 명시).

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM 백본 | Qwen2.5-0.5B |
| 시각 인코더 | DINOv2 + SigLIP (third-view + wrist) |
| Action Expert | N-layer (백본 layer와 정렬) Bridge Attention + FFN 스택 |
| 토큰화 | 256-bin uniform quantization, 2D (T x D) 격자 |
| 마스킹 스케줄 | cosine, r = cos(pi/2 * t), t ~ Uniform(0,1) |
| 학습 손실 | masked CE (L_mask) + step-unroll CE (L_unroll), L = (L_mask + lambda*L_unroll)/(1+lambda) |
| lambda | 0.5 (최적) |
| action chunk | 8-step |
| Fine-tuning | LoRA |
| Hardware | 단일 NVIDIA RTX 4090 |

---

## 5. 실험 설계 및 평가 프로토콜

평가는 **시뮬레이션 2종 + 실물 로봇**:
1. **LIBERO** — 4 suite success rate (%)
2. **CALVIN ABC->D** — 1~5 연속 task 성공률 및 avg-len (out of 5)
3. **Real-world** — UR5e에서 3개 task, 각 20 trial, OpenVLA-OFT / pi0와 비교

📌 시뮬레이션 baseline을 Large(7B급) / Small / Tiny 세 스케일로 분류해 파라미터 효율을 강조하는 설계.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table I) — success rate %, 0.5B

| Suite | VLA-Adapter-Pro* (0.5B) | pi0 (3B) | GR00T N1 (2B) | **TS-Mask VLA (0.5B)** |
|-------|------|------|------|------|
| Spatial | 95.0 | 96.8 | 94.4 | **95.4** |
| Object | 99.0 | 98.8 | 97.6 | **99.4** |
| Goal | 94.0 | 95.8 | 93.0 | **96.2** |
| Long | 80.8 | 85.2 | 90.6 | **91.6** |
| **Avg** | 92.2 | 94.2 | 93.9 | **95.7** |

- **19배 큰 FlowVLA(8.5B, 88.1) 대비 +7.6%p**, pi0(94.2) 대비 +1.5%p
- **Long suite에서 91.6%로 최고** — long-horizon에서 시공간 모델링 효과 두드러짐 (GR00T N1 대비 +1%)

### CALVIN ABC->D (Table II) — success rate %, avg-len

| Completed | OpenVLA-OFT (7B) | OpenHelix (7B) | **TS-Mask VLA (0.5B)** |
|-----------|------|------|------|
| 1 task | 96.3 | 97.1 | **97.4** |
| 2 tasks | 89.1 | 91.4 | **92.5** |
| 3 tasks | 82.4 | 82.8 | **85.2** |
| 4 tasks | 75.8 | 72.6 | **77.0** |
| 5 tasks | 66.5 | 64.1 | **66.9** |
| **Avg len** | 4.10 | 4.08 | **4.19** |

- **모든 7B급을 avg-len에서 능가** (4.19), 14배 적은 파라미터로 OpenVLA-OFT(4.10) 초과
- OpenVLA(3.27) 대비 4-task 52.1->77.0, 5-task 43.5->66.9로 long-horizon에서 극적 향상

> ❓ **예상 질문**: LIBERO는 압도적인데 CALVIN avg-len 4.19는 상대적으로 소폭 우위다. 왜?
> **답변**: CALVIN ABC->D는 5-task chaining이라 절대 난이도가 높아 최상위 모델들도 4.0~4.1대에 몰려 있다(포화 아님, head-room 압축). TS-Mask의 4.19는 그 밀집대에서 **최고**이며, 특히 0.5B라는 점에서 파라미터 효율 우위가 본질. 절대 격차보다 "동급 스케일 대비"가 핵심 메시지.

---

## 7. Ablation 분석

### Masking Strategy (Table III) — 1D vs 2D

| Mask | Spatial | Object | Goal | Long |
|------|---------|--------|------|------|
| 1D | 94.6 | 98.4 | 95.2 | 85.0 |
| **2D** | **95.4** | **99.4** | **96.2** | **91.6** |

- Spatial/Object/Goal에서 약 +1%p, **Long에서 +6.6%p** — temporal 추론 의존 task에서 2D 구조의 효과가 압도적
- 결론: 시간-공간 구조를 명시적으로 모델링하는 것이 long-sequence에서 결정적

### Step Unroll Strength (Table IV) — LIBERO-Spatial %

| 설정 | Success |
|------|---------|
| No unroll | 90.8 |
| **lambda = 0.5** | **95.4** |
| lambda = 1.0 | 91.5 |

- unroll 없으면 train-test 불일치로 90.8, **lambda=0.5에서 +4.6%p** 향상
- 단 lambda=1.0은 과도한 정규화로 91.5로 하락 -> 주 masking loss와의 **균형이 필수**

### Step Unroll 메커니즘
- 1차 forward 후 high-confidence 예측을 채워 넣고, 남은 mask 위치에 2차 예측 -> 그 CE가 L_unroll. inference의 반복 ReMask 절차를 학습에 근사해 discrepancy 완화.

---

## 8. 관련 연구 비교

| 모델 | 액션 생성 | Params | LIBERO Avg | CALVIN avg-len |
|------|----------|--------|-----------|---------------|
| OpenVLA | AR token | 7B | 76.5 | 3.27 |
| pi0 | flow-matching | 3B | 94.2 | — |
| GR00T N1 | — | 2B | 93.9 | — |
| OpenVLA-OFT | parallel decode | 7B | — | 4.10 |
| VLA-Adapter-Pro* | — | 0.5B | 92.2 | — |
| **TS-Mask VLA** | **discrete diffusion + 2D mask** | **0.5B** | **95.7** | **4.19** |

### 핵심 차이
- **이산 diffusion을 액션 청크에 확장**한 드문 시도 (텍스트/이미지 위주였던 masked discrete diffusion을 로봇 액션으로)
- **2D temporal-spatial 마스킹**: 1D random masking(InterMask 등)과 달리 시간축/차원축 양쪽에 구조적 perturbation
- **Bridge Attention**: VLA-Adapter의 layer-aligned 조건화를 self/AQ/task 3-스트림 융합으로 확장

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **자원 제약 시나리오만 검증**: 저자 스스로 Limitations에서 "abundant training resources 하의 성능은 미검증"이라 명시. 단일 RTX 4090, 벤치마크 데이터 fine-tuning만 다룸
2. **오픈소스 미공개**: 코드/가중치 링크가 논문에 없음 -> 재현성 제약
3. **Real-world 정량표 부재**: Fig 5의 bar chart로만 제시(20 trial/task)되어 정확한 수치 테이블이 없음. 3개 task로 범위 제한적
4. **Hyperparameter 민감도**: lambda 외에 마스킹 스케줄, 반복 횟수 I, ReMask scheduler gamma 등에 대한 sweep이 얕음
5. **백본 의존성**: Qwen2.5-0.5B에서만 검증. 다른 백본/스케일로의 확장성 미검증

### Attribution 문제
- LIBERO/CALVIN 우위가 **discrete 2D 모델링** 덕인지 **Bridge Attention의 다층 조건화** 덕인지 완전 분리 안 됨 (Table III는 masking만, Bridge Attention ablation은 부재)

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — masked discrete diffusion을 로봇 액션에 2D 구조로 이식한 점이 참신 |
| **Technical depth** | ★★★★☆ — Bridge Attention(3-스트림+gating), 2D 마스킹, step-unroll 등 설계가 촘촘 |
| **Experimental rigor** | ★★★☆☆ — 시뮬 결과는 강력하나 real-world 정량표/백본 다양성/Bridge ablation 부재 |
| **Practical impact** | ★★★★★ — 0.5B, 단일 4090으로 7B급 초과 -> 배포 효율 측면 매우 실용적 |
| **Writing quality** | ★★★☆☆ — 명확하나 일부 표기/오타(â 양자화 서술 등) 존재 |

**강점**: 0.5B라는 tiny 스케일로 LIBERO 95.7 / CALVIN 4.19를 달성하는 파라미터 효율. 2D temporal-spatial 마스킹의 Long-suite +6.6%p 효과가 설계 가설을 강하게 지지.
**약점**: 자원 제약 세팅에 국한, 코드 미공개, real-world 정량화 부족, Bridge Attention 단독 기여 미분리.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 0.5B로 pi0(3B)를 이기는 게 정말 구조 덕인가, 데이터/튜닝 차이 아닌가? | 동일 벤치 데이터 fine-tuning으로 +1.5%p. 저자는 2D 이산 inductive bias를 원인으로 지목하나 대규모 자원 하 검증은 미완 |
| 2 | 왜 VQ-VAE 대신 uniform 256-bin 양자화인가? | VQ-VAE는 과도 압축으로 시간 의존성/차원 결합 등 fine-grained 정보를 잃음. uniform bin이 구조 보존에 유리 |
| 3 | 2D 마스킹이 1D 대비 이득이 Long에 몰리는 이유는? | Long-horizon은 cross-time 의존성이 지배적. temporal 프레임 통째 마스킹이 이 의존성 학습을 강제 -> +6.6%p |
| 4 | Step-unroll을 왜 lambda=0.5로 제한하나? | inference의 2단계 ReMask를 학습에 근사하나, lambda=1.0이면 주 masking objective를 침해(91.5로 하락). 균형점이 0.5 |
| 5 | Bridge Attention에서 task 브랜치에 tanh gating을 왜 두나? | 초기 학습 시 task 조건이 과도하게 지배하는 것을 완화. 학습 가능한 g로 점진적 task 주입 |
| 6 | 반복 refinement I 회는 몇 번이고 latency는? | 논문은 I-step coarse-to-fine ReMask를 기술하나 구체적 I 값/실측 latency 테이블은 제시 안 됨(한계) |
| 7 | CALVIN 4.19가 4.10 대비 유의미한가? | 절대 격차는 작지만 0.5B로 7B급을 넘긴다는 파라미터 효율이 본질. avg-len 밀집대(4.0~4.1)에서 최고 |
| 8 | Real-world 일반화 근거는? | UR5e 3-task(각 20 trial)에서 OpenVLA-OFT/pi0 초과(Fig 5). 다만 정량표 부재로 강도 판단엔 데이터 부족 |

<!-- VERIFIED: pdf -->
