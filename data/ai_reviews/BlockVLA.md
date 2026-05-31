# BlockVLA: Accelerating Autoregressive VLA via Block Diffusion Finetuning 세미나 리뷰

> **한 줄 요약**: 자기회귀(AR) VLA(Prismatic-7B + SigLIP/DINOv2)를 **block diffusion**으로 fine-tuning — 블록 간에는 AR 의존성, 블록 내에는 병렬 denoising을 유지함으로써 prefix KV-cache 재사용 + NFE 감소를 동시에 달성. 표준 이산 확산 대비 **3.3×**, OpenVLA AR 대비 **8.0×** 추론 가속(186.7 token/s, RTX 4090). LIBERO 4-suite 평균 **91.7%** (DDVLA 83.2% 대비 +8.5pp), SimplerEnv-WidowX 평균 ~35.4%.

---

## 1. 배경 및 동기

VLA에는 두 가지 디코딩 패러다임이 경쟁한다:

- **AR VLA(OpenVLA, Prismatic 등)**: 한 토큰씩 순차 디코딩 — long-horizon에서 latency가 누적되고 오차도 누적.
- **이산 확산 LM(dLLM, DDVLA)**: 양방향 attention으로 토큰을 병렬 정제(parallel refinement) 가능. 그러나 (a) NFE가 반복적으로 필요, (b) 양방향 반복 디코딩에 표준 KV cache를 그대로 못 씀.

BlockVLA는 두 패러다임의 장점을 **블록 단위**로 결합 — "블록은 AR, 토큰은 dLLM"이라는 hybrid를 제안한다. 이로써 (i) 이전 블록들의 KV는 cache로 재사용, (ii) 현재 블록 내의 토큰들은 한 번의 forward로 병렬 denoising.

---

## 2. 방법론 심층 분석

### 2.1 백본 및 토크나이저

- **Vision-language backbone**: **Prismatic-7B** (SigLIP + DINOv2 dual encoder)
- **Action tokenization**: 256 bin × 7 token/timestep — **3 translation + 3 rotation + 1 gripper**
- **Sequence format**: `[BOS, visual, proprioceptive, language, action_sequence, EOS]`

### 2.2 Block Diffusion Mask 설계

토큰 시퀀스를 길이 $B$의 블록으로 분할:

- **블록 간 attention**: causal (이전 블록만 보임)
- **블록 내 attention**: bidirectional (블록 내 모든 토큰이 서로 보임)

이 마스크 디자인이 핵심이다 — 이전 블록까지의 K/V는 변하지 않으므로 prefix cache로 그대로 재사용할 수 있고, 현재 블록 내에서는 양방향 정제가 가능하다. **B=1로 두면 AR, B=∞로 두면 표준 dLLM**이라는 일반 프레임워크 역할.

### 2.3 학습 목적 함수 (Equation 4)

블록별 masked denoising loss를 평균:

$$
\mathcal L = \frac{1}{B} \sum_{b=1}^B \mathbb E_{t, x^b_t} \big[ - \log p_\theta(x^b_0 | x^b_t, x^{<b}_\text{context}) \big]
$$

여기서 $x^{<b}_\text{context}$ 처리에 두 가지 변종:

- **Teacher Forcing**: 이전 블록을 **clean ground-truth**로 condition
- **Diffusion Forcing**: 이전 블록도 **earlier diffusion step의 noisy 상태**로 condition — 추론 분포와의 train-test gap을 줄임

저자는 Figure 7에서 **Diffusion Forcing이 Teacher Forcing보다 LIBERO-Object/Long 모두에서 일관되게 우월**함을 보인다.

### 2.4 Token Shift 제거 (Table 2)

표준 dLLM에서 흔히 쓰이는 token shift trick은 BlockVLA에는 오히려 해롭다:

| 설정 | 5k step 성공률 |
|---|---|
| With token shift | 51.2% |
| **W/o token shift** | **59.6%** |

이는 BlockVLA의 action 토큰이 LLM 텍스트와 다른 분포를 가져, AR-pretraining의 shift convention이 fine-tuning을 오히려 늦추기 때문으로 해석.

---

## 3. 데이터셋 및 평가 프로토콜

- **LIBERO**: 4 suite × 10 task × 50 demo = suite당 500 episode
- **SimplerEnv**: WidowX 4 태스크 — Put Carrot on Plate / Put Spoon on Towel / Stack Green on Yellow / Put Eggplant in Basket. 태스크당 **24 episode**, Grasp Count(G)와 Success Count(S) 별도 리포트
- **학습 예산**: **50k step on only 2 GPUs**

---

## 4. 실험 결과

### 4.1 LIBERO Success Rate vs Step (Table 3)

50k step 시점의 BlockVLA vs DDVLA:

| Suite | OpenVLA (reported) | DDVLA @50k | **BlockVLA @50k** | Δ vs DDVLA |
|---|---|---|---|---|
| Spatial | 84.7 | 89.8 | **90.6** | +0.8 |
| Object | 88.4 | 96.6 | **97.6** | +1.0 |
| Goal | 79.2 | 92.6 | **93.2** | +0.6 |
| **Long** | 53.7 | 53.6 | **85.2** | **+31.6** |
| **Avg** | 76.5 | 83.2 | **91.7** | **+8.5** |

**LIBERO-Long에서의 +31.6pp**가 BlockVLA의 가장 인상적인 결과 — long-horizon에서 양방향 정제(블록 내 parallel denoising)가 AR의 오차 누적을 크게 누른다. 게다가 BlockVLA는 학습 곡선 측면에서도 빠르게 수렴: **5k step에서 이미 평균 54.7%** (DDVLA는 같은 step에서 39.8%).

### 4.2 SimplerEnv WidowX (Table 4, 60k step)

| Task | DDVLA (G/S out of 24) | BlockVLA (G/S out of 24) |
|---|---|---|
| Put Carrot on Plate | 5/5 | **4/4** |
| Put Spoon on Towel | 7/7 | **12/12** |
| Stack Green on Yellow | 3/3 | **3/3** |
| Put Eggplant in Basket | 15/15 | **15/15** |

평균 success rate ≈ **35.4%** (DDVLA 31.3%). 단 SimplerEnv 24-episode 표본은 분산이 큰 평가이므로 절대치보다는 trend로 해석.

### 4.3 추론 가속

| 지표 | 값 |
|---|---|
| BlockVLA throughput | **186.7 tokens/s** |
| vs 표준 discrete diffusion | **3.3×** 빠름 |
| vs Autoregressive OpenVLA | **8.0×** 빠름 |
| 하드웨어 | RTX 4090 단일 GPU |

---

## 5. Ablation 분석

### 5.1 Block Size (Figure 6)

| Block size | LIBERO-Object @50k | LIBERO-Long @50k |
|---|---|---|
| B=7 | 낮음 | 낮음 |
| **B=14** | **일관되게 더 높음** | **일관되게 더 높음** |

블록을 너무 작게 하면 AR에 가까워져 병렬화 이득이 사라지고, 너무 크게 하면 NFE가 늘어남 — 7-DoF × N timestep 시퀀스에서 B=14가 sweet spot.

### 5.2 Denoising Steps per Block (Figure 8, LIBERO-Object @50k)

| Steps | 성능 |
|---|---|
| 1 step | 낮음 |
| 2 steps | 균형 |
| **3 steps** | **peak** |
| 4 steps | diminishing returns |

블록당 **3 step**이 최적 — NFE 4 이상으로는 latency만 늘고 정확도는 정체.

### 5.3 Teacher Forcing vs Diffusion Forcing (Figure 7)

LIBERO-Object/Long 모두에서 **Diffusion Forcing이 일관 우위** — 추론 시 분포(이전 블록도 partial noisy일 가능성)와 align되기 때문.

### 5.4 Token Shift 제거 (Table 2)

- 5k step: w/ shift 51.2% vs w/o shift 59.6% (+8.4pp)
- 결론: AR-pretraining의 token shift convention은 block diffusion fine-tuning을 늦춤

---

## 6. 한계 및 의의

**한계**:
- **모델 파라미터 수, 학습 비용**(GPU-hour) 미보고 — "50k step / 2 GPU"만 명시.
- 코드 공개 여부 미명시 (논문 본문에서 URL 발견 안됨).
- SimplerEnv 24-episode는 표본이 작아 분산이 큼 — 통계적 유의성을 강하게 주장하기 어려움.
- 실로봇 검증 없음. simulation only.
- 블록 size sweep이 7 vs 14만 — 더 넓은 범위(28, 56 등)의 행동 곡선이 없음.

**의의**:
- "AR vs dLLM"이라는 이진 선택을 **블록 size B로 연속 파라미터화**한 첫 작업. B=1 ↔ AR, B=∞ ↔ dLLM 사이의 sweet spot을 명시적으로 탐색 가능.
- LIBERO-Long **+31.6pp**(DDVLA 대비)는 long-horizon에서 양방향 정제의 가치를 보여주는 가장 강한 단일 결과.
- prefix KV-cache 재사용을 dLLM 정책 영역으로 확장 — 사전학습 AR 자산을 그대로 재활용할 수 있는 실용성.
- RTX 4090에서 186.7 token/s — 실배포 가능한 latency 영역.

---

## 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | LIBERO-Long +31.6pp가 너무 큰데, DDVLA가 underfitting된 것 아닌가? | DDVLA도 50k step까지 학습된 동일 setup. AR/dLLM은 50k에서 53.6%로 수렴 정체 — block diffusion이 capacity 측면에서 유리 |
| 2 | B=14가 정말 sweet spot인가? B=28, B=56은? | 본 논문은 B=7 vs B=14만 비교. 더 큰 블록의 trade-off는 future work |
| 3 | 학습 시 noise schedule은? | 명시 미보고. 표준 masked diffusion 사용 추정 |
| 4 | KV-cache hit rate / wall-clock 분해는? | throughput만 보고(186.7 token/s, 3.3× speedup). per-stage 분해 부재 |
| 5 | SimplerEnv 24 episode는 너무 적지 않은가? | 동의. trend로만 해석. LIBERO 4 suite × 500 trial이 주된 평가 |
| 6 | OpenVLA Prismatic-7B 그대로면 trainable param은 7B 전체인가? | 본 논문에서 PEFT/full fine-tune 여부 명시 안됨. "2 GPU 50k step"은 PEFT 시나리오에 가까움 |
| 7 | Diffusion Forcing이 Teacher Forcing보다 항상 우월한가? | LIBERO-Object/Long 둘 다 우월. 다른 suite/태스크에서는 미보고 |
| 8 | 실로봇 검증 없음 — sim2real gap은? | 본 논문 한계. SimplerEnv가 sim2real proxy 역할이지만 실제 robot validation 부재 |

<!-- VERIFIED: pdf -->
