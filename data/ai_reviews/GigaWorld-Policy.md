# GigaWorld-Policy: An Efficient Action-Centered World–Action Model

> **한 줄 요약**: Wan 2.2 5B DiT를 backbone으로 action token과 future-video token을 causal mask로 결합한 action-centered World–Action Model. Inference 시 video branch는 optional이라 action-only 360 ms (Motus 대비 9× 가속), RoboTwin 2.0 86%·실세계 0.83 SR 달성.

---

## 1. 배경 및 문제의식

VLA 학습은 **supervision sparsity** 문제를 가진다 — 관찰·언어는 고차원/풍부하지만 action 라벨은 희박하다. 기존 World-Action Model(WAM) 계열(Motus, Cosmos-Policy, VideoVLA, Mimic-Video, LingBot-VA)은 미래 비디오 생성을 강제하여 supervision density를 높이지만, (i) inference 시 매 control step마다 비디오 토큰을 sampling해야 해서 latency가 크고, (ii) 픽셀 예측 오차가 action으로 전파된다 (Fig. 2 b–c, Sec. 1). GigaWorld-Policy는 학습 시에만 비디오를 dense supervision으로 쓰고 추론 시에는 video branch를 끌 수 있는 action-centered 설계를 제안 (Fig. 2 d).

## 2. 아키텍처

### 2.1 Backbone & Tokenization (Sec. 3.2)
- **Wan 2.2 5B Diffusion Transformer**(Wan et al., 2025)를 정책 초기화로 사용
- 3개 view(left/front/right)를 한 장의 composite image로 결합(Eq. 4) → 단일 backbone으로 cross-view consistency 확보
- VAE로 현재/미래 관찰을 spatiotemporal latent $T_o, T_f$로 인코딩
- 모든 token을 단일 transformer stack이 처리(MoE 아님). 단 positional encoding은 type별 분리 — 시각 토큰은 2D, state/action은 1D temporal
- 언어 instruction은 self-attention 시퀀스에 포함되지 않고 cross-attention 외부 conditioning으로 들어감

### 2.2 Causal Block-wise Attention Mask (Fig. 4, Eq. 5)
시퀀스 $T_t = [T_o; T_s; T_a; T_f]$에 대해:
1. $T_s, T_o$ ↔ 상호 attend, $T_a/T_f$는 못 봄
2. $T_a$ → $\{T_s, T_o\}$만 attend, $T_f$는 못 봄
3. $T_f$ → $\{T_s, T_o, T_a\}$ attend (action-conditioned dynamics)

이 비대칭 마스크가 핵심 — future-video 정보가 action 예측에 leak되는 것을 막아서 추론 시 video branch 제거가 가능해진다.

### 2.3 Flow Matching Objective (Eqs. 6–10)
$x(s) = (1-s)\epsilon + sx$, target velocity $\dot x(s) = x - \epsilon$. Action·video 두 modality에 모두 적용:
$$\mathcal{L}_{all} = \lambda_{video}\mathcal{L}_{video} + \lambda_{action}\mathcal{L}_{action}$$
실험 weight: $\lambda_{action}=5, \lambda_{video}=1$. Action chunk $p=48$, future stride $\Delta=12$ (즉 $K=4$ frame 예측).

### 2.4 Curriculum Pre-training (Tab. 1, Sec. 3.3)
1. Web-video 초기화 (Wan)
2. **Embodied data pre-training** ~10,000h: Agibot 2,500h + EGO4D 3,500h + Open X-Embodiment 3,500h + EgoDex 800h + DROID 350h + RoboMind 300h + Something-V2 200h + RDT 25h + ATARA 10h
3. Target-robot post-training (image+language+action triplet)

## 3. 핵심 결과

### 3.1 RoboTwin 2.0 (Tab. 2, 50+ task 평균)
| Method | Clean | Rand. |
|---|---|---|
| π₀.₅ | 0.43 | 0.44 |
| X-VLA | 0.73 | 0.73 |
| Motus | 0.89 | 0.87 |
| **GigaWorld-Policy** | **0.87** | **0.85** |

VLA-Tracker YAML의 robotwin_v2_clean_avg=87.0, rand_avg=85.0과 정확히 일치(논문 본문은 평균 0.86으로 인용).

### 3.2 Inference Latency on A100 (Tab. 3)
| Method | Time (ms) | SR Sim | SR Real |
|---|---|---|---|
| π₀.₅ | 225 | 0.48 | 0.69 |
| GigaBrain-0 | 452 | – | 0.68 |
| Motus | 3231 | 0.88 | 0.76 |
| Cosmos-Policy | 1413 | – | 0.58 |
| **Ours** | **360** | 0.86 | **0.83** |

Motus 대비 ≈9× 가속(3231→360 ms = 8.97×) 하면서 real-world +7%. 실세계 4 task 평균(Tab. 4): 0.83 vs π₀.₅ 0.69.

### 3.3 Data Efficiency (Fig. 7)
GigaWorld-Policy는 demonstration 10%만으로 π₀.₅의 100% 데이터 성능에 도달.

## 4. Ablation 핵심 (Sec. 4.5)

- **Stride sweep $\Delta$ (Tab. 5, $p=48$ 고정)**: $K=0$ (no future) 0.60 → $K=4$ ($\Delta=12$) **0.83** → $K=12$ ($\Delta=4$) 0.76. 즉 dense future prediction은 오히려 손해.
- **Pre-training 구성 (Tab. 7)**: scratch 0.45 → video init only 0.57 → embodied only 0.73 → **video+embodied 0.83**. 둘 다 필요.
- **Embodied data scale (Fig. 8)**: 0% → 100%로 갈수록 0.57 → 0.83 단조 증가
- **Causal mask vs full self-attn (Tab. 6)**: SR 0.83 vs 0.81, 그리고 video 생성 PSNR 28.41/SSIM 0.901 vs 27.87/0.892 — causal mask가 성능과 deployability 둘 다 우수

## 5. 의의 및 비판적 분석

핵심 통찰은 "video supervision은 학습 신호로만 쓰고 추론에서는 분리 가능하게 둔다"는 것이다. Causal masking이 이 분리를 제도적으로 강제 — full self-attn 변형은 SR이 낮을 뿐 아니라 inference에서 future token을 드롭할 수 없다(action token이 future에 의존). RoboTwin 2.0에서 Motus와 평균 success가 거의 동등(0.86 vs 0.88)함에도 9× 가속한 점, 그리고 real-world에서는 오히려 +7%인 점은 "낮은 latency가 closed-loop error correction을 강화한다"는 저자의 주장(Sec. 4.3)과 일관된다.

## 6. 한계 및 향후 방향

- π₀.₅(225 ms)보다는 여전히 60% 느림 — 5B DiT의 FLOPs 한계
- $K=4$가 최적이라는 결과는 "어느 정도의 future modeling이면 충분하고 더는 오히려 noisy"임을 시사 → 더 긴 horizon 일반화는 미해결
- VLA-Tracker YAML의 inference_hz=2.8는 360 ms latency(≈2.78 Hz)와 일관 ✓
- Wan 2.2의 web video bias가 robot task와 mismatch될 수 있고, embodied pre-training에서 EGO4D 3,500h 의존도가 큼

<!-- VERIFIED: pdf -->
