# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

> **한 줄 요약**: Wan2.2-5B video DiT + 1B action expert (6B total)의 WAM에서, test-time video generation을 제거해도 성능이 **"highly comparable"** (RoboTwin 91.8%, LIBERO 97.6%)하면서 latency **190ms (vs IDM 810ms, 4.3x 가속)** 달성. 핵심: video co-training 제거 시 **-8%p** 하락.

---

## 1. 배경 및 동기

- WAM(GR-1, UniPi, Motus)은 test-time에 미래 비디오 생성 → latency 높음
- **핵심 질문**: 비디오 생성이 test-time에 정말 필요한가, **학습 시의 auxiliary signal로만 유용한가?**

---

## 2. 방법론 심층 분석

### 2.1 Architecture

- **Video DiT**: Wan2.2-5B (pretrained video generation model)
- **Action Expert**: 1B parameters
- **Total**: **6B parameters**
- **MoT (Mixture-of-Transformer)**: Shared attention between video and action branches
- Action horizon: **32 steps**, 9 video frames/chunk (4x temporal downsampling)

### 2.2 Training

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{vid}}$$

- Flow matching for both video and action
- Logit-normal noise schedule, 10 denoising steps at inference
- CFG scale 1.0
- **Structured attention masks**: 미래 video token → action token 영향 차단

### 2.3 Inference: Video Branch OFF

```
[Training] Obs → [Video + Action branch, shared attention]
[Inference] Obs → [Action branch ONLY] → 190ms
```

### 2.4 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (LR 1e-4, WD 0.01) |
| Schedule | Cosine annealing |
| Precision | Mixed + gradient clip 1.0 |
| GPU | Single **NVIDIA RTX 5090D V2** (32GB) |
| Embodied pretraining | **None** (unlike most baselines) |

---

## 3. 실험 결과 심층 분석

### Controlled Variants

| Variant | 설명 |
|---------|------|
| **Fast-WAM** | Train with video, **inference WITHOUT video** |
| Fast-WAM-Joint | Video + action jointly denoised |
| Fast-WAM-IDM | Video first → action conditioned on video |
| w.o. video co-train | No video objective during training |

### RoboTwin 2.0 (50+ bimanual tasks)

| Method | Clean | Random | **Avg** |
|--------|-------|--------|---------|
| **Fast-WAM** | 91.88 | 91.78 | **91.8** |
| Fast-WAM-Joint | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | 91.16 | 91.34 | 91.3 |
| **w.o. video co-train** | 82.76 | 84.80 | **83.8 (−8.0%p)** |
| LingBot-VA (pretrained) | 92.90 | 91.50 | 92.2 |
| Motus (pretrained) | 88.66 | 87.02 | 87.8 |

### LIBERO

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| **Fast-WAM** | 98.2 | 100.0 | 97.0 | 95.2 | **97.6** |
| Joint | 99.6 | 99.4 | 98.2 | 96.8 | **98.5** |
| IDM | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| **w.o. co-train** | 89.2 | 99.2 | 95.4 | 90.0 | **93.5 (−4.1%p)** |

### 핵심 발견

1. **Fast-WAM ≈ Joint ≈ IDM**: Test-time video가 있든 없든 성능 "highly comparable"
2. **Video co-training 제거 시 큰 하락**: RoboTwin −8.0%p, LIBERO −4.1%p, Real-world severe
3. → **Video prediction의 가치는 training-time representation 개선에 있음**

### Latency

| Method | Latency |
|--------|---------|
| **Fast-WAM** | **190ms** |
| Fast-WAM-IDM | 810ms |
| **Speedup** | **4.3x** |

### Real-World Towel Folding

- **Fast-WAM**: Competitive with pretrained baselines
- **w.o. video co-train**: **10% success** → dramatic failure
- → Real-world에서 video co-training의 중요성이 더욱 극명

---

## 4. 한계 및 미해결 문제

1. **Joint (98.5%) > Fast-WAM (97.6%)**: Test-time video가 +0.9%p 기여. 이 0.9%p가 safety-critical에서 중요할 수 있음
2. **Wan2.2-5B 의존**: 대형 pre-trained video model 필요. From-scratch 학습의 한계 불명확
3. **No embodied pretraining**: 다른 baseline(LingBot, Motus)은 pretrained. 공정 비교를 위해 non-pretrained variant 비교가 핵심
4. **Single GPU (RTX 5090D)**: Consumer-level이지만 최신 GPU 필요

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — "Test-time imagination이 불필요"라는 도발적 질문에 명쾌한 답 |
| **Technical depth** | ★★★★☆ — 4가지 variant의 체계적 비교 |
| **Experimental rigor** | ★★★★★ — RoboTwin + LIBERO + Real-world + ablation |
| **Practical impact** | ★★★★★ — 4.3x 가속의 실용적 가치 |
| **Writing quality** | ★★★★★ |

**강점**: "Video co-training은 필수, test-time generation은 불필요"라는 명확한 결론. 4.3x 가속. **약점**: Joint이 0.9%p 더 높아 완전한 "불필요"는 아님.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Joint이 +0.9%p면 test-time video가 약간은 유용한 거 아닌가? | 맞음. 0.9%p는 marginal하나 zero는 아님. Risk-averse 환경에서는 Joint이 바람직 |
| 2 | Video co-train 없이 real-world 10%면, video가 representation에 핵심인 건가? | 정확히 이것이 핵심 발견. Video prediction이 encoder에 dynamics awareness를 부여 → action에 간접 기여 |
| 3 | DiT4DiT과의 차이는? | DiT4DiT는 intermediate feature extraction, Fast-WAM은 complete branch removal. 둘 다 test-time generation bypass |
| 4 | FLARE (latent prediction)와의 관계는? | FLARE도 학습 시만 world model 사용. Fast-WAM은 pixel-level video, FLARE는 latent-level. 원리는 동일 |

<!-- VERIFIED: pdf -->
