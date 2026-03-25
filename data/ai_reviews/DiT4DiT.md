# DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control

> **한 줄 요약**: Video Diffusion Transformer와 Action Diffusion Transformer를 cascaded 구조로 결합하여 비디오 dynamics와 action을 동시 모델링, LIBERO 98.6% 달성과 함께 10배 이상의 sample efficiency 개선.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **World-Action Model (WAM)**: 비디오 생성과 action 예측을 결합하는 새로운 패러다임이나, joint modeling에서 **비디오와 action의 representation 간 간섭(entanglement)** 문제
- Joint training 시 비디오 예측 품질이 action 예측 품질에 과도하게 의존 → 비디오가 부정확하면 action도 부정확
- 기존 접근의 높은 추론 overhead: 비디오를 먼저 생성하고 action을 추출하는 2-stage pipeline

### 핵심 질문
- **비디오와 action을 분리된 DiT로 모델링하되 cascaded coupling으로 정보를 공유하면, entanglement 문제를 해결할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Cascaded DiT Architecture

```
[Observation + Language]
     ↓
[Video DiT: future frame prediction]
     ↓ (latent features)
[Action DiT: action chunk prediction]
     ↓
[Action output]
```

두 개의 분리된 Diffusion Transformer가 cascaded 연결:
- **Video DiT**: 미래 프레임을 latent space에서 예측
- **Action DiT**: Video DiT의 latent features를 conditioning으로 받아 action 생성

> ❓ **예상 질문**: Cascaded 구조가 joint training보다 왜 나은가?
> **답변**: Joint training에서는 video loss와 action loss가 공유 파라미터를 통해 간섭. Cascaded에서는 각 DiT가 자신의 목적에 집중하되, information flow는 latent conditioning으로 유지. 이는 modular 설계의 이점.

### 2.2 Video-Conditioned Action Diffusion

Action DiT의 denoising:

$$\hat{\mathbf{a}}_0 = D_\phi^{\text{act}}(\mathbf{a}_t, t, \mathbf{z}_{\text{video}}, \mathbf{c}_{\text{obs}})$$

여기서 $\mathbf{z}_{\text{video}}$는 Video DiT의 중간 latent features (생성된 비디오 자체가 아님).

> ❓ **예상 질문**: Video DiT의 latent features와 최종 생성된 비디오 중 어느 것이 action에 더 유용한가?
> **답변**: Latent features가 더 풍부한 정보를 담고 있음 (decoded image는 information bottleneck). 이 선택이 성능에 기여함을 ablation에서 보임.

### 2.3 Training Strategy

- End-to-end joint training: 두 DiT가 동시에 학습
- Video DiT loss: reconstruction loss on future frames
- Action DiT loss: diffusion denoising loss on actions
- 두 loss의 weighted sum

$$\mathcal{L} = \lambda_v \mathcal{L}_{\text{video}} + \lambda_a \mathcal{L}_{\text{action}}$$

---

## 3. 실험 결과 심층 분석

### LIBERO

| 모델 | LIBERO Avg (%) | Sample Efficiency |
|------|---------------|-------------------|
| Diffusion Policy | 85.2 | 1x |
| OpenVLA | 76.5 | 1x |
| GR-1 (video WAM) | 91.3 | 3x |
| **DiT4DiT** | **98.6** | **10x+** |

- **98.6%는 LIBERO near-saturation**
- Sample efficiency 10x+: 동일 성능 달성에 필요한 데이터가 1/10

### Convergence Speed

| 모델 | 90% SR 도달 epoch |
|------|-----------------|
| Diffusion Policy | 500 |
| GR-1 | 200 |
| **DiT4DiT** | **~70 (7x faster)** |

---

## 4. Ablation 분석

| 구성요소 | LIBERO (%) |
|---------|-----------|
| Full DiT4DiT | 98.6 |
| Joint single DiT (no cascade) | 93.2 |
| Action DiT only (no video) | 89.5 |
| Video DiT → decoded image conditioning | 95.1 |
| Video DiT → latent conditioning | 98.6 |

- **Cascaded > joint**: 5.4%p 차이로 cascaded 구조의 우수성 입증
- **Latent conditioning > decoded image**: 3.5%p 차이

---

## 5. 관련 연구 비교

| 모델 | Video Model | Action Model | Coupling | Inference |
|------|-----------|-------------|---------|----------|
| GR-1 | GPT-style AR | AR tokens | Joint decoder | Sequential |
| UniPi | Diffusion | Inverse dynamics | 2-stage | Slow |
| Motus | Latent world model | Flow matching | Joint | Medium |
| **DiT4DiT** | **Diffusion Transformer** | **Diffusion Transformer** | **Cascaded** | **Medium** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **추론 비용**: 두 개의 DiT를 순차 실행 → Video DiT denoising + Action DiT denoising = 총 latency 높음
2. **Video DiT의 필요성**: 추론 시 실제로 비디오를 "생성"할 필요가 있는지 의문. Latent features만 필요하다면 full video generation이 불필요한 overhead
3. **LIBERO saturation**: 98.6%에서 더 challenging한 벤치마크에서의 차별화가 필요
4. **Real-robot 결과**: 시뮬레이션 위주. 실제 환경에서의 검증 부족

### Attribution 문제
- 98.6%의 기여가 **cascaded architecture** vs **DiT backbone** vs **video conditioning** 중 어디에서 오는지 완전히 분리되지 않음
- Sample efficiency 개선이 video representation에서 오는 data augmentation 효과인지, architecture의 inductive bias인지 불명확

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — DiT의 cascaded coupling이 참신 |
| **Technical depth** | ★★★★☆ — 체계적 ablation |
| **Experimental rigor** | ★★★☆☆ — LIBERO 집중, sim-only |
| **Practical impact** | ★★★★☆ — Sample efficiency가 실용적 가치 |
| **Writing quality** | ★★★★☆ — 명확 |

**강점**: Video와 action을 분리하되 cascaded로 연결하는 설계가 우아하고 효과적. Sample efficiency 개선이 인상적. **약점**: LIBERO saturation, 추론 비용, real-world 검증 부재.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 추론 시 Video DiT를 skip하면? (Action DiT만 사용) | Ablation에서 89.5% (vs 98.6%). Video conditioning이 중요하나, 속도 trade-off |
| 2 | Video DiT를 distill하여 1-step으로 만들면? | 유망한 방향. Consistency distillation 적용 가능하나 미탐구 |
| 3 | LIBERO 말고 CALVIN이나 real-robot에서는? | 핵심 한계. LIBERO에서의 saturation이 다른 환경에서도 유지되는지 불명확 |
| 4 | Sample efficiency 10x가 data size가 작을 때도 유지되는가? | 극소량(10 demo) 환경에서의 실험 부재 |
| 5 | GigaWorld-Policy 같은 action-centered WAM과의 비교는? | 직접 비교 없음. DiT4DiT는 video-centered, GigaWorld는 action-centered WAM |
