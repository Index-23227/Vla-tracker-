# TBD-VLA: Temporal Block Diffusion VLA — Block 단위 Discrete Diffusion으로 Temporal AR과 Parallel Decoding 통합

> **한 줄 요약**: Qwen3-VL 2B 백본 위에 **temporal block(m=4) 단위 masked discrete diffusion**을 도입해, block 내부는 parallel, block 사이는 autoregressive로 액션을 생성. LIBERO 97.7%, SimplerEnv-WidowX 66.8%, SimplerEnv-Google 91.0% / 86.3%, 실세계 67.1%를 달성하면서도 0.086s latency로 RTC와 자연스럽게 호환.

---

## 1. 배경 및 동기

### 기존 Discrete VLA의 한계
- **Autoregressive token decoding** (OpenVLA): 액션 chunk를 left-to-right로 풀어내 inference latency가 폐쇄루프 제어에 부적합
- **Parallel decoding** (OpenVLA-OFT, Discrete Diffusion VLA): 빠르지만 **token 간 temporal dependency**를 명시적으로 모델링하지 못함
- **Compact action tokenization** (π0-FAST, VQ-VLA): 토큰 수는 줄이지만 timestep-token 대응이 약화됨

### 핵심 질문
- **Temporal autoregression의 인과 구조와 parallel decoding의 속도를 한 프레임워크에서 동시에 얻을 수 있는가?**
- VLM-only 구조에서 별도 action expert 없이 가능한가?

---

## 2. 방법론 심층 분석

### 2.1 아키텍처

**Base**: Qwen3-VL 2B (VLM-only, action expert 없음)
- VLM tokenizer에 mask, placeholder, action token을 추가
- Proprioception과 action 모두 **Nb bins**로 양자화 후 공유 dictionary 사용
- Prompt 형식: `"State: {state tokens}, Task: {instruction}, Actions: {placeholder tokens}"`

### 2.2 Temporal Block Diffusion 정식화

액션 시퀀스 $a_{1:H_p}$를 $K = H_p/m$개의 temporal block으로 분할:

$$p(a_{1:H_p} \mid o, g) = \prod_{k=0}^{K-1} p_\theta(a_{km+1:(k+1)m} \mid o, g, a_{1:km})$$

- **Forward**: 각 block $k$의 각 token을 $t_{k,i} \sim \mathcal{U}(0,1)$ 확률로 [MASK]로 corruption
- **Reverse**: shifted predictor block $z_k$ (이전 clean block들) 위에서 masked token 예측
- **Loss**: masked positions에 대한 평균 cross-entropy

### 2.3 Temporal-level Token Shift

- VLM 백본의 next-token prediction 목적에 정렬하기 위해, **현재 block의 토큰이 다음 block을 예측**하도록 target을 shift
- Discrete diffusion의 self-reconstruction과 AR의 next-token 사이의 gap을 메움

### 2.4 Doubled-layout 학습 트릭

- Clean sequence $x^0$와 noised sequence $x^t$를 **같은 RoPE 위치를 공유하며** concat 입력
- Custom attention mask로 모든 block을 단일 forward pass에서 병렬 학습
- 학습 효율 대폭 향상

### 2.5 추론 최적화

| 기법 | 효과 |
|------|------|
| **Decoding as Needed** | rollout horizon Ha=8에 필요한 $K_{exec} = \lceil H_a/m \rceil$ block만 디코딩 → 0.185s → 0.125s |
| **Prefix KV Cache** | 시각/프롬프트/이전 block 토큰의 KV 재사용 → 0.125s → 0.113s |
| **VLM Compile** | PyTorch compile → 0.113s → **0.086s** |
| **Expectation Sampling** | $a_{t+h,j} = \sum_x p_\theta(x) c_j(x)$ — 가장 가능성 높은 단일 토큰이 아닌 전체 분포의 기대값 사용 |
| **RTC (in-painting)** | latency window 내의 이전 액션 꼬리를 freeze해 비동기 실행. Masked-diffusion 학습 목적과 자연스럽게 정렬 |

---

## 3. 학습 설정

| 항목 | 값 |
|------|-----|
| VLM | **Qwen3-VL 2B** |
| Temporal block size $m$ | **4** |
| Prediction horizon $H_p$ | **16** (→ K=4 blocks) |
| Diffusion steps per block $n_d$ | **2** |
| Pre-training | DROID, Open-X Embodiment, RoboSet, RoboMIND, RH20T |
| LIBERO fine-tune | 80K steps |
| SimplerEnv-WidowX fine-tune | Bridge-V2 20K steps |
| SimplerEnv-Google fine-tune | Fractal 40K steps |
| 추론 측정 환경 | NVIDIA RTX A40 |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 2)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 | 95.4 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| Disc Diff VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| dVLA | 97.4 | 97.9 | 98.2 | 92.2 | 96.4 |
| **TBD-VLA** | **97.6** | **99.6** | 97.4 | **96.6** | **97.7** |

- **Long-horizon (LIBERO-Long) +4.2%p** 우위가 가장 인상적
- 2B 모델이 3B (π0.5), 7B (Disc Diff VLA)를 능가

### 4.2 LIBERO under Inference Latency (Table 11)

| Latency $L$ | TBD-VLA + RTC | TBD-VLA (no RTC) |
|-------------|---------------|------------------|
| L=0 | 97.7 | 97.7 |
| L=2 | 96.4 | — |
| L=4 | **93.2** | 72.3 |

- RTC 사용 시 L=4에서도 **+20.9%p** (72.3→93.2) — RTC 학습-추론 alignment의 효과 명확
- π0.5 (90.0 at L=4)보다 **+3.2%p** 우위

### 4.3 LIBERO-Plus (Robustness, Table 12)

| Suite | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg** |
|-------|--------|-------|----------|-------|-----------|-------|--------|---------|
| Pretraining 有 | 87.81 | 60.38 | 77.36 | 95.77 | 88.77 | 89.94 | 84.39 | **83.49** |
| Pretraining 無 | 29.43 | 62.86 | 52.12 | 89.35 | 88.84 | 61.65 | 79.02 | 66.18 |

- 2위 대비 **+15.1%p** (paper 본문) — 강건성 격차가 크다
- **Camera perturbation에서 pretraining의 효과(+58.4%p)** 가 가장 결정적

### 4.4 SimplerEnv-WidowX (Table 3)

| 모델 | Spoon | Carrot | Stack | Eggplant | **Avg** |
|------|-------|--------|-------|----------|---------|
| UniVLA | 83.3 | 66.7 | 33.3 | 95.8 | **69.8** |
| **TBD-VLA** | 52.0 | **86.8** | 31.2 | **97.2** | 66.8 |
| Disc Diff VLA | 29.2 | 29.2 | 20.8 | 70.8 | 37.5 |
| π0.5 | 44.4 | 29.2 | 18.1 | 63.9 | 38.9 |

- UniVLA(7B)에 3.0%p 뒤지지만 **모델 크기 2B 대비 효율 우수**
- Carrot/Eggplant에서 최강. Spoon에서는 약점 노출

### 4.5 SimplerEnv-Google Robot (Table 4)

| | Visual Matching | | | Variant Aggregation | | |
|--|---|---|---|---|---|---|
| Model | Pick Can | Move Near | Drawer Avg | Pick Can | Move Near | Drawer Avg |
| InternVLA-M1 | 95.3 / 90.0 / 52.5 / **79.3** | | | 97.1 / 82.0 / 72.0 / **83.7** | | |
| **TBD-VLA** | **99.2 / 85.0 / 88.9 / 91.0** | | | **97.2 / 78.3 / 83.4 / 86.3** | | |

- Drawer task에서 **+36.4%p (88.9 vs 52.5)** — temporal 모델링이 long-action에 유리함을 시사

### 4.6 실세계 (Figure 5)

3개 Franka FR3 태스크 (Bin / Toaster / Liquid), 각 20 rollout × 4 조건 = 240 rollout/method.

| 방법 | Avg SR |
|------|--------|
| π0.5 (DROID checkpoint fine-tune) | 50.0% |
| TBD-VLA (no RTC) | 60.0% |
| **TBD-VLA + RTC** | **67.1%** |

- π0.5 대비 **+17.1%p**, RTC가 **+7.1%p** 기여

---

## 5. Ablation 분석 (Table 5, SimplerEnv-Google)

| Config | SR (%) | Inference (s) | VLM forward passes |
|--------|--------|--------------|-------------------|
| $m=1$, $n_d=2$ (full temporal AR) | 84.6 | 0.223 | 16 |
| $m=16$, $n_d=2$ (no temporal AR) | 84.0 | 0.061 | 2 |
| $m=4$, $n_d=1$ (1 step) | 85.7 | 0.060 | 2 |
| $m=4$, $n_d=2$, Argmax | 81.6 | 0.086 | 4 |
| **$m=4$, $n_d=2$, Expectation** | **88.7** | 0.086 | 4 |

- **Expectation vs Argmax: +7.1%p** — 가장 큰 단일 디자인 효과
- $m=4$ optimal: 둘 다 극단 ($m=1$ 또는 $m=16$)은 -4%p 손실

### 추론 속도 분해 (Table 6)

| Component | Latency |
|-----------|---------|
| Baseline | 0.185s |
| + Decode-as-Needed | 0.125s (-0.060) |
| + KV Cache | 0.113s (-0.012) |
| + VLM Compile | **0.086s** (-0.027) |

---

## 6. 추론 효율 vs 경쟁 모델 (Table 1)

| 모델 | Size | Temporal AR | Decoder | Latency |
|------|------|------------|---------|---------|
| OpenVLA | 7B | × | AR | 0.344s |
| OpenVLA-OFT | 7B | × | Parallel | **0.031s** |
| π0.5 | 3B | × | Flow Matching | 0.208s |
| Discrete Diffusion VLA | 7B | × | Disc Diff | 0.069s |
| **TBD-VLA** | **2B** | **✓** | Block Disc Diff | **0.117s** |

- OpenVLA-OFT (parallel)가 더 빠르지만 temporal AR 없음
- TBD-VLA는 **temporal AR이 있는 모델 중 최저 latency**

---

## 7. 관련 연구와의 차별점

| 모델 | Token-level Temporal | Parallel Decoding | VLM-only | Backbone Size |
|------|--------------------|------------------|----------|--------------|
| OpenVLA | ✓ (AR) | ✗ | ✓ | 7B |
| OpenVLA-OFT | ✗ | ✓ | ✓ | 7B |
| Discrete Diffusion VLA | ✗ | ✓ | ✓ | 7B |
| dVLA | ✗ | ✓ | ✓ | 8B |
| **TBD-VLA** | **✓ (block)** | **✓ (intra-block)** | **✓** | **2B** |

- "block-AR + intra-block parallel"이라는 새로운 axis 점유
- 작은 백본으로 SOTA — 효율-성능 frontier 갱신

---

## 8. 한계 및 미해결 문제

1. **Camera viewpoint OOD 취약**: "transfer the liquid" task에서 뷰포인트 변경 시 완전 실패. 시각 충실도 의존 태스크에 약점
2. **Auxiliary VLM 목적 미탐색**: Co-training (VLM cap. 보존) 없이 진행
3. **Bin 수 $N_b$ 미공개 (본문)**: Quantization 정밀도 trade-off 분석 부재
4. **WidowX에서 UniVLA에 -3.0%p**: Bridge-V2 fine-tune 양이 다르거나 token shift 효과가 task별로 다름
5. **2B 백본 한계**: scaling law 측면에서 7B+ 비교 부재

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Block diffusion을 VLA에 처음 적용. Temporal AR×Parallel 통합 |
| **Technical depth** | ★★★★★ — Token shift, doubled-layout, expectation sampling, RTC 정렬 모두 일관됨 |
| **Experimental rigor** | ★★★★★ — LIBERO + LIBERO-Plus + SimplerEnv 2종 + 실세계 + latency ablation |
| **Practical impact** | ★★★★★ — 2B로 SOTA, RTC 자연 정렬, 0.086s latency |
| **Writing quality** | ★★★★☆ — 명료하나 Table 2 Avg(97.7)와 본문(97.6) 약간 불일치 |

**강점**: VLM-only 구조에서 block diffusion으로 temporal/parallel을 통합한 우아한 정식화. **Expectation sampling +7%p, RTC +20%p**라는 큰 단일 효과가 명확. **약점**: viewpoint OOD 약점과 auxiliary 목적 미탐색.

---

## 10. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | OpenVLA-OFT가 0.031s로 더 빠른데 TBD-VLA의 이점은? | Temporal AR을 보존하면서 LIBERO 97.7 vs 95.4(+2.3%p), latency window 하에서 RTC alignment, 모델 크기 1/3.5 |
| 2 | $m$=1과 $m$=16 모두 -4%p 손실인 이유는? | $m$=1은 step별 AR → 학습-추론 misalignment + KV 재사용 불가, $m$=16은 temporal 인과 모델링 상실 |
| 3 | Expectation sampling이 7%p 향상시키는 이유는? | Argmax는 양자화 격자의 가장 가까운 점만 선택. Expectation은 인접 bin 확률을 가중평균해 **연속 동작 정밀도 회복** |
| 4 | RTC inpainting과 학습 목적의 정렬이란? | Masked diffusion은 "이미 채워진 일부 + 마스크된 일부"를 예측하도록 학습되므로, RTC에서 freeze된 액션 꼬리를 그대로 조건으로 사용 가능. AR 모델은 이 정렬이 약함 |
| 5 | Discrete Diffusion VLA (7B, 0.069s)와 비교 시 격차의 원천은? | Temporal block 구조 (intra-AR), Qwen3-VL 백본, expectation sampling. LIBERO +1.4%p, SimplerEnv-Bridge +29%p 격차 |
| 6 | 2B 모델이 7B를 이기는 것이 scaling law에 반하는가? | 백본 (Qwen3-VL)이 더 강력하고, temporal 구조 자체가 효율적 inductive bias 제공 |

---

## 11. 핵심 기여 정리

1. **Block-wise masked discrete diffusion for VLA**: $m=4$ 블록 단위로 intra-block parallel, inter-block AR
2. **Token-shift + doubled-layout**: VLM pretraining (next-token)과 정렬되는 학습 파이프라인
3. **Expectation sampling**: 양자화 격자를 넘는 연속 동작 복원 → +7%p
4. **RTC와의 native 정렬**: Masked diffusion 학습으로 RTC inpainting이 학습-추론 alignment 보장 → L=4에서 +20.9%p

---

## 12. 결론 및 시사점

TBD-VLA는 **"temporal AR이냐 parallel decoding이냐"라는 기존의 이분법을 block diffusion으로 해소**한 작품이다. 핵심은 (1) VLM-only 구조 유지, (2) 학습 목적 (masked diffusion) 과 추론 모드 (RTC inpainting) 의 정렬, (3) expectation sampling이라는 디코딩 트릭이다. LIBERO 97.7%, 실세계 +17%p, latency 0.086s라는 결과는 discrete VLA의 새로운 효율-정확도 frontier를 정의한다. 후속 연구는 (1) viewpoint robustness, (2) auxiliary VLM 목적 co-training, (3) 더 큰 백본으로의 scaling을 다룰 것으로 보인다.

<!-- VERIFIED: pdf -->
