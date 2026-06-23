# ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean Flow Action Generation

> **한 줄 요약**: SigLip2 + SmolVLM 멀티모달 백본 위에 **improved Mean Flow(iMF)** action head와 **Attention Residual(AttnRes)** 깊이별 동적 feature routing을 결합한 0.39B 파라미터 VLA로, **2-step 추론**만으로 LIBERO Avg 88.0%(SmolVLA 0.45B의 87.3%) 달성하면서 inference latency를 18.3ms로 단축한 reactive 로봇 제어 프레임워크(arXiv 2606.14255, 2026).

---

## 1. 배경 및 동기

- Diffusion Policy [1] 및 후속 VLA(π₀, OpenVLA)는 expressive multimodal action 분포를 모델링하지만, **수십 step의 iterative denoising**이 필요해 control cycle당 수백 ms의 지연이 발생.
- Reactive closed-loop manipulation(접촉 동역학, 동적 객체, 실행 불확실성에 적응)에는 high-frequency 제어가 필수.
- Rectified Flow [5] 등 straighter transport path는 step 수 축소를 시도하지만, 여전히 **local instantaneous velocity field**를 학습해 극저 step에서 truncation error 누적.
- 저자들의 핵심 질문(Sec. I): "generative robot policy가 diffusion-style action modeling의 표현력을 유지하면서 one-to-few-step inference로 단축될 수 있는가?"

---

## 2. 방법론

### 2.1 Improved Mean Flow (iMF) Action Generation (Sec. III, Eq. 1–4)

- 표준 Mean Flow [6]는 instantaneous velocity v(z_t, t) 대신 **finite-interval 평균 transport velocity**를 학습:
  $$u(z_t, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau, \tau)\,d\tau$$
- 양변을 (t−r)로 곱하고 Leibniz rule로 t에 대해 미분 후 chain rule을 적용하면 (Eq. 3–4):
  $$u(z_t, r, t) = v(z_t, t) - (t-r)\frac{d}{dt}u(z_t, r, t)$$
- **개선점**: 학습 시 JVP(Jacobian-Vector Product) correction 항을 도입하고 **stop-gradient** constraint로 path inconsistency 안정화 (Fig. 3(b)). 학습 안정성 위해 **Pseudo-Huber loss(δ=1)** 사용 — MSE 대비 large error에서 sub-linear scaling으로 JVP 폭발 방지(부록 B에서 |e_d|≪δ에서 ½e_d², |e_d|≫δ에서 δ|e_d|−δ² 점근식 증명).

### 2.2 Attention Residual (AttnRes) Backbone (Sec. III, Eq. 32–33)

- 표준 PreNorm 잔차 누적은 깊이 증가 시 hidden state가 **uniformly summed**되어 representation dilution 발생(부록 A).
- AttnRes는 각 sublayer가 이전 모든 layer 출력을 indexed cache로 유지하고, 학습된 **structural pseudo-query q_m**을 통해 softmax-normalized routing으로 동적 retrieval:
  $$\bar{h}^{(m)}_\tau = \sum_{j<m} \alpha_{j,\tau} h^{(j)}_\tau,\quad \alpha_{j,\tau} = \mathrm{softmax}_j(q_m^\top \mathrm{RMSNorm}(h^{(j)}_\tau))$$
- 저step 생성에서는 각 evaluation에 representational demand가 큼 → 동적 depth-wise routing이 핵심.

### 2.3 전체 파이프라인 (Fig. 2)

- **Visual encoder**: 동결된 SigLip2(dual-view RGB 256×256, agent-view + wrist).
- **Language encoder**: SmolVLM의 text transformer layers.
- **State token**: 8-D joint state → linear projection.
- **Action head**: 16-block AttnRes Transformer(hidden 768, 8 query/8 KV heads, RoPE base 10k, RMSNorm ε=10⁻⁶, dropout 0.05, SwiGLU FFN dim 2048).
- **Trajectory**: H=16, action chunk K=8, da=7 relative joint control, n_obs=1.
- **Time sampling**: r, t ~ logit-normal(μ=−0.4, σ=1.0), 정렬 후 0.5 확률로 r=t(standard flow matching target 혼합).
- **Inference**: 이산 시간 그리드 1.0 → 0.5 → 0.0의 **2-step Euler** 적분.

> ❓ **예상 질문**: Mean Flow [6]와 무엇이 다른가?
> **답변**: 기존 Mean Flow는 평균 velocity 직접 최적화 시 conditional transport target과 marginal velocity dynamics 사이 path inconsistency가 저step에서 불안정화를 유발. iMF는 (1) Eq. 4의 관계를 JVP correction으로 학습 신호에 명시적으로 주입, (2) stop-gradient로 자기참조 폭주 차단, (3) Pseudo-Huber로 JVP gradient 폭주 억제. 부록 식 (35): v̂ = u_θ + (t−r)D_z f_θ(z_t, r, t, C)·[f_θ(z_t, t, t, C), 0, 1]^⊤.

---

## 3. 실험 결과

### 3.1 LIBERO Multi-task (Table I)

40개 task aggregate 학습, 100 trial 평균:

| Method | Params | Spatial | Object | Goal | Long | Avg | Latency(ms) |
|---|---|---|---|---|---|---|---|
| Diffusion Policy | 0.46B | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | 178.8 |
| OpenVLA | 6.74B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 115.0 |
| π₀ (Paligemma-3B) | 5.97B | 87.0 | 63.0 | 89.0 | 48.0 | 71.8 | 94.3 |
| π₀ | 4.03B | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 | 93.4 |
| SmolVLA | 0.24B | 87.0 | 93.0 | 88.0 | 63.0 | 82.8 | 71.1 |
| SmolVLA | 0.45B | 90.0 | 96.0 | 92.0 | 71.0 | 87.3 | 74.1 |
| **ReactVLA** | **0.39B** | **93.0** | 95.0 | 92.0 | **72.0** | **88.0** | **18.3** |

- **4× 빠른 추론**으로 최대 baseline 능가, π₀ 4B 대비 **5× 빠르면서 +2.0%p**.

### 3.2 RoboIMI Dual-Arm (Table II, 100 rollouts)

| Method | Peg-in-Socket Reward | Object Transfer Reward | Latency(ms) |
|---|---|---|---|
| ACT | 289.60 | 115.64 | 4.5 |
| Diffusion Policy | 1019.39 | 319.20 | 398.0 |
| **ReactVLA** | **1513.56** | **526.22** | **15.1** |

- Peg-in-socket에서 Diffusion Policy 대비 **+48.5%** reward, latency **398→15.1ms**(26.4×). SmolVLA는 bimanual에서 수렴 실패(–).

### 3.3 실세계 Diana 7 Robot (Sec. V-F, Fig. 8)

- 50개 teleop demo/task, 60 Hz 제어, **average latency 38.6 ms**.
- Orange Pick-and-Place: ReactVLA / SmolVLA 모두 95% (19/20).
- Block Stack(정밀 제어 필요): **ReactVLA 90% (18/20)** vs SmolVLA 75% (15/20) — SmolVLA의 82.2 ms latency가 error accumulation 유발.

### 3.4 Ablation

- **AttnRes vs Vanilla(Fig. 4)**: AttnRes 88.0% / 40k step만에 50% 돌파 vs Vanilla 28.7% / 느린 수렴. Feature dilution 방지 효과 입증.
- **Pseudo-Huber vs MSE(Fig. 5)**: MSE는 빈번한 loss spike 및 JVP 폭주, Pseudo-Huber는 부드러운 수렴.
- **추론 throughput**: diffusion 대비 평균 **15.2×** 빠름.

---

## 4. 한계 및 미해결 문제 (Sec. VI 명시)

1. **실세계 평가가 통제된 tabletop으로 한정**: cluttered scene, long-horizon task, 다양한 환경 확장 미검증.
2. **하드웨어 표준화 미흡**: benchmark suite별로 서로 다른 GPU 플랫폼 사용 → 절대 latency 비교는 benchmark 내부로만 유효, relative speedup으로 해석.
3. **시연 수 적음**(task당 50): scaling law 검증 부재.
4. **2-step inference 외 step trade-off curve 부재**: 1-step / 4-step 비교 ablation 없음.
5. **AttnRes 메모리 비용**: 모든 이전 layer hidden state cache → 깊이 확장 시 비용 증가, 본 논문 16 block 한정.
6. **SmolVLM-기반 VLM의 robotic adaptation 부재**: SmolVLA와 동일한 OCR-편향 backbone 사용.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Mean Flow의 robot 도메인 첫 적용 + JVP correction + AttnRes 결합 |
| **Technical depth** | ★★★★☆ — Eq. 1–4 derivation, Pseudo-Huber 점근분석(부록 B), 명료한 알고리즘 기술 |
| **Experimental rigor** | ★★★★☆ — LIBERO(40 task) + RoboIMI(dual-arm) + 실세계 Diana 7, 핵심 ablation 2개 |
| **Practical impact** | ★★★★★ — 0.39B로 4B π₀ 능가하면서 5× 빠름, 38.6ms 실세계 latency |

**강점**:
- **속도-품질 trade-off의 실질적 돌파**: 2-step inference로 SmolVLA 0.45B(74.1ms) 능가하면서 4× 빠름.
- iMF의 수학적 유도(Eq. 1–4)와 JVP/stop-gradient 안정화의 명확한 motivation.
- AttnRes ablation(88.0% vs 28.7%)의 극적 차이로 design choice 정당화.
- Bimanual에서 SmolVLA 수렴 실패를 dual-arm RoboIMI로 명시.

**약점**:
- 실세계 evaluation의 task 다양성 및 demo scale 한정적.
- AttnRes [10]는 Kimi Team(2026)의 LM-domain idea를 차용 — 순수 novelty는 robot adaptation에 국한.
- Inference step 수에 대한 sweep ablation 부재(왜 정확히 2-step인지 정량적 분석 부족).

---

## 6. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | iMF가 Rectified Flow 대비 본질적 우위는? | Rectified Flow는 transport path를 직선화해도 여전히 local v(z_t, t) 학습 → 저step에서 truncation. iMF는 finite-interval mean transport를 직접 모델 → step 수를 명시적 학습 대상에 포함, JVP correction으로 conditional/marginal 일관성 강제. |
| 2 | 왜 2-step인가, 1-step은? | Logit-normal sampling에서 0.5 확률로 r=t(즉 1-step flow matching target)을 혼합 학습 → 1-step도 지원. 그러나 LIBERO 같은 multi-modal 분포에서 2-step이 expressivity 보존(논문은 정량 비교 미제시, future work 영역). |
| 3 | AttnRes는 단순 DenseNet/Hyper-connection [15]과 무엇이 다른가? | DenseNet/Hyper-connection은 dense 또는 multi-stream concatenation으로 고정 routing. AttnRes는 **input-dependent softmax retrieval**(structural pseudo-query) → token마다 다른 layer aggregation. |
| 4 | SmolVLA가 RoboIMI bimanual에서 0인 이유는? | 논문은 "training complexities" 명시. 추정: SmolVLA의 단일 embodiment pretrain(SO100 위주)이 dual-arm RoboIMI joint space에 부적합 + flow matching의 multi-modal action 표현이 dual-arm coordinated chunk에서 발산. |
| 5 | Pseudo-Huber δ=1을 어떻게 선택? | 부록 B 점근분석에 따라 δ는 quadratic↔linear 전환점. δ=1은 정규화된 action space의 typical error 크기를 가정한 default. δ sensitivity 분석은 본 논문에 없음. |
| 6 | 실세계 38.6 ms = 25.9 Hz인데 60 Hz 제어와 어떻게 매칭? | Action chunk K=8을 큐로 60 Hz에 분배 실행. 큐 소진 시점에 새 chunk 생성 → 큐 충전 주기가 1/26 s ≈ 38.6 ms. Async execution(open-loop chunk + reactive replan) 구조. |

<!-- VERIFIED: pdf -->
