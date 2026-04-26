# GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware VLA Models

> **한 줄 요약**: 2D patch token 대신 3D anisotropic Gaussian primitive($\mu, \sigma, \alpha$)로 장면을 토큰화하고 Depth-Aware Chain-of-Thought($c_1$–$c_4$)를 결합하여 LIBERO 96.4%, SimplerEnv 80.2%, Data-Efficient 83.1% 달성.

---

## 1. 배경 및 동기

VLA 모델은 시각 입력을 2D patch token($N_p$=256)으로 인코딩하는데, 이 토큰들은 (i) pixel-uniform 하게 token budget을 분배하고, (ii) surface normal/곡률 정보를 담지 못하며, (iii) depth 신뢰도(geometric confidence)를 표현할 방식이 없다. DepthVLA는 별도 depth expert stream을 추가했지만 여전히 픽셀당 scalar depth로, edge와 평면이 동일 깊이일 때 동일한 표현을 만든다 (논문 Sec. I). GST-VLA는 (a) 3D anisotropic Gaussian primitive로 surface orientation·confidence를 명시적으로 인코딩하고, (b) DA-CoT으로 3D 추론을 supervised intermediate generation으로 강제한다.

## 2. 방법론

### 2.1 Gaussian Spatial Tokenizer (GST)
- Frozen depth estimator + frozen SigLIP semantic encoder → patch별 metric anchor $p_k$를 back-projection (Eq. 1)
- 4-layer MLP($f_\theta$)이 $[\mu_k \in \mathbb{R}^3, \sigma_k \in \mathbb{R}^3, \hat\alpha_k]$를 예측 (Eq. 2). 공분산은 axis-aligned $\Sigma_k = \mathrm{diag}(\exp(2\sigma_k))$
- Multi-scale image pyramid 기반 opacity MLP (Eq. 3): 3×1152 → 1차원, 표면 confidence를 텍스처 gradient에 연동
- 3D Fourier PE $L=6$ octave (Eq. 4) — ablation에서 2D learned PE 대체 시 −2.8 pp로 GST 내 가장 큰 단일 효과 (Table IV)
- Spatial attention pooling으로 $N_p$=256 raw token → $N_g$=128 pooled token (Eq. 5). Average pooling 대체 시 −2.1 pp
- Differentiable depth rendering (Eq. 6) + scale-invariant log loss(Eq. 13)로 geometric regularization

### 2.2 Depth-Aware Chain-of-Thought
- VLM 모든 transformer block에 cross-attention sublayer 추가 (Eq. 8) — pooled가 아닌 raw 256-primitive에 직접 access
- 4단계 thought를 autoregressive 생성: $c_1$ 3D centroid → $c_2$ grasp contact + 표면 normal → $c_3$ metric pairwise distance → $c_4$ SE(3) waypoint
- Annotation 자동화: 3D centroid는 open-vocab detection + depth point cloud, grasp point는 pretrained planner, waypoint는 demonstration end-effector 속도 zero-crossing

### 2.3 Flow-Matching Action Expert
300M 파라미터, 6 layers × $d_e$=512, dual cross-attention(VLM hidden / DA-CoT action token), MoE FFN(8 expert/layer, top-2 routing, expert hidden 2048). $L_{\text{act}}$=10 chunks, ODE 10 Euler steps (Eq. 10).

### 2.4 3-Stage Training
- **S1** (80K steps): GST + action expert만 학습 (encoders/VLM frozen). $L_{\text{depth}}$ on ScanNet/Hypersim/ARKitScenes
- **S2** (40K steps): LoRA(r=16,α=32) on Ψ projector + VLM self-attn, $L_{\text{flow}}+L_{\text{CoT}}+L_{\text{depth}}$
- **S3** (20K steps): full fine-tune at LR 3e-5
- Composite loss(Eq. 11): $\lambda_{\text{CoT}}=0.5, \lambda_{\text{depth}}=0.1$

## 3. 실험 결과

| Benchmark | GST-VLA | Δ vs SOTA |
|---|---|---|
| LIBERO Spatial / Object / Goal / Long (Table II) | 98.2 / 97.4 / 97.1 / 92.6 → 96.4 avg | +2.0 vs SpatialVLA 94.4 |
| SimplerEnv Pick-Can / Move-Near / Drawer-Open / Drawer-Close (Table III) | 83.1 / 77.5 / 75.6 / 84.7 → 80.2 avg | +5.4 vs SpatialVLA 74.8 |
| Data-Efficient (Table I) Avg | 83.1 | +6.3 vs SpatialVLA 76.8 |

LIBERO-Long에서 +3.1 pp의 가장 큰 향상 — $c_4$ SE(3) waypoint supervision 덕분의 trajectory level coherence(Sec. IV.B). SimplerEnv Close-Drawer에서 +5.9 pp는 grasp contact $c_2$ 효과.

## 4. 핵심 Ablation (논문 Tables IV–VII)

- 3D Fourier PE → 2D learned: **−2.8 pp** (GST 내 최대)
- $S_1$ pretraining 제거: **−6.2 pp** (전체 ablation 중 최대; geometric calibration 선결 필요)
- $L_{\text{action}}$ conditioning 제거: −3.1 pp (DA-CoT가 $H_{\text{vlm}}$과 redundant 아님)
- DA-CoT 전체 제거: −3.9 pp; $c_4$만 제거: −2.3 pp; $c_1$만 제거: −1.9 pp
- Dense scalar depth(DepthVLA-style) 대비 full Gaussian: +4.5 pp (Table VII)
- $N_g$=64는 −3.3 pp, $N_g$=256은 +0.4 pp만 — 128이 saturation point

## 5. 분석 및 의의

- DA-CoT 출력 중 $c_1$ centroid 오차 5 cm 초과 시 task success 30% 미만 — 런타임 monitoring 신호로 사용 가능 (Sec. IV.D.1)
- Median 3D centroid 오차 2.3 cm, grasp contact 1.8 cm, SE(3) waypoint 3.1 cm. CoT 정확도 vs success Pearson 0.71
- Inference 6.2 Hz on A100-80GB: encode 18ms + GST 12ms + DA-CoT 38ms + ODE 22ms (DepthVLA 8.1 Hz, OpenVLA 9.4 Hz)
- 60–70% of pooling query를 object surface/edge에 할당; specular/textureless 표면에서 $\alpha<0.05$로 자동 suppression

## 6. 한계 및 발전 방향

- Frozen depth estimator의 정확도가 bottleneck — reflective/투명 물체에서 $c_1$ 오차 누적되어 task success 급락
- 6.2 Hz는 contact-rich high-frequency control에 부족 — DA-CoT 38 ms가 주요 원인
- 논문 부록의 disclaimer: "results presented are preliminary, experiments ongoing" — 최종 수치 변동 가능성 명시
- Future work: $c_2$ grasp contact를 6-DoF orientation까지, opacity confidence를 active perception(다음 viewpoint 선택)에 활용 가능

<!-- VERIFIED: pdf -->
