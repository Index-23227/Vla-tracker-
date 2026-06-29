# DriveStack-VLA: Render-Teacher Alignment for BEV-Based DeepStack Vision-Language-Action Model

> **한 줄 요약**: 자율주행 VLA가 perspective image token과 language prior에만 의존해 **driving-oriented spatial intelligence**가 부족하다는 문제를 지적하고, Qwen3-VL-4B 백본 위에 (1) BEVFormer 기반 top-down feature를 **DeepStack 방식으로 LLM decoder에 주입**, (2) rasterized image를 teacher로 삼아 real image의 perceptual focus를 정렬하는 **Render-Teacher Alignment**, (3) best-of-K trajectory를 ranking·refine하는 **self-critic**을 결합한 3단 학습 프레임워크. NAVSIMv1 91.6 PDMS, NAVSIMv2 91.0 EPDMS, Bench2Drive 79.49 driving score를 달성.

---

## 1. 배경 및 동기

### 기존 driving VLA의 한계
- 대부분의 VLA driving 모델은 scene을 **2D perspective image token**으로 표현 → viewpoint-dependent, 카메라/history 간 redundant, planning에 필요한 top-down geometry를 implicit하게만 인코딩.
- 정밀한 motion planning은 **metric geometry, top-down scene structure, safety-critical perceptual cue**에 대한 명시적 attention을 요구.
- Expert demonstration만으로는 rare/safety-critical/recovery 시나리오 coverage가 부족.
- Rasterized/rendered view는 scalable augmentation이지만, real image와 단순 혼합 시 action decoder가 **양 domain에서 동일한 planning-relevant cue에 attend하는지**가 보장되지 않음.

### 핵심 질문
- BEV geometric prior를 LLM decoder에 안정적으로 주입할 수 있는가?
- 합성(rasterized) 이미지를 어떻게 활용해야 실제 배포 시 fragile한 visual evidence 의존을 줄일 수 있는가?
- 느린 generate-critique-rewrite loop 없이 multimodal trajectory selection을 할 수 있는가?

📌 [Figure 1 삽입] — Action codebook / generative planner / DriveStack-VLA 패러다임 비교

---

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Actor-Critic)
- **Actor**: multi-view image I, navigation instruction x, ego state u를 입력받음. perspective vision encoder + BEV encoder를 가지며, BEV branch가 top-down representation을 생성해 DeepStack-style connection으로 decoder에 주입. decoder는 action-token sequence를 생성하고 frozen action codebook이 이를 연속 trajectory로 디코딩.
- **Critic**: LLM decoder의 last-layer hidden state를 재사용. scoring head가 후보 trajectory에 scalar quality score를 부여해 ranking, refinement head가 top score < threshold일 때 bounded residual을 예측.

### 2.2 BEV DeepStack Injection
- BEVFormer 기반 encoder E_bev로 top-down feature map 추출 → Qwen3-VL의 PatchMerger와 동일 구조의 projection P_bev로 BEV token Z_bev ∈ R^{Nb×d} 생성.
- Qwen3-VL의 DeepStack 인터페이스(per-layer visual memory)를 활용. 주입 대상 layer 집합 L_ds의 각 layer ℓ에서 layer-specific BEV token과 camera token을 concat: V_ℓ = [Z_cam_ℓ ; Z_bev_ℓ] → BEV·camera cue를 multi-level로 통합.

### 2.3 Render-Teacher Alignment
RAP 방식으로 rasterized image를 구성해 lane/agent/traffic signal을 controllable하게 강조. real image를 student, rasterized image를 teacher(stop-gradient)로 삼아 두 가지 loss로 정렬.

- **Masked Camera-Token Alignment (Eq.3)**: rasterized image의 큰 검은 배경 영역이 정렬을 지배하지 않도록 render-guided soft mask w_k (Eq.2, foreground threshold τ, sharpness γ) 적용한 mask-weighted MSE. 배경 patch는 작은 weight, foreground patch는 큰 weight.
- **Action-to-Vision Attention Distillation (Eq.4-6)**: action token query → camera key의 attention 분포를 head/query 평균 후 normalize, temperature T_a softmax. teacher(rasterized) 분포를 real 분포로 KL distillation. → action token이 safety-critical visual region에 attend하도록 유도.

### 2.4 SFT Loss (Stage-1, Eq.7)
L_SFT = L_CE^r + λ_meta·L_CE^m + λ_mask·L_mask + λ_attn·L_attn (real/rasterized pass 각각의 autoregressive CE + 두 정렬 loss).

📌 [Figure 2 삽입] — Actor-Critic 아키텍처 (DeepStack 주입 + scoring/refinement head)

---

## 3. Action Tokenization

- 연속 trajectory를 VQ-VAE 스타일 codebook으로 S개 segment로 discretize (S = planning horizon in seconds). segment s마다 scale token ⟨scale_qs⟩ + code token ⟨traj_cs⟩ 1개씩 예측.
- frozen VQ decoder(=action codebook)가 action tail a → 연속 trajectory τ = {(x_t, y_t, ψ_t)}로 매핑. 4초 horizon, 2Hz.
- 이 discrete codebook 구조가 본 모델의 action head 핵심 (discrete latent token → 연속 궤적 디코딩).

---

## 4. Reinforcement Fine-Tuning (Stage-2)

- inference가 best-of-K stochastic sampling에 의존하므로, GRPO objective로 proposal distribution을 정렬.
- joint reward (Eq.8): r_i = r_driving(τ_i) + α_fmt·r_fmt(a_i).
  - **Driving reward (Eq.S2)**: NC/TTC/DAC normalized sub-score 평균 (collision-free, collision buffer, drivable-area compliance).
  - **Format reward (Eq.S3-S6)**: length / index-range / schema 제약을 강제해 strictly decodable action token 보장. λ_len=0.05, λ_range=0.35, λ_schema=0.60, α_fmt=0.20.
- Stage-1 checkpoint를 frozen reference로 두고 KL penalty (Eq.S1)로 mode collapse 방지.

---

## 5. Self-Critic: Scoring & Refinement (Stage-3)

- **Environment token**: last-layer hidden state H에서 BEV+camera token 위치(K_env)를 추출, linear map으로 head dim 투영 후 pooled vector e_h 생성.
- **Scoring head (Eq.9)**: 후보 τ와 e_h로 ŝ∈[0,1] 예측. SmoothL1 regression + gap-weighted ranking loss(best vs worst)로 학습 → scalar 회귀가 아닌 robust ranking 능력 확보.
- **Refinement head (Eq.10)**: τ_ref = τ_0 + Δ_max ⊙ tanh(R(τ_0, E_h)). second-best 후보를 τ_0, best를 supervision target으로 SmoothL1 학습. → 느린 iterative rewrite loop 없이 conditional residual refinement.

---

## 6. 실험 설정

- **데이터**: NAVSIM(102k train, Stage-1/2), RAP rasterized image, Stage-3는 filtered 40k subset (≥3 후보 & best-worst PDMS gap 존재하는 샘플만 retain, 약 60% filter out). Bench2Drive base set 1,000 clips(950 train/50 val).
- **벤치마크**: NAVSIMv1(PDMS), NAVSIMv2(EPDMS, human penalty filter False/True), Navhard(Gaussian splatting counterfactual), Bench2Drive(CARLA 220 routes, closed-loop).
- **구현**: visual resolution 32×32×384, 32× A100 GPU. Stage-1 4 epoch(AdamW, cosine, per-GPU batch 2), Stage-2 1 epoch RFT, Stage-3 scoring/refinement head 각 10 epoch.

---

## 7. 주요 결과

### NAVSIMv1 (Table I)
- SFT 89.8 PDMS, RFT **91.6 PDMS** → 모든 기존 VLM-based 방법 능가 (SGDrive 91.1, WAM-Diff 91.0).
- safety-critical metric NC 99.4, DAC 98.4로 최고.
- Qwen3-VL-4B baseline(84.0) 대비 SFT+Stage3에서 +5.8, RFT 추가로 91.6.

### NAVSIMv2 (Table II)
- EPDMS filter=True **91.0**, filter=False **87.3** → 양 setting 모두 새로운 SOTA.

### Bench2Drive (Table III)
- Driving score **79.49**, success rate **56.36%**, efficiency 164.52. RFT 없이 real-world pretrain → simulation SFT만으로 달성 → 강한 transferability.

### Navhard (Table IV)
- overall EPDMS **34.9** (TransFuser 23.1, DiffusionDrive 24.2 대비 큰 향상). counterfactual S2 평가에서도 robust.

---

## 8. Ablation 분석

### BEV DeepStack & Render-Teacher Alignment (Table V)
- DeepStack BEV 주입(Exp.c)이 baseline(a) 대비 PDMS +2.5, 단순 추가 token stream보다 효과적.
- rasterized image 단순 추가(d)는 무의미, unmasked MSE(e)도 제한적.
- masked MSE + attention distill(g)가 Exp.d 대비 PDMS +2.2, **DAC +2.3** → 정렬이 action token을 safety-critical region으로 유도.

### Self-Critic & RFT Reward (Table VI)
- Self-critic(b)가 baseline(a) 대비 PDMS +1.6 (oracle scorer 없이 ranking·refine).
- format + driving reward(d)로 추가 향상, sampling-based planning 검증.

---

## 9. 정성적 분석

- Render-Teacher Alignment의 attention heatmap(Fig.4): baseline은 foreground pedestrian에만 집중하거나 강한 햇빛 반사 background에 distract됨. 본 모델은 교차로 측면 차량 등 모든 safety-critical agent에 강한 attention, 무관한 background는 suppress.

---

## 10. 강점

- BEV geometric prior를 **DeepStack per-layer 주입**으로 통합 → 기존 단순 BEV token 추가보다 우수.
- rasterized teacher를 adversarial domain head 없이 **attention 정렬**로 활용 → raster augmentation 이점 보존.
- **Discrete action codebook + GRPO RFT**로 strictly decodable proposal 보장.
- self-critic이 generate-critique-rewrite loop 제거 → inference 효율.
- open-loop(NAVSIM, Navhard) + closed-loop(Bench2Drive) 모두에서 SOTA.

---

## 11. 한계 및 논의

- **Open-source 미공개** (code_url null), anonymous project page만 제공.
- Bench2Drive **Comfortness 11.31**로 매우 낮음 (UniAD 43.58, AutoVLA 39.33 대비) → trajectory smoothness 희생.
- 32× A100, 3-stage 학습으로 reproduction cost가 높음.
- BEVFormer encoder/camera parameter 의존 → calibration·multi-camera setup에 sensitive할 수 있음.
- Stage-3가 약 60% 샘플을 filter out → critic 학습 데이터 효율 이슈.

---

## 12. 종합 평가

DriveStack-VLA는 "VLA driving에 부족한 spatial intelligence"라는 명확한 문제의식을 **BEV DeepStack 주입 + Render-Teacher Alignment + self-critic**이라는 세 직교 축으로 동시에 공략한 well-engineered 연구다. data 측(rasterized augmentation + 정렬)과 model 측(geometric grounding + best-of-K ranking)을 모두 개선하며, ablation이 각 component의 기여를 깔끔히 분리해 보여준다. NAVSIMv1/v2/Navhard real-log 및 Bench2Drive closed-loop 전반에서 SOTA를 기록한 점이 설득력 있다. 다만 미공개·고비용·낮은 comfort는 실전 적용 시 고려 필요. action head는 frozen VQ codebook으로 연속 궤적을 디코딩하는 discrete latent token 구조로 분류됨.

**Score: 8.5 / 10**

<!-- VERIFIED: pdf -->
