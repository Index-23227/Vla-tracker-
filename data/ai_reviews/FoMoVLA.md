# FoMoVLA: Bridging Visual Foresight and Motion Guidance for Vision-Language-Action Models

> **한 줄 요약**: StarVLA-GR00T(약 4.6B, flow-matching DiT 액션 헤드) 위에, 미래 feature 예측(K=16 <Foresight> 토큰 + EMA teacher)과 sparse 2D point tracking(frozen CoTracker-v3 teacher)을 **학습 전용** 보조 과제로 얹고, 둘을 zero-init cross-attention 모듈(FCCA)로 결합해 "어디로(goal state)"와 "어떻게(motion path)"를 일관되게 학습시킨 VLA. 추론 시 보조 브랜치를 전부 제거해 오버헤드가 +9.4ms/+0.1GB에 불과하며 LIBERO 평균 98.8%, RoboCasa GR-1 Tabletop 56.9%로 SOTA 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA는 현재 관측·언어를 액션으로 직접 매핑하는 **reactive** 패러다임 → 미래 scene evolution이나 long-horizon object dynamics를 명시적으로 모델링하지 못함
- **Visual foresight 계열**(DreamVLA, UniVLA 등)은 미래 시각 상태를 예측하지만 "어디로 갈지(goal)"만 보여주고 "어떻게 갈지(motion path)"를 담지 못함
- **Dense pixel-level future prediction**은 정보는 풍부하나 제어와 무관한 static content까지 학습해 표현 중복·연산 부담이 큼
- **Point tracking 계열**(FlowVLA, JOPAT)은 motion을 compact하게 담지만 미래 goal state는 예측하지 않음 → 둘을 **분리**해서 학습

### 핵심 질문
- **미래 feature 예측(goal)과 sparse point tracking(motion)을 명시적으로 상호작용시키면 단순 multi-task 합산보다 더 나은 시너지가 나는가?**
- **추론 비용을 늘리지 않으면서 spatio-temporal foresight를 표현에 주입할 수 있는가?**

📌 [Figure 1 삽입] — (a) Vanilla VLA, (b) foresight를 보조 과제로 붙인 VLA, (c) FoMoVLA: foresight와 motion을 함께 예측하고 그 일관성을 강제.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요
FoMoVLA는 VLA 정책에 **학습 시에만 동작하는 3개 브랜치**를 추가:
1. **Point Tracking 브랜치** (§3.2): image token hidden state에서 2D point trajectory 예측 (how to move)
2. **Future Feature Prediction 브랜치** (§3.3): K개 <Foresight> 토큰이 최종 프레임 o_{t+T}의 시각 feature를 인코딩 (where to end up)
3. **FCCA** (§3.4): 예측된 미래 표현으로 motion 예측을 conditioning해 두 목적을 결합

Backbone은 **StarVLA-GR00T**, 입력 224×224, ViT patch 16 → 14×14 map을 spatial merger로 8×8=64 image token(M=N=64)으로 압축.

### 2.2 Point Tracking (Motion Guidance)
- **Goal-aware 전제**: causal VLM에서 image token이 뒤에 오는 text token을 못 보므로, **text를 image 앞에 재배치**해 각 image token의 hidden state가 전체 instruction에 조건화되도록 함
- **Sparse point selection**: 8×8 패치 셀 중심에 대응하는 N=64 query point → image token과 1:1 대응
- **Frozen CoTracker-v3 teacher**가 action chunk 전체 T 프레임의 displacement d* ∈ R^{N×T×2}, visibility v*를 GT로 생성 (dense pixel rendering 회피)
- **Motion loss**: L_track = L_disp + L_vis + λ_smooth·L_smooth (displacement L2, visibility BCE, 2차 시간차분 smoothness). 모든 loss는 GT visibility로 masking해 occluded point의 noisy gradient 차단

### 2.3 Future Feature Prediction (Visual Foresight)
- image·instruction 토큰 뒤에 **K=16 <Foresight> 토큰**을 붙여 o_{t+T}의 시각 feature를 인코딩하도록 학습
- **EMA teacher**(vision encoder의 shadow copy, momentum μ=0.999)가 target frame feature z* 제공
- **MAE decoder**: K 토큰 + (M−K=48) mask 토큰을 2-layer ViT decoder에 통과 → z_hat 복원, **cosine-similarity loss** L_foresight
- K-token bottleneck은 의도적으로 compact → patch-level 암기가 아닌 **global scene-level 변화**를 포착하도록 유도

> ❓ **예상 질문**: 왜 raw pixel이 아니라 feature를 예측하나?
> **답변**: pixel 예측은 static background 등 제어와 무관한 정보까지 재구성해 overhead가 큼. Feature space 예측은 compact하고 action-aligned. 게다가 EMA teacher가 안정적 regression target을 제공해 collapse를 막음.

### 2.4 FCCA (Future-Conditioned Point Tracking)
- 두 목적을 독립 loss로 두면 "무관한 regularizer"에 그침. 가설: **미래 시각 상태를 알면 motion 예측이 더 잘 제약된 문제가 된다**
- **zero-initialized MHA(8-head)** 모듈: H̃_vis = H_vis + MHA(LN(H_vis), LN(H_fut), LN(H_fut))
- 출력 projection을 **zero-init**해 학습 초기엔 identity → 사전학습 VLM feature를 교란하지 않고, 학습이 진행되며 점진적으로 미래 정보를 spatial token에 주입

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO**: 4 suite(Spatial/Object/Goal/Long) × 10 task × 50 demo, T=8
- **RoboCasa GR-1 Tabletop**: 24 tabletop pick-place task, task당 1000 demo(총 24000), GR-1 휴머노이드 egocentric, T=16
- **LIBERO-Plus**: zero-shot OOD 평가 전용(7개 perturbation 차원, 10,030 instance) — 학습 X

### 데이터 사용 패턴
- 모든 보조 supervision은 **학습 전용**, 추론 시 제거
- CoTracker-v3(motion teacher), EMA teacher(foresight target) 모두 gradient 없이 supervisory signal만 생성

> ❓ **예상 질문**: teacher(CoTracker, EMA)에 의존하는데 배포 시 문제 없나?
> **답변**: teacher는 학습 중 GT 생성에만 쓰이고 추론에서 완전히 사라짐. 모든 motion/foresight 지식이 VLM backbone 파라미터에 흡수되므로 배포 footprint는 Vanilla와 동일(4599.3M).

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | StarVLA-GR00T |
| 액션 헤드 | Conditional flow-matching DiT (v_θ 적분 τ:0→1) |
| Image tokens | 8×8 = 64 (M=N=64) |
| Foresight tokens | K=16 |
| Point tracker teacher | CoTracker-v3 (frozen) |
| EMA momentum | μ=0.999 |
| Loss weights | λ1=0.1(foresight), λ2=0.3(track), λ_smooth=0.1 |
| Optimizer | AdamW (β1=0.9, β2=0.95), per-module LR (VLM 1e-5, aux heads 1e-4) |
| Hardware | 8× H20 GPU, DeepSpeed ZeRO-2 |
| Steps | LIBERO 30K (batch 12), RoboCasa 100K (batch 8) |
| 학습 추가 파라미터 | +60.7M (+1.3%): foresight decoder 30.5M, FCCA 26.2M, track projector 2.6M, visibility head 1.3M |
| 추론 오버헤드 | +9.4ms median latency, +0.1GB GPU (Table 8) |

---

## 5. 실험 설계 및 평가 프로토콜

평가는 **세 축**:
1. **LIBERO** — 4 suite, 20 rollout/task, T=8 (Table 1)
2. **RoboCasa GR-1 Tabletop** — 24 task, 50 rollout/task, T=16 (Table 3)
3. **LIBERO-Plus** — zero-shot OOD robustness, 7 perturbation (Table 2)

또한 (a) component ablation, (b) point grid density / foresight decoder / coupling design ablation, (c) 다른 policy head(StarVLA-π, StarVLA-OFT)로의 scalability를 평가.

📛 **Real-world 실험 부재**: 시뮬레이션(LIBERO, RoboCasa) 한정. 3D geometry 미모델링을 limitation으로 명시.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1) — 전체 모델 vs 대표 baseline

| Type | Method | Spatial | Object | Goal | Long | Avg |
|------|--------|---------|--------|------|------|-----|
| VLA/WAM | π0 (RSS'25) | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| VLA/WAM | X-VLA (ICLR'26) | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| VLA/WAM | LingBot-VA (RSS'26) | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Future Pred. | HiF-VLA (CVPR'26) | 98.8 | 99.4 | 97.4 | 96.4 | 98.0 |
| Point Track | JOPAT (arXiv'26) | 97.2 | 98.9 | 98.4 | 96.4 | 97.8 |
| **Ours** | **Full (+FP+Track+FCCA)** | **98.4** | **99.6** | **99.4** | **97.6** | **98.8** |

- **평균 98.8%로 최고**. Goal(99.4)에서 특히 강함
- Component ablation: Base 96.5 → +Future Pred 97.5 → +Tracking 97.8 → +둘 다 98.3 → **+FCCA 98.8**. 단순 합산(98.3) 대비 FCCA가 +0.5, **LIBERO-Long에서 +1.8%p**

### RoboCasa GR-1 Tabletop (Table 3)

| Method | Avg |
|--------|-----|
| π0.5 | 37.0 |
| StarVLA-GR00T (backbone) | 47.8 |
| GR00T-N1.6 | 47.6 |
| StarVLA-OFT | 48.8 |
| **FoMoVLA** | **56.9** |

- backbone(StarVLA-GR00T 47.8) 대비 **+9.1%p** → egocentric 휴머노이드에서도 dual-objective 효과 큼
- Per-task ablation(Table 7): Vanilla 47.8 → +FP 54.4 → +Track 55.6 → +둘 56.6 → +FCCA 56.9

### LIBERO-Plus zero-shot OOD (Table 2)

| Method | Pretrain | Total |
|--------|----------|-------|
| π0 | ✓ | 53.6 |
| Abot-M0 | ✓ | 80.5 |
| StarVLA | ✗ | 74.1 |
| **FoMoVLA** | ✗ | **80.5** |

- **pretrain 없이 80.5%**로 Abot-M0(pretrain 有)와 동률, StarVLA 대비 **+6.4%p**
- language(+5.5%), background에서 특히 강함. camera/robot-state shift는 상대적으로 약함 (고정 viewpoint 학습 + 순수 2D motion cue의 한계)

> ❓ **예상 질문**: LIBERO는 이미 포화(98%+)인데 의미가 있나?
> **답변**: 의미는 **Long-horizon(LIBERO-Long +1.8%p by FCCA)**, **RoboCasa(+9.1%p)**, **LIBERO-Plus OOD(+6.4%p)**에 있다. atomic task가 아닌 long-horizon·egocentric·OOD에서 foresight+motion 결합의 이득이 두드러진다.

---

## 7. Ablation 분석

### FCCA 효과 (Table 1 & Figure 6)
- FCCA 유무로 LIBERO-Long +1.8%p. point tracking 품질(별도 CoTracker GT 대비): ATE-all 1.2→0.9px(−25%), ATE-moving 3.8→2.3px(−39.5%), Median TE 3.7→2.1px(−43.2%), **Survival@10px 78.0→95.3%(+17.3pp)** → 미래 conditioning이 강한 geometric prior 제공

### Point Grid Density (Table 4)
| Grid | Long | Avg |
|------|------|-----|
| 8×8 (64) | **97.6** | **98.8** |
| 16×16 (256) | 94.8 | 98.1 |
- **sparse 8×8이 dense보다 우수**(특히 long-horizon). 4× sparse로도 task-relevant motion 충분히 포착

### Foresight Architecture (Table 4)
| Design | Long | Avg |
|--------|------|-----|
| MAE (16→64) | **97.6** | **98.8** |
| Direct-64 | 95.0 | 97.8 |
- compact K-token bottleneck이 global goal-state encoding 유도

### Foresight–Motion Coupling (Table 4)
| Design | Long | Avg |
|--------|------|-----|
| Shared goal tokens | 85.8 | 94.8 |
| Separate query tokens | 95.8 | 98.2 |
| Shared image tokens | 95.6 | 97.4 |
| **Image + Goal CrossAttn (FCCA)** | **97.6** | **98.8** |
- 두 브랜치가 같은 token을 공유하면 **gradient interference**로 long-horizon 급락(85.8). image token은 motion decoder 입력으로 두고 goal은 cross-attention으로 주입하는 최종 설계가 최고

### 다른 policy head로의 scalability (Table 5)
- StarVLA-π: 95.7 → 97.9 (+2.2, Long +8.2)
- StarVLA-OFT: 96.6 → 98.0 (+1.4)
- → 특정 action decoder에 종속되지 않는 **plug-in** 성격

---

## 8. 관련 연구 비교

| 모델 | 미래 예측 | motion tracking | 상호작용 | 추론 오버헤드 |
|------|----------|-----------------|----------|--------------|
| DreamVLA / UniVLA | ✓ (feature/latent) | ✗ | – | 有 |
| FlowVLA | ✗ | ✓ (optical flow CoT) | – | 有 |
| JOPAT | ✗ | ✓ (diffusion으로 track+action) | – | 有 |
| Spatial Forcing | teacher distill | 부분 | 분리 | 없음(training-only) |
| **FoMoVLA** | ✓ (compact feature) | ✓ (sparse 2D point) | **✓ FCCA** | **거의 없음(+9.4ms)** |

### 핵심 차이
- 미래 feature(goal)와 point tracking(motion)을 **명시적으로 결합**(FCCA)한 최초 계열 — 단순 multi-task 합산과 차별
- 모든 보조 supervision이 **학습 전용** → 배포 파라미터 Vanilla와 동일

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **2D image-space motion만**: tracking supervision이 view-dependent 2D motion에 국한 → dynamic scene의 3D geometry 미포착. 저자도 향후 3D motion prediction을 제안
2. **Camera/robot-state OOD 취약**(Table 2): 고정 viewpoint·초기 pose로 학습해 novel configuration 일반화가 language/background 대비 약함
3. **Real-world 부재**: LIBERO/RoboCasa 시뮬레이션 한정. 실 하드웨어 검증 없음
4. **Teacher 의존**: motion GT는 CoTracker-v3 품질에, foresight target은 EMA teacher에 상한이 걸림. teacher 오류가 supervision noise로 전이될 여지
5. **Hyperparameter 민감도**: λ1=0.1, λ2=0.3, K=16 등 고정. sensitivity 분석 부족

### Attribution 문제
- LIBERO 향상이 **결합(FCCA)** 덕인지 단순 보조 파라미터(+1.3%) 덕인지 — component ablation과 Table 4 coupling 비교가 상당 부분 해소하나(shared token 94.8 vs FCCA 98.8), "동일 파라미터의 무의미 보조 head" 대조는 없음

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — foresight(goal)와 point tracking(motion)을 zero-init cross-attention으로 결합한 관점이 신선. "where + how" 분해가 명확 |
| **Technical depth** | ★★★★☆ — coupling design, grid density, foresight decoder, tracking quality(ATE/Survival) ablation이 체계적 |
| **Experimental rigor** | ★★★☆☆ — LIBERO/RoboCasa/LIBERO-Plus는 강력하나 real-world 부재, teacher 의존 |
| **Practical impact** | ★★★★★ — 학습 전용이라 추론 오버헤드 +9.4ms/+0.1GB로 거의 공짜. 다른 policy head에도 plug-in 가능 |
| **Writing quality** | ★★★★☆ — 동기·방법·ablation 흐름이 명확 |

**강점**: "미래 상태(goal)"와 "motion path(how)"의 상보성을 FCCA로 명시적으로 엮은 깔끔한 프레이밍. 학습 전용 설계로 배포 비용이 사실상 0이면서 LIBERO 98.8, RoboCasa +9.1%p, OOD +6.4%p. point tracking 품질 지표(Survival@10px +17.3pp)가 FCCA의 실효를 정량 입증.
**약점**: 2D motion 한정, real-world 부재, camera/robot-state OOD 취약, teacher 품질 상한.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 보조 브랜치가 학습 전용인데 왜 성능이 오르나? | 지식이 VLM backbone 파라미터에 흡수됨. 추론엔 K=16 foresight 토큰만 남고(action head가 cross-attend) 나머지 제거 |
| 2 | 단순 multi-task와 뭐가 다른가? | 독립 loss 합산은 modest gain(98.3). FCCA로 motion을 미래에 conditioning하면 98.8, Long +1.8%p. Table 4 shared-token(94.8)이 gradient interference 증거 |
| 3 | 왜 sparse 8×8이 dense 16×16보다 좋은가? | dense는 long-horizon에서 열화(94.8). sparse가 task-relevant motion을 충분히 담으면서 노이즈·중복 감소 |
| 4 | LIBERO 포화인데 기여는? | RoboCasa +9.1%p, LIBERO-Plus OOD +6.4%p, LIBERO-Long이 진짜 무대. atomic task는 head-room 부족 |
| 5 | teacher(CoTracker/EMA) 실패 시? | supervision noise로 전이 가능. visibility masking으로 occluded point noise는 차단하나 tracker 상한은 존재 |
| 6 | 다른 backbone에 이식 가능? | Table 5에서 StarVLA-π(+2.2), StarVLA-OFT(+1.4)로 검증 — action decoder 비종속 plug-in |
| 7 | camera/robot OOD 약한 이유? | 고정 viewpoint·초기 pose 학습 + 순수 2D motion cue라 novel 3D configuration 일반화 한계. 저자가 3D motion을 향후 과제로 제시 |
| 8 | zero-init FCCA의 의미? | 출력 projection zero-init → 초기 identity로 사전학습 feature 보존, 학습하며 점진적 미래 주입. 안정적 fine-tuning의 핵심 |
| 9 | RoboCasa egocentric에서 ego-motion과 object motion 구분? | Fig 7에서 global camera displacement(ego)와 local arm motion을 함께 포착. change heatmap이 task-relevant 영역에 집중 |

<!-- VERIFIED: pdf -->
