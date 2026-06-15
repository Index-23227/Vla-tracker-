# GEAR-VLA: Learning Geometry-Aware Action Representations for Generalizable Robotic Manipulation

> **한 줄 요약**: Qwen2.5-VL 백본에 VGGT 3D 공간 인코더를 zero-initialized connector로 주입하고, FAST discrete action token + causal VQ-VAE latent action ID로 coarse-to-fine 사전학습한 뒤 gradient-decoupled DiT continuous action expert를 붙여, LIBERO 98.7%·LIBERO-Plus 88.7% zero-shot·RoboTwin 2.0 91.06/89.92%·실로봇 AgileX 85.9%·미사전학습 embodiment LDT-01 81.0%·6,360-trial universal grasping 90.1%를 달성한 geometry-aware cross-embodiment VLA.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Action token VLA**(OpenVLA, π₀, FAST 계열): 연속 액션을 양자화한 토큰으로 VLM에 흘려보내는데, 이 토큰이 **low-level trajectory에 종속**되어 VLM이 추론보다 모방에 치우침
- **3D-aware VLA**(SpatialVLA, depth/3D positional encoding 계열): 깊이·3D feature는 VLM의 semantic 공간과 **자연스럽게 정렬되지 않음**. VLM backbone에 직접 주입하면 사전학습된 의미 표현이 disturb 됨
- **Cross-embodiment**(X-VLA soft prompt, embodiment-specific action head): robot identity가 공유 정책 표현에 entangle 되어 데이터가 imbalanced하거나 target embodiment가 unseen이면 전이 효율 저하

### 핵심 질문
- **action semantics, 3D geometry, embodiment 정규화를 하나의 통합된 manipulation representation으로 학습할 수 있는가?**
- **VLM의 semantic 표현을 망가뜨리지 않으면서 3D 기하 정보를 주입하는 방법은?**
- **사전학습되지 않은 새로운 robot embodiment로 전이가 가능한가?**

📌 [Figure 1 삽입] — GEAR-VLA 3축 설계: coarse-to-fine action learning, semantic-aligned 3D integration, embodiment canonicalization.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

GEAR-VLA는 세 가지 모듈을 결합:
- **VLM 백본**: Qwen2.5-VL (2D visual encoder + LLM); semantic 시각 pathway는 **freeze**
- **3D Spatial Encoder**: VGGT (multi-view consistency 기반 feed-forward 3D model), **trainable**, **zero-initialized connector**로 VLM에 주입
- **Action Expert**: gradient-decoupled DiT (Diffusion Transformer); FAST discrete action token + causal VQ-VAE latent action ID 조건 → 30-step (1초 @ 30 Hz) continuous action chunk

### 2.2 Coarse-to-Fine Action Learning

학습은 두 단계:
1. **Embodied VLM Pretraining (stage 1, 350K iter)**: FAST 토크나이저로 discrete action 감독 + causal VQ-VAE로 action-free manipulation video에서 latent action ID 추출. VLM이 embodied reasoning과 discrete action 의미를 모두 학습.
2. **Continuous Policy Learning (stage 2, 700K iter)**: latent action token이 DiT action expert에 조건을 전달. 핵심은 **stop-gradient (sg)**: continuous action loss가 VLM 백본으로 backprop되지 않도록 차단.

> ❓ **예상 질문**: 왜 continuous loss를 VLM에 backprop하지 않는가? 학습 신호가 줄어들지 않나?
> **답변**: continuous action regression loss는 noisy하고 low-level이라 VLM의 high-level semantic 표현을 손상시킬 위험이 크다. discrete action token과 latent action ID로 이미 "action semantics"를 학습했으므로, DiT는 그 위에서 detail만 채우는 역할. Table 14 ablation에서 latent action ID 제거 시 88.7→87.1로 떨어지는 게 latent supervision의 효과를 보여줌.

### 2.3 Semantic-Aligned 3D Integration

- **VGGT** 선택 이유: depth/단일 프레임이 아닌 **multi-view consistency**로 scene layout, object shape, spatial relation을 포착
- **Zero-initialized connector**: 학습 초기엔 3D feature 기여가 0 → VLM의 사전학습 의미 공간 보존
- **2D pathway freeze + 3D pathway trainable**: 점진적으로 3D feature가 학습되며 의미와 기하가 정렬됨

> ❓ **예상 질문**: 왜 2D ViT를 같이 학습하지 않는가?
> **답변**: Table 3 ablation: VGGT 제거 시 88.7→85.1 (-3.6), VGGT는 두되 trainable 적응을 끄면 의미 공간 disturb. Figure 7에서 frozen 2D + trainable VGGT 조합이 ImageNet feature space의 class boundary를 가장 안정적으로 유지함을 visualization으로 보임.

### 2.4 Embodiment Canonicalization

- **Embodiment-aware state projector** (per-embodiment lightweight projector): joint angle, EE pose 등 proprioceptive 차이를 low-level interface에서 흡수
- **Embodiment-invariant action space**: **현재 EE pose 기준 relative end-effector action** — robot kinematics 차이를 외부화
- soft prompt 등 robot identity를 semantic 토큰으로 주입하는 방식은 **회피**

> ❓ **예상 질문**: X-VLA의 soft prompt와 비교했을 때 정말 더 나은가?
> **답변**: Table 13에서 직접 비교 — soft prompt 추가하면 오히려 88.7→85.0 (-3.7). robot identity가 shared VLA 표현과 entangle되어 imbalanced data에서 해로움. 본 논문은 state-level adaptation을 선호.

---

## 3. 데이터 전략

### 학습 데이터 (사전학습)
- Vision-language understanding (instruction tuning data)
- Spatial grounding, trajectory reasoning, pointing, affordance understanding
- Robot trajectories (FAST 토큰으로 discretize)
- Action-free manipulation videos (causal VQ-VAE로 latent action ID 추출)
- 모든 action-labeled data를 **30 Hz로 resample**, **30-step chunk** = 1초 예측 horizon

### 다운스트림 데이터
- **LIBERO**: libero-union (4 suite 통합 학습)
- **LIBERO-Plus**: zero-shot 평가만, 학습 없음
- **RoboTwin 2.0**: 50 task, clean/randomized 두 설정
- **실로봇**:
  - AgileX (14-DoF bimanual): task당 200 demos, 1 색상 학습
  - LDT-01 (16-DoF, 사전학습에 미포함): task당 200 demos lightweight adaptation
  - Universal grasping: 35 object 100 demos 학습 → 212 unseen objects test

> ❓ **예상 질문**: 사전학습 데이터의 정확한 규모는?
> **답변**: 논문 본문에 robot trajectory + video 데이터 표는 있으나(Tables 4-5, 부록) 총 시간/궤적 수치는 본문에 명시 X. 240 H200 × 350K + 700K iteration이라는 컴퓨트로 간접 추정 가능.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM 백본 | Qwen2.5-VL (semantic 시각 pathway freeze) |
| 3D Encoder | VGGT (trainable, zero-init connector) |
| Action Expert | gradient-decoupled DiT |
| Action 토큰 | FAST tokenizer (discrete) + causal VQ-VAE latent action ID |
| Action chunk | 30-step @ 30 Hz = 1초 horizon |
| Action space | embodiment-invariant relative EE pose |
| Optimizer | AdamW, bf16, LR 2e-5, 3% warmup, constant schedule |
| Pretrain (stage 1) | 240 H200 GPUs, per-GPU batch 8, **350K iter** |
| Pretrain (stage 2) | 240 H200 GPUs, per-GPU batch 4, **700K iter** |
| LIBERO finetune | 56 H200, batch 4, 12K iter |
| 기타 | gradient checkpointing, image augmentation, wrist + state attention dropping |

---

## 5. 실험 설계 및 평가 프로토콜

평가는 4축:
1. **표준 시뮬레이션** — LIBERO (in-distribution), LIBERO-Plus (zero-shot OOD), RoboTwin 2.0 (clean/rand)
2. **실로봇 외형 일반화** — AgileX, 3 bimanual tasks, 3 unseen colors/task
3. **Cross-embodiment 전이** — LDT-01 (사전학습에 미포함 embodiment)
4. **대규모 universal grasping** — 212 unseen objects × 3 scene settings × 10 trial = 6,360 trial per method

모든 baseline은 가능한 한 동일 프로토콜로 reproduce, 일부는 official 수치.

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO (Table 1, Table 8) — Standard Simulation

| Suite | OpenVLA | π₀ | π₀.₅ | X-VLA | ACoT | **GEAR-VLA** |
|-------|---------|----|----|-------|------|--------------|
| Spatial | 84.7 | 98.0 | 98.8 | 98.2 | 99.4 | **99.7** |
| Object | 88.4 | 96.8 | 98.2 | 98.6 | 99.6 | **99.8** |
| Goal | 79.2 | 94.4 | 98.0 | 97.8 | 98.8 | **98.4** |
| Long | 53.7 | 88.4 | 92.4 | 97.6 | 96.0 | **96.8** |
| **Avg** | 76.5 | 94.4 | 96.9 | 98.1 | 98.5 | **98.7** |

- ACoT 대비 +0.2pp, 사실상 **saturate**된 영역
- Goal에서는 ACoT(98.8) > GEAR(98.4) — 매우 좁은 격차

### 6.2 LIBERO-Plus (Table 9) — Zero-shot OOD

7개 perturbation (Cam, Robot, Lang, Light, BG, Noise, Layout) zero-shot:

| Method | Cam | Robot | Lang | Light | BG | Noise | Layout | Avg |
|--------|-----|-------|------|-------|----|----|-------|------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| π₀ | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | 69.4 |
| π₀.₅ | 75.8 | 79.4 | 83.3 | 95.5 | 95.0 | 89.6 | 87.0 | 85.7 |
| ACoT | 72.6 | 82.6 | 87.5 | 97.7 | 96.5 | 87.8 | 88.1 | 86.6 |
| **GEAR-VLA** | **82.6** | **84.1** | 82.4 | **97.9** | 93.1 | **90.0** | **89.4** | **88.7** |

- 가장 어려운 **Camera (+10pp)**, **Robot (+1.5pp)** 차원에서 큰 격차 — geometry-aware + embodiment-invariant 설계의 효과
- Language perturbation에서는 ACoT(87.5) > GEAR(82.4) — semantic CoT의 우위가 잔존

### 6.3 RoboTwin 2.0 (Tables 10-11) — 50 task aggregate

| Setting | π₀.₅ | X-VLA | ACoT | **GEAR-VLA** |
|---------|------|-------|------|--------------|
| Clean Avg | 82.74 | 72.88 | 80.06 | **91.06** |
| Rand Avg | 76.76 | 72.84 | 78.72 | **89.92** |

- **Clean +11.0pp / Rand +11.2pp** 격차 — 이번 논문의 가장 인상적 결과
- 특히 어려운 task (Hanging Mug 47/38, Scan Object 83/79, Move Stapler Pad 89/84)에서 baseline 대비 큰 향상

### 6.4 Real-World

**AgileX (14-DoF bimanual, 3 task)**: GEAR 85.9% > π₀.₅ 83.0%, X-VLA 76.0%, ACoT 77.8%

**LDT-01 (16-DoF, 미사전학습 embodiment)**: GEAR 81.0% > π₀.₅ 73.9%, ACoT 70.3%
- 사전학습에 유사 embodiment가 없음에도 **+7.1pp over π₀.₅** — embodiment canonicalization 효과 입증

### 6.5 Universal Grasping (Table 2) — 6,360 trials

| Object Category | π₀.₅ | DexGraspVLA | **GEAR-VLA** |
|----------------|------|-------------|--------------|
| Axisymmetric | 81.9 | 86.4 | **90.7** |
| Block-like | 83.0 | 87.7 | **92.4** |
| Irregular | 69.9 | 77.1 | **86.7** |
| Tool | 66.7 | 75.6 | **86.7** |
| Bagged | 73.3 | 81.3 | **86.7** |
| **Overall Avg** | 79.1 | 84.4 | **90.1** |

- Irregular/Tool 같은 형태가 어려운 객체에서 **+10~16pp** 격차 — 3D geometry encoding의 효과 명확

> ❓ **예상 질문**: 6,360 trial이 충분한 통계인가?
> **답변**: 5 category × 3 setting (Sparse, Dense, BG/Light) × ~14 object × 10 trial 구조. 객체별 통계는 sparse하지만 aggregate 단위에서는 baseline 대비 큰 격차(+5.7pp over DexGraspVLA)라 noise level 위.

---

## 7. Ablation 분석

### 7.1 Discrete Action Learning (Table 3)

| Variant | Avg | Δ |
|---------|-----|---|
| Full model | 88.7 | — |
| w/o latent action IDs | 87.1 | -1.6 |
| w/o FAST tokens | 85.4 | -3.3 |

- FAST 토큰이 latent ID보다 더 큰 기여 — discrete trajectory supervision의 중요성

### 7.2 3D Geometry Integration

| Variant | Avg | Δ |
|---------|-----|---|
| Full | 88.7 | — |
| w/o VGGT | 85.1 | -3.6 |
| Frozen VGGT (no adapt) | 86.9 | -1.8 |
| w/o zero-init connector | 84.8 | -3.9 |
| Unfreeze 2D ViT | 83.7 | -5.0 |

- **zero-init connector 제거가 가장 치명적** (-3.9), 2D ViT freeze가 핵심 (Figure 7 visualization)
- 단순히 3D feature 더하는 것이 아니라 *어떻게 정렬하는가*가 결정적

### 7.3 Embodiment Canonicalization (Table 13)

| Variant | Avg | Δ |
|---------|-----|---|
| Ours | 88.7 | — |
| w/o all | 86.9 | -1.8 |
| X-VLA soft prompt 추가 | 85.0 | **-3.7** |
| w/o State Projector | 86.7 | -2.0 |

- soft prompt가 오히려 해로움 — robot identity entanglement 가설 지지

### 7.4 Latent Action Modeling (Table 14)

| Variant | Avg |
|---------|-----|
| w/o latent action IDs | 87.1 |
| LAPA-style (initial/final only) | 88.1 |
| Ours (5 frames in 1s) | **88.7** |

- 5-frame continuous latent action이 LAPA-style 양 끝점 supervision보다 우수

### 7.5 Attention-level Modality Dropping (Table 12)

| Variant | Avg |
|---------|-----|
| w/o Dropping | 87.7 |
| Drop Wrist only | 88.1 |
| Drop State only | 88.5 |
| **Drop Wrist + State** | **88.7** |

- wrist-view와 robot state 둘 다에 의존하지 않도록 dropout — head-view + visual evidence 강제

---

## 8. 관련 연구 비교

| 모델 | 3D | Action Token | Embodiment | LIBERO Avg | LIBERO-Plus Avg | RoboTwin2.0 Clean |
|------|----|----|------------|-----------|----------------|-------------------|
| OpenVLA | ✗ | discrete | shared head | 76.5 | 15.6 | 38.3 |
| π₀ | ✗ | flow matching | shared | 94.4 | 69.4 | 48.4 |
| π₀.₅ | △ | flow matching | shared | 96.9 | 85.7 | 82.7 |
| X-VLA | ✗ | continuous | soft prompt | 98.1 | 68.3 | 72.9 |
| ACoT | ✗ | discrete CoT | shared | 98.5 | 86.6 | 80.1 |
| SpatialVLA | partial (3D PE) | discrete | shared | 78.1 | — | — |
| **GEAR-VLA** | **VGGT trainable** | **FAST + latent + DiT** | **state canonicalization** | **98.7** | **88.7** | **91.1** |

### 핵심 차이
- **3D 통합 방식**: depth/positional encoding 대신 multi-view VGGT를 zero-init connector로 점진 주입 — VLM 의미 표현 보존
- **Embodiment 처리**: shared head/soft prompt 대신 **state-level adaptation + invariant action space**
- **Action representation**: FAST discrete + latent ID + DiT continuous의 **계층화** — discrete가 semantics를, DiT가 continuous detail을 담당

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Open-source 부재**: 코드/체크포인트 공개 여부 불확실 (project page만 존재). 재현 어려움.
2. **컴퓨트 진입 장벽**: 240 H200 GPU × (350K + 700K) iteration은 대부분 lab가 재현 불가. 학계 reproducibility 약함.
3. **Language perturbation에서 ACoT에 뒤짐** (LIBERO-Plus Lang 82.4 vs 87.5) — semantic reasoning에 약점
4. **Action expert size 미보고**: DiT의 정확한 layer/parameter 미공개. 추론 latency도 본문에 없음.
5. **VGGT 의존성**: VGGT는 별도 사전학습된 large 3D 모델. 더 가벼운 대안과의 비교 ablation 없음.
6. **LDT-01 외 embodiment 다양성 부족**: cross-embodiment 주장이 강하지만 실제 실험은 AgileX + LDT-01 두 개로 한정.

### Attribution 문제
- LIBERO 98.7 vs ACoT 98.5는 거의 saturation, novelty가 RoboTwin 2.0과 real-world에 집중되어 있음
- 사전학습 데이터 규모/품질 효과와 architectural novelty의 분리 분석 부족

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — coarse-to-fine action learning + zero-init VGGT integration + embodiment canonicalization 3축 결합은 신선. 개별 component는 기존이지만 조합과 alignment 방식이 독창적 |
| **Technical depth** | ★★★★★ — Table 1~14 + 부록 표까지 체계적 ablation. 특히 3D integration 방식별 비교(Figure 7)와 embodiment 방식 비교(Table 13)가 강력한 근거 제공 |
| **Experimental rigor** | ★★★★★ — sim(LIBERO/LIBERO-Plus/RoboTwin 2.0) + real(AgileX, LDT-01) + 6,360-trial universal grasping. cross-embodiment까지 다룬 종합 평가 |
| **Practical impact** | ★★★☆☆ — 240 H200 GPU 컴퓨트와 open-source 불확실성이 채택 장벽. 그러나 cross-embodiment 결과(LDT-01 +7.1pp)는 산업체 유인이 큼 |
| **Writing quality** | ★★★★☆ — 3축 설계가 명확. Figure 1의 overview가 직관적 |

**강점**: 3D geometry, action token, embodiment 세 표현 gap을 **분리 처리 + low-level 인터페이스로 변형 흡수**한다는 통합적 시각. RoboTwin 2.0 50-task에서 +11pp, universal grasping에서 +5.7pp는 단순 SOTA push가 아닌 일반화 능력의 실증. LDT-01 cross-embodiment 결과가 가장 인상적.

**약점**: 거대 컴퓨트, open-source 불확실, language reasoning에서 ACoT에 약간 뒤짐. VGGT 대체 가능성 ablation 부재.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO에서 ACoT 대비 +0.2pp가 의미 있나? | LIBERO는 saturation. 의의는 LIBERO-Plus(+2.1), RoboTwin2.0(+11.0), 실로봇(+7.1)에 있음. LIBERO는 sanity check에 가까움 |
| 2 | 왜 zero-init connector가 필수인가? | Table 3에서 -3.9pp로 가장 치명적. 학습 초기 3D feature 기여가 0이라 VLM 의미 공간이 disturb되지 않고, 점진적으로 학습 신호가 흐르도록 함. Figure 7에서 class boundary 안정성 시각화 |
| 3 | VGGT 의존성 — 더 가벼운 3D encoder로 대체 가능? | 논문은 VGGT만 사용. multi-view consistency가 핵심이라 단일 frame depth로는 fallback 어려울 듯. ablation 없음 |
| 4 | Stop-gradient (DiT → VLM)가 정말 필요한가? | continuous action loss가 noisy/low-level이라 VLM semantic을 corrupt. 직접 비교 ablation은 본문에 명확히 없으나 "gradient-decoupled"라는 명명이 핵심 설계 의도 |
| 5 | X-VLA soft prompt가 왜 해로운가? | Table 13: -3.7pp. robot identity가 shared VLA 표현과 entangle되어 imbalanced data에서 잘못된 inductive bias. state-level adaptation은 표현 공간 외부에서 차이 흡수 |
| 6 | LDT-01에서 200 demos로 81%는 진짜인가? | "lightweight adaptation" — pretrained backbone freeze 정도/방법은 부록 참조 필요. 적응 효율은 embodiment-invariant action space가 가능하게 함 |
| 7 | RoboTwin 2.0 +11pp 격차의 원인은? | RoboTwin 2.0은 multi-task 50개로 randomization 강함. 3D geometry + embodiment canonicalization이 이런 종류의 시뮬레이션 variability에 robust. Hanging Mug(47/38), Move Stapler Pad(89/84) 같은 fine-grained task에서 격차 |
| 8 | Universal grasping 90.1%가 기존 dex-grasping보다 좋다는 의미? | DexGraspVLA(84.4) 대비 +5.7pp. 6,360 trial이라는 큰 표본. tool/irregular object에서 +10pp 이상 — 3D representation 효과 |
| 9 | FAST + latent action ID 둘 다 필요한가? | Table 3: latent 제거 -1.6, FAST 제거 -3.3. FAST가 더 큰 기여지만 latent ID는 action-free video 활용 — 둘은 보완적 데이터 소스에 대응 |
| 10 | 추론 latency는? | 본문 미보고. 30-step chunk @ 30 Hz = 1초 horizon. DiT inference + VLM forward 결합이므로 X-VLA 같은 단순 continuous decoder보다 느릴 가능성 |
| 11 | Coarse-to-fine 2단계가 1단계 joint training보다 나은가? | 본문에 stage 분리 ablation 명시는 없으나, embodied pretraining이 discrete action semantics를 먼저 학습한 뒤 DiT를 위에 얹는 설계 철학이 stop-gradient와 일관 |
| 12 | 진정한 cross-embodiment vs 단순한 fine-tuning의 차이는? | LDT-01에 200 demos는 fine-tuning이지만 **embodiment-invariant action space + 사전학습된 shared representation** 덕에 효율적. π₀.₅(73.9) 대비 +7.1은 representation 일반화 증거 |

<!-- VERIFIED: pdf -->
