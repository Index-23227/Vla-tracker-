# AVDC: Learning to Act from Actionless Videos through Dense Correspondences

> **한 줄 요약**: Action annotation 없는 비디오에서 text-conditioned video diffusion model로 미래 프레임을 생성하고, optical flow 기반 dense correspondence + rigid body transformation으로 3D action을 역추론하여, 165개 비디오만으로 Meta-World 43.1% (BC 16.2% 대비 2.7배) 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 로봇 학습 데이터는 **action annotation이 필수** → 수집 비용이 극도로 높음
- 인터넷 비디오에는 풍부한 manipulation dynamics가 있으나 action label 없음
- 기존 video-based 접근(R3M, VIP)은 representation learning에 국한 → action 직접 생성 불가
- UniPi는 video→action을 시도했으나, **inverse dynamics model에 action label이 필요** → 모순

### 핵심 질문
- **Action label 없이 비디오만으로 로봇의 실행 가능한 3D action을 생성할 수 있는가?**
- **Dense optical flow correspondence로 정확한 rigid body transformation을 복원할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Video Diffusion Model

Modified U-Net with **factorized spatial-temporal convolution**:
- Spatial conv: 각 timestep에 독립 적용
- Temporal conv: 각 spatial location에 독립 적용
- **CLIP-Text encoder (63M params)** → Perceiver attention pooling → time embedding injection
- Input: 첫 프레임을 noisy future frames에 concatenate하여 conditioning

| 환경 | Params | Resolution | Batch | Steps | GPU |
|------|--------|-----------|-------|-------|-----|
| Meta-World | 201M | 128×128 | 16 | 60K | 4× V100, ~24h |
| iTHOR | 109M | 64×64 | 32 | 80K | 4× V100, ~24h |
| Bridge | 166M | 48×64 | 32 | 180K | 4× V100, ~48h |

### 2.2 Dense Correspondence → 3D Action

핵심 파이프라인:

```
Generated video → GMFlow (optical flow) → Dense correspondence
     → Object mask (Language-SAM) → Contact point sampling (N=500)
     → RANSAC → 2D transformation → 3D rigid body transform via camera intrinsics
     → Subgoal sequence → Position controller
```

Rigid body transformation:

$$\mathcal{L}_{\text{Trans}} = \sum_i \left\|\mathbf{u}_t^i - \frac{(K T_t \mathbf{x}_i)_1}{(K T_t \mathbf{x}_i)_3}\right\|_2^2 + \left\|\mathbf{v}_t^i - \frac{(K T_t \mathbf{x}_i)_2}{(K T_t \mathbf{x}_i)_3}\right\|_2^2$$

> ❓ **예상 질문**: Rigid body transformation 가정이 얼마나 제약적인가?
> **답변**: 변형 가능한 물체(천, 로프)에는 부적합. Rigid assumption이 Meta-World의 단단한 물체에서는 적합하나, 실제 환경의 다양한 물체에서는 한계. 논문도 이를 인정.

### 2.3 Replanning

Open-loop 실행 중 실패 감지 시 replan:
- **Trigger**: "로봇 움직임 < 1mm for 15 consecutive steps while task unfulfilled"
- 새로운 observation으로 video regeneration → action 재계산
- 1-6회 replanning per episode

> ❓ **예상 질문**: Replanning이 비용이 높지 않은가?
> **답변**: 매우 비쌈. Video generation 10.57초 + flow 0.28초/frame + action regression 1.31초 + execution 1.53초 = **~18초/replan** (RTX 3080Ti). 1-6회 replan하면 총 30초~2분 per episode. Real-time 제어와는 거리가 멂.

---

## 3. 데이터 전략

| 데이터셋 | 비디오 수 | Resolution | Action Label | 용도 |
|---------|---------|-----------|-------------|------|
| Meta-World | **165** (5 demo × 11 task × 3 cam) | 128×128 | ✗ | 주요 평가 |
| iTHOR | **240** (12 obj × 4 room × 5 ep) | 64×64 | ✗ | Navigation |
| Visual Pusher | **198** (human videos) | Variable | ✗ | Cross-embodiment |
| Bridge | **33,078** | 48×64 | ✗ | Pre-training |
| Real-world FT | **20** | Variable | ✗ | Fine-tuning |

핵심: **165-240개 비디오만으로 학습** — action label 전혀 불필요.

---

## 4. 실험 결과 심층 분석

### Meta-World (Table 1, 11 Tasks)

| 모델 | Action Labels | Overall SR (%) |
|------|-------------|---------------|
| BC-Scratch | ✓ (15,216 pairs) | 16.2 |
| BC-R3M | ✓ | 15.4 |
| UniPi | ✓ (inverse dynamics) | 6.1 |
| AVDC-Flow (direct) | ✗ | 13.7 |
| AVDC-NoReplan | ✗ | 19.6 |
| **AVDC-Full** | **✗** | **43.1** |

**태스크별 편차가 매우 큼**:
- Best: Door-Close **89.3%**, Handle-Press **81.3%**, Door-Open **72.0%**
- Worst: Assembly **6.7%**, Hammer **8.0%**
- Assembly, Hammer처럼 **정밀 접촉이 필요한 태스크에서 극도로 약함**

### iTHOR Navigation (Table 2)

| 모델 | Overall SR (%) |
|------|---------------|
| BC-Scratch | 2.1 |
| BC-R3M | 0.4 |
| **AVDC** | **31.3** |

- Navigation에서 BC 대비 압도적 (2.1% → 31.3%)
- Camera motion = inverse object motion 이중성 활용

### Cross-Embodiment (Visual Pusher)

- **90% zero-shot success rate** (40 runs)
- Human video만으로 학습 → 로봇에서 zero-shot 실행

### Segmentation Ablation

| Masks | SR (%) |
|-------|--------|
| GT masks | 43.1 |
| Predicted masks (Language-SAM) | 34.5 (-8.6%p) |

---

## 5. Ablation 분석

| Variant | Overall SR (%) | 특징 |
|---------|---------------|------|
| AVDC (ID) = UniPi style | 6.1 | Inverse dynamics (action label 필요) |
| AVDC (Flow) | 13.7 | Direct flow → action (no video gen) |
| AVDC (No Replan) | 19.6 | Open-loop (replan 없음) |
| **AVDC (Full)** | **43.1** | **Two-stage + replanning** |

- Replanning 기여: **+23.5%p** (19.6→43.1)
- Video generation 기여: +5.9%p (direct flow 13.7 vs no-replan 19.6 → video gen 없이도 어느 정도)

### DDIM Acceleration

| Steps | SR (%) | Speedup |
|-------|--------|---------|
| 100 (DDPM) | 43.1 | 1x |
| 10 (DDIM) | 37.5 (-5.6%p) | 10x |

---

## 6. 추론 비용 (RTX 3080Ti)

| 단계 | 시간 |
|------|------|
| Video generation | **10.57s** (~1.51s/frame) |
| Flow prediction | 0.28s/frame pair |
| Action regression | 1.31s |
| Execution | 1.53s |
| **Total per replan** | **~18s** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점 (저자 명시 포함)
1. **Occlusion**: 저자 명시 — "when majority of object occluded by robot arm, algorithm may lose track"
2. **Optical flow 한계**: 저자 명시 — "struggles under rapidly changing lighting or large object movements." 특히 작은 물체의 pixel-level 오류가 3D 공간에서 큰 오류로 증폭
3. **Force 정보 부재**: 저자 명시 — "force information crucial for manipulation, unobtainable from RGB videos"
4. **Grasp/contact 예측**: 저자 명시 — 비디오에서 로봇의 grasp 방식을 직접 유추 불가 (사람 손 vs 로봇 그리퍼)
5. **극도로 느린 추론**: ~18s/replan × 1-6회 = 30s~2min per episode. Real-time 제어와 거리가 멂
6. **정밀 조작 약점**: Assembly 6.7%, Hammer 8.0% — contact-rich task에서 거의 실패
7. **Real-world 실패 분석**: 10건 중 8건 실패: 75%가 wrong plan (잘못된 물체/목표), 25%가 video generation discontinuity

### Attribution 문제
- 43.1%의 성능이 **video generation** 품질 때문인지, **replanning 전략** 때문인지 분리 필요. Ablation에서 replanning 기여(+23.5%p)가 video gen 기여보다 훨씬 큼 → "좋은 action 추출"보다 "실패 시 재시도"가 핵심 요인일 수 있음

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Actionless video → executable policy의 패러다임이 매우 참신 |
| **Technical depth** | ★★★★☆ — Dense correspondence + rigid transform의 기하학적 접근이 우아 |
| **Experimental rigor** | ★★★★☆ — 4개 환경, cross-embodiment, ablation 포괄적 |
| **Practical impact** | ★★★☆☆ — 18s/replan의 latency, 43.1% SR로 실용성 제한 |
| **Writing quality** | ★★★★☆ — 명확한 동기와 체계적 실험 |

**강점**: Action 없는 비디오 활용이라는 근본적으로 중요한 문제를 다루며, cross-embodiment (human→robot) 가능성을 실증. **약점**: 43.1% SR, ~18s latency, contact-rich 실패로 practical deployment는 아직 먼 미래.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Replanning이 +23.5%p면, 좋은 video보다 "많이 시도"가 핵심 아닌가? | 맞음. Replanning = closed-loop retry. Video 품질보다 failure recovery가 성능의 주요 요인. Replanning 없으면 19.6%에 불과 |
| 2 | 165개 비디오 vs 15,216개 action-labeled frames — 공정한 비교인가? | AVDC는 label 없이 165 video, BC는 15K labeled. 데이터 "양"은 BC가 많으나 "annotation 비용"은 AVDC가 훨씬 적음 |
| 3 | Rigid body assumption이 깨지는 경우는? | Deformable objects (천, 로프), articulated objects (서랍 내부 물체). Meta-World는 rigid 위주라 유리 |
| 4 | RT-2나 OpenVLA 같은 대규모 VLA와 비교하면? | 직접 비교 없음. 그러나 action label 없이 학습한다는 점에서 상보적 — AVDC를 VLA의 data augmentation에 활용 가능 |
| 5 | 추론 10.57초를 줄이려면? | DDIM 10-step으로 10x 가속 가능하나 -5.6%p 하락. Consistency model이나 latent diffusion으로 추가 가속 가능 |
| 6 | 왜 Assembly와 Hammer에서 실패하는가? | 두 태스크 모두 정밀한 접촉 제어 필요. Optical flow의 pixel-level 오류가 3D에서 증폭되어 contact positioning 실패 |

<!-- VERIFIED: pdf -->
