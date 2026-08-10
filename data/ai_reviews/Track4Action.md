# Track4Action: Distilling World-Centric 3D Tracker into Vision-Language-Action Policies

> **한 줄 요약**: 시연 영상에서 액션 청크와 시간적으로 정렬된 K+1 프레임 클립을 frozen world-centric 3D tracker(Track4World)로 인코딩해 privileged supervision 타깃으로 만들고, 이를 current-observation VLA의 learnable track query에 distill해 flow-matching 액션 헤드를 조건화함으로써, 배포 시에는 tracker 없이 실행되면서 zero-shot LIBERO-Plus 82.3%, RoboTwin 2.0 clean/randomized 80.44%/81.48%, 실제 양팔 4개 태스크 평균 67.5%를 달성한 VLA.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **액션 라벨의 정보 부족**: 액션 청크는 로봇이 *어떻게 움직여야 하는지*는 지정하지만, *어떤 장면 포인트가 함께 움직이는지*, *접촉이 발생하는지*, *가시성이 어떻게 변하는지*는 인코딩하지 않음. 동일한 end-effector 변위가 자유 공간 통과일 수도, 물체 운반일 수도, 카메라 시점 변화일 수도 있음.
- **World-model policy**(3D-VLA, World Guidance, VLA-JEPA 등)는 미래 이미지/기하/시각 특징을 *예측*하지만, 그 예측 자체가 배포 시 부담이 되거나 타깃이 일반적인 비디오 콘텐츠에 머묾.
- **Geometry-aware policy**(SpatialVLA, Spatial Forcing, GeoPredict, LaMP)는 3D 구조나 frozen spatial representation과 정렬하지만, 타깃이 **단일 상태(single state)**에서 추출된 특징이거나 명시적 기하 변수임.
- 공통 공백: **시연된 액션 청크가 실제로 실현한(realized) 정확히 그 구간**의 motion-specialized 표현을 활용하지 않음.

### 핵심 질문
> **"action-aligned 3D tracker가 학습 중에는 VLA를 supervise하고, 로봇이 실제로 행동할 때는 사라질 수 있는가?"**

📌 [Figure 1 삽입] — Flow-tracking 모델은 4D 장면 진화를 포착하지만 태스크 인식이 없고, 기존 VLA는 태스크는 이해하나 4D 미래 동역학을 명시적으로 모델링하지 않음. Track4Action은 두 상보적 강점(Spatial Intelligence × Semantic Intelligence)을 연결.

---

## 2. 방법론 심층 분석

### 2.1 문제 정식화 (Section 3.1)

시각 t에서 정책은 언어 지시 l, 현재 RGB 관측 I_t, 선택적 로봇 상태 s_t를 받아 정규화된 액션 청크를 예측:

```
A*_t = (a_t, ..., a_{t+K-1}) ∈ R^{K × d_a}
```

학습 시에만 동일 샘플이 primary-view 클립 `V_{t:t+K} = (o_t, ..., o_{t+K})`를 제공. **K개 액션은 K+1 프레임 사이의 K개 전이(transition)에 정확히 대응**한다. 배포 시 정책은 `Â_t = π_θ(l, I_t, s_t)`만 계산하며 클립은 입력에서 완전히 제거됨.

> ❓ **예상 질문**: 왜 "미래 프레임 예측"이 아니라 "tracker feature 정렬"인가?
> **답변**: 픽셀 재구성은 태스크와 무관한 외형 변화까지 학습 부담으로 떠안는다. Track4World feature는 3D 궤적·장면 모션·visibility·카메라 파라미터로 이미 구조화된 표현이며, 게다가 **클립이 정확히 A*_t가 유발한 전이만 덮기 때문에** 타깃이 "무관한 미래 사건"이 아니라 "그 액션 청크에 결부된 world change"를 서술한다.

### 2.2 World-Centric Tracker Supervision (Section 3.2)

Frozen Track4World 모델 T가 action-aligned 클립을 받아 world-centric 3D trajectory + scene/motion/visibility/camera cue를 추정하고, 이를 pooling해 compact tracker feature를 만든다:

```
f_trk_t = sg[ Pool( T(V_{t:t+K}) ) ]
```

`sg`는 stop-gradient — Track4World로는 gradient가 흐르지 않는다 (완전 frozen teacher).

### 2.3 Track Query Distillation (Section 3.3)

1. VLA 백본이 현재 이미지+지시를 인코딩: `H_t = E_θ(l, I_t)`
2. **Learnable track queries Q**가 cross-attention으로 H_t를 읽음: `Z_t = D_θ(Q, H_t)`
3. Z_t를 pooling해 공유 정렬 공간에서 f_trk와 L2 매칭:

```
L_align = || z̄_t − f̄_trk_t ||²₂
```

여기서 Z_t는 멀티모달 백본과 액션 헤드 사이의 **action-facing bottleneck** 역할을 한다.

### 2.4 액션 생성과 Tracker-Free 배포 (Section 3.4)

- Z_t를 VLA 시퀀스와 **learned feature-wise gate**로 융합해 액션 조건 `H_cond_t` 구성
- Gate는 H_t의 semantic context를 보존하면서 distilled tracker cue를 액션 헤드에 노출
- **teacher feature f_trk는 절대 H_cond_t에 들어가지 않음**
- Flow-matching 액션 헤드(Lipman et al., 2023)의 표준 conditional velocity 목적함수 L_act 사용

```
L = λ_act · L_act + λ_align · L_align
```

배포 시 action-aligned 클립, frozen tracker, tracker 타깃, 정렬 브랜치를 모두 폐기하고 track-query + gated-fusion 경로만 유지.

> ❓ **예상 질문**: track query가 "student representation"인 동시에 "액션 헤드 입력"인 게 왜 중요한가?
> **답변**: 논문이 명시적으로 강조하는 두 번째 설계 선택이다. 만약 정렬 손실이 분리된 auxiliary decoder에만 걸리면 학습 신호가 제어에 도달하지 않는다. 같은 query가 양쪽 역할을 하기 때문에 alignment loss가 **action-facing representation 자체를 형성**한다. Ablation(alignment-free variant)의 큰 격차가 이를 뒷받침.

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO**: 4개 suite(Spatial, Object, Goal, Long) — **하나의 정책**을 4개 suite 전체에 걸쳐 학습
- **LIBERO-Plus**: 별도 학습 없음. 원본 LIBERO 시연으로만 학습하고 7개 perturbation 차원에 **zero-shot** 평가
- **RoboTwin 2.0**: 50개 양팔 태스크, clean / randomized 조건
- **실제 로봇**: 4개 태스크 × 50 시연

### 데이터 사용 패턴의 제약
- **시간 순서가 있는 시연 필수**: 학습 타깃이 클립 V_{t:t+K}를 읽으므로 **isolated image-action pair로는 supervision 불가**
- 독립적으로 움직이는 물체가 있으면 로봇 명령과 관측된 장면 전이 사이의 연관이 약해짐
- 전처리는 offline 비용만 발생, 배포 비용은 0

> ❓ **예상 질문**: 대규모 웹/OXE 데이터 사전학습 없이 되는가?
> **답변**: 논문은 OXE 등 대규모 사전학습을 사용하지 않는다. 대신 alignment-free variant와 **데이터·optimizer·schedule·모델 스케일·rollout 인터페이스를 모두 고정**한 통제 비교로 distillation 경로 자체의 효과를 분리한다. 이게 이 논문 실험 설계의 가장 깔끔한 부분.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLA 백본 | Qwen3.5-2B |
| 액션 헤드 | DiT-B flow matching (Peebles & Xie, 2023 아키텍처) |
| 총 파라미터 | **3.3B** |
| Teacher | Track4World (frozen, stop-gradient) |
| LIBERO 관측 | agent-view + wrist-view RGB, 8-D robot state |
| LIBERO 액션 | 7-D delta end-effector, **K = 8** |
| RoboTwin 관측 | 1 external + 2 wrist view, **robot state 없음** |
| RoboTwin 액션 | 14-D 양팔 joint + gripper, **K = 16** |
| 실제 로봇 | AgileX ALOHA-style, 6-DoF Piper 팔 ×2, 평행 그리퍼 |
| 손실 | λ_act·L_act + λ_align·L_align |
| Hardware / GPU | **논문에 명시 X** |
| λ 값 | **논문에 명시 X** |

---

## 5. 실험 설계 및 평가 프로토콜

논문은 평가를 세 질문 중심으로 구성:
1. **Comparison** — general / video-based / trace-guided / 3D-aware VLA 대비 우위가 있는가? (§4.2–4.4)
2. **Tracker distillation** — 완전한 distillation 경로가 alignment-free 대조군보다 제어를 개선하는가? (§4.5)
3. **Challenging settings** — long-horizon, 7종 zero-shot shift, 50-task 양팔 시뮬, 실물 조작에서 이득이 유지되는가? (§4.2–5)

평가 축:
- **In-distribution LIBERO** (4 suite, one policy)
- **Zero-shot LIBERO-Plus** (7 perturbation, 무적응, **official benchmark aggregation Avg** 사용 — 열 단순평균 재계산 아님)
- **RoboTwin 2.0** (50 태스크, clean + domain randomization)
- **실제 양팔 로봇** (4 태스크, 10 trial/태스크, success rate + 4단계 process score)

모든 벤치마크에서 **online task success rate** 보고.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1) — 단일 정책, 4 suite 평균

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | **98.8** | 98.2 | **98.0** | 92.4 | 96.9 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| FlowVLA | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| Track4Action **w/o Align.** | 94.0 | 99.2 | 97.4 | 86.2 | 94.2 |
| **Track4Action** | 95.0 | **99.6** | 97.6 | **95.8** | **97.0** |

- π₀.₅ 대비 **+0.1**, F1 대비 **+1.3** — Object와 Long에서 1위
- **핵심은 ablation**: alignment 제거 시 97.0 → 94.2 (**−2.8**), 그런데 그 격차가 **LIBERO-Long에 집중**(95.8 → 86.2, **−9.6**). Spatial/Object/Goal은 각각 1.0 / 0.4 / 0.2 변화에 불과.
- 해석: tracker supervision은 **다단계에 걸쳐 task-relevant state를 보존해야 할 때** 가장 유용하며, 이미 포화된 단기 horizon 점수를 균일하게 올리는 게 아님.

### LIBERO-Plus Zero-Shot OOD (Table 2)

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg** |
|--------|--------|-------|----------|-------|-----------|-------|--------|---------|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| LaMP | 64.5 | **69.6** | **88.2** | 95.3 | **97.4** | 76.9 | 73.8 | 79.3 |
| Track4Action **w/o Align.** | 58.3 | 52.0 | 84.3 | 92.7 | 90.2 | 79.4 | 76.1 | 74.7 |
| **Track4Action** | **76.6** | 60.7 | 87.1 | **96.3** | 96.1 | **88.4** | **78.7** | **82.3** |

- LaMP 대비 **+3.0**, alignment-free 대비 **+7.6** (74.7 → 82.3)
- **7개 차원 전부에서 alignment-free 대비 개선**, 카테고리 단위 회귀(regression) 없음 → 평균 이득이 특정 유리한 perturbation 하나에 기인하지 않음
- 최대 이득: **Camera +18.3**, Noise +9.0, Robot +8.7 — 3D tracker feature가 카메라 변화 정보를 담는다는 점과 정합적
- 전 차원에서 top-2 진입. 다만 LaMP가 Robot / Language / Background에서 여전히 강함 → 추가 robustness 여지 존재

### RoboTwin 2.0 (Table 3) — 50 태스크 양팔

| Method | Model Size | Clean | Rand. |
|--------|-----------|-------|-------|
| π₀.₅ | 3.3B | 42.98 | 43.84 |
| X-VLA | 0.9B | 72.80 | 72.84 |
| Motus w/o Pretrain | 8B | 77.56 | 77.00 |
| Track4Action **w/o Align.** | 3.3B | 41.32 | 39.90 |
| **Track4Action** | 3.3B | **80.44** | **81.48** |

- Motus w/o Pretrain 대비 **+2.88 / +4.48**, X-VLA 대비 **+7.64 / +8.64**
- **Randomization에서 오히려 +1.04 상승** (80.44 → 81.48). 같은 변화에서 Motus는 −0.56, X-VLA는 +0.04 → randomization 하에서 Motus 대비 마진이 2.88 → 4.48로 **확대**
- Distillation 경로의 기여가 여기서 압도적: **clean +39.12, randomized +41.58**

> ❓ **예상 질문**: RoboTwin에서 alignment-free가 41.32밖에 안 되는 게 이상하지 않은가?
> **답변**: 가장 날카로운 지점. LIBERO에서 격차가 2.8인데 RoboTwin에서 39.12는 한 자릿수가 다르다. 논문의 설명은 "포화된 in-distribution LIBERO는 headroom이 없고, 14-D 양팔 액션 공간 + 3-view 관측 + robot state 없음이라는 훨씬 어려운 설정에서 world-state 정보의 가치가 커진다"는 것. 다만 alignment-free 3.3B 베이스라인이 π₀.₅(3.3B, 42.98)와 거의 같은 수준이라는 점은, 이 백본이 tracker supervision 없이는 RoboTwin에서 특별히 강하지 않다는 뜻이기도 하다.

### 실제 양팔 로봇 (Figure 4, 5)

**In-distribution (10 trial × 4 태스크)**

| 지표 | alignment-free | π₀.₅ | **Track4Action** |
|------|---------------|------|-----------------|
| 평균 success rate | 42.5% | 65.0% | **67.5%** |
| 평균 process score | 60.6 | 75.0 | **75.0** |

- 4개 태스크 **전부**에서 alignment-free 대비 최소 **+20점**, 최대 **+30점**(drawer, cabbage — 둘 다 순서화된 다단계 태스크)
- Process score는 4개 태스크 모두에서 상승, 이득 범위 +7.5 ~ +25.0
- π₀.₅와의 비교는 균일하지 않음: drawer는 60% → 80%로 개선하지만 **towel folding은 40% vs π₀.₅ 70%** → 변형체(deformable) 조작이 남은 물리적 난제

**OOD (3개 held-out 변화)**

| 설정 | alignment-free | **Track4Action** |
|------|---------------|-----------------|
| 새 물체 배치 (chili transfer) | 30% | **60%** |
| Unseen towel color | 10% | **20%** |
| Textured desk background | 30% | **70%** |
| **평균** | 23.3% | **50.0%** |

- Unseen-color towel 20%는 여전히 낮음 — 물리 OOD deformable 조작의 어려움을 저자도 인정

---

## 7. Ablation 분석

이 논문의 ablation은 **단일 시스템 레벨 대조군**(alignment-free variant)에 집중되어 있다. 백본·액션 헤드·모델 스케일·rollout 인터페이스·데이터·optimizer·학습 스케줄을 모두 고정하고 tracker-distillation 경로만 제거.

| 평가 축 | w/o Align. | Track4Action | Δ |
|---------|-----------|-------------|---|
| LIBERO Avg | 94.2 | 97.0 | **+2.8** |
| LIBERO-Long | 86.2 | 95.8 | **+9.6** |
| LIBERO-Plus Avg | 74.7 | 82.3 | **+7.6** |
| RoboTwin 2.0 Clean | 41.32 | 80.44 | **+39.12** |
| RoboTwin 2.0 Rand. | 39.90 | 81.48 | **+41.58** |
| 실물 4-task 평균 success | 42.5% | 67.5% | **+25.0** |
| 실물 process score | 60.6 | 75.0 | **+14.4** |
| 실물 OOD 평균 | 23.3% | 50.0% | **+26.7** |

**패턴**: 가장 작은 마진은 포화된 in-distribution LIBERO, 가장 큰 마진은 zero-shot shift와 다른 양팔 액션 공간. 이는 "world state와 motion에 대한 요구가 큰 설정일수록 이득이 크다"는 논문의 주장과 일관.

📛 **부재하는 ablation**: λ_act / λ_align 비율, track query 개수, K(청크 길이)에 대한 민감도, gate 없이 단순 concat했을 때, alignment 손실을 auxiliary decoder에만 걸었을 때 — 어느 것도 정량화되지 않음.

---

## 8. 관련 연구 비교

| 모델 | Supervision 대상 | 배포 시 추가 의존성 | LIBERO Avg | LIBERO-Plus Avg |
|------|-----------------|-------------------|-----------|----------------|
| OpenVLA | expert action likelihood | 없음 | 76.5 | 15.6 |
| π₀ | flow-matching denoising | 없음 | 94.2 | 53.6 |
| π₀.₅ | flow matching + open-world | 없음 | 96.9 | — |
| SpatialVLA | 3D spatial representation | 3D 인코딩 | 78.1 | — |
| TraceVLA | visual trace prompting | trace 생성 | 74.8 | — |
| FlowVLA | visual CoT motion reasoning | 미래 프레임 예측 | 88.1 | — |
| WorldVLA | autoregressive action world model | world model | 81.8 | 25.0 |
| LaMP | 3D scene flow를 latent motion prior로 | — | — | 79.3 |
| **Track4Action** | **realized 구간의 pooled 3D tracker feature** | **없음 (tracker-free)** | **97.0** | **82.3** |

### 핵심 차이
- **Track2Act / TraceGen / 3DFlowAction / JOPAT**은 track·flow를 *예측 대상 또는 계획 매개체*로 노출 → 배포 시 그 예측이 파이프라인에 남음
- Track4Action의 타깃은 **배포 입력도 아니고 로봇이 재구성해야 하는 출력도 아님**. 순수하게 학습 중 표현을 형성하는 privileged signal
- **LaMP**(같은 그룹의 선행 연구, arXiv 2603.25399)와 가장 가까움 — LaMP는 3D scene flow를 latent motion prior로 쓰는 반면, Track4Action은 pretrained world-centric dense 3D tracker의 pooled feature를 action-aligned 구간에 대해 distill

---

## 9. 한계 및 미해결 문제

### 저자가 명시한 한계 (Section 6)
1. **시간 순서 시연 필수**: 학습 타깃이 realized 클립 V_{t:t+K}를 읽으므로 isolated image-action pair로는 supervision 제공 불가 → 대규모 이종 데이터 활용에 제약
2. **독립 운동 물체**: 로봇과 무관하게 움직이는 물체가 있으면 로봇 명령 ↔ 관측된 장면 전이의 연관이 약화
3. **Pooled 타깃의 정보 손실**: segment-level world motion을 강조하는 대신 **local contact correspondence를 버릴 수 있음**
4. **Track4World 오류 상속**: occlusion이나 빠른 모션에서 tracker가 틀리면 그대로 전파
5. **단일 primary camera view만 사용**
6. **물리 평가 범위 협소**: 1개 양팔 플랫폼, 4개 태스크, 3개 visual shift

### 리뷰어 관점의 추가 미비점
7. **하이퍼파라미터 미보고**: λ_act, λ_align, track query 수, 학습 GPU/스텝 — 재현성에 필요한 정보 부재
8. **백본 일반성 미검증**: Qwen3.5-2B + DiT-B 조합에서만 검증. "different VLA backbones와 호환"이라 주장하나 실증은 없음
9. **RoboTwin ablation 격차의 attribution**: 39점 차이가 tracker distillation의 본질적 기여인지, alignment-free 대조군이 그 설정에서 학습이 잘 안 된 것인지 분리되지 않음
10. **Deformable 조작 실패**: towel folding 40% vs π₀.₅ 70% — pooled segment-level 타깃이 세밀한 변형체 접촉을 못 담는다는 한계 3과 정합적

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "realized action segment"를 privileged supervision으로 쓰고 배포에서 완전히 제거한다는 프레이밍이 깔끔. 단, LaMP 등 선행 3D-alignment 계열과 연속선상 |
| **Technical depth** | ★★★☆☆ — 방법 자체는 단순(pooled feature + L2 정렬 + gated fusion). 수식은 명료하나 설계 선택의 정량적 근거가 얇음 |
| **Experimental rigor** | ★★★★☆ — 데이터/optimizer/스케일/rollout을 고정한 alignment-free 대조군, 시뮬 2종 + 50-task 양팔 + 실물 + OOD까지 커버리지 우수. 반면 내부 ablation은 부족 |
| **Practical impact** | ★★★★★ — **배포 비용 0**. 학습 시 offline 전처리만 추가하고 추론 인터페이스는 원래 VLA 그대로 유지 |
| **Writing quality** | ★★★★☆ — 세 질문 중심의 실험 구성이 명확. 실물 태스크 milestone 정의도 상세 |

**강점**: 아이디어가 "공짜 supervision"에 가깝다. 시연 영상은 이미 존재하고, tracker는 frozen이며, 배포 시 아무것도 남지 않는다. 그런데 LIBERO-Plus에서 7개 차원 전부 개선, RoboTwin randomization에서 오히려 성능 상승, 실물 OOD에서 23.3% → 50.0%. 이 정도 일관성이면 신호가 실재한다고 보기 충분하다.

**약점**: 방법론적 ablation이 사실상 on/off 하나뿐이라 "무엇이 효과를 내는가"가 분해되지 않는다. λ, query 수, gate 설계, pooling 방식 중 어느 것이 중요한지 알 수 없다. 하드웨어/하이퍼파라미터 미보고는 재현성 문제. RoboTwin alignment-free 41.32는 대조군 학습이 제대로 되었는지 의심하게 만드는 수치다.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO에서 97.0 vs π₀.₅ 96.9 — 0.1점 차이가 의미 있나? | LIBERO 순위 자체는 무의미. 논문의 실제 주장은 zero-shot LIBERO-Plus 82.3(LaMP +3.0)과 RoboTwin 80.44/81.48에 있다. LIBERO는 "in-distribution에서 손해 없음"의 sanity check |
| 2 | RoboTwin alignment-free 41.32는 대조군이 망가진 것 아닌가? | 가장 정당한 의심. 논문은 데이터·optimizer·스케일·rollout 고정을 명시하나, 39점 격차는 LIBERO의 2.8점과 스케일이 다르다. 14-D 양팔 + robot state 없음 + 3-view라는 어려운 설정 때문이라는 설명은 있으나 추가 진단(학습 곡선 등)은 제시되지 않음 |
| 3 | Randomized(81.48)가 clean(80.44)보다 높은 건 이상하지 않나? | +1.04. 논문은 "clean 우위가 randomization에서 유지된다"는 해석만 제시. Domain randomization이 일종의 augmentation으로 작용했을 가능성이 크지만 논문은 분석하지 않음. Motus는 −0.56, X-VLA는 +0.04로 대조 |
| 4 | Tracker feature를 pooling하면 정보가 다 뭉개지지 않나? | 저자도 한계로 인정 — segment-level world motion을 강조하고 local contact correspondence를 버릴 수 있음. 실제로 towel folding(정밀 변형체 접촉)에서 π₀.₅에 40% vs 70%로 패배 |
| 5 | Track4World가 틀리면? | Occlusion·빠른 모션에서의 오류를 그대로 상속한다고 명시. 다만 stop-gradient teacher라 오류가 정책 파라미터에 직접 backprop되지는 않고 잘못된 정렬 타깃으로만 작용 |
| 6 | 왜 LIBERO-Long에서만 +9.6이 나오나? | Tracker supervision이 "여러 단계에 걸쳐 task-relevant state를 보존"해야 할 때 유용하다는 게 논문의 해석. Spatial/Object/Goal은 1.0/0.4/0.2 변화에 그침. 실물에서도 최대 이득이 drawer·cabbage 같은 다단계 태스크에서 발생해 일관 |
| 7 | 다른 VLA 백본에 붙일 수 있나? | 논문은 "current-observation VLA 인터페이스를 보존하므로 다양한 백본·액션 디코더와 호환"이라 주장하지만 **실증은 Qwen3.5-2B + DiT-B 하나뿐**. Autoregressive 액션 토큰화 백본에서 gated fusion이 동일하게 작동할지는 미검증 |
| 8 | K(청크 길이)가 LIBERO 8, RoboTwin 16인데 정렬 타깃 품질에 영향은? | K가 클수록 클립이 길어져 pooled feature가 더 많은 전이를 요약. 민감도 분석 없음. K가 커질수록 pooling의 정보 손실(한계 3)이 심해질 것으로 예상되나 정량화되지 않음 |
| 9 | Alignment loss가 단순 L2인데 contrastive나 distribution matching이 낫지 않나? | 논문은 공유 정렬 공간에서의 L2만 사용하고 대안을 비교하지 않음. 명백한 후속 연구 여지 |
| 10 | "Privileged supervision"이라는 프레이밍이 기존 것과 다른가? | 개념 자체(teacher가 학습 때만 특권 정보 관찰)는 오래됨. 새로운 건 **특권 정보가 "액션 청크와 시간적으로 정확히 정렬된 realized 전이"라는 점**과, 그 표현이 auxiliary decoder가 아니라 액션 헤드를 조건화하는 query에 직접 distill된다는 점 |
| 11 | 실물 67.5% vs π₀.₅ 65.0% — 2.5점 차이로 우위를 주장할 수 있나? | 4 태스크 × 10 trial = 40 trial이라 통계적으로 약함. Process score는 75.0으로 동률. 실물에서의 진짜 주장은 π₀.₅ 대비가 아니라 **alignment-free 42.5% 대비 +25.0**과 OOD 23.3% → 50.0% |
| 12 | 학습 비용 증가는? | Track4World forward는 offline 전처리로 한 번만 수행 가능(frozen, stop-gradient). 논문은 "offline preprocessing은 추가하나 배포 비용은 없다"고 명시. 다만 실제 전처리 시간/저장 용량은 미보고 |

---

## 12. 재현 및 후속 연구 체크리스트

### 재현 시 확인해야 할 항목
- [ ] λ_act, λ_align 비율 — **논문 미보고**, 저자 문의 또는 grid search 필요
- [ ] Learnable track query 개수 및 cross-attention 디코더 D_θ 구성 — 미보고
- [ ] 공유 정렬 공간으로의 projection head 구조(z̄, f̄_trk 매핑) — 미보고
- [ ] Track4World pooling 방식(mean / attention / token-type별) — "scene, motion, camera 토큰을 pooling"까지만 명시
- [ ] Feature-wise gate의 구체적 형태(FiLM류인지, sigmoid gate인지) — 미보고
- [ ] 학습 GPU 종류·개수·스텝 수 — 미보고
- [ ] LIBERO-Plus는 반드시 **official benchmark aggregation Avg**를 사용 (열 단순평균 아님)

### 유망한 후속 방향
1. **다중 뷰 tracker 타깃** — 현재 primary view 하나만 사용 (한계 5)
2. **Local contact-aware 타깃** — pooled segment 표현에 접촉 지점 수준 정렬을 추가해 deformable 조작 실패 보완
3. **백본 이식성 검증** — π₀, OpenVLA-OFT, GR00T 등에 plug-in
4. **Isolated image-action pair 대응** — pseudo-clip 생성이나 인접 프레임 근사로 시간 순서 제약 완화
5. **Tracker 신뢰도 가중** — occlusion 구간에서 Track4World confidence로 L_align 가중해 오류 전파 억제
6. **Alignment 목적함수 대안** — contrastive / distribution matching 비교
7. **K 스케일링 분석** — 청크 길이와 pooled 타깃 정보량의 trade-off 곡선

<!-- VERIFIED: pdf -->
