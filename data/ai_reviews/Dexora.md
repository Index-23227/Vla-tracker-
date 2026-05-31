# Dexora: Open-source VLA for High-DoF Bimanual Dexterity

> **한 줄 요약**: T5(언어) + SigLIP(시각) 인코더에 28-layer decoder-only transformer diffusion policy를 결합하여 36-DoF dual-arm + dual-hand 시스템을 제어하는 오픈소스 VLA. 핵심 기여는 *offline discriminator(30M)가 클립-수준 quality weight*를 제공해 저품질 teleoperation 데모를 down-weighting하는 **data-quality-aware training recipe**. 기본 12 태스크 89.6%, 정밀 dexterous 6 태스크 66.7%로 GR00T-N1·π0·Diffusion Policy를 크게 상회. ICRA 2026.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 오픈소스 VLA(OpenVLA, π0, GR00T-N1)는 대부분 **6-7 DoF gripper** 기반 — 진정한 dexterous manipulation(pen twirling, cap twisting, leek cutting)에는 자유도 부족
- 36-DoF 시스템(dual 6-DoF arm + dual 12-DoF XHAND)을 다루는 오픈소스 VLA는 사실상 부재
- Teleoperation 데이터는 본질적으로 *품질 편차*가 큼 — 같은 task에서도 jerky/missed grasp/recovery가 섞임. 표준 imitation learning은 모든 demo를 동등하게 학습 → 저품질 trajectory가 정책을 망침

### 핵심 질문
- **36-DoF 양손 dexterous manipulation을 다루는 오픈소스 VLA를 어떻게 만들 것인가?**
- **Teleoperation demo의 *품질 차이*를 어떻게 정책 학습에서 명시적으로 활용할 것인가? (re-labeling이 아니라 *clip-level weighting*으로)**

📌 [Figure 1] — Custom exoskeleton backpack + Apple Vision Pro hand tracking, 36-DoF AIRBOT+XHAND, MuJoCo digital twin

---

## 2. 방법론 심층 분석

### 2.1 시스템 구성

**로봇 플랫폼 (36 DoF)**:
- 2 × AIRBOT 6-DoF arms = 12 DoF
- 2 × XHAND 12-DoF dexterous hands = 24 DoF
- 합계 **36-dim joint command space**

**Teleoperation rig**:
- 팔: custom exoskeleton backpack — 직접 kinematic mapping
- 손: Apple Vision Pro hand tracking
- MuJoCo digital twin으로 sim teleop 동시 수집

### 2.2 정책 아키텍처

| 컴포넌트 | 사양 |
|---------|-----|
| Language encoder | **T5** (frozen 추정) |
| Vision encoder | **SigLIP** (multi-view) |
| Policy backbone | Decoder-only Transformer **diffusion policy**: 28 layers, hidden 1024, 16 attention heads |
| Output | 36-dim synchronized joint commands (action chunk) |
| Discriminator | 12 layers, hidden 512, 8 heads, **~30M params** |

> ❓ **예상 질문**: 왜 T5 + SigLIP인가, Qwen·LLaVA 같은 unified VLM은 안 쓰나?
> **답변**: 본 논문은 *큰 VLM 없이도* 36-DoF dexterity를 demonstrate하는 minimal 디자인을 의도. T5 + SigLIP은 가볍고 fine-tune 비용이 낮음. 트레이드오프: 일반적 instruction following 능력은 unified VLM보다 약하지만, 양손 motor control에는 충분.

### 2.3 Data-Quality-Aware Training Recipe (핵심 기여)

**3단계**:
1. **Discriminator pretraining**: clip(예: 16-frame 윈도우) 단위로 trajectory quality score $q \in [0, 1]$ 예측. Supervision은 human-labeled high/low quality 페어로부터.
2. **Discriminator scoring**: 전체 dataset의 모든 clip에 대해 $q$ inference.
3. **Weighted diffusion training**: clip의 quality score를 학습 loss에 곱함:

$$
\mathcal{L}_\text{weighted} = \mathbb{E}_{(x_0, c, q) \sim \mathcal{D}}\; q \cdot \| \epsilon_\theta(x_t, t, c) - \epsilon \|^2
$$

여기서 $c$는 condition (vision + language), $q$는 clip-level weight. 결과적으로 *저품질 clip의 gradient 기여가 down-weighted*되고 부드러운 trajectory가 학습됨.

> ❓ **예상 질문**: Filtering(저품질 제거)과 weighting(부드러운 가중치)의 차이는?
> **답변**: Hard filtering은 정보 손실(특히 demo가 적은 dexterous task에서 치명적). Weighting은 모든 demo의 정보를 보존하되 기여도를 조정 → low-quality에서도 *recovery motion*같은 유용 정보를 학습 가능. 논문은 explicit 비교 ablation 부재이나 정성적 정당화.

### 2.4 Cross-Embodiment Projection

학습 후 정책을 다른 embodiment(single-arm gripper, dual-arm gripper, single-arm hand)에 *action projection*만으로 전이:
- 36-dim joint command → target embodiment의 active joint subset으로 선형 projection
- 모델 재학습 불필요, 동일 체크포인트 재사용

---

## 3. 데이터셋

### 3.1 Composition

| 소스 | Trajectories | Frames | Hours |
|------|------------|--------|-------|
| Simulation | 100,000 | 6.5M | 361h |
| Real teleop | 10,000 | 2.92M | 40.5h |
| **High-quality real subset** | ~1,500 (15%) | ~440K | ~6h |

전체 약 **9.4M frames / 400h**. High-quality fraction이 15%라는 점이 quality weighting 정당화.

### 3.2 Task 다양성
- **Basic 12 tasks**: pick-and-place (apple→plate 100%), assemble/disassemble, articulated objects (laptop opening 90%)
- **Dexterous 6 tasks**: use pen, fetch book, cut leek, place plates, rough dough, twist cap

---

## 4. 실험 결과

### 4.1 Basic Tasks (12 tasks, Table I)

| Method | Avg Success |
|--------|------------|
| Diffusion Policy | 34.2 |
| π0 | 50.4 |
| GR00T-N1 | 82.1 |
| **Dexora** | **89.6** |

Dexora는 12개 중 **7개에서 ≥90%** 달성.

### 4.2 Dexterous Tasks (6 tasks, Table II)

| Task | Diffusion Policy | π0 | GR00T-N1 | **Dexora** |
|------|----------------|-----|---------|----------|
| Use pen | – | – | – | 65 |
| Fetch book | – | – | – | 80 |
| Cut leek | – | – | – | 80 |
| Place plates | – | – | – | 70 |
| Rough dough | – | – | – | 80 |
| Twist cap | – | – | – | 25 |
| **Average** | **6.7** | **26.7** | **51.7** | **66.7** |

> ❓ **예상 질문**: Twist cap이 25%로 유독 낮은 이유는?
> **답변**: 논문은 "tactile feedback 부재와 비교적 *low-friction rigid fingertip pads*" 때문이라고 명시. Twist cap은 grip stability + 회전 토크 + slip detection이 필요한데, vision-only로는 slip 발생을 감지하기 어렵다.

### 4.3 OOD 일반화
- 6개 조건(unseen background, lighting, objects, occlusion, clutter, height variation)에서 견고한 성능 유지
- 정확 수치는 paper Table III/부록.

---

## 5. 어블레이션

### 5.1 Discriminator 효과 (Table III)

| Task | w/o discriminator | **w/ discriminator** |
|------|------------------|--------------------|
| Corn → plate | 85% | **95%** |
| Lift basket (bimanual) | 55% | **80%** |

**Trajectory smoothness**:
- 가속도 메트릭: **-41%** (낮을수록 좋음)
- Jerk 메트릭: **-26%**

→ Quality weighting이 *성공률* 뿐 아니라 *제어 품질*도 개선.

### 5.2 Sim vs Real Data Mixing

| Setting | Use Pen Success |
|---------|---------------|
| Sim only | 0 |
| Sim + 50% real | 35 |
| Sim + 100% real | 65 |

→ Dexterous skill은 **real data가 절대적**. Sim-only는 0%(zero generalization). 50% real만으로도 35% → 80% sim 데이터의 *증류 효과*는 제한적이고, real demo가 필수.

> ❓ **예상 질문**: 100K sim trajectory가 0% use pen이라면 sim 데이터의 가치는 무엇인가?
> **답변**: Sim 데이터는 *basic task*(pick-place)에서 sample efficiency 향상에 기여. Dexterous task에서는 contact dynamics, finger compliance가 sim2real gap이 커서 zero. 즉 sim 데이터는 *coarse manipulation prior*로 기능하고 dexterity는 real이 담당.

---

## 6. 한계 및 미해결 문제

1. **Tactile sensing 부재**: twist cap 25%가 단적 — vision-only는 slip / force-sensitive task의 한계가 명확
2. **Discriminator label cost**: clip-level quality label을 어떻게 수집했는지(human label 양, 비용)가 부록 수준에서만 처리됨
3. **Hardware 의존성**: AIRBOT + XHAND + custom exoskeleton + Vision Pro는 재현 비용이 높음. 오픈소스라 해도 *접근성*은 제한적
4. **모델 크기 미언급**: policy의 정확한 파라미터 수(layers × hidden × heads로부터 ~350M 추정되나 명시 X)
5. **Cross-embodiment transfer 수치 부재**: "성공한다"는 정성 서술 + 실험 영상은 있으나 success rate 표가 약함
6. **Ablation 부분적**: discriminator가 두 task에서만 명시적으로 비교(corn→plate, lift basket). 전체 18 task에서의 contribution map은 부재

### Attribution
- 성능 향상이 (a) data quality weighting, (b) 36-DoF 양손 데이터 자체의 풍부함, (c) diffusion policy backbone, (d) sim+real 큰 데이터 중 어느 것인지 부분 분리됨 — discriminator on/off는 명확하나 *데이터 양 ablation*은 약함

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 36-DoF 오픈소스 VLA + clip-level quality weighting의 결합 |
| **Technical depth** | ★★★★☆ — discriminator-weighted diffusion이 깔끔한 formulation |
| **Experimental rigor** | ★★★★☆ — 12 basic + 6 dexterous + ablation + OOD, baselines 비교 충실 |
| **Practical impact** | ★★★★★ — *오픈소스*로 36-DoF dexterous VLA 공개는 커뮤니티에 큰 가치 |
| **Writing quality** | ★★★★☆ — 시스템·데이터·방법론 분리 명확 |

**강점**: dexterous manipulation을 향한 *오픈소스 system+policy 풀스택*. Quality-aware training은 teleoperation data의 본질적 noise 문제에 대한 실용적 솔루션. **약점**: tactile 부재, discriminator label cost, 일부 ablation의 task coverage 한정.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Clip-level quality label은 어떻게 수집했는가? 비용·시간은? | 본문은 human labeling으로 high/low quality pair 수집. 정확한 수치(label 양, $/label)는 미공개. Scale-up시 가장 큰 bottleneck |
| 2 | Discriminator 자체가 high-quality data만으로 over-fit 가능성? | 30M discriminator는 충분히 작아서 over-fit risk 낮음. 다만 *bias*(특정 motion style을 high quality로 oversample)는 가능. Cross-validation 결과 부재 |
| 3 | 36-DoF action space에서 diffusion sampling의 mode collapse 가능성? | Multi-mode action(좌손 잡고 우손 잡는 두 방식)이 dexterous에 있을 때 diffusion이 한 모드만 학습할 수 있음. 본 논문은 정량 분석 부재. Action chunking으로 부분 완화 |
| 4 | Cross-embodiment projection은 정말 zero-shot인가? | 동일 dynamics(diffusion noise schedule)는 share되지만 action space의 *분포 shift*가 있을 수 있음. 본 논문은 projection 후 success를 정성적으로 보고 |
| 5 | GR00T-N1(82.1% basic, 51.7% dex)과의 격차의 *근본 원인*은? | (a) Dexora는 36-DoF native, GR00T-N1은 6-7 DoF 가정에서 변환, (b) Dexora의 real teleop이 GR00T-N1 사전학습 분포와 다르고 더 풍부함, (c) discriminator weighting의 +5~+15 contribution. 세 요인 분리 ablation은 부분적 |
| 6 | 100K sim trajectory가 use pen에 0% 기여 — sim 데이터를 그렇게 많이 모은 가치는? | Basic task의 sample efficiency, instruction-following 사전학습, MuJoCo digital twin 자체 가치. Dexterous skill에 직접 transfer 안 되는 것이 limitation |
| 7 | T5 + SigLIP은 modern unified VLM(Qwen3-VL 등)에 비해 vision-language alignment가 약하지 않은가? | 맞음. Trade-off: 작은 footprint, fine-tune 용이. Language의 *복잡 instruction following*은 약하지만 dexterous task의 instruction은 단순 — practical justification |
| 8 | Tactile/force sensor 추가가 twist cap 같은 task를 얼마나 향상시킬까? | 본 논문은 future work으로 명시. Sensorized fingertip(예: GelSight)을 같은 framework에 추가하면 contact-rich dexterous에서 50%+ 향상 가능성. 다만 hardware/data cost가 다시 증가 |

<!-- VERIFIED: pdf -->
