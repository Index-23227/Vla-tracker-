# OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation

> **한 줄 요약**: Chameleon-7B 위에 per-slot vector를 frozen 32-dim 식별 주소(address)와 256-dim 시간변화 콘텐츠(content)로 분리하고, 모든 transformer layer의 cross-slot attention key를 address 부분에만 의존하도록 마스킹·재주입함으로써, "어떤 물체를 조작할지"의 결정을 "그 물체가 지금 어떻게 보이는지"와 아키텍처 수준에서 분리한 flow-matching VLA. LIBERO 97.8%로 in-distribution SOTA에 근접하면서 LIBERO-Plus의 기하학적 OOD축(camera/robot/layout)에서 π0.5 대비 +4.8% 강건성을 입증.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 한계

- OpenVLA, π0, π0.5 등 주류 VLA는 **holistic token mixing**: 모든 시각·언어 토큰이 self-attention에서 자유롭게 섞임 → 모델이 "물체를 가리키는 신호"와 "그 물체의 현재 외형/배치"를 자체적으로 disentangle해야 함
- 결과적으로 카메라 시점/배치/조명이 바뀌면 **잘못된 물체에 grounding이 슬며시 옮겨가는 binding failure**가 발생 (LIBERO-Plus camera 축에서 baseline들이 60~75%로 급락)
- World-action 계열(F1-VLA, MemoryVLA, VLA-JEPA 등)이 미래 예측을 보조 신호로 쓰지만, **binding 그 자체에 대한 구조적 강제**는 없음

### 핵심 질문

- **"어떤 물체를 잡을지"의 결정을 "그 물체가 지금 어떻게 보이는지"와 분리할 수 있는가?**
- 그 분리를 **post-hoc regularization이 아니라 attention 연산 자체에 내장**할 수 있는가?
- 그 결과가 in-distribution 성능을 해치지 않고 OOD 견고성을 어디까지 끌어올리는가?

📌 [Figure 1 삽입] — OA-WAM 아키텍처: per-slot address/content 분해 + 모든 layer에서의 key-mask + reset hook

---

## 2. 방법론 심층 분석

### 2.1 Slot 분해

각 frame은 robot 1개 + object N개(최대 N=16)로 분해되며, 각 slot의 vector는 4개 부분의 concat:

| 구성요소 | 차원 | 출처 | 시간 변화 |
|---------|------|------|----------|
| addr (identity address) | 32 | 언어 라벨 + 초기 DINOv3 feature, episode 시작 시 1회 계산 | **frozen** |
| cnt (content) | 256 | 매 frame SAM 3 mask + DINOv3 feature | 변화 |
| π (frame index) | 16 | sinusoidal | 변화 |
| ρ (role) | 16 | lookup (robot/object/padding) | frozen |

총 320-dim slot → 학습 가능한 adapter로 4096-dim 트렁크 차원에 projection되어 Chameleon-7B에 입력.

### 2.2 Object-Addressable Attention (핵심)

OA-WAM의 정체성은 **parameter-free 두 가지 연산**에 있음:

1. **Key-projection masking**: cross-slot attention의 K 입력에서 32-dim address 외 모든 좌표를 0으로 마스킹. 즉 "어느 slot이 어느 slot을 보는가"의 결정은 오직 frozen address로만.
2. **Per-layer reset hook**: residual 누적 과정에서 layer마다 slot vector의 address slice를 캐시된 원본으로 덮어쓰기. 깊어질수록 content와 섞이는 drift를 차단.

> ❓ **예상 질문**: 왜 key만 마스킹하고 query·value는 안 하는가?
> **답변**: Query는 "내가 무엇을 찾는지"를 표현 — content와 task context가 자유롭게 들어가야 함. Value는 정보 전달자 — content가 흘러야 행동이 결정됨. 오직 key("나는 누구인가")만 frozen identity로 묶으면, "잘못된 외형의 슬롯이 자신을 표적으로 가장하는" binding spoofing이 차단됨.

> ❓ **예상 질문**: 32 dim address가 16개 object를 구별하기에 충분한가?
> **답변**: 32-dim continuous space에서 무작위로 뽑힌 16개 unit vector의 평균 cosine 유사도는 ~0.18 — 일반적으로 충분히 분리 가능. 다만 cluttered scene(동종 물체 다수)에서는 이론적 bottleneck.

### 2.3 World Action Head

Action head 외에 **per-slot 다음 frame state(content + pose)를 예측하는 보조 head**가 있음. 이것이 "World"의 의미 — 명시적 video generation은 아니지만 slot-level latent forecasting을 함으로써 trunk가 dynamics를 학습하도록 압박.

**Action**: 16-step continuous 7-DoF chunk, flow matching MLP가 velocity field $v_\xi(A_t^\tau, \tau, H_q^{act})$를 회귀.

손실:

$$\mathcal{L}_{act} = \mathbb{E}_{\tau,\epsilon}\Big\|v_\xi(A_t^\tau, \tau, H_q^{act}) - (A_t - \epsilon)\Big\|_2^2, \quad \tau\sim\mathcal{U}(0,1), \epsilon\sim\mathcal{N}(0,I)$$

추론은 4-step forward Euler. Trunk+head ~5.6ms (perception 95ms 별도).

> ❓ **예상 질문**: 왜 diffusion이 아니라 flow matching인가?
> **답변**: 4-step Euler로 충분히 수렴하므로 sampling cost가 낮음. 또한 velocity field formulation이 chunk 전체에 대해 한 forward pass로 학습/추론되어 chunk-wise consistency가 좋음. π0와 동일한 선택.

---

## 3. 학습 레시피

| 단계 | Steps | 데이터 | 학습 대상 |
|------|-------|--------|-----------|
| Stage 0 | ~600k | DROID + RoboCasa + OXE | slot-aware trunk 사전학습 |
| Stage I | 50k | LIBERO | slot-adapter alignment만 |
| Stage II | 100k | LIBERO | 전체 시스템 LoRA fine-tuning (80M LoRA + 47M heads = ~127M trainable) |

복합 손실:

$$\mathcal{L}(\theta) = \mathcal{L}_{act} + 0.5\,\mathcal{L}_{world} + 0.04\,\mathcal{L}_{vq} + 0.1\,\mathcal{L}_{compose} + 0.05\,\mathcal{L}_{role}$$

> ❓ **예상 질문**: 7B 모델인데 trainable이 127M (1.8%)밖에 안 되면 LIBERO 97.8%는 사실상 LoRA 효과인가?
> **답변**: 부분적으로 그렇다. 다만 Stage 0의 slot-aware trunk pretraining(전체 trunk를 slot 분해된 input으로 학습)이 LoRA만으로 얻기 어려운 representation을 만든다. Ablation에서 OA constraint를 끄면(V2) LIBERO 95.4%로 떨어지므로 정확히 분리 측정 가능.

---

## 4. 실험 결과

### 4.1 In-distribution: LIBERO (Table 1)

| Suite | OA-WAM |
|-------|--------|
| Spatial | 98.9% |
| Object | 99.0% |
| Goal | 97.4% |
| Long | 95.9% |
| **Avg** | **97.8%** |

SimplerEnv WidowX visual matching: 79.3% avg.

### 4.2 LIBERO-Plus 7축 강건성 (Table 2) — 논문의 핵심

| 축 | OA-WAM | 최고 baseline | 차이 |
|----|--------|--------------|------|
| Camera | **80.5%** | π0.5: 75.4% | **+5.1** |
| Robot Init | 89.6% | X-VLA: 89.7% | −0.1 |
| Layout | 82.8% | π0.5: 85.7% | −2.9 |
| **Geo Avg (Cam/Robot/Layout)** | **84.3%** | π0.5: 79.5% | **+4.8** |
| Lighting | 96.5% | π0.5: 96.9% | −0.4 |
| Background | 95.9% | π0.5: 94.6% | +1.3 |
| Language | 85.3% | π0.5: 85.6% | −0.3 |
| Sensor Noise | 75.6% | Cosmos-Policy: 92.7% | **−17.1** |
| 7-axis Avg | 83.9% | π0.5: 85.7% | −1.8 |

핵심 관찰:
- **기하학적 축(Camera/Robot/Layout)**에서 명확한 우위 — object-addressing의 직접적 효과
- **Sensor Noise**에서 압도적 열위 — frozen address가 깨끗한 perception을 전제로 함. SAM 3/DINOv3 mask가 noisy하면 addr 자체가 오염
- **Lighting/Background**은 거의 평탄 — content stream이 이미 처리하던 모달리티
- **Language**는 약간 열위 — 언어 paraphrase로 address가 바뀌면 binding이 흔들림

### 4.3 Ablation (Table 3) — OA constraint만 격리

| Variant | Key Mask | Reset Hook | LIBERO | LP Camera | Swap Binding |
|---------|----------|-----------|--------|-----------|--------------|
| V2 (no OA) | off | off | 95.4% | 60.5% | 0.06 |
| V1 (mask off) | off | on | 96.3% | 67.2% | 0.19 |
| **V0 (full)** | **on** | **on** | **97.8%** | **80.5%** | **0.87** |

**가장 강력한 주장의 근거**: key mask 하나만 끄면 LP Camera가 13.3% 떨어지는데 LIBERO는 1.5%만 떨어짐. 즉 key mask는 **OOD-specific 효과**이지 generic capacity 증가가 아님.

### 4.4 Causal Swap Binding Test (Table 4) — 가장 인상적

언어가 가리키는 target slot의 address를 다른 object의 address로 교체했을 때 trajectory가 그 새 target으로 옮겨가는 정도(cosine alignment):

| 모델 | Swap Binding |
|------|-------------|
| OpenVLA | ≤0.09 |
| π0, π0.5 | ≤0.09 |
| OpenVLA-OFT | ≤0.09 |
| WorldVLA, VLA-JEPA, ThinkAct, Cosmos-Policy, GE-Act | ≤0.09 |
| **OA-WAM** | **0.87** |

> ❓ **예상 질문**: 0.87 vs ≤0.09의 격차가 너무 큰데 측정 정의가 baseline에 불리하지 않은가?
> **답변**: 매우 타당한 의심. Baseline들은 명시적 slot vocabulary가 없으므로 "address를 swap한다"는 개입 자체가 적용 불가 — 대체 정의(예: 언어에서 object 이름만 바꿔서 prompt 재입력)로 측정한 것으로 보임. 직접 비교라기보다 **OA-WAM이 가진 새로운 인과적 핸들의 존재 자체**를 시연하는 실험.

---

## 5. 어떤 baseline과 비교했는가

LIBERO/SimplerEnv 비교: OpenVLA, SpatialVLA, π0, π0.5, InternVLA-M1, CogACT, F1-VLA, MemoryVLA, VLA-JEPA, CoWVLA, ThinkAct, VITA.

LIBERO-Plus 추가: OpenVLA-OFT, X-VLA, AVA-VLA, WorldVLA, GE-Act, HoloBrain-0, Cosmos-Policy.

> ❓ **예상 질문**: 왜 PerAct, RVT 같은 3D manipulation 전문 모델은 없는가?
> **답변**: 그쪽은 voxel/point-cloud action grounding 계열로 입력 modality가 다름. OA-WAM은 2D RGB를 받는 VLA 계열 내부에서의 비교에 집중.

---

## 6. 한계 및 미해결 문제

### 방법론적

1. **Sensor Noise 축 −17.1%**는 가장 심각한 약점. Frozen address가 깨끗한 perception을 전제로 하므로 noise/occlusion 환경에서 binding 신호가 오염. SAM 3 mask 신뢰도가 낮은 영역에서 address가 식별 기능을 잃음.
2. **Layout −2.9%, Robot Init −0.1%**: "기하학적 강건성"이라는 핵심 주장 안에서도 Layout과 Robot은 미세하게 열위 — Camera만 큰 폭으로 우위. 즉 OA의 효과는 "카메라 시점 변화에 대한 robustness"가 가장 본질이며 layout 변화에는 제한적.
3. **Real-world 실험 없음**: 논문이 명시적으로 "Validation is simulator-only; reported robustness does not yet prove real-robot deployment." sim2real에서 SAM 3/DINOv3 perception의 안정성이 보장되지 않음.
4. **Perception 95ms 병목**: trunk가 5.6ms로 빠르지만 SAM 3 + DINOv3가 95ms — closed-loop 30Hz 이상 제어가 불가능.
5. **Stage 0 비용 미보고**: 600k step의 전체 trunk pretraining 비용이 명확하지 않음. LoRA만으로 보이지만 사실상 7B 모델의 representation을 새로 빚는 비용.

### Attribution 문제

- LIBERO 97.8%의 어느 정도가 Chameleon-7B의 backbone 능력에서 오고, 어느 정도가 Stage 0 slot-aware trunk pretraining에서 오고, 어느 정도가 OA constraint에서 오는지 완전 분리는 어려움. Table 3 ablation으로 OA 자체의 기여(95.4→97.8 = +2.4)는 측정되지만 backbone vs Stage 0의 분리는 없음.
- Swap binding 0.87은 OA의 binding 메커니즘이 "작동함"의 증거이지 "필요함"의 증거는 아님 — baseline들도 cluttered scene에서 잘못된 객체에 안 가는 한 swap binding 점수가 낮아도 무방.

---

## 7. 관련 연구 비교

| 모델 | Binding 메커니즘 | LIBERO Avg | LP Camera | Swap Binding |
|------|----------------|-----------|-----------|--------------|
| OpenVLA | implicit attention | ~75% | <60% | ≤0.09 |
| π0.5 | implicit attention | ~96% | 75.4% | ≤0.09 |
| WorldVLA | implicit + world model | ~95% | <70% | ≤0.09 |
| VLA-JEPA | latent forecasting | ~94% | <70% | ≤0.09 |
| **OA-WAM** | **explicit slot address (frozen K)** | **97.8%** | **80.5%** | **0.87** |

### 핵심 차이

OA-WAM은 binding을 "더 좋은 표현으로 잘 학습되길 기대"가 아니라 **attention 연산 자체에 내장된 invariant**로 만듦. 이는 slot-attention 계열(IODINE, Slot Attention)의 정신을 VLA에 도입한 첫 시도라 평가 가능.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — VLA에서의 attention-level identity binding은 새로운 영토 |
| **Technical depth** | ★★★★☆ — key mask + reset hook이 단순하지만 분명한 ablation을 가짐 |
| **Experimental rigor** | ★★★★☆ — LIBERO-Plus 7축 + swap binding 인과 실험은 모범적이나 real-world 부재 |
| **Practical impact** | ★★★☆☆ — Sensor noise 약점과 95ms perception 병목이 실배포 가로막음 |
| **Writing quality** | ★★★★☆ — Ablation 설계가 깔끔하고 주장이 측정 가능 |

**강점**: VLA 강건성 연구에서 "binding"이라는 추상 개념을 attention-level operation으로 환원한 것 자체가 큰 기여. Swap binding test로 모델이 무엇을 학습했는지를 인과적으로 검증한 점이 인상적. **약점**: Sensor noise와 real-world 부재가 결합되면 sim-to-real 갭이 클 가능성이 큼. 핵심 메커니즘인 frozen address는 깨끗한 perception을 전제하는데 그 전제 자체가 실세계에서 자주 깨짐.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO 97.8%인데 기존 OpenVLA-OFT는 이미 97%대. OA-WAM이 기여한 것은 정확히 무엇? | LIBERO 자체에선 +0.X% — 거의 saturated. 진짜 기여는 LIBERO-Plus Camera에서 +5.1%, Geo Avg에서 +4.8%. 즉 in-distribution이 아니라 OOD에서의 차별성. |
| 2 | Frozen address가 perception 오류에 깨진다면 SAM 3 신뢰도가 낮은 실세계에서 어떻게 되나? | 정확히 sensor noise −17.1%가 그 신호. Cosmos-Policy(92.7%)는 noise robust 학습을 했기에 우위. OA-WAM은 clean perception 전제. 실세계 배포 위해선 noisy address에 대한 robust mechanism 필요. |
| 3 | Swap binding 0.87 vs ≤0.09 — 비교 정의가 baseline에 불리한 것 아닌가? | 가능성 있음. Baseline은 명시적 slot vocabulary가 없으므로 "address swap" 자체가 동치 개입이 아님. 이 실험은 "OA-WAM이 인과적 binding handle을 갖는다"의 시연이지 정량 비교라기보다는. |
| 4 | Stage 0 600k steps의 비용 미보고는 큰 누락 아닌가? | 그렇다. 사실상 7B 모델의 representation을 새로 만드는 비용인데 GPU·시간 표기가 없음. "LoRA로 fine-tune했다"가 misleading하게 들릴 수 있음. |
| 5 | Sensor Noise 75.6%로 Cosmos-Policy 92.7% 대비 17% 뒤짐. 핵심 OA 주장과 모순 아닌가? | 모순이라기보다 trade-off. OA는 기하학적 binding 강건성을 얻는 대신 perception quality에 강하게 의존. Two-stage 융합(noisy 환경에서 OA constraint relaxation) 등 후속 연구 여지. |
| 6 | 7B backbone 위에 127M LoRA로 97.8%인데 이게 backbone 능력의 평탄화 효과인가? | 부분적으로 그렇다. 다만 V2 (no OA, same backbone) 95.4% → V0 97.8% (+2.4%)로 OA constraint 자체의 기여가 분리 측정됨. |
| 7 | 16개 object 한계를 어떻게 푸는가? | 논문 명시 한계. N=16 hard cap이므로 cluttered scene이나 멀티-에이전트 환경엔 직접 적용 불가. Slot 동적 할당(slot attention의 random init) 후속 작업 필요. |
| 8 | Closed-loop 95ms × Action chunk 16 step이면 실제 제어 주파수는? | Chunk 16개를 한번에 받으므로 chunk 단위로는 perception cost가 분담됨. 그러나 새 perception이 95ms 걸리니 한 chunk를 다 쓰기 전 새 관측이 들어오지 않음 → re-planning latency 증가. Reactive 조작에 불리. |
| 9 | Real-world 부재는 critical limitation 아닌가? | Yes. 논문 자체가 인정. LIBERO-Plus가 sim 내 OOD라 sim-to-real gap을 substitute하지 못함. 다음 단계 연구로 가장 중요한 axis. |
| 10 | "World Action Model"인데 명시적 video generation 없는 게 misleading한 naming 아닌가? | 일리 있음. World 부분은 per-slot latent state forecasting (auxiliary loss)에 한정. Video/point cloud generation을 기대하는 독자에겐 misnomer. F1-VLA, WorldVLA 등이 갖는 video forecasting과는 다름. |

<!-- VERIFIED: pdf -->
