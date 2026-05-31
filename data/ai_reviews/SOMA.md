# SOMA: Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action

> **한 줄 요약**: GR00T N1.5 위에 movable head camera로 수집한 multi-view 관측을 융합한 persistent spatial memory(구축→정제→검색의 3-stage)를 얹어, 조작 대상이 현재 시야 밖으로 사라져도 행동을 생성할 수 있게 만든 VLA augmentation framework. RoboCasa-GR1 평균 52.0%, SimplerEnv visual-matching 63.2%, 실제 5개 out-of-vision 태스크 평균 28.3% 달성.

---

## 1. 배경 및 동기

### 기존 VLA의 시야 제한 문제
- 대부분의 VLA(π₀, OpenVLA, GR00T)는 **현재 프레임만** 정책 입력으로 사용 → 카메라가 흔들리거나 head가 움직이면 직전 관측은 영구 소실
- Bimanual 작업이나 long-horizon 조작에서 **타깃 물체가 frustum 밖으로 나가는 순간** 정책이 기능을 잃음 — "어디에 있었는지"를 기억하지 못함
- 인간은 *작업 메모리*를 통해 시야 밖 물체의 위치를 유지하지만, VLA는 이 능력이 본질적으로 결여

### 핵심 질문
- **시야 밖(out-of-vision) 물체에 대한 조작이 가능한 VLA를 어떻게 만들 것인가?**
- Movable head camera에서 얻은 정보를 어떻게 정책이 활용 가능한 형태로 유지·갱신할 것인가?

📌 [Figure 1] — SOMA pipeline: Spatial Memory Construction → Dynamic Memory Refinement → Contextual Memory Retrieval → GR00T N1.5 action decoder

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

SOMA = **GR00T N1.5(frozen language decoder) + DiT action head + 3-stage spatial memory module**

세 모듈이 카메라 motion을 따라가며 메모리를 유지:

| 단계 | 역할 | 입력/출력 |
|------|------|----------|
| Spatial Memory Construction | Multi-view feature를 3D scene representation으로 통합 | head-camera 시퀀스 → persistent feature volume |
| Dynamic Memory Refinement | Head가 움직일 때마다 메모리 갱신, stale entry 제거 | 현재 frame + 기존 memory → updated memory |
| Contextual Memory Retrieval | 현재 instruction과 관련된 메모리만 정책에 전달 | task token + memory → retrieval된 token set |

### 2.2 왜 frozen language decoder인가?
- VLM의 언어 추론 능력은 유지하고, **시각 표현과 행동 디코더만 fine-tuning**하여 spatial memory가 정책에 통합되도록 함
- 기존 GR00T N1.5의 instruction-following 능력을 그대로 보존

> ❓ **예상 질문**: Memory가 영구히 자라지 않게 하려면?
> **답변**: Dynamic Memory Refinement가 head pose 기반으로 중복/노후 항목을 정리. 그러나 논문에 정확한 capacity bound나 forgetting policy 수치는 부재.

> ❓ **예상 질문**: 왜 단순히 multi-frame stacking을 쓰지 않는가?
> **답변**: Multi-frame은 short context(<10 frames)에만 유효. SOMA의 spatial memory는 작업 전체(수십 초~분)에 걸친 정보를 압축하여 유지하는 것이 차별점.

---

## 3. 데이터 전략

| 데이터 | 규모 | 용도 |
|--------|------|------|
| Real-world VR-teleop demos | 400 demos/task × 5 tasks | Real out-of-vision 평가 |
| RoboCasa Tabletop (Full) | 300 demos per setting | Simulation 평가 |
| Simulation ablation | 50 episodes/ablation | 모듈별 기여 분석 |

> ❓ **예상 질문**: 400 demos/task가 충분한가?
> **답변**: Out-of-vision은 본질적으로 정보 손실이 큰 도메인. 400 demos는 다른 manipulation 연구의 50~200 demos보다 많지만, 5개 태스크 중 하나(dual-arm coordination)에서 16.7%까지 떨어지는 점은 데이터 양의 한계라기보다는 태스크 자체 난이도일 가능성.

---

## 4. 실험 결과 심층 분석

### 4.1 Real-World Out-of-Vision (Figure 4)

| Task | Success |
|------|---------|
| Invisible→Invisible Pick-and-Place | 30.0% |
| Visible→Invisible Pick-and-Place | 35.0% |
| Invisible→Visible Pick-and-Place | 27.5% |
| Sequential Dual-Object Pick-and-Place | 32.5% |
| Dual-Arm Coordination Pick-and-Place | 16.7% |
| **평균** | **28.3%** |

- **28.3% 평균은 절대값으로는 낮지만**, GR00T N1.5 baseline은 대부분의 out-of-vision 시나리오에서 사실상 0% — 시야를 벗어나는 순간 정책이 무력화되기 때문
- Dual-arm coordination(16.7%)이 가장 어려운 이유: 두 팔의 모든 타깃이 동시에 시야 밖일 수 있음

### 4.2 RoboCasa-GR1 (Table 3, 300 demos)

| Category | SOMA |
|----------|------|
| Container Interaction | 53.3% |
| Cooking Preparation | 48.4% |
| Tabletop Serving | 54.0% |
| Dish Transfer | 55.0% |
| Tray Organization | 48.8% |
| **평균** | **52.0%** |

- Qwen-VLA의 RoboCasa-GR1 56.7%에 약간 못 미치나, **out-of-vision augmentation에 특화한 모델이 generalist Qwen-VLA에 근접**한 점은 의미 있음

### 4.3 SimplerEnv (Table 4)

| Setting | Average |
|---------|---------|
| Visual Matching | 63.2% |
| Variant Aggregation | 52.5% |

- Variant aggregation(VA)이 visual matching(VM)보다 10.7%p 낮음 — VA는 분포 외 변형(질감/조명)을 포함하므로 spatial memory가 시각 변형에는 큰 도움이 안 됨을 시사

---

## 5. 한계 및 미해결 문제

### 방법론적 미비점
1. **시각 변형에는 도움 부족**: VM vs VA의 10.7%p 격차는 spatial memory가 *기하* 정보 보존에는 강하지만 *외관* 변화에는 무력하다는 증거
2. **메모리 capacity 미보고**: persistent memory의 크기, GPU 메모리 footprint, 추론 latency 수치 부재
3. **Coding 미공개**: "Code will be released soon" — 재현성 보장 없음
4. **Dual-arm 16.7%**: 가장 도전적인 시나리오에서 여전히 낮은 성능 — coordination을 위한 추가 메커니즘 필요

### Attribution 문제
- 28.3% 향상이 **spatial memory** 덕분인지, **GR00T N1.5 backbone**의 강력함 덕분인지 분리 어려움
- 같은 데이터로 single-frame GR00T N1.5 fine-tune했을 때의 baseline 수치가 명시되지 않음

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA의 out-of-vision 한계를 명시적으로 다룬 첫 연구급 |
| **Technical depth** | ★★★★☆ — 3-stage memory module이 체계적 |
| **Experimental rigor** | ★★★☆☆ — Real 5 tasks + sim 2 benchmarks는 적절하나 baseline 비교 부족 |
| **Practical impact** | ★★★☆☆ — code 미공개, latency 미보고 |
| **Writing quality** | ★★★★☆ — ICML 2026 채택 |

**강점**: "Out-of-vision manipulation"이라는 명확하고 underexplored된 문제를 정의하고, 실제 로봇에서 28.3% 성공률을 보이는 동작 가능 시스템을 제시. **약점**: VA에서의 성능 저하는 spatial memory의 시각 robustness 한계를 노출하며, code 미공개와 latency 부재가 채택의 발목을 잡음.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Memory 크기 bound는? | 미보고. Dynamic refinement가 forgetting을 담당한다고만 명시 |
| 2 | 추론 latency는? | 미보고. GR00T N1.5 자체가 ~10Hz이므로 SOMA는 그 이하로 예상 |
| 3 | Baseline GR00T N1.5는 같은 OOV 태스크에서 몇 %? | 명시 없음 — 사실상 0%로 추정되나 정량 비교 부재 |
| 4 | Spatial memory를 SimplerEnv VA에서 비활성화하면? | Ablation 부재. VM/VA gap이 memory의 시각 invariance 한계를 시사 |
| 5 | Dual-arm coordination 16.7%가 의미하는 바? | OOV + 양팔 동시 = SOMA에도 한계. 향후 두 head 또는 multi-agent memory가 필요할 가능성 |
| 6 | Qwen-VLA 같은 generalist와 비교하면? | 직접 비교 없음. RoboCasa-GR1 52.0% vs Qwen-VLA 56.7% — 작지만 generalist에 미치지 못함 |
| 7 | Movable head camera가 필수 하드웨어 요구사항인가? | 그렇다 — fixed camera 로봇은 적용 불가 |

<!-- VERIFIED: pdf -->
