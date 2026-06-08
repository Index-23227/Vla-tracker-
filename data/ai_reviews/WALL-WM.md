# WALL-WM: Carving World Action Modeling at the Event Joints

> **한 줄 요약**: Wan2.2-5B video tower를 embodied 데이터로 재학습시킨 14B 규모의 event-centric world action model. Action DiT + Staircase Decoding으로 가변 길이 action chunk를 생성하며, real-robot Diverse Manipulation 75.86 / Reasoning Manipulation 71.60 Task Progress로 π0.5(55.64/56.40) 대비 20점대 우위.

---

## 1. 배경 및 동기

### 기존 연구의 한계
- **순수 video generation 모델** (Wan2.1, Wan2.2, CogVideoX, Aether)은 perceptual fidelity는 좋지만 **embodied 환경에서 contact/dynamics/instruction-following이 약함**
- **기존 VLA** (π0.5, LingBot-VA, DreamZero)는 fixed-length 액션 청크만 출력하며 event 경계(grasp/transfer/release)에 대한 명시적 구조가 없음
- 두 흐름이 분리되어 있어 **world model을 활용한 행동 생성**이 어색했음

### 핵심 질문
- **대규모 video prior(Wan)를 embodied physical prior로 변환할 수 있는가?**
- **Event 경계를 inference 시 명시적으로 활용하면 long-horizon manipulation이 안정화되는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

WALL-WM은 두 모듈의 layer-coupled 결합:

1. **Video tower** (Wan2.2-5B, 3D Causal VAE 기반) — 다중 뷰 영상 → 잠재 표현
2. **Action DiT** — 무작위 초기화, video-DiT 각 레이어의 features에 cross-attention
3. **Reasoning module** (event mode 전용): fine-tuned Qwen3.5-VL-9B — task instruction + multi-view observation → next-event description

### 2.2 Staircase Decoding

핵심 아이디어: transformer를 **relay depth N_r**에서 분할
- N_r 하층: 공유 visual-language grounding features 인코딩
- N_r 상층: 점진적으로 다른 reasoning step으로 specialize
- 결과: **연속 잠재 reasoning sequence의 parallel generation**

### 2.3 Wan-style Flow Matching

- v-prediction objective
- Variable-length action chunks (event mode) / fixed-length (unified mode)
- Action transformer 구성요소: self-attention over action tokens + cross-attention to state token + cross-attention to matched video-DiT layer features + gated FFN

### 2.4 두 가지 Inference 모드

| 모드 | 동작 |
|------|------|
| **Event mode** | Qwen3.5-VL-9B가 task → next-event 설명 변환 → 이를 prefix로 받아 event-conditioned 행동 생성 |
| **Unified mode** | 명시적 event 추론 없이 직접 fixed-length chunk 디코딩 (Staircase Decoding 활용) |

---

## 3. 데이터 전략

- **Generalized embodied data mixture**: 다양한 verb, object 카테고리, scene layout, camera 구성, robot embodiment 포괄
- **Embodied Video Generation benchmark** (자체 구축): held-out 200 in-distribution + 50 OOD 태스크
- OOD split는 novel object-verb pairings, paraphrased instructions, unseen scene arrangements로 text/compositional 일반화 검증
- 학습 옵티마이저: **Muon optimizer**

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Video tower | Wan2.2-5B (Wan Series text-to-video) |
| Reasoning LLM | Qwen3.5-VL-9B (event mode) |
| VAE | 3D Causal VAE |
| Flow matching | Wan-style, v-prediction |
| Optimizer | Muon |
| Action 청크 | Variable (event) / Fixed (unified) |

---

## 5. 실험 결과 (PDF Table 2, 3, 5; Figure 16, 17 직접 확인)

### 5.1 Embodied Video Generation (Table 2)

| Metric (선택) | Wan2.1-1.3B | Wan2.2-5B | **WALL-WM** |
|---|---|---|---|
| Motion Smoothness | 0.619 | 0.683 | **0.771** |
| Subject Consistency | 0.476 | 0.769 | **0.795** |
| Semantic — Background | 0.522 | 0.817 | **0.838** |
| Interaction Quality | 0.219 | 0.226 | **0.434** |
| Physical — Instruction Following | 0.308 | 0.298 | **0.391** |

### 5.2 3D Awareness on CO3Dv2 (Table 3)

| Model | Point Err↓ | Depth Err↓ | AUC@5↑ | AUC@30↑ |
|---|---|---|---|---|
| DINOv2 | 0.559 | 0.209 | 0.051 | 0.508 |
| V-JEPA | 0.439 | 0.214 | 0.076 | 0.619 |
| WAN2.1-14B | 0.284 | 0.151 | 0.200 | 0.736 |
| **WALL-WM** | **0.271** | **0.132** | **0.210** | 0.727 |

### 5.3 Real-Robot Task Progress (Figure 16, 17)

| Suite | LingBot-VA | DreamZero | π0.5 | WALL-WM-U-Scratch | **WALL-WM (event)** |
|---|---|---|---|---|---|
| **Diverse Manipulation** (avg) | 29.71 | 39.97 | 55.64 | 63.00 | **75.86** |
| **Reasoning Manipulation** (avg) | 31.60 | 32.70 | 56.40 | 59.50 | **71.60** |

- Diverse 5 tasks: Arrange Cup Inverted Triangle, Put Spoon to Bowl, Put Glasses on Woodshelf, Put Ring onto Rod, Put Blocks to Color, Pour Water from Bottle, Pick Items into Basket
- Reasoning 5 tasks: Sort Headphone, Classify Items as Shape, Press Button In Order, Pair Up Items, Pick Fruits into Basket
- Event mode가 U-Scratch 대비 +12.86 (Diverse), +12.10 (Reasoning) → event-centric pretraining의 기여 입증

### 5.4 Dexterous & Generalization (정량 수치는 일부만 도표)

- **Dexterous**: Put Stationery in Case, Insert Wireline, Unbox fan — event mode가 평균 최고 (정확한 수치는 Table 5)
- **Generalization** (7.2.4): cluttered scenes에서도 event mode 우위

---

## 6. Ablation 분석

| Ablation | 변경 | 효과 |
|---|---|---|
| WALL-WM-U-Scratch | event-centric pretraining 제거 | Diverse 75.86 → 63.00 (-12.86) |
| Wan2.2-5B prior 제거 | random video tower | Table 2 전반 큰 폭 하락 (Wan2.2-5B baseline 대비) |
| 7.2.5 Event & View Modeling | (paper 본문 참조) | event-conditioned reasoning + 다중 뷰 cross-attention이 robustness 기여 |

---

## 7. Related Work 비교

| Method | 백본 | 모드 | 평가 메트릭 |
|---|---|---|---|
| π0.5 | (separate) | fixed-chunk | Task Progress 0-100 |
| DreamZero | (separate) | fixed-chunk | Task Progress 0-100 |
| LingBot-VA | (separate) | fixed-chunk | Task Progress 0-100 |
| **WALL-WM** | **Wan2.2-5B + Qwen3.5-VL-9B** | **variable (event) + fixed (unified)** | **Task Progress 0-100** |

---

## 8. Limitations

1. **표준 sim 벤치마크 미보고**: LIBERO/CALVIN/SimplerEnv 수치 없음 → 다른 VLA와 직접 비교 어려움
2. **Task Progress 메트릭 의존**: 0-100 dense score는 partial credit 포함이라 binary success보다 후함. 직접 비교 시 주의
3. **추론 비용**: 14B 규모 + event-mode는 Qwen3.5-VL-9B 추가 호출 필요 → latency 부담
4. **데이터 비공개**: "internally developed deployment platform suite" → 정확한 데이터 분포/규모 비공개

---

## 9. 종합 평가

| 항목 | 평점 (5점) | 비고 |
|---|---|---|
| **혁신성** | 4.5 | Video prior → embodied prior 변환 + event 경계 명시화 |
| **재현성** | 3.0 | 코드 공개되었으나 학습 데이터/하드웨어 미공개 |
| **실험 폭** | 4.0 | 4개 real-robot suite + video gen + 3D awareness 평가 |
| **이론적 깊이** | 3.5 | Staircase Decoding 아이디어는 깔끔, flow matching 기존 기법 활용 |
| **실용성** | 3.5 | 코드 공개, 다만 표준 sim 벤치마크 결과 부재로 다른 VLA와의 직접 비교 어려움 |

**총평**: Video generation 백본을 embodied 정책으로 전환하는 가능성을 가장 잘 보여준 사례 중 하나. 실 로봇 Task Progress에서 π0.5 대비 +20점대 우위는 상당한 결과. 다만 표준 sim 벤치마크 부재가 평가의 한계.

---

## 10. 예상 세미나 질문

> ❓ **Wan 백본이 embodied 데이터로 재학습되면 원래의 video gen 능력은 손상되는가?**
> Table 2에서 Visual Quality (Image Aesthetic 0.503 vs Wan2.2 0.527, Dynamic 0.393 vs 0.409)는 약간 떨어졌으나, Motion / Semantic / Physical 차원에서는 크게 향상. Trade-off는 의도된 것 — embodied prior로 특화시킨 결과.

> ❓ **Task Progress 메트릭이 너무 후하게 점수를 주는 것 아닌가?**
> Table 6에 task-specific rubric이 명시되어 partial credit 기준이 공개됨. 모든 baseline에 동일 적용. 다만 binary success로 환산하면 차이가 더 줄어들 가능성 있음.

> ❓ **Event mode가 항상 우위인가? Latency 비용은?**
> Qwen3.5-VL-9B 호출이 추가되므로 unified mode 대비 latency 비용. Reasoning Manipulation 같이 instruction grounding이 중요한 태스크에서 event mode 우위가 크고, Diverse Manipulation에서도 +12.86. 단순 reactive 태스크에서는 unified mode가 충분할 수 있음.

> ❓ **`pi0.5`보다 큰 모델 (14B vs ~3B)인데 fair comparison인가?**
> 정확한 지적. 백본 규모 차이가 있다. 다만 WALL-WM-U-Scratch (동일 14B, event 제거)도 63.00에 그쳐서 단순 규모만으로 설명 안 됨 → event-centric pretraining의 기여를 분리해서 보여줌.

---

## 11. 코드 & 재현

- **Code**: https://github.com/X-Square-Robot/wall-x
- **Optimizer**: Muon (구현 공개 여부 확인 필요)
- **데이터**: internal embodied mixture (비공개)
- **체크포인트**: 공개 여부 코드 저장소 확인

---

## 12. 결론

WALL-WM은 (1) video generation 모델의 prior를 embodied policy로 효과적으로 전이하고, (2) event-centric inference로 long-horizon Task Progress를 크게 끌어올렸다는 두 기여를 가진다. 표준 sim 벤치마크 부재라는 한계는 있으나, real-robot 결과는 명확하다.

---

<!-- VERIFIED: pdf -->
