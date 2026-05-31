# Evo-Depth: A Lightweight Depth-Enhanced Vision-Language-Action Model

> **한 줄 요약**: InternVL3-1B + Qwen2.5-0.5B 기반 0.9B 경량 VLA에 **Implicit Depth Encoding Module (IDEM)**과 **Spatial Enhancement Module (SEM)**을 결합하고 **Progressive Alignment Training**으로 학습하여, LIBERO 95.4%, MetaWorld 84.4%, real-world 90% 평균 성공률을 가장 작은 모델로 달성한 depth-enhanced flow-matching VLA.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대부분의 VLA(OpenVLA, π₀, CogACT, RoboVLM 등)는 **2D 멀티뷰 RGB**만 입력 → 3D 공간 추론 약점
- 명시적 depth 입력(RGB-D)을 받는 모델은 **하드웨어 의존성**(센서 등록·노이즈)이 크고 추가 입력 channel로 inference cost↑
- 7B+ 대형 VLA의 성능은 좋으나 **edge 배포가 어려움** — 1B 미만 경량 모델은 정확도 격차가 큼

### 핵심 질문
- **별도 depth 센서 없이** RGB만으로 3D-aware feature를 뽑아낼 수 있는가?
- 그러한 feature가 **0.9B 경량 모델에서도** 대형 VLA에 견줄 성능을 낼 수 있는가?

📌 [Figure 1 삽입] — Evo-Depth 아키텍처: Multi-view RGB → InternVL3-1B → IDEM(implicit depth) → SEM(depth-aware modulation) → Qwen2.5-0.5B + flow matching head

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요
- **Vision encoder**: InternVL3-1B (frozen 또는 LoRA)
- **Implicit Depth Encoding Module (IDEM)**: 멀티뷰 RGB token으로부터 **명시적 depth 라벨 없이** depth-aware token을 생성
- **Spatial Enhancement Module (SEM)**: IDEM 출력을 vision token에 depth-aware modulation으로 융합
- **LM**: Qwen2.5-0.5B (14 layers, 일부 layer만 사용해 경량화)
- **Action head**: Flow matching DiT (continuous action chunk 예측)

### 2.2 IDEM: Implicit Depth Encoding

핵심 아이디어: 외부 monocular depth estimator(ZoeDepth 등)를 호출하지 않고, **멀티뷰 RGB token 간의 cross-attention**으로 implicit 3D structure를 학습.

```
D_token = CrossAttn( view_i, view_j )  → depth-aware embedding
```

**Pretraining stage**에서는 외부 depth(예: dataset에 포함된 RGB-D)를 supervised target으로 사용해 IDEM이 implicit depth를 "예측"하도록 학습하고, **deployment 시에는 외부 depth 없이도 IDEM token만으로 inference**.

> ❓ **예상 질문**: ZoeDepth 같은 monocular depth estimator를 직접 쓰는 것과 비교해 무엇이 다른가?
> **답변**: (1) 추가 model을 inference time에 호출하지 않아 latency↓, (2) IDEM이 action-relevant한 depth feature만 학습 → task-aware compression, (3) depth estimator의 metric scale 오류에 비독립적으로 강인.

### 2.3 SEM: Spatial Enhancement Module

IDEM이 만든 depth token을 **vision token에 modulation(FiLM-style)** 으로 주입:

```
v'_i = γ(D_token) * v_i + β(D_token)
```

이는 단순 concatenation이나 cross-attention보다 **parameter cost가 낮고**, depth가 vision feature의 분포를 직접 reshape하도록 한다.

### 2.4 Progressive Alignment Training (PAT)

3단계 학습:
1. **Stage 1 — Depth pretraining**: IDEM이 ground-truth(또는 estimated) depth를 예측하도록 supervised
2. **Stage 2 — Vision-depth alignment**: IDEM/SEM 출력이 LM의 token space와 호환되도록 alignment loss로 학습
3. **Stage 3 — Action learning**: Flow matching loss로 end-to-end policy fine-tuning. 이 단계에서는 depth supervision을 제거하고 행동 신호만 사용

> ❓ **예상 질문**: 왜 한 번에 end-to-end로 학습하지 않는가?
> **답변**: Depth와 action은 서로 다른 표현을 요구. 0.9B 경량 모델이 두 신호를 한꺼번에 받으면 underfitting. 단계적 학습으로 **representation을 단계별로 anchor**.

---

## 3. 데이터 전략

- **Stage 1 (depth)**: RGB-D를 포함하는 robot dataset(LIBERO-RGBD, real-world calibrated)
- **Stage 2 (alignment)**: 동일 데이터셋의 vision-language pair
- **Stage 3 (action)**: LIBERO, MetaWorld, real-world bench 4종 + 자체 수집 demonstration

> ❓ **예상 질문**: Stage 1에서 사용한 RGB-D 데이터 양이 충분한가?
> **답변**: 정확한 수치 미공개. depth pretraining의 scale이 IDEM 품질을 좌우하므로 ablation 부재가 약점.

---

## 4. 시스템/학습 세부사항

| 컴포넌트 | 사양 |
|--------|------|
| Vision | InternVL3-1B |
| LM | Qwen2.5-0.5B (14 layers) |
| Total parameters | **0.9B** |
| Action head | Flow matching DiT |
| Inference | "highest inference frequency" (구체 수치 미공개) |
| GPU memory | "lowest" (구체 수치 미공개) |

---

## 5. 실험 설계 및 평가 프로토콜

평가 4축:
1. **LIBERO** (Spatial / Object / Goal / Long) — 표준 SR
2. **LIBERO-Plus** — robustness 변형 (조명/시점/노이즈)
3. **MetaWorld** — 50 task suite
4. **VLA-Arena** — multi-task generalization
5. **Real-world** — 4개 평가 suite, average SR

---

## 6. 실험 결과 심층 분석

### LIBERO (Table)

| 모델 | Params | LIBERO Avg | LIBERO-Plus |
|------|--------|-----------|-------------|
| OpenVLA (baseline) | 7B | ~80% | <60% |
| π₀ | 3B | ~91% | ~65% |
| **Evo-Depth** | **0.9B** | **95.4%** | **69.6%** |

**해석**: 8× 작은 모델로 π₀급 또는 그 이상 — depth feature의 효과가 lightweight 환경에서 특히 큼.

### MetaWorld

| 모델 | Avg SR |
|------|--------|
| OpenVLA | 미공개 |
| **Evo-Depth** | **84.4%** |

### VLA-Arena

| 모델 | Total |
|------|-------|
| **Evo-Depth** | **41.1%** |

> ⚠️ VLA-Arena 점수의 절대값(41.1%)이 100점 만점인지 정규화 점수인지 불명확. Baseline 표가 함께 있어야 의미 해석 가능.

### Real-World

- 4개 task suite 평균 **90.0% SR**
- 가장 적은 GPU memory, 가장 높은 inference frequency를 동시에 달성했다고 주장

---

## 7. Ablation 분석

논문에서 명시한 ablation (정확한 수치는 PDF에서만 확인 가능):
- **w/o IDEM**: depth implicit encoding 제거 → LIBERO에서 수 % drop
- **w/o SEM**: depth-aware modulation 제거 → spatial reasoning 태스크에서 큰 drop
- **w/o PAT**: end-to-end joint training → 수렴 불안정, 평균 SR 하락

> ❓ **예상 질문**: IDEM과 SEM 중 어느 쪽이 더 중요한가?
> **답변**: SEM이 vision token에 직접 작용하므로 정책 영향이 크나, IDEM 없이는 SEM에 줄 depth signal 자체가 없음 → **complementary**.

---

## 8. 관련 연구 비교

| 모델 | Params | 입력 | Depth 사용 | LIBERO Avg |
|------|--------|------|-----------|-----------|
| OpenVLA | 7B | RGB | ✗ | ~80% |
| π₀ | 3B | RGB | ✗ | ~91% |
| CogACT | 7B | RGB | ✗ | ~91% |
| 3D-VLA | 4B+ | RGB-D | Explicit | LIBERO 미보고 |
| **Evo-Depth** | **0.9B** | **RGB** | **Implicit (IDEM)** | **95.4%** |

핵심 차별점:
- **Implicit depth** 접근으로 RGB-D 센서 의존성 제거
- **Lightweight first** — edge 배포 가능
- 그럼에도 unseen LIBERO-Plus에서 robust(69.6%)

---

## 9. 한계 및 미해결 문제

1. **Depth supervision의 출처가 불명확**: Stage 1에서 어떤 depth(GT vs estimated)를 사용했는지 ablation 부재 → "implicit"의 진정한 가치 판단 어려움
2. **추론 latency/Hz 수치 미공개**: "highest inference frequency"라 하지만 실제 Hz 미보고
3. **LIBERO-Plus 69.6%**: SR이 LIBERO 95.4% 대비 25%p 떨어짐 → robustness gap이 여전
4. **VLA-Arena 41.1%의 절대값 해석**: 비교 baseline 부재로 의미 평가 어려움
5. **3D-VLA / SpatialVLA와의 직접 비교 부재**: depth-aware 경쟁 모델과의 head-to-head 없음
6. **MetaWorld 84.4%만으로는 어떤 sub-task에서 강점이 있는지 불명확**

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Implicit depth encoding은 SpatialVLA 등 explicit 대비 신선한 접근 |
| **Technical depth** | ★★★★☆ — IDEM/SEM/PAT의 3단계 설계가 합리적 |
| **Experimental rigor** | ★★★★☆ — 4개 벤치마크 + LIBERO-Plus robustness까지 평가 |
| **Practical impact** | ★★★★★ — **0.9B**로 95.4% LIBERO는 edge 배포에 강력 |
| **Writing quality** | ★★★★☆ — 명확한 motivation |

**강점**: 1B 미만의 가장 작은 VLA 중 하나로 LIBERO SOTA급(95.4%)을 달성. **Depth implicit**으로 inference cost 추가 없이 3D-aware feature 활용. **약점**: implicit depth의 학습이 결국 ground-truth depth supervision에 의존하면 "implicit"이라는 주장이 약해짐. Inference Hz 같은 핵심 efficiency 지표를 정량 보고하지 않음.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|--------------|
| 1 | "Implicit depth"가 결국 GT depth supervision에 의존한다면 SpatialVLA와 무엇이 다른가? | Inference time에 외부 depth estimator 호출이 없다는 점이 핵심. Pretraining only — deployment is RGB-only |
| 2 | 0.9B 모델이 진짜 7B 모델보다 빠른가? | "Highest inference frequency"라 주장하지만 절대 Hz 미공개. Memory도 "lowest"만 표시 |
| 3 | LIBERO-Plus 69.6%는 충분한가? | LIBERO 95.4% 대비 -25%p — robustness gap이 크고, 실제 unseen 환경에서 성능 변동성 우려 |
| 4 | InternVL3-1B + Qwen2.5-0.5B가 어떻게 0.9B인가? | 두 모델의 layer/parameter 일부만 사용하는 형태로 추정 — 정확한 breakdown 미공개 |
| 5 | Flow matching의 step 수는? | 미공개. π₀의 10 step 대비 효율성 비교가 필요 |
| 6 | VLA-Arena 41.1%의 baseline은? | 본문에서 baseline 비교 부족 — 단독 수치만으로는 의미 해석 어려움 |
| 7 | Real-world 4 suite의 task 종류와 난이도? | "4 task suites" 외 구체적 정보 부족 — pick&place 위주인지 long-horizon인지 |
| 8 | 3D-VLA, SpatialVLA, DepthVLA와의 직접 비교는? | 본문에 head-to-head 부재 — depth-aware VLA 영역에서 가장 큰 약점 |

<!-- VERIFIED: pdf -->
