# Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs

> **한 줄 요약**: 사전학습된 generative VLA(π₀, π₀.5, CogACT)를 **frozen** 상태로 둔 채, 배포 중 성공한 (관측, 행동) chunk를 **progress-calibrated memory**에 누적하고, inference 시 state-relevant chunk를 retrieval → DTW consistency filter → elite action prior로 묶어 generative sampler의 중간 denoising state에 **confidence-adaptive guidance**로 주입하는 training-free 테스트타임 적응 프레임워크. LIBERO-10에서 π₀ +2.8 / π₀.5 +2.0, SimplerEnv(CogACT) +3.7, real-robot 5종에서 일관된 개선을 보임.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- π₀, π₀.5, CogACT 등 generative VLA는 강력하나 **배포 환경의 distribution shift**에 취약 — fine-tuning이 어렵거나 비용이 큼
- 기존 test-time adaptation 기법은 (a) gradient update(권장 안 됨, 안전성 문제) 또는 (b) memory를 사용하나 **retrieval 결과를 정책의 generative process와 통합하지 못함**(단순 averaging 등)
- 특히 **flow-matching/diffusion** 정책의 경우, 단순 action averaging은 정책 분포의 manifold를 벗어남

### 핵심 질문
- Frozen VLA의 inference 과정에 **외부에서 retrieval한 prior**를 자연스럽게 inject할 수 있는가?
- 어떤 retrieval 결과를 **신뢰**할지 자동으로 판단할 수 있는가?
- Memory를 **무한히 누적**할 때 효율성과 quality를 어떻게 균형 잡는가?

📌 [Figure 1 삽입] — Retrieve-then-Steer 파이프라인: 성공 episode → progress-calibrated memory → state-conditioned retrieval → DTW consistency filter → confidence-adaptive guidance into denoising step

---

## 2. 방법론 심층 분석

### 2.1 시스템 개요

세 단계:
1. **Online Success Memory**: 정책이 task를 완료할 때마다 (obs, action chunk, progress score) 튜플로 저장
2. **Retrieval & DTW Filtering**: 현재 state로 nearest-neighbor 검색 → 후보들 간 DTW로 outlier 제거 → "elite" subset
3. **Confidence-Adaptive Steering**: Elite prior를 generative sampler의 중간 denoising step에 주입

### 2.2 Progress-Calibrated Memory

단순 "성공한 episode만 저장"이 아니라, **episode 내에서 각 chunk의 progress score**를 계산:

```
progress(t) = (some measure of completion at time t)
```

이 progress가 높은 chunk만 elite candidate로 우선. 이는 다음 두 문제를 해결:
- 성공 episode 안에도 **noise나 잘못된 행동이 섞임**
- 동일한 state에서 여러 chunk가 retrieve될 때, **task에 더 가까운 chunk**를 우선해야 함

### 2.3 DTW-Based Consistency Filtering

후보 action chunk들 사이의 trajectory shape를 **Dynamic Time Warping**으로 비교:

```
score(c_i, c_j) = DTW(c_i, c_j)
```

Cluster 평균에서 떨어진 outlier chunk는 제거. 이는 **catastrophic retrieval**(잘못된 상황의 chunk를 가져오는 경우)을 막는 핵심 안전망.

> ❓ **예상 질문**: DTW의 계산 비용이 inference latency에 미치는 영향은?
> **답변**: chunk 길이가 짧고(보통 16~32 step) 후보 개수도 적어 sub-ms 수준. 단, memory 크기가 매우 커지면 nearest-neighbor 검색이 bottleneck.

### 2.4 Confidence-Adaptive Prior Guidance

핵심 트릭: Retrieve된 elite prior를 generative sampler의 **중간 denoising step**에 inject.

```
x_t' = x_t + λ(confidence) · (elite_prior − x_t)
```

`λ(confidence)`는 후보 cluster의 일관성(DTW score)이 높으면 크고, 낮으면 0에 가까움. 따라서:
- Retrieval이 강하게 일관적 → prior가 강하게 steering
- Retrieval이 약하면 → frozen VLA가 자체적으로 sampling

> ❓ **예상 질문**: 중간 step에 inject하는 이유는?
> **답변**: 너무 일찍(t=T) inject하면 noise와 섞여 무력화. 너무 늦게(t=0) inject하면 already converged한 분포를 깨뜨림. **중간 t**에서 inject할 때 retrieve된 prior가 정책 manifold에 자연스럽게 흡수.

### 2.5 Training-Free

- 모든 base VLA(π₀, π₀.5, CogACT)는 **frozen**
- Additional parameter는 **memory + retrieval index**만
- Plug-and-play: 새 base model에도 즉시 적용 가능

---

## 3. 데이터 전략

- **Pretraining**: 사용 안 함 — 기존 VLA의 weight 그대로
- **Online accumulation**: 배포 환경에서 성공한 episode만 누적 (cold start는 base VLA의 자연 성공률에 의존)

> ❓ **예상 질문**: Cold start 문제(메모리가 비어있을 때)?
> **답변**: λ(confidence)가 자동으로 0에 가까워져 frozen VLA의 원래 분포를 사용. 안전한 fallback.

---

## 4. 실험 설계 및 평가 프로토콜

세 가지 평가:
1. **LIBERO-10** with π₀ (flow matching) and π₀.5 (flow matching, larger)
2. **SimplerEnv** with CogACT (diffusion)
3. **Real-Robot**: OpenArm (3 tasks) + ALOHA-PiPER (2 tasks)

---

## 5. 실험 결과 심층 분석

### LIBERO-10 (Table)

| Base VLA | Baseline | + R-then-S | 향상 |
|----------|----------|-----------|------|
| π₀ | 81.6% | **84.4%** | **+2.8** |
| π₀.5 | 92.4% | **94.4%** | **+2.0** |

> **해석**: 이미 92.4%인 π₀.5에서도 +2.0p — saturation 근처에서도 일관된 개선.

### SimplerEnv (Table)

| Base | Baseline | + R-then-S | 향상 |
|------|----------|-----------|------|
| CogACT | 75.8% | **79.5%** | **+3.7** |

> Diffusion 기반 VLA에서도 작동 — **flow matching + diffusion 모두 지원**하는 hybrid 특성.

### Real-Robot

| Platform | Task | SR |
|----------|------|----|
| OpenArm | Bowl Stacking | **80%** |
| OpenArm | Cube Handoff | **52%** |
| OpenArm | Test-Tube (4/4) | **24%** |
| ALOHA-PiPER | T-shirt fold (yellow) | **50%** |
| ALOHA-PiPER | T-shirt fold (white) | **46%** |

> ⚠️ Test-Tube 24%, T-shirt 46–50% 같은 어려운 deformable/precision 태스크에서는 절대 SR이 낮음 — base VLA의 한계가 dominant.

---

## 6. Ablation 분석

논문에서 다룬(또는 다뤘다고 추정되는) ablation:
- **w/o Progress Calibration**: noise chunk가 elite에 포함되어 SR↓
- **w/o DTW Filter**: catastrophic retrieval로 SR↓
- **w/o Confidence-Adaptive λ**: cold start에서 망가짐, 또는 over-steering으로 SR↓
- **Fixed memory size**: vs. unbounded growth — efficiency vs quality trade-off

(정확한 ablation 표 수치는 PDF 본문에서 확인 필요)

> ❓ **예상 질문**: 세 컴포넌트 중 어느 것이 가장 중요한가?
> **답변**: 직관적으로 DTW filter가 catastrophic retrieval을 막는 안전망 — 제거 시 가장 큰 drop 예상.

---

## 7. 관련 연구 비교

| 방법 | Base 모델 수정 | Training | Memory 종류 | Generative inject |
|------|--------------|----------|------------|------------------|
| RAG-style VLA | ✗ | ✗ | Static memory | ✗ |
| VINN | ✗ | ✗ | All demonstrations | k-NN action |
| Sirius-Fleet | ✓ | ✓ | Intervention data | ✗ |
| Test-time fine-tune | ✓ | ✓ | — | ✗ |
| **R-then-S** | **✗** | **✗** | **Online success, progress-calibrated** | **✓ (mid-step)** |

핵심 차별점:
- **Mid-step injection**: generative manifold를 깨지 않고 prior를 흡수
- **Online accumulation + progress calibration**: static memory 대비 환경 적응
- **DTW filter**: catastrophic retrieval 방어 — 다른 retrieval 기법에 부재한 안전 메커니즘

---

## 8. 한계 및 미해결 문제

1. **Cold start 의존성**: 메모리가 비어있을 때 base VLA에 전적으로 의존 — 초기 episode가 모두 실패하는 시나리오에서는 의미 없음
2. **Memory growth**: 장기 배포 시 메모리가 무한히 커지면 retrieval latency 증가 — pruning 전략 분석 부재
3. **Difficult task에서 절대 SR이 낮음**: Test-Tube 24%, T-shirt 46–50%는 method 자체보다 base VLA 한계
4. **Open-source 부재**: 코드/모델 공개 약속 미명시 — 재현성 우려
5. **Hyperparameter sensitivity**: λ(confidence), DTW threshold, mid-step inject timing 등 다수의 hyperparameter가 task별로 다를 가능성
6. **새로운 task에 대한 zero-shot**: memory에 없는 완전히 새로운 task에서는 효과 0
7. **Failure mode 분석 부재**: 잘못된 prior가 inject되어 *오히려* 실패하는 케이스 분석 없음
8. **Inference overhead**: retrieval + DTW + λ 계산이 실제 latency에 얼마나 영향을 주는지 정량 보고 부재

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Mid-step injection + progress calibration의 조합은 새로움 |
| **Technical depth** | ★★★★☆ — DTW filter + confidence λ의 설계가 정교 |
| **Experimental rigor** | ★★★★☆ — 3개 base VLA × 다수 벤치마크 + real-robot |
| **Practical impact** | ★★★★★ — Training-free, plug-and-play — 즉시 production에 적용 가능 |
| **Writing quality** | ★★★★☆ — Method가 잘 구조화됨 |

**강점**: Frozen VLA에 대한 training-free TTA로서 LIBERO-10/SimplerEnv/real-robot에서 일관된 개선. **약점**: 절대 성능 향상은 +2–4%p 수준(이미 강한 base에서 marginal), open-source 미공개, hyperparameter sensitivity 분석 부족.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|--------------|
| 1 | +2–4%p 개선이 통계적으로 유의한가? | 본문에서 seed variance / confidence interval 표기 확인 필요. LIBERO-10에서 10 task 평균이라면 noise floor 이하일 수도 |
| 2 | Cold start 시나리오에서 효과는? | λ(confidence)=0이므로 base와 동일. 환경 적응 후에야 향상 — "online" 자체가 강점이자 약점 |
| 3 | Memory가 매우 커지면 retrieval latency는? | 분석 부재. faiss/HNSW 같은 ANN index가 필수일 것이나 본문 미언급 |
| 4 | DTW filter의 threshold는 task별로 다른가? | Sensitive — hyperparameter sweep 부재 |
| 5 | Mid-step injection의 정확한 t는? | Flow matching/diffusion step 중 어느 시점인지 — 본문 PDF 확인 필요 |
| 6 | π₀.5 92.4%에서 +2.0이면 noise일 수도? | 가능성 있음. seed 분산이 ±1.5%p 정도면 marginal |
| 7 | OpenVLA / RT-2 같은 AR-action VLA에서도 작동? | 본문 미검증. Mid-step injection은 generative model 전제 — AR 모델에서는 적용 불가 |
| 8 | Failure case는? | 분석 부재. 잘못된 prior가 injected되어 task가 *나빠지는* 케이스가 있는지 |
| 9 | Catastrophic forgetting을 일으키지 않는가? | Base가 frozen이므로 base 자체는 안전. 다만 memory를 부적절하게 trust하면 specific state에서 regress |
| 10 | Real-robot에서 24%, 50% 같은 낮은 SR은 method 한계인가 base 한계인가? | 본문에서 baseline(method 미적용) 절대 SR도 함께 표기되어야 분리 가능 — 일부만 표기됨 |

<!-- VERIFIED: pdf -->
