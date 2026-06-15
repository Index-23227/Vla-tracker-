# CT-VAM: A Cerebello-Thalamic-Inspired Vision-Action Model for Efficient Visuomotor Control

> **한 줄 요약**: 언어를 저수준 제어 루프에서 빼고 one-hot task token으로 대체한 뒤, DINOv3-S+ 듀얼뷰 비주얼 백본 + TARS(스트림 분리 조건부 어텐션) + rectified-flow 액션 디코더 + FCI(flow-consistent inpainting)로 구성된 68M 파라미터 vision-action 정책. LIBERO 4 suite 평균 82.1%로 4000M~7000M VLM 정책들과 경쟁하면서 Jetson Orin NX에서도 실시간 동작.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **OpenVLA / RT-2 / π₀ 같은 VLM 기반 VLA**는 강력한 의미 일반화를 갖지만, 매 제어 스텝마다 언어 조건부 backbone을 다시 통과시키며 **지연시간/메모리 비용**이 큼 (3.3B~7B 파라미터)
- 매니퓰레이션 태스크에서 **raw instruction은 의도(intent)만 지정**하고, 실제 fine-grained motor command는 시각·고유수용 피드백과 학습된 정책 동역학이 결정 (ACT, Diffusion Policy 류의 발견)
- **TinyVLA, BitVLA** 등이 경량화를 시도했으나 여전히 언어 조건부 backbone을 제어 루프 안에 두는 구조

### 핵심 질문
- **task intent가 grounding된 이후 raw language를 저수준 제어 루프에서 완전히 제거해도 강력한 visuomotor 성능을 유지할 수 있는가?**
- **heterogeneous한 입력(action token / 학습 가능 쿼리 / dense vision+proprio / task)을 단일 softmax 어텐션이 아닌 스트림별로 분리해 융합하면 어떤 이득이 있는가?**

### 생물학적 모티프
- 인간은 매 motor step마다 언어 명령을 재해석하지 않음. **소뇌(cerebellum)**: 학습된 운동 동역학과 빠른 sensorimotor feedback. **시상(thalamus)**: 다중 감각 스트림을 라우팅·게이팅하여 운동 영역으로 전달.
- CT-VAM은 이 분업을 모방: language → compact task condition(시상의 입력 라우팅 역할)으로 단발 grounding, 이후 저수준 정책(소뇌의 procedural memory 역할)이 닫힌 루프로 동작.

📌 [Figure 1 삽입] — CT-VAM 아키텍처: high-level decision(선택적) → Task Encoder(one-hot) | DINOv3 듀얼뷰 → cross-view patch fusion | TARS 스트림 분리 어텐션 + rectified-flow → 액션 청크 | FCI로 비동기 실행 | OpenArm 실기 배포

---

## 2. 방법론 심층 분석

### 2.1 문제 정식화 — Grounded Visuomotor Execution

저자는 raw instruction L과 초기 컨텍스트 X₀에서 grounded intent G ~ q_φ(·|L, X₀)를 추출하고, 저수준 정책 π_θ(A_t | G, O_t)로 분리하는 framework을 제안.

**Control-sufficiency 정의 (Definition 1)**: G가 ε_t-control-sufficient 하다는 것은 I(A_t; L | G, O_t) ≤ ε_t.

**Assumption 1 + Proposition 1**: G가 L의 action-relevant information을 보존하면, 저수준 정책은 raw L을 더 이상 참조할 필요가 없다.

> ❓ **예상 질문**: 이게 정말 새로운가? Decoupling 자체는 RT-1 시절부터 알려진 아이디어 아닌가?
> **답변**: 정확하다. 새로움은 information-theoretic formalization(I(A;L|G,O) ≤ ε)으로 명시화한 점, 그리고 실제로 **one-hot task token만**으로 4000M~7000M VLM 정책에 근접하는 성능을 보여줬다는 실증적 증거에 있다. 다만 본 논문은 grounding 모듈 자체는 다루지 않고 G=one-hot으로 단순화 — 곧 Limitation에서 다룰 약점.

### 2.2 아키텍처 — 세 가지 핵심 구성

| 구성 | 역할 |
|------|------|
| Dual-view visual encoder (DINOv3-S+) | primary + wrist 카메라를 layer-wise visual memory D_ℓ로 인코딩 |
| TARS (Thalamic Action Routing Stream) | 4개 스트림(self / action-query / dense / task)을 분리 어텐션 |
| Flow-consistent inpainting (FCI) | 비동기 chunk 추론 시 boundary 연속성 보장 |

### 2.3 TARS — Stream-Separated Conditional Attention

핵심 차별점. 디코더 레이어 ℓ에서 4개의 분리된 메모리 스트림:
- **Self stream** M^self_ℓ = X_ℓ (현재 action token)
- **Action-query stream** M^aq_ℓ = E_aq (DINOv3 register token으로 초기화된 N_q=32개 learnable slot)
- **Dense stream** M^dense_ℓ = [D_ℓ; S] (patch-level visual evidence + proprioception)
- **Task stream** M^task_ℓ = {r} (one-hot에서 임베딩된 단일 task token)

각 스트림은 **독립적으로 LN → 독립 softmax**를 거친 뒤 게이트로 결합:
```
X̂_ℓ = X_ℓ + (Σ_b γ_b A_b) W_o
```
- γ_self = γ_task = 1 (고정)
- γ_aq, γ_dense는 학습 가능 게이트

> ❓ **예상 질문**: 그냥 concat해서 하나의 softmax로 처리하는 VLA-Adapter[21]와 뭐가 다른가?
> **답변**: shared softmax는 토큰 수가 많은 dense visual+proprio가 분모를 지배해 **compact task token (단 1개)이 거의 무시**되는 현상이 생긴다. 스트림별 분리 정규화는 각 스트림에서 독립적으로 확률 분포를 만든 뒤 게이트로 weighted sum하므로 task token이 dense stream에 휩쓸리지 않는다. Appendix B.5의 비교가 이 효과를 확인.

### 2.4 Rectified-Flow Action Decoder

- Flow time τ ∈ [0, 1]에서 noised action state A_τ → predicted velocity
- TARS 디코더가 M번 반복되며 X_0 → ... → X_M의 action token 업데이트
- FFN block은 time-conditioned (τ-embedding 주입)

### 2.5 Flow-Consistent Inpainting (FCI)

비동기 chunk 실행의 핵심 트릭:
- 현재 chunk를 실행하는 동안 다음 chunk를 병렬 추론
- K_ov만큼 남은 액션이 **overlap region** Y_ov ∈ R^(K_ov × d_a)
- naive hard clamp는 rectified-flow의 noise-to-data trajectory를 깨므로, 대신 학습 시 사용한 rectified-flow interpolation을 강제:
```
A^(i)_{1:K_ov} = (1 - τ_i) ε_ov + τ_i Y_ov
```
- 매 flow step마다 overlap constraint 재적용 → boundary 부드럽게 + 학습 분포와 일치하는 디코더 입력 유지

> ❓ **예상 질문**: 그냥 마지막에 액션을 linearly interpolate하면 안 되나?
> **답변**: 안 된다. Rectified flow는 noise → data로의 ODE trajectory를 따르도록 학습되었는데, 디코더 입력에 hard-clamped 값이 들어가면 학습 분포 밖이라 속도 예측이 깨진다. FCI는 overlap 영역도 동일한 (1−τ)ε + τY 인터폴레이션을 따르게 함으로써 디코더가 in-distribution input을 보도록 한다.

---

## 3. 데이터 전략

### 학습 데이터
- **시뮬레이션**: LIBERO 4 suite 표준 학습 split
- **실기**: 각 태스크당 텔레오퍼레이션 30 에피소드 (OpenArm 좌측 팔 관절 공간)

### 데이터 사용 패턴
- One-hot task identifier (N_task개 클래스) — open-vocabulary 아님, **closed task set 가정**
- 배치 크기 32 (실기), 시뮬은 미명시
- Real-world 모든 비교법(Diffusion Policy, π₀, CT-VAM)은 동일한 demo / observation-action space / 평가 프로토콜

> ❓ **예상 질문**: One-hot task가 본질적으로 한계 아닌가? 새 task 추가 시 재학습해야 하는데.
> **답변**: 명확한 한계다. 저자도 Limitations §6에서 인정하며, 향후 raw language → intent로 매핑하는 grounding 모듈을 통합할 것이라고 언급. 본 논문은 "intent가 grounding된 이후 저수준 visuomotor 효율"만 격리해서 평가하는 게 목적.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Visual backbone | DINOv3-S+ (사전학습) |
| Total params | 68M |
| Action chunk length H | 미명시 (Appendix 추정) |
| Action queries N_q | 32 (learnable, DINOv3 register token 초기화) |
| Rectified-flow NFE | 8 steps (CT-VAM, Diffusion Policy 동일); π₀는 50 steps |
| Control frequency | 20 Hz (real-world) |
| Optimizer | AdamW |
| Batch size | 32 (real-world) |
| Hardware (학습) | 미명시 |
| Hardware (배포) | RTX 4080 / Jetson Orin NX (TensorRT) |
| Inference latency (chunk) | RTX 4080: 56.8 ms; Jetson Orin NX: 200.6 ms |

---

## 5. 실험 설계 및 평가 프로토콜

### 5.1 시뮬레이션
- **LIBERO** 4 suite (Spatial / Object / Goal / Long), 표준 프로토콜, **태스크당 50 rollouts**
- One-policy 설정 (4 suite 통합 정책), one-hot task identifier 사용

### 5.2 실기
- **Ball Pouring** (정량 비교용, 20 trials × 4 method-platform 조합)
- **Box Opening and Placement** (long-horizon subtask switching 검증)
- 메트릭: success rate, execution time, inference time
- 인간 개입 없을 때만 성공으로 카운트

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1)

| 모델 | Params | Spatial | Object | Goal | Long | Avg |
|------|--------|---------|--------|------|------|-----|
| OpenVLA | 7000M | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 4000M | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π₀-FAST | 3300M | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀ | 3300M | 90.0 | 86.0 | **95.0** | **73.0** | **86.0** |
| Diffusion Policy | – | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| MDT | ~225M | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| **CT-VAM** | **68M** | **89.0** | **94.6** | 78.4 | 66.2 | **82.1** |

**관찰**:
- **Non-VLM 정책 중 최강** (Diffusion Policy 72.4, MDT 76.1 → CT-VAM 82.1)
- π₀ (3300M) 대비 **48배 작은 파라미터**로 평균 -3.9%p 차이만 — 효율성 측면에서 매우 강력
- Spatial / Object에서 π₀를 능가; Goal / Long에서는 여전히 π₀가 우위 → 장기 horizon에서 VLM의 의미 일반화 효과가 잔존
- **OpenVLA(7000M) 대비 +5.6%p**, **SpatialVLA(4000M) 대비 +4.0%p**로 백본 크기를 1/100 수준으로 줄이고도 우위

### Scaling 비교 (Appendix C.2)
같은 flow-matching 정책의 small/base/large 변종이 65.8 / 76.9 / 77.3% → CT-VAM 82.1%는 **단순 크기 증대로 도달할 수 없는 지점**임을 보임. 즉 TARS + action query design이 기여.

### Real-World — Ball Pouring (Table 2)

| 방법 | RTX 4080 SR | RTX 4080 ExecT | RTX 4080 InfT | Jetson Orin NX SR | Jetson Orin NX ExecT | Jetson Orin NX InfT |
|------|-------------|----------------|---------------|-------------------|---------------------|---------------------|
| Diffusion Policy | 70.0% | 27.21s | 303.3 ms | N.T. | – | – |
| π₀ | 95.0% | 7.82s | 117.2 ms | **N.D.** (메모리 부족) | – | – |
| CT-VAM w/o FCI | **100%** | 8.33s | 56.3 ms | 85% | 10.24s | 256.2 ms |
| **CT-VAM w/ FCI** | 95% | **6.41s** | 56.8 ms | **90%** | **7.23s** | 200.6 ms |

**관찰**:
- RTX 4080에서 CT-VAM의 chunk 추론 latency는 **56.8 ms — π₀의 117.2 ms 대비 2배, Diffusion Policy의 303.3 ms 대비 5.3배 빠름**
- **Jetson Orin NX에서 π₀는 OOM으로 배포 불가**, CT-VAM은 90% 성공률 유지
- FCI는 success rate를 거의 유지하면서 (100→95%) **execution time을 8.33s → 6.41s로 단축** — 추론을 액션 실행과 오버랩
- 20 Hz 제어 주기에서 FCI는 inference latency 대부분을 hide → real-time-equivalent frequency가 목표 20 Hz에 근접

> ❓ **예상 질문**: 100% → 95%로 떨어진 5%p는 통계적으로 유의미한가?
> **답변**: 20 trials × 4 setting이라 통계적 검정력은 제한적 (one trial = 5%p). 즉 1번의 실패가 5%p 차이로 보이는 셈. 저자도 "comparable success rate" 표현으로 다룸. 의미 있는 건 8.33s → 6.41s execution time (23% 단축)이 안정적으로 관찰된다는 점.

---

## 7. Ablation 분석

논문 Table 1 본문 + Appendix C.1/C.2에 산재.

### TARS action queries (Appendix C.1)
- N_q = 32 learnable action queries가 capacity와 optimization stability의 균형점
- learnable이 아닐 때(고정) 성능 저하 — DINOv3 register token 초기화 + joint training이 핵심

### Flow-matching 정책 scaling (Appendix C.2)
| Variant | LIBERO Avg |
|---------|-----------|
| Small | 65.8 |
| Base | 76.9 |
| Large | 77.3 |
| **CT-VAM** | **82.1** |

→ CT-VAM의 향상은 **단순한 파라미터 증대로 설명 안 됨**. TARS + action query 조합의 효과.

### Shared-softmax vs Stream-separated (Appendix B.5)
- VLA-Adapter[21]의 shared-softmax는 dense token이 task token을 dominate
- TARS의 스트림별 분리 LN/softmax가 task token의 영향력 보존

### FCI vs hard inpainting (Appendix B.6)
- Hard clamp는 rectified-flow trajectory를 깨 디코더 입력이 OOD → 성능 저하
- FCI는 trajectory를 유지하며 boundary 연속성 확보

---

## 8. 관련 연구 비교

| 모델 | Params | Language at exec | LIBERO Avg | Edge 배포 |
|------|--------|------------------|------------|-----------|
| OpenVLA | 7000M | ✓ (full VLM) | 76.5 | ✗ |
| SpatialVLA | 4000M | ✓ | 78.1 | ✗ |
| π₀ | 3300M | ✓ | 86.0 | ✗ (Orin NX OOM) |
| π₀-FAST | 3300M | ✓ | 85.5 | ✗ |
| Diffusion Policy | – | – | 72.4 | △ (느림) |
| MDT | ~225M | △ (goal) | 76.1 | △ |
| TinyVLA | – | ✓ | – | △ |
| **CT-VAM** | **68M** | **✗** (one-hot intent) | **82.1** | **✓ (Orin NX)** |

### 핵심 차이
- **언어를 저수준 루프에서 제거**한 첫 번째 경량 정책 중 하나 (TinyVLA는 여전히 VLM 유지)
- **TARS의 스트림 분리** — VLA-Adapter, MDT 등의 fusion 방식과 명확히 구분
- **FCI를 통한 비동기 chunk 추론** — Diffusion Policy / π₀에는 없는 시스템 레벨 최적화

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **One-hot task tokens만 평가**: 실제 사용 시 raw language → intent 매핑이 필요한데, 본 논문은 grounding 모듈을 다루지 않음. 저자도 §6에서 인정.
2. **Closed task set 가정**: 학습 시 본 N_task개 태스크 외 새 태스크에 zero-shot 적용 불가. open-vocabulary VLA의 강점을 포기.
3. **Goal / Long suite에서 π₀ 대비 열세** (78.4 vs 95.0; 66.2 vs 73.0) — VLM의 의미 일반화가 long-horizon에서 여전히 우위. CT-VAM은 효율성-성능 trade-off 지점.
4. **Real-world 평가가 tabletop 2개 태스크에 한정**: Ball Pouring, Box Opening + Placement. broader task distribution, 다양한 embodiment 검증 부재.
5. **학습 하드웨어 / 학습 시간 미보고**: 재현성 측면 정보 부족.
6. **Long-horizon subtask switching이 수동**: Box Opening + Placement 실험에서 subtask 전환을 사람이 트리거. automatic subtask recognition은 future work.

### Attribution 문제
- 82.1%의 향상이 (a) DINOv3-S+ 백본의 강력함, (b) TARS 스트림 분리, (c) rectified flow + action query 중 어느 것의 기여인지 완전 분리 안 됨. Appendix C.2의 scaling 비교가 부분적 답이긴 함.
- "compact policy인데도 강력하다"는 주장에서 68M이 정말 compact한지 — DINOv3-S+ 자체가 사전학습 비용을 흡수하고 있음. fair comparison인가는 논쟁 여지.

### 실용성 측면
- Jetson Orin NX에서 200 ms chunk inference latency는 5 Hz에 해당. FCI로 hide한다 해도 chunk 단위라 reactive control에는 한계.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — TARS 스트림 분리 어텐션 + FCI + 언어 디커플링 조합이 기존 VLA / 비주얼 정책과 명확히 구분 |
| **Technical depth** | ★★★★☆ — information-theoretic formalization (Definition 1, Prop 1) + flow-consistent inpainting 수학이 탄탄 |
| **Experimental rigor** | ★★★☆☆ — LIBERO 결과는 강력하나 real-world가 2개 태스크에 한정; 학습 하드웨어 미보고 |
| **Practical impact** | ★★★★★ — Jetson Orin NX 실시간 배포가 가능한 68M 모델 — 실용성 측면에서 거의 독보적 |
| **Writing quality** | ★★★★☆ — 명확한 구조, biological analogy가 직관적 |

**강점**:
- 68M 파라미터로 LIBERO 82.1%, π₀(3300M, 86.0%)에 근접한 효율성
- TARS의 스트림 분리는 일반화 가능한 디자인 원칙 (다른 multi-modal 정책에도 적용 가능)
- FCI는 chunk 기반 정책의 비동기 실행 문제에 깔끔한 해법
- 실제로 Jetson Orin NX에서 동작하는 — π₀가 OOM되는 환경에서

**약점**:
- One-hot task token이라는 큰 단순화 — open-vocabulary 시나리오 미평가
- Long suite에서 VLM 정책에 열세
- Real-world 평가 범위 제한

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | One-hot task token이 raw language 대비 정보 손실 없나? | Definition 1의 ε-control-sufficiency가 형식적 근거. 단 closed task set 가정이 필수. Open-vocab에선 별도 grounding 모듈 필요 — 미평가. |
| 2 | TARS가 정말 shared softmax보다 우월한가? | Appendix B.5의 직접 비교가 답이지만, 본문에 정량 수치 부족. Scaling 실험(65.8/76.9/77.3 → 82.1)이 간접 증거. |
| 3 | π₀ Long suite 73.0 vs CT-VAM 66.2 — 결국 VLM이 필요? | 그렇다. 의미 일반화가 중요한 long-horizon에선 VLM 우위 잔존. CT-VAM은 "효율성 우선" 포지셔닝. |
| 4 | FCI가 boundary continuity를 정말 보장? | (1−τ)ε_ov + τ Y_ov 인터폴레이션이 학습 시 rectified-flow training과 동일 형식이므로 디코더가 in-distribution input을 봄. Hard clamp와의 비교는 Appendix B.6. |
| 5 | DINOv3-S+ 의존도가 크지 않나? | 사전학습 백본의 강점을 활용하는 게 맞고, 이게 fair comparison인지는 논쟁 여지. 다만 SpatialVLA 등도 비슷한 사전학습 자산을 씀. |
| 6 | 68M이 정말 compact한가? Embedded 관점에서? | Jetson Orin NX 8GB에서 동작한다는 것 자체가 실증. π₀는 OOM. 다만 200 ms chunk inference는 high-frequency 제어엔 빠듯. |
| 7 | Action chunk length H가 명시 안 됨 | 본문 미명시. Appendix B 참조 필요. 일반적으로 ACT/Diffusion Policy 류는 16~32 step 사용. |
| 8 | Real-world 20 trials는 통계적으로 부족하지 않나? | 부족하다. one trial = 5%p. 95% vs 100% 같은 차이는 통계적으로 의미 없을 수 있음. 저자도 "comparable" 표현으로 다룸. |
| 9 | Cerebello-thalamic analogy는 마케팅 아닌가? | 부분적으로 그렇다. 본질은 "fast/local execution vs slow/semantic grounding" 분리. Appendix B.1의 functional analogy는 motivation 수준. |
| 10 | 다른 embodiment(bimanual, dexterous hand) 적용 가능? | 미평가. OpenArm 단일 팔만 검증. Future work. |

<!-- VERIFIED: pdf -->
