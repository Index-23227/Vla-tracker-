# ECHO: Continuous Hierarchical Memory for Vision-Language-Action Models

> **한 줄 요약**: Xi'an Jiaotong University 팀의 long-horizon 메모리 프레임워크. pi_0의 hidden state를 **513-d Lorentz 매니폴드**(hyperbolic 공간)로 매핑하는 가벼운 MLP autoencoder + entailment 페널티로 **semantic memory tree**를 구성하고, **Cone Tree 빔서치**(빔폭 k=3)로 검색, 백그라운드 consolidation으로 보간·split하는 plug-and-play 모듈. LIBERO-Long에서 vanilla pi_0(80.70%)을 **93.48%** (+12.78pp), Spatial/Object/Goal/Long 평균 **97.3%** 로 끌어올렸고, 실제 Franka에서도 평균 70.0%로 pi_0 58.3% 대비 +11.7pp.

---

## 1. 배경 및 동기

- VLA의 long-horizon manipulation 성능은 결국 **경험 메모리**에 의해 결정된다. MemoryVLA, MAP-VLA 등 최근 모델들이 이를 시도하지만 대부분 **flat / Euclidean 메모리**다.
- Flat memory의 한계:
  - 카테고리 간 계층 구조를 표현하지 못함 → 효율적 검색 불가.
  - 메모리 크기가 커지면 kNN linear scan 비용 폭증.
  - cross-suite 일반화에서 task 사이의 의미적 관계를 활용 못함.
- 인간 의미 기억의 hierarchical organization과, **hyperbolic 공간이 트리를 거의 distortion-free로 임베딩한다**는 사실에서 영감.

### 연구 질문
1. VLA의 hidden state를 hyperbolic 공간으로 옮기면 manipulation memory가 자연히 트리화되는가?
2. 트리 구조를 검색에 활용해 latency와 정확도 둘 다 잡을 수 있는가?
3. 메모리를 **자동 보강(consolidation)** 하면 데이터 없이도 성능을 더 끌어올릴 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 Hyperbolic Autoencoder (HAE)
- Encoder: lightweight MLP (GELU) → Lorentz 매니폴드 위의 (n+1)차원 표현, **n=512** (총 513-d).
- Lorentz bilinear form: ⟨x, y⟩_ℒ = -x₀ y₀ + Σᵢ xᵢ yᵢ.
- 인코딩은 **exponential map**, 검색·디코딩은 **logarithmic map** 으로 Euclidean 영역과 왕복.
- 학습 손실:
  1. reconstruction loss
  2. Lorentz graph regularization
  3. **entailment penalty** (부모-자식 cone constraint)
  4. latent norm regularization
- Optimizer: AdamW, LR 1e-3, weight decay 1e-6, batch 256.

### 2.2 Cone Tree Retrieval
- 쿼리 q에 대해 노드 z_μ가 entailment cone 조건 ϕ(q, z_μ) ≤ ω(z_μ) 를 만족하면 자식으로 확장.
- **빔 폭 k = 3** 의 hierarchical beam search → 관련 서브트리만 탐색, global linear scan 회피.
- Appendix E.2: 메모리 노드 수가 늘어나도 standard kNN 대비 **CPU 검색 latency가 sublinear** 로 스케일.

### 2.3 Background Consolidation
- 신규 경험이 누적되면 백그라운드 스레드에서:
  - **Geometric interpolation**: 같은 부모를 갖는 이웃 노드 사이를 보간 → 신규 "virtual" 경험 합성.
  - **Structural splitting**: 너무 커진 노드를 분할해 트리 균형 유지.
- 데이터 추가 없이도 representation richness 증가 — Table 2의 +1.44pp가 이 모듈의 순수 기여.

### 2.4 Memory Injection into pi_0
- **pi_0 백본은 그대로 둠** (plug-and-play).
- suffix embedding에 잔차 형태로 주입: `E'_suffix = E_suffix + proj(α_t · (v_prior - h_last))`.
- 주입 강도 α_0 = 0.03 (보수적 hyperparameter).

---

## 3. 실험 결과

### 3.1 LIBERO + LIBERO-Plus (Table 1, 3 seeds, % SR)

| Method | Spatial | Object | Goal | Long-10 | LIBERO-Plus |
|---|---|---|---|---|---|
| Octo | 78.9±1.0 | 85.7±0.9 | 84.6±0.9 | 51.1±1.3 | - |
| OpenVLA | 84.7±0.9 | 88.4±0.8 | 79.2±1.0 | 53.7±1.3 | 17.3±3.2 |
| MAP-VLA | 96.3 | 98.4 | 95.4 | 83.4±0.7 | - |
| MemoryVLA | 98.0±0.6 | 97.4±0.9 | 96.4±1.3 | 92.4±1.1 | - |
| Vanilla π₀ | 97.5±1.7 | 97.0±1.2 | 92.3±2.5 | 80.7±2.0 | 54.2±2.9 |
| **ECHO (Ours)** | **98.3±1.0** | **98.8±0.5** | **98.6±1.0** | **93.5±2.6** | **56.5±2.0** |

- 4개 suite 평균 **97.3%** — 모든 suite에서 SOTA.
- 가장 큰 향상은 Long(-10) 에서 (+12.78pp over pi_0); 짧은 task에서도 1-6pp 향상.
- **LIBERO-Plus**(perturbed eval)에서도 OpenVLA·pi_0를 상회 → robustness 검증.

### 3.2 Real-World Franka (Table 5)

| Model | Place Banana | Stack Blocks | Insert Circle | **Avg** |
|---|---|---|---|---|
| Vanilla π₀ | 75.0 | 45.0 | 55.0 | 58.3 |
| **ECHO** | **90.0** | **55.0** | **65.0** | **70.0** |

- Franka Emika Panda 15 Hz, remote policy server inference.
- 모든 task에서 5-15pp 향상.

### 3.3 Cross-Suite Generalization
- LIBERO-Spatial/Object/Goal로만 memory를 학습한 뒤 **LIBERO-Long**(unseen long-horizon)에 평가: **89.31%**.
- 동일 setting의 vanilla pi_0 80.70% 대비 +8.6pp — **memory tree가 task 카테고리 간 의미를 일반화**함을 시사.

---

## 4. 어블레이션 (Table 2, LIBERO-Long)

| 구성 | Memory Space | Retrieval | Success (%) |
|---|---|---|---|
| Vanilla π₀ | – | – | 80.70±2.01 |
| + Short-term buffer | Euclidean | None | 88.81±2.01 |
| + Flat memory | Euclidean | kNN | 83.25±5.01 |
| + Hyperbolic memory | Hyperbolic | kNN | 91.11±3.50 |
| + Cone Tree | Hyperbolic | Cone Tree | 92.04±3.50 |
| **ECHO (전체)** | **Hyperbolic** | **Cone Tree + Consolidation** | **93.48±2.89** |

기여 분해:
- Short-term buffer만으로 +8.11pp (놀랍게도 큼).
- Flat Euclidean kNN: +2.55pp (오히려 short-term보다 낮음 — 단순 검색은 노이즈).
- Hyperbolic 공간 도입: +10.41pp.
- Cone Tree 추가: +11.34pp.
- Background consolidation 추가: **+12.78pp** (consolidation 자체 +1.44pp).

→ **Hyperbolic 공간 자체**가 가장 큰 단일 기여, 그 위에 검색 구조와 consolidation이 +2-3pp씩 누적.

---

## 5. 한계

| # | 한계 | 코멘트 |
|---|---|---|
| 1 | semantic keyframe 의존 | Appendix G: "performance still depends on the quality of semantic keyframe extraction." keyframe 선택 자체의 robustness 분석 부족. |
| 2 | real-world 좁은 범위 | 3 tabletop task만 평가 — household / mobile / dexterous 미검증 |
| 3 | scaling | "lifelong memory banks"로의 확장은 future work으로 명시 |
| 4 | 코드 비공개 | GitHub URL 부재 — 재현성 한계 |
| 5 | wall-clock 학습 시간 미보고 | 메모리 뱅크 구성 4.1-4.6분만 보고. policy 학습 GPU-hours 부재 |
| 6 | LIBERO 외 sim 부재 | CALVIN/SimplerEnv/RoboCasa 평가 없음 — long-horizon 외 task에서의 효과 검증 부족 |
| 7 | α_0=0.03 고정 | 주입 강도 sensitivity 미공개 |

---

## 6. 의의

- VLA + hierarchical memory에 **연속 hyperbolic 공간**이라는 강한 수학적 prior를 도입한 첫 사례.
- pi_0 같은 강력한 foundation model 위에서도 **+12.78pp**라는 큰 절대 향상 → long-horizon에서 memory가 여전히 미해결 문제임을 재확인.
- Cone Tree retrieval은 메모리가 lifelong로 커질수록 latency 측면에서 점점 더 유리 — **scaling-friendly memory** 의 구체적 청사진.
- Plug-and-play 잔차 주입(α_0=0.03) 디자인은 다른 VLA(pi_0.5, GR00T, OpenVLA)에도 그대로 이식 가능할 가능성.

---

## 7. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | Hyperbolic 공간이 왜 manipulation memory에 적합한가? | 트리 구조(task 카테고리 → sub-skill)를 distortion 거의 없이 임베딩. Cone constraint로 entailment 의미가 자연스럽게 인코딩. Euclidean 대비 Long-10에서 +7.86pp가 그 증거. |
| 2 | Short-term buffer만으로도 +8.11pp인데 hyperbolic이 정말 필요한가? | 추가 +4.67pp + consolidation 효과 + scaling-friendly 검색. 단순 buffer는 long-horizon에서 memory size↑ 시 비효율. |
| 3 | 메모리 뱅크 구성 4분이 데이터셋마다 매번 필요한가? | task suite별 일회성 구성. inference time에는 Cone Tree 검색만 발생. |
| 4 | Cone Tree 빔 k=3 — k 변화 ablation은? | paper 본문에는 미공개. k↑는 정확도↑ latency↑ trade-off 예상. |
| 5 | π₀ 외 다른 VLA(pi_0.5, GR00T)에 그대로 적용 가능? | 잔차 주입 구조이므로 hidden state가 있는 모든 VLA에 이식 가능. 다만 hidden state semantic이 다르면 HAE 재학습 필요. |
| 6 | LIBERO-Plus 56.5%는 절대값이 낮은데, perturbation에 대한 robustness가 부족한 것 아닌가? | OpenVLA 17.3% 대비 3배 이상. pi_0 54.2% 대비도 +2.3pp. 절대값이 낮은 것은 LIBERO-Plus 자체 난이도 때문. |
| 7 | Consolidation의 virtual memory가 실제 경험을 오염시킬 위험은? | Geometric interpolation은 같은 부모 노드 자식들 사이에서만 일어남(entailment 보장). +1.44pp 기여로 net positive. 다만 long-tail task에서의 risk 미평가. |
| 8 | pi_0 대비 latency는? | 본문 명시 latency 수치 없음. Cone Tree 검색이 sublinear scaling이라는 보장만 제공. policy step latency 자체는 잔차 주입 1회로 거의 변화 없을 것. |

<!-- VERIFIED: pdf -->
