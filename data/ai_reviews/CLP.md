# CLP: Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think

> **한 줄 요약**: π0/GR00T-N1.5/SmolVLA 같은 현대 continuous-control VLA에서 single forward pass로 측정한 CKA 유사도를 이용해 representational 중복 레이어를 fine-tuning 이전에 정적으로 제거(최대 50% depth ↓)하는 calibration-only 압축 프레임워크. LIBERO 4 suite에서 π0-CLP 93.9% (base 94.6%) · GR00T-N1.5-CLP 93.0% (base 93.9%) · SmolVLA-CLP 76.75% (base 77.15%)를 ×1.39–1.47 speedup으로 달성하며, 10% LIBERO에서는 π0 baseline 77.7% → 84.6%로 오히려 향상.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Training-free token pruning** (FastV, DivPrune, EfficientVLA, SpecPrune-VLA, ADP, VLA-Cache) — inference만 가속, fine-tuning 비용(LIBERO 20h on 4×A100) 미해결
- **Lightweight from-scratch architectures** (RoboMamba, FLOWER-VLA, SmolVLA, NORA) — 대형 사전학습 VLA의 광범위한 capability를 완전히 계승하지 못함
- **Training-adaptive routing** (DeeR-VLA, MoLe-VLA, AC²-VLA) — auxiliary router/early-exit head/distillation 등이 아키텍처 복잡도와 학습 부담을 키움

### 핵심 질문
- **현대 continuous-control VLA에서 fine-tuning 이전에 정적으로 layer를 제거하면서 policy 성능을 유지할 수 있는가?**
- **token이나 token-routing이 아니라 layer 단위 redundancy를 single forward pass로 정량화할 수 있는가?**

📌 [Figure 1 삽입] — CLP 파이프라인: calibration data → CKA 유사도 행렬 → contiguous 고-유사도 블록 → anchor 유지 + 나머지 제거 → 정적 compressed 정책 → 일반 fine-tuning

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

CLP는 **모델 자체가 아니라** 다음 4개 입출력을 가지는 **압축 파이프라인**:
- Input: 사전학습된 VLA π_θ (π0/GR00T-N1.5/SmolVLA), 대상 모듈 M, 예산 k_M, threshold τ
- Output: 정적으로 truncate된 π_θ^pruned

대상 모듈 M은 (i) VLM backbone (ii) action expert / action head 둘 다 적용 가능. 모든 transformer block이 동일한 hidden dim을 공유하므로 단순히 predecessor↔successor를 연결만 하면 됨 → auxiliary routing/adapter 불필요.

### 2.2 CKA-guided block 식별

Centered Kernel Alignment (CKA) 정의 (Eq. 3):
- CKA(H_i, H_j) = ||H_j^T H_i||_F² / (||H_i^T H_i||_F · ||H_j^T H_j||_F)
- 0~1, orthogonal/scale invariant
- 인접 layer (H_{l-1}, H_l)의 CKA가 ≈1이면 layer l이 거의 변환을 가하지 않는다는 신호

순서:
1. 작은 calibration set D_cal에서 single forward pass
2. 각 layer hidden state 누적 H̄_l 구축
3. s_l = CKA(H̄_{l-1}, H̄_l) 계산 (l=2..L_M)
4. s_l ≥ τ인 연속 layer를 한 block B로 grouping
5. 각 block의 첫 layer(anchor) 유지, 나머지를 candidate pool P_M로 모음
6. R_M = TopK_{l∈P_M}(s_l, k_M) → RemoveLayers(π_θ, R_M)

τ는 |P_M| ≥ k_M이 되도록 자동 조정.

### 2.3 적용 사례 (Table 5)

| Backbone | Module | 원본 layer | 제거 layer | 제거 인덱스 |
|---------|---------|-----------|-----------|------------|
| π0 | VLM + Action expert | 18 | 12 | 1, 2, 4, 6, 8, 9 |
| GR00T-N1.5 | VL-self-attention | 4 | 3 | 2 |
| GR00T-N1.5 | DiT action head | 16 | 8 | 1, 2, 4, 5, 6, 7, 10, 11 |
| SmolVLA | VLM + Action expert | 16 | 10 | 1, 2, 5, 6, 14, 15 |

> ❓ **예상 질문**: π0에서 18 layer 중 12개를 제거하는데도 어떻게 성능이 거의 유지되나?
> **답변**: 표 7의 Table 5 인덱스를 보면 제거된 1,2,4,6,8,9는 모두 "anchor가 아닌 후속 layer". 즉 거의 동일한 representation을 출력하는 "twin" layer들이라는 게 CKA로 사전 정량화된다. 더 결정적으로, fine-tuning이 잔존 layer로 하여금 latent 매니폴드를 재구성(manifold restoration, Figure 1)하게 만든다는 게 PCA 분석에서 확인됨.

---

## 3. 데이터 전략

### 학습/평가 데이터
- **LIBERO**: 4 suite (Spatial, Object, Goal, Long). full fine-tune + 10% subset (low-data 회복력 검증)
- **RoboCasa**: 30 demos / 100 demos × 5 tasks (PnPCabToCounter, PnPCounterToCab, SetUpCoffeeMug, TurnOffStove, TurnOnMicrowave) — 24개 중 일부
- **SimplerEnv WidowX**: Bridge fine-tuning 후 7 tasks
- **Real-world**: 10 tasks × 4 embodiment (UR10, UR5, single-arm ALOHA, bimanual ALOHA), 100~2800 데모

### Calibration set
- 학습 episode에서 sample한 소수 batch
- single forward pass만 필요 (gradient 없음)

> ❓ **예상 질문**: calibration set이 task에 의존하면 generalization이 떨어지지 않나?
> **답변**: 그렇기 때문에 CLP는 task-agnostic redundancy를 잡으려고 CKA representation 통계만 사용한다. 다만 저자도 limitation에서 "modality-specific token dynamics를 명시적으로 고려하지 않는다"고 인정, action/state token 별 redundancy가 다를 수 있음을 미래 과제로 남김.

---

## 4. 시스템/학습 세부사항 (Table 1)

| 항목 | π0 → π0-CLP | GR00T-N1.5 → CLP | SmolVLA → CLP |
|------|--------------|-------------------|----------------|
| Model size | 3.5B → 2.7B (-22.9%) | 2.7B → 2.0B (-25.9%) | 450M → 354M (-21.3%) |
| Trainable params | 3.1B → 2.3B (-25.8%) | 1.07B → 0.75B (-30.1%) | 100M → 63M (-37%) |
| Training time (60k steps) | 15.5h → 11.2h (-27.8%) | 10.7h → 7.4h (-30.8%) | 24.75h → 8.83h (-64.3%) |
| GFLOPs | 3073 → 2196.5 (-28.5%) | 1010 → 512.4 (-49.3%) | 598.4 → 536.1 (-10.41%) |
| Inference latency (RTX 4070) | 211ms → 152ms (-27.9%) | 121ms → 85ms (-29.8%) | 201ms → 137ms (-31.84%) |

- Batch size: 64 (LIBERO 4 GPU), 48 (π0 RoboCasa, 4×H100), 32 (GR00T-N1.5 RoboCasa/SimplerEnv, 1×H100)
- Eval: 50 episodes/task, 10-step execution (LIBERO/RoboCasa), 8-step (SimplerEnv)

---

## 5. 실험 설계 및 평가 프로토콜

4개 연구 질문(RQ):
- **RQ1** (압축 trade-off): 어디까지 layer를 잘라도 성능이 유지되나? — Figure 3-a,b
- **RQ2** (latent 분석 + ablation): CKA가 유일하게 좋은 선택 기준인가? 압축이 latent에 어떤 영향? — Figure 3-d,f
- **RQ3** (baseline 비교): training-free vs training-adaptive 대비 어떤가? — Tab. 2/3, Fig. 3-c
- **RQ4** (real-world): FLOP 절감이 실제 wall-clock 절감으로 이어지나? — Tab. 4, Fig. 3-e

📛 **Real-world 평가 포함**: 10 tasks × 4 embodiment, 단 GR00T-N1.5 기반만 검증.

---

## 6. 실험 결과 심층 분석

### LIBERO Full Fine-tuning (Table 2)

| Method | Spatial | Object | Goal | Long | Avg | Speedup |
|--------|---------|--------|------|------|-----|---------|
| OpenVLA-OFT | 97.6 | 96.5 | 97.9 | 94.5 | 96.6 | 1.00× |
| FastV | 94.6 | 95.8 | 94.0 | 88.8 | 93.3 | 1.44× |
| DivPrune | 92.4 | 91.2 | 89.0 | 84.8 | 89.4 | 1.46× |
| EfficientVLA | 96.5 | 91.1 | 96.0 | 72.1 | 88.9 | 1.52× |
| ADP | 97.6 | 98.4 | 97.4 | 84.2 | 94.4 | 1.35× |
| π0 (base) | 94.6 | 98.2 | 95.4 | 90.0 | 94.6 | 1.00× |
| π0-SpecPrune-VLA | 96.6 | 98.0 | 95.2 | 84.2 | 93.5 | 1.31× |
| **π0-CLP** | **95.0** | **99.2** | **95.0** | **86.4** | **93.9** | **1.39×** |
| GR00T-N1.5 (base) | 90.8 | 98.4 | 95.4 | 91.0 | 93.9 | 1.00× |
| **GR00T-N1.5-CLP** | **89.4** | **98.8** | **95.8** | **88.6** | **93.0** | **1.42×** |
| SmolVLA (base) | 71.8 | 92.2 | 87.4 | 57.2 | 77.15 | 1.00× |
| **SmolVLA-CLP** | **75.6** | **93.0** | **81.6** | **56.2** | **76.75** | **1.47×** |

- 모든 backbone에서 평균 손실 ≤ 1%p, speedup 1.39~1.47×
- token-pruning 계열(FastV, DivPrune, EfficientVLA)보다 평균이 분명히 높고 fine-tuning까지 가속

### LIBERO 10% data, MoLe-VLA 비교 (Table 6, Fig. 3-c)

| Model | Long | Goal | Object | Spatial | Avg | Hours |
|-------|------|------|--------|---------|-----|-------|
| π0 | 58.8 | 87.8 | 82.6 | 81.6 | 77.7 | 15.5 |
| π0-MoLe | 60.2 | 88.2 | 86.0 | 84.4 | 79.7 | 15.6 |
| **π0-CLP** | **66.2** | **90.6** | **89.0** | **92.6** | **84.6** | **11.2** |

- low-data에서 base 대비 **+6.9%p**, 학습 시간은 **-27.7%**
- MoLe-VLA 같이 trainable router를 추가하는 방식보다 단순히 layer를 제거하는 게 더 효과적이라는 흥미로운 결과

### RoboCasa 30 demos (Table 9)

| Model | PnP Cab→Counter | PnP Counter→Cab | Coffee Mug | Turn Off Stove | Turn On Microwave | Avg | Hours |
|-------|----------------|-----------------|------------|----------------|--------------------|-----|-------|
| π0 | 14 | 16 | 2 | 2 | 44 | 15.6 | 17.5 |
| π0-MoLe | 14 | 18 | 2 | 4 | 50 | 17.6 | 17.7 |
| **π0-CLP** | **16** | **16** | **4** | **4** | **50** | **18** | **13.5** |

- low-data RoboCasa에서도 +2.4%p, 23% 학습 시간 절약

### SimplerEnv (Table 3, GR00T-N1.5)

| Model | Carrot | Eggplant Basket | Spoon | Stack Cube | Eggplant Sink | Close Drawer | Open Drawer | Avg | Hours |
|-------|--------|-----------------|-------|------------|---------------|--------------|-------------|-----|-------|
| GR00T-N1.5 | 26 | 34 | 18 | 8 | 8 | 12 | 10 | 16.57 | 22.9 |
| **GR00T-N1.5-CLP** | **34** | 14 | **38** | 4 | **16** | **24** | 10 | **20** | **15.7** |

- 평균 +3.43%p, 학습 시간 -31.4%
- Eggplant Basket와 Stack Cube에서 base보다 낮은 등 task-wise 분산이 큼

### Real-world (Table 4, GR00T-N1.5, 10 tasks)

| Task | GR00T-N1.5 | GR00T-N1.5-CLP |
|------|-----------|----------------|
| Groceries→Basket | 90 | 89 |
| Open Kettle | 100 | 95 |
| Close Kettle | 100 | 100 |
| Serve Napkin | 45 | **65** (+20) |
| Screwdriver→Basket | 15 | **30** (+15) |
| Banana→Pot | 65 | **75** (+10) |
| Cube→Drawer | 75 | 60 |
| Block Stacking | 80 | 75 |
| Fold Shorts | 90 | **95** (+5) |
| Fly Towel | 75 | 70 |
| **Avg** | **73.5** | **75.9** |

- 평균 +2.4%p, 일부 어려운 long-horizon (napkin/screwdriver/banana)에서 큰 향상
- Fig. 3-e 기준 최대 1.94× 학습 속도

> ❓ **예상 질문**: LIBERO full data에선 base보다 살짝 낮은데 어떤 의미가 있나?
> **답변**: 핵심 가치는 (i) 속도 1.39~1.47× (ii) low-data에서 base를 능가 (iii) real-world에서도 평균 향상. full data LIBERO 같이 이미 포화된 환경에서는 base와 동등 수준이 사실상 best case.

---

## 7. Ablation 분석

### Block-selection 기준 (Fig. 3-d, GR00T-N1.5 on LIBERO)
- CKA (제안) — 가장 안정적, baseline에 근접
- MSE — 부분적으로 동작하나 long-horizon에서 불안정
- COSINE — local similarity라 global topology 보존 실패
- RANDOM — 명백한 하한
- KEEP-FIRST (last k 제거) — 일부 backbone에서는 통하나 일반화 X

PCA 시각화 (Fig. 3-f): CKA만 base 모델의 latent topology를 보존, 나머지 기준은 isolated subspace로 distort.

### Pruning ratio (Fig. 3-a,b)
- π0 on LIBERO와 GR00T-N1.5 on RoboCasa 모두 **50% pruning ratio까지 success rate 평탄**
- 50% 초과부터 급격 저하 → 50% 가 실용적 sweet spot

### Manifold restoration (Figure 1 PCA)
- pruning 직후: 잠재 공간이 좁은 subspace로 collapse
- fine-tuning 후: base 모델 분포 근처로 회복
- 이게 "왜 잘 동작하는가"의 핵심 메커니즘

---

## 8. 관련 연구 비교

| 분류 | 방법 | Training-free | Fine-tune 가속 | 추가 모듈 | 적용 대상 |
|------|------|---------------|----------------|------------|------------|
| Token pruning | FastV/DivPrune/EfficientVLA/SpecPrune/ADP/VLA-Cache | ✓ | ✗ | △ | 주로 OpenVLA |
| Light-weight | RoboMamba/FLOWER-VLA/SmolVLA/NORA | — (from scratch) | — | — | 신규 |
| Adaptive routing | DeeR-VLA/MoLe-VLA/AC²-VLA | ✗ | △ | ✓ (router/early exit) | OpenVLA 등 |
| **CLP (이 논문)** | **CKA layer pruning** | **✓ (calibration only)** | **✓ (-28~64%)** | **✗** | **π0/GR00T-N1.5/SmolVLA** |

핵심 차이:
- token이 아닌 **layer** 단위 정적 제거 → 모듈 단순화, 추가 파라미터 0
- inference + training 둘 다 가속
- 현대 continuous-control 모델(π0, GR00T)에 검증 (이전 acceleration 연구는 OpenVLA 중심)

---

## 9. 한계 및 미해결 문제

### 저자 명시
1. **Global pruning criterion**: action/state token 같은 modality-specific dynamics 미고려 → 미래 작업
2. **Post-pretraining fine-tuning만 검증**: pretraining stage에 적용해 layer selection prior로 쓰는 가능성 미탐구

### 평론자 시각
3. **τ 선정 규칙이 모호**: |P_M| ≥ k_M을 만족하도록 조정한다 했지만 구체적 search 절차가 본문에 없음 → 재현성 우려
4. **Real-world는 GR00T-N1.5만**: π0/SmolVLA의 real-world 일반화는 미검증
5. **SimplerEnv task-wise 변동**: avg는 좋지만 Eggplant Basket 34→14, Stack Cube 8→4처럼 일부 task는 분명히 후퇴. low-precision dexterous task에서 layer 제거가 부정적일 수 있다는 신호
6. **Calibration set 크기/구성의 민감도 분석 부재**: D_cal이 얼마나 작아도 되는지, task 분포가 바뀌면 제거 인덱스가 어떻게 바뀌는지 plot 없음
7. **Anchor=첫 layer 휴리스틱의 정당성**: 왜 block의 first layer가 가장 중요한지에 대한 quantitative 분석 부족 (intuition만)

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — CKA 자체는 알려진 지표지만 modern continuous-control VLA에 layer-level 정적 압축을 적용한 깔끔한 통합 |
| **Technical depth** | ★★★★☆ — calibration → CKA → block aggregation → fine-tune의 단순함이 오히려 강점. Algorithm 1로 명료 |
| **Experimental rigor** | ★★★★☆ — 3 backbone × 3 simulator × 10 real-world. token-pruning 5개와 routing-based MoLe 동시 비교 |
| **Practical impact** | ★★★★★ — auxiliary module 0개, fine-tuning까지 가속, 4070급 GPU에서도 inference 30% 빠름. 즉시 채택 가능 |
| **Writing quality** | ★★★★☆ — RQ 구조와 figure 흐름이 분명. ablation이 풍부 |

**강점**: (1) inference + training 양쪽 가속 (2) low-data 환경에서 base 능가 (구조적 regularizer 효과) (3) 추가 학습 모듈 없음 → MoLe/DeeR과 차별 (4) 3개 backbone에 일관 적용 (5) real-world까지 검증.

**약점**: SimplerEnv task별 분산, τ/D_cal 민감도 분석 부재, anchor=first layer 휴리스틱의 정당성, real-world가 GR00T-N1.5에 한정.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 왜 CKA가 다른 similarity(MSE, cosine)보다 layer pruning에 좋은가? | CKA는 orthogonal/scale invariant라 representation의 functional 동치성을 잡는다. cosine은 magnitude에 민감, MSE는 absolute distance라 layer 간 학습된 scale 차이에 약함. Fig. 3-d,f에서 PCA로 CKA만 base topology 보존 확인 |
| 2 | LIBERO full에서 base보다 0.7~0.9%p 낮은데 trade-off로 정당한가? | (i) 1.39~1.47× 속도, (ii) 학습시간 -27~31%, (iii) 동등급의 token-pruning 방법(EfficientVLA 88.9)보다 5%p 높음, (iv) low-data 84.6 vs base 77.7로 base를 능가. 종합적으로 favorable |
| 3 | τ는 어떻게 정하나? | |P_M| ≥ k_M을 만족하도록 calibrate. 다만 본문에 grid search 등 구체적 절차 명시 X — 재현성 측면 약점 |
| 4 | calibration set이 작으면 pruning 결정이 바뀌나? | 본문에 명시적 분석 없음. CKA는 representation 통계라 대수의 법칙으로 어느 정도 안정적일 것으로 기대되나 검증 부재 |
| 5 | π0 18 layer 중 12개를 제거해도 되나? 왜 6 layer만 남기지 않나? | Table 5 — VLM + Action expert 합쳐 18 중 12 제거, 즉 6 layer 잔존. Fig. 3-a에서 50% 이상 제거 시 급격 저하라 6/18 = 33% 잔존은 limit 근처. 그래서 평균 0.7%p 손실은 한계점에서의 trade-off로 해석 |
| 6 | MoLe-VLA(dynamic layer skip)보다 단순 제거가 왜 더 효과적인가? | (i) MoLe는 router를 학습해야 → 추가 파라미터/학습 부담, (ii) low-data에서 router가 과적합, (iii) CLP는 정적이라 inference path 안정. Fig. 3-c에서 84.6 vs 79.7로 5%p 차이 |
| 7 | Real-world에서 일부 task는 실패하는데(Open Kettle -5) 어떤 패턴? | 정확한 pinpoint는 없지만, 짧고 정밀한 task(open kettle, stack block)에서 미세 후퇴, long-horizon/조작 다양성 큰 task(napkin, screwdriver, banana)에서 큰 향상. 즉 capacity 감소가 over-specialization을 줄이는 regularizer로 작용 |
| 8 | π0 vs GR00T-N1.5 어느 쪽 압축에 더 적합? | trainable param 절감은 GR00T-N1.5(-30.1%)가 더 큼. GFLOPs 절감도 -49.3% vs -28.5%. DiT action head가 매우 redundant (16 → 8)라는 발견이 흥미. GR00T-N1.5가 architecturally 더 over-parameterized |
| 9 | Anchor=block 첫 layer 휴리스틱은 왜 합리적? | 저자 직관: 첫 layer가 block의 input representation을 형성하므로 functional anchor. 다만 quantitative 검증 없음 — middle layer 유지 vs first layer 유지 ablation 필요 |
| 10 | OpenVLA(autoregressive)에는 적용 가능한가? | 본문에서 다루지 않음. CKA는 transformer 일반에 적용 가능하나 autoregressive 디코딩 path에서는 layer skip이 sequential dependency를 깨뜨릴 수 있음. 향후 연구 과제 |

<!-- VERIFIED: pdf -->
