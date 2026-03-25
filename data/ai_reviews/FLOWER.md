# FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies

> **한 줄 요약**: LLM layer를 50% pruning하고 그 capacity를 diffusion head에 재할당하는 "intermediate-modality fusion"과 Global-AdaLN conditioning으로, 950M 파라미터의 효율적 VLA를 구축하여 CALVIN ABC에서 4.53 SOTA 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 VLA (7B+)는 **학습 비용이 극도로 높음** (수천 GPU-hours) → 연구 접근성 제약
- VLM backbone의 모든 layer가 action prediction에 필요한 것은 아님 → **compute 낭비**
- Diffusion/flow action head에 충분한 capacity가 할당되지 않음 (VLM 대비 매우 작음)

### 핵심 질문
- **VLM의 후반부 layer를 pruning하고 그 capacity를 action head에 재할당하면, 총 파라미터는 줄이면서 성능은 올릴 수 있는가?**
- **1B 미만 VLA로 7B 모델과 경쟁할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Intermediate-Modality Fusion

VLM의 중간 layer에서 hidden state를 추출하고 이후 layer를 제거:

$$\mathbf{h}_{\text{mid}} = \text{VLM}_{1:L/2}(\mathbf{o}, \mathbf{l}), \quad \text{VLM}_{L/2+1:L} \text{ 제거}$$

"중간" representation이 action prediction에 충분한 정보를 담고 있다는 가설.

> ❓ **예상 질문**: 왜 중간이 충분한가? 후반부 layer가 더 추상적이고 유용하지 않은가?
> **답변**: VLM의 후반부 layer는 주로 **language generation**에 특화 → action prediction에는 과잉. 중간 layer의 representation이 시각적·공간적 정보를 더 직접적으로 보유. 이는 ViT에서도 중간 layer가 가장 풍부한 spatial feature를 가진다는 연구와 일맥상통.

### 2.2 Global-AdaLN Conditioning

Action-specific Adaptive Layer Normalization:

$$\text{AdaLN}(\mathbf{x}; \gamma, \beta) = \gamma \cdot \text{LN}(\mathbf{x}) + \beta, \quad (\gamma, \beta) = \text{MLP}(\mathbf{h}_{\text{mid}})$$

- 기존 cross-attention 대비 **파라미터 20% 절감**
- VLM representation을 diffusion head의 각 layer에 주입

> ❓ **예상 질문**: AdaLN이 cross-attention보다 정보 전달 capacity가 작지 않은가?
> **답변**: AdaLN은 global conditioning (전체에 동일 scale/shift), cross-attention은 token-level conditioning. 복잡한 spatial conditioning에서는 cross-attention이 유리하나, action prediction에서는 global conditioning이 충분할 수 있음. Ablation에서 AdaLN ≥ cross-attention 성능.

### 2.3 950M Architecture

| 구성요소 | 파라미터 |
|---------|---------|
| Visual encoder (SigLIP) | ~400M |
| VLM (pruned, 50%) | ~350M |
| Flow action head | ~200M |
| **Total** | **~950M** |

학습: **200 H100 GPU-hours** (OpenVLA의 ~1/50)

---

## 3. 실험 결과 심층 분석

### CALVIN ABC→D

| 모델 | Params | Avg. Len | Training Cost |
|------|--------|---------|---------------|
| GR-1 (1.5B) | 1.5B | 3.86 | 2000 GPU-hrs |
| OpenVLA (7B) | 7B | 3.45 | 10000 GPU-hrs |
| **FLOWER (950M)** | **950M** | **4.53 (SOTA)** | **200 GPU-hrs** |

- **950M으로 CALVIN SOTA** (4.53 avg len)
- **학습 비용 50배 절감** (OpenVLA 대비)

### 10 Benchmarks 종합

| 모델 | 10-bench 평균 SR (%) |
|------|---------------------|
| OpenVLA | 62.3 |
| Octo | 58.7 |
| **FLOWER** | **68.5** |

---

## 4. Ablation 분석

| 구성요소 | CALVIN Avg Len |
|---------|---------------|
| Full FLOWER | 4.53 |
| Full VLM (no pruning) | 4.21 |
| VLM 25% pruning | 4.38 |
| VLM 75% pruning | 4.15 |
| AdaLN → cross-attention | 4.48 |
| Flow head → MLP | 3.85 |

- **50% pruning이 최적** — full VLM보다 오히려 나음 (+0.32)!
- 이는 pruning이 regularization 효과를 내거나, 절감된 capacity가 action head에 더 유용하게 쓰임을 시사

---

## 5. 관련 연구 비교

| 모델 | Params | Efficient | Action Head | Training Cost |
|------|--------|----------|-------------|--------------|
| OpenVLA | 7B | ✗ | Token pred | ~10K GPU-hrs |
| Octo | 93M | ✓ | Diffusion | ~500 GPU-hrs |
| CogACT | 7B | ✗ | DiT | ~5K GPU-hrs |
| **FLOWER** | **950M** | **✓** | **Flow DiT** | **200 GPU-hrs** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **Pruning 최적 지점의 일반성**: 50%가 항상 최적인지 불명확. 태스크·데이터에 따라 최적 pruning ratio가 변할 수 있음
2. **VLM 능력 보존**: 50% layer 제거 시 language understanding, spatial reasoning 등 VLM 능력이 어디까지 보존되는지 체계적 평가 부재
3. **대규모 데이터 scaling**: 200 GPU-hours는 small-scale 설정. 대규모 데이터에서 pruned model의 scaling behavior 미검증
4. **Real-robot 결과의 깊이**: 다수 벤치마크에서 테스트하나, 각 벤치마크당 결과의 상세 분석이 부족

### Attribution 문제
- "Pruning이 성능을 높인다"는 결과가 **capacity 재분배** 덕분인지, **overfitting 방지** 덕분인지, **학습 최적화 용이성** 덕분인지 분리 어려움
- 동일 총 파라미터(950M)에서 VLM과 action head 비율을 다르게 한 비교가 더 설득력 있을 것

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — VLM pruning + capacity 재분배라는 반직관적 접근 |
| **Technical depth** | ★★★★☆ — 체계적 설계와 ablation |
| **Experimental rigor** | ★★★★★ — 10 벤치마크, 다양한 embodiment |
| **Practical impact** | ★★★★★ — Democratization of VLA 연구 |
| **Writing quality** | ★★★★★ — 명확하고 설득력 있음 |

**강점**: "Less VLM, more action head"라는 역발상이 실험적으로 입증됨. 950M/200 GPU-hrs로 SOTA는 연구 접근성 혁명. **약점**: Pruning의 이론적 근거가 부족, large-scale scaling 미검증.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VLM을 80% pruning하면? 한계점은 어디인가? | 75% pruning에서 성능 하락 시작. 최소한의 VLM capacity threshold 존재하나 정확한 정의 어려움 |
| 2 | Pruning 대신 처음부터 작은 VLM + 큰 action head로 학습하면? | 이 비교가 핵심. Pre-trained VLM의 knowledge transfer vs from-scratch의 차이를 분리해야 함 |
| 3 | CALVIN 4.53이 다른 VLM backbone에서도 재현되는가? | 현재 특정 VLM (SigLIP + Phi-2 계열)에서만 검증. 다른 backbone에서의 일반성 미확인 |
| 4 | 200 GPU-hours 학습으로 long-horizon dexterous task까지 해결 가능한가? | 현재는 standard manipulation 위주. Complex task에는 더 많은 데이터·compute 필요할 가능성 |
| 5 | Flow matching vs diffusion: FLOWER에서 flow를 선택한 이유는? | Flow matching이 적은 step으로 고품질 생성 가능 → 효율성 철학과 일치 |
