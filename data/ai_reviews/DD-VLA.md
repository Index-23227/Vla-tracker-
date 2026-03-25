# DD-VLA: Discrete Diffusion VLA — Bringing Discrete Diffusion to Action Decoding

> **한 줄 요약**: 기존 VLA의 autoregressive 또는 continuous diffusion 방식 대신, **이산 확산(discrete diffusion)**으로 action chunk를 생성하여 적응적 디코딩 순서와 오류 교정을 지원하고 LIBERO 96.3% 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Autoregressive VLA** (OpenVLA, RT-2): 왼쪽→오른쪽 순서로 action token 생성 → 초기 토큰 오류가 후속 토큰에 전파(exposure bias)
- **Continuous diffusion VLA** (CogACT, pi0): 고품질 action 생성 가능하지만, VLM의 discrete token 공간과 action의 continuous 공간 간 **아키텍처적 단절** → 별도 diffusion head 필요
- 두 패러다임을 통합하는 시도 부재: discrete token 기반이면서도 diffusion의 iterative refinement 이점을 갖는 방법

### 핵심 질문
- **Discrete diffusion이 autoregressive와 continuous diffusion의 장점을 모두 취할 수 있는가?**
- **비순차적(non-sequential) 디코딩이 action 생성에 실질적 이점이 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Discrete Diffusion for Actions

Action chunk를 discretized token sequence로 표현하고, **마스킹 기반 discrete diffusion**으로 생성:

$$q(\mathbf{a}_t | \mathbf{a}_0) = \text{Cat}(\mathbf{a}_t; (1-\beta_t)\mathbf{a}_0 + \beta_t \mathbf{m})$$

여기서 $\mathbf{m}$은 [MASK] token, $\beta_t$는 마스킹 스케줄.

Reverse process:
$$p_\theta(\mathbf{a}_0 | \mathbf{a}_t, \mathbf{c}) = \prod_{i} p_\theta(a_0^{(i)} | \mathbf{a}_t, \mathbf{c})$$

> ❓ **예상 질문**: Discrete diffusion과 BERT-style masked prediction의 차이는?
> **답변**: 핵심 차이는 **iterative refinement**. BERT는 한 번에 모든 [MASK]를 예측하지만, discrete diffusion은 여러 step에 걸쳐 점진적으로 unmask → 이전 step의 예측을 조건으로 다음 예측이 개선됨. 이것이 "diffusion"의 본질.

### 2.2 Adaptive Decoding Order

Autoregressive와 달리, **모델이 자체적으로 디코딩 순서를 결정**:
- Confidence가 높은 토큰을 먼저 unmask
- 어려운(불확실한) 토큰은 나중에, 더 많은 context를 활용하여 예측

$$\text{order} = \text{argsort}(-\text{confidence}(\hat{a}_0^{(i)}))$$

> ❓ **예상 질문**: 적응적 순서가 고정 순서(왼→오른) 대비 정말 유리한가? 어떤 task에서 차이가 큰가?
> **답변**: Ablation에서 적응적 순서가 고정 순서 대비 ~2%p 향상. Bimanual task처럼 양팔의 동기화가 중요한 경우, 한 팔의 action을 먼저 결정하고 다른 팔을 조건부로 예측하는 것이 유리할 수 있음.

### 2.3 Secondary Re-masking for Error Correction

생성 후 **low-confidence token을 다시 마스킹하고 재생성**:

$$\mathbf{a}' = \text{ReMask}(\hat{\mathbf{a}}_0, \text{threshold}=\tau) \to \hat{\mathbf{a}}'_0 = p_\theta(\mathbf{a}_0 | \mathbf{a}')$$

> ❓ **예상 질문**: Re-masking이 수렴하지 않고 oscillate할 수 있지 않은가?
> **답변**: 이론적으로 가능하나, 실험에서 1-2회 re-masking으로 충분. Re-masking 횟수와 threshold의 trade-off는 ablation에서 다뤄지나 수렴 보장에 대한 이론적 분석은 없음.

---

## 3. 데이터 전략

- 기존 VLA 학습 데이터를 그대로 활용
- Action discretization: Uniform binning (256 bins per dimension)
- 연속 action → discrete token 변환의 quantization error는 bin 수로 조절

---

## 4. 실험 결과 심층 분석

| 모델 | LIBERO Avg (%) | SimplerEnv Avg (%) |
|------|---------------|-------------------|
| OpenVLA (AR) | 76.5 | 48.7 |
| OpenVLA-OFT (parallel) | 84.2 | 53.1 |
| CogACT (cont. diffusion) | 89.7 | 67.8 |
| **DD-VLA (discrete diffusion)** | **96.3** | **72.5** |

- LIBERO에서 **96.3%**는 near-saturation 수준
- SimplerEnv에서도 continuous diffusion(CogACT) 대비 우위

### Parallel Decoding

| 디코딩 방식 | LIBERO (%) | Wall-clock (ms) |
|-----------|-----------|----------------|
| Full iterative (T=10) | 96.3 | 120 |
| Accelerated (T=5) | 95.8 | 65 |
| 1-step (parallel) | 93.2 | 15 |

---

## 5. Ablation 분석

| 구성요소 | LIBERO 변화 |
|---------|-----------|
| Discrete diffusion → Autoregressive | -8%p |
| Discrete diffusion → Continuous diffusion | -3%p |
| Adaptive order → Fixed L→R | -2%p |
| Re-masking 제거 | -1.5%p |
| 256 bins → 64 bins | -2%p |

---

## 6. 관련 연구 비교

| 모델 | Token Type | Generation | Error Correction | Unified Architecture |
|------|-----------|-----------|-----------------|---------------------|
| OpenVLA | Discrete | Autoregressive | ✗ | ✓ |
| CogACT | Continuous | Continuous diffusion | Implicit (denoising) | ✗ (separate head) |
| **DD-VLA** | **Discrete** | **Discrete diffusion** | **✓ (re-masking)** | **✓** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Quantization error**: 256-bin discretization은 action space의 precision을 ~1/256로 제한. 고정밀 task(삽입, 조립)에서 이 quantization error가 dominant error source가 될 수 있음
2. **Iterative decoding latency**: Full 10-step에서 120ms → 10Hz 제어에는 적합하나, 더 높은 주파수에서는 병목. 1-step으로 줄이면 3%p 성능 하락
3. **Discrete diffusion의 이론적 한계**: Continuous diffusion의 score matching에 비해 discrete diffusion의 수렴 보장이 약함
4. **Real-robot 결과 미포함**: LIBERO, SimplerEnv 모두 시뮬레이션

### Attribution 문제
- 96.3%라는 높은 수치가 **discrete diffusion** 덕분인지, **학습 레시피·데이터 전처리** 최적화 때문인지 분리 어려움
- CogACT와의 비교에서 모델 크기, 학습 데이터, epoch이 완전히 동일한지 불명확

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Discrete diffusion의 VLA 적용이 매우 참신 |
| **Technical depth** | ★★★★☆ — Re-masking, adaptive order 등 깊은 설계 |
| **Experimental rigor** | ★★★☆☆ — Sim-only, LIBERO near-saturation |
| **Practical impact** | ★★★★☆ — Unified architecture의 실용적 가치 |
| **Writing quality** | ★★★★☆ — 깔끔한 구조 |

**강점**: Autoregressive와 diffusion의 장점을 결합하는 새로운 패러다임 제시. Error correction이 구조적으로 가능. **약점**: Sim-only 평가, quantization error의 근본적 한계, 실제 배포에서의 검증 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 256 bins이면 precision이 ~0.4%인데, 이것으로 mm 단위 정밀도가 가능한가? | Action range에 따라 다름. ±1m range면 ~7.8mm resolution. 정밀 삽입(~1mm)에는 부족 |
| 2 | Continuous diffusion 대비 discrete diffusion의 이론적 장점은? | Unified token space(LLM과 동일), error correction 가능, parallel decoding. 다만 precision 손실 |
| 3 | LIBERO 96.3%는 saturation 아닌가? 차별화 가능한 더 어려운 벤치마크에서의 결과는? | 맞음. LIBERO-Long이나 real-robot에서의 차이가 더 의미 있을 것 |
| 4 | Re-masking을 무한 반복하면 성능이 계속 올라가는가? | 실험에서 2회 이후 수렴. Diminishing returns + latency 증가 |
| 5 | 이 방법이 language token 생성에도 적용 가능한가? | 이론적으로 가능(masked language model). VLA에서 언어 이해와 action 생성을 동일 프레임워크로 통합 가능 |
