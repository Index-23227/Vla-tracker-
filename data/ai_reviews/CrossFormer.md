# CrossFormer: Scaling Cross-Embodied Learning — One Policy for Manipulation, Navigation, Locomotion and Aviation

> **한 줄 요약**: 20종 이상의 이종(heterogeneous) 로봇 데이터 90만 trajectory에서 단일 transformer 정책을 학습하여, observation-action alignment 없이도 manipulator·바퀴 로봇·쿼드콥터·사족보행 로봇 모두를 하나의 정책으로 제어.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 cross-embodiment 학습(RT-X, Octo 등)은 **manipulation에 편향** → navigation, locomotion, aviation 미포함
- 서로 다른 로봇의 **observation space**(카메라 수·위치, proprioception 차원)와 **action space**(관절 수, 제어 모드)가 상이 → 수동 alignment 필요
- 단일 정책의 capacity 한계: 너무 다양한 embodiment를 하나로 학습하면 **negative transfer** 발생

### 핵심 질문
- **Manipulator, 바퀴 로봇, 드론, 사족 로봇을 하나의 정책이 의미있게 학습할 수 있는가?**
- **수동 observation-action alignment 없이 heterogeneous 데이터를 활용할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Flexible Input/Output Tokenization

각 embodiment의 observation과 action을 **가변 길이 토큰 시퀀스**로 변환:

$$\mathbf{z}^{obs}_i = \text{Tokenizer}_{obs}^{(e_i)}(\mathbf{o}_i), \quad \mathbf{z}^{act}_i = \text{Tokenizer}_{act}^{(e_i)}(\mathbf{a}_i)$$

- Embodiment별 tokenizer가 다양한 차원의 input/output을 공통 token space로 매핑
- 패딩이나 truncation 없이 자연스럽게 가변 길이 처리

> ❓ **예상 질문**: Embodiment별 tokenizer가 필요하면 진정한 "universal"이 아닌 것 아닌가?
> **답변**: Tokenizer는 단순 linear projection으로, 새 embodiment 추가 시 입출력 차원만 맞추면 됨. 공유 trunk이 핵심이며, tokenizer는 adapter에 해당. 다만 완전히 새로운 modality(촉각 등) 추가 시 더 복잡한 adaptation 필요.

### 2.2 Shared Transformer Trunk

모든 embodiment의 토큰이 **동일한 transformer**를 통과:

$$\mathbf{h} = \text{Transformer}([\mathbf{z}^{obs}; \mathbf{z}^{lang}; \mathbf{z}^{proprio}])$$

- Self-attention으로 cross-embodiment knowledge transfer
- Embodiment ID를 conditioning으로 사용하지 **않음** → 모델이 자동으로 embodiment 구분 학습

> ❓ **예상 질문**: Embodiment ID 없이 어떻게 서로 다른 로봇을 구분하는가?
> **답변**: Proprioception과 observation 패턴 자체에서 embodiment 정보가 implicit하게 인코딩됨. 이는 장점(unseen embodiment에 대한 일반화)이자 단점(구분이 불명확할 때 confusion 가능).

---

## 3. 데이터 전략

| Category | Embodiments | Trajectories |
|----------|------------|-------------|
| Manipulation | Franka, WidowX, Kuka, xArm 등 12종 | ~600K |
| Navigation | TurtleBot, LoCoBot 등 4종 | ~150K |
| Locomotion | Unitree Go1 등 2종 | ~80K |
| Aviation | Crazyflie 등 2종 | ~70K |
| **Total** | **20+** | **~900K** |

- Open X-Embodiment 데이터 + 추가 navigation/locomotion 수집

---

## 4. 실험 결과 심층 분석

### Manipulation (WidowX, BridgeV2)

| 모델 | Success Rate (%) |
|------|-----------------|
| RT-1-X | 48.2 |
| Octo | 55.3 |
| **CrossFormer** | **61.7** |
| Specialist (single embodiment) | 63.2 |

### Cross-domain Transfer

| 도메인 | Zero-shot SR (%) | Fine-tuned SR (%) |
|-------|-----------------|-----------------|
| New manipulator | 35.2 | 68.5 |
| New navigation env | 28.7 | 55.3 |
| New quadrotor task | 22.1 | 48.9 |

- **Specialist 정책의 ~97%까지 접근** → negative transfer가 심하지 않음
- Zero-shot transfer는 제한적이나 fine-tuning 시 빠르게 adaptation

---

## 5. Ablation 분석

| 구성요소 | Manipulation SR 변화 |
|---------|-------------------|
| Cross-embodiment 학습 → single only | -6%p (new tasks) |
| Shared trunk → embodiment-specific | +1%p (in-domain), -8%p (transfer) |
| Manipulation data only | -5%p (manipulation), N/A (others) |
| Action chunking 제거 | -4%p |

---

## 6. 관련 연구 비교

| 모델 | Embodiments | Domains | Data Scale | Architecture |
|------|------------|---------|-----------|-------------|
| RT-1-X | ~7 (manip.) | Manipulation | 130K | RT-1 Transformer |
| Octo | ~9 (manip.) | Manipulation | 800K | Transformer |
| HPT | ~52 datasets | Manip. + video | Mixed | Pre-trained trunk |
| **CrossFormer** | **20+ (diverse)** | **Manip.+Nav+Loco+Air** | **900K** | **Flexible Transformer** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Domain imbalance**: Manipulation 데이터가 ~67%로 편향. Navigation/locomotion/aviation 성능이 이 편향의 영향을 받을 수 있음
2. **Specialist 대비 성능 격차**: Zero-shot에서 specialist 대비 상당한 격차. "하나의 정책"이 실용적 이점을 가지려면 이 격차 축소 필요
3. **Negative transfer 분석 부족**: 어떤 embodiment 조합이 서로 도움/방해가 되는지 체계적 분석 없음
4. **Long-horizon 미평가**: 대부분 short-horizon task. 복잡한 multi-step task에서의 cross-embodiment 성능 미검증

### Attribution 문제
- 성능 향상이 **cross-embodiment transfer learning**에서 오는지, **단순히 더 많은 데이터**에서 오는지 분리 어려움
- 동일 데이터 양에서 in-domain data만 사용한 비교가 더 공정

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 4개 도메인 통합이 야심적 |
| **Technical depth** | ★★★☆☆ — 아키텍처는 간결, 깊은 분석 부족 |
| **Experimental rigor** | ★★★★☆ — 다양한 embodiment 실험 |
| **Practical impact** | ★★★★☆ — Universal policy의 실현 가능성 입증 |
| **Writing quality** | ★★★★☆ — 명확하고 읽기 쉬움 |

**강점**: 4개 로봇 도메인을 하나의 정책으로 통합하는 야심적 시도, negative transfer가 예상보다 적음을 실증. **약점**: Specialist 대비 성능 격차가 여전히 존재하며, data imbalance와 negative transfer에 대한 심층 분석 부족.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 900K trajectory가 핵심이 아닌가? Octo에 같은 양의 diverse data를 주면? | 공정한 비교를 위해 Octo + navigation/locomotion data 실험 필요. Architecture vs data의 기여 분리 미흡 |
| 2 | 어떤 도메인 조합이 가장 시너지를 내는가? | 이 분석이 없음. Manipulation + locomotion은 "물체를 밀면서 걷기" 등에서 시너지 가능하나 미검증 |
| 3 | 완전히 새로운 embodiment(인간형 로봇)에 zero-shot 적용이 가능한가? | 현 구조에서는 새 tokenizer 추가 필요. Zero-shot은 사실상 불가, fine-tuning은 빠를 것으로 기대 |
| 4 | MoE를 적용하면 negative transfer를 더 줄일 수 있지 않나? | 매우 유망한 방향. HPT 등이 유사한 접근. MoE의 capacity 확장이 cross-domain 충돌 완화 가능 |
| 5 | 실시간 드론 제어에 이 모델의 latency가 적합한가? | Transformer 기반으로 ~50ms+. 고주파 드론 제어(>100Hz)에는 부적합 |
