# HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist VLA Policies

> **한 줄 요약**: 로봇 데이터의 다층적 이질성(embodiment, action space, sensor, frequency)을 계층적 MoE로 계층별 적응적 처리하여, heterogeneous cross-embodiment 학습에서 일관적 성능 향상 달성.

---

## 1. 배경 및 동기

- Cross-embodiment VLA (Octo, CrossFormer)는 **이질적 데이터를 단일 네트워크로 처리** → 서로 다른 embodiment 간 interference
- 이질성이 **다층적**: embodiment 차이, action space 차이, sensor 차이, control frequency 차이
- 기존 MoE는 단일 수준의 routing만 수행 → 다층적 이질성 미처리

---

## 2. 방법론 심층 분석

### 2.1 Hierarchical MoE

각 transformer layer에서 **다중 수준의 routing**:

```
Layer l:
  Router 1: Embodiment-level routing → expert subset 선택
  Router 2: Task-level routing → expert subset 내에서 세부 선택
  Shared experts: 모든 입력에 공통 적용
```

$$h_l = \sum_{i \in \text{shared}} E_i^s(x) + \sum_{j \in \text{selected}} g_j(x) \cdot E_j^{h}(x)$$

> ❓ **예상 질문**: Hierarchical routing의 overhead는? 일반 MoE 대비 얼마나 추가 연산인가?
> **답변**: Routing 자체는 경량 linear projection이므로 overhead 미미. 다만 expert 수 증가에 따른 메모리 사용량이 단일 MoE 대비 높을 수 있음.

### 2.2 Gradual Abstraction

하위 layer: embodiment-specific features 처리 (sensor alignment, action space normalization)
상위 layer: task-agnostic shared knowledge (물체 인식, 공간 추론)

> ❓ **예상 질문**: 이 abstraction 패턴이 자연스럽게 emerge하는가, 아니면 강제하는가?
> **답변**: Routing이 학습을 통해 자동으로 결정되므로 emerge하는 것으로 주장. 그러나 이 패턴의 발현을 정량적으로 검증하는 분석(예: router decision 시각화)은 부분적.

---

## 3. 실험 결과 심층 분석

| 모델 | Cross-embodiment 평균 SR (%) |
|------|---------------------------|
| Octo | 55.3 |
| CrossFormer | 61.7 |
| OpenVLA (single) | 68.5 |
| **HiMoE-VLA** | **73.2** |

- Cross-embodiment에서 일관적 향상
- 특히 **이질성이 큰 조합** (manipulator + mobile)에서 개선폭이 큼

---

## 4. 한계 및 미해결 문제

1. **Expert 수의 scalability**: Embodiment가 수십 개로 늘어나면 expert 수 폭발
2. **새로운 embodiment 추가**: 기존 routing을 재학습해야 하는지, 새 expert만 추가하면 되는지 불명확
3. **Routing의 해석가능성**: 어떤 expert가 어떤 embodiment에 특화되는지의 분석 부족
4. **Standard benchmark 결과**: LIBERO/CALVIN 등 단일 embodiment 벤치마크에서의 경쟁력 미제시

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Hierarchical MoE for heterogeneous robotics |
| **Technical depth** | ★★★★☆ — 다층 routing 설계 |
| **Experimental rigor** | ★★★★☆ — Cross-embodiment 포괄적 평가 |
| **Practical impact** | ★★★★☆ — Multi-robot deployment에 직접적 가치 |
| **Writing quality** | ★★★★☆ |

**강점**: 로봇 데이터의 다층적 이질성을 구조적으로 다루는 최초의 체계적 접근. **약점**: Scalability 우려, 새 embodiment 추가의 용이성 미검증.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Embodiment 수가 100개면 expert 몇 개 필요한가? | Linear scaling이면 비실용적. Expert sharing/clustering이 필요하나 미탐구 |
| 2 | 단일 MoE (hierarchical 아닌)와의 차이는 얼마인가? | Ablation 포함. Hierarchical이 flat MoE 대비 ~3%p 향상 |
| 3 | Load balancing은 어떻게 하는가? | Auxiliary loss로 balanced routing 유도하나, 특정 embodiment에 expert가 편중될 수 있음 |
