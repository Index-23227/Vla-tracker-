# SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation

> **한 줄 요약**: SAM2의 visual foundation model 표현을 multi-resolution upsampling으로 활용하고, memory mechanism을 추가하여 RLBench 86.8% 및 memory-dependent 태스크에서 94.3% (MemoryBench) 달성.

---

## 1. 배경 및 동기

- Visual foundation model (SAM, DINOv2)의 풍부한 표현을 로봇 정책에 직접 활용
- **Memory**: 이전 관측을 기억하여 partially observable task 해결 (물체가 가려졌다가 다시 나타나는 경우 등)

---

## 2. 방법론

### SAM2 Features + Multi-resolution Upsampling

SAM2의 다양한 해상도 feature를 fusion하여 fine-grained spatial 정보 보존.

### Memory Mechanism (SAM2Act+)

이전 timestep의 feature를 memory bank에 저장하고, 현재 관측과 cross-attention:

$$\mathbf{h}_t = \text{CrossAttn}(\mathbf{f}_t, \text{MemBank}(\mathbf{f}_{t-1}, ..., \mathbf{f}_{t-K}))$$

---

## 3. 실험 결과

| 모델 | RLBench (%) | MemoryBench (%) |
|------|-----------|----------------|
| PerAct | 49.4 | 52.1 |
| 3D Diffuser Actor | 76.3 | 65.3 |
| **SAM2Act** | **86.8** | **78.5** |
| **SAM2Act+** (memory) | **86.8** | **94.3** |

- Memory 추가로 MemoryBench에서 +16%p 향상

---

## 4. 한계 및 미해결 문제

1. **SAM2 의존**: SAM2가 실패하는 시각적 조건(투명 물체, 극단적 조명)에서 취약
2. **Memory 관리**: Memory bank 크기와 retention policy의 최적화 필요
3. **RLBench 특화**: 다른 벤치마크에서의 일반성 미검증
4. **Sim-only**: 실제 환경 미테스트

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — SAM2 + memory for robots |
| **Practical impact** | ★★★★☆ — Memory-dependent task 해결 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | SAM2 대신 DINOv2를 쓰면? | DINOv2도 강력하나 SAM2의 segmentation-aware feature가 manipulation에 유리 |
| 2 | Memory가 noise로 오염되면? | Attention으로 자연스럽게 irrelevant memory 무시. 그러나 adversarial noise에 대한 robustness 미검증 |
