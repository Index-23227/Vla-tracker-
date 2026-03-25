# RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation

> **한 줄 요약**: Mamba state space model을 VLA backbone으로 활용하여, Transformer 대비 3배 빠른 추론과 우수한 robotic reasoning을 최소한의 파라미터 fine-tuning으로 달성.

---

## 1. 배경 및 동기

- Transformer의 **O(n²) attention**이 long-context robot observation에서 bottleneck
- **Mamba (SSM)**: O(n) 복잡도, 긴 시퀀스 효율적 처리 → VLA에 적합할 가능성
- 기존 VLA의 추론 속도 개선 필요

---

## 2. 방법론

### Mamba-based VLA

Transformer 대신 **Mamba block**으로 vision-language-action 처리:
- Linear time complexity: observation history가 길어져도 latency 일정
- Selective state space: 관련 정보를 선택적으로 보존

> ❓ **예상 질문**: Mamba의 단방향(unidirectional) 특성이 spatial reasoning에 불리하지 않은가?
> **답변**: 맞음. Mamba는 causal (왼→오른) 처리에 특화. 이미지의 2D spatial 관계를 1D sequence로 처리하면 정보 손실. Bidirectional Mamba나 2D scanning이 대안이나 구현 복잡성 증가.

---

## 3. 실험 결과

| 모델 | Reasoning Score | Manipulation SR | Inference Speed |
|------|---------------|----------------|----------------|
| OpenVLA (7B Transformer) | 65.3 | 76.5% | 1x |
| **RoboMamba** | **68.7** | **74.2%** | **3x** |

- **3x 빠른 추론**이면서 reasoning은 오히려 약간 우수
- Manipulation은 소폭 하락 (-2.3%p) → speed-accuracy trade-off

---

## 4. 한계 및 미해결 문제

1. **Spatial reasoning**: Mamba의 1D 특성으로 2D/3D spatial reasoning에서 Transformer 대비 약할 수 있음
2. **커뮤니티 성숙도**: Mamba가 아직 Transformer만큼 검증되지 않음
3. **Action head**: Mamba backbone + diffusion head 결합의 최적성 미탐구
4. **Manipulation 성능 소폭 하락**: 속도 이점이 이 하락을 상쇄하는지 태스크에 따라 다름

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — SSM의 VLA 적용 개척 |
| **Practical impact** | ★★★★☆ — 추론 속도 3x |

**강점**: Mamba의 로봇 적용 가능성 입증, 3x 속도 향상. **약점**: Transformer 대비 spatial reasoning 열위 가능.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Mamba-2나 Jamba (hybrid)를 적용하면? | Hybrid Transformer+Mamba가 양쪽 장점을 가질 가능성. 미탐구 |
| 2 | Long observation history에서 Mamba의 이점이 더 커지는가? | 이론적으로 그렇다. History 길이에 따른 speed/performance 분석 필요 |
