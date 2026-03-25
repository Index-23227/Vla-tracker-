# HPT: Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers

> **한 줄 요약**: 52개 데이터셋의 heterogeneous 로봇 데이터에서 공유 가능한 policy trunk를 사전학습하여, embodiment-agnostic representation을 학습하고 unseen 태스크에서 20%+ fine-tuning 성능 향상 달성.

---

## 1. 배경 및 동기

- 로봇 데이터의 이질성(embodiment, sensor, task)이 대규모 사전학습의 최대 장벽
- **Proprioception과 vision의 차원이 로봇마다 다름** → 공유 모델 학습 어려움

---

## 2. 방법론 심층 분석

### 2.1 Heterogeneous Alignment

각 embodiment의 proprioception과 vision을 **짧은 token sequence로 정렬**:

$$\mathbf{z}_i = \text{Aligner}_i(\mathbf{o}_i^{\text{vis}}, \mathbf{o}_i^{\text{prop}})$$

Aligner: embodiment-specific lightweight MLP/Conv.

### 2.2 Shared Policy Trunk

정렬된 token이 **공유 transformer trunk**를 통과:
- Cross-embodiment knowledge transfer
- Task-agnostic motor primitives 학습

### 2.3 52 Datasets Scale

| 소스 | 규모 |
|------|------|
| Open X-Embodiment | ~25 datasets |
| Simulation (RLBench, CALVIN 등) | ~15 datasets |
| Human video | ~7 datasets |
| Deployed robots | ~5 datasets |

---

## 3. 실험 결과 심층 분석

| 설정 | From scratch | HPT pre-trained | 향상 |
|------|------------|----------------|------|
| New sim task (avg) | 58.3% | 72.1% | +13.8%p |
| New real task (avg) | 42.7% | 63.5% | +20.8%p |
| New embodiment | 35.2% | 52.8% | +17.6%p |

- **실제 로봇에서 20%+ 향상**이 핵심 기여

---

## 4. 한계 및 미해결 문제

1. **Aligner의 overhead**: 새 embodiment마다 aligner 재학습 필요
2. **Negative transfer**: 52 datasets 중 어떤 조합이 해로운지 미분석
3. **Trunk 크기의 sweet spot**: Scaling law 분석 부분적
4. **Action prediction만 평가**: Perception, reasoning 능력의 향상은 미측정

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Heterogeneous alignment + shared trunk |
| **Technical depth** | ★★★★☆ — 52 datasets에서의 scaling 분석 |
| **Experimental rigor** | ★★★★★ — 다양한 downstream 평가 |
| **Practical impact** | ★★★★★ — Pre-trained robot policy backbone |
| **Writing quality** | ★★★★☆ |

**강점**: 52 datasets 규모의 포괄적 사전학습, 실질적 fine-tuning 성능 향상. **약점**: Negative transfer 분석 부족, aligner의 embodiment-specific 의존성.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 52 datasets 중 가장 유용한 subset은? | Data selection/pruning 분석 부분적. 일부 dataset 제거 시 성능 하락하는 것도, 상승하는 것도 존재 |
| 2 | VLM backbone (OpenVLA처럼) 위에 HPT를 쌓으면? | 유망한 결합이나 미탐구. VLM의 language 능력 + HPT의 motor primitive 지식 |
| 3 | Aligner 없이 universal tokenization이 가능한가? | CrossFormer가 이를 시도. Aligner는 오히려 필수적 유연성을 제공 |
