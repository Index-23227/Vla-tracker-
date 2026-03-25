# Motus: A Unified Latent Action World Model

> **한 줄 요약**: Latent action space에서의 통합 world model로, 기존 pretrained model과 motion information을 결합하는 Mixture-of-Transformer 아키텍처를 통해 video prediction, action prediction, planning을 유연하게 수행.

---

## 1. 배경 및 동기

- World model과 action model의 **분리된 학습**이 비효율적
- Latent action space에서 통합하면 action-labeled data 없이도 world dynamics 학습 가능
- 기존 pretrained model (video generation, VLM)의 지식을 활용하면서 motion 정보를 추가

---

## 2. 방법론 심층 분석

### Mixture-of-Transformer Architecture

여러 모델링 모드를 하나의 아키텍처에서 유연하게 전환:
- **Mode 1**: Video prediction (future frames)
- **Mode 2**: Action prediction (current → action)
- **Mode 3**: Joint (video + action)

Latent action:
$$\mathbf{z}_a = E_a(\mathbf{o}_t, \mathbf{o}_{t+1})$$

Inverse dynamics로 latent action을 추출하고, 이를 통해 action-unlabeled video에서도 학습.

> ❓ **예상 질문**: Latent action이 실제 robot action과 어떻게 대응되는가?
> **답변**: Latent→real action decoder를 소량의 labeled data로 학습. Latent space의 구조가 물리적 의미를 갖도록 contrastive/regression loss 적용. 완벽한 대응은 보장되지 않음.

---

## 3. 실험 결과

| 모델 | LIBERO Avg (%) | Inference (ms) |
|------|---------------|---------------|
| GR-1 | ~91 | ~500 |
| pi0 | ~90 | ~200 |
| **Motus** | **~93** | **~400** |

- GR-1, pi0 대비 competitive하면서 unified framework의 유연성 제공

---

## 4. 한계 및 미해결 문제

1. **Latent action의 해석가능성 부족**: Latent space가 black-box
2. **Inference latency**: Video + action joint 모드에서 400ms+
3. **Decoder 학습의 데이터 의존**: Labeled data 필요
4. **GigaWorld-Policy에 의해 추월**: Action-centered 접근이 더 효율적임이 이후 보여짐

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Latent action world model |
| **Technical depth** | ★★★★☆ — MoT, multi-mode 설계 |
| **Practical impact** | ★★★☆☆ — Latency 문제 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM처럼 test-time video를 제거하면? | 이후 GigaWorld-Policy가 이를 검증. Motus에서의 직접 실험은 부재 |
| 2 | Latent action space의 차원은 성능에 얼마나 민감한가? | Ablation 부분적. 너무 작으면 정보 손실, 너무 크면 학습 어려움 |
