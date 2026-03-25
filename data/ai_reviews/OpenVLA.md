# OpenVLA: An Open-Source Vision-Language-Action Model

> **한 줄 요약**: Llama 2 + DINOv2 + SigLIP을 결합한 7B 파라미터 오픈소스 VLA로, 97만 real-world demonstration에서 학습하여 RT-2-X (55B)를 16.5%p 절대 성능으로 능가하면서 7배 적은 파라미터 사용.

---

## 1. 배경 및 동기

- RT-2가 VLA의 가능성을 보였으나 **비공개 55B 모델** → 접근 불가
- Octo는 오픈소스이나 VLM backbone 없이 성능 제한
- **오픈소스 + VLM backbone + 대규모 데이터**의 결합 필요

---

## 2. 방법론 심층 분석

### Architecture

```
[DINOv2] → visual features ┐
[SigLIP] → visual features ├→ [Projector] → [Llama 2 (7B)] → action tokens
[Language] → text tokens    ┘
```

- **Dual visual encoder**: DINOv2 (spatial features) + SigLIP (semantic features)
- **Action tokenization**: 256-bin discretization per dimension
- **Autoregressive action generation**: Llama 2가 action token을 순차 생성

> ❓ **예상 질문**: 왜 두 개의 visual encoder인가?
> **답변**: DINOv2는 spatial/geometric feature에 강하고, SigLIP은 semantic/language-aligned feature에 강함. 두 encoder의 상보적 정보가 manipulation에 모두 필요.

### Fine-tuning

LoRA 기반 효율적 fine-tuning:
- Consumer GPU (RTX 3090)에서 fine-tuning 가능
- ~5 GPU-hours for new task adaptation

---

## 3. 실험 결과 심층 분석

| 모델 | Params | SimplerEnv Avg (%) | Real Robot SR (%) |
|------|--------|-------------------|------------------|
| RT-2-X | 55B | 52.3 | 41.5 |
| Octo | 93M | 38.2 | 35.2 |
| **OpenVLA** | **7B** | **48.7** | **58.0** |

- RT-2-X (55B) 대비 **7x 적은 파라미터로 16.5%p 절대 우위** (real robot)
- SimplerEnv에서는 RT-2-X에 약간 못 미치나, real robot에서 역전

---

## 4. 한계 및 미해결 문제

1. **Action tokenization**: 256-bin discretization의 precision 한계 → CogACT, OpenVLA-OFT 등에서 개선
2. **추론 속도**: 7B autoregressive → ~200ms per action, 5Hz 제어
3. **Autoregressive action의 exposure bias**: Left-to-right generation의 error cascading
4. **Long-horizon 제한**: 단일 action prediction, planning 능력 부재

### 후속 영향
- 사실상 **VLA 연구의 표준 baseline**으로 자리 잡음
- OpenVLA-OFT, CogACT, FLOWER, NORA 등 모두 OpenVLA를 baseline으로 비교

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Open-source VLA의 정의적 연구 |
| **Technical depth** | ★★★★☆ — Dual encoder, efficient fine-tuning |
| **Experimental rigor** | ★★★★★ — 포괄적 평가 |
| **Practical impact** | ★★★★★ — VLA 연구 생태계 형성 |
| **Writing quality** | ★★★★★ |

**강점**: VLA 연구의 democratization. 모든 후속 연구가 OpenVLA를 기반으로 비교·개선. **약점**: Discretization, autoregressive 생성의 한계는 이후 연구에서 해결.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Diffusion head로 교체하면 얼마나 좋아지는가? | CogACT가 정확히 이를 수행: +35% (sim). Discretization이 주요 bottleneck |
| 2 | LoRA fine-tuning vs full fine-tuning? | Full이 약간 낫지만 memory 5배+ 필요. LoRA가 practical sweet spot |
| 3 | 7B가 필수인가? 3B로 줄이면? | NORA (3B)가 OpenVLA 능가. VLM backbone 품질이 크기보다 중요 |
