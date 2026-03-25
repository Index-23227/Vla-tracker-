# OpenVLA: An Open-Source Vision-Language-Action Model

> **한 줄 요약**: SigLIP+DINOv2 dual encoder (600M) + Llama 2 (7B)를 970K OXE demonstration에서 학습한 오픈소스 VLA로, BridgeV2 **76.2%**, Google Robot **79.4%**로 RT-2-X (55B)를 능가하며 LoRA로 **RTX 3090에서 fine-tuning** 가능. **64×A100, 14일 (21,500 A100-hrs)** 학습.

---

## 1. 배경 및 동기

- RT-2가 VLA의 가능성을 보였으나 **비공개 55B 모델**
- Octo는 오픈소스이나 VLM backbone 없이 성능 제한
- **오픈소스 + VLM + 대규모 데이터**의 결합 필요

---

## 2. Architecture

| Component | Params | Details |
|-----------|--------|---------|
| Visual encoder | 600M | **SigLIP + DINOv2** (channel-wise concat) |
| Projector | ~2M | 2-layer MLP |
| LLM backbone | 7B | **Llama 2** |
| Action tokenization | — | **256 bins/dim** (least-used 256 tokens 덮어씀) |
| Input resolution | — | **224×224** |

### Training

| 항목 | 값 |
|------|-----|
| Data | **970K demos** (Open X-Embodiment) |
| GPU | **64× A100** |
| Duration | **14 days** |
| Total compute | **~21,500 A100-hours** |
| Batch | 2048 |
| LR | 2e-5 (fixed, no warmup) |
| Epochs | **27** iterations through full dataset |

---

## 3. 실험 결과 심층 분석

### BridgeData V2 WidowX (170 rollouts)

| Model | Params | SR (%) |
|-------|--------|--------|
| RT-1-X | 35M | 31.8 |
| Octo | 93M | 45.3 |
| RT-2-X | **55B** | 59.7 |
| **OpenVLA** | **7B** | **76.2** |

- **RT-2-X (55B) 대비 +16.5%p**, 7x 적은 파라미터

### Google Robot (60 rollouts)

| Model | SR (%) |
|-------|--------|
| RT-1-X | 38.2 |
| Octo | 52.7 |
| RT-2-X | 78.1 |
| **OpenVLA** | **79.4** |

### Fine-tuning (Franka, 129 rollouts)

| Method | SR (%) | Trainable Params | VRAM |
|--------|--------|-----------------|------|
| Full FT | 69.7±7.2 | 7,188M | - |
| **LoRA (r=32)** | **68.2±7.5** | **97.6M** | **59.7GB** |
| LoRA (r=64) | 68.2±7.8 | 195.2M | - |
| Sandwich | 62.1±7.9 | 914.2M | - |

### Quantization

| Precision | SR (%) | VRAM |
|----------|--------|------|
| bfloat16 | 71.3±4.8 | 16.8GB |
| int8 | 58.1±5.1 | 10.2GB |
| **int4** | **71.9±4.7** | **7.0GB** |

- **int4가 bfloat16보다 약간 높음** (71.9 vs 71.3) → 놀라운 결과

### Inference Speed

- A100: **~6Hz** (bfloat16)
- A5000 int4: ~3Hz

---

## 4. 핵심 발견

1. **Vision encoder fine-tuning 필수**: VLM 관례와 달리, vision encoder를 함께 fine-tune해야 로봇 성능 향상
2. **Multiple epochs 필요**: LLM의 단일 pass와 달리, 27 epochs이 중요
3. **224×224 충분**: 384×384와 동등 성능, 연산 절감

---

## 5. 한계 및 미해결 문제

1. **256-bin discretization**: Action precision 한계 → CogACT, OpenVLA-OFT에서 개선
2. **6Hz**: Autoregressive generation이 느림 → parallel decoding으로 해결 (OFT)
3. **21,500 A100-hrs**: 학습 비용 높음 → FLOWER (200 H100-hrs)가 50x 효율적
4. **VLA 연구의 표준 baseline**: 이후 모든 VLA가 OpenVLA를 baseline으로 비교

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Open-source VLA의 정의적 연구 |
| **Technical depth** | ★★★★★ — Dual encoder, LoRA, quantization 분석 |
| **Experimental rigor** | ★★★★★ — 170+60+129 rollouts, 포괄적 |
| **Practical impact** | ★★★★★ — VLA 연구 생태계 형성 |
| **Writing quality** | ★★★★★ |

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | int4 (71.9%) > bfloat16 (71.3%)의 이유는? | 통계적 변동 범위 내. 또는 quantization이 regularization 효과를 낼 가능성 |
| 2 | LIBERO에서는? | OpenVLA: 76.5%. OFT로 fine-tune하면 97.1%. Fine-tuning recipe가 결정적 |
| 3 | 7B가 필수인가? 3B면? | NORA (3B)가 real-world에서 OpenVLA 능가 (56.7% vs 40%). Backbone quality > size |

<!-- VERIFIED: pdf -->
