# RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation

> **한 줄 요약**: 1.2B 파라미터 Diffusion Transformer를 1M+ 멀티로봇 trajectory에 사전학습하고, Physically Interpretable Unified Action Space를 도입해 ALOHA 양팔 로봇에서 zero-shot 일반화·few-shot 학습·dexterous 제어를 동시에 달성한 ICLR 2025 논문.

---

## 1. 배경 및 동기

- Bimanual manipulation은 dual-arm action space(높은 차원) + multi-modal action distribution(Fig. 2b의 cube 잡기 toy example) 때문에 unimanual VLA를 그대로 이식할 수 없음.
- Dual-arm 데이터는 < 10K trajectory 수준으로 매우 희소 → cross-embodiment pretrain + target fine-tune 파이프라인이 필요.
- 기존 접근들의 한계: ACT(VAE)는 표현력 부족, OpenVLA(discrete tokenization)는 양손 협응에서 quantization error 누적, Octo(93M diffusion)는 모델 크기가 부족.

---

## 2. 방법론

### 2.1 Diffusion Transformer (Sec. 4.1, Fig. 3)

- **Backbone**: SigLIP(vision, frozen) + T5-XXL(language, frozen) + 1.2B DiT denoiser. Action chunk $a_{t:t+T_a}$를 K=100 → DPM-Solver++로 5 step에 denoising. 추론 주파수 6 Hz chunk / 381 Hz action(RTX 4090).
- **DiT 핵심 수정 3가지** (Fig. 4)
  1. **QKNorm + RMSNorm**: 로봇 물리량의 unstable numerical range로 인한 attention 불안정·loss explosion을 방지(Fig. 4a). LayerNorm의 centering이 시계열의 symmetry를 깨므로 RMSNorm 채택.
  2. **MLP Decoder**: 마지막 linear projection 대신 nonlinear MLP를 써야 dexterous task(Robot Dog walk straight)에서 성공률 유지(Fig. 4b).
  3. **Alternating Condition Injection (ACI)**: image·language를 cross-attention으로 주입하되, 매 layer 동시 주입 시 image token이 text token을 dominate함. Layer 단위로 번갈아 주입하여 instruction following 정확도 회복.

### 2.2 Physically Interpretable Unified Action Space (Sec. 4.2)

- 다양한 로봇의 proprioception/action을 EE pose + joint + wheel + gripper의 통합 슬롯에 물리적 의미를 보존하며 매핑(Fig. 3 좌측). Padding으로 dimension 차이를 흡수.
- 46개 데이터셋 / 1M+ trajectory / 21TB 규모로 pretrain → ALOHA 양팔 fine-tune 데이터(6K+ trajectory, 300+ task, 100+ object, 15+ room)에서 refinement.

> ❓ **예상 질문**: Unified space의 "padding"이 학습에 noise가 되지 않는가?
> **답변**: Robot마다 슬롯 점유가 다르므로 stochastic independent masking을 함께 적용해 modality dependency가 한쪽으로 쏠리지 않도록 했다고 Sec. 4.1에서 언급.

---

## 3. 실험 결과 (Table 2, 3)

### Q1·Q2: Zero-shot 일반화 (Wash Cup·Pour Water 전체 성공률, %)
| Method | Wash Cup unseen2 | Pour Water unseen3 | Pour-L-1/3 Correct Amount |
|---|---|---|---|
| ACT | 0 | 12.5 | N/A |
| OpenVLA | 0 | 0 | 0 |
| Octo | 0 | 0 | 0 |
| RDT (scratch) | 0 | 25 | 12.5 |
| **RDT (ours)** | **50** | **62.5** | **75** |

### Q3·Q4: Few-shot & Dexterity
- Handover (5-shot): RDT 40 vs OpenVLA·Octo 0
- Fold Shorts (1-shot): RDT 68 vs Octo 4 vs ACT·OpenVLA 0
- Robot Dog Walk Straight: RDT 48 vs Octo·ACT·OpenVLA ≤ 32 (정확한 joystick 각도 제어)

### Q5: Ablation (Table 2)
- RDT(regress) Unseen Object 12.5%, RDT(small 166M) 37.5%, RDT(scratch) 0%, RDT(ours) 50% — **diffusion·scale·pretrain 모두 필수**.

---

## 4. 한계 및 미해결 문제

1. **추론 비용**: 1.2B DiT는 실용 환경에서 여전히 GPU(RTX 4090) 필요. Edge 배포 미해결.
2. **Bimanual 데이터 의존**: Fold Shorts·Handover는 5 shot 미만에서 성능 급락 가능성(논문 Sec. 5.1 demos: Fold Shorts 1, Handover 5).
3. **Unified action space의 padding**: redundant DOF 로봇(humanoid 등)에서 정보 손실 가능 — 논문은 ALOHA 중심 평가만 제공.
4. **Vision encoder freezing**: SigLIP/T5-XXL freeze 정책으로 인해 새로운 시각적 도메인(예: 야외)에서 generalization 미검증.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Bimanual 특화 1.2B DiT + Unified Action Space |
| **Technical depth** | ★★★★★ — QKNorm, MLP decoder, ACI 등 robotic-aware DiT 수정 |
| **Experimental rigor** | ★★★★☆ — 7 task, 4 baseline, 4 ablation variant, 실로봇 평가 |
| **Practical impact** | ★★★★★ — Bimanual VLA의 사실상 첫 foundation model |

**강점**: 양팔 multi-modal action을 표현하는 diffusion modeling, 물리 의미를 보존하는 unified space, 1M trajectory pretrain의 가시적 효과(Table 2의 scratch vs ours 비교).
**약점**: 9.4 Hz 미만 chunk 주파수, fine-tuning에 6K+ trajectory 필요(데이터 수집 비용).

---

## 6. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Unified action space가 unimanual VLA에 비해 negative transfer를 만들지 않는가? | Sec. 4.2의 stochastic independent masking + 물리 의미 보존 매핑이 negative transfer를 완화. Table 2에서 scratch < small < ours로 pretrain의 양의 효과 검증. |
| 2 | DPM-Solver++ 5 step이 정밀 task에서도 충분한가? | Robot Dog Walk Straight 48% vs Octo 0%로 dexterous task에서도 5 step이 충분함을 시사. 다만 더 까다로운 manipulation에서는 step 증가 필요할 수 있음. |
| 3 | Vision/Language encoder를 freeze한 이유와 trade-off는? | GPU 메모리 절감 목적(Sec. 4.1). 신규 도메인 적응이 어려울 수 있으나 multi-robot pretrain의 일반화로 보완. |
| 4 | ACI가 instruction following을 정말 결정짓는가? | Fig. 4b에서 w/o ACI 시 Pour Water-L-1/3 Correct Amount 성공률이 큰 폭으로 하락 — 정량 검증됨. |

<!-- VERIFIED: pdf -->
