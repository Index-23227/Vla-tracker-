# TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation

> **한 줄 요약**: 1B 미만 VLM(LLaVA-Phi 계열, Pythia 백본)에 LoRA fine-tuning + Diffusion Policy head를 결합하여, **OpenX 사전학습 없이도** 단일팔 5 task 평균 SR 94.0%를 달성하고 OpenVLA(7B, 970K demos pretrain) 대비 +25.7%p 우위와 **20배 빠른 추론**(14 ms vs 292 ms)을 보인 RA-L 2025 논문.

---

## 1. 배경 및 동기

- 기존 VLA(RT-2, OpenVLA)의 두 가지 병목: (1) 7B+ VLM과 (2) discrete action token의 autoregressive 생성 → 추론이 매우 느리고 OpenX(970K demos) 같은 대규모 사전학습이 필요.
- 핵심 질문(Sec. I): "VLA의 장점을 유지하면서 빠르고 data-efficient한 모델을 만들 수 있는가?"
- 실용적 시사: 소규모 연구실/edge 배포 가능성, bimanual 실험에서 OpenX(single-arm only) pretrain의 negative transfer 가능성 검증.

---

## 2. 방법론

### 2.1 Compact VLM Backbone (Sec. III-A, Fig. 2)

- Pythia[30] LM(70M~1.4B) + ViT를 LLaVA[13] 파이프라인으로 vision-language 사전학습.
- 3가지 크기: **TinyVLA-S(422M)**, **TinyVLA-B(740M)**, **TinyVLA-H(1.3B)**.
- Robot data 단계에서 VLM은 **LoRA**(trainable param 5%)만 업데이트. 학습 후 re-parameterization으로 inference latency 영향 없음.

### 2.2 Diffusion Policy Decoder (Sec. III-B)

- Action token discretization(RT-2식) 대신 Diffusion Policy(DDPM)[3] head를 부착.
- 파이프라인: VLM이 raw observation+language를 multimodal embedding → adaptive pooling + LayerNorm으로 fixed feature → proprioception과 concat → 3-layer MLP → DP head conditioning → noise 예측.
- DP head는 **full parameter training**, VLM은 LoRA — 두 학습 전략 분리.

> ❓ **예상 질문**: LoRA로 5%만 업데이트해도 정말 robot dynamics를 학습 가능한가?
> **답변**: VLM의 pre-trained world knowledge를 보존하면서 attention의 Q/K/V에만 low-rank 추가. DP head가 full-train되므로 action distribution 학습은 head가 담당, VLM은 perception/언어 이해를 보존(Sec. III-B).

---

## 3. 실험 결과

### 3.1 MetaWorld 50-task Multi-task (Table I)
- TinyVLA-H Avg 31.6 vs Diffusion Policy 10.5 — Hard 카테고리에서 DP 대비 6배 성능(11.4 vs 1.9).

### 3.2 Single-arm Franka 5 Tasks (Table II)
| Model | Pretrain | Total/Train Params | Avg. SR (%) |
|---|---|---|---|
| Diffusion Policy | N/A | 111M / 111M | 35.3 |
| OpenVLA | 970K (OpenX) | 7.2B / 195M | 68.3 |
| **TinyVLA-S** | **N/A** | **422M / 101M** | 23.3 |
| **TinyVLA-B** | **N/A** | **740M / 138M** | 77.4 |
| **TinyVLA-H** | **N/A** | **1.3B / 143M** | **94.0** |

- TinyVLA-H FlipMug 98.3% / StackCubes 98.3% — OpenVLA(51.7 / 40.0) 대비 압도적.

### 3.3 Bimanual UR5 (Table III) — **OpenX pretrain의 한계 노출**
- OpenVLA는 PlaceBread/StackCubes/PlaceTennisBag 모두 0%; TinyVLA-H는 76.7/36.7/30.0%. OpenX(single-arm only)의 bimanual negative transfer.

### 3.4 일반화 (Fig. 5–9)
- **View(±15~30°)**: TinyVLA 25/48 vs OpenVLA 4/48, DP 0/48 (Fig. 5).
- **Background**: TinyVLA 10/12 vs OpenVLA 9/12 vs DP 0/12 (Fig. 6).
- **Spatial OOD**: OpenVLA 13/16 > TinyVLA 12/16 (Fig. 9) — 970K OpenX의 spatial 다양성 우위 영역.

### 3.5 Inference Latency (Table IV) & Policy Head Ablation (Table V)
- OpenVLA-7B 292 ms → OpenVLA-1B 140 ms → **TinyVLA-1B 14 ms** (~20×). VLM 축소(2×) + diffusion head(AR 제거)의 결합 효과.
- Head ablation: MLP 0.0%, ACT ~13%, **Diffusion ~94%** — diffusion head가 핵심.

---

## 4. 한계 및 미해결 문제

1. **Spatial OOD에서 OpenVLA에 패배**(Fig. 9): "no pretrain" 주장의 부분적 약점.
2. **Failure mode가 모델 크기에 따라 변함**(Fig. 10): 0.4B는 instruction misinterpret, 1.3B는 inaccurate positioning.
3. **Bimanual 실험은 3 task만**: long-horizon dual-arm 미검증.
4. **Pythia + LLaVA 의존**: modern VLM(PaliGemma) 대체 시 정량적 평균 SR 미보고.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "Pretrain-free + LoRA + DP head" 조합의 검증 |
| **Technical depth** | ★★★☆☆ — 구성요소는 기존 기법의 조합 |
| **Experimental rigor** | ★★★★★ — sim/real, single/bimanual, 5+ 일반화 시나리오 |
| **Practical impact** | ★★★★★ — 14 ms 추론, edge 배포 현실화 |

**강점**: OpenX pretrain 없이 SOTA에 근접, 20× 추론 속도, bimanual에서 OpenVLA를 능가.
**약점**: Spatial OOD 한계, modern VLM(PaliGemma) 활용 시 추가 이득 불확실.

---

## 6. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | OpenX pretrain 없이 어떻게 OpenVLA를 능가하나? | DP head가 full-train되어 action distribution을 정밀 학습. Multi-view 입력을 OpenVLA에 맞춰 fair화한 후에도 +25.7%p(Sec. IV-B). |
| 2 | Bimanual에서 OpenVLA 0%는 너무 극단적이지 않나? | 저자 진단: OpenX는 single-arm 전용. 이는 OpenVLA의 한계가 아니라 OpenX 분포의 한계. TinyVLA는 robot data만으로 학습되어 bimanual 데이터에 직접 적합. |
| 3 | DP를 ACT/MLP로 대체하면? | Table V: MLP 0.0%, ACT ~13%, DP ~94% — DP의 multi-modal 표현력이 결정적. |
| 4 | 14 ms 추론이 실제 control loop에 충분한가? | 7-DOF action 1회 예측 기준이며 chunking 미사용 — chunking 결합 시 추가 향상 가능. |

<!-- VERIFIED: pdf -->
