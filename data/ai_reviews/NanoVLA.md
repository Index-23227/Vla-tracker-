# NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies

> **한 줄 요약**: Xiaomi EV의 edge-device 지향 경량 VLA 패밀리. (1) vision-language 입력을 late stage에서만 융합하는 decoupled fusion으로 instruction caching, (2) 길게 학습 짧게 실행(LSAC) chunking, (3) Beta-Binomial 기반 dynamic router로 small/large backbone 전환. ResNet18 + BERT-base(NanoVLA-S, 161M) / Qwen2.5-0.5B(NanoVLA-L, 520M) 구성으로 Jetson Orin Nano(8GB)에서 OpenVLA 대비 52× FPS, +13.8% SR.

---

## 1. 배경 및 동기

- VLA 배포의 3가지 edge 병목:
  1) 추론이 느리고 compute-intensive,
  2) long-horizon 행동이 jerky·brittle,
  3) 단일 backbone이 task 난이도와 mismatch (단순 grasp에 over-compute, 장기 task에 under-perform).
- 단순 파라미터 축소 대신 **추론을 재구성**하자는 문제의식: "compute only where it matters".

---

## 2. 방법론 심층 분석

### 2.1 Vision-Language Decoupling with Caching (Section 3.1)
- 기존 VLA는 cross-attention으로 매 step vision·language 모두 재계산.
- NanoVLA는 visual encoder(ResNet18/ViT)와 language encoder(BERT/Qwen)를 **frozen** 유지하고 독립 인코딩 후 lightweight transformer가 late fusion (self-attn → cross-attn → MLP).
- **Caching**: instruction 임베딩은 1회 계산 후 재사용. Qwen 0.5B backbone에서 추론 시간 −62% (Figure 5b).

### 2.2 Long-Short Action Chunking (LSAC, Section 3.2)
- Train: 긴 horizon H_train개 action을 ℓ-loss로 회귀 (식 2).
- Inference: 첫 h ≪ H_train만 실행 후 replan (식 3).
- 효과: Fixed AC는 30–40 step 이후 SR 70–74로 급락. **LSAC는 20–60 step 평탄, 100 step에서도 84.0** (Figure 4b).

### 2.3 Dynamic Routing (Section 3.3)
- Per-task posterior p_{m,l} ∼ Beta(s_{m,l}+α₀, n_{m,l}−s_{m,l}+β₀).
- Pairwise win probability π_{i≻j}(l) (식 6) — closed-form 또는 Monte Carlo (MCB) 근사.
- Bernoulli 로그우도(식 8)로 text-conditioned classifier f_θ 학습. Inference 시 default lightweight, p̂_L(l) > τ면 large model로 escalate.

### 2.4 모델 변종
| 변종 | Vision | LLM | Total / Trainable |
|------|--------|-----|-------------------|
| NanoVLA-S | ResNet18 | BERT-base | 161M / 52M |
| NanoVLA-L | ResNet18 | Qwen2.5 0.5B | 520M / 52M |
| NanoVLA-R | router | both | 296M(평균) / 52M |

---

## 3. 실험 결과

### 3.1 LIBERO 4-suite (Table 1)
| Policy | Total | Spatial | Object | Goal | Long | Avg |
|--------|-------|---------|--------|------|------|-----|
| OpenVLA | 7.5B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 3.5B | 87.0 | 63.0 | 89.0 | 48.0 | 71.8 |
| SpatialVLA | 3.5B | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| SmolVLA | 450M | 72.8 | 69.8 | 84.0 | 52.6 | 78.6 |
| NanoVLA-S | 161M | 81.6 | 93.6 | 89.6 | 49.8 | 78.7 |
| NanoVLA-L | 520M | 87.2 | 89.8 | 90.0 | 55.2 | 80.4 |
| **NanoVLA-R** | 296M* | **89.8** | **96.2** | **93.0** | **57.4** | **84.1** |

- OpenVLA 대비 +6.0–12.3% Avg, **<10% 파라미터**.

### 3.2 LIBERO-90 / 실세계 / Edge
- LIBERO-90: NanoVLA-L 83.3% (+21.3% vs OpenVLA 62.0%, +14.4% vs SmolVLA). NanoVLA-S 55.1% — BERT의 long-instruction 한계.
- LeRobot 11 real task Avg: SmolVLA 58.0 / π0 51.6 / OpenVLA 80.4 / **NanoVLA-L 85.6** / **NanoVLA-R 85.6**. Open/Close lid에서 NanoVLA-L 76/68 vs OpenVLA 56/54.
- Jetson Orin Nano 8GB: OpenVLA 대비 **52× FPS, +13.8% SR**. Caching으로 NanoVLA-L 추론 35.49s → 13.36s (−62%).

---

## 4. 한계

1. **Vision encoder는 ResNet18 고정**: 시각 표현력 한계. ViT 변종 비교 부재.
2. **Encoder freeze**: 새 환경 적응이 action expert 52M에 집중 → 시각 분포 변화에 민감 (논문도 night→day 실험에서 일부 영향 보고).
3. **Router 학습 비용**: per-task 시도 데이터 n_{m,l}, s_{m,l} 필요 → 새 task에 cold-start.
4. **NanoVLA-S의 Long suite 49.8%**: BERT가 long-horizon planning에 약함을 인정.
5. **CALVIN/SimplerEnv 미보고**: 다른 표준 벤치마크 부재.
6. **Open-source 미공개**(YAML 명시).

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — decoupling+chunking+routing 조합의 edge 특화 |
| Technical depth | ★★★★☆ — Beta-Binomial routing이 흥미로움 |
| Experimental rigor | ★★★★☆ — sim+실세계+edge 측정 |
| Practical impact | ★★★★☆ — Jetson 배포 검증 |
| Writing | ★★★☆☆ — under review, 일부 표현 거침 |

**강점**: edge에서 멀티-billion 모델 능가, caching/chunking/routing 모듈화. **약점**: encoder freeze로 인한 시각 적응 한계, 비공개.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | Decoupled fusion이 본질적으로 SmolVLA와 무엇이 다른가? | SmolVLA는 distillation 중심 단일 backbone. NanoVLA는 instruction caching까지 활용한 inference 재설계 + routing. |
| 2 | 52× FPS는 4-bit quantized OpenVLA 기준 — fair comparison? | 논문도 OpenVLA가 FP에서 OOM이라 4-bit 추정치 사용을 명시. SmolVLA 비교(43.8% FPS, +3.2% SR)가 더 fair. |
| 3 | Router의 Bayesian 모델이 실제 효용? | NanoVLA-R가 NanoVLA-L 대비 −1.7% SR이지만 −41% 파라미터. trade-off 우수. 다만 task pool이 LIBERO-90에 종속. |
| 4 | LSAC가 ACT(Zhao 2023)와 본질적으로 다른가? | ACT는 fixed chunk를 길게 잡고 짧게 실행. LSAC는 동일 idea를 명시화·정량화. 신규성은 제한적, 엔지니어링 가치는 큼. |
| 5 | ResNet18 freeze에서 fine-tuning을 더 하면? | 미검증. encoder unfreeze 시 trainable 파라미터 폭증으로 edge 메리트 잠식 가능. |

<!-- VERIFIED: pdf -->
