# CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for VLA Models

> **한 줄 요약**: 자기회귀(AR) VLA에만 적용되어 온 DPO(Direct Preference Optimisation)를 연속 행동(flow-matching) backbone에도 확장하기 위한 surrogate log-probability 추정기를 제안하고, LoRA vs DoRA, denoise 루프 latency, prefix KV-cache 가속의 상한 등 cross-paradigm post-training과 추론 최적화를 실증 분석한 연구.

---

## 1. 배경 및 동기

최근 VLA 모델은 크게 두 가지 패러다임으로 수렴해 왔다: OpenVLA류의 **이산 토큰 자기회귀(autoregressive)** 방식과 pi-0.5류의 **연속 행동 flow-matching** 방식이다. 그러나 언어 모델에서 사실상 표준이 된 사후 학습(post-training) 절차인 **DPO**는 거의 전적으로 자기회귀 VLA에서만 연구되어 왔으며, flow-matching backbone에 DPO를 적용하려면 probability-flow ODE 적분이 필요해 비용이 크다는 문제가 있다. CrossVLA는 이 cross-paradigm 격차를 좁히고 동시에 VLA 추론 latency의 실제 병목을 정밀하게 분해하는 것을 목표로 한다.

## 2. 핵심 아이디어

- **Surrogate flow-matching log-probability 추정기**: probability-flow ODE 적분 없이 DPO가 연속 행동 backbone 위에서 동작하도록 만드는 대체 추정기를 제안.
- **LoRA vs DoRA head-to-head**: VLA DPO의 PEFT(파라미터 효율 미세조정) 계층으로서 LoRA와 DoRA를 정면 비교.
- **Inference-time anatomy**: `sample_actions` 호출의 latency 구성을 분해해 denoise 루프가 78.6%를 차지함을 정량화하고, prefix K/V 캐시(VLA-Cache류) 가속의 상한을 측정.
- **Multi-view + temporal projection head 사전학습**: 6000 LIBERO 프레임으로 학습해 same-task retrieval에서 99.5% k-NN recall@1 달성, downstream initialisation으로 공개.

## 3. 방법론 요약

CrossVLA는 새로운 정책 모델을 처음부터 학습시키는 것이 아니라 **사후 학습 + 추론 단계**에서의 cross-paradigm 분석을 수행한다. 우선 surrogate log-probability 추정기를 통해 DPO 손실을 flow-matching backbone에서 ODE 적분 없이 계산할 수 있도록 한다. 이를 OpenVLA를 SFT한 모델 위에 적용해 LoRA와 DoRA를 PEFT 계층으로 사용하는 두 변종을 비교한다. 평가는 LIBERO 4개 suite, 각 600 trial × 3 seed로 수행되며, 추론 분석은 chunk-level 및 token-level 캐시 전략의 성공률 영향을 함께 측정한다. 추가로 downstream 초기화용으로 multi-view + temporal projection head를 LIBERO 프레임에 사전학습한다.

## 4. 실험 결과

- **DoRA + DPO 평균 +10.4 pp** (OpenVLA SFT 대비, LIBERO 4-suite 평균, 600 trials × 3 seeds).
- Per-suite 상세:
  - Object **+20.0 pp** (3 seeds 모두 38/50 — seed variance 0)
  - Long-horizon **+11.0 pp**
  - Goal **+8.0 pp**
  - Spatial **+2.7 pp**
- **Inference anatomy**:
  - Denoise 루프가 `sample_actions` latency의 **78.6%** 차지.
  - Prefix K/V 캐시(VLA-Cache류)의 가속 상한 ≈ **21%**.
  - Chunk-level과 token-level 캐시 전략 모두 성공률을 **0–80%까지 저하**시킬 수 있음.
- **Projection head 사전학습**: same-task retrieval에서 **k-NN recall@1 99.5%** (랜덤 대비 36배).
- LIBERO 절대 점수, 모델 파라미터 수, 학습 비용 등은 abstract에 미명시.

## 5. 한계 및 의의

abstract 기준으로 CrossVLA는 새로운 백본을 제시하는 연구가 아니라 **기존 VLA 패러다임의 post-training/inference 디자인 공간을 cross-paradigm으로 정리**하는 실증 연구다. DoRA가 자명한 선택이 아니라 LoRA 대비 실제로 +10.4 pp의 향상을 줄 수 있고, denoise 루프가 latency의 80% 가까이를 차지하므로 prefix-cache류 기법으로는 본질적인 가속이 어렵다는 점을 명확히 보여준 것이 핵심 기여다. 다만 (a) 저자가 한 명으로만 표기되어 있고 affiliation이 공개되지 않은 점, (b) LIBERO 절대 점수가 abstract에 없어 다른 VLA와의 직접 비교가 어려운 점, (c) workshop draft로 표기된 점은 결과 일반화에 주의가 필요한 부분이다. 코드와 체크포인트가 공개(https://github.com/lz-googlefycy/vla-lab)되어 후속 cross-paradigm 연구의 baseline으로 재현/확장 가능하다는 점은 강한 강점.

<!-- VERIFIED: abstract-only -->
