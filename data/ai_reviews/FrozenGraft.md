# Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale: A Frozen-Backbone Grafting Diagnostic

> **한 줄 요약**: 공개된 VLA 체크포인트의 vision tower만 deterministic AvgPool + LayerNorm + linear projector로 교체해 2,000-step 학습하는 frozen-backbone "grafting" 진단법으로, SmolVLA-450M에서 우승한 encoder(SigLIP)가 π0.5-3.3B에서도 상위권을 차지한다고 신뢰할 수 없음을 40-cell 그리드로 보여준 방법론/진단 논문.

---

## 1. 배경 및 동기

- VLA(Vision-Language-Action) 정책의 vision encoder는 보통 upstream VLM 릴리스에서 그대로 상속(SmolVLA→SmolVLM SigLIP, π0.5→PaliGemma SigLIP, OpenVLA→SigLIP+DINOv2 fused). 그러나 소형 백본에서 검증된 encoder 선택이 대형 백본으로 옮겨가는지에 대한 통제된 증거는 없다.
- 기존 연구(VLM4VLA, OpenVLA-OFT, Theia)는 encoder를 백본과 **co-train**하므로 encoder 품질과 백본-encoder 공동적응이 분리되지 않는다.
- 실무 시나리오: 이미 출시된 π0.5/SmolVLA 체크포인트를 inherit한 사용자는 VLM pretraining을 다시 할 수 없고, **post-hoc encoder swap**만 가능하다. 이 변수를 격리하는 것이 본 논문의 목표.

---

## 2. 방법론

### Frozen-backbone grafting (Sec. 3.2)
공개 VLA의 단일 visual entry point를 monkey-patch로 가로채:
1. **Foreign encoder forward**: 후보 encoder(SigLIP / DINOv2-small / FastViT-SA12 / RepViT-M1)를 224×224 RGB에 forward. timm pretrained, frozen.
2. **Deterministic spatial pooling**: feature map을 AdaptiveAvgPool2d로 SmolVLA용 8×8=64 token 또는 π0.5용 16×16=256 token으로 정규화.
3. **Token-wise LayerNorm + linear projector**: 백본 hidden size(960 또는 2048)로 투영. **이 0.37M-1.58M projector만 학습 가능**, encoder/LM/action expert는 모두 frozen.

### Backbones (Sec. 3.1)
- **SmolVLA-450M**: SmolVLM2-500M-Video-Instruct + SigLIP-base/16-224 + pixel-shuffle 4× connector → (B, 64, 960). policy.model.vlm_with_expert.embed_image를 패치.
- **π0.5-3.3B**: PaliGemma-3B + shape-optimized SigLIP-So400m + multi_modal_projector → Gemma expert(hidden 2048), (B, 256, 2048). paligemma.model.vision_tower.forward를 교체.

### Training/eval (Sec. 3.3-3.4)
- LeRobot lerobot/libero_{spatial,object}_image (24,913 train / 6,457 val window, episode-split). T=50 chunk, delta-timestamps {i/10}^49.
- AdamW lr=1e-4, wd=1e-4, β=(0.9,0.999), no schedule, BF16, **2,000 steps**.
- SmolVLA: batch 8, single GPU; π0.5: micro-batch 2 + grad accum 4 (effective 8).
- Seed: SigLIP/DINOv2 셀은 {42,43,44} 3-seed, FastViT/RepViT 셀은 {42,43} 2-seed → 총 **40 main run**.
- 평가는 episode-split validation MSE/L1 (val_mse가 ranking metric). 동일 action-sampling seed, 동일 loader 순서.
- **closed-loop는 보고하지 않음**: SmolVLA/π0.5는 SO-100 데이터로 학습되었고 LIBERO Franka와 action space가 불일치하여 native/grafted 양쪽 모두 success ≈ 0.

---

## 3. 실험 결과

### Main grafting matrix (Table 1)
seed-averaged val_mse (낮을수록 좋음):

| Backbone | Suite | SigLIP | DINOv2 | FastViT | RepViT |
|---|---|---|---|---|---|
| SmolVLA | spatial | **0.0706** | 0.0734 | 0.0929 | 0.1557 |
| SmolVLA | object | **0.0628** | 0.0675 | 0.0794 | 0.1351 |
| π0.5 | spatial | 0.0267 | **0.0256** | 0.0283 | 0.0459 |
| π0.5 | object | **0.0215** | 0.0217 | 0.0221 | 0.0357 |

→ SmolVLA는 두 suite 모두 SigLIP 우승. π0.5-spatial은 DINOv2 우승, π0.5-object는 top 3가 2.7% 상대 범위의 near-tie band.

### Native anchor (Table 5)
| Backbone | Suite | Native MSE | Best grafted | Rel. |
|---|---|---|---|---|
| SmolVLA | spatial | 0.0544 | 0.0706 (SigLIP) | **+29.8%** higher |
| SmolVLA | object | 0.0475 | 0.0628 (SigLIP) | +32.1% higher |
| π0.5 | spatial | 0.0440 | 0.0256 (DINOv2) | **−41.8%** lower |
| π0.5 | object | 0.0379 | 0.0215 (SigLIP) | −43.2% lower |

→ 그러나 native-encoder-through-graft-interface control(Table 8)에서 π0.5의 −42~43% 개선은 거의 전부 **wrapper 자체의 효과**(−49.84%~−51.94%)로 설명됨.

### P2 LoRA ablation (Table 6)
rank-8 LoRA를 encoder 내부에 풀면 4개 셀 모두 −9%~−22% 개선. SmolVLA×SigLIP은 P1 0.0682 → P2 0.0534로 native(0.0544)와 2% 이내 근접.

### Pooling ablation (Table 7, SmolVLA-spatial-seed42)
Perceiver ≺ AvgPool ≺ AttnPool 순서가 SigLIP/DINOv2 모두에서 보존. 단 Perceiver는 ~60M trainable로 17% 개선 대비 100× 비용.

### Sanity controls (Sec. 4.13, Table 8)
- SmolVLA: vision-gap +13~25%, shuffle-gap +48~51% → spatial-action correspondence 진짜 사용.
- π0.5-object: vision-gap +95.5%, shuffle-gap +22.2%, encoder gap(1.7~7%) < shuffle band → narrow but real.
- **π0.5-spatial: vision-gap +50.5%, shuffle-gap +3.9%** → encoder가 low-frequency 이미지 통계(색 분포, texture marginal)에 의존, fine-grained spatial 구조가 아님.

---

## 4. 한계 및 미해결 문제

1. **Closed-loop 미보고**: SO-100/Franka embodiment 불일치로 모든 셀이 success ≈ 0. 정확히 이 regime이 본 진단의 존재 이유이지만, deployable policy로서의 가치는 0.
2. **Wrapper non-neutrality**: grafting harness가 SmolVLA에서 +45-56% 손해, π0.5에서 −50-52% 이득의 **opposite-sign** 효과. 모든 결론은 fixed grafting protocol 하에서만 성립.
3. **π0.5-libero_spatial은 shuffle-tolerant**: 이 셀의 encoder ranking은 spatial-action correspondence가 아니라 low-frequency 통계에 기인할 수 있어 weakest supporting cell.
4. **N=4 encoder, N≤3 seed**의 작은 그리드: cross-backbone Spearman ρ(+0.80/+1.00)이 positive인 것은 RepViT가 항상 꼴찌이기 때문이며 top-1 신뢰성과는 무관. permutation p-value(spatial 0.33, object 0.04)도 24개 perm 한계로 significance claim 미지원.
5. **OpenVLA 미포함**: harness 확장 가능하지만 현 grid에는 SmolVLA/π0.5만. 세 번째 backbone에서 sign이 어떻게 바뀔지 미지수.
6. **Encoder 사이즈 축**: SigLIP-Large나 DINOv2-Giant 같은 더 큰 encoder를 배제 — latency-quality Pareto가 달라질 수 있음.
7. **Projector-only 학습이 보수적**: stronger encoder가 단순히 linearly projectable하기 때문에 이기는 confound. P2 LoRA가 부분적으로 해소하지만 4-cell gate-level 수준.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — 새로운 정책이나 architecture가 아니라 "encoder ablation을 어떻게 하면 fair하게 할 것인가"라는 방법론 contribution. 단일 entry-point monkey-patch + 1.58M projector라는 minimal harness는 깔끔하다. |
| **Practical impact** | ★★★★☆ — π0.5/SmolVLA 같은 출시된 VLA를 사용하는 실무자가 encoder 교체를 결정하기 전에 < 6 GPU-hour로 돌릴 수 있는 cheap diagnostic. zero/shuffled-image control은 후속 encoder 논문이 반드시 채택해야 할 sanity check. |
| **Rigor** | ★★★★★ — 40 main + 8 native anchor + 4 LoRA + 6 pooling + 12 sanity control = 70+ run을 episode-split, fixed eval seed, 11/12 cell-level breakdown으로 정직하게 보고. 가장 약한 셀(π0.5-spatial shuffle-tolerant)을 자기검열로 표시한 것이 인상적. |

핵심 메시지는 단순하다: **"소형 VLA에서 가장 좋은 encoder가 대형 VLA에서도 가장 좋다고 가정하지 마라."** 그리고 이를 확인할 cheap protocol을 제공한다. 새로운 정책 모델은 아니지만, 향후 encoder 선택 논문의 기준선이 될 가능성이 있는 방법론적 논문.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Closed-loop success가 0인데 offline MSE가 ranking에 의미가 있나? | 저자들도 인정. 다만 동일 (backbone, suite, eval pipeline) 안에서 encoder만 교체한 within-condition contrast는 여전히 encoder 효과를 가르킨다. zero/shuffled-image control이 vision-utility를 확인. |
| 2 | π0.5-object에서 SigLIP 0.02149 / DINOv2 0.02166 / FastViT 0.02206이면 사실상 동률 아닌가? | 그렇다 — 저자들이 "near-tie band within 2.7% relative"로 명시하고 top-1을 주장하지 않음. seed-44 추가로 top-1 identity가 flip되는 것이 핵심 finding. |
| 3 | Wrapper가 π0.5에서 −50%, SmolVLA에서 +50%인 이유? | 가설(Sec 5.4): (i) SmolVLM SigLIP은 이미 (B,64,960)에 맞춰진 token grid라 추가 pool이 destructive; PaliGemma SigLIP-So400m은 더 큰 grid를 emit해 wrapper의 16×16 pool이 useful bottleneck. (ii) Gemma 2048 expert가 unfamiliar token을 re-interpret할 수 있음. 둘 다 testable, future work. |
| 4 | RepViT가 항상 꼴찌인데 cross-backbone correlation이 positive로 나오는 이유? | bottom-of-pool ordering(RepViT < 나머지)이 양쪽 backbone에서 동일해 ρ가 positive로 끌려옴. 하지만 top-1 stability는 별개라는 것이 본 논문의 주된 주장. |
| 5 | 왜 OpenVLA에서는 안 했나? | OpenVLA의 native embodiment가 SmolVLA/π0.5보다 LIBERO Franka에 가깝지만 7B-class라 compute가 더 큼; harness는 확장 가능하다고만 명시. 후속 작업. |

<!-- VERIFIED: pdf -->
