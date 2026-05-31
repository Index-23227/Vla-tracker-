# GridS: See What Matters — Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model

> **한 줄 요약**: VLA의 256-token visual encoder 출력 중 task-critical 영역만 남기되, 기존 token-pruning이 사용하는 **이산적 인덱스 선택** 대신 **연속 (x, y) 좌표** 를 예측해 **differentiable bilinear sampling** 으로 sub-patch precision의 visual token K개(K≪256)를 추출. 16-token 설정에서 **FLOPs 76% 감소, LIBERO 94.4% → 96.0%** (π0 기준), 그리고 SmolVLA 실로봇에서 baseline 7.6% → 60.0% (Stack Cubes), OOD 38.1%. ICML 2026. **Code: github.com/Fediory/Grid-Sampler**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA는 256개 visual token을 LLM에 공급 → 추론 비용이 token 수에 quadratic.
- 기존 token-pruning(FastV, SparseVLM, VLA-Cache)은 **patch-level discrete selection** — 어느 patch index를 keep할지 binary 선택.
- Discrete selection의 두 가지 문제:
  1. **Quantization error**: 16×16 patch grid 내에서 contact-relevant 미세 영역(gripper-object contact, edge)이 patch 경계 위에 걸치면 한 patch 통째로 keep 하거나 drop → spatial precision 손실.
  2. **Non-differentiable**: argmax/top-k 선택은 미분 불가능 → straight-through estimator나 Gumbel trick으로 우회 → 학습 신호 약함.

### 핵심 질문
- **Visual token 선택을 연속 좌표 공간에서 직접 학습할 수 있는가?**
- **Sub-patch precision sampling이 동일 token 수에서 더 좋은 성능을 내는가?**

📌 [Figure 1 삽입] — Discrete patch selection vs. continuous grid sampling 비교 (contact-point 손실 사례).

---

## 2. 방법론 심층 분석

### 2.1 연속 좌표 예측

GridS는 lightweight coordinate head를 도입:
- Input: VLA의 visual encoder 출력 (e.g., SigLIP 256-token feature map H×W=16×16).
- Output: K개의 연속 좌표 {(x_k, y_k)}, x_k, y_k ∈ ℝ (sub-patch precision).
- K는 hyperparameter; 기본 K=16 (256→16, 16× 압축).

### 2.2 Differentiable Bilinear Sampling

각 좌표 (x, y)에서 인접 4 patch (P_00, P_01, P_10, P_11)의 feature를 bilinear interpolation:
```
F_sampled(x, y) = Σ_k ω_k · P_k
ω_k = bilinear weights (linear in fractional x, y)
```
- ω가 (x, y)에 대해 미분 가능 → 좌표 예측 head가 end-to-end gradient로 학습.
- Sub-patch precision: x=3.4, y=7.7 같은 비정수 좌표 가능 → contact point 같은 미세 영역 정확히 sampling.

> ❓ **예상 질문**: Bilinear sampling은 Spatial Transformer Networks (Jaderberg et al., 2015)에서 이미 잘 알려진 기술 아닌가?
> **답변**: 맞음. GridS의 novelty는 (a) STN을 VLA visual token pruning에 적용한 첫 사례, (b) coordinate head를 host VLA와 **end-to-end co-training**, (c) sub-patch precision이 contact-rich manipulation에서 실질적 이득을 가져옴을 보임.

### 2.3 End-to-End Co-training

Coordinate head + host VLA(π0 / π0.5 / SmolVLA)를 단일 loss(action prediction)로 jointly 학습. 별도 supervision 없이도 coordinate head가 task-relevant region을 학습.

📌 [Figure 2/3 삽입] — Learned coordinates의 시각화 (gripper, object center 등에 자연스럽게 수렴).

---

## 3. 데이터 전략

| Host | Dataset | Steps | Batch | LR |
|------|---------|-------|-------|-----|
| π0 / π0.5 | LIBERO 전체 | 30,000 | 32 | 5e-5 |
| SmolVLA | SO100 (real-world Stack Cubes) | 50,000 | 16 | 1e-4 |

추가 데이터 없이 기존 train set으로 co-training만 수행.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Token budget K | {4, 16, 32}; default 16 |
| FLOPs (256→16) | 216.01G → 51.65G (**−76%**) |
| Training speedup | 3.4× (π0), 2.9× (π0.5) |
| Inference speedup | 1.2× @ batch 1, **3.2× @ batch 128** |
| Coordinate head | lightweight MLP (parameter overhead negligible) |

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (π0 host, Table 1)

| Suite | π0 baseline (256) | + GridS16 | + GridS4 |
|-------|-------------------|----------|----------|
| Spatial | 97.2 | **98.0** | 96.6 |
| Object | 98.8 | **99.2** | 99.4 |
| Goal | 96.0 | **96.4** | 96.4 |
| Long | 85.6 | **90.2** | 89.6 |
| **Avg** | 94.4 | **96.0** | 95.5 |

- **16× 압축에서 baseline 대비 +1.6%p 향상** — pruning이 성능 저하가 아니라 향상을 가져오는 매우 이례적 결과.
- Long-horizon에서 +4.6%p 가장 큰 향상 → noise visual token이 제거되며 정책의 long-horizon decision이 안정화됨을 시사.

### 5.2 LIBERO (π0.5 host)

| Suite | π0.5 baseline | + GridS16 | + GridS4 |
|-------|---------------|----------|----------|
| Avg | 96.7 | **97.7** | 96.7 |

→ 더 강한 baseline(π0.5)에서도 GridS16가 +1.0%p 향상.

### 5.3 실로봇 (SmolVLA on SO100)

| Task | Baseline | + GridS |
|------|---------|---------|
| Stack Cubes | ~7.6% | **60.0%** |
| Stack Cubes OOD | ~0% | **38.1%** |

- **Stack Cubes에서 +52.4%p의 극적 향상** — baseline SmolVLA가 fundamental하게 실패하던 task를 GridS가 풀어냄.
- OOD 38.1%는 visual token reduction이 단순 효율 기법이 아니라 **regularizer로 작동** 한다는 강력한 증거.

### 5.4 효율성

- **FLOPs**: 216.01G → 51.65G (76% ↓).
- **Training**: π0에서 **3.4× speedup**.
- **Inference**: 배치 큰 경우(B=128) **3.2× speedup**, B=1에서는 1.2×(작은 배치에서는 visual encoder가 bottleneck이 아니라 LLM이라 효과 작음).

---

## 6. 관련 연구 비교

| 방법 | 선택 방식 | Differentiable | Sub-patch | LIBERO Δ |
|------|----------|---------------|----------|---------|
| FastV | top-k attention | × (STE) | × | 보통 −1~−3%p |
| SparseVLM | adaptive sparsity | △ | × | 보통 0~−2%p |
| VLA-Cache | temporal caching | × | × | 보통 0%p |
| **GridS** | **연속 좌표 + bilinear** | **✓** | **✓** | **+1.6%p** |

- **유일하게 positive Δ** 를 보이는 pruning 기법.

---

## 7. 한계 및 미해결 문제

1. **B=1 inference speedup 1.2×**: real-time robot control(보통 single-batch)에서는 효율 이득이 제한적. LLM이 bottleneck이라 visual encoder pruning만으로는 부족.
2. **Coordinate head 학습의 stability**: bilinear sampling은 미분 가능하나 gradient가 (x, y) 좌표를 통해 흐를 때 학습 안정성/수렴성에 대한 분석 부재.
3. **K=16의 일반성**: LIBERO/실로봇에서 K=16이 sweet spot이지만, dexterous / multi-object task에서는 더 많은 token이 필요할 수 있음. K의 자동 선택 부재.
4. **OOD 일반화의 이유 불명**: GridS가 OOD에서 baseline 0%→38%는 큰 향상이지만, regularization 효과인지, denoising 효과인지, 정확한 메커니즘 분석 없음.
5. **Multi-view setting 미검증**: GridS는 single-view를 가정. Multi-camera VLA에서 view별로 별도 sampling이 필요한지 미검증.
6. **Author affiliations 정보 부재**: arXiv listing에 affiliation 표기 없음 (저자 7명) — 재현/communication에 약간의 불편.
7. **ICML 2026 venue 표기**: 본문에서는 명시되어 있으나 abstract listing에서만 보임 — 정식 proceedings 확인 필요.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — STN-style sampling을 VLA token pruning에 적용한 첫 사례. 단순하지만 효과적. |
| **Technical depth** | ★★★★☆ — 수식은 간단하나 end-to-end co-training의 실용적 효과가 큼 |
| **Experimental rigor** | ★★★★★ — π0 / π0.5 / SmolVLA 세 backbone, LIBERO + 실로봇 + OOD, K-sweep |
| **Practical impact** | ★★★★★ — Code 공개, 76% FLOPs reduction, 성능 향상 — 즉시 채택 가능 |
| **Writing quality** | ★★★★☆ — 명확한 motivation과 ablation, 다만 OOD mechanism 분석 부재 |

**강점**: 압축하면서도 **성능이 오르는** 흔치 않은 결과. 16× token 압축에서 LIBERO +1.6%p, 실로봇 +52.4%p. STN을 VLA에 적용한 단순함이 오히려 강력. **약점**: B=1 inference에서 효율 이득이 제한적, OOD 일반화의 mechanism 분석 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Pruning을 했는데 성능이 오르는 이유는? | 두 가지 가설: (a) Sub-patch precision으로 contact point를 더 정확히 sampling, (b) 256 visual token 중 다수가 noise이고 16개로 줄이면 noise 제거 효과. Long-horizon (+4.6%p)에서 가장 큰 향상이 후자 가설 지지. |
| 2 | STN(2015)과 본질적으로 무엇이 다른가? | 기술적으로 동일한 bilinear sampling이나 (a) VLA pretrained visual encoder feature map에 적용, (b) host VLA와 end-to-end co-train, (c) action prediction loss로 좌표 학습 — 새로운 응용 도메인과 학습 방식이 contribution. |
| 3 | K=16이 너무 적지 않은가? FastV 등은 보통 64-128개 token을 keep. | LIBERO/SO100 같은 단순 manipulation에서는 16개 contact/object center로 충분. Dexterous task에서는 K↑ 필요할 수 있음. K=32에서도 동등 성능이라 16이 효율-정확도 trade-off optimum. |
| 4 | Coordinate head가 어디로 수렴하는가? | Visualization에서 gripper, object center, 그리고 contact region에 자연스럽게 수렴. 어떤 explicit supervision 없이 action loss만으로 학습되는 emergence. |
| 5 | OOD에서 baseline 7.6%→60%는 과한 향상 아닌가? | SmolVLA baseline이 SO100에서 매우 약한 상태였음 (small model). GridS의 visual focus가 noisy distractor를 제거해 base의 action-relevant 부분만 남김. 큰 model에서는 이정도의 향상은 보이지 않을 가능성. |
| 6 | Inference speedup B=1에서 1.2×에 불과한 이유는? | 추론 시 LLM forward가 bottleneck. Visual token 16개라도 LLM forward 비용은 거의 동일. B=128에서 visual encoder의 GPU memory/bandwidth가 saturate되어 token reduction의 효과가 크게 보임. |
| 7 | 다른 token pruning과의 직접 head-to-head는? | FastV, SparseVLM, VLA-Cache와 비교 — 모두 GridS보다 LIBERO에서 열위. 정확한 표 수치는 본문 참조. |
| 8 | Code가 공개되어 있다면 재현 가능한가? | github.com/Fediory/Grid-Sampler 에 공개. 다만 affiliation이 표기되지 않아 institutional support / long-term maintenance에 의문. |
| 9 | LongCat-Image-Edit, VGGT 같은 multi-view 기법과 결합 가능한가? | 원리적으로 호환. GridS는 visual encoder output에 적용되므로 multi-view encoder 출력에도 동일 적용 가능. 미검증. |
| 10 | π0.5 baseline 96.7%에서 +1.0%p는 ceiling effect가 아닌가? | 가능. 다만 long-horizon suite에서 여전히 향상폭이 큼 → ceiling은 long-horizon에서 아직 미도달. |

<!-- VERIFIED: pdf -->
