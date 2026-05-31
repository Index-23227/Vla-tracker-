# GR00T-N1.5: NVIDIA's Open Humanoid Foundation Model with FLARE Objective

> **한 줄 요약**: NVIDIA GEAR Lab이 Computex 2025에서 공개한 GR00T-N1 dual-system 후속작. **Eagle 2.5 VLM(2.1B, frozen)** + diffusion transformer action head에 **FLARE(Future LAtent Representation Alignment) objective** (loss 계수 0.2)를 추가하여 action label 없는 human/heterogeneous 비디오를 사전학습에 통합. 250K steps · 1K H100 · global batch 16384 학습 후 HuggingFace에 가중치 공개(nvidia/GR00T-N1.5-3B). **DreamGen 12-task 38.3%** (N1 13.1%), **RoboCasa 30-demo 47.5%** (N1 17.4%), **Real GR-1 language following 93.3%** (N1 46.6%), **Unitree G1 post-training 98.8%** (N1 44.0%).

> 본 리뷰는 NVIDIA 공식 프로젝트 페이지(research.nvidia.com/labs/gear/gr00t-n1_5/)의 수치를 PDF 수준의 1차 출처로 활용. arXiv 논문은 별도 발표되지 않았다.

---

## 1. 배경 및 동기

- **GR00T-N1(2025-03)** 은 humanoid용 open foundation model로 RoboCasa 32.1%, GR-1 tabletop 50.0%를 달성했지만 한계가 남았다:
  - 인간 비디오를 action label 없이 활용하는 방법 부재.
  - DreamGen·neural trajectory 같은 합성 데이터 통합이 ad-hoc.
  - Multi-embodiment(Unitree G1 등)에서 빠른 적응이 어려웠음.
- 연구 질문:
  1. action label 없는 대규모 video를 어떻게 robot policy 학습에 사용할 것인가?
  2. humanoid 데이터 희소성을 합성으로 메우면서도 학습 효율을 어떻게 유지할 것인가?
  3. 다양한 embodiment(Fourier GR-1, Unitree G1)에 단일 모델로 빠르게 적응할 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 Dual-System Architecture
- **System 2 — Eagle 2.5 VLM (2.1B, frozen)**: vision + language reasoning. N1.5에서는 **frozen** 으로 두고 위에 가벼운 **MLP adapter + LayerNorm** 만 추가. (N1은 어댑터를 학습했음.)
- **System 1 — Diffusion Transformer**: action chunk 생성 head, VLM embedding에 cross-attention.

### 2.2 FLARE — Future LAtent Representation Alignment
- **목적**: action label이 없는 human/heterogeneous video를 representation learning에 활용.
- **방법**: 현재 시점 representation으로부터 미래 latent를 직접 예측·정렬. 프레임을 generatively 모델링하는 대신 **target future embedding과의 alignment loss**를 추가.
- **loss 계수**: pretraining·posttraining 모두 **0.2** 로 고정.
- 효과: action-free 데이터에서도 시간적 인과 representation을 얻음 → robot embodiment로 transfer.

### 2.3 GR00T-Dreams Synthetic Pipeline
- 텍스트·이미지 조건으로 새로운 robot task 영상을 생성하는 neural trajectory pipeline.
- **DreamGen 12 verb suite**로 zero/few-shot 적응 평가 → N1 13.1% → **N1.5 38.3%** (+25.2pp).

### 2.4 학습 규모

| 항목 | 값 |
|---|---|
| 파라미터 | 3B (VLM 2.1B + DiT action expert) |
| GPU | 1K × H100 |
| Steps | 250K |
| Global batch | 16384 |
| Optimizer | AdamW + cosine LR (warmup ratio 0.05) |
| 데이터 | Real GR-1 + Sim GR-1(DexMG) + DreamGen 합성 + OpenXE + AgiBot-Beta |

VLM frozen + 가벼운 adapter라는 디자인 덕에 250K step도 36시간 가량의 wall-clock으로 가능했다고 보고.

---

## 3. 실험 결과 (NVIDIA 공식 페이지 Table 검증)

### 3.1 Architecture Validation (Sim)

| Benchmark | GR00T-N1 | **GR00T-N1.5** | Δ |
|---|---|---|---|
| Language Table | 52.8% | **93.2%** | +40.4 |
| Sim GR-1 Language following | 36.4% | **54.4%** | +18.0 |

### 3.2 Data-Limited Post-Training

| Setting | GR00T-N1 | **GR00T-N1.5** | Δ |
|---|---|---|---|
| RoboCasa 30 demos | 17.4 | **47.5** | +30.1 |
| Sim GR-1 zero-shot | 39.6% | **43.9%** | +4.3 |
| Sim GR-1 30 demos | 43.2% | **47.4%** | +4.2 |

### 3.3 Real GR-1 Humanoid

| Metric | GR00T-N1 | **GR00T-N1.5** | Δ |
|---|---|---|---|
| Language following rate | 46.6% | **93.3%** | +46.7 |
| Overall success rate | 43.3% | **83.0%** | +39.7 |

### 3.4 Novel Object Generalization

| Setting | GR00T-N1 | **GR00T-N1.5** |
|---|---|---|
| 0-shot | 0% | **15.0%** |
| Post-trained on human videos | – | **55.0%** |

→ **FLARE의 직접 증거**: 인간 비디오만으로 새 객체 일반화가 55%까지 도달.

### 3.5 Cross-Embodiment

| Embodiment | GR00T-N1 | **GR00T-N1.5** |
|---|---|---|
| Unitree G1 post-training | 44.0% | **98.8%** |

### 3.6 외부 보고 / 본 트래커 참조 수치
- **DreamGen 12-task**: 38.3% (N1: 13.1%) — NVIDIA research page.
- **RoboCasa Kitchen 100-demo, 24-task**: 64.1% (HAMLET paper baseline).
- **Vlaser paper Table 3** (SimplerEnv): GR00T-N1.5(2.1B) Visual Matching 52.4%, Variant Aggregation 43.7%.

---

## 4. 어블레이션 (페이지에 명시된 비교)

- N1 vs N1.5는 모든 축에서 N1.5 우세하지만 **개별 변경(FLARE / frozen VLM / adapter 단순화 / GR00T-Dreams)** 의 분리 ablation은 공개되지 않음.
- FLARE의 loss 계수 0.2 단일값만 보고 — 0 / 0.1 / 0.5 등 sweep 부재.

---

## 5. 한계

| # | 한계 | 코멘트 |
|---|---|---|
| 1 | arXiv 부재 | NVIDIA blog + project page + HF model card뿐. peer review 미통과 |
| 2 | 표준 manipulation 벤치 미직접 보고 | LIBERO/CALVIN 공식 보고 없음. SimplerEnv는 외부(Vlaser)에서 baseline으로 인용 |
| 3 | 개별 컴포넌트 ablation 부재 | FLARE 단독·VLM freeze·GR00T-Dreams 데이터의 marginal contribution 분리 불가 |
| 4 | RoboCasa 보고 분산 | NVIDIA 페이지의 30-demo 47.5%와 HAMLET 페이퍼의 100-demo 64.1% — 평가 protocol 서로 다름 |
| 5 | 하드웨어 의존 | Fourier GR-1·Unitree G1 등 특정 humanoid에 특화. 새 embodiment로의 zero-shot transfer는 추가 데이터 필요 |
| 6 | 재현성 | weights는 open, 학습 코드/데이터(GR00T-Dreams)는 closed |
| 7 | 단일 3B scale | 더 작은/큰 사이즈 변종 부재 |

---

## 6. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — FLARE objective + frozen VLM + GR00T-Dreams 통합 |
| Technical depth | ★★★☆☆ — 핵심 수치는 공개되었으나 paper-grade 디테일 부족 |
| Experimental rigor | ★★★★☆ — N1과 직접 비교한 다축 벤치마크 다수 |
| Practical impact | ★★★★★ — HF open weights, humanoid 생태계 핵심 |
| Writing/Communication | N/A — 공식 paper 부재 |

**강점**: open weights, FLARE로 unlabeled video 활용, frozen VLM으로 36h 단기 학습, novel object 55% 같은 generalization 증거. **약점**: 학술 검증 부재, 컴포넌트 ablation 부재, 표준 벤치(LIBERO/CALVIN) 미보고.

---

## 7. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | FLARE가 R3M·VIP 같은 video pretraining과 무엇이 다른가? | "**Future** latent representation alignment" — past contrastive가 아닌 미래 latent와의 정렬. loss 계수 0.2로 main BC loss를 보조. 그러나 정확한 loss 형식은 별도 FLARE 페이지 참조로 우회. |
| 2 | N1 → N1.5의 핵심 driver는 FLARE인가 VLM freeze인가? | 분리 ablation 부재. 페이지는 "architecture changes + FLARE 둘 다"라고만 표현. Novel object 0→15% 는 FLARE 기여 가능성 큼. |
| 3 | 250K steps가 36시간 — 정말 from scratch인가? | Eagle 2.5 VLM·DiT 모두 prior weight 사용. 36h는 GR00T-Dreams 데이터로의 fast adaptation을 뜻함. from-scratch 아님. |
| 4 | LIBERO 미보고 이유? | humanoid 지향 — single-arm 벤치는 우선순위 낮음. 단 직접 비교 가능성 손실. |
| 5 | RoboCasa 64.1(N1.5) vs 47.6(N1.6) — N1.5가 더 나은가? | 평가 protocol 자체가 다름 (Kitchen 100-demo vs GR1 Tabletop 24-task). 동일 setting 비교 부재. |
| 6 | Unitree G1 98.8%는 너무 높은데 평가가 쉬운 task인가? | 특정 post-training task에서의 saturation으로 추정. 평가 task 종류 명시 부족. |
| 7 | FLARE loss 계수 0.2 — sweep 결과는? | 페이지 명시 없음. 일반적 multi-task weighting trick과 동일하게 시행착오로 추정. |

<!-- VERIFIED: pdf -->
