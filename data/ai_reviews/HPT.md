# HPT: Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers

> **한 줄 요약**: MIT/Meta FAIR(Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He)가 제안한 stem-trunk-head 모듈러 아키텍처로, 52개 이질적 robot dataset(real teleop + simulation + deployed robots + human videos, 200K trajectory)에서 sharable transformer trunk를 사전학습하고 새 embodiment로 transfer 시 fine-tuned 성능을 20%+ 향상시킨 NeurIPS 2024 Spotlight.

---

## 1. 배경 및 동기 (§1)

로봇 데이터의 핵심 난점은 **heterogeneity**: 각 embodiment가 서로 다른 proprioception 차원, end-effector, action space, sensor 구성, environment를 가진다(§1, footnote 1). 기존 generalist policy(RT-X, Octo, OpenVLA)는 (1) vision pre-training만 하고 proprioception은 후행 추가하거나, (2) language로 통일된 단일 dataset format을 강제. 저자들은 stem이 embodiment마다의 sensor를 16-token short sequence로 정렬하고, shared trunk가 pre-train되도록 분리한다(Fig. 1-2).

---

## 2. 방법론 심층 분석 (§3, Fig. 2-3, Table 1)

### 2.1 Stem (§3.1, Fig. 3)

- **Proprioceptive tokenizer**: dim d_p (예: 7 EEF pose) → MLP → sinusoidal PE → cross-attention with 16 learnable tokens(N_p=16, dim 128~1024).
- **Vision tokenizer**: 사전학습 frozen ResNet18 feature → cross-attention with 16 learnable tokens(N_v=16). 다중 view/시간을 단일 short token sequence로 압축.

### 2.2 Trunk (§3.1, Table 1)

5개 크기: HPT-Small(3.1M), Base(12.6M), Large(50.5M), XLarge(226.8M), Huge(1.1B). 모두 depth 16~80, width 128~1024. Trunk는 embodiment 수와 무관하게 고정.

### 2.3 Head

Embodiment-task별 MLP. Pooled trunk feature → 정규화된 action trajectory.

### 2.4 Training (§3.2, Eq. 1)

L = Huber loss (정규화 action vs 예측). Trunk만 매 iteration 업데이트, stem/head는 sampled batch에 따라 업데이트. Default 27 datasets, Scaled 52 datasets/170k trajectory/2048 batch (Table 2).

---

## 3. 실험 결과 심층 분석

### 3.1 Pre-training Scaling (§4, Fig. 5-7)

- **Data scaling (Fig. 5a)**: trajectory 수 270 → 170K로 늘릴 때 validation loss 단조 감소; 그러나 model size를 함께 키워야 함(blue line은 1K traj 부근에서 plateau).
- **Dataset scaling (Fig. 5b)**: 10 → 52 dataset으로 늘릴수록 validation loss 개선.
- **Model scaling (Fig. 7)**: 1M → 1B param까지 데이터/배치를 동시에 늘리면 validation loss 단조 감소(red line). Depth/width 차이는 미미.
- **Sim+Human videos joint pre-training (§4.3, Fig. 8)**: 7 sim datasets + EPIC kitchen + PoCo(300 traj × 2)도 joint pre-training 가능 입증.

### 3.2 Transfer Learning (§5)

- **Simulation (Fig. 10a)**: Meta-world / RoboMimic / Fleet-Tools 3 벤치마크 50 episode 평가. HPT-B → HPT-XL transfer 시 from-scratch 대비 일관 향상.
- **Simpler benchmark (Fig. 10b)**: Google EDR Close Drawer, Move Near, Pick Coke Can — Octo / RT-1-X / RT-2-X 대비 경쟁력 확보(>300 episode 총합).
- **Real-world (Fig. 12, Table 3)**: Sweep Leftover, Fill Water, Scoop Food, Switch Insertion (45 trial 평균). Sweep Leftover에서:

| Method | Success (%) |
|---|---|
| From Scratch | 43.3±3.8 |
| R3M | 50.0±3.0 |
| Voltron | 46.7±3.8 |
| VC-1 | 53.3±2.6 |
| **HPT-B Finetuned** | **70.0±3.0** |
| **HPT-XL Finetuned** | **76.7±3.3** |

학습 가능 파라미터(stem+head)는 전체의 **2%** 수준. Inference time: HPT-B는 RTX 3070에서 47Hz, HPT-XL은 19Hz; A100은 3-4× 가속.

---

## 4. 한계 (§6)

1. **Embodiment split이 단순**: dataset mixture의 균형이 거친 수준이며 정교한 curation 부족.
2. **Pre-training objective가 단순 BC**: supervised Huber만 사용; self-supervision/contrastive는 미탐구.
3. **Heterogeneous pre-training의 수렴 속도**: 모듈러 구조 때문에 느림(§6).
4. **단기 horizon, 고정 embodiment 평가**: real eval은 short-horizon manipulation 위주 (§6).
5. **신뢰성 한계**: 실제 평가 task SR이 일반적으로 90% 미만.
6. **LIBERO/SimplerEnv 직접 보고 부재**: 본 paper에는 LIBERO 점수가 직접 보고되지 않음. **YAML의 `libero_spatial=86.4 / object=89.2 / goal=78.6 / long=55.8`은 paper의 Table 2가 아닌 외부 평가**(YAML 주석은 "Table 2, HPT paper"라고 적혀 있으나 실제 paper의 Table 2는 데이터셋 통계 — YAML 메타데이터 오류 가능성 flag).

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — Stem-trunk-head 모듈러 + heterogeneous alignment |
| Technical depth | ★★★★★ — 1B까지 scaling, 52 dataset 통합 |
| Experimental rigor | ★★★★★ — Pre-train val + transfer 모두 |
| Practical impact | ★★★★★ — Open-source, generalist robot backbone의 청사진 |

**강점**: Heterogeneity 문제의 명확한 formulation, scaling law 실증, 실전 transfer 성능. **약점**: BC objective 단순, real-task SR 90% 미만, language conditioning 부재.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | OpenVLA처럼 VLM을 trunk로 쓰면? | HPT는 language-free; VLM trunk가 multi-modal grounding엔 좋으나 본 작업의 목표는 proprio-vision alignment에 한정. |
| 2 | 16 token이 너무 짧지 않은가? | Octo는 256 token 사용. 짧은 sequence는 inference 효율 + scaling 유리(§3.1)이나 fine-grained 정보 손실 가능. |
| 3 | Negative transfer를 분석했는가? | Fig. 5b에서 dataset 수 증가 시 일관된 개선 보고 — 큰 negative transfer는 관찰되지 않음. 그러나 dataset별 기여도 분석은 부분적. |

<!-- VERIFIED: pdf -->
