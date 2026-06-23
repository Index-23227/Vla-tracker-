# X-Tokenizer: A Multimodal Action Tokenizer for VLA Pretraining

> **한 줄 요약**: Action tokenization을 단순 압축이 아닌 **VLM과 robot control 사이의 semantic interface 학습**으로 재정의. Residual VQ에 비대칭 supervision을 가하는 **Semantic Residual Quantization (SRQ)** — top-level은 Masked Action Modeling(BERT-style)으로 "discrete action language"가 되고, 깊은 level들은 reconstruction residual을 담당. 2.4M trajectory / 2.0B frame / 17 arm family로 pretraining 후 frozen 상태로 **Wall-OSS + Flow Matching** policy에 붙여 RoboTwin 2.0 dual-arm 50-task에서 Easy 84.7% / Hard 80.88% / Avg 82.79%를 달성, π0·π0.5 baseline을 능가. 실제 deployment 시에는 X-Tokenizer 자체가 inference path에서 제거되므로 추가 비용 0.

---

## 1. 배경 및 동기 (Section 1, 2)

- **표상 mismatch**: pretrained VLM은 의미론적으로 구조화된 discrete token 공간에서 동작하는 반면, robot policy는 정밀한 연속 motor command를 생성해야 함. action tokenizer는 그 둘을 잇는 다리지만 기존 tokenizer는 *compression module*로만 설계됨.
- **Hybrid discrete-continuous VLA의 함정** (Section 1): discrete action-token 예측 loss는 단순한 보조 목적이 아니라 **shared hidden state를 형성하는 supervisor** 역할을 함. token target이 임의의 reconstruction index이면 autoregressive loss가 VLM hidden state를 *geometric code pattern* 쪽으로 끌어당겨 multimodal grounding을 침식시킴.
- **기존 한계**: FAST·VQ-BeT·VQ-VLA·FASTer는 signal fidelity는 강하지만 VL 표상과 명시적으로 정렬되지 않음. ActionCodec은 contrastive supervision을 시도하나 frozen VLM 표상 공간에 직접 anchor되지 않고 semantic intent / residual을 분리하지 않음.
- 핵심 주장: action tokenizer는 *"representation-shaping target"* 이어야 한다 — 두 요건이 동시에 필요: (a) 코드가 backbone과 의미적으로 정렬, (b) 충분한 low-level detail로 정밀 reconstruction.

---

## 2. 방법론: Encoder-SRQ-Decoder (Section 3)

### 2.1 전체 구조 (Eq. 1)

`a_{t:t+T-1} -[Encoder E_theta]-> h_{1:M} -[SRQ Q_psi]-> tau_{1:M} -[Decoder D_phi]-> a_hat_{t:t+T-1}`

- **Encoder**: Perceiver-style cross-attention로 T=64 frame delta-action chunk를 M=16 latent slot으로 downsample. **Delta action** 사용 (absolute가 아닌 직전 proprioceptive anchor o 기준 offset) — embodiment 간 공유 가능한 motion pattern으로 codebook을 절약. 학습된 embodiment token m에 CFG-style none slot 포함하여 unseen embodiment robustness 확보.
- **SRQ (핵심)**: Q-level Residual Vector Quantization, **비대칭 supervision**. 표준 RVQ는 모든 level이 동일한 reconstruction loss를 받아 perplexity가 uniform해지고 level별 역할 분화가 안 됨 → SRQ는 이 자연스러운 구조 ("무엇을 할지" coarse intent vs "어떻게 할지" fine correction)를 RVQ depth에 mapping.
- **Decoder**: Perceiver IO-style read-out head, 가볍게 설계. 대부분의 capacity는 encoder + SRQ에 위치.

### 2.2 Joint Pretraining Loss (Eq. 3)

```
L_pre = L_rec + lambda_mam · L_mam + lambda_align · L_align + lambda_pred · L_pred
```

- **L_rec**: action reconstruction (모든 RVQ level이 받음).
- **L_mam (Masked Action Modeling, Section 3.3, Eq. 4)**: top-level discrete index c^(1)_{1:M}에만 BERT-style masked prediction. 일부 위치를 mask하고 작은 Transformer가 주변 context로 복원 → top-level code stream을 **자체 예측 가능한 internal action language**로 만듦.
- **L_align**: pre-quantization latent h_{1:M}을 frozen Qwen2.5-VL-7B로부터 추출한 fused VL feature와 contrastive alignment.
- **L_pred**: quantized latent z_tilde_{1:M}에서 next-frame VL feature를 예측 — 물리적 결과 (forward-aware) 정보 보존.

> **비대칭의 핵심**: 오직 첫 RVQ level만 discrete-level semantic supervision (MAM)을 받음. 깊은 level (q>1)은 supervision 없이 residual error만 흡수 → "Zipf vs Uniform" 분포 (§4.2.1).

### 2.3 Deployment-time 비대칭 설계

- 세 auxiliary head는 **pretraining에만** 사용되고 deployment에서 제거됨.
- Downstream VLA co-training: frozen X-Tokenizer가 expert trajectory를 offline에서 multi-level token으로 encoding하여 autoregressive supervision 제공. **Inference 시점에는 X-Tokenizer가 호출되지 않음** — 연속 Flow Matching expert만 hidden state에서 action을 regress.

---

## 3. SRQ Codebook Structure 검증 (Section 4.2.1, Fig. 5)

| Level | Active % | 역할 |
|-------|---------:|------|
| q0 (MAM-regularized) | **76.4%** | Long-tailed "motion words", 빈도가 4 order of magnitude로 분포 (Zipf-like) |
| q1 | 93.8% | Reconstruction residual |
| q2 | 99.3% | Reconstruction residual |
| q3 | 99.8% | Reconstruction residual |

- Top-level은 의도적으로 long-tail → 자주 쓰이는 "motion word" 집합 + rare pattern. 깊은 level은 uniform → residual correction. **collapse 없음** (모든 level >10%).
- 이는 SRQ가 "coarse intent + residual correction"의 hierarchy를 실제로 만들어냈음을 입증.

---

## 4. Tokenizer Ablation (Section 4.2.2, Table 1)

Reconstruction L1 (lower is better) vs RVQ perplexity (q0 낮을수록 집중, q3 높을수록 broad usage).

| Method | L1 | Δ% vs FAST | PPL q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|---:|---:|
| FAST | 0.01446 | — | — | — | — | — |
| 256-bin uniform | 0.00486 | -66% | — | — | — | — |
| X-Tok no aux | 0.00815 | -44% | 751 | 693 | 756 | 757 |
| w/o Align+Pred | 0.00830 | -43% | 687 | 904 | 853 | 793 |
| w/o MAM | 0.01564 | +8% | 603 | 677 | 830 | 871 |
| **X-Tokenizer (full)** | 0.01693 | +17% | **510** | 700 | 828 | **916** |

- Full 모델만 "monotone increasing PPL" (510 → 700 → 828 → 916) — SRQ가 의도한 intent/residual 분리가 완성됨.
- **Trade-off**: reconstruction L1은 256-bin/no-aux 대비 *나쁨* (의도된 결과). 단순 reconstruction이 아니라 downstream policy가 사용할 **structured semantic interface**를 만드는 게 목적이기 때문.
- MAM 제거 시 q0가 집중되지 않고, Align+Pred 제거 시 deeper level 정렬이 무너짐 → 두 supervision이 상보적.

---

## 5. Noise Robustness (Section 4.2.3, Table 2)

Word Error Rate (lower is better), Gaussian noise σ 주입.

| σ | **X-Tokenizer** | FAST | 256-bin | RDT2 VQ |
|---|---:|---:|---:|---:|
| 0.004 | **0.313** | 0.313 | 0.454 | 0.325 |
| 0.006 | **0.437** | 0.899 | 0.533 | 0.439 |
| 0.008 | **0.526** | 1.445 | 0.597 | 0.549 |

- FAST는 BPE re-segmentation 때문에 noise가 커질수록 WER이 폭증 (시퀀스 길이 자체가 변함).
- X-Tokenizer는 변화가 주로 q1:3 (residual)로 흡수되고 q0 (intent)은 안정 — backbone이 보는 "coarse action label"이 noise에 강건.

---

## 6. RoboTwin 2.0 결과 (Section 4.3, Appendix C.1 Table 6)

**Setting**: Wall-OSS (Qwen2.5-VL-3B + Flow Matching expert) 공개 checkpoint에 frozen X-Tokenizer 부착 후 70k step fine-tune. 50 dual-arm task × 100 rollout × {Easy=Clean, Hard=Randomized strong DR}.

| Method | Easy | Hard | Avg |
|---|---:|---:|---:|
| π0 | 65.9 | 58.4 | 62.1 |
| π0.5 | 82.7 | 72.9 | ≈77.8 |
| X-VLA | 84.7 | 76.8 | 80.9 |
| **Wall-OSS + X-Tokenizer** | **84.7** | **80.88** | **82.79** |

(Appendix Table 6 per-task 50-task 평균: Easy 84.70%, Hard 80.88%)

- 가장 강한 published baseline π0.5를 Easy/Hard 모두에서 능가.
- **Hard에서 gain이 더 큼** → aligned action-token interface가 visual condition shift 하에서 가장 유용. domain randomization에 의해 backbone hidden state가 흔들릴 때 semantic anchor 역할.

### 6.1 Cross-Embodiment (Section 4.3, Fig. 9)

5개 single-arm embodiment (Agilex/Arx5/Franka/Piper/UR5).

| Setting | Easy | Hard | Avg |
|---|---:|---:|---:|
| Single-embodiment (5개 별도 모델) | 70.9 | 64.0 | 67.5 |
| **Joint (5-embodiment 통합 1개)** | **77.9** | **74.4** | **76.2** |

- 공유 action token 공간이 motion structure를 cross-embodiment로 재사용. Hard에서 격차가 더 크게 벌어짐.

---

## 7. Real-World Evaluation (Section 4.4)

**Setting**: 7 tabletop task, Wall-OSS Qwen2.5-VL-3B backbone, 4개 action interface 비교 (원본 Wall-OSS flow head / FAST / RVQ-only no-aux / X-Tokenizer). 동일 training schedule, 동일 480k multimodal grounding sample 25% mix.

| Action interface | VQA (%) | Short-h (5 tasks) | Long-h (2 tasks) | All-7 Avg |
|---|---:|---:|---:|---:|
| FAST | 75.7 | — | 61.0 | 73.0 |
| RVQ no-aux | 79.4 | — | — | 69.1 |
| **X-Tokenizer** | **85.9** | **80.6** | **69.3** | **77.4** |

- vs FAST: **multimodal grounding +13.5% relative (75.7 → 85.9)**, **long-horizon +8.25 absolute (61.0 → 69.25)**.
- **Cross-backbone transfer 입증**: X-Tokenizer는 frozen Qwen2.5-VL-7B feature에 align되었지만 downstream에서는 Qwen2.5-VL-3B에 consume됨 → semantic alignment가 backbone size에 의존하지 않음.
- **RVQ no-aux 교훈**: multi-level discrete structure만으로는 VQA를 살릴 수 있으나 (75.7→79.4) action 품질은 오히려 하락 (73.0→69.1). MAM + Align + Pred가 *모두* 있어야 양쪽이 같이 올라감.

---

## 8. 핵심 통찰

1. **"Tokenizer = compression"이 아니라 "tokenizer = supervision signal"**: discrete token loss는 backbone hidden state를 형성하는 작용을 하므로 token의 의미 구조가 곧 VLM의 보존된 grounding 능력으로 이어짐.
2. **비대칭 supervision의 정당성**: 모든 RVQ level에 동일 reconstruction loss를 주면 perplexity가 uniform해지고 level별 interpretable role이 사라짐 (no-aux 결과). SRQ의 비대칭이 *명시적 inductive bias*.
3. **Pretrain-only auxiliary head**: 세 head를 deployment에서 빼는 설계 — pretraining 시 풍부한 신호 vs deployment 시 가벼운 core라는 두 마리 토끼를 동시 확보. (FAST/RVQ 등 기존 tokenizer와 inference cost 동등.)
4. **VLM-driven reconstruction probe**: VL feature → codebook → reconstruction 경로의 cosine similarity 0.85-0.95 (Section 4.1) — 학습된 alignment가 단순 embedding plot 인접이 아니라 *기능적으로 사용 가능*함을 보임. 표준 action-only tokenizer (FAST, RDT-VQ)는 이 경로 자체가 존재하지 않음.

---

## 9. 한계

- **Reconstruction L1이 의도적으로 나쁨** (FAST 대비 +17%): semantic structure를 위한 trade-off지만 high-precision 산업 task (예: mm-scale insertion)에서 부담이 될 수 있음. Section 4.1도 fine pre-contact task (insert/plug, press/button)에서 VL-driven reconstruction의 L1 error가 가장 큼을 인정.
- **Wall-OSS에 종속된 평가**: tokenizer ablation은 모두 Wall-OSS backbone + Flow Matching expert 위에서만 측정됨. 다른 hybrid VLA (예: π0.5의 discrete pre-train + flow post-train) 또는 pure autoregressive policy (RT-2, OpenVLA)에 붙였을 때의 효과는 미검증.
- **공개 자산 부재**: paper 시점 기준 weight·code 미공개 (project page만). 재현/검증 어려움.
- **17 arm family에 한정**: humanoid·mobile manipulator·legged 등 더 광범위한 morphology에는 적용 안 됨. embodiment token m의 generalization은 same-class 내에서만 입증.
- **Q=4, T=64, M=16의 hyperparameter는 ablation 부족**: SRQ의 비대칭 supervision은 depth 선택과 강하게 결합되어 있을 가능성.

---

## 10. 총평

본 논문의 가장 중요한 기여는 **"action tokenizer는 무엇을 위한 도구인가"라는 질문의 재정의**다. FAST·VQ-BeT 같은 기존 작업이 *얼마나 잘 reconstruction*하는지를 묻는다면, X-Tokenizer는 *얼마나 잘 backbone을 supervise*하는지를 묻는다. RoboTwin Hard에서의 gain (Fig. 8)과 multimodal grounding VQA의 +13.5% 향상 (Section 4.4)이 이 관점 전환의 정량적 근거다. 특히 RVQ-only no-aux 변종이 reconstruction은 가장 좋지만 downstream action 품질은 떨어진다는 결과 (Table 1 + Section 4.4)는 "compression ≠ representation"이라는 주장에 강력한 증거를 제공한다.

또한 비대칭 supervision (top-level만 MAM·deeper level은 raw residual)이라는 **단순하지만 옳은 inductive bias**가 인상적이다. 이는 음성 합성의 SoundStream/EnCodec이 coarse-to-fine RVQ를 사용하는 것과도 정신적으로 닮아 있지만, 여기서는 *cross-modal alignment*를 위해 그 구조를 적극 활용했다는 점이 차별점.

Pretrain-only auxiliary head 설계도 실용적이다. Inference 시 X-Tokenizer가 사라지므로 기존 hybrid VLA pipeline에 *부담 없이* 끼울 수 있는 plug-in 형태 — 후속 작업이 채택하기 쉬운 형태로 packaging되어 있다.

다만 X-Tokenizer 자체는 **policy가 아닌 supervision 도구**라는 점이 항상 따라다닌다 — 이 paper는 Wall-OSS와 결합된 "Wall-OSS + X-Tokenizer"라는 named policy로 평가되며, X-Tokenizer만 떼어서 비교하는 건 의미가 없음. VLA-Tracker에는 "Wall-OSS + X-Tokenizer" deliverable로 등록하는 것이 자연스럽다.

---

## 11. 후속 연구 방향

- **다른 backbone에 plug-in**: π0, X-VLA, RDT 등 다양한 hybrid VLA에 X-Tokenizer를 frozen으로 붙였을 때의 효과 측정. cross-backbone transfer가 paper 안에서는 부분적으로만 입증됨 (Qwen2.5-VL 7B→3B).
- **Discrete-only policy로의 응용**: 현재는 hybrid (discrete supervision + Flow Matching regression) 안에서만 평가. RT-2 / OpenVLA처럼 *순수 discrete autoregressive*가 X-Tokenizer code stream으로 직접 action을 생성하면 어떤지.
- **다른 modality로의 SRQ**: 비대칭 RVQ supervision 자체는 일반화 가능한 아이디어. video tokenization, speech tokenization에 적용 시 어떤가.
- **MAM masking ratio·schedule**: BERT의 15% mask, span masking 등 NLP에서 발전한 기법을 action에 어떻게 옮길지.
- **Long-horizon에서의 token reuse 분석**: top-level "motion word"가 task 간에 얼마나 reuse되는지 정량 분석 → 진정한 "discrete action language" 형성 여부.

---

## 12. 예상 질문

- **Q1**: SRQ는 RVQ에 단지 MAM 하나를 더한 것 아닌가?
  - A: 그것만이 아님. (a) Align + Pred head로 latent를 VL space에 anchor, (b) auxiliary head를 deployment에서 제거하는 비대칭 train/deploy 구조, (c) 비대칭 supervision을 통해 perplexity의 monotone spectrum을 강제 — 세 요소가 함께 작동. Table 1의 "w/o MAM"·"w/o Align+Pred" 모두 full보다 나쁨.

- **Q2**: Wall-OSS + X-Tokenizer의 84.7% Easy가 X-VLA의 84.7%와 동률인데 정말 SOTA인가?
  - A: Easy는 동률이지만 Hard에서 80.88% vs X-VLA 76.8% → 약 +4pt. Hard가 domain randomization을 포함한 진정한 generalization 지표이므로 의미 있는 격차. Avg도 82.79 vs 80.9.

- **Q3**: X-Tokenizer 자체가 policy가 아닌데 VLA-Tracker에 추가하는 게 맞나?
  - A: 본 논문은 **"Wall-OSS + X-Tokenizer"라는 구체적 named policy**를 deliverable로 제공하고 그 RoboTwin 2.0 점수를 보고함. tokenizer만이라면 framework로 제외해야 하지만, paper의 main result table이 policy 형태이므로 ACCEPTED.

- **Q4**: Inference 시 X-Tokenizer가 사라진다면 어떻게 동작하나?
  - A: Pretrain 시점에는 frozen X-Tokenizer가 expert trajectory를 *offline에서* multi-level discrete token으로 encoding → 이 token을 autoregressive prediction target으로 사용해 VLM hidden state를 형성. Inference 시점에는 VLM이 만든 hidden state를 *continuous Flow Matching expert*가 받아 action chunk를 직접 regress. 즉 X-Tokenizer는 "training-time supervision 공급자"이고 inference path 외부.

- **Q5**: FAST 대비 reconstruction L1이 17% 더 나쁜데 어떻게 downstream이 더 좋은가?
  - A: Reconstruction 품질 ≠ supervision 품질. 임의의 reconstruction index로 학습된 backbone hidden state는 *geometric pattern*에 끌려가 multimodal grounding을 잃음 (Section 4.4의 FAST VQA 75.7%). X-Tokenizer는 reconstruction을 약간 희생하고 *backbone이 보기에 의미 있는* token을 제공 → VQA 85.9%, 결과적으로 action 품질도 향상. 이 trade-off의 정량적 입증이 본 논문의 핵심.

- **Q6**: MAM이 BERT처럼 작동하는데 robot action에서 정말 "language" 같은 syntactic regularity가 있는가?
  - A: Fig. 5의 q0 active 76.4% + Zipf-like long-tail 분포가 그 증거. 일부 "motion word"가 자주 반복되고 (장 코너 도달, 잡기, 들어올리기 등) 그 외 rare pattern이 long tail. 이 분포는 자연어의 token frequency와 정성적으로 유사 → masked prediction이 학습 가능한 구조.

- **Q7**: 2.4M trajectory / 17 arm family는 어떻게 모았나?
  - A: Appendix Table 4·5. Open X-Embodiment (RT-1, BridgeV2 등 다수), DROID, RoboTwin 2.0 sim, X Square 내부 데이터셋 등 public + proprietary 혼합. 17 arm family는 Franka / UR5 / Arx5 / Piper / Aloha bimanual / Agilex bimanual / ... 등 (Appendix Table 5).

<!-- VERIFIED: pdf -->
