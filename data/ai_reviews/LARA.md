# LARA: Latent Action Representation Alignment for Vision-Language-Action Models

> **한 줄 요약**: Latent Action Model(LAM)과 diffusion VLA를 **각각 독립적으로 학습/freeze**하던 기존 패러다임을 깨고, DiT의 L-2 레이어 hidden state와 LAM의 **online continuous latent action** z_t를 cosine similarity로 정렬하는 단순한 보조 손실(w1=0.01)을 도입하여 LAM·VLA를 동시 최적화. OXE 사전학습만으로 LIBERO 88.6%, SIMPLER 65.2% — vanilla DiT 대비 LIBERO +5%p / SIMPLER +15%p, LAM refinement 시 SIMPLER 5-task 평균 +15.7%p, GR00T-N1.6에 post-training으로 plug-in 시 G1 real +5.56%p.

---

## 1. 배경 및 동기

- **VLA 데이터 병목**: 행동 라벨이 달린 로봇 데이터는 비용이 크고 embodiment 가로 일반화가 어렵다. 인터넷 비디오는 풍부하나 action label이 없다.
- **기존 LAM-VLA 패러다임의 한계** (Fig. 2 left):
  1. LAM은 **순수 visual reconstruction**(VQ-VAE)으로 학습 → 조명/배경 등 task-irrelevant visual change까지 latent에 인코딩 ("spurious dynamics").
  2. VLA는 **frozen LAM**이 생성한 pseudo-label에 종속 → latent 공간이 실제 robot action에 맞춰 진화 불가능.
- **저자들의 핵심 가설**: REPA(image generation에서 DINOv2 frozen 정렬)와 달리, **action 영역에서는 alignment target 자체가 학습 가능해야** LAM↔VLA가 상호 보완할 수 있다. 즉 정렬 신호도 "live"여야 한다.

---

## 2. 방법론 심층 분석

### 2.1 두 구성 요소

**(A) Latent Action Model (LAM)** — Moto-GPT(Chen et al., 2025b) 설계 차용
- Frozen ViT(MAE pretrained) encoder가 (I_t, I_{t+C}) patch embedding 추출.
- M-Former (4-layer transformer, 8 learnable query) → continuous latent z_t.
- VQ codebook (size **128**)로 z_t^q 양자화.
- 12-layer ViT decoder (hidden 768)로 I_{t+C} 재구성. 손실: L_LAM = MSE + commitment(VQ-VAE, β).

**(B) Flow-matching VLA** — GR00T-N1 계열의 cross-attn DiT
- **Frozen Eagle-2 VLM** + learnable self-attention adapter (visual+text token 처리).
- Embodiment-specific MLP encoder가 proprioceptive state를 공유 latent로 사영 (max action dim 32, max state dim 64, max 64 embodiment IDs).
- **DiT L=16 layers**, alternating self-attn / cross-attn (VLM token에 cross). Flow matching 목표:

  L_ACT(θ) = E[||v_θ(A_t^τ, c_t) − (A_t − ε)||²]

### 2.2 LARA Loss (Eq. 6, 7)

DiT를 encoder-decoder로 보고 hidden h_t^θ = E_θ(A_t^τ, c_t)를 추출. **L−2 layer**의 hidden state 중 **t+C 시점(action chunk 끝)** 토큰을 선택:

L_LARA = − E[ CosSim( z_t, f_ψ(h_t^θ) ) ]

여기서 **z_t는 quantize 이전의 online continuous LAM latent**(z_t^φ). f_ψ는 학습 가능한 MLP projection head. 전체 손실:

L = L_ACT(θ) + w1·L_LARA(θ, φ, ψ) + w2·L_LAM(φ),  w1 = w2 = **0.01**.

φ(LAM)와 θ(DiT) **둘 다 gradient를 받아 공동 업데이트**되는 것이 핵심.

### 2.3 양방향 정규화 (저자들의 이론적 주장)

1. **Inverse Dynamics Regularization for LAM**: action 표현과 정렬되도록 강제 → LAM이 조명/그림자 같은 spurious factor를 버리고 control-relevant visual change만 인코딩 (Fig. 5 attention map으로 정성 검증 — LARA-LAM은 end-effector·target object에 집중, baseline LAM은 배경에 분산).
2. **Forward Dynamics Grounding for Action Diffusion**: VLA hidden state가 LAM의 forward-predictive latent에 묶이므로, 단순 패턴 매칭이 아닌 **future state evolution**을 암시적으로 학습 → "kinematically plausible but functionally null" trajectory 환각 감소.

### 2.4 3-stage 학습 파이프라인 (Sec. 4.2)

| 단계 | 데이터 | 손실 | 업데이트 |
|------|--------|------|----------|
| 1. LAM Pre-training | OXE + unlabeled video, C=5 | L_LAM only | φ |
| 2. LARA Joint Pre-training | OXE labeled, C=16 | L_ACT + w1·L_LARA + w2·L_LAM | θ, φ, ψ |
| 3. LARA Joint Post-training | target embodiment demos | 동일 | θ, φ, ψ |

### 2.5 두 가지 응용 모드

- **(i) Post-training Enhancement**: 임의의 pre-trained diffusion VLA(GR00T-N1.6, π0.5)에 LARA를 부착해 post-training만 수행. Architecture 변경 없음.
- **(ii) LAM Refinement**: LARA로 정련된 LAM을 떼어 LAPA·Moto-GPT 같은 latent-token 기반 VLA의 pseudo-label provider로 재사용.

---

## 3. 데이터 전략

- **사전학습**: Open-X-Embodiment(OXE) subset, single-arm end-effector trajectory만 필터. Bridge 8.27% 등 균형 샘플링.
- **시간 stride**: LAM은 C=5 (fine-grained visual motion), VLA는 C=16 (action chunk와 일치).
- **Post-training**:
  - LIBERO (Liu et al., 2023): 4 suites, 표준 fine-tuning protocol.
  - SIMPLER-ENV (Li et al., 2024): Pick Coke Can / Object Movement / Open & Close Drawer.
  - GR1-Sim-24(30): 30 demo/task fine-tuning, 24 bimanual task.
  - G1-Real(50): Unitree G1 humanoid, "Pick Green Tomato → Green Basket", "Grasp Bottle and Pour to Cup", 50 demo per task, 50 trial 평가.

---

## 4. 시스템/학습 세부사항

- **하드웨어**: 4× NVIDIA A100 GPU.
- **LAM**: 350k step, global batch 512, AdamW, lr=1e-4 cosine decay, wd=1e-5.
- **LARA full / DiT-only**: 200k step pre-train, batch 384, 동일 optimizer.
- **GR00T-N1.6-LARA post-train**: 20k step(50k for GR1-Sim-24), batch 384, lr=1e-4.
- **Alignment depth**: DiT **L−2 layer**가 GR00T 계열에서 최적(Fig. 6: L−2 = 92.5% > L = 87% > 8 = 89% > 4 = 86.5%). π0.5에서는 **L(최종)이 최적** — architecture-dependent.
- **Token 선택**: action chunk의 **마지막 timestep(t+C)** hidden token을 정렬 대상으로 사용 → "완성된 action trajectory의 표현이 LAM의 visual effect와 일치"하도록 강제.

---

## 5. 실험 설계 및 평가 프로토콜

저자들은 **두 설정**으로 공정성을 분리:

| 설정 | 의미 |
|------|------|
| **OXE-Constrained** | 모든 모델이 OXE 범위 안에서만 pre-train → 순수 method effect 격리 |
| **Unconstrained** | 임의의 데이터·모델 크기 허용 → 절대 성능 한계 비교 |

**연구 질문 3개** (Sec. 5):
1. 전체 VLA 학습으로서 LARA의 우월성?
2. Post-training enhancement / LAM refinement 모듈로서의 효과?
3. 새 embodiment·task에 대한 일반화?

---

## 6. 실험 결과 심층 분석

### 6.1 Full VLA Training (Table 1, OXE-Constrained)

| Method | LIBERO Avg | SIMPLER Avg |
|--------|-----------|-------------|
| OpenVLA | 76.5 | 32.7 |
| Octo | 75.1 | 14.6 |
| Moto-GPT | — | 61.4 |
| LAPA | 65.7 | — |
| LARA (DiT-only) | 84.4 | 55.8 |
| **LARA (full)** | **88.6** (+5.0%p) | **65.2** (+16.8%p) |

- DiT-only는 baseline diffusion만. **LARA full vs DiT-only 격차가 method의 순효과**: LIBERO Long에서 76.5 → 86.0 (+12.4%p)로 가장 큼 — long-horizon task에서 forward dynamics grounding이 결정적.
- SIMPLER Pick Coke Can: DiT-only 62.3 → full 82.3 (**+32.1%p**), 단 Move Near는 -0.4%p로 거의 saturation. Drawer는 21.0 → 29.5 (+40.5%p).
- OXE만 썼는데도 **Unconstrained 설정의 π0-FAST(85.5), Magma, SpatialVLA(78.1)를 상회** → data-efficient VLA framework로서 유의미.

### 6.2 Post-training Enhancement (Table 1·2 하단)

| | LIBERO Avg | SIMPLER Avg | GR1-Sim-24 | G1-Real |
|---|-----------|-------------|-----------|---------|
| GR00T-N1.6 (baseline) | 95.0 | 78.9 | 47.0 | 72.0 |
| **GR00T-N1.6-LARA** | **95.6** | **79.9** | **48.5** | **76.0** |
| Δ | +0.6%p | +1.3%p | +3.2%p | **+5.56%p** |

- 이미 SOTA인 GR00T-N1.6 위에 **post-training만** LARA로 했음에도 G1 real에서 +5.56%p 개선. LAM은 OXE pre-train된 것 그대로 사용.
- G1-Real Pick-n-Place: 76.0 → 84.0 (+10.5%p), Grasp-n-Pour Pour: 87.2 → 94.4 (+7.2%p). DreamVLA/UniVLA처럼 full re-training이 필요 없으면서 더 큰 효율성.

### 6.3 Fast Adaptation (Table 2 상단, OXE-Constrained)

OXE에 없는 GR-1 / G1 embodiment에 적응시켰을 때 LARA(full) vs LARA(DiT-only):
- GR1-Sim-24: 6.4 → 11.4 (+78.1%)
- G1-Real Pick-n-Place: 58.0 → 80.0 (+37.9%p), Grasp-n-Pour: 54.0 → 68.0 (+25.9%p).

저자 해석: LARA가 학습한 latent action 공간이 **embodiment-agnostic** semantic motion에 가깝기 때문에 적은 demo로 빠르게 전이.

### 6.4 LAM Refinement (Table 3)

Moto-GPT의 LAM을 LARA-LAM으로 교체 → SIMPLER 5-task:

| Task | LAM | LARA-LAM | Δ |
|------|-----|----------|---|
| Pick Object | 36.3 | 41.0 | +12.9% |
| Move Near | 61.0 | 63.7 | +4.4% |
| Open Drawer | 25.7 | 29.3 | +14.0% |
| **Close Drawer** | **38.0** | **53.7** | **+41.3%** |
| Pick Coke Can | 53.0 | 59.7 | +12.6% |
| **Avg** | **42.8** | **49.5** | **+15.7%** |

- LARA로 정렬되어 더 깨끗해진 latent를 pseudo-label로 쓰는 것만으로 downstream VLA가 평균 +15.7%p 향상 → bi-directional 효과의 강력한 증거.

---

## 7. Ablation 분석

### 7.1 Alignment Depth (Fig. 6, LIBERO-Long)

| Layer | 4 | 8 | **L-2** | L |
|-------|---|---|---------|---|
| SR | 86.5 | 89 | **92.5** | 87 |

- 너무 얕으면 semantic abstraction 부족, 너무 깊으면 action-specific하여 representation 정렬 여지 없음. L−2가 sweet spot. 단 π0.5에서는 L이 최적 → **architecture-dependent**.

### 7.2 Joint Optimization vs Frozen LAM (Fig. 6)

- LARA full(joint, L-2): **92.5%**
- Frozen LAM(alignment 있음, L-2): **85.5%**
- → 정렬 자체보다 **LAM이 함께 학습되는 것**이 결정적. 이것이 REPA(frozen DINOv2 정렬)와의 본질적 차이.

### 7.3 Loss Weight w1, w2

- Appendix Sec. B.2에서 ablation 보고. 본문에서는 w1=w2=0.01로 정착. "loss design also plays a crucial role" — 너무 크면 ACT 손실을 압도, 너무 작으면 효과 미미.

### 7.4 정성 분석 (Fig. 5)

- Baseline LAM의 attention: 배경·테이블·조명 영역에 분산.
- LARA-LAM의 attention: end-effector + 조작 대상 객체에 집중적으로 sharp. → inverse dynamics regularizer 효과의 직접 시각화.

---

## 8. 관련 연구 비교

- **LAM 계열** (LAPO, Genie, LAPA, Moto-GPT, villa-X, UniVLA): LAM을 pseudo-label 공급원 또는 frozen feature로 사용. **decoupled training**.
- **REPA(image generation)** [Yu et al., 2024]: DiT를 frozen DINOv2로 정렬. **target frozen**이라 generative 영역에서 잘 작동.
- **FLARE, TraceVLA, Kachaev 2025**: VLA를 frozen visual-language feature로 정렬 — alignment target이 정적이라는 점이 LARA와 다름.
- **DreamVLA, UniVLA**: explicit world model로 효과를 얻으나 full re-training 필요. LARA는 plug-in post-training으로 비슷한 효과를 더 싸게 달성.
- **저자들이 강조하는 gap**: "alignment target should be an updatable action representation" — z_t를 online으로 쓰는 발상이 핵심 차별점.

---

## 9. 한계 및 미해결 문제

1. **OXE-subset만 사용**: 저자도 인정. Full open-source dataset으로 scale-up 시 효과 폭은 미검증 (Sec. 5.4 끝).
2. **Architecture-dependent alignment depth**: GR00T는 L-2, π0.5는 L → 새 backbone마다 sweep 필요. Universal rule 부재.
3. **G1-Real Grasp-Right에서 -4.0%p**: post-training enhancement 일부 sub-task에서 regression. 모든 task에서 monotonic improvement는 아님.
4. **z_t를 quantize 이전 continuous로 사용**: discrete token 기반 downstream(LAPA·Moto-GPT)에서는 그대로 못 쓰고 LARA-LAM을 다시 떼어 써야 함 — 추론 파이프라인 분리 필요.
5. **Compute**: 4×A100·350k step LAM + 200k step DiT + 20-50k post-train. "data-efficient"이긴 하나 academic group이 from-scratch로 재현하기엔 부담.
6. **Cycle-time/latency 미보고**: DiT 16-layer + LAM forward를 학습 시 동시 돌리는 비용 보고 부재. Inference 시 LAM은 분리 가능하다고 가정되나 명시적 latency 표기 없음.
7. **Ablation w/o LAM reconstruction (w2=0)**: forward dynamics grounding이 정말 핵심인지 확인하는 ablation이 본문에 부재 (Appendix에 일부).
8. **모든 task가 short-horizon manipulation**: Long-horizon mobile manipulation, navigation 등에서의 검증 부재.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — REPA-style alignment를 action 영역으로 가져오되 **"target도 학습 가능"**으로 비튼 단 한 줄의 통찰이 강력. 단, 기술적 구성요소(LAM, DiT, flow matching, alignment) 자체는 모두 기존 |
| **Technical depth** | ★★★★☆ — 3-stage 학습, L-2 token-at-t+C 선택, w1=0.01 등 디테일이 잘 정돈. Ablation도 alignment depth/joint vs frozen/loss weight로 체계적 |
| **Experimental rigor** | ★★★★★ — OXE-Constrained / Unconstrained 분리, 3개 sim + 1개 real(50 trial), full / post-train / LAM-refine 3가지 사용 모드 모두 검증 |
| **Practical impact** | ★★★★★ — w1=0.01 단 한 줄 추가로 SOTA VLA에 +5%p, LAM 교체만으로 downstream +15%p. Plug-and-play 가치 매우 큼 |
| **Writing quality** | ★★★★☆ — Figure 2의 좌우 비교가 핵심 메시지 즉시 전달. Sec. 4의 reciprocal regularization 논증은 깔끔하나 일부는 정성적 |

**총평**: LARA는 "왜 LAM과 VLA를 따로 학습해 왔는가?"라는 단순한 질문에서 출발해, **alignment target도 학습 가능해야 한다**는 한 가지 통찰만으로 LAM 기반 VLA 패러다임 전체를 단순화시켰다. REPA가 "frozen target이면 충분하다"고 보여준 image diffusion 영역과 달리, robot action처럼 **embodiment 종속적이고 task-specific한 신호**에서는 정렬 신호 자체가 evolve해야 한다는 주장을 ablation(joint vs frozen, +7%p)으로 직접 입증한 점이 가장 큰 기여. 약점은 alignment depth가 architecture별로 다르다는 점과 LAM 자체 디자인은 Moto-GPT 그대로라는 점 — 그러나 "loss 한 줄로 SOTA를 +5%p 끌어올린다"는 단순함이 본 논문의 가장 큰 무기.

---

## 11. 적용 가능성 및 향후 방향

- **즉시 응용**: 현존 diffusion-based VLA(GR00T-N1.6, π0.5, RDT-1B, FLOWER 등)에 plug-in post-training. 0.01 weight, L-2 alignment의 단순한 leveraging.
- **확장 방향**:
  1. **Multi-step latent prediction**: 현재는 t+C 한 시점만 정렬. 여러 시점(t+C/2, t+C 등)으로 dense alignment 시 추가 효과?
  2. **Hierarchical LAM**: short-horizon visual dynamics + long-horizon subgoal latent를 분리하여 long-horizon task 강화.
  3. **Cross-embodiment LAM transfer**: G1 humanoid에 OXE-pretrained LAM이 통한 점을 활용, dexterous hand 등 high-DoF embodiment로 zero-shot.
  4. **VLM도 unfreeze**: 현재 Eagle-2 frozen. LARA loss로 VLM까지 정렬하면 instruction-following 개선 가능성.
  5. **Discrete LAM token 기반 generation 모델과의 직접 비교**: LAPA, UniVLA, villa-X와 같은 setting에서 LARA-LAM을 in-place 교체했을 때 어디까지 SOTA를 갱신하는가.
  6. **Cycle-time 측정 및 deploy**: 학습 시 LAM 함께 forward — inference 시 LAM 제거의 정확한 효과(추정과 실제 GPU memory) 정량화.

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | REPA와의 본질적 차이는 무엇인가? Frozen DINOv2로 대체하면 안 되나? | Fig. 6의 frozen LAM ablation이 직접 답: L-2 frozen 정렬은 85.5%, joint는 92.5%. action signal은 generative와 달리 **target 자체가 robot data로 진화**해야 의미가 있음. DINOv2 같은 일반 visual feature는 control-relevant signal에 specialize되지 않음. |
| 2 | w1=0.01은 너무 작지 않은가? 정렬 효과가 실제로 어디서 오는가? | 0.01도 작아 보이지만 cosine similarity가 [-1,1]이고 ACT 손실은 ||·||² scale이므로 gradient 비율로 보면 적절. Appendix Sec. B.2 ablation에서 0.01 부근이 plateau. 핵심은 w1보다 **joint optimization** 자체 (Fig. 6). |
| 3 | LARA 효과가 단순히 multi-task supervision의 이득 아닌가? | 두 가지 반박: (i) DiT-only는 동일 데이터·step으로 학습됐는데 LIBERO 84.4 vs LARA full 88.6 (+4.2%p) — 데이터·compute 동일. (ii) Fig. 5 attention map 정성 분석으로 LAM이 실제로 action-centric으로 진화함을 시각화. |
| 4 | Alignment depth가 architecture-dependent라면 새 backbone마다 sweep 필요. 실용성 손상 아닌가? | 인정되는 한계. 다만 저자들의 경험적 규칙 "deeper layers close to action head"는 일반 가이드라인 제공. π0.5는 L, GR00T-N1.6은 L-2. Layer 4·8은 모든 architecture에서 suboptimal. 새 backbone에서는 2-3 layer만 sweep하면 충분. |
| 5 | LAM이 unlabeled human video로 학습된다면 embodiment gap은 어떻게 해소되는가? | LAM은 visual dynamics만 학습 → embodiment-agnostic latent. Sec. 5.4가 직접 검증: GR-1/G1은 LAM pre-train에 없었으나 LARA full이 +30%p 이상 향상 → "control-relevant visual change"가 추상적으로 전이됨. |
| 6 | Moto-GPT의 LAM 디자인 그대로 차용했는데, LAM 자체 개선 없이 alignment만으로 LAPA·UniVLA를 능가하나? | 동일 LAM 구조에서 정렬만 추가한 LARA-LAM이 Moto-GPT(원본 LAM)의 SIMPLER 평균 42.8 → 49.5 (+15.7%p). 단 UniVLA 같은 unconstrained SOTA(LIBERO 95.2)는 더 큰 데이터로 학습된 결과이며 LARA(OXE only) full이 88.6이므로 직접 비기지 못함. 단, GR00T-N1.6-LARA가 95.6로 UniVLA를 약간 상회. |
| 7 | G1-Real Grasp-Right에서 -4.0%p regression이 발생한다. 신뢰할 수 있는 방법인가? | 50 trial 단위에서 -4%p는 ±2 sample. 다른 sub-task에서 +5~+10%p이고 평균 +5.56%p이므로 statistical noise 가능성 큼. Single failure mode가 본질적 한계인지는 더 큰 trial이 필요. |
| 8 | Inference cost는? LAM forward도 매번 돌려야 하지 않나? | 학습 시에만 LAM forward가 필요(L_LARA 계산용). Inference 시에는 DiT만 동작 → overhead 0. 단 논문은 LAM 분리 후의 정확한 ms 측정을 보고하지 않음 — 개선 여지. |
| 9 | π0.5에서는 L(최종)이 최적이라는 주장은 어떻게 일관성을 가지는가? | 저자들은 "deeper layers near action head"를 핵심 원리로 제시. π0.5는 backbone 구조상 action head가 최종에 직접 붙어 L이 그 직전. GR00T는 L-2가 high-level/action 사이의 인터페이스. 즉 architecture별 "second-to-action-decoder" 위치가 본질. |
| 10 | LAM의 codebook size=128은 너무 작지 않은가? Discretization으로 정보 손실은? | LARA에서 alignment에 쓰는 z_t는 **양자화 이전 continuous** (Eq. 6). 즉 codebook 손실의 영향을 우회. VQ는 LAM의 자체 reconstruction objective에만 필요. 이 설계가 LARA의 핵심 implementation 선택 중 하나. |
| 11 | Long-horizon이나 mobile manipulation에서도 통하는가? | 본 논문 범위 밖. LIBERO-Long에서 86.0%(DiT-only 76.5%, +9.5%p)로 long-horizon 단서는 있으나 진정한 multi-step navigation·tool-use는 미검증. Future work. |
| 12 | LARA를 RL fine-tuning과 결합하면? | 본 논문은 supervised behavior cloning만. SimpleVLA-RL 같은 PPO/GRPO scheme에 LARA loss를 보조 regularizer로 추가하면 hallucinated trajectory 감소 효과로 RL 안정성 강화 가능. 실험 필요. |

<!-- VERIFIED: pdf -->
