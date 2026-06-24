# EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models

> **한 줄 요약**: 동결된 VLM(GR00T N1.5 3B) + flow-matching DiT 구조의 VLA에 token-level Frame Averaging 기반 EquiPerceptor와 steerable-layer 기반 정확한 SO(2)-equivariant EquiActor를 결합하여, pretrained 가중치를 수정하지 않고 카메라→액션 전 파이프라인에 근사 SO(2) 동변성을 부여한 최초의 large-scale 동변 VLA. LIBERO avg 92.6%, CALVIN ABCD→D avg-len 4.03.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 현대 VLA(π₀, GR00T N1.5, OpenVLA 등)는 회전 대칭(SO(2))이라는 manipulation의 본질적 기하 구조를 architecture 차원에서 활용하지 않음 — 같은 task를 회전된 객체 orientation마다 **독립적으로** 학습해야 함
- 기존 equivariant policy(EquAct, Diffusion Policy 변형 등)는 **scratch 학습** 또는 **point cloud** 기반이라 사전학습된 대형 VLM의 표현력을 활용 못함
- Eq.Bot 같은 **canonicalization wrapper**는 frame 추정 품질에 의존하고 action head에 기하구조를 전파 못함

### 핵심 질문
- **동결된 ViT 기반 VLM + flow-matching DiT 위에 어떻게 architecturally SO(2) equivariance를 보장할 수 있는가?**
- **공간적으로 indexed된 ViT 패치 토큰 시퀀스에 Frame Averaging을 어떻게 일반화하는가?** (기존 FA는 globally pooled vector에만 적용)

📌 [Figure 1 삽입] — EquiPerceptor(top/wrist image → Frame Average → equivariant/invariant projector → frozen VLM → equivariant adapter) + EquiActor(state·noisy action → equivariant cross/self attention via steerable layers)

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

EquiVLA = GR00T N1.5(frozen VLM 3B + flow-matching DiT action head) + 두 개의 composable 모듈:
- **EquiPerceptor**: top-down 이미지에 **token-level Frame Averaging**을 적용해 ViT 패치 토큰 시퀀스로부터 *근사* SO(2)-equivariant 시각 표현 생성. Frame Averaging이 spatial permutation τ(h⁻¹)과 regular rep ρ_reg(h)의 곱으로 작용하여 패치 단위 위치-내용 일관성 유지
- **EquiActor**: steerable layer로 attention/state encoding/action decoding을 구성한 **정확한** SO(2)-equivariant flow-matching DiT — regular feature space에서 동작

### 2.2 Token-level Frame Averaging (핵심 기여)

기존 FA[14]는 globally pooled vector에만 적용 — ViT 패치 토큰처럼 공간적으로 indexed된 시퀀스에 직접 적용하면 회전된 입력에서 각 패치가 다른 grid 위치로 이동하므로 naive averaging은 위치 정보를 파괴.

**해결**: A(h) = τ(h) ⊗ ρ_reg(h) — spatial permutation τ(h)가 패치 위치를 재매핑하고 regular representation ρ_reg(h)가 token feature를 회전. 두 작용을 동시에 수행하여 위치-feature 정렬을 유지.

> ❓ **예상 질문**: 왜 *approximate* equivariance인가?
> **답변**: 유한한 n×n grid에서 회전된 패치 중심이 grid 셀에 정확히 안 떨어지므로 nearest-neighbor 재매핑이 필요 → spatial homomorphism defect Δ 발생. Table 6에서 C₈ × ViT grid에 대한 Δ 폐형 계산을 제공. C₄ × C₄ 쌍은 exact (Δ=0).

### 2.3 EquiActor: Steerable Flow-matching DiT

- **State/action encoding**: regular representation에서 동작, 입력 회전이 feature 회전으로 정확히 대응
- **Cross/self attention**: steerable layer로 구성되어 동변성 정확히 보존
- **Flow-matching objective**: conditional flow matching loss는 **G-invariant**이고 ODE 적분이 동변성을 보존하므로 학습/추론 모두에서 정확한 SO(2) 보장

> ❓ **예상 질문**: EquiActor만 적용 vs 전체 EquiVLA의 기여 분리는?
> **답변**: Table 1에서 GR00T N1.5(78.1%) → +EquiActor(91.0%) → EquiVLA(92.6%, 상대 제어). EquiActor가 대부분의 향상(+12.9pp)을 차지하고 EquiPerceptor가 추가 +1.6pp 기여 → action-head equivariance가 dominant factor.

---

## 3. 데이터 전략

### 학습 데이터
- **LIBERO**: 4개 suite (10/Goal/Object/Spatial), 각 10 tasks, suite당 500 rollouts(50/task) × 2 seeds 평가
- **CALVIN ABCD→D**: 환경 A,B,C,D 학습, hold-out D에서 1000 instruction chain 평가, single-step observation
- **Mobile ALOHA**: 5개 tabletop task, ±10 cm / ±10° 위치/방향 변동
- 모든 모델 동일 hyperparameter, 동일 GR00T N1.5 3B checkpoint에서 출발

### 데이터 효율 실험
LIBERO 학습 demo의 10%/40%/100%로 sample efficiency 비교:
| 방법 | 10% | 40% | 100% |
|------|-----|-----|------|
| GR00T N1.5 | 58.4 | 73.9 | 78.1 |
| +EquiActor | 58.8 | 84.1 | 91.0 |
| **EquiVLA** | **60.2** | **84.5** | **92.6** |

> ❓ **예상 질문**: 10% 영역에서 우위가 작은데(60.2 vs 58.4) 동변성의 sample efficiency 주장은 약한 게 아닌가?
> **답변**: 정당한 비판. 40%에서 +10.6pp로 격차가 크게 벌어지지만 10%에서는 +1.8pp에 그침. 매우 저데이터 영역에서는 사전학습 VLM의 prior가 dominant하고 동변 구조의 inductive bias가 충분히 표현되려면 일정 데이터가 필요함을 시사.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | GR00T N1.5 3B (frozen VLM + flow-matching DiT) |
| Group | C₈ (8-fold cyclic, 기본 설정) |
| Optimizer | (논문 명시 없음, 기본 AdamW로 추정) |
| Learning rate | 1×10⁻⁴ |
| Batch size | 64 (sim), 32 (real-robot) |
| LIBERO steps | 데이터셋 크기에 비례 |
| CALVIN steps | 120k |
| Real-robot steps | 150k |
| 학습 시간 (H100) | LIBERO 17h, CALVIN 63h, real 60h (EquiVLA 기준) |
| Inference latency | C₈ 194 ms/step, C₄ 161, C₁₆ 243 |

---

## 5. 실험 설계 및 평가 프로토콜

세 가지 변형 비교:
1. **GR00T N1.5** — non-equivariant baseline
2. **GR00T N1.5 + EquiActor** — action-head equivariance만 isolate
3. **EquiVLA (ours)** — 전체

평가:
- LIBERO 4 suite × relative/absolute control 두 가지
- CALVIN ABCD→D, 1000 chain, single-frame
- Mobile ALOHA 5 task × 20 trial
- Empirical equivariance error ε_eq (M=500, all g ∈ C₈)

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1, Relative Control)

| 모델 | 10 | Goal | Object | Spatial | **Avg** |
|------|----|----|----|----|----|
| π₀† | 73.0 | 93.0 | 86.0 | 90.0 | 86.0 |
| OpenVLA† | 55.0 | 79.2 | 88.4 | 84.7 | 76.8 |
| SmolVLA | 61.0 | 61.4 | 66.0 | 74.0 | 65.6 |
| GR00T N1.5 | 72.0 | 75.0 | 83.4 | 82.0 | 78.1 |
| +EquiActor | 82.6 | 88.0 | 95.2 | 98.2 | 91.0 |
| **EquiVLA** | **87.6** | **89.4** | **98.0** | 95.4 | **92.6** |

### LIBERO (Absolute Control)

| 모델 | Avg |
|------|-----|
| GR00T N1.5 | 62.6 |
| +EquiActor | 73.6 |
| EquiVLA | 76.1 (+13.5pp over baseline) |

- LIBERO-10(long-horizon)에서 +15.6pp(상대) 향상이 가장 큼 — 동변성이 누적 오차 완화에 기여
- Absolute control에서도 +13.5pp 일관된 향상 — 동변성 이점이 제어 모드에 강건

### CALVIN ABCD→D (Table 2, single-frame)

| 모델 | T1 | T2 | T3 | T4 | T5 | **Avg-len** |
|------|----|----|----|----|----|----|
| GR00T N1.5 | 89.0 | 79.2 | 68.7 | 59.4 | 48.5 | 3.45 |
| +EquiActor | 93.7 | 85.8 | 77.8 | 70.1 | 61.9 | 3.89 |
| **EquiVLA** | **95.0** | **88.5** | **81.1** | **73.8** | **64.3** | **4.03** |

- HULC(3.07, multi-frame) 대비 single-frame으로도 우위, MoDE(multi-frame baseline)에 근접
- 5-task에서 GR00T 48.5 → EquiVLA 64.3 (+15.8pp) — long-horizon에서 격차 최대

### Real-robot (Mobile ALOHA, Table 3)
GR00T N1.5 54% → EquiVLA 72% 평균(5 task), 모든 task에서 동등 또는 우수.

### Empirical Equivariance Error (Table 7)
| 모델 | ε_eq |
|------|------|
| GR00T N1.5 | 7.754 ± 3.572 |
| +EquiActor | 0.837 ± 0.738 |
| EquiVLA | (추가 감소, paper Table 7 참조) |

- EquiActor만으로 ε_eq를 ~9배 감소 → architectural equivariance가 실제로 정량적 동변 향상으로 측정됨

> ❓ **예상 질문**: CALVIN에서 single-frame인데 multi-frame MoDE에 근접한다는 게 정말 동변성 덕인가?
> **답변**: 직접 비교 시 baseline GR00T N1.5(3.45)도 이미 HULC multi-frame(3.07) 초과 — VLA pretraining 자체의 강력함이 base. EquiActor의 +0.44, EquiPerceptor의 추가 +0.14가 동변성 기여 정량. Multi-frame 우회는 large-scale pretraining + architectural bias의 결합으로 해석.

---

## 7. Ablation 분석

### Group order (Table 5)
| Group | 10 | Goal | Object | Spatial | Avg | ms/step |
|-------|----|----|----|----|----|---------|
| C₄ | 82.5 | 94.0 | 94.5 | 95.5 | 91.6 | 161 |
| **C₈** | 87.5 | 89.5 | 98.0 | 95.5 | **92.6** | **194** |
| C₁₆ | 87.0 | 95.4 | 98.2 | 96.4 | **94.3** | 243 |

- C₁₆에서 +1.7pp 추가 향상이나 latency 25% 증가 → **C₈가 accuracy-latency sweet spot**
- C₄ → C₈에서는 +1.0pp만 향상, C₈ → C₁₆에서 +1.7pp — 더 fine-grained group이 추가 이득

### EquiActor only vs full EquiVLA
- 상대 제어: 91.0 vs 92.6 (+1.6pp from perceptor)
- 절대 제어: 73.6 vs 76.1 (+2.5pp from perceptor)
- EquiPerceptor가 절대 제어에서 더 큰 기여 — 시각 표현의 회전 일관성이 절대 좌표 예측에 더 중요

### Rotation Generalization (LIBERO-Object ±25° in 5° increments)
- 학습 시 보지 못한 회전 각도에서 EquiVLA가 baseline 대비 우위 유지 — architectural equivariance의 일반화 보장

---

## 8. 관련 연구 비교

| 모델 | Equivariance | Backbone 사용 | Pipeline 범위 | LIBERO Avg |
|------|------|------|------|------|
| EquAct[9] | SE(3) | scratch | point cloud only | (RLBench) |
| Diffusion Policy + FA[4] | SO(2) (approx) | ResNet, scratch | global feature | — |
| Eq.Bot[10] | canonicalization | frozen | wrapper-only | — |
| **EquiVLA** | **SO(2) (chain)** | **frozen VLM** | **vision + DiT 전체** | **92.6** |

### 핵심 차이
- **Frozen pretrained VLM**을 그대로 활용 + architectural equivariance — 둘을 동시에 만족하는 최초 방법
- Token-level FA로 ViT 패치 시퀀스에 FA를 확장 — 이전엔 global pooled vector만 가능
- Flow-matching DiT에 steerable layer를 적용한 최초 사례

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **SO(2) only**: tabletop top-down 카메라가 dominant symmetry라는 가정. Out-of-plane 회전(SO(3))이나 SE(3) full symmetry는 미지원 — mobile manipulation의 일반 시나리오엔 부족
2. **Approximate equivariance**: 유한 grid의 spatial discretization 때문에 정확한 동변성 불가 — Theorem 3의 bound가 작긴 하나 0이 아님
3. **Latency 증가**: C₈에서 baseline 64 ms/step → 194 ms/step (3배). 고주파 제어엔 부담
4. **학습 시간 증가**: LIBERO 5h → 17h (3.4배), CALVIN 19h → 63h (3.3배)
5. **단일 backbone 검증**: GR00T N1.5에만 적용 — π₀, OpenVLA 등 다른 flow-matching/diffusion VLA에 plug-and-play 가능성 미검증
6. **Wrist image는 invariant 처리**: 회전 동변에서 제외 — wrist view가 SO(2) 작용을 받지 않는다는 가정에 의존

### Attribution 문제
- EquiVLA의 향상이 **순수 동변성** 덕인지 **EquiActor의 추가 파라미터/표현력** 덕인지 분리가 부분적 — same-param non-equivariant DiT 비교 없음

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Frozen VLM + flow-matching DiT pipeline 전체에 SO(2) equivariance를 부여한 최초 사례. Token-level FA는 ViT 시대에 의미 있는 일반화 |
| **Technical depth** | ★★★★★ — Spatial homomorphism defect의 폐형 계산(Table 6), end-to-end equivariance bound(Theorem 3) 등 이론적 깊이 강함 |
| **Experimental rigor** | ★★★★☆ — LIBERO/CALVIN/real 3개 벤치마크, 그룹 order ablation, sample efficiency 모두 갖춤. 다만 다른 backbone 검증 부재 |
| **Practical impact** | ★★★☆☆ — Latency 3배 증가가 실제 배포 장벽. 그러나 sample efficiency 측면에서 적은 데이터로 더 강력한 일반화 가능 |
| **Writing quality** | ★★★★☆ — 이론 정리와 실험 흐름이 명확, defect/bound 형식적 정의가 깔끔 |

**강점**: 대규모 사전학습 VLM의 표현력과 architectural equivariance의 inductive bias를 양립시킨 최초의 framework. EquiActor만 단독으로도 +12.9pp 향상이라는 강력한 결과. CALVIN long-horizon에서 single-frame으로 multi-frame baseline 추월.
**약점**: SO(2)에 한정, latency 3배 증가, 단일 backbone 검증. Wrist image 처리의 비대칭성 가정.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 SO(3) 아닌 SO(2)인가? | Tabletop top-down 카메라의 dominant symmetry. SE(3)는 backbone 자체의 viewpoint 일반화로 우회. 일반 mobile manip은 future work |
| 2 | C₈에서 C₁₆ 가면 +1.7pp인데 왜 C₈가 default? | Latency 25% 증가 대비 ROI. Accuracy-latency Pareto frontier에서 C₈가 knee point |
| 3 | EquiPerceptor가 EquiActor 대비 기여 작은 이유? | Wrist image는 invariant 처리되고 frozen ViT의 정보가 이미 풍부 → top-down의 회전 일관성만 정밀화. Action head 쪽이 회전된 trajectory를 생성하는 주체이므로 dominant |
| 4 | Approximate equivariance가 실제 task에 영향? | Table 6의 ∆ bound가 작아서 실용적으론 무시 가능. 다만 super-resolution ViT나 매우 작은 grid에서는 영향 커질 수 있음 |
| 5 | 동변성이 sample efficiency 약속을 데이터 10%에서 못 지키는 이유? | 사전학습 prior가 매우 강하면 architectural bias의 marginal contribution 작음. 40% 이상에서 격차 벌어지는 게 inductive bias의 본질적 효과 |
| 6 | Flow-matching loss가 G-invariant라는 보장의 의미? | Conditional FM loss는 입력 회전 + 출력 회전 쌍에서 동일한 loss 값 → equivariant policy가 같은 gradient signal 받음. ODE 적분이 deterministic이라 inference 시에도 보존 |
| 7 | Wrist image fixed 가정의 영향은? | Real-robot에서 wrist view가 회전 작용을 적게 받음(grasp 자세 중심) → 합리적. 다만 시점 다양성 큰 mobile setup에선 재고 필요 |
| 8 | π₀/OpenVLA 같은 다른 flow-matching VLA에 plug-and-play? | EquiActor가 frozen VLM 뒤에 붙는 module이므로 동일한 DiT 인터페이스를 가진 backbone에 적용 가능할 가능성. 미검증 |
| 9 | Empirical ε_eq가 0이 아닌 이유? | Frozen ViT가 회전 동변하지 않으므로 EquiPerceptor의 token-level FA는 approximate만 — Theorem 3의 spatial discretization defect |
| 10 | 학습 시간 3배 증가가 실제 production에 합리적? | 1회 학습 비용은 inference 시간 ↑보다 덜 중요. 다만 빈번한 fine-tuning이 필요한 환경엔 부담 |

<!-- VERIFIED: pdf -->
