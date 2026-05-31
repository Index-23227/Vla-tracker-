# DyGRO-VLA: Cross-Task Scaling of Vision-Language-Action Models via Dynamic Grouped Residual Optimization

> **한 줄 요약**: RL로 VLA를 최적화할 때 발생하는 "generalist → narrow specialist 붕괴" 문제를, **2단계 프레임워크**(Stage 1: Information Bottleneck 기반 cross-task latent representation, Stage 2: Mixture-of-RL-Residuals)로 해결. DINOv2 + SigLIP + Qwen2.5-0.5B 백본 위에서 LIBERO 97.1% avg (+4.4pp over offline base, Long 95.0%), RoboTwin2 4-task 79.2%, Sim2Real 4-task 57.5%를 달성.

---

## 1. 배경 및 동기

### 문제 정의: Generalist-to-Specialist Collapse
- RL은 trajectory imitation을 넘어 환경 상호작용으로 VLA의 정밀 제어를 개선할 수 있는 유망한 경로
- 그러나 대부분의 RL 최적화는 **task-specific** → 본래 generalist controller로 설계된 VLA를 좁은 태스크 집합에 과적합시키는 부작용
- 이 collapse는 다른 태스크 성능 저하 + distribution shift 취약성을 동시에 야기

### 핵심 질문
- **Cross-task representation을 RL fine-tuning에서 보존할 수 있는가?**
- **Multi-task RL의 task interference를 어떻게 명시적으로 완화할 것인가?**
- **Representation 학습과 policy 학습을 분리해야 할까?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 컴포넌트 | 역할 |
|---------|------|
| **DINOv2** | Vision encoder (geometric/spatial features) |
| **SigLIP** | Language model (instruction encoding) |
| **Qwen2.5-0.5B** | LLM core (multimodal fusion) |
| Multi-head attention fusion | Vision + language + proprio 결합 |
| **MoRR (Mixture-of-RL-Residuals)** | 8개 expert residual policy + dynamic routing |

### 2.2 Stage 1 — Offline Pre-training (IB Objective)

Information Bottleneck 원리:
```
max  I(Z; A) - λ_IB · I(Z; O)
```
- Z: latent representation
- A: action, O: observation
- λ_IB: compression-vs-task-relevance trade-off

Variational lower bound 형태:
```
L_base = E[-log π_θ(a|z)] + λ_IB · [E_{P_OZ}[T_ψ(o,z)] - log E_{P_O·P_Z}[exp(T_ψ(o,z))]]
```
- 두 번째 항은 MINE-style mutual information 추정
- Task embedding은 **contrastive loss**로 동시 학습:
```
L_CL(ψ) = -E_{(z̃_T, ζ)} [ log(exp(sim(z̃_T, e_ζ)/τ) / Σ_{ζ'} exp(sim(z̃_T, e_ζ')/τ)) ]
```

### 2.3 Stage 2 — Online Fine-tuning (MoRR)

Mixture-of-RL-Residuals는 base policy를 **직접 수정하지 않고** residual을 예측:
```
Δa = Σ_i ω_i(z̃_T) · ã_i
```
- ω_i: task embedding z̃_T에 의해 routing되는 gating weight
- ã_i: i번째 expert residual
- Base policy + Δa = final action

RL 알고리즘: Soft Actor-Critic 변형 + h-step action chunking.

> ❓ **예상 질문**: Mixture가 아닌 단일 residual head로는 안 되는가?
> **답변**: Ablation에서 Single Expert 93.4% vs 4 Experts 97.0% vs 8 Experts 97.1%. Multi-expert가 +3.7pp 결정적 기여. 단일은 task 간 interference 완화 capacity 부족.

> ❓ **예상 질문**: Mixture-of-experts의 routing instability·expert collapse는?
> **답변**: Task embedding이 contrastive로 명시적으로 정렬되어 routing의 결정성이 보장. Ablation에서 "no contrastive learning" -2.8pp로 가장 큰 손실 → contrastive가 routing 안정성의 핵심.

---

## 3. 데이터 전략

- LIBERO 4-suite co-training (Spatial/Object/Goal/Long 동시)
- RoboTwin 2.0 4-task simulation subset
- Real robot Sim2Real: 같은 4-task, 20 trials/task, domain randomization 사전 적용

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO 4-Suite

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| Diffusion Policy | 59.6 | 73.8 | 51.6 | 41.0 | 56.5 |
| MT-ACT | 50.2 | 72.0 | 60.2 | 50.2 | 58.2 |
| Octo | 78.7 | 85.5 | 84.2 | 51.0 | 74.9 |
| OpenVLA | 82.1 | 87.3 | 77.4 | 51.8 | 74.7 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π₀-FAST* | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| DyGRO-VLA (SFT) | 95.4 | 96.0 | 93.8 | 85.0 | 92.6 |
| DyGRO-VLA (Offline) | 95.6 | 96.0 | 94.0 | 85.2 | 92.7 |
| **DyGRO-VLA** | **97.6** | **98.6** | **97.2** | **95.0** | **97.1** |

- **LIBERO-Long 95.0%** — π₀의 88.4% 대비 +6.6pp, 가장 큰 격차
- Offline base 92.7% → Full RL 97.1% = +4.4pp absolute 개선
- 특히 Long에서 +9.8pp 개선 → RL이 long-horizon에서 가장 큰 기여

### 4.2 RoboTwin 2.0 (4-task subset)

| Method | Beat Block Hammer | Pick Dual Bottles | Stack Bowls Two | Place Empty Cup | Avg |
|--------|------------------|-------------------|----------------|----------------|-----|
| OpenVLA-oft (SFT) | 54.0 | 32.0 | 88.0 | 50.0 | 56.0 |
| OpenVLA-oft (RFT) | 71.9 | 54.0 | 92.0 | 96.1 | 78.5 |
| **DyGRO-VLA** | **72.2** | **57.0** | **90.4** | **97.0** | **79.2** |

- OpenVLA-oft RFT(78.5%) 대비 +0.7pp — modest 격차
- Place Empty Cup 97% — near saturation
- 가장 어려운 Pick Dual Bottles에서 +3pp (54 → 57)

### 4.3 Sim2Real Transfer (20 trials/task)

| Task | Sim | Real | Δ (sim-real) |
|------|-----|------|--------------|
| Beat Block Hammer | 72.2 | 30.0 | -42.2 |
| Pick Dual Bottles | 57.0 | 40.0 | -17.0 |
| Stack Bowls Two | 90.4 | 90.0 | -0.4 |
| Place Empty Cup | 97.0 | 70.0 | -27.0 |
| **Avg** | **79.2** | **57.5** | **-21.7** |

- vs OpenVLA-oft (RFT) Real: 55.0% → DyGRO +2.5pp
- **Stack Bowls Two는 sim-real gap이 거의 없음**(0.4pp) — domain randomization이 효과적인 task
- **Beat Block Hammer의 sim-real gap이 42.2pp**로 가장 큼 — 정밀 접촉/타격 task가 sim-real 격차에 취약

---

## 5. Ablation 분석

| Component Removed | Impact |
|-------------------|--------|
| Without IB Objective | -0.3pp |
| **Without Contrastive Learning** | **-2.8pp** |
| Without Difficulty-Aware Sampling | -0.7pp |
| Single Expert (vs 4) | -3.7pp (93.4 vs 97.0) |
| 4 Experts | 97.0pp |
| **8 Experts (default)** | **97.1pp** |

### 해석
1. **Contrastive learning이 가장 결정적 (-2.8pp)** — task embedding의 정렬이 routing의 핵심
2. **IB objective는 -0.3pp만 영향** — 의외로 modest. 명명만큼 결정적이지 않음
3. **Expert 수 4→8은 +0.1pp만**으로 saturation — 8개가 충분
4. **Difficulty-aware sampling -0.7pp** — hard task에 weight 부여가 약하지만 일관 양의 기여

> ❓ **예상 질문**: IB objective가 -0.3pp만 영향이면 paper의 이름값을 정당화하는가?
> **답변**: 이름과 ablation 영향의 불균형. IB는 representation에 inductive bias로 작동하나 contrastive와 MoRR가 실제 driver. Paper title의 "Dynamic Grouped Residual"이 더 본질적 기여.

---

## 6. 관련 연구 비교

| 모델 | RL 사용 | Multi-task interference 처리 | Cross-task representation |
|------|--------|------------------------------|--------------------------|
| OpenVLA-oft (RFT) | ✓ | ✗ (단일 head) | △ |
| iVideoGPT-RL | ✓ | ✗ | ✗ |
| **DyGRO-VLA** | **✓ (2-stage)** | **✓ (MoRR)** | **✓ (IB + contrastive)** |

핵심 차이: **MoRR**가 expert-level에서 task별 적응을 제공하면서 base policy는 generalist로 보존.

---

## 7. 한계 및 미해결 문제

1. **GitHub 미공개**: code release URL 없음 → 재현 부담
2. **RoboTwin 4-task subset의 협소함**: 31-task 전체가 아니라 4개에 한정 → RoboTwin 평균에서의 generalist 능력 검증 부족
3. **Beat Block Hammer의 sim-real gap 42pp**: domain randomization이 정밀 접촉 task에서 한계
4. **OpenVLA-oft RFT 대비 격차가 작음**: RoboTwin 79.2 vs 78.5 = +0.7pp는 MoRR의 추가 복잡도를 정당화하기에 작을 수 있음
5. **IB objective의 modest 기여(-0.3pp)**: 명명과 실제 ablation 영향의 불일치
6. **Affiliations 부재**: arXiv abstract에서 명확한 institutional affiliation이 listed되지 않음
7. **CALVIN, SimplerEnv 평가 부재**: LIBERO + RoboTwin 4-task에 한정
8. **MoRR routing의 해석성**: 어떤 expert가 어떤 task에 활성화되는지의 시각화 부재

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — VLA × RL의 generalist collapse 문제를 정면 제기, MoRR로 elegant solution |
| Technical depth | ★★★★☆ — IB + contrastive + MoRR + SAC 통합이 체계적 |
| Experimental rigor | ★★★☆☆ — LIBERO는 강력하나 RoboTwin은 4-task subset; Sim2Real는 통계적 분산 부족 |
| Practical impact | ★★★★☆ — LIBERO-Long 95% + Sim2Real 57.5%는 generalist VLA RL의 실용성 시사 |
| Writing quality | ★★★★☆ — 2-stage framework의 동기와 ablation이 명확 |

**강점**: VLA × RL 연구의 핵심 함정인 **generalist 붕괴**를 정면 제기. Representation(Stage 1)과 policy(Stage 2)를 분리하고, 후자에서 mixture-of-residuals라는 modular 구조 제안. LIBERO-Long +9.8pp 개선이 강력한 결과. **약점**: IB ablation impact(-0.3pp)가 modest해 이름값과 다소 불일치. RoboTwin은 4-task subset에만 평가. Sim-real gap 42pp(Beat Block Hammer)는 sim2real의 본질적 한계 미해결.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | IB objective가 -0.3pp만 영향인데 이름에 "IB"가 핵심처럼 보이는 이유? | Ablation에서 IB는 modest. 실제 driver는 contrastive(-2.8pp)와 MoRR(-3.7pp). Paper title의 "Dynamic Grouped Residual"이 더 본질적 |
| 2 | LIBERO Long 95% vs π₀ 88.4%의 격차가 정말 RL 덕분인가, 데이터 덕분인가? | DyGRO-VLA (Offline) 85.2 → Full 95.0 = +9.8pp가 RL 단계의 기여. 그러나 SFT 85.0과 Offline 85.2 차이는 미미 → IB stage 자체 기여는 작음 |
| 3 | RoboTwin 4-task 79.2 vs OpenVLA-oft RFT 78.5 = +0.7pp는 noise level 아닌가? | Statistical significance 명시 부재. Place Empty Cup +0.9, Pick Dual Bottles +3.0 등 task별 일관 양의 기여는 단순 noise는 아닐 가능성 |
| 4 | MoRR의 8 experts vs 4 experts가 +0.1pp만 차이면 과잉 아닌가? | 그렇다, saturated. 4 experts가 sweet spot으로 보임. 8 experts는 minor diminishing return |
| 5 | Sim-Real gap 42pp(Beat Block Hammer)는 method의 generalization 한계인가? | Domain randomization 한계 + 정밀 접촉 task의 본질적 sim-real 차이. Method가 generalist를 보존하나 sim-real bridging까지 해결하지는 못함 |
| 6 | Contrastive learning이 -2.8pp로 가장 큰데, routing 시각화는? | 본문 확인 필요. Routing weight visualization이 없으면 expert specialization 주장이 weak |
| 7 | Backbone Qwen2.5-0.5B는 작은데 LIBERO 97.1%가 가능한 이유? | RL fine-tuning이 작은 backbone에서도 큰 폭의 향상 가능함을 시사 (Offline 92.7 → RL 97.1). 0.5B는 RL의 sample efficiency가 잘 작동하는 sweet spot일 수 있음 |
| 8 | "Generalist 붕괴" 검증은 — task interference가 실제로 측정되었나? | Ablation의 expert 수 변화가 indirect evidence. Direct measurement(예: task A 학습 후 task B 성능 drop)는 본문에서 확인 필요 |

<!-- VERIFIED: pdf -->
