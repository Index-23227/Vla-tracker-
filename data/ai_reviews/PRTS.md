# PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations

> **한 줄 요약**: Qwen3-VL-4B + 675M flow-matching DiT 액션 전문가에 **Contrastive RL(CRL) auxiliary head**를 추가하여, state-action 임베딩과 goal 임베딩의 inner product가 **log-discounted goal occupancy**에 근사하도록 대규모 사전학습. 404M 샘플로 학습하여 LIBERO 98.4%, LIBERO-Plus 81.4%, LIBERO-Pro 58.8%, SimplerEnv WidowX 77.1%, 실로봇 95.9%를 기록.

---

## 1. 배경 및 동기

### VLA 사전학습의 사각지대
- π0, GR00T, OpenVLA-OFT 등 대규모 VLA는 **behavior cloning (BC)** 손실만으로 사전학습됨 → 데이터의 분포를 모방할 뿐 **task-goal 구조** 자체를 학습하지 않음
- 그 결과 distribution shift (LIBERO-Plus 카메라 각도 변화, LIBERO-Pro 새로운 명령 조합) 에서 성능이 급락
- 강화학습(RL)에서 사용되는 **goal-conditioned value function**은 task 구조를 직접 모델링하지만, demonstration 기반 사전학습과 결합되지 않았음

### 핵심 질문
- **VLA 사전학습 단계에서 BC와 goal-conditioned RL 신호를 어떻게 결합할 수 있는가?**
- **이 결합이 OOD generalization (특히 instruction-level 일반화)에 실질적 이득을 주는가?**

📌 [Figure 1] Qwen3-VL backbone + DiT action expert + Contrastive head (s,a) ↔ g

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 컴포넌트 | 구성 | 파라미터 |
|---------|------|---------|
| VLM backbone | Qwen3-VL-4B-Instruct | 4B |
| Action expert | Flow-matching Diffusion Transformer (DiT) | 675M |
| Contrastive head | (s,a) ↔ g projection MLP | (소형) |
| **합계** | | **~4.7B** |

- **Action expert**는 horizon H의 action chunk를 5 denoising step으로 생성
- **Contrastive head**는 사전학습 시에만 사용되는 auxiliary, 추론 시 미사용

### 2.2 Contrastive RL (CRL) 목적함수

PRTS의 핵심 통찰: **goal-conditioned RL의 Q-function 학습을 contrastive representation learning으로 환원**.

- φ(s, a): state-action 임베딩 (encoder)
- ψ(g): goal 임베딩 (encoder)
- 학습 목표: 동일 trajectory 내 (s, a, g) 쌍을 positive, 다른 trajectory의 g를 negative로 InfoNCE loss로 학습
- 이론적 정당화: 수렴 시 **φ(s,a)ᵀψ(g) ≈ log p_π(g | s, a) / p(g)** — 즉, log-discounted goal occupancy에 근사

> ❓ **예상 질문**: 이것이 단순히 InfoNCE를 VLA에 붙인 것과 본질적으로 다른가?
> **답변**: Hafner / Eysenbach 등의 Contrastive RL(CRL) 이론에 기반. 일반 InfoNCE는 representation alignment만 학습하지만, CRL은 **Q-function의 surrogate**로 해석됨. 즉, 학습된 representation이 단순 similarity가 아니라 "이 (s,a)에서 goal g에 도달할 확률"의 로그값으로 의미를 가진다.

> ❓ **예상 질문**: Goal g는 어디서 오는가? Language goal인가 image goal인가?
> **답변**: 논문은 future state representation (또는 future image feature)을 g로 사용. 한 trajectory에서 시간 t의 (s_t, a_t) 와 시간 t+k의 g_{t+k}를 positive 쌍으로 처리. 명령어가 아니라 **임베딩 공간의 미래 상태** — 그래서 instruction-level 일반화에 기여.

### 2.3 학습 손실

총 손실은 BC(flow matching) + CRL의 weighted sum:

L = L_FM(a_t | s_t, ℓ) + λ_CRL · L_InfoNCE((s,a), g)

- L_FM: 표준 flow matching loss on action chunks
- L_InfoNCE: 배치 내 다른 trajectory의 g를 negative로
- λ_CRL: 균형 하이퍼파라미터 (논문 미공개)

### 2.4 추론 시 동작
- 추론에서는 **CRL head를 사용하지 않음** — BC head만 호출
- 즉 추론 비용은 일반 flow-matching VLA와 동일 (5-step DiT denoising)
- CRL은 **representation 사전학습 신호**로만 기능

---

## 3. 데이터셋

### 사전학습 데이터 (404M 샘플, 167.8B 토큰)

| 데이터셋 | 유형 | 비중 |
|---------|------|------|
| AgiBotWorld | Real-world humanoid manipulation | 대규모 |
| RoboMind | Multi-embodiment | 대규모 |
| Open X-Embodiment | Cross-embodiment 22 robots | 중간 |
| Self-collected | TeleAI 자체 수집 | 보조 |

- **167.8B 토큰**은 GR00T-N1 (수십 B) 보다 크고 π0.5와 유사한 규모
- "Packed sequences" 256개로 학습 — token packing으로 효율화

### 평가 데이터셋

| 벤치마크 | 목적 |
|---------|------|
| LIBERO (4 suites) | 표준 fine-tuned 평가 |
| LIBERO-Plus | Zero-shot, 환경 변화 (조명, 카메라, 분포 변화) |
| LIBERO-Pro | Zero-shot, 명령어 재조합 (instruction-level OOD) |
| SimplerEnv WidowX | 4 task 평균 |
| Real RealMan dual-arm | 11 tasks |
| Real Flexiv single-arm | 3 tasks |

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 64 × NVIDIA H100 |
| Global batch | 256 packed sequences |
| Pre-training steps | ~220K (≈ 1 week) |
| Post-training batch | 32 ~ 1,024 (task dependent) |
| Post-training steps | 20K ~ 100K |
| Action denoising steps | 5 |
| Optimizer / LR | 미공개 |

> ❓ **예상 질문**: 64 H100 × 1주는 약 ~10K GPU-hours. 이 규모에서 CRL의 추가 비용은?
> **답변**: Contrastive head는 작은 MLP이고 negative는 in-batch에서 추출하므로 forward/backward 비용 증가는 미미 (~5% 이하 추정). 다만 batch size를 키워야 더 강한 negative를 얻을 수 있어 효과적 batch가 contrastive 품질을 결정.

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (in-distribution, fine-tuned)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.4 | 85.8 | 94.2 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| **PRTS** | **98.8** | **99.8** | **98.4** | **96.6** | **98.4** |

- **LIBERO-Long 96.6%** — 모든 baseline보다 명확히 우세 (vs π0 85.8, GR00T 90.6)

### 5.2 ⭐ LIBERO-Plus (zero-shot, environment shift)

| 모델 | Avg |
|------|-----|
| π0 | ~70% |
| GR00T-N1 | ~74% |
| **PRTS** | **81.4%** |

- Camera angle, lighting, texture 변화에 대한 강건성

### 5.3 ⭐ LIBERO-Pro (zero-shot, instruction recombination)

| 모델 | Avg |
|------|-----|
| baseline | ~50% |
| **PRTS** | **58.8%** |

- **명령어 재조합**: 학습에 없던 동사+명사 조합. 가장 어려운 generalization

### 5.4 SimplerEnv (WidowX, 4 task 평균)

| 모델 | Avg |
|------|-----|
| π0 | ~60% |
| **PRTS** | **77.1%** |

### 5.5 Real-Robot

| Robot | Tasks | SR |
|-------|-------|-----|
| RealMan dual-arm | 11 | **95.9%** |
| Flexiv single-arm | 3 | **90.0%** |

- 11-task dual-arm 95.9%는 동시기 π0.5(약 90%)보다 우세

---

## 6. ⭐ Ablation 분석 (Table 6) — CRL의 실제 가치

| 평가 | w/o CRL | w/ CRL | Δ |
|------|---------|--------|-----|
| LIBERO Avg | 97.8 | 98.4 | **+0.6** |
| LIBERO-Plus Avg | 76.5 | 81.4 | **+4.9** |
| LIBERO-Pro Avg | 53.8 | 58.8 | **+5.0** |

### LIBERO-Pro 세부 (가장 큰 이득)

| 분해 차원 | w/o CRL | w/ CRL | Δ |
|----------|---------|--------|-----|
| Robot state perturbation | ~60% | ~76% | **+16.3** |
| Object position shift | ~55% | ~66% | **+10.5** |
| Task instruction shift | ~50% | ~61% | **+11.4** |
| Visual shift | ~100% | 100% | 0 |

> ❓ **예상 질문**: CRL의 +0.6 (in-dist) vs +5.0 (OOD) 격차는 무엇을 의미하는가?
> **답변**: BC는 in-distribution을 이미 잘 맞춤. CRL의 진짜 가치는 **OOD에서 task structure에 기반한 generalization**. Goal occupancy를 학습한 representation이 unseen 명령에서도 "어떤 (s,a)가 이 goal에 가까운가"를 판단 가능.

> ❓ **예상 질문**: Visual shift에서 100% → 0 이득. CRL이 visual robustness에는 기여하지 않는가?
> **답변**: 정확히 그렇다. Visual 강건성은 backbone (Qwen3-VL) 자체의 vision feature에서 오고, CRL은 **task semantics**에 작용. 두 신호는 직교한다.

---

## 7. 한계 및 미해결 문제

1. **Instruction-level OOD가 여전히 73.8%**: LIBERO-Pro task generalization에서 가장 어려운 부분 — 시각 변화는 100%지만 instruction 재조합은 한계
2. **CRL 손실 가중치 λ_CRL 미공개**: 재현성에 핵심인데 보고 누락
3. **Goal 정의의 ambiguity**: Future state embedding이 정확히 어느 시간 step의 무엇인지 표기 모호 (논문 §3에 sketch만)
4. **In-distribution 이득 미미 (+0.6)**: CRL이 in-dist에 도움을 안 준다면 OOD가 본질적으로 필요하지 않은 deployment에서는 cost-benefit이 약함
5. **Sim-to-real 직접 비교 부재**: SimplerEnv → Real transfer 실험 없음
6. **추론 비용**: 5-step DiT denoising — π0(10-step)보다 적으나 CF-VLA(2-step) 등 최신 효율화 방법 대비 보고된 latency 없음
7. **Contrastive negative 다양성**: In-batch negative만 사용하면 hard negative 부족 — memory bank나 momentum encoder 미사용

### Attribution 문제
- LIBERO 98.4%의 향상이 (a) Qwen3-VL-4B backbone scale, (b) 404M 샘플 데이터 규모, (c) CRL 손실, (d) 5-step flow matching DiT 중 어디서 오는지 분리 어려움
- Ablation Table 6의 CRL 기여는 명확하나, backbone/데이터 ablation 부재
- "CRL이 핵심 contribution"이라는 주장 대비 +0.6 (LIBERO) 이득은 작아 보일 수 있음 — 단, OOD 이득이 본 논문의 셀링 포인트

---

## 8. 관련 연구 비교

| 모델 | Backbone | Aux pretraining loss | LIBERO Avg | OOD 평가 |
|------|----------|---------------------|-----------|---------|
| OpenVLA | Llama-7B + SigLIP | (none) | 76.5 | ✗ |
| π0 | PaliGemma-3B | (none) | 94.2 | ✗ |
| π0.5 | PaliGemma-3B | High-level VLM reasoning | 94.2+ | △ |
| GR00T-N1 | Cosmos | Video prediction aux | 93.9 | △ |
| OpenVLA-OFT | Llama-7B | (none, OFT recipe) | 95+ | ✗ |
| **PRTS** | **Qwen3-VL-4B** | **Contrastive RL (CRL)** | **98.4** | **LIBERO-Plus/Pro** |

### 핵심 차이
- 다른 VLA가 사전학습에서 video prediction, high-level reasoning을 aux로 쓰는 반면, PRTS는 **goal-conditioned value function**을 representation으로 학습
- OOD 평가 (LIBERO-Plus, Pro)를 본격적으로 보고하는 거의 유일한 동시기 연구

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — CRL을 VLA pretraining에 통합한 첫 사례. 이론적 토대 있음 |
| **Technical depth** | ★★★★☆ — log-discounted goal occupancy 해석, Qwen3-VL + DiT 결합 |
| **Experimental rigor** | ★★★★★ — LIBERO/Plus/Pro/Simpler/Real dual+single-arm 망라, OOD ablation 풍부 |
| **Practical impact** | ★★★★☆ — 추론 비용 동일, 사전학습만 강화 → 기존 VLA 파이프라인에 plug-in 가능 |
| **Writing quality** | ★★★☆☆ — Goal 정의/하이퍼파라미터 표기 모호 |

**강점**: VLA 사전학습에 task-structural 신호(CRL)를 도입하여 OOD generalization을 크게 개선. LIBERO-Plus +4.9, LIBERO-Pro +5.0은 동시기 VLA 중 최상위급. Open-source. **약점**: In-distribution 이득은 작고(+0.6), goal 정의/λ 등 핵심 디테일 누락, instruction-level OOD는 여전히 60% 미만.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | CRL 없이 같은 backbone/데이터로 LIBERO Avg가 97.8%인데, +0.6이 핵심 contribution인가? | In-dist 이득은 작지만 OOD(+4.9, +5.0)가 진짜 셀링. "Generalization 강화 사전학습"으로 재포지셔닝 |
| 2 | Goal g의 정확한 정의는? Future state? Future image? 그 time horizon은? | 논문 §3 표기 모호. Future state embedding을 trajectory에서 임의 future step에서 추출하는 것으로 추정 |
| 3 | InfoNCE의 negative는 in-batch만? Memory bank나 hard negative mining은? | In-batch only로 보임. Batch 256 packed sequences가 충분한 다양성을 주는지 의문 |
| 4 | LIBERO-Pro의 visual shift에서 100% → CRL 기여 0. 정말로 task shift에만 작용하는가? | Yes. 이는 CRL이 representation의 **semantic 축**만 강화함을 입증. Backbone과 직교 |
| 5 | CRL과 BC(flow matching) loss의 weight λ_CRL은? | 논문 미보고 — 재현성의 핵심 누락 |
| 6 | π0.5의 high-level reasoning aux 대비 CRL의 이론적 우위는? | π0.5는 language model의 reasoning에 의존(implicit). CRL은 **수렴 시 log p(g|s,a)/p(g)** 라는 명시적 의미. RL theory가 뒷받침 |
| 7 | Goal-conditioned RL인데 왜 actual RL fine-tuning은 안 했는가? | 본 논문은 supervised pretraining만. RL fine-tuning은 future work — CRL representation이 RL value 초기화로 쓸 수 있다는 함의 |
| 8 | 404M sample, 167.8B 토큰의 데이터 source가 AgiBotWorld 중심인데, embodiment bias 우려는? | RoboMind + OpenX 포함으로 완화. 다만 SO-100 등 desktop arm 데이터 부족 가능 |
| 9 | LIBERO-Pro에서 instruction shift +11.4가 task generalization 최댓값. 그래도 73.8%인 이유는? | "Pick the red cup and put on blue plate" → "Place the blue plate under the red cup" 같은 reordering이 여전히 어렵다. 명령 파싱-action mapping의 compositional gap |
| 10 | Real-robot 95.9%는 11 task 평균인데, 가장 낮은 task는? | 논문 보고 없음 — 평균만 |
| 11 | 추론 latency 미보고. 5-step DiT는 몇 ms? | 미보고. CF-VLA(2-step 7.81 ms) 대비 정량 비교 불가 |
| 12 | CRL이 작동하려면 trajectory가 goal-reaching이어야 하는데 실패 demonstration에서는? | 흥미로운 한계. 본 논문은 success-only data 가정 — 실패 데이터의 contrastive negative 활용은 future work |

<!-- VERIFIED: pdf -->
