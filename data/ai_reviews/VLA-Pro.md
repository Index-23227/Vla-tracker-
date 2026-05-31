# VLA-Pro: Cross-Task Procedural Memory Transfer for Vision-Language-Action Models

> **한 줄 요약**: 학습 시 task별 LoRA adapter를 "절차적 기억(procedural memory)"으로 저장하고, 추론 시 action type/object geometry/end-effector orientation/target interaction point로 top-k 메모리를 검색해 softmax 가중 융합하는 **backbone-agnostic** VLA 향상 프레임워크. RoboTwin held-out tasks에서 RDT 기준 **+207%**, pi0.5 기준 47% 상대 향상.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 VLA(π0, RDT, X-VLA)는 학습한 task 분포 안에선 강하지만, **새로운 조작 task에 zero-shot/few-shot 일반화**가 약함.
- 일반화 향상을 위한 기존 접근:
  - **데이터 확장** → expensive, scene-specific demo가 필요.
  - **Skill library / retrieval** → 보통 raw observation 또는 language embedding으로 retrieval, **물리적 절차(procedural knowledge)** 자체를 저장하진 않음.
  - **LoRA 다중 expert (MoE-LoRA)** → 학습 시 expert 결정이 고정, 새 task에 동적 적응 불가.

### 핵심 질문
- "**컵을 옆으로 눕혀 쥐는 절차**"는 컵 → 와인잔 → 머그잔 등 여러 task에서 재사용 가능. 이 **절차 자체를 파라미터로 저장**할 수 있을까?
- LoRA adapter를 **task의 미분가능한 procedural memory unit**으로 보고, 새 task에 대해 retrieval + fusion으로 즉시 활용할 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 전체 파이프라인

VLA-Pro는 두 단계 학습:
1. **Base LoRA**: 8개 training task의 혼합 데이터로 backbone(pi0.5 / RDT / X-VLA) 위에 공통 base LoRA를 학습.
2. **Task-Specific LoRA**: 각 task별로 별도의 LoRA adapter를 fine-tune해 procedural memory bank $\mathcal{M} = \{(k_i, \theta_i^{\text{LoRA}})\}$ 구축. $k_i$는 task 메타 키, $\theta_i^{\text{LoRA}}$는 adapter weight.

### 2.2 Memory Key 설계

각 LoRA는 다음 네 가지 속성으로 indexing:
- **Action type**: pick / place / rotate / pour 등 high-level verb
- **Object geometry**: target object의 형상 클래스 (cylindrical, rectangular, spherical, ...)
- **End-effector orientation**: 접근 시 gripper 방향 (top-down, side, angled)
- **Target interaction point**: 접촉 위치 (center, edge, handle)

이는 단순 instruction embedding보다 **물리적 절차에 가까운** indexing이라는 점이 핵심.

### 2.3 추론 시 Retrieval & Fusion

새 task에 대해:
1. VLM이 위 네 속성을 명령/이미지로부터 추출 → query $q$.
2. Cosine similarity로 top-k 메모리 검색.
3. Softmax 가중 융합:
$$\theta^{\text{LoRA}}_{\text{use}} = \sum_{i \in \text{top-}k} \frac{\exp(\text{sim}(q, k_i)/\tau)}{\sum_j \exp(\text{sim}(q, k_j)/\tau)} \theta_i^{\text{LoRA}}$$
4. 융합된 adapter를 base backbone에 합성 → action chunk 생성.

LoRA의 low-rank 특성 덕에 가중합이 잘 정의되고 메모리 크기도 task당 수 MB 수준.

### 2.4 Backbone-Agnostic 검증

같은 procedural memory 메커니즘을 세 가지 backbone에 적용:
- **pi0.5** (flow-matching)
- **RDT** (Robotics Diffusion Transformer)
- **X-VLA**

세 backbone 모두에서 일관된 향상을 보여, framework가 specific architecture에 묶이지 않음을 입증.

> **예상 질문**: top-k fusion이 conflict하는 절차를 평균낼 위험은?  
> **답변**: τ를 작게(peaked) 두면 사실상 winner-take-all이 되어 conflict 완화. 논문은 또한 retrieval key의 직교성(action type × geometry × ...) 덕에 자연스러운 disentangle이 일어난다고 주장.

---

## 3. 데이터셋

| 데이터 | 용도 | 비고 |
|--------|------|------|
| RoboTwin (8 training tasks) | base LoRA + per-task LoRA | dual-arm sim |
| RoboTwin (9 held-out test tasks) | 평가 (zero-shot / few-shot) | 학습에 미포함 |
| 실제 로봇 6 held-out tasks | real-world 평가 | tabletop manipulation |

memory bank 크기는 8 (train task 수)로 작은 편이지만, 본 논문의 평가 가정은 "**small bank + smart retrieval**".

---

## 4. 실험 결과

### 4.1 RoboTwin 9 Held-Out Tasks (성공률, %)

| Backbone | Baseline | **VLA-Pro** | 상대 향상 |
|----------|----------|-------------|----------|
| pi0.5 | ≈40 | **59.3** | +47% |
| RDT | ≈11 | **34.1** | **+207%** |
| X-VLA | ≈17 | **30.0** | +76% |

특히 RDT 기준 **+207%**의 향상이 두드러짐 — base backbone의 일반화가 약할수록 procedural memory의 효과가 큼.

### 4.2 RLBench Zero-Shot

| 모델 | RLBench Avg |
|------|-------------|
| pi0.5 baseline | ≈13.8 |
| **VLA-Pro (pi0.5)** | **20.9** (+51%) |

학습 분포 밖 환경(RLBench)에 대한 zero-shot transfer에서도 향상 — procedural memory가 단순 task overfitting이 아님을 시사.

### 4.3 실제 로봇 6 Held-Out Tasks

| 모델 | 평균 성공률 |
|------|------------|
| pi0.5 baseline | 5.8% |
| **VLA-Pro (pi0.5)** | **65.0%** |

11x 향상은 인상적이나, baseline 5.8%는 매우 낮은 수치라 절대 비교라기보다 "procedural memory가 fine-tuning 데이터 없이 작동"의 증거로 해석.

---

## 5. 어블레이션 (논문 보고 요약)

| 설정 | 성공률 변화 |
|------|------------|
| Full retrieval (top-k=3) | baseline |
| Top-1 retrieval | 약간 ↓ |
| Random LoRA selection | ↓↓ (procedural matching이 핵심) |
| Action type만 indexing | ↓ |
| Object geometry만 indexing | ↓ |
| No memory (base LoRA만) | ↓↓ (가장 큰 drop) |

네 가지 indexing 속성이 모두 기여하며, 단일 속성으로는 부족.

---

## 6. 한계

1. **Memory bank 확장성 미검증**: 본 실험은 8 task base만 사용. 1000 task scale에서 retrieval cost / 융합 품질이 어떻게 변할지 불명.
2. **Procedural matching 가정**: 새 task의 4가지 속성을 VLM이 정확히 추출한다는 가정 — 추출 오류가 retrieval을 망가뜨릴 수 있음.
3. **Backbone 학습 비용 미공개**: pi0.5/RDT/X-VLA를 각각 fine-tune하는 hardware/시간이 명시되지 않음.
4. **LIBERO 미평가**: 트래커가 보유한 다른 모델과의 직접 비교가 어려움.
5. **Conflict resolution**: 두 메모리가 정반대 절차를 권할 때(예: top-down vs side approach) softmax 평균이 정말 안전한가는 추가 연구 필요.

---

## 7. 예상 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 왜 instruction embedding이 아니라 4-속성 indexing인가? | Instruction은 표면적 paraphrase에 흔들리지만, action type / object geometry / orientation / interaction point는 **물리적으로 grounded**. "병뚜껑을 돌려 열어라"와 "snap off the cap"이 같은 절차임을 instruction embedding보다 더 잘 매칭. |
| 2 | RDT에서 +207%면 base가 너무 낮은 것 아닌가? | 그렇다. RDT held-out 성능이 11% 수준이라 절대 향상이 23%p — 인상적이지만 절대 SOTA는 pi0.5 backbone(59.3%). 상대 향상보다 backbone별 절대 수치를 같이 봐야 함. |
| 3 | LoRA를 softmax-평균하면 rank가 부풀려져 효과적 rank가 떨어지지 않나? | 이론적으로 가능. 다만 top-k가 작고(=3) τ가 작아 사실상 dominant adapter 한두 개의 가중합 → 실효 rank 손실은 제한적. |
| 4 | 학습/추론 비용은? | 학습 시 task당 LoRA hyperparameter (rank, alpha)와 fine-tune step이 추가. 추론 시 retrieval은 O(memory size × key dim)으로 무시 가능. LoRA 합성도 한 번의 weight 합산. |
| 5 | Real-world 6 tasks에서 baseline 5.8%는 비정상적으로 낮음 — 비교가 공정한가? | pi0.5는 시뮬레이션 fine-tune 후 zero-shot real → 5.8%는 sim-to-real gap. VLA-Pro의 65%는 procedural memory가 sim-to-real을 부분적으로 메운다는 신호지만, fair comparison엔 동일 sim-to-real strategy의 baseline이 필요. |

<!-- VERIFIED: pdf -->
