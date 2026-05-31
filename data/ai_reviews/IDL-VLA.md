# IDL-VLA: Mitigating State Aliasing in Vision-Language-Action Models via Inverse Dynamics Learning

> **한 줄 요약**: VLA의 vision encoder를 inverse-dynamics objective(현재·미래 관측으로부터 그 사이 action chunk를 예측)와 Pseudo Time Reversal(PTR)로 직접 supervise하는 plug-in 학습 기법. **추론 시 auxiliary head를 제거**하므로 deployed 모델은 baseline과 동일한 컴퓨트지만 state aliasing이 줄어 visual fine-grained 구분 능력이 향상. VLM4VLA(Qwen3-VL) + IDL은 CALVIN ABC→D에서 4.03 → **4.10**, SpatialVLA + IDL은 SimplerEnv-Bridge에서 30.2% → **33.3%**.

---

## 1. 배경 및 동기

### "State aliasing" 문제 정의
- VLA의 vision encoder는 *주로 contrastive/language alignment*로만 pretraining되어, **시각적으로 유사하지만 행동이 달라야 하는 상태**를 구분하지 못함
- 예: 같은 컵이 약간 다른 각도로 놓였을 때 grasp pose가 달라야 하지만, encoder가 두 상태를 동일하게 인코딩 → policy가 평균 행동을 출력 → 실패
- 이를 *state aliasing*이라 명명. 직접 회귀(direct regression) VLA에서 특히 심각

### 핵심 질문
- **Vision encoder를 행동 차이에 민감하도록** 추가 supervision으로 어떻게 가르칠 것인가?
- 추론 비용을 늘리지 않고 baseline 위에 얹을 수 있는가?

📌 [Figure 1] — Auxiliary inverse-dynamics head가 (current obs, future obs)로 action chunk 예측 → encoder를 직접 supervise. 추론 시 head 제거.

---

## 2. 방법론 심층 분석

### 2.1 Inverse Dynamics Learning (IDL)

표준 VLA loss:
$$\mathcal{L}_{\text{VLA}} = \|a_{t:t+H} - \pi_\theta(o_t, l)\|$$

IDL augmented:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VLA}} + \lambda \cdot \|a_{t:t+H} - g_\phi(\text{enc}(o_t), \text{enc}(o_{t+H}), \text{enc}(o_t) - \text{enc}(o_{t+H}))\|$$

- $g_\phi$ = patch-wise fusion MLP (auxiliary head)
- Encoder gradient flows from both terms → encoder가 state-discriminative 표현 학습

### 2.2 Pseudo Time Reversal (PTR)
- Trajectory의 일부를 역순으로 뒤집어 augmentation
- 효과: encoder가 *양방향* action을 모두 학습 → action 방향 다양성이 늘어 state aliasing 해소
- 단, 실제 물리는 비가역이므로 PTR로 생성된 trajectory가 실행 가능하지 않을 수 있음 → encoder *only* supervision으로 한정

### 2.3 추론 시 head 제거
- 학습 후 $g_\phi$는 폐기, $\pi_\theta$만 deploy
- → **추론 컴퓨트는 baseline과 동일**, 메모리도 동일
- 이는 LoRA/adapter류와 차별화되는 점: 추론 비용 zero overhead

> ❓ **예상 질문**: 왜 forward dynamics가 아닌 inverse dynamics인가?
> **답변**: Forward는 (o_t, a) → o_{t+H}로 image 생성 → diffusion 등 무거운 decoder 필요. Inverse는 (o_t, o_{t+H}) → a로 작은 MLP로 충분. 또한 inverse는 *action에 필요한 visual 차이*를 정확히 학습하므로 state aliasing 직격.

> ❓ **예상 질문**: λ는 어떻게 정하는가?
> **답변**: 논문에 명시적 sweep 부재. 통상 0.1~1.0 범위로 추정.

---

## 3. 데이터 전략

| 데이터 | 사용처 |
|--------|--------|
| CALVIN ABC | VLM4VLA, FLOWER + IDL 학습 |
| Bridge | SpatialVLA + IDL 학습 (SimplerEnv 평가) |
| LIBERO-90 | Pretraining 평가 |

- 추가 annotation **없음** — 기존 (obs, action) pair만 사용

---

## 4. 실험 결과 심층 분석

### 4.1 CALVIN ABC→D (Table 1)

| 모델 | Avg Seq Len |
|------|------------|
| VLM4VLA baseline | 4.03 |
| **VLM4VLA + IDL (Ours)** | **4.10 (+0.07)** |
| FLOWER baseline | 4.54 |
| **FLOWER + IDL** | **4.56 (+0.02)** |

- VLM4VLA에서 +0.07, FLOWER에서 +0.02 — 향상 폭이 baseline 강도에 반비례
- FLOWER가 이미 saturate 영역(4.54/5) → 추가 마진 적음

### 4.2 SimplerEnv-Bridge (Table 2)

| 모델 | Average | Eggplant | Cube |
|------|---------|----------|------|
| SpatialVLA baseline | 30.2 | 54.2 | 16.7 |
| **SpatialVLA + IDL** | **33.3 (+3.1)** | **66.6 (+12.4)** | **20.8 (+4.1)** |

- Eggplant +12.4%p는 인상적 — 시각적으로 비슷한 회전 상태에서 다른 grasp이 필요한 태스크에서 효과 큼
- 전체 평균 +3.1%p는 일관된 향상

### 4.3 Baseline 다양성
- Direct regression (VLM4VLA), flow-matching (FLOWER), 3D-spatial (SpatialVLA) 세 가지 architectural family에서 모두 향상 → IDL의 plug-and-play 주장 강화

---

## 5. Ablation 분석
- PTR 없이 IDL only? IDL 없이 PTR only? 등 명시적 ablation 본문에 부재 (정보 한계)
- λ sweep 부재

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **향상 폭이 작음**: CALVIN +0.07 (1.7% relative), SimplerEnv +3.1%p — 의미는 있으나 paradigm-shifting은 아님
2. **Model name이 명확하지 않음**: "IDL", "auxiliary inverse dynamics learning"로만 호칭 — paper 자체에 모델명 부재 (이 리뷰는 IDL-VLA로 호칭)
3. **Code 미공개**: 재현성 보장 없음
4. **PTR의 물리적 정당성**: 비가역 동역학에서 역순 trajectory가 어떻게 유의미한지 이론적 분석 부재
5. **LIBERO/RoboTwin 미평가**: 핵심 manipulation benchmark에서의 검증 부재

### Attribution 문제
- Encoder에 추가 gradient를 흘리는 것 자체의 효과(단순 regularization)와 inverse dynamics specific한 효과를 분리하기 어려움
- Random auxiliary task control 부재 → "inverse dynamics가 특별히 좋다"는 주장이 약함

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Inverse dynamics는 RL 문헌에서 오래된 아이디어, VLA에 적용은 새로움 |
| **Technical depth** | ★★★☆☆ — Simple objective + PTR — 직관적이지만 깊이 부족 |
| **Experimental rigor** | ★★★☆☆ — 3 baseline은 좋으나 ablation 부재 |
| **Practical impact** | ★★★★☆ — Zero inference overhead는 매우 매력적 |
| **Writing quality** | ★★★☆☆ |

**강점**: 추론 비용 zero overhead로 일관된 +3.1%p (SimplerEnv) / +0.07 (CALVIN). 3개 architectural family에서 작동. **약점**: 향상 폭이 marginal, ablation 부재, code 미공개, paper 내 모델 이름이 모호.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Random auxiliary task와 비교하면? | 통제 실험 부재 — auxiliary loss 자체의 regularization 효과를 분리 못함 |
| 2 | PTR의 물리적 정당성? | 비가역 동역학에서 역순 trajectory는 invalid. Encoder supervision으로만 한정한다고 논문에 명시 |
| 3 | λ는 어떻게 정하는가? | 명시적 sweep 부재 — task별 hyperparameter sensitivity 미보고 |
| 4 | CALVIN +0.07이 통계적으로 유의한가? | seed/표준편차 미보고 — 단일 run일 가능성 |
| 5 | LIBERO에서는? | 평가 부재 |
| 6 | OpenVLA, π₀ 같은 mainstream baseline은? | 평가 부재 — VLM4VLA, FLOWER, SpatialVLA만 |
| 7 | Inverse dynamics head 크기는? | Patch-wise fusion MLP — 정확한 크기 미보고 |
| 8 | State aliasing이 실제로 줄었는지 측정? | 정성적 분석 부재 — 정량 증거가 benchmark 점수 뿐 |

<!-- VERIFIED: pdf -->
