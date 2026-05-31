# AttenA+: Rectifying Action Inequality in Robotic Foundation Models

> **한 줄 요약**: VLA의 모든 timestep loss를 동등하게 다루는 관행이 **action inequality**(저속·정밀 구간이 task success를 결정하지만 supervision은 고속·이동 구간이 압도)를 야기한다는 가설 아래, **w_t = 1/v_t²** 의 inverse-velocity 가중치를 per-timestep loss에 곱해주는 **architecture-agnostic plug-in**. OpenVLA-OFT + AttenA+ → LIBERO 98.6 (+1.5), Pi0.5 + AttenA+ → 97.95 (+1.1), Fast-WAM + AttenA+ → RoboTwin 2.0 92.46 (+0.6). 파라미터 0개 추가, inference cost 0, 코드 변경 최소.

---

## 1. 배경 및 동기

### Action Inequality 가설
- VLA 학습에서 일반적으로 timestep마다 동등한 weight로 L1/L2/flow-matching loss를 적용함.
- 그러나 manipulation에서 **저속(low-velocity) 구간**(예: 접촉 전 미세 정렬, gripper close, precision insertion)은 task success를 결정.
- **고속(high-velocity) 구간**(reach, transit)은 error tolerance가 크고 task success에 marginal.
- 결과적으로 dataset의 시간 비율로 보면 transit가 우세 → loss가 transit-dominated → 모델이 precision moment를 underfit.

### 핵심 질문
- **각 timestep loss를 그 시점의 velocity에 반비례하게 가중하면, 핵심 순간의 supervision이 커져 성능이 오르는가?**
- **이를 architecture-agnostic으로 plug-in 가능하게 만들 수 있는가?**

📌 [Figure 1 삽입] — 같은 trajectory에서 low-vel vs high-vel 구간 시각화 및 success contribution 차이

---

## 2. 방법론 심층 분석

### 2.1 디자인 철학: "Zero-Cost Plug-in"

AttenA+가 의도적으로 만족하는 제약:
- 추가 파라미터 0
- 추가 inference latency 0
- 모델 architecture 변경 없음
- 적용 코드 변경 ≈ training loop에서 loss 계산 한 줄 수정

→ 즉 "supervision balance"라는 같은 문제를 푸는 FrameSkip이 **frame selection** 으로 접근한 반면, AttenA+는 **loss reweighting**으로 접근.

### 2.2 Velocity 정의와 Weight

- **Velocity magnitude**: $v_{b,t} = |a_{t+1} - a_t|_2$ (action vector L2 norm difference)
- **기본 weight**: $w_t = 1/v_t$
- **Main configuration**: $w_t = 1/v_t^2$ — 제곱이 더 강한 boosting을 줌
- **Clipping**: clipmax ∈ {2.0, 3.0, 5.0} — 너무 큰 weight 폭주 방지

### 2.3 모델별 적용

#### Discriminative (regression) 모델
$$\mathcal{L} = \sum_t w_t \cdot |a^{pred}_t - a^{gt}_t|$$
- OpenVLA-OFT, Pi0.5 등

#### Generative (flow matching / diffusion) 모델
- 각 denoising step에서 per-timestep loss에 동일 $w_t$ 적용
- Diffusion Policy formulation은 Appendix B에 수식 명시

#### World-Action Model
- Fast-WAM 등 world model에 적용 시에도 same scheme

> ❓ **예상 질문**: Velocity가 0 가까운 dwell frame은 weight가 polynomial하게 폭주하는데?
> **답변**: clipmax로 hard cap. 즉 $w_t = \min(1/v_t^2, \text{clipmax})$. clipmax=2~5 sweep을 ablation에서 보고.

> ❓ **예상 질문**: Inverse가 아니라 inverse-square인 이유?
> **답변**: $1/v$ 는 dwell vs transit ratio가 작음. $1/v^2$는 더 aggressive하게 dwell을 boost. Ablation에서 inverse_squared가 inverse / log / exp_decay 중 평균 best.

### 2.4 Architecture 요약

| 항목 | 값 |
|------|----|
| 추가 파라미터 | 0 |
| 추가 inference cost | 0 |
| Backbone | Plug-in: OpenVLA-OFT, Pi0, Pi0.5, Fast-WAM 등 |
| Loss reweight | $w_t = 1/v_t^2$ with clipmax |
| Compatible action heads | regression, FM, diffusion |

---

## 3. 학습 인프라

| 변형 | Hardware | Wall-clock |
|------|----------|-----------|
| AttenA+OFT | 1 × H800, 200K steps (ckpt every 5K) | ~35 hours |
| AttenA+WAM | 2 × H800, 1 epoch RoboTwin 2.0 (ckpt every 2K) | ~4 days |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO with OpenVLA-OFT (Table 1)

| Suite | OpenVLA-OFT | **AttenA+OFT** | Δ |
|-------|------------|---------------|----|
| Spatial | 97.6 | **99.0** | +1.4 |
| Object | 98.4 | **100.0** | +1.6 |
| Goal | 97.9 | **98.8** | +0.9 |
| Long | 94.5 | **96.6** | +2.1 |
| **Avg** | **97.10** | **98.60** | **+1.50** |

- **Object suite 100%** — clean ceiling.
- **Long suite +2.1** — 가장 큰 향상. Long-horizon task일수록 trajectory에 dwell/transit imbalance가 누적되어 reweight 효과가 큼.

### 4.2 LIBERO with Pi0.5 (Table 3)

| 모델 | LIBERO Avg |
|------|-----------|
| Pi0.5 | 96.85 |
| **AttenA+Pi0.5** | **97.95** (+1.10) |

- Generative(flow-matching) 모델에도 적용 가능함을 입증.

### 4.3 RoboTwin 2.0 with Fast-WAM (Table 5)

| 설정 | Fast-WAM | **AttenA+WAM** |
|------|---------|---------------|
| Clean | - | 93.06 |
| Randomized | - | 91.86 |
| **Avg (50 bimanual tasks)** | **91.8 (prior SOTA)** | **92.46** (+0.6) |

- 50개 bimanual task 평균에서 새로운 SOTA.
- Task별 success rate 65~100% 범위 (논문 본 표).

### 4.4 Real-World Franka (Table에 50-trial 결과)

| Task | Trials | AttenA+OFT | Baseline 대비 Δ |
|------|--------|-----------|----------------|
| Close the open drawer | 50/50 | 100% | 0% |
| Put Green Cube into Green Bowl | 50/50 | 100% | +4% |
| Put Object-A into Green Bowl | 49/50 | 98% | +8% |
| Put A then B (long-horizon) | 45/50 | 90% | +6% |

- **Multi-object와 Long-horizon에서 가장 큰 향상** — sim 결과와 일관: 복잡할수록 reweight 효과 큼.

### 4.5 Ablation (Table 4)

| Weighting | LIBERO 평균 경향 |
|-----------|-----------------|
| inverse ($1/v$) | 양호 |
| **inverse_squared ($1/v^2$)** | **최우수** |
| exp_decay | 중간 |
| log | 약함 |

- Clipmax 1.0 → baseline에 수렴 (효과 사라짐).
- Clipmax 2.0~5.0 sweep에서 task별 best가 다름 — "universal best"는 없음.

---

## 5. 관련 연구 비교

| 라인 | 대표 연구 | 접근 | AttenA+와 차이 |
|------|----------|------|--------------|
| Frame selection | FrameSkip | dataloader에서 informative frame만 선택 | AttenA+는 모든 frame 사용, weight만 변경 |
| Curriculum | task-level curriculum | task 난이도 순서 학습 | AttenA+는 timestep-level weighting |
| Focal loss류 | objection detection | 어려운 example boost | AttenA+는 velocity-driven, 도메인 적응 |
| Action chunking | RDT, Pi0 | chunk 단위 prediction | architecture 변경, AttenA+는 loss만 |

핵심 차별점: **"Velocity inverse"라는 simple하고 임베디드 가능한 reweighting** — explainable, zero-cost, 다른 plug-in과 결합 가능.

> ❓ **예상 질문**: FrameSkip과 AttenA+가 함께 적용되면?
> **답변**: 둘 다 supervision balance를 다른 축으로 해결 — frame selection (FrameSkip) + per-frame weight (AttenA+) 결합 자연스러움. 본문에 해당 실험 없음. 향후 연구 자연 연장.

---

## 6. 한계 및 미해결 문제

### 방법론적
1. **Velocity 정의가 단순**: action vector L2 norm 차이. End-effector의 6-DoF에서 rotation과 translation을 같은 norm으로 합치는 것이 부정확할 수 있음 (rotation 1 rad ≠ translation 1 m).
2. **Clipmax tuning이 task별로 다르다**: "universal best 없음" 인정. 즉 deployment마다 sweep 필요.
3. **Dwell frame (v ≈ 0)에서 weight 폭주 위험**: clipmax로 막지만, 본질적으로 1/v 라는 함수가 singular함을 hack로 우회.

### 평가
1. **Real-world task 4개**: 다양성 부족. 모두 Franka, 모두 bin pick-and-place 계열.
2. **Diffusion Policy 결과 누락**: Appendix B에 formulation만 있고 실험 결과 부재.
3. **각 task별 baseline 평균 standard deviation 부재**: ±0.16, ±0.00 등 LIBERO subset에 stddev 보고가 부분적.

### 해석
- **+0.6 ~ +2.1%p의 게인이 "transformative"한가**: 이미 OpenVLA-OFT 97.1, Pi0.5 96.85 등 ceiling 가까운 baseline에서 +1%p 이상 일관된 향상은 의미 있음. 그러나 Fast-WAM의 +0.6은 noise 가능.
- **"Action inequality" framing의 일반성**: 본 논문은 manipulation에 집중. Locomotion, navigation 등 다른 robotic domain에서 동일한 inequality가 있는지 일반화는 미입증.

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "Action inequality"라는 framing이 깔끔. Loss reweighting 자체는 새롭지 않으나 robotic domain에 velocity 기반으로 instantiate한 것이 신선. |
| **Technical depth** | ★★★☆☆ — 단순 수식 ($w_t=1/v_t^2$). 깊은 이론 분석은 없으나 단순함이 미덕. |
| **Experimental rigor** | ★★★★☆ — OpenVLA-OFT, Pi0.5, Fast-WAM 세 backbone에 일관된 게인. Real-world Franka 50 trial × 4 task. |
| **Practical impact** | ★★★★★ — 0 파라미터, 0 inference cost, 한 줄 변경. Adopt cost 최저. |
| **Writing/Clarity** | ★★★★☆ — Action inequality 동기 부여가 직관적. |

**강점**: 가장 cheap한 형태의 VLA 성능 향상 — plug-in으로 어디든 적용 가능. 세 가지 다른 backbone family (discriminative regression, FM generative, world-action model) 모두에서 동작 검증. **약점**: Velocity 정의의 단순함, clipmax tuning 필요, dwell frame 처리.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 동일 effect를 단순히 "low-velocity frame oversampling"으로 얻을 수 있지 않은가? | 부분적으로 가능. AttenA+는 reweighting이라 dataloader 변경 없음 — 더 가볍고 안정적. FrameSkip이 sampling 접근. |
| 2 | Velocity = action delta L2가 rotation과 translation을 같은 단위로 보는 문제는? | 정확한 지적. 본문은 단순화 채택. 6-DoF별 component-wise weight나 SE(3) metric 도입은 자연스러운 후속. |
| 3 | clipmax = 1.0이면 baseline에 수렴한다고 했는데 왜? | $w=1$이 되어 reweighting 효과 사라짐. Sweep 의미: clipmax > 1에서만 효과 발현. |
| 4 | Pi0.5 + AttenA+ (+1.1) 가 OpenVLA-OFT + AttenA+ (+1.5) 보다 작은 이유? | Pi0.5 baseline이 더 낮고 noisy. Or Pi0.5의 flow matching loss가 이미 velocity 정보를 implicit하게 활용. |
| 5 | Diffusion Policy 결과는 왜 빠졌나? | Appendix B에 수식만. 실험 결과는 본문 fetch 결과에서 확인 안됨 — Reviewer 지적 가능 지점. |
| 6 | RoboTwin 2.0 clean 93.06 vs random 91.86 — randomization 강도? | 객체 위치, 색, lighting random 일반적. Δ 1.2 차이는 robustness가 어느 정도 보존됨을 의미. |
| 7 | LIBERO Object 100%에서 +1.6 의미? | 시드 평균 100% (50 trial 50 success). Variance 매우 낮음. ceiling 도달. |
| 8 | FrameSkip과 결합하면 시너지인가, redundancy인가? | 가설: shared mechanism (supervision balance)이므로 marginal returns. 그러나 frame과 weight는 다른 축이라 일부 시너지 가능. 본문 미실험. |

<!-- VERIFIED: pdf -->
