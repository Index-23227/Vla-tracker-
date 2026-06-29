# ROAD-VLA: Robust Online Adaptation via Self-Distillation for Vision-Language-Action Models

> **한 줄 요약**: 희소 보상의 scalar advantage를 action-token logit에 perturbation으로 주입해 "현재 정책 자신"으로부터 proximal teacher를 만들고, 이를 token-level KL로 self-distill 함으로써 OpenVLA-7B의 online 적응을 PPO보다 안정적·강건하게 수행하는 advantage-guided self-distillation 프레임워크.

---

## 1. 배경 및 동기

- VLA(OpenVLA, π0 등)는 대규모 사전학습으로 일반화하지만, 배포 시 새로운 외형·물체 배치·센서 노이즈·실행 오류 등 분포 변화에 취약하여 **online 적응**이 필수적이다.
- RL(PPO, DPO, GRPO)이 자연스러운 사후학습 틀이지만, 로봇 task의 보상은 sparse·delayed하여 policy-gradient가 high variance, 최적화 불안정, 사전학습 능력의 catastrophic forgetting을 겪는다.
- LLM에서 성공한 self-distillation(특권 teacher로 dense step-wise 신호 생성)을 VLA에 적용할 수 있는가? 가 핵심 질문.

---

## 2. 핵심 관찰: text-guided teacher의 실패

- 저자들이 가장 먼저 시도한 "demonstration / 검색 경험 / high-level plan 같은 **텍스트 기반** 특권 정보로 teacher를 조건화"하는 방식은 작동하지 않는다(Sec. 5.5).
- 이유: (i) embodied 사후학습 후 VLA는 LLM backbone의 in-context 추론 능력을 거의 상실, (ii) 텍스트 서술과 low-level 연속 제어 사이의 **modality gap**.
- 따라서 언어 기반 특권 맥락을 **action-centric, value 기반 신호**로 대체해야 한다는 결론.

---

## 3. 방법론: Advantage-Guided Self-Distillation

### Calibrated, agreement-gated advantage (Eq. 7-9)
- 현재 정책의 intrinsic advantage Â_int와 frozen PPO critic의 reference Â_ref를 batch 통계로 스케일 정합 후, **부호가 일치할 때만** 혼합(α=0.5 기본). gate g_t = 1[sign 일치].
- 혼합 advantage를 표준화·clip(c=2.0)하여 perturbation weight ω_t 산출. signed weight 유지(ReLU 불필요).

### Advantage-guided teacher (Eq. 10-11)
- teacher는 student logit에 η·ω_t 를 sampled token 방향으로 더한 **logit-perturbed copy**: q* = softmax(z + η·ω_t·e). η=1.0 고정.
- 이 perturbation은 KL-regularized local improvement 문제의 **closed-form 해**(exponential tilt). τ=1 → η=1/τ=1.
- OpenVLA action은 K=7 discrete token이므로 각 token 위치마다 독립적으로 적용, autoregressive하게 합성.

### Distillation objective (Eq. 12)
- token-level **forward KL** KL(q*‖p^θ)로 student를 teacher에 증류(JSD보다 mode-covering이라 안정적·강력). LAGD로 표기.

---

## 4. 이론적 결과 (Theorem 1)

- centered token reward를 student conditional 하에 중심화하면 teacher는 불변이고 student 기대값은 0.
- calibration 조건(teacher의 shaped reward가 true advantage와 정렬, β>0, ε_cal) 하에서 **policy-improvement lower bound** 유도:
  J(π_θ') ≥ J(π_θ) + (1/T)Σ E[βτ·KL(q*‖π_θ) − CB·√D_dist] − ε_cal (Pinsker, C=√2).
- 즉 teacher-policy KL이 개선항, student-teacher mismatch가 증류 비용으로 나타나며, autoregressive chain rule로 token-level 감독과 연결된다.

---

## 5. 실험 설정

- Base: **OpenVLA-7B**. 140 expert trajectory로 warm-up한 공유 checkpoint에서 ROAD-VLA와 PPO 모두 초기화.
- 동일한 rollout buffer, reward 구조, 최적화 hyperparameter로 통제 비교.
- 7개 manipulation 환경, 3개 분포 변화 축: Visual Robustness(VR-UnseenTable, VR-DynamicTexture, VR-DynamicNoise), Compositional Reasoning(CR-MultiObject, CR-MultiReceptacle), Execution Robustness(ER-InitPose, ER-Repositioning). 3 seed.

---

## 6. 주요 결과 (Table 1)

| 지표 (평균) | PPO | ROAD-VLA |
|---|---|---|
| ID 성공률 | 85% | **88%** |
| OOD 성공률 | 69% | **73%** |
| 열화 Δ (ID-OOD) | 16.3% | **14.6%** |

- 대부분 환경에서 ROAD-VLA가 PPO를 능가. 특히 VR-DynamicTexture(Δ 22→19), ER-Repositioning(Δ 16→11)에서 강건성↑.
- 주의: 일부 row(ER-Repositioning ID 89 vs 88, VR-DynamicNoise Δ 19 vs 20)에서는 PPO가 동급/근소 우위 — "거의 모든" 설정에서 우세이지 전부는 아님.

---

## 7. 적응 동역학 (Figure 2, Sec. 5.3)

- OOD baseline 27-31%에서 시작. ROAD-VLA가 mid-training 내내 PPO 선도, sample efficiency 우위.
- VR-DynamicNoise: peak 70.8%(PPO 64.1%), 종료 시 +4%. ER-Repositioning: step 159에서 peak 76.0%, +8점.
- 최종 30% 구간에서 분산이 더 낮음(VR-DynamicTexture 4% vs 6%) — agreement gate가 critic 불일치 gradient를 걸러 안정화.

---

## 8. 왜 작동하는가 (Sec. 5.4)

- **Policy entropy**: 수렴 시 ROAD-VLA가 더 높은 entropy 유지(3.24 vs 3.15 nats) → implicit diversity regularizer, premature policy collapse 방지 → OOD 강건성.
- **Advantage weight**: ω̄가 −0.033 → +0.037~+0.040으로 이동, 56% step에서 양수 → 고품질 전이에 선택적 강조.
- **Critic agreement**: 초기 ~90% → 수렴 71-75%, chance 대비 충분히 높음. gate가 합의 시에만 mixed advantage 적용.

---

## 9. Ablation (Table 2, VR-UnseenTable OOD)

| 방법 | 성공률 (%) |
|---|---|
| PPO | 87.2 ± 3.6 |
| ROAD-VLA (RelSpatial PI, 텍스트) | 4.68 ± 0.0 |
| ROAD-VLA (Plan+RelSpatial PI) | 4.68 ± 0.0 |
| ROAD-VLA (MCTS PI) | 75.8 ± 2.0 |
| ROAD-VLA (JSD loss) | 85.9 ± 1.5 |
| ROAD-VLA (w/o gate) | 89.8 ± 0.1 |
| **ROAD-VLA (full)** | **91.5 ± 1.2** |

- 텍스트 PI는 4.68%로 **붕괴** → advantage-guided 신호가 필수. MCTS PI도 75.8%로 PPO 이하.
- forward KL > JSD(85.9), gate 제거 시 −1.7점(distillation inversion 방지). full이 PPO 대비 +4.3점.
- α는 0.5가 stability-adaptability 최적(α=1.0 빠르나 stale, α=0.0 안정성 부족).

---

## 10. 한계 및 미해결 문제

1. **시뮬레이션·pick-and-place 한정**: 물리 로봇, long-horizon, 더 넓은 task 분포에서의 검증 부재.
2. **reference PPO critic 의존**: 큰 분포 변화 시 critic 품질 저하 가능 → critic-free / uncertainty-aware teacher가 향후 과제.
3. **벤치마크 미보고**: LIBERO/SimplerEnv 등 표준 공개 벤치마크 점수가 없고 자체 7개 환경의 상대 비교만 제시 → 절대 성능 위치 파악 어려움.
4. ROAD-VLA가 모든 환경에서 PPO를 이기지는 못함(일부 ID/Δ에서 동급/열세).

---

## 11. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — scalar advantage를 logit perturbation의 closed-form proximal teacher로 환원해 token-level dense 감독으로 바꾼 발상이 깔끔하고, 이론적 개선 보장까지 동반 |
| **Practical impact** | ★★★☆☆ — OpenVLA online 적응의 안정성·강건성을 실증하나, 시뮬레이션·자체 벤치 한정 및 reference critic 의존으로 실배포 일반화는 미검증 |

ROAD-VLA의 핵심 기여는 "특권 teacher를 외부에서 가져오지 말고 현재 정책 자신을 advantage로 살짝 밀어 만든다"는 통찰이다. text-guided PI의 명시적 실패(4.68%)는 사후학습된 VLA의 modality gap을 강하게 드러내며, action-space에서의 dense 감독이 sparse-reward RL의 본질적 약점을 보완함을 보여준다.

---

## 12. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | logit perturbation이 왜 "정당한" teacher인가? | KL-regularized local improvement 문제 max_q E[r] − τ·KL(q‖π_θ)의 해가 exponential tilt이고, one-point token reward r(u)=η·ω·1[u=â]에서 이것이 정확히 logit shift η·ω가 됨(Eq. 11, 19, 24). ad-hoc이 아니라 proximal 최적해. |
| 2 | 텍스트 기반 teacher는 왜 붕괴(4.68%)하나? | 사후학습이 LLM backbone의 in-context 추론을 약화시키고, 이산 텍스트 서술과 연속 제어 사이 modality gap 때문에 정밀 grounding 신호를 제공하지 못함. advantage 기반의 action-centric 신호만이 유효. |
| 3 | agreement gate가 하는 일은? | online critic과 frozen reference critic의 advantage 부호가 일치할 때만 mixed advantage를 적용. 불일치 시 online으로 복귀하여 stale·conflicting gradient("distillation inversion")를 차단. 제거 시 −1.7점, 후반 분산도 증가. |
| 4 | PPO 대비 본질적 차이는? | PPO는 advantage를 sampled action 한 개의 likelihood에 곱하는 scalar로만 사용. ROAD-VLA는 모든 action token·모든 on-policy step에 걸친 dense teacher 분포로 확장하면서 KL-proximal 구성으로 teacher를 정책 근처에 유지. |

<!-- VERIFIED: pdf -->
