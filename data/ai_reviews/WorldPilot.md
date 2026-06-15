# World Pilot: Steering Vision-Language-Action Models with World-Action Priors

> **한 줄 요약**: 사전학습된 World-Action Model(Cosmos Policy)을 동결한 채 두 개의 상보적 경로(Latent Steering: scene-evolution latent를 cross-attention residual로 VLM hidden state에 주입 / Action Steering: 예측 궤적을 단일 prefix 토큰으로 압축해 flow-matching action generator에 조건화)로 VLA에 dynamics prior를 주입하여, LIBERO-Plus zero-shot OOD에서 84.7% Total (ABot-M0 대비 +4.2)을 달성하고 4개 실로봇 태스크 12개 ID/OOD 셀 전부에서 1위를 차지한 VLA steering 프레임워크.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 한계
- 표준 VLA([1, 2, 3, 4])는 VLM(image-text 사전학습)의 hidden state에서 action을 생성한다. 그러나 image-text pair는 정적이므로 **"scene이 action에 따라 어떻게 진화하는가"** 라는 dynamics 표현이 본질적으로 결여되어 있다.
- 결과: 시점(camera), geometry, contact tolerance가 학습 분포에서 벗어나면 정책이 부서진다([16-19]).

### World-Action Model(WAM)의 보완 가능성
- Cosmos Policy[23], mimic-video[24], DreamZero[25] 등 video-pretrained WAM은 action-conditioned scene evolution을 학습하며 embodiment·visual condition 간 전이가 광범위하다.
- WAM의 출력은 VLA가 빠진 것을 정확히 채운다: (i) scene-evolution latent Z_w (어떻게 보이는 상태가 바뀔지), (ii) coarse action trajectory hypothesis Ã_w (그 변화를 일으키는 action sketch). 둘은 공유 encoder에서 나오므로 structurally aligned.

### 핵심 질문
단순히 두 모델을 나란히 놓는 것으로는 부족하다. **어떤 signal을, 어떤 형태로, VLA의 어느 layer에 주입할 것인가?** 이것이 정책 능력 향상으로 이어지는지 결정한다.

---

## 2. 방법론 심층 분석

### 2.1 문제 정의 (Sec. 3.1)
시점 t에서 정책은 관측 O_t, 언어 ℓ, optional proprioception q_t를 받아 action chunk A_t = (a_t, …, a_{t+K-1})을 예측. WAM은 같은 입력에서 (Z_w_t, Ã_w_t)를 반환하고 World Pilot은:

```
(Z_w_t, Ã_w_t) = W_phi(O_t, ℓ, q_t)
Â_θ,t = π_θ(O_t, ℓ, q_t ; Z_w_t, Ã_w_t)
```

### 2.2 Latent Steering (Sec. 3.2)
- WAM은 VAE로 O_t를 encode 후 DiT로 denoise하여 per-view 미래 latent Z_w_t를 산출.
- Dynamics encoder f_dyn으로 projection 후 **future-scene 토큰임을 명시하는 temporal embedding rho_fut**를 더한다: D_w_t = f_dyn(Z_w_t) + rho_fut. (논문: rho_fut가 없으면 prior 기여가 경험적으로 감소.)
- VLM hidden state H_t에서 D_w_t로의 cross-attention의 출력을 **residual**로 더한다:

```
H̄_t = H_t + CrossAttn(H_t, D_w_t)
```

- Residual 형식: token 순서·hidden-state 구조 보존 → 기존 VLA action 생성 경로에 그대로 결합. Cross-attention: 각 VLM 토큰이 자신의 spatial region에 해당하는 D_w_t 부분에 **선택적으로** 주목 (전역 modulation 대비 장점).
- **왜 latent인가**: decoded future image는 texture·lighting·background·generation artifact 같은 action-irrelevant 정보를 동반해 dynamics 구조를 희석한다.

### 2.3 Action Steering (Sec. 3.3)
- WAM이 만든 Ã_w_t는 horizon·action dim이 task별로 다르므로 VLA horizon K로 resample.
- Action encoder f_act로 **단 하나의 prefix 토큰**으로 압축:

```
s_w_t = f_act(Align_K(Ã_w_t))
```

- Flow-matching generator 입력을 [u_t ; s_w_t ; Q_t ; X_τ,t]로 확장. u_t는 state 토큰, Q_t는 learned future-query 토큰, X_τ,t는 flow time τ의 noisy trajectory. 조건은 dynamics-enhanced H̄_t (cross-attention condition).
- s_w_t는 **prefix**이지 noisy trajectory의 일부가 아니므로 denoising 자체는 거치지 않고 self-attention으로 generator의 denoising recurrence를 조건화.
- **왜 single token인가**: per-step token은 매 step을 WAM의 noisy step에 묶어 error를 누적시킴. Single token은 trajectory의 overall shape만 hint로 주고 generator가 dynamics-enhanced hidden state와 함께 자기 chunk를 선택할 자유를 보존.

### 2.4 두 경로의 상호 독립성
- Latent Steering은 H_t에 residual을 더해 token sequence를 보존.
- Action Steering은 generator에 prefix 1개만 삽입, denoising recurrence를 바꾸지 않음.
- 따라서 둘은 additive이고 독립적으로 ablate 가능 (Sec. 4가 실증).

### 2.5 정책 학습 (Sec. 3.4)
- WAM W_phi는 **전 학습 과정 동결**. Gradient는 VLM backbone, f_dyn + Latent Steering cross-attn, f_act, flow-matching generator에만 흐른다. → WAM 사전 prior가 VLA fine-tuning으로 오염되지 않음. WAM forward는 사전계산·캐싱 가능 (inner training loop에서 제외).
- ABot-M0[6] 따라 **clean-action parameterization** flow-matching 채택. Noisy trajectory X_τ,t = τ·A*_t + (1-τ)·ε.
- Objective:

```
L_WorldPilot = E_{τ,ε} [ w(τ) || Â_θ,t - A*_t ||_2^2 ],   w(τ) = 1/(1-τ)^2
```

w(τ)는 velocity-space loss의 reweighting. WAM prior는 오직 conditioning path를 통해서만 들어가므로 별도 prior loss가 필요 없다.

---

## 3. 데이터 전략

| 항목 | 내용 |
|---|---|
| **시뮬레이션 학습** | LIBERO[68] (LIBERO-Plus 평가는 zero-shot OOD), RoboCasa[43] |
| **실로봇** | Stack Blocks / Fold Towel / Fruit-to-Plate / Container-Lid Alignment 4 task |
| **실로봇 데모** | Task당 100 ID teleoperated demonstration |
| **OOD 평가** | LIBERO-Plus 10,030 perturbed task (7축: background, camera, language, light, layout, robot, noise); 실로봇은 task당 2개 OOD variant |
| **WAM prior** | Cosmos Policy[23] (5-step denoising) 기본; Latent Steering ablation에서 Cosmos-Predict[71] (scene-prediction-only) 와도 비교 |

---

## 4. 시스템/학습 세부사항

- **VLM**: Qwen3-VL[67]
- **Action head**: DiT-based flow-matching (ABot-M0 스타일, clean-action parameterization)
- **WAM**: Cosmos Policy, 5-step denoising, **동결**
- **Dropout**: WAM 조건 D_w_t와 s_w_t에 dropout rate 0.3 → 정책이 prior에 과의존하지 않도록
- **하드웨어**: 8× RTX PRO 6000
- **실로봇 fine-tuning**: 모든 method 10,000 step, optimizer/batch/LR schedule 매칭, 20 trial per task setting
- **추론**: VLA와 WAM 둘 다 online으로 동작, fusion path는 train/infer에서 동일한 모양의 prior를 소비 → 학습된 fusion이 직접 transfer

---

## 5. 실험 결과 (PDF Table 1, 2, 3, 4, 5, 6 직접 확인)

### 5.1 시뮬레이션 (Table 1)
| Method | LIBERO | LIBERO-Plus Total | RoboCasa |
|---|---|---|---|
| OpenVLA[69] | 84.7 | 15.6 | – |
| π0[12] | 94.4 | 53.6 | 42.4 |
| π0.5[5] | 96.9 | 77.4 | 41.4 |
| Being-H0.7[66] | 99.2 | 82.1 | 62.1 |
| Cosmos Policy[23] | 98.5 | 79.7 | 67.1 |
| ABot-M0[6] | 98.6 | 80.5 | 54.0 |
| **World Pilot** | **98.5** | **84.7** | **65.5** |

LIBERO-Plus per-axis (Camera/Robot/Language/Light/Background/Noise/Layout): **82.8 / 60.6 / 87.2 / 98.6 / 96.4 / 93.6 / 80.5**. Camera에서 +13.2(가장 큰 단일 축 gain), Background/Noise/Light(외형 축) 모두 1위. Language·Robot·Layout은 strongest baseline에 근접.

### 5.2 실로봇 (Table 2, 20 trial/setting)
| Task | Setting | π0.5 | ABot-M0 | Cosmos Policy | **World Pilot** |
|---|---|---|---|---|---|
| Stack Blocks | ID / Color / Height | 40 / 15 / 0 | 60 / 25 / 10 | 65 / 30 / 15 | **70 / 55 / 50** |
| Fold Towel | ID / Direction / Novel | 55 / 25 / 10 | 50 / 20 / 5 | 45 / 15 / 10 | **85 / 75 / 70** |
| Fruit-to-Plate | ID / Novel Fruit / Layout | 35 / 10 / 5 | 65 / 30 / 15 | 70 / 35 / 20 | **90 / 75 / 70** |
| Container-Lid | ID / Novel Obj / Lid Pose | 40 / 15 / 5 | 65 / 30 / 15 | 60 / 25 / 10 | **80 / 70 / 65** |

12셀 전부 1위. **ID→OOD drop이 ≤20pt** (다른 baseline은 25–50pt). Container-Lid OOD에서 World Pilot은 13–14/20 성공, baseline 최대 6/20.

---

## 6. Ablation 분석

### 6.1 각 경로의 독립 기여 (Table 3)
- ABot-M0 baseline: 80.5%
- Latent Steering only: 83.7% (+3.2)
- Action Steering only: 83.1% (+2.6)
- Full World Pilot: 84.7% (+4.2)
→ 두 경로가 **상보적**으로 기여.

### 6.2 World prior가 action post-training 없이도 작동하는가? (Table 4)
WAM을 Cosmos-Predict[71] (scene-prediction-only, action post-training 안 됨)로 교체, Latent Steering only:
| Benchmark | ABot-M0 | + LS (Cosmos-Predict) |
|---|---|---|
| LIBERO-Plus | 80.5 | **82.6 (+2.1)** |
| RoboCasa | 54.0 | **62.7 (+8.7)** |
| RoboTwin2.0 (clean) | 81.2 | **85.3 (+4.1)** |

→ 단순 video-pretrained world model의 latent도 이득. Cosmos-Predict→Cosmos Policy로 action post-training 추가 시 LIBERO-Plus +1.1 추가(83.7%).

### 6.3 Latent 형태 (Table 5, LIBERO-Plus Total)
| Future Information | Success (%) |
|---|---|
| Future latent (1 step) | 84.6 |
| Future latent (3 steps) | 84.5 |
| Future latent (5 steps) | **84.7** |
| Decoded future image | 83.5 |

Denoising depth 1/3/5에 거의 무관(편차 ≤0.2). Decoded image로 바꾸면 −1.2pt → pixel-level realism은 도움이 안 되고 dynamics 구조를 희석.

### 6.4 Action prior 형태 (Table 6)
| Action Prior Form | Success (%) |
|---|---|
| **Single encoded token (Ours)** | **84.7** |
| Per-step encoded tokens | 83.6 |
| Flow init. from Ã_w_t | 84.1 |
| Raw Ã_w_t | 83.0 |

Per-step은 step-level noise 전파, raw는 WAM trajectory에 그대로 묶임, flow init은 generator의 prior 보정 여지를 빼앗음. Single token이 최선.

---

## 7. Related Work 비교

| 접근 | 대표 | 한계 vs World Pilot |
|---|---|---|
| **Future image + action 공동 생성** | Motus[58], DreamVLA[59] | Visual reconstruction loss가 action representation을 외형 detail에 흡수시킴 |
| **Predicted future / subgoal image로 정책 가이드** | π0.7[60], VISTA[61] | Pixel-space 출력이 texture·lighting·artifact 등 control과 무관한 정보를 포함 → control-relevant 구조 희석 |
| **Latent / implicit feature로 world model 지식 전달** | Being-H0.7[66], WoG[41] | Static future snapshot에 의존, 연속적 spatiotemporal evolution을 못 다룸 |
| **World Pilot** | 본 논문 | Scene-evolution **latent** (pixel 아님) + trajectory-level **single-token** action prior를 **frozen WAM**에서 추출해 두 layer에 별도 주입 |

---

## 8. Limitations (논문 Sec. 5 자체 언급 + 분석)

1. **WAM coverage 의존**: 테스트 scene이 WAM video pretraining 분포 밖이면 두 prior 모두 약화 → World Pilot의 gain 감소.
2. **개선이 균일하지 않다**: LIBERO-Plus의 Language/Robot/Layout 축에서는 strongest baseline에 뒤짐. 실로봇 OOD에서도 ID 대비 10–20pt drop은 남는다 (prior가 OOD shift를 줄일 뿐 제거 못 함).
3. **Modular coupling의 trade-off**: WAM과 VLA는 action loss로만 연결 → 컴포넌트 교체는 용이하지만 **joint co-adaptation**이 줄 수 있는 추가 이득은 못 얻음.
4. **추론 비용**: 매 결정 step에 WAM forward 1회 추가 → **고주파 reactive control에 부적합**. 저자들이 prior distillation, adaptive querying을 향후 과제로 명시.

---

## 9. 종합 평가

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★★☆ — frozen WAM + 두 layer에 각자 다른 형태(latent residual + single prefix token)의 prior를 동시 주입한 점이 깨끗하고 ablation으로 강하게 정당화됨 |
| **Technical depth** | ★★★★☆ — clean-action flow-matching, residual cross-attention, future temporal embedding rho_fut 같은 디테일이 모두 ablation으로 검증 |
| **Empirical rigor** | ★★★★★ — LIBERO-Plus 7축 + RoboCasa + 4 실로봇 task × 3 setting × 20 trial. WAM 종류 교체, denoising depth 변화, action prior 형태 4종 비교까지 포함 |
| **Practical impact** | ★★★★☆ — frozen WAM이므로 더 강한 video world model이 나오면 그대로 plug-in 가능. 단, per-step WAM forward 비용은 제약 |
| **Reproducibility** | ★★★☆☆ — 코드/체크포인트 공개 명시 없음 (project page만). VLM, action head, WAM 모두 외부 의존(Qwen3-VL, ABot-M0 head, Cosmos Policy). |

### 핵심 기여
1. VLA에 WAM prior를 주입하는 **"무엇을, 어디에"** 라는 질문을 명확히 던지고 ablation 행렬로 답한다.
2. **두 경로(perception layer / action generator)** 가 다른 정보를 다른 형태로 carry해야 한다는 설계 원칙(residual latent vs. single prefix token).
3. **Frozen WAM**과 action-post-training이 없는 단순 scene-prediction WAM(Cosmos-Predict)으로도 이득이 발생함을 증명 → world prior가 정책 미세조정에 의존하지 않는다는 강한 주장.

---

## 10. 예상 세미나 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | 왜 latent를 쓰고 decoded image를 안 쓰나? 결국 같은 정보 아닌가? | Table 5에서 −1.2pt 차이로 실증. Pixel 공간은 texture/lighting/artifact를 동반 → action selection에 무관한 detail로 dynamics 구조를 희석. Latent는 contact·motion·state-change에 집약적. |
| 2 | Single prefix token이 trajectory 정보를 충분히 carry하는가? Per-step이 더 풍부할 것 같다. | Per-step은 WAM 궤적의 step-level noise를 매 출력 step에 직접 묶어 error를 누적. Single token은 overall shape만 hint로 주고 generator가 dynamics-enhanced H̄_t와 함께 자기 chunk를 commit. Table 6: +1.1pt. |
| 3 | WAM을 동결하는 게 정말 최선인가? Joint training이 더 좋지 않을까? | 저자도 limitation으로 인정 (modular vs joint trade-off). 그러나 동결의 장점: ① VLA fine-tuning이 WAM의 broad pretrained prior를 망가뜨리지 않음, ② WAM forward 캐싱 가능, ③ 더 좋은 WAM이 나오면 swap만으로 업그레이드. |
| 4 | LIBERO-Plus Robot 축에서 ABot-M0(67.9)보다 낮은(60.6) 이유는? | WAM은 video로 학습되어 시각/동역학에는 강하지만, robot embodiment 자체의 변화에는 약하다. WAM 자체의 pretraining 분포에 robot variation이 적기 때문. |
| 5 | Cosmos-Predict만으로도 이득이 있다면 Cosmos Policy의 추가 가치는? | Table 3 vs Table 4 비교: LIBERO-Plus에서 LS-only 기준 82.6 → 83.7 (+1.1). Action-post-training이 trajectory 신호를 sharpen해 marginal gain은 있지만, video pretraining 자체가 이미 대부분의 prior를 제공. |
| 6 | rho_fut(future temporal embedding) 없으면 왜 약해지나? | Latent을 "지금이 아니라 미래"로 명시적으로 표지해야 cross-attention이 미래 token으로서 다른 분포로 attend하게 됨. 표지 없으면 현재 token과 섞여 dynamics signal이 흐려진다. 정량은 본문에 "empirically diminishes"로만 명시. |
| 7 | Real-world에서 ID→OOD drop이 10–20pt 남는데, prior가 정말 OOD에 도움이 되나? | Baseline은 25–50pt drop. 같은 ID 성능에서 출발해도 OOD drop이 절반 이하 → prior가 robustness gap을 줄이는 건 분명. 단, 완전 해결은 아님. |
| 8 | Inference latency는 얼마나 늘어나나? | 매 decision step에 WAM(5-step DiT denoising) 추가. 저자는 정량을 안 주지만 limitation으로 명시. 고주파 reactive control(>50 Hz)에는 적합하지 않을 가능성. |

---

## 11. 코드 & 재현

- **Project page**: https://world-pilot.github.io/
- **arXiv**: https://arxiv.org/abs/2606.12403 (v1, 2026-06-10)
- **External 의존성**:
  - VLM backbone: Qwen3-VL ([67])
  - Action head: ABot-M0 flow-matching head ([6])
  - WAM: Cosmos Policy ([23]) / Cosmos-Predict ([71])
- **재현 난이도**: 코드/가중치 공개 명시 없음 → 직접 구현 시 (i) Qwen3-VL hidden state로의 cross-attention residual, (ii) Cosmos Policy의 scene-evolution latent 추출 인터페이스, (iii) flow-matching prefix conditioning이 핵심 구현 포인트.
- **하드웨어**: 8× RTX PRO 6000 (논문 명시)

---

## 12. 결론

World Pilot은 "VLA에 world prior를 주입한다"는 큰 흐름 위에서, **prior의 form과 entry point가 자유롭게 교환 가능하지 않다**는 명제를 실증적으로 확립한 작업이다. 핵심은:

1. **Latent ≠ decoded image** (Latent Steering은 residual cross-attention으로 perception layer에 들어가야 함).
2. **Single trajectory token ≠ per-step / flow-init / raw** (Action Steering은 generator의 prefix로 한 토큰만 들어가야 함).
3. **WAM은 동결되어야** prior가 VLA fine-tuning에 흡수되지 않고 그 dynamics 표현이 보존된다.

이 세 원칙이 ablation matrix 전체에서 일관되게 winning configuration이라는 점이 논문의 가장 강한 기여이며, LIBERO-Plus 84.7% Total과 실로봇 12/12 셀 1위라는 결과는 그 원칙이 simulation·real-world 모두에서 transfer됨을 보여준다. 향후 더 강한 video WAM(e.g., 더 큰 Cosmos 계열, 4D world model)이 등장하면 frozen-swap 형태로 그대로 업그레이드될 수 있다는 점에서 **VLA + world model 결합 패러다임의 modular한 표준 레시피** 후보로 의미가 있다.

<!-- VERIFIED: pdf -->
