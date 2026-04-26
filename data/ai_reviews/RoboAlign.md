# RoboAlign: Learning Test-Time Reasoning for Language-Action Alignment in Vision-Language-Action Models

> **한 줄 요약**: Qwen2.5-VL-7B-Instruct를 SFT로 FAST action token 생성 능력에 노출시킨 뒤 GRPO로 reasoning을 action-accuracy reward에 직접 맞추는 2-stage MLLM 학습으로, 이후 frozen MLLM 위에 diffusion head를 붙여 LIBERO Avg 86.8%, CALVIN ABC→D Avg.Len 2.57, real-world +106.6% 상대 이득을 달성한 KAIST/RLWRLD/Yonsei/UC Berkeley의 language-action alignment framework.

---

## 1. 배경 및 동기

- 최근 VLA는 MLLM을 backbone으로 쓰지만 embodied-reasoning을 강화한 variant(RoboBrain 2.0, Cosmos-Reason1 등)가 오히려 VLA 성능을 악화시키는 경우가 보고됨(Fig. 1). Qwen2.5-VL-7B-Ins baseline 73.9% 대비 RoboBrain 2.0 기반 variant는 57.35%로 **-16.5%p** 감소(Fig. 1).
- 원인은 MLLM-level reasoning supervision이 low-level action generation과 modality gap을 갖는 데 있다. VLM4VLA(Zhang et al., 2026)도 "embodied reasoning 능력과 VLA 성능의 상관관계는 일관되지 않다"고 관찰.
- 저자들은 reasoning의 quality를 **language output이 아닌 실제 action token accuracy**로 평가해야 한다고 주장. 이를 위해 MLLM이 low-level FAST action token(Pertsch et al., 2025)을 zero-shot chain-of-thought reasoning의 결과로 직접 생성하도록 학습시킨 뒤 RL로 reasoning을 action accuracy에 정렬시킨다.

---

## 2. 방법론

### Stage 1: Robotics-aware SFT (Sec. 4)
- Qwen2.5-VL-7B-Ins에 대해 **1.88M MLLM 학습 샘플 + 400K BridgeV2 FAST-token subset = 2.28M**으로 SFT.
- Prompt template(Fig. 5)에 따라 `<think>...</think>` 내부 zero-shot reasoning 후 최종 FAST token 시퀀스를 출력.
- Vision encoder는 frozen, cosine scheduler lr 2e-5, warmup ratio 0.03, 1 epoch.

### Stage 2: GRPO-based action alignment (Sec. 4, Eq. 1-2)
- 12.8K BridgeV2 FAST subset(<1% of SFT 데이터)에 대해 GRPO(Shao et al., 2024)로 RL.
- Reward = (format reward r_f + FAST-token accuracy reward r_a)/2. Accuracy reward는 prefix 일치 길이(Eq. 2): `r_a = (1/m) · max{i : T_{1:i} = T^*_{1:i}}`.
- EasyR1 repo 사용; rollout batch 512, update batch 128, 5 sample/prompt, lr 1e-6, 1 epoch.

### VLA training
- 학습된 MLLM을 **frozen backbone**으로 사용, 새로 초기화된 **diffusion-based action head**(Gr00t-N1.5 계열)를 부착해 각 benchmark 데이터로 처음부터 학습(LIBERO 60K, CALVIN 100K, real-robot 30K step).

---

## 3. 실험 결과

### LIBERO (Table 2, 500 trials/category)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Qwen2.5VL-7B-Ins baseline | 95.2 | 95.0 | 42.4 | 63.2 | 73.9 |
| w/ Language-Only SFT (1.88M) | 91.0 | 94.4 | 67.8 | 65.0 | 79.6 |
| w/ Action-Only SFT (1.88M) | 89.8 | 95.8 | 82.8 | 57.6 | 81.5 |
| w/ RoboAlign w/o RL (2.28M) | 92.8 | 97.4 | 59.0 | 65.6 | 78.7 |
| **w/ RoboAlign (2.28M+12.8K)** | **93.8** | **96.0** | **87.2** | **70.0** | **86.8** |

Baseline 73.9 → 86.8, **+17.5% 상대 이득**. Goal(42.4→87.2)과 Long(63.2→70.0)에서 특히 큰 gain.

### CALVIN ABC→D (Table 3, 1000 chains)

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg.Len |
|---|---|---|---|---|---|---|
| Qwen2.5VL-7B-Ins | 77.8 | 55.0 | 38.6 | 26.6 | 18.1 | 2.16 |
| Language-Only SFT | 87.4 | 62.2 | 41.9 | 25.2 | 15.3 | 2.32 |
| Action-Only SFT | 66.1 | 34.7 | 15.3 | 7.1 | 3.2 | 1.26 |
| RoboAlign w/o RL | 74.6 | 49.6 | 31.5 | 21.2 | 12.2 | 1.89 |
| **RoboAlign** | **87.6** | **67.2** | **47.1** | **32.8** | **22.2** | **2.57** |

길이 5까지 유일하게 단조 증가(length-5 18.1→22.2). 다른 baseline은 길이가 늘어날수록 하락.

### Real-world Franka Research 3 (Table 4, 96 trials/task)
4개 pick-and-place task 평균: Qwen2.5VL-7B 32.3% → RoboAlign 66.7% (**+106.6% 상대**). Basket→bowl 20.8→70.8이 가장 극적.

### Ablations

- **Alignment strategies (Table 6)**: Action-base RL(Ours) 86.8 vs Language-base RL 83.6, Visual-base RL 85.1 → Long category(70.0 vs 58.2/64.6)에서 action-level RL만 유일하게 큰 개선.
- **SFT-based alignment 비교 (Table 7)**: ECoT-style SFT alignment는 RoboAlign SFT(78.7) 대비 오히려 **67.7로 하락** → SFT로 reasoning을 주입하면 오히려 악화, RL이 필수.
- **Backbone 일반화 (Table 5)**: Qwen3-VL-8B-Ins에서도 85.2→92.5(+7.3%p), Long 60.0→78.6.
- **KNN representation probing (Table 8)**: LIBERO task 20개 trajectory에 대한 state classification 정확도가 39.06→69.79%로 상승 → MLLM 내부 표현이 fine-grained robot state를 더 잘 encode.

---

## 4. 한계 및 미해결 문제

1. **Closed-source 구현**: KAIST/RLWRLD collab이지만 code_url이 비공개. GRPO reward shaping과 VLA training script의 재현성은 공개되지 않음.
2. **Single-task VLA 학습**: 각 benchmark마다 diffusion head를 처음부터 학습하므로 multi-task/generalist 시나리오는 미검증.
3. **FAST tokenization 의존성**: Pertsch et al.의 FAST가 reward 신호의 단위이기 때문에 다른 action representation(e.g. continuous diffusion target)에 대한 직접 이식은 추가 설계 필요.
4. **12.8K RL 데이터의 조성**: BridgeV2 FAST subset만 사용해 domain이 편향. LIBERO/CALVIN의 out-of-domain task에 대한 reasoning 일반화는 간접 증거만 제시.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "Reasoning quality를 language가 아닌 action accuracy로 평가"라는 framing은 깔끔하고 후속 연구의 기준점이 될 만함 |
| **Practical impact** | ★★★★☆ — SFT 대비 <1% 데이터로 +17.5%/+18.9%/+106.6% 상대 gain이 simulation/real 모두에서 일관됨. 단 MLLM backbone 재학습이 필요해 training cost는 존재 |

핵심 메시지는 "VLA에 필요한 reasoning은 language-level VQA가 아니라 action-token-level RL로 획득해야 한다"이다. Table 6(Language-base RL vs Action-base RL)과 Table 7(SFT-based alignment의 성능 저하)이 이 주장을 뒷받침한다. 특히 RoboBrain 2.0 같은 embodied reasoning SOTA MLLM이 VLA에서는 최저 성능(Fig. 1)이라는 발견은 커뮤니티의 방향성에 재고를 촉구한다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | SFT만으로 action token을 학습시킨 후 왜 RL이 추가로 필요한가? | SFT(78.7)는 reasoning trace를 보여주지만 reasoning 자체가 최적은 아님. GRPO는 여러 reasoning trajectory를 sampling하여 action accuracy가 높은 reasoning을 강화. Table 2에서 w/o RL 78.7 → RoboAlign 86.8, 특히 Goal 59.0→87.2는 RL이 "어떤 reasoning이 정답 action으로 이어지는가"를 학습함을 보여줌. |
| 2 | FAST token accuracy reward가 diffusion action head의 continuous output에 어떻게 도움이 되는가? | MLLM은 frozen backbone으로만 기능하고 hidden state를 diffusion head가 consume. FAST-aligned MLLM의 hidden state가 fine-grained robot state를 더 잘 encode(Table 8: KNN 39→70%)하므로 downstream diffusion head가 같은 데이터로 더 정확한 action을 regress. |
| 3 | Language-Only SFT는 왜 Long에서 개선이 미미한가(65.0 vs RoboAlign 70.0)? | Language-level reasoning은 high-level task decomposition엔 도움이 되지만 low-level continuous action alignment는 약함. Table 6에서 Language-base RL도 Long 58.2로 오히려 하락 → 장기 manipulation은 action-level alignment가 필수. |

<!-- VERIFIED: pdf -->
