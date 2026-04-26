# DualCoT-VLA: Visual-Linguistic Chain of Thought via Parallel Reasoning for Vision-Language-Action Models

> **한 줄 요약**: 16개의 visual CoT + 4개의 linguistic CoT learnable query token을 VLM 입력에 삽입하고 각각을 Depth Anything 3(기하)와 Qwen3-0.6B(언어)로 distill하여, autoregressive CoT 없이 단일 forward pass로 dual-modal 추론을 수행. LIBERO 98.8%, RoboCasa GR1 55.1%, VLM latency 3156ms → 58.1ms.

---

## 1. 배경 및 동기

- 기존 CoT-VLA 계열은 단일 modality에 국한: linguistic CoT(Embodied CoT, ThinkAct)는 logical planning에 강하지만 spatial grounding 부족; visual CoT(CoT-VLA, FlowVLA)는 그 반대.
- Autoregressive CoT decoding은 N길이 token에 O(N)의 sequential forward가 필요 — closed-loop 로봇 제어에 치명적인 latency.
- Implicit latent reasoning(Coconut, Fast-ThinkAct)도 잠재 공간 내부에 AR 구조를 유지해 parallel화 불가.
- 저자들의 제안: **두 modality를 parallel하게 internalize**하되, frozen teacher model로 distill하여 collapse 방지.

---

## 2. 방법론

### Parallel Implicit CoT via Query Tokens (Sec. 3.2)
Input sequence: X_input = [V_obs, Q_vis, L_instr, Q_lin] with |Q_vis|=M=16, |Q_lin|=N=4 (Eq. 1). Single forward pass로 H_vis ∈ R^(16×d), H_lin ∈ R^(4×d) 추출.

### Visual CoT: Geometric Distillation from DA3 (Sec. 3.3)
Frozen Depth Anything 3 encoder로 dense spatial feature F_DA3 추출. 차원 불일치 해결을 위해 learnable spatial query Q_spatial가 H_vis에 cross-attend하여 고해상도 feature map 재구성(Eq. 2): F̂_DA3 = CrossAttention(Q_spatial, P(H_vis), P(H_vis)). MSE loss(Eq. 3): `L_vis = MSE(F̂_DA3, F_DA3)`. 16개 continuous latent에 spatial prior 압축.

### Linguistic CoT: Step-level Text Supervision (Sec. 3.4)
Frozen Qwen3-0.6B이 H_lin(linear projection 후)을 prefix로 받아 explicit CoT text Y_cot = (y_1, ..., y_L)를 생성. CE loss(Eq. 4): `L_lin = -Σ log p_φ(y_i | P_lin(H_lin), y_<i)`. CoT text는 "state tracking + spatial location + action formulation" 3-part 구조.

### Flow-Matching Action Head (Sec. 3.5)
DiT가 마지막 layer의 VLM hidden H_vlm에 cross-attend; noisy action a_t, t ~ U(0,1), robot state를 sequence input으로. Flow matching loss(Eq. 5): `L_act = E[||v_θ(a_t, t, H_vlm) - (A - a_0)||^2]`. 총 손실(Eq. 6): `L_total = λ_vis L_vis + λ_lin L_lin + λ_act L_act` with λ_vis=0.1, λ_lin=0.1, λ_act=1.0.

### Architecture
- Backbone: Qwen3-VL-4B
- Teachers (training만): Depth Anything 3 + Qwen3-0.6B (inference 시 폐기)
- CoT annotation: LIBERO는 Yin et al. 2025, RoboCasa는 Qwen3-VL-32B가 생성
- LIBERO: lr 2.5e-5, action window 7, batch 48. RoboCasa: lr 3e-5, window 15, batch 256. H100 GPU 학습.

---

## 3. 실험 결과

### LIBERO (Table 1, 500 episodes/suite)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.6 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| DeepThinkVLA | 96.6 | 99.0 | 96.4 | 96.2 | 97.0 |
| LaRA-VLA | 96.4 | 98.6 | **99.8** | 96.6 | 97.9 |
| **DualCoT-VLA** | **99.4** | **99.8** | 97.8 | **98.2** | **98.8** |

YAML 값(spatial 99.4 / object 99.8 / goal 97.8 / long 98.2 / avg 98.8)과 정확히 일치.

### RoboCasa GR1 (Table 2, 24 tasks × 50 rollouts)
Avg **55.1%** (GR00T-N1.5 48.2, GR00T-N1.6 47.6, Qwen3-GR00T 47.8). 특히 spatially demanding한 task에서 강점: CuttingboardToPan 80.0, PlacematToPlate 74.0, PlateToPlate 76.0. YAML robocasa_gr1_avg: 55.1와 일치.

### Inference Latency (Table 3, H100 GPU)

| Metric | Non-CoT | AR CoT | **Parallel CoT (DualCoT)** |
|---|---|---|---|
| VLM Forward (ms) | 53.7 | 3156.0 | **58.1** |
| Action Head (ms) | 22.5 | 27.5 | 25.1 |
| Total (ms) | 76.2 | 3178.5 | **83.2** |

AR 대비 ~38x 빠름, Non-CoT 대비 +6.9ms overhead.

### Ablation (Table 4, LIBERO)

| Visual | Linguistic | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| ✓ | ✗ | 99.4 | 99.6 | 97.4 | 95.0 | 97.9 |
| ✗ | ✓ | 98.4 | 98.4 | 96.6 | 96.0 | 97.4 |
| ✓ | ✓ | 99.4 | 99.8 | 97.8 | 98.2 | 98.8 |

Visual-only는 Spatial(+1.6)에 집중적 기여, Linguistic-only는 Long(+4.0)에 기여하여 **complementarity 입증**.

---

## 4. 한계 및 미해결 문제

1. **Teacher model 품질에 의존**: DA3가 geometric prior를, Qwen3-0.6B가 linguistic decoding을 담당하므로 이들의 capacity 한계가 CoT 품질을 제약. DA3보다 약한 geometric model로 교체 시 영향 미검증.
2. **Token 개수(M=16, N=4) 고정**: 더 복잡한 scene / 장기 task에서 16개 visual token이 충분한지 불명. Scaling 연구 부재.
3. **Linguistic CoT annotation의 수동성**: LIBERO는 외부 source, RoboCasa는 Qwen3-VL-32B 자동 생성 — annotation 품질이 다른 scheme일 때의 민감도 분석 없음.
4. **Real-world 평가가 AgileX 3 task만**: Easy/Medium/Hard로 구분되지만 25 trial씩으로 통계 제한. Fig. 4(b)에는 수치가 시각화되어 있으나 정량 표는 제공되지 않음.
5. **Qualitative latent probing**의 해석 가능성(Fig. 3): visual probe로 depth map을 복원할 수 있다지만, reconstruction이 어느 정도 구조적인지에 대한 quantitative metric이 부족.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Parallel + dual-modality + implicit CoT의 조합은 CoT-VLA, ThinkAct, Coconut 등의 장점을 엮어낸 clean 재설계 |
| **Practical impact** | ★★★★★ — AR CoT 대비 ~38x latency 개선 + LIBERO 98.8%의 SOTA. 실전 배포 가능한 "생각하는 VLA" 청사진 |

DualCoT-VLA의 핵심 기여는 "CoT를 생성하지 말고 **distill**하라"는 설계 철학이다. Visual CoT는 DA3의 dense feature로, linguistic CoT는 Qwen3-0.6B가 탐지할 수 있는 "decodable latent"로 각각 supervise함으로써 latent collapse를 피하면서도 parallel forward로 속도를 확보한다. Table 3의 3156ms → 58.1ms는 단순 속도 개선이 아니라 "CoT를 실전 로봇 제어 루프에 넣을 수 있는가"라는 질문에 대한 답이다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Linguistic CoT에서 frozen Qwen3-0.6B가 explicit text를 재생산하게 하는 목적은? 단순 latent distillation으로 충분하지 않은가? | 저자들은 Sec. 3.4에서 "latent representation collapse"를 명시적으로 피하기 위해 text-level supervision을 선택. 순수 latent alignment는 쉽게 degenerate해서 planning 정보가 사라질 수 있음. Sim-CoT(Wei et al. 2025) 참조. |
| 2 | Inference 시 teacher model이 필요 없다면 학습 시 추가 비용은 어느 정도인가? | DA3 + Qwen3-0.6B는 둘 다 frozen이므로 forward pass만 필요. Batch 단위로 cache 가능하며, total training은 H100 기반으로 수행되었고 action window 7(LIBERO)/15(RoboCasa)로 설정됨. |
| 3 | LIBERO-Long 92.0 → 98.2 개선에서 visual vs linguistic 중 무엇이 주된 기여인가? | Table 4: Linguistic-only 추가 시 92.0 → 96.0 (+4), Visual-only 추가 시 92.0 → 95.0 (+3), full은 98.2. 장기 planning은 linguistic CoT가 주도하지만 visual이 final +2.2를 더해 complementary. |

<!-- VERIFIED: pdf -->
