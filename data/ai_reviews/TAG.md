# TAG: Target-Agnostic Guidance for Stable Object-Centric Inference in Vision-Language-Action Models

> **한 줄 요약**: Pi0/Pi0.5 위에 "foreground-erased counterfactual observation"을 unconditional branch로 두고 **classifier-free guidance 스타일로 predicted velocity field를 residual extrapolation**하는 inference-time guidance. 학습은 counterfactual calibration(p_cf=0.1) 30K step만 필요하며 LIBERO Avg 97.9%(π0.5 baseline 95.2 대비 **+2.7%p**), LIBERO-Plus +4.7%p, VLABench +26.01%p의 개선을 학습 비용 없이 얻는 test-time intervention.

---

## 1. 배경 및 동기

- 기존 flow-matching VLA(π0, π0.5)는 near-saturation 성능이지만 **visually cluttered / long-horizon 시나리오에서 target-object attention이 distractor로 분산**되는 문제(Fig. 4, VLABench "pick poker 3 of diamonds" 예시)가 남아 있음.
- 저자들은 generative modeling의 classifier-free guidance(CFG; Ho & Salimans)에서 영감을 얻어, **"target이 있는 observation과 target이 erase된 counterfactual observation의 velocity 차이"**를 guidance signal로 사용하면 object-centric reasoning을 inference time에 강화할 수 있다고 주장.
- Key insight: foreground-erased image로 policy를 실행하면 "target 없이 environment만 볼 때 어떻게 움직일까"를 묻는 shortcut; 원본 대비 residual이 target-object 관련 action component가 된다.

---

## 2. 방법론

### Counterfactual 생성 pipeline (Sec. 3)
- **Grounding DINO** + **SAM 2** + **MiniMaxRemover** inpainting으로 target object의 mask를 만들고 삭제해 두 가지 variant 생성:
  - **TAG-bg (Background)**: target 영역을 배경으로 inpainting 복원
  - **TAG-black (Black)**: target 영역을 pure black으로 덮음
- Closed-loop quality audit: inpainted frame을 VLM이 재검사, artifact 발견 시 최대 3회 reroute. 실패한 sample은 학습 분포에서 영구 제외.

### Training-time Calibration (Sec. 3)
- Baseline policy(π0/π0.5) fine-tuning 시 **p_cf=0.1 확률로 원본 observation을 counterfactual erased frame으로 치환**.
- 30,000 step, cosine decay lr(peak 2.5e-5, final 2.5e-6, warm-up 1K), batch size 32(LIBERO)/24(VLABench), EMA decay 0.999, 단일 NVIDIA RTX PRO 6000.

### Inference-time CFG (Sec. 3)
- Flow-matching action head의 velocity field에 대해 **residual extrapolation**: `v_guided = v(x|obs) + γ·(v(x|obs) − v(x|obs_erased))`.
- 이 조작은 pretrained weights는 건드리지 않으며 추가 forward pass 하나(erased obs)만 필요.
- γ는 guidance scale hyperparameter; Sec. 4의 sweep(Table 7)로 튜닝.

---

## 3. 실험 결과

### LIBERO (Table 1, 500 trials/category)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| π0 FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| GR00T-N1.6 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| π0.5 baseline | 95.4 | 98.4 | 97.2 | 89.6 | 95.2 |
| **π0.5 + TAG-bg** | **97.8** | **99.2** | **97.4** | **97.0** | **97.9 (+2.7)** |
| π0.5 + TAG-black | 97.6 | 99.8 | 97.4 | 95.8 | 97.7 (+2.5) |

Long category에서 π0.5 89.6→97.0 (**+7.4%p**)로 long-horizon 누적 오류를 뚜렷하게 완화.

### LIBERO-Plus Robustness (Table 2, 7 perturbation axes)

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|---|
| π0.5 baseline | 64.8 | 71.8 | 83.0 | 93.5 | 92.2 | 78.8 | 85.5 | 81.4 |
| **π0.5 + TAG-bg** | 75.8 | 80.0 | 79.8 | 97.5 | 96.8 | 88.0 | 85.0 | 86.1 (+4.7) |
| **π0.5 + TAG-black** | **77.5** | **80.5** | 81.5 | **98.0** | **97.2** | **89.0** | **87.0** | **87.2 (+5.8)** |

특히 Camera(viewpoint shift, 64.8→77.5) 및 Noise(78.8→89.0)에서 visual robustness 개선 확연.

### VLABench Track 1 (Table 3, SR) & Table 4 (Process Score)

| Base / Task | Select Toy | Select Fruit | Select Painting | Select Poker | Select Mahjong | Avg SR |
|---|---|---|---|---|---|---|
| π0 base | 54.0 | 48.0 | 16.0 | 6.0 | 6.98 | 26.20 |
| π0 + TAG-bg | 48.0 | 50.0 | 36.0 | 48.0 | 15.22 | **39.44 (+13.24)** |
| π0.5 base | 28.0 | 44.0 | 26.0 | 32.0 | 17.02 | 29.40 |
| **π0.5 + TAG-bg** | **38.0** | **78.0** | **78.67** | **53.06** | **58.16** | **55.41 (+26.01)** |

Cluttered visual distractor가 있는 fine-grained selection에서 **평균 SR 2배 가까이 상승**. Process Score(Table 4)도 π0.5 0.4206→0.5920(+0.1714).

### Ablation
- **Training strategy (Table 5)**: Training=**Erase** + Eval=BG가 55.41%로 최적. Training에서 단순 black/BG는 47.01/39.33%로 대폭 열위 → 명시적 object erasure가 foreground-agnostic representation 학습에 필수.
- **Inference-time unconditional (Table 6)** / **Guidance scale γ (Table 7)**: 각각 inpainting style과 γ sweep 제시, default setting이 LIBERO/VLABench 모두 최적.

---

## 4. 한계 및 미해결 문제

1. **Auxiliary vision stack 의존성**: Grounding DINO + SAM 2 + MiniMaxRemover가 필요. Offline 학습용 counterfactual 생성은 비용이 크며 real-time scenario에서 erased observation을 online 합성하려면 expensive.
2. **Language category에서 후퇴**: LIBERO-Plus Language axis에서 TAG-bg 79.8 < baseline 83.0(Table 2) — background를 지워도 language grounding에는 해가 될 수 있음.
3. **Target-object 정의 의존**: 하나의 manipulation target을 가정. Bimanual / multi-object sequential task에서 "erase 대상"을 결정하는 logic이 단순화되어 있음.
4. **π0/π0.5 flow-matching만 시연**: Diffusion / autoregressive / FAST-token VLA로의 extension은 미제시(velocity field residual extrapolation이 다른 head에 그대로 적용될지 불명).

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — CFG의 robotics 적용은 개념적 transplant지만 "target-agnostic counterfactual = foreground erase"라는 concrete instantiation과 closed-loop inpainting audit은 새롭고 실용적 |
| **Practical impact** | ★★★★★ — 30K step calibration만으로 SOTA baseline 대비 일관된 +2–26%p 개선. 특히 이미 near-saturate인 LIBERO에서 Long +7.4%p, cluttered VLABench에서 배 가까운 개선은 drop-in upgrade |

핵심 메시지: **"VLA inference에도 CFG가 작동한다. Target-object를 지운 counterfactual observation이 classifier-free guidance의 unconditional branch 역할을 하여 velocity field를 object-centric 쪽으로 extrapolate할 수 있다."** Table 3의 VLABench gain(π0.5 29.40→55.41)이 특히 강력한데, 이는 visually similar distractor가 많은 시나리오에서 policy가 "target 없이 볼 때의 trajectory"와 "target 포함한 trajectory"의 차이에 집중하게 만드는 메커니즘을 정량적으로 보여준다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 Training=Erase + Eval=Background가 최적인가? | Erase(inpaint)는 가장 realistic하여 학습 시 distribution shift를 최소화하면서도 "target 없음"을 명확히 signal. Evaluation에서는 Background(원복된 배경)가 inpainting artifact 없는 clean counterfactual을 제공. Table 5: E+BG 55.41 vs B+B 47.03. |
| 2 | CFG residual이 LIBERO-Plus의 Camera/Noise 같은 visual perturbation에 왜 특히 효과적인가? | Perturbation하에서 baseline velocity는 noise에 sensitive. Counterfactual(target 제거) observation도 같은 perturbation을 겪으므로 **perturbation-induced velocity error가 subtract됨** → object-specific signal이 선명해짐. Table 2: Camera 64.8→77.5, Noise 78.8→89.0. |
| 3 | 왜 Language category에서는 오히려 하락하는가? | Language perturbation은 visual 변경이 아닌 instruction 변경. TAG는 visual-counterfactual guidance라 language shift를 상쇄하지 못함; 오히려 foreground masking이 weak language grounding과 결합하면 noise가 증폭될 수 있음. TAG-black(81.5)이 TAG-bg(79.8)보다 나아 erase 강도에 민감. |

<!-- VERIFIED: pdf -->
