# VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models

> **한 줄 요약**: HKUST/CUHK/SmartMore의 dual-system VLA. **System 2 Planner**(Qwen3-VL-4B-Instruct + SAM3)가 instruction을 subtask로 decompose하고 target object 중심에 crosshair, placement region에 bounding box를 **overhead camera에 직접 그린** visual prompt를 생성, **System 1 Controller**(QwenOFT = OpenVLA-OFT의 Prismatic VLM을 Qwen3-VL-4B-Instruct로 교체)가 이 visual overlay를 조건으로 continuous action을 regress. SimplerEnv WidowX Avg **58.3**, RoboCasa-GR1 Avg **53.8**, Realworld ID/OOD generalization gap 2.5%.

---

## 1. 배경 및 동기

- Monolithic VLA(OpenVLA, π0)는 single-system으로 reasoning과 motor control을 함께 수행하지만 **(i) complex spatial instruction grounding 약함, (ii) OOD appearance/position에서 generalization 제한**(Sec. 1, Table 3의 QwenOFT ID 80% → OOD 63.3% drop).
- Textual instruction을 policy에 그대로 전달하는 것은 "pick the 3rd blue block from the left" 같은 spatial reference에서 bottleneck. 저자들은 **visual prompt**(overlay된 crosshair, bounding box)가 언어보다 우월한 interface라고 주장 — "a picture is worth a thousand words".
- Prior "System 1 / System 2" dual-system(Kahneman)의 robotics 적용(RoboDual, Helix 등)은 존재하나, **interface modality를 visual prompt로 통일**한 것은 VP-VLA의 독창성.

---

## 2. 방법론

### System 2: VLM Planner (Sec. 3)
- **Qwen3-VL-4B-Instruct + SAM3**로 구성.
- **Event-driven** 실행: gripper state transition(close↔open)을 trigger로 새 subtask가 결정될 때만 호출 → per-step VLM inference 불필요.
- Pipeline: (1) instruction을 subtask list로 decompose, (2) 현재 subtask의 target object와 goal location을 language로 식별, (3) SAM3 segmentation으로 object centroid(crosshair)와 placement region(bounding box)을 overhead camera frame에 **렌더링**.
- Prompt template은 Appendix Table 9(task decomposition)와 Table 10(subtask completion detection)에 명시.

### System 1: QwenOFT Controller (Sec. 3)
- **OpenVLA-OFT의 Prismatic VLM backbone을 Qwen3-VL-4B-Instruct로 교체**한 QwenOFT 아키텍처 (starVLA 커뮤니티 구현).
- Continuous action decoder, L1 action loss.
- **Auxiliary visual grounding objective**: 1000개 spatial bin에 대한 cross-entropy로 2D anchor(object centroid)와 bounding box coordinate를 예측. Gradient는 **VLM backbone으로만** backprop(action head에는 영향 없음), 이로써 visual prompt understanding을 VLM level에서 강화. loss weight λ=0.1.
- Visual prompt는 observation image에 overlay되어 controller의 기존 pipeline에 자연스럽게 편입 — 추가 modality port 불필요.

### Training
- Framework: **starVLA** (open source community fork).
- 8 GPUs, AdamW, VLM lr 1e-5 / action model lr 1e-4.
- SimplerEnv: Open X-Embodiment의 BridgeDataV2 + Fractal subset, 70K step, batch 32/device.
- Real-world: Franka Research 3, 8 GPUs, batch 256, action chunk 16.

---

## 3. 실험 결과

### SimplerEnv WidowX (Table 2)

| Method | Spoon on Towel | Carrot on Plate | Stack Green Block | Eggplant in Basket | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 34.2 | 30.0 | 30.0 | 72.5 | 41.8 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| π0 | 29.2 | 62.5 | 29.2 | 91.6 | 53.1 |
| π0.5 | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| Isaac-GR00T-N1.6-Bridge | 64.5 | 65.5 | 5.5 | 93.0 | 57.1 |
| QwenOFT + Qwen3VL (baseline) | 58.3 | 50.0 | 20.8 | 70.8 | 50.0 |
| **Ours + Qwen3VL (VP-VLA)** | **66.7** | 50.0 | 20.8 | **95.8** | **58.3** |

QwenOFT baseline 대비 **+8.3%p**, π0.5 / GR00T-N1.6-Bridge 대비 **+1.2%p**. Put Eggplant in Basket에서 70.8→95.8로 +25%p는 visual prompt가 object grounding을 크게 돕는 evidence.

### RoboCasa-GR1-Tabletop (Table 1, 25 task averaged)

| Block / Method | Isaac-GR00T N1.5 | Isaac-GR00T N1.6 | QwenGR00T | QwenPI | QwenOFT | QwenFAST | **Ours** |
|---|---|---|---|---|---|---|---|
| PnP * to * Close (Avg) | 45.3 | 24.2 | 50.3 | 42.3 | 43.7 | 35.0 | **54.3** |
| PnP Novel Cuttingboard→* Avg | 46.4 | 56.9 | 52.8 | 46.0 | 50.4 | 50.4 | **60.8** |
| PnP Novel Placemat→* Avg | 45.5 | 51.9 | 38.0 | 43.5 | 41.5 | 33.5 | **54.5** |
| PnP Novel Tray→* Avg | 48.8 | 55.1 | 39.2 | 44.0 | 49.2 | 32.0 | 46.0 |
| PnP Novel Plate→* Avg | 56.5 | 57.6 | 58.5 | 44.0 | 61.0 | 45.0 | 53.5 |
| **Overall Average** | 48.2 | 47.6 | 47.8 | 43.9 | 48.8 | 39.0 | **53.8** |

5개 category 중 3개에서 SOTA. Novel Placemat→Plate에서 QwenOFT 52.0 → VP-VLA 70.0, Novel Tray→Plate 56.0→66.0 등 novel appearance/distractor 조건에서 개선 뚜렷.

### Real-world Robot Experiments

- **Waste sorting categorization (Table 3, Franka Research 3, 10K steps)**: ID 87.5% vs QwenOFT 80%, OOD **85% vs 63.3%** — generalization gap이 **2.5% vs 16.7%**. Scrambled Rubik's cube(visual OOD) 9/10 vs 3/10.
- **Color-based egg pick (Table 4)**: ID 77.1% vs 58.3%, OOD Color **75.0% vs 29.2%**, OOD Position 75.0 vs 54.2. Attribute binding이 visual prompt로 대폭 강화.
- **Egg carton placement (Table 5, 4×4 grid)**: ID 91.25% vs 70.63% — coordinate(L1C2 등) 기반 precise placement에서 visual bounding box가 linguistic ambiguity 해소.

### Ablation (Appendix, Table 7/8)
- Table 7: per-task RoboCasa ablation, visual grounding loss와 task decomposition 각각의 기여를 분해.
- Table 8: SimplerEnv에서 task decomposition 없이 실행하면 성능이 QwenOFT 수준으로 회귀 → **decomposition이 visual prompting의 효과를 amplify**.

---

## 4. 한계 및 미해결 문제

1. **Task decomposition의 dependency**: Table 8이 시사하듯 decomposition 실패 시 visual prompt 효과가 희석. 복잡한 interleaved task에서 Planner의 decomposition 품질이 bottleneck.
2. **SAM3 / VLM compute overhead**: Event-driven이라 every-step은 아니지만 gripper transition마다 Qwen3-VL-4B + SAM3 실행이 필요. Edge deployment에서 latency issue 가능.
3. **Overhead camera 의존**: Crosshair/bounding box가 top-down view에 그려진다고 가정. Mobile manipulation이나 non-tabletop 시나리오에서는 visual prompt 좌표계 정의가 난제.
4. **SimplerEnv 일부 task 미개선**: Stack Green Block 20.8(baseline 동일), Carrot on Plate 50.0(baseline 동일) — visual prompt가 이 task에서는 추가 signal을 제공하지 못함. Novel Tray→* / Novel Plate→* category도 baseline 대비 후퇴.
5. **Closed-source**: code_url은 project page(visualprompt-vla.github.io)만, 실제 code release는 미공개.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual prompting을 VLA interface로 공식화하고 Planner-Controller 사이의 communication protocol로 확립한 framing이 새롭다. Auxiliary visual grounding loss로 VLM-level visual prompt understanding을 명시적으로 학습시키는 설계도 합리적 |
| **Practical impact** | ★★★★☆ — SimplerEnv 58.3 / RoboCasa 53.8은 기존 dual-system method 대비 경쟁력 있으며, real-world OOD에서 generalization gap 2.5%는 인상적 |

핵심 메시지: **"Linguistic ambiguity가 존재하는 spatial reference task에서 visual prompt(crosshair, bounding box)는 language instruction보다 robust한 interface이며, 이를 통해 dual-system VLA의 Planner↔Controller 통신 채널을 통일할 수 있다."** Table 4의 OOD Color generalization gap(VP-VLA 2.1% vs QwenOFT 29.1%)과 Table 3의 Rubik's cube 9/10 vs 3/10 대비가 가장 강력한 evidence. 언어만 받는 baseline이 "color heuristics 또는 positional memorization"에 의존하는 실패 pattern을 정량적으로 보여주었다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Auxiliary visual grounding loss의 gradient가 왜 VLM에만 흐르고 action head에는 안 흐르는가? | Action head가 grounding coordinate에 직접 fit하면 action regression과 grounding이 objective 혼선. Grounding은 **VLM이 visual prompt를 parse하는 능력**만 강화해야 하므로 VLM layer까지만 backprop. 결과적으로 VLM hidden state가 더 잘 "crosshair/bbox를 이해"하게 되고 action head는 이 representation을 자연스럽게 활용. λ=0.1로 weight도 작음. |
| 2 | Event-driven Planner invocation이 안정성을 해치지 않는가? (중간에 object가 바뀌면?) | Gripper transition은 대부분의 manipulation에서 subtask boundary와 일치(pick→place는 close→open으로 자연 분할). Planner가 subtask를 decompose할 때 sequential list를 미리 생성하므로 다음 subtask 조회는 O(1). 예: "place wine in cabinet + close cabinet"에서 wine 배치 후 target이 door로 전환되는 예시가 Sec. 4에서 정상 작동 확인. |
| 3 | Qwen3-VL-4B-Instruct를 왜 Prismatic VLM 대신 쓰는가? QwenOFT의 기여는? | Prismatic VLM은 상대적으로 약한 OCR/visual-reasoning 능력. Qwen3-VL은 stronger multimodal reasoning과 grounded visual understanding을 제공해 visual prompt overlay를 더 잘 해석. Table 2에서 QwenOFT baseline(50.0)이 이미 OpenVLA-OFT(41.8) 대비 +8.2%p → backbone 교체 자체가 유의미한 기여이며, VP-VLA의 58.3은 여기에 visual prompting 설계가 추가로 +8.3%p 얹은 결과. |

<!-- VERIFIED: pdf -->
