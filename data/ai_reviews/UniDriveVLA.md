# UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving

> **한 줄 요약**: Xiaomi EV + HUST + UMacau의 autonomous-driving VLA. Qwen3-VL(2B Base / 8B Large) 위에 **Mixture-of-Transformers(MoT) 3-expert 아키텍처(Understanding / Perception / Action)**를 masked joint attention으로 결합하여 representation interference를 해소하고, sparse perception(detection, map, occupancy, motion, ego anchor) + 3-stage progressive training(pretrain → joint LoRA → freeze-VLM refine)으로 Bench2Drive Driving Score 78.37, nuScenes(no-ego) L2-UniAD avg 0.90m를 달성. Code 공개.

---

## 1. 배경 및 동기

- 기존 driving VLA는 monolithic shared-weight Transformer로 vision-language 처리와 spatial perception을 동시에 학습. 저자들은 이를 "**perception–reasoning conflict**"로 규정(Fig. 1, Fig. 2).
- Fig. 2a: LLM token과 perception token 간 cosine similarity가 학습 진행에 따라 수렴 → 서로 다른 표현 요구가 간섭. Fig. 2b: shared-weight decoder에서 understanding/perception/planning 성능이 trade-off.
- Sparse 2D VLA는 semantic reasoning은 강하나 spatial grounding이 약하고, dense BEV-based VLA는 반대 (Fig. 1 a/b/c). 두 축을 하나의 모델에서 **표현 분리**로 함께 얻는 것이 목표.

---

## 2. 방법론

### MoT Architecture (Fig. 3)
- Qwen3-VL을 backbone으로 사용(SigLIP-2 vision encoder + MLP merger + Qwen3 LM), input resolution 960×544(32× downsampling).
- 세 expert가 **parameter-wise 분리되지만 masked joint attention**으로 cross-expert interaction:
  - **Understanding Expert**: Qwen3-VL language pathway, VQA / scene 이해 담당
  - **Perception Expert**: detection / map / occupancy / motion anchor token 처리 (sparse perception paradigm)
  - **Action Expert**: flow-matching trajectory generator — Gaussian noise와 target velocity sequence를 interpolate
- Masked joint attention(Fig. 4): 각 expert가 자신의 token을 query로, 허용된 다른 expert의 key/value를 attend하되 control은 mask로 수행하여 **불필요한 gradient 오염 방지**.

### 3-Stage Progressive Training (Sec. 3)
- **Stage 1**: VLM backbone full fine-tune 3 epoch, lr 4e-5, **driving:general = 3:7** data mixture (general 부분은 FineVision). ReCogDrive pretrain recipe 채택.
- **Stage 2**: 30 epoch joint 최적화(AdamW), base lr 2e-4, **VLM 0.5× multiplier(effective 1e-4)**, LoRA on language model, EMA. "Perception/Action expert는 적극 학습, VLM은 보수적 update"가 의도.
- **Stage 3**: VLM freeze, Perception + Action expert 15 epoch fine-tune, lr 1e-4, **motion forecasting objective** 추가. Dynamic prior가 action expert에 주입됨.

### Sparse Perception
BEV dense map 대신 detection / map / occupancy / motion / ego를 **anchor token**으로 sparse하게 표현. Perception expert가 이 anchor token을 language token과 같은 space에서 처리하므로 VLM의 semantic pathway를 유지.

---

## 3. 실험 결과

### Bench2Drive Closed-Loop (Table 1)

| Method | Avg. L2↓ | Driving Score↑ | Success Rate↑ | Efficiency↑ | Comfortness↑ |
|---|---|---|---|---|---|
| UniAD | 0.73 | 45.81 | 16.36 | 129.21 | 43.58 |
| DriveTransformer | 0.62 | 63.46 | 35.01 | 100.64 | 20.78 |
| ReCogDrive (ICLR26) | — | 71.36 | 45.45 | 138.18 | 17.45 |
| DriveMOE (CVPR26) | 0.38 | 74.22 | 48.64 | 175.96 | 15.31 |
| Orion (ICCV25) | 0.68 | 77.74 | 54.62 | 151.48 | 17.38 |
| **UniDriveVLA** | 0.72 | **78.37** | 51.82 | **198.86** | 11.78 |

PDM-Lite expert 없이 학습된 method 중 Driving Score **SOTA 78.37** 및 Efficiency SOTA 198.86.

### Bench2Drive Multi-ability (Table 2)

| Method | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|---|---|---|---|---|---|---|
| DriveMOE | 34.67 | 40.00 | 65.45 | 40.00 | 59.44 | 47.91 |
| Orion | 25.00 | 71.11 | 78.33 | 33.00 | 69.15 | 54.72 |
| **UniDriveVLA** | **38.75** | **80.00** | 50.00 | 30.00 | 58.95 | 51.53 |

Merging / Overtaking 등 interactive scenario에서 최고 성능(각각 38.75 / 80.00), 단 Emergency Brake / Give Way는 Orion 대비 열위(comfortness 11.78로 aggressive한 주행 정책 시사).

### nuScenes Open-Loop Planning (Table 3, UniAD metrics, no ego-state)

| Method | L2 1s | L2 2s | L2 3s | L2 Avg | Coll. Avg | LLM |
|---|---|---|---|---|---|---|
| UniAD | 0.59 | 1.01 | 1.48 | 1.03 | 0.77 | — |
| FSDrive | 0.40 | 0.89 | 1.60 | 0.96 | 0.40 | Qwen2-VL-3B |
| SparseDrive‡ | 0.38 | 0.92 | 1.66 | 0.99 | 0.21 | — |
| **UniDriveVLA-Large** | **0.36** | **0.83** | **1.50** | **0.90** | **0.27** | Qwen3-VL-8B |

no-ego setting에서 **L2 avg 0.90m로 SOTA**. with-ego setting(ST-P3)은 L2 avg 0.42로 OpenDriveVLA 0.33 대비 열위이지만 collision은 0.10으로 경쟁적.

### nuScenes Perception (Table 4)

| Method | Det mAP↑ | NDS↑ | Map mAP↑ | Motion minADE↓ | minFDE↓ |
|---|---|---|---|---|---|
| SparseDrive | 0.418 | 0.525 | 0.551 | 0.600 | 0.960 |
| HiP-AD | 0.424 | 0.535 | 0.571 | 0.610 | — |
| UniDriveVLA-Large | 0.407 | 0.460 | 0.535 | 1.264 | 2.121 |

Detection/Map은 specialized baseline과 경쟁 가능하지만 **motion prediction(ADE/FDE)은 열위** — VLA joint training의 명확한 trade-off.

### Ablations

- **Planning components (Table 5)**: Ego 0.75→0.61, +Det 0.58 / CR 0.10, +Occ best L2 0.53. Motion/Map 추가는 open-loop에선 marginal.
- **MoT vs shared-weight (Table 7)**: General VQA 31.1→45.5, DriveBench 50.8→54.9, L2 0.641→0.533, CR 0.175→0.140 — **모든 축에서 MoT 우세**, "decoupling이 interference를 해소한다"는 핵심 claim을 뒷받침.
- **DriveBench (Table 6)**: UniDriveVLA 평균 51.97, Behavior 60.97로 GPT-4o의 45.40보다 높음.
- **General multimodal (Table 8)**: 8B model이 MMStar 43.3, RealWorldQA 49.9, AI2D 76.3 — Qwen3-VL-8B(63.0 / 69.0 / 83.2) 대비 저하되나 "driving adaptation 후에도 multimodal reasoning이 완전히 붕괴되지 않음"을 보여줌.

---

## 4. 한계 및 미해결 문제

1. **Motion forecasting 약점 (Table 4)**: minADE 1.264 / minFDE 2.121로 SparseDrive(0.600/0.960) 대비 두 배 이상 열위. Stage 3에서 motion objective를 주입했지만 specialized baseline 수준엔 미도달.
2. **General VLM 성능 후퇴 (Table 8)**: 8B 모델의 MMStar 63.0→43.3 등 driving adaptation cost가 상당. 저자들도 "general benchmark에서는 specialized VLM에 미치지 못한다"고 인정.
3. **Comfortness 저하 (Table 1)**: Driving Score SOTA지만 comfortness 11.78로 Orion 17.38/DriveMOE 15.31 대비 낮음. Efficiency-oriented 궤적이 승차감을 희생하는 trade-off 존재.
4. **Closed-loop는 Bench2Drive 한정**: CARLA simulator 의존, 실제 주행 검증 부재. nuScenes는 open-loop.
5. **PDM-Lite 미사용 disclaimer**: Table 1의 "PDM-Lite trained" 그룹(R2SE 등)과 동일 조건 비교를 피하기 위한 설정.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — MoT를 VLA에 적용하고 perception-reasoning conflict를 명시적으로 정량화한 것이 기여. Sparse perception × flow-matching action × 3-stage training의 조합은 현재까지 driving VLA에서 가장 체계적 |
| **Practical impact** | ★★★★☆ — Xiaomi가 production 관점에서 발표, open-source(github.com/xiaomi-research/unidrivevla). Bench2Drive SOTA(78.37) + nuScenes no-ego SOTA + interactive ability에서 가장 균형 있음 |

핵심 메시지: **"Autonomous driving VLA의 성능은 단일 Transformer로 perception/reasoning/action을 모두 처리하는 monolithic 설계의 한계에 부딪히며, MoT로 parameter를 분리하고 masked joint attention으로 선택적으로 섞을 때 모든 축이 함께 개선된다."** Table 7의 shared-weight vs MoT 직접 비교가 이를 깔끔하게 증명한다(General VQA +14.4%p, DriveBench +4.1%p, L2 −16.9%).

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Stage 2에서 VLM에 0.5× lr multiplier를 주는 이유? | VLM은 Stage 1에서 이미 driving+general pretraining을 마친 상태. Stage 2는 Perception/Action expert를 처음 학습시키므로 expert에는 aggressive update(lr 2e-4), VLM은 보수적 update(effective 1e-4)를 적용해 semantic reasoning을 보존하면서 expert가 spatial prior를 학습. LoRA 적용으로 VLM parameter drift를 추가로 억제. |
| 2 | Motion prediction이 specialized baseline 대비 약한데 이게 planning SOTA와 어떻게 양립하는가? | nuScenes planning은 ego-trajectory 예측이 주, agent motion forecast는 보조. Table 4에서 detection/map은 경쟁력 유지하므로 "where obstacles are + semantic context"로 planning이 가능. Table 5 ablation에서 motion 제거해도 L2 큰 변화 없음(0.53→0.54). 단, dense-interaction long-horizon scenario에서는 motion 약점이 드러날 가능성 있음. |
| 3 | MoT와 단순한 task-specific head split의 차이는? | MoT의 핵심은 "각 expert가 **독립 transformer stack**을 갖지만 masked joint attention으로 선택적으로 cross-attend"하는 점. Shared-weight decoder(Table 7)는 모든 layer가 parameter 공유 → gradient conflict; task-specific head는 마지막 layer만 분리 → early feature에서 conflict 여전. MoT는 stack 전체를 분리하여 근본적 interference를 차단. |

<!-- VERIFIED: pdf -->
