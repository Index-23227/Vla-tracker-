# TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies

> **한 줄 요약**: Co-Tracker로 추출한 active point trajectory를 원본 이미지 위에 시각적으로 overlay하는 **visual trace prompting**으로 OpenVLA-7B를 finetune. SimplerEnv Google Robot 3-task 평균에서 OpenVLA 대비 +7.5%p, real-world WidowX-250 8 task에서 OpenVLA 대비 약 3.5× success를 달성한 ICLR 2025 논문 (UMD/MSR).

---

## 1. 배경 및 동기 (Sec. 1)

- VLA는 현재 frame을 보고 reactive하게 행동 → **과거 motion에 대한 spatial-temporal awareness가 결여**.
- 단순히 history frame을 concat하면 정보 중복/distraction 문제 (Sec. 3 도입부).
- 저자 제안: **point trajectory를 시각적으로 overlay**해 모델이 자기 과거를 자연스럽게 파악하도록 함.

---

## 2. 방법론

### Visual Trace Generation (Sec. 3.1, Fig. 2)
- Co-Tracker로 timestep $t-N$ ~ $t$의 dense $K\times K$ point trajectory 추출 (구현: $K=40$).
- Active point set $\hat{\mathcal{P}} = \{p \in \mathcal{P} \mid \sum |\Delta p_{t'}| > \kappa\}$로 움직임이 큰 점만 보존.
- 그 중 **$M=5$개 active trace**를 random sample → 이미지에 overlay.
- 시간 window $N=6$, 매 $N$ step마다 dense tracking을 갱신해 연산량 절감.

### Model Architecture (Sec. 3.2, Fig. 1)
- **OpenVLA(7B)** finetune이 base. (1) 원본 image, (2) trace overlay image를 모두 입력하고, **learnable separator token**으로 분리.
- Text prompt에 trace 존재를 명시 ("You are given two images: one with the original robot observation, and another marked with historical traces…").
- **Dropout α**: 학습 시 확률 α로 trace image를 원본으로 대체 → Co-Tracker 실패 시(test-time) robust 작동.

### Implementation (Sec. 3.3)
- Finetune dataset 150K trajectory (BridgeData-v2 + Google RT-1 + WidowX 120 demo).
- 5 epoch finetune; TraceVLA-Phi3는 4B Phi-3-Vision backbone, OXE 970K trajectory 30 epoch pretrain (32 H100).

---

## 3. 실험 결과 (PDF 표 인용)

### SimplerEnv Google Robot (Table 1, Sec. 4.1)
| Model | Visual Matching: Move Near / Pick Coke / Drawer | Variant Aggregation: Move Near / Pick Coke / Drawer | Overall |
|---|---|---|---|
| OpenVLA-Phi3 (4B) | 46.1 / 46.7 / 22.5 | 51.9 / 49.7 / 22.2 | 39.9% |
| **TraceVLA-Phi3** | 50.4 / 52.2 / 31.0 | 55.0 / 52.4 / 23.2 | **44.0%** (+4.1) |
| OpenVLA (7B) | 47.1 / 15.3 / 49.5 | 54.0 / 52.8 / 22.5 | 40.2% |
| **TraceVLA (7B)** | 53.7 / 28.0 / 57.0 | 56.4 / 60.0 / 31.0 | **47.7%** (+7.5) |
| Octo-Base | – | – | 8.2% |
| RT1-X | – | – | 45.8% |

**관찰**: Visual Matching 설정에서 **Pick Coke Can 15.3 → 28.0 (+12.7pp)**, Drawer 49.5 → 57.0 (+7.5pp). 즉 동일 시나리오에서도 visual trace가 일관되게 향상.

### Environmental Variant (Fig. 4, Sec. 4.1)
- Lighting: OpenVLA 41.7 → TraceVLA 50.8% (+9.1pp).
- Background: 41.2 → 52.3 (+11.1).
- Distractor: 54.3 → 66.7 (+12.4).
- Camera/orientation 등 모든 5 axis에서 향상, 일부 카테고리는 +20% 이상.

### Real-world WidowX-250 (Fig. 6, Sec. 4.2)
- 8 task, 각 10 trial 기준.
- **Pickplace Corn (training data 미포함)**: OpenVLA 1/10 → TraceVLA **8/10** (8×).
- Fold Cloth 2 vs 7, Pickplace Knife 4 vs 8, Lift Battery 4 vs 9 등 거의 모든 task에서 두 배 이상.
- 4개 unseen 일반화 task에서도 일관되게 향상.

---

## 4. 한계 및 비판적 고찰

1. **Co-Tracker 의존성**: Tracking 실패 시 trace 정보 손실 → dropout으로 완화하지만 lighting 극단 조건 (Sec. 3.2 명시)에선 약화 가능.
2. **Visual occlusion**: trace가 end-effector나 객체를 가릴 수 있음 → 원본 이미지 추가 입력으로 회피하나 token 비용 2× 증가.
3. **2D trace의 한계**: 깊이 정보 없음. Camera viewpoint shift 시 trace의 의미가 변함.
4. **Real-time inference**: dense Co-Tracker 호출은 비용이 높아 sparse query (M active points)로 줄였으나, latency 명시 수치는 paper에 없음.
5. **150K trajectory**: 학습 비용 (5 epoch × OpenVLA-7B fine-tune)이 작지 않음.

### YAML 정합성 검토
- YAML `simpler_env.google_robot_pick_coke_can: 72.8` → **PDF Table 1에는 그런 수치 없음**. PDF 최고치는 Visual Matching Pick Coke 28.0% / Variant Aggregation 60.0% / 둘 평균 44.0%. **72.8% 출처 불명 → 수정/검증 필요.**
- YAML `simpler_env.google_robot_move_near: 56.4` → PDF Variant Aggregation Move Near 56.4%와 정확히 일치하지만 Visual Matching은 53.7. **부분만 일치.**
- YAML `parameters: 7B` 일치, `backbone: PrismaticVLM (SigLIP+DinoV2)`, `llm: Llama-2 7B` 일치 (OpenVLA 기반).
- YAML `compute: 16 A100 GPUs` → PDF는 Phi3 pretrain에 32 H100 사용. 7B finetune의 GPU 수는 본문에 미명시. **검증 필요.**

---

## 5. 총평

**강점**: 단순한 시각적 prompting만으로 7B VLA에 spatial-temporal memory를 부여. ICLR 2025 게재. **약점**: Co-Tracker dependency, 2D 한계, YAML SimplerEnv 수치 불일치.

---

## 6. 예상 날카로운 질문

| # | 질문 | 답 (논문 근거) |
|---|------|----------------|
| 1 | Naive history concat과 차이는? | Sec. 3 도입: history frame은 시각적 redundancy로 distraction 유발. Trace는 핵심 motion 정보만 압축적으로 시각화 → Table 1에서 +7.5pp의 gain은 단순 concat baseline 대비 유의미한 차이. |
| 2 | Trace가 잘못 그려지면? | Sec. 3.2 dropout α: 학습 시 trace image를 원본으로 교체하는 augmentation으로 robustness 확보. Co-Tracker 실패 케이스에서도 작동. |
| 3 | Phi-3 4B 모델로도 효과? | Table 1: TraceVLA-Phi3 39.9 → 44.0 (+4.1pp). 7B에서 더 큰 gain (+7.5pp)이지만 작은 모델에서도 일관된 효과. |

<!-- VERIFIED: pdf -->
