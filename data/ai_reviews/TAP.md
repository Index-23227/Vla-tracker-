# TAP: Learning to Move Before Learning to Do — Task-Agnostic Pretraining for VLAs

> **한 줄 요약**: VLA 학습을 "어떻게 움직일 것인가(physical competence)"와 "무엇을 할 것인가(semantic alignment)"로 분해하는 **Decomposition Hypothesis**에 기반, 버려지는 off-task 궤적과 자율 random play 같은 **언어 라벨 없는 task-agnostic 데이터**로 self-supervised **Inverse Dynamics** 사전학습(Stage 1)을 수행한 뒤, 소량의 expert 데이터로 언어 정렬(Stage 2 BC)만 하는 2단계 프레임워크. SIMPLER에서 Standard BC 대비 **+10%p 절대 향상**(Avg-All 23.15→33.32%), 실세계 WidowX에서 viewpoint 변화 시 OXE 기반 NORA가 0%로 붕괴할 때 **15–25% 성공률 유지**. (Fudan University, Shanghai Innovation Institute — ICML 2026)

---

## 1. 배경 및 동기

- **데이터 병목**: VLA 학습은 (observation, instruction, action) triplet 형태의 expert demonstration에 의존하며, human teleoperation 기반 수집은 비용·노동 집약적이고 운영자 가용성에 의해 근본적으로 non-scalable.
- **발달심리학적 영감**: 인간 영아는 expert demonstration 없이 curiosity-driven, task-unaware한 자기 탐색(뻗기, 만지기, 떨어뜨리기)으로 물리·affordance·감각운동 규칙을 먼저 습득 [9–11]. "how to move"의 grounding은 "what to do"와 분리되어 능동적 상호작용에서 자연 발생.
- **핵심 관찰**: task instruction은 "what to do"에만 필요하고 "how to move"에는 불필요. 그러나 현행 VLA 파이프라인은 instruction 없는 궤적을 전부 폐기 → 풍부하고 저렴한 자원의 낭비.
- **본 논문의 질문**: 언어 주석 없는 상호작용 데이터에서 물리적 사전지식을 추출해 expert 데이터 의존을 수십~수백 배 줄일 수 있는가?

---

## 2. 문제 설정 (Task-Agnostic Data란)

- Task-agnostic 데이터: 언어 지시 `l`이 없는 궤적 `τ = (o_0, a_0, o_1, ..., o_T)`. 유효한 물리적 상호작용은 담지만 인간이 정의한 "목적"은 없음.
- **두 가지 공급원**:
  1. **Repurposed 기존 데이터셋**: Bridge/OXE에서 타깃 태스크와 무관해 폐기되던 궤적 (예: 타깃이 "put carrot on plate"일 때 "open drawer" 궤적) — grasping dynamics, 충돌 반응, EEF 제어 등 전이 가능한 물리 prior 함유.
  2. **Autonomous Random Play**: 무작위 EEF 명령으로 로봇이 스스로 밀기/쓸기/넘어뜨리기/잡기 수행. 인간 개입 거의 없이 "사실상 무료"로 수집되며 해당 embodiment/workspace 특화.

---

## 3. 방법론 심층 분석

### 3.1 Stage 1: Task-Agnostic Pretraining (Inverse Dynamics)
- 목표: `p(a_t | o_t, o_{t+1})` — 두 관측 사이 전이를 일으킨 행동 예측.
- **왜 ID인가**: 프레임 간 "무엇이 변했는가"(EEF 이동, 물체 변위)에 집중하고 정적 배경(조명, 텍스처, clutter)을 무시하도록 강제 → "세상이 어떻게 보이는가"가 아닌 "세상이 어떻게 변하는가"를 인코딩하는 dynamics-aware 표현 학습.
- 구현: `o_{t+1}`을 암묵적 visual goal로 취급한 visual-only 입력 시퀀스. 손실은 MSE `L_ID = E ||â_t − a_t||²` (식 2–3). Stage 1에서는 visual encoder 동결, VLM backbone + action head만 학습.

### 3.2 Stage 2: Task-Specific Alignment
- 조건 신호를 visual goal `φ(o_{t+1})` → 언어 지시 `ψ(l)`로 교체: `â_t = f_θ(φ(o_t), ψ(l))` (식 4). 표준 BC 손실 (식 5), 전체 파라미터 joint finetuning.
- **작동 원리**: backbone과 action head가 재사용되므로 시각→운동 매핑은 이미 확립됨. Stage 2는 semantic space에서 기확립된 dynamics space로의 경량 projection만 학습 → 필요 라벨 수 대폭 감소.

### 3.3 Autonomous Collection Pipeline (Algorithm 1, Appendix A)
1. **Safe Pose Library**: 무목적 teleoperation으로 workspace를 조밀하게 커버 → 안전 경계 필터링 → Voxel Grid Downsampling(leaf 5cm³)으로 균일·이산 pose library `P_safe` 구축.
2. **Constrained Trajectory Generation**: 최소 거리 제약 하에 waypoint 확률 샘플링 → **contact heuristic**(고도 z_thresh 초과가 c_max 스텝 지속 시 강제 하강)으로 hovering 방지, contact-rich 상호작용 유도 → cosine interpolation → boundary-aware Gaussian noise 주입.
3. 인간 개입은 ~30분마다 물체 셔플로 한정. 25Hz 원시 데이터를 **5Hz로 다운샘플** — 인접 프레임 시각 변화가 센서 노이즈가 아닌 행동에 인과적으로 귀속되고, "approach/grasp/lift" 수준의 의미 있는 primitive를 학습하게 함.

### 3.4 아키텍처 및 학습 세부
- **Qwen2.5-VL (3B)** backbone + **SigLIP** ViT visual encoder (400M, 224×224), action head는 **2-layer MLP** → 7-D delta-pose EEF action (Δx, Δy, Δz, 3D axis-angle, gripper). 상대 운동 예측으로 workspace 좌표 불변성 확보.
- 100k steps, 8×H100 단일 노드 (~24 GPU hours), global batch 128, AdamW (wd 0.05), lr 5e-5 cosine decay (warmup 5k), grad clip 1.0, bfloat16.

---

## 4. 데이터 전략

| 모델 | 사전학습 데이터 | 규모 | 목적함수 (라벨) |
|---|---|---|---|
| RT-1-X / Nora | OXE | ~1.0M | BC (필요) |
| OpenVLA | OXE | ~970k | BC (필요) |
| Octo | OXE | ~800k | Masked BC (필요) |
| π0 | Multi-Emb. | Massive | BC (필요) |
| **TAP Stage 1** | Task-agnostic (off-task Bridge / self-play) | **20k (Sim) / 30h (Real)** | **Inverse Dyn. (불필요)** |
| **TAP Stage 2** | Expert | **5k (Sim) / 0.2k (Real)** | BC (필요) |

Expert 데이터 요구량을 수 orders of magnitude 절감 (Table 1, Table 4). 실세계는 태스크당 expert 200개만 사용.

---

## 5. 주요 실험 결과

### 5.1 SIMPLER (WidowX, Table 2) — RQ1
- **TAP-20k: Avg-All 33.32%** vs Standard BC 23.15% (**+10.2%p**, 동일 아키텍처·동일 Stage 2 데이터), OpenVLA 7.75%, RT-1-X 3.03%, Nora 20.06%. Octo 31.31%를 상회, π0 40.08%에 근접.
- 태스크별 Entire (TAP-20k): Spoon on cloth **58.3%**, Carrot on plate 0.0%, Stack Blocks 16.7%, Eggplant in Basket 8.3% (Avg-Entire 20.82%).
- **스케일링**: 8k→14k→20k 에피소드에서 24.47→30.21→33.32%로 단조 증가 — task-agnostic 노출 확대가 곧바로 downstream 성능으로 전환되며 아직 포화되지 않음.
- 공정성: OpenVLA/NORA/π0는 TAP과 동일한 5k Stage 2 데이터로 finetuning; RT-1-X/Octo는 SIMPLER 원논문 인용.

### 5.2 Partial vs Entire 분해 — RQ2
- TAP Avg-Partial **45.82%** — Octo(42.30%)와 대등, π0(53.10%)에 근접. Partial(성공적 grasping)은 언어 감독 없이 학습된 저수준 물리 역량에 의존 → **물리 역량이 semantic grounding과 분리되어 습득 가능하다는 직접 증거**. Partial success가 31.8→45.8%로 상승(결론부).

### 5.3 실세계 WidowX 250s (Table 3, 600+ trials) — RQ3
- Expert 200개 제한, Stage 1은 30시간 자율 self-play. 5개 시나리오 × 태스크당 20 trials.
- **Push pumpkin 평균: TAP 61% > NORA 56%** (from-scratch 21%); Carrot 평균: TAP 28% vs NORA 36% (from-scratch 9%).
- **Viewpoint Variation**: NORA·BC 모두 **0%** (허공 grasp) vs TAP **15%/25%**.
- **Background Texture Shift**: pushing에서 NORA 55% vs TAP 65%; carrot에서 NORA 10% vs TAP 25%.
- **Visual Distractors**(미학습 과일 clutter): BC 5%로 붕괴, NORA 60%, TAP 65% — self-exploration이 배경의 spurious correlation 대신 gripper-object 인과 역학에 주목하게 만듦.

---

## 6. Ablation / 분석

- **Early saturation 극복 (Fig. 3)**: 초기 학습 속도는 Baseline과 유사하나, Baseline은 ~23%에서 진동·정체하는 반면 사전학습 모델은 30%+까지 계속 상승 → TAP은 semantic 습득을 가속하는 게 아니라 **도달 가능한 성능 상한을 올림** (loss landscape 재편).
- **Stage 1×Stage 2 sweep (Fig. 4, Table 5)**: 성능은 사전학습 규모에 의해 상한이 결정됨. Stage 1 20k steps에서는 Stage 2를 늘려도 ~18% 정체; 100k steps로 확장 시 30%+ (최고 100k/100k = 33.32%). Task-agnostic 데이터가 소량 expert 데이터 overfitting을 막는 regularizer로 작용.
- **Grad-CAM (Fig. 5)**: Stage 1 후 텍스트 없이도 attention이 gripper·조작 가능 물체에 자동 집중(implicit affordance map), 배경 억제. Stage 2의 언어 지시는 attention을 gripper로 강하게 수렴시키는 constraining filter. 실세계 novel background에서도 focus 유지 → domain-invariant 표현.
- **Error 분석 (4.5절)**: 실패의 ~25%는 Execution/Dynamics(미세 접촉, mm 단위 misalignment, depth 모호성), **~75%는 Semantic/Reasoning**(distractor를 완벽한 동작으로 잡기, long-horizon에서 freezing/looping) — 저수준 실행은 견고해졌으나 reactive VLA의 추론 능력이 새 병목.

---

## 7. 효율성

- 총 학습 ~**24 GPU hours** (8×H100 단일 노드) vs OpenVLA 64×A100, RT-1-X/Octo TPU v4 Pods 수 주 — 학계 규모 자원으로 재현 가능 (Table 4).
- 실세계 expert 궤적 <1k (태스크당 200) + ~100k steps의 "무료" self-play로 1M+ expert 궤적 기반 NORA와 대등한 종합 성능.

---

## 8. 핵심 인사이트

1. **Decomposition Hypothesis 실증**: "how to move"는 언어 없이 학습 가능하며, 언어는 "what to do"에만 필요 — Partial success 지표가 이를 정량 검증.
2. Inverse dynamics를 auxiliary loss나 pseudo-labeling 도구가 아닌 **독립 사전학습 단계**로 격상한 것이 기존 연구(VPT, GR-1, SMART, PACT)와의 차별점.
3. Self-exploration 데이터의 다양성이 **OOD 강건성**의 원천 — expert 데이터 스케일링만이 유일한 길이라는 통념에 도전.
4. 사전학습 규모가 finetuning 규모보다 성능 상한을 지배 (Fig. 4의 수직 gradient > 수평 gradient).

---

## 9. 한계 및 비판

- **절대 성능**: TAP-20k Avg-All 33.32%는 π0(40.08%)에 못 미치고, Carrot on plate Entire 0%, Eggplant 8.3% 등 태스크별 편차 큼. Stage 2 후에도 정밀 placement는 미해결.
- 실세계 평가가 **2개 태스크·단일 embodiment(WidowX 250s)**에 한정 — cross-embodiment 전이는 주장만 있고 실증 없음.
- Standard Setup에서는 NORA가 여전히 우위(65% vs 40%, 85% vs 75%) — in-domain에서는 대규모 expert 사전학습이 유리.
- Random play는 tabletop pushing/grasping 위주 — 관절 물체, 변형체, long-horizon 태스크로의 확장성 미검증.
- 실패의 75%가 semantic/reasoning 오류 — TAP이 해결하는 것은 물리 병목이지 추론 병목이 아님(저자 인정).
- Figure 1에는 "Fast+ Decoder"가 그려져 있으나 본문/Appendix B.1은 2-layer MLP + MSE로 기술 — 표기 불일치.

---

## 10. Open Questions

1. Random play 데이터의 스케일링 법칙은 어디까지 유지되는가 (30h → 300h)?
2. ID 목적함수를 flow-matching/diffusion action head와 결합하면 정밀 placement 문제가 완화될까?
3. Cross-embodiment: 한 로봇의 play 데이터로 학습한 물리 prior가 다른 로봇으로 전이되는가?
4. Stage 1을 video prediction 등 forward dynamics와 결합한 hybrid 목적함수의 효과는?
5. Semantic failure(75%)를 줄이기 위한 planning/reasoning 모듈과의 결합 가능성?

---

## 11. 관련 연구 비교

| 접근 | 대표 | ID의 역할 | TAP과의 차이 |
|---|---|---|---|
| 대규모 expert BC | RT-1/2, OpenVLA, π0, Nora, Octo | 없음 | 1M+ 라벨 궤적 필요; TAP은 라벨 불필요 데이터로 사전학습 |
| 명시적 dynamics 표현학습 | MIDAS, SMART, PACT | auxiliary objective | TAP은 ID를 standalone 사전학습 단계로 사용 |
| 암묵적 dynamics | Vi-PRoM, MaskDP | temporal reordering/masking | TAP은 행동 자체를 직접 회귀 |
| 비디오 기반 | VPT, GR-1 | 인터넷 비디오 pseudo-labeling / 미래 프레임 예측 | TAP은 로봇 자체 상호작용 데이터에서 물리 prior 학습 |

---

## 12. 결론 및 기여 요약

1. **Decomposition Hypothesis**: 물리 역량과 semantic 정렬의 분리 학습이라는 관점 제시 및 실증.
2. **TAP 프레임워크**: off-task 궤적 + 자율 random play → Inverse Dynamics 사전학습 → 소량 expert BC 정렬의 2단계 파이프라인, 안전한 자율 수집 알고리즘(Algorithm 1) 포함.
3. **실증**: SIMPLER에서 Standard BC 대비 +10%p (Avg-All 33.32%), 1M+ 궤적 학습 모델들과 대등/상회; 실세계에서 viewpoint 변화 시 NORA 0% vs TAP 15–25%, pushing 평균 61% vs 56%.
4. **효율**: expert 데이터 수 orders of magnitude 절감, 8×H100 ~24 GPU hours의 학계 규모 예산.

**핵심 메시지**: Embodied AI 스케일링의 다음 지렛대는 더 많은 expert teleoperation이 아니라, 영아의 motor babbling처럼 **로봇이 스스로 만들어내는 task-agnostic 물리 경험을 사전학습 자원으로 전환하는 것**이다.

<!-- VERIFIED: pdf -->
