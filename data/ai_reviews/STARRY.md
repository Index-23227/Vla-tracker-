# STARRY: Spatial-Temporal Action-Centric World Modeling 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

STARRY(Tian, Jin, Yu et al., 2026, arXiv:2604.26848, Beijing Institute of Technology / Zhongguancun Academy / Zhongguancun Institute of AI / USTC / HIT / ECNU)는 robotic manipulation을 위한 spatial-temporal action-centric world model이다. 핵심 동기는 기존 VLA 정책들이 (a) text-token 중심 추론에 의존하여 정밀한 spatial-temporal coordination에 약하고, (b) 2D visual representation과 3D metric control 사이에 명확한 alignment 메커니즘이 없으며, (c) world modeling과 action generation이 분리되어 학습된다는 점이다.

핵심 아이디어:
- **Unified diffusion**: future spatial-temporal latents와 actions를 단일 diffusion process에서 jointly denoise.
- **GASAM (Geometry-Aware Selective Attention Modulation)**: 예측된 depth/end-effector geometry를 token-aligned weights로 변환, action attention을 geometry에 맞춰 modulate.
- **Hierarchical pretraining**: L1~L6의 web video → ego-centric video → synthetic/sim → real robot 데이터로 단계적 사전학습.

## 2. 아키텍처: ST World Model + Action Expert + Geometry Expert

STARRY는 4개 expert를 통합한 diffusion 기반 multi-stream Transformer다.

**Understanding Expert**: Qwen-VL backbone. multi-view RGB-D + instruction을 latent context embedding으로 인코딩.

**ST World Model**: video generation model에서 가중치를 상속한 diffusion Transformer. Hidden 3072, 30 layer, 24 head. Multi-view RGB-D observation과 end-effector trajectory를 입력으로 future spatial-temporal latents를 denoise.

**Action Expert**: ST World Model과 구조적으로 정렬된 30-layer Transformer (hidden 1024, 16 head). Action chunk를 diffusion process로 denoise하며 ST latents와 동일 timestep에서 joint training.

**Geometry Expert**: diffusion-aligned Transformer. 미래 depth map과 end-effector position을 예측하고, GASAM이 이 출력을 token-aligned weights로 변환해 Action Expert의 attention을 selective하게 조절.

## 3. 학습 데이터 및 최적화

논문은 6단계 hierarchical dataset 구성을 채택한다:
- **L1–L2**: Web / egocentric video (Ego4D, Ego-Dex) — 일반적인 시각 dynamics priors.
- **L3–L4**: Geometry-enriched synthetic / sim data (EmbodiedMAE) — depth, geometry supervision.
- **L5–L6**: Real robot datasets (DROID, BridgeData V2, OpenX) — manipulation-specific fine-tuning.

학습 목표는 (a) future ST latent denoising, (b) future depth + end-effector geometry prediction, (c) action diffusion regression의 multi-task joint loss이다. Geometry Expert와 ST World Model이 video generation pretraining 가중치로 초기화되어 시각/기하 priors를 직접 활용한다.

## 4. 핵심 실험 결과: RoboTwin 2.0

**RoboTwin 2.0 (50 bimanual tasks, fine-tuned)**:
- Clean: **93.82%**
- Randomized: **93.30%**

**Baseline 비교** (Clean / Randomized 평균):
| Model | Clean | Randomized |
|-------|-------|-----------|
| π0.5 | 62.86 | 60.30 |
| X-VLA | 72.80 | 72.84 |
| Motus | 88.66 | 87.02 |
| LingBot-VA | 92.93 | 91.55 |
| **STARRY** | **93.82** | **93.30** |

기존 SOTA였던 LingBot-VA를 Clean +0.89%p, Randomized +1.75%p 능가하며, π0.5 대비 30%p 이상의 큰 마진을 보인다.

**Real-world**: 5개 실제 task 평균 70.8% 성공률 (π0.5 baseline 42.5% 대비 +28.3%p). Sim2real 전이가 ST world modeling 덕에 안정적임을 시사.

## 5. Ablation: ST modeling + GASAM 효과

논문은 representation 종류와 GASAM 적용 여부를 교차 분석한다 (Randomized 기준):

| Representation | GASAM Off | GASAM On | Gain |
|----------------|-----------|----------|------|
| Action-only | 64.96 | 75.88 | +10.92 |
| Appearance-only | 85.80 | 86.96 | +1.16 |
| Full ST Model | 88.82 | **93.30** | +4.48 |

핵심 관찰:
- **GASAM은 모든 representation에서 일관된 향상**을 제공하나, action-only에서 가장 큰 절대 gain (+10.92).
- **Full ST modeling**이 단일 modality 대비 +3~+24%p 우위. trajectory evolution과 spatial geometry를 함께 모델링하는 것이 정밀 manipulation에 결정적.
- GASAM 단독으로도 Full ST 대비 +4.48%p — geometry-aware attention modulation이 단순 multi-task 학습보다 효과적임을 확인.

## 6. 평가 및 한계

**강점**:
- (a) RoboTwin 2.0 Clean 93.82 / Randomized 93.30으로 명확한 SOTA, 이전 SOTA LingBot-VA 능가.
- (b) Real-world 70.8% 성공률 — π0.5 대비 28%p 이상 향상으로 sim2real 우수성 입증.
- (c) GASAM이라는 명확한 mechanism: geometry/depth → token-aligned attention weights로 변환하여 2D↔3D alignment 문제를 해결.
- (d) Unified diffusion(ST latent + action 동시 denoise) 설계로 world modeling과 action prediction의 mutual reinforcement.

**약점**:
- (a) 총 parameter 수가 명시되지 않음 — 비교 fairness 평가 어려움.
- (b) LIBERO, CALVIN, SimplerEnv 등 다른 표준 benchmark 미보고 — RoboTwin 2.0 외 generalization 검증 부족.
- (c) Code/모델 공개 정보 부재(open_source=false, code_url=null) — 재현성 제약.
- (d) Diffusion 기반 unified denoising의 inference latency가 보고되지 않음 — 실시간 closed-loop 제어 적용성 미검증.
- (e) GASAM이 의존하는 depth 예측 품질이 떨어지는 환경(투명/반사 객체)에서의 robustness 추가 분석 필요.

**YAML 점검**:
- `action_head_category=diffusion` 적절 — Action Expert가 diffusion process로 학습.
- `backbone`에 ST World Model + Qwen-VL Understanding Expert 명시.
- `benchmarks.robotwin_v2.clean_avg=93.82`, `randomized_avg=93.30`은 논문 main results와 일치.
- `parameters="N/A"` — 논문이 aggregate count를 명시하지 않으므로 보수적으로 기록.

<!-- VERIFIED: pdf -->
