# LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning

> **한 줄 요약**: LLaVA-1.5 기반 LMM에 **2-D visual trace 예측**을 보조 태스크로 instruction tuning하여, RLBench 12-task에서 **avg 43.3%** (Image-BC ViT 1.3% 대비) 달성하고 real-Franka에서 SOTA. CLIP ViT-L + Llama-2 7B + 8.5M OXE 이미지-trace 페어 사전훈련.

---

## 1. 배경 및 동기

- LMM (LLaVA, InstructBLIP)은 VL 태스크에서 강하나 robotics 도메인에서는 일관성이 부족.
- 기존 robot LMM들은 next-word prediction과 robot control 사이의 gap을 충분히 메우지 못함.
- **핵심 통찰**: instruction prompt에 robot type / task / 5-step proprioception을 구조화하여 입력하고, 출력으로 **2-D visual trace** + action을 동시에 예측하면 vision-action alignment 향상.
- Open X-Embodiment에서 8.5M image-trace 페어 자동 추출 (end-effector bounding box detector).

---

## 2. 방법론

### Structured Instruction Prompt
- "You are a {robot} using {control}. The task is '{task}'. The previous five steps are {state}. Predict the 2-D visual trace and next {n} action(s)."
- Robot model + control mode + task + proprioception + horizon `n` 모두 자연어 prefix.

### Visual Trace as Auxiliary Target
- End-effector center pixel을 episode 끝까지 ordered list로 출력.
- Action prediction과 jointly 디코딩 (단일 LMM 출력 시퀀스).
- Action 8-dim: 7 joint velocities + binary gripper.

### Architecture
- Base: **LLaVA-1.5** (CLIP ViT-L/14 + Llama-2 7B + linear projection).
- 사전훈련: 8.5M OXE image-trace pairs로 instruction tuning (Step 1).
- 파인튜닝: target 환경에서 추가 학습 (Step 2).
- Hardware: **8× A6000 GPUs** (PyTorch + 공식 LLaVA 코드 기반).

---

## 3. 실험 결과

### Table 1 — RLBench Multi-Task (12 tasks, 25 episodes each, 5 seeds)

| 방법 | 정보 | Avg |
|------|------|----:|
| Image-BC (CNN) | 2-D | 1.3 |
| Image-BC (ViT) | 2-D | 1.3 |
| C2FARM-BC | 3-D voxels | 22.0 |
| PerAct | 3-D voxels | 35.7 |
| **LLARVA** | **2-D only** | **43.3** |

→ 3-D voxel 정보 없이도 PerAct(3-D)을 능가.

### Table 2 — Real Franka 7-DoF

LLARVA가 RPT, Octo 대비 모든 real-world 태스크에서 가장 높은 성공률.

### Table 3 — Ablation (4 RLBench tasks)

- Instruction pretraining 추가: **+17.5pp**
- Visual trace target 추가: **+15pp**
- 둘 다 필수임을 ablation으로 입증.

### Cross-Embodiment

Sawyer로 fine-tune해도 Franka와 유사한 성능 — 단일 LMM이 robot 사이를 일반화.

---

## 4. 한계 및 미해결 문제

1. **Action 8-dim에 한정**: 7 joint velocity + 1 gripper의 단순 representation. End-effector pose나 high-DoF dexterous에는 미검증.
2. **2-D trace 의존**: Bounding box detector 품질이 trace 품질을 결정. 새 end-effector마다 detector 재훈련 필요.
3. **Single-step action prediction (n=1)**: chunking을 사용하지 않아 연속 액션의 jitter 가능성 (CoT 모델 대비 불리).
4. **CLIP ViT-L의 시공간 한계**: 복잡한 occlusion / multi-frame 추론에서 한계.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 2-D visual trace를 LMM의 보조 출력으로 사용한 최초 시도 (CoRL 2024) |
| **Practical impact** | ★★★☆☆ — 2-D-only로 3-D voxel 모델을 능가하는 점은 인상적이나, 후속 flow-matching VLA들에 비하면 dated |

**강점**: visual trace 보조 supervision이 vision-action alignment에 효과적임을 ablation으로 입증. 8 A6000으로 학습 가능한 합리적 모델 크기.
**약점**: action representation 단순. 후속 flow-matching/diffusion VLA들의 chunking과 비교 시 실시간 성능 낮음.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 2-D trace가 3-D voxel만큼 효과적인가? | LMM이 이미 2-D representation에 강하므로 추가 supervision이 효율적. 3-D voxel은 별도 인코더가 필요해 LMM의 pretrained 강점을 살리기 어려움. |
| 2 | OXE의 8.5M trace는 어떻게 만들었나? | OXE 37개 서브셋의 13개 robot/end-effector마다 별도로 학습된 bounding box detector → center point. Detector 품질 의존. |
| 3 | n=1만 예측하는 이유? | LMM의 length budget을 visual trace에 할당하기 위해. action chunking은 후속 연구 영역. |
| 4 | OpenVLA와의 차이는? | OpenVLA는 action token 직접 디코딩, LLARVA는 visual trace를 함께 디코딩. LLARVA가 RLBench multi-task에서 우위, OpenVLA는 OXE-scale 일반성에서 우위. |

<!-- VERIFIED: pdf -->
