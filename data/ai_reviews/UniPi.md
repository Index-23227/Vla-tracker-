# UniPi: Learning Universal Policies via Text-Guided Video Generation

> **한 줄 요약**: 순차적 의사결정 문제를 text-conditioned video generation으로 재정의(UPDP, Unified Predictive Decision Process)하고, 생성된 frame sequence에서 inverse dynamics로 action을 추출하는 **video-as-policy** 패러다임의 선구적 연구. CLIPort/조합 생성 task에서 baseline 대비 큰 폭의 향상을 보고 (NeurIPS 2023, Du et al.).

---

## 1. 배경 및 동기 (Sec. 1, 2)

- 환경마다 state/action space가 달라 일반화가 어려움 → **이미지를 universal interface, 텍스트를 universal task spec**으로 사용.
- MDP의 reward 정의/state 정의 한계를 회피하기 위해 **UPDP** $G = \langle \mathcal{X}, \mathcal{C}, H, \rho\rangle$를 제안 (Sec. 2.2). $\rho(\cdot|x_0,c)$는 첫 프레임/텍스트 조건 video diffusion.
- Diffusion-based video planner + 학습된 inverse dynamics model이 핵심 두 모듈 (Fig. 2).

---

## 2. 방법론 (Sec. 3)

### Universal Video-Based Planner
- Imagen-Video 기반 cascaded video diffusion 사용 (Sec. 3.1, Appendix A.1).
- **First-frame conditioning**: 학습 시 첫 프레임을 explicit context로 추가 → 단순 fixed-frame trick보다 안정적.
- **Trajectory consistency via tiling**: noise denoising 시 첫 frame을 모든 timestep에 channel-wise concat → 환경 상태 유지.
- **Hierarchical planning**: coarse temporal sparse video → super-resolution으로 fine-grained 시간 보간.
- T5-XXL(4.6B) text encoder, 1.7B video diffusion + 1.7B temporal SR + 1.4B/1.2B real-world cascades, 256 TPU-v4 (Appendix A.1).

### Inverse Dynamics (Sec. 3.2)
- 3×3 conv stack + residual + mean pool + MLP(128, 7) → **7-dim continuous robot control**을 MSE로 회귀.
- Adam, lr 1e-4, gradient clip 1, 2M steps. Planner와 **decoupled**으로 학습 → 작은 sub-optimal dataset만으로 가능.
- Open-loop control 사용 (Sec. 3.2).

---

## 3. 실험 결과 (PDF 수치 인용)

### 조합 생성 task (Table 1, ± std)
| Model | Seen Place | Seen Relation | Novel Place | Novel Relation |
|---|---|---|---|---|
| State+Transformer BC | 19.4±3.7 | 8.2±2.0 | 11.9±4.9 | 3.7±2.1 |
| Image+Transformer BC | 9.4±2.2 | 11.9±1.8 | 9.7±4.5 | 7.3±2.6 |
| Image+TT | 17.4±2.9 | 12.8±1.8 | 13.2±4.1 | 9.1±2.5 |
| Diffuser | 9.0±1.2 | 11.2±1.0 | 12.5±2.4 | 9.6±1.7 |
| **UniPi** | **59.1±2.5** | **53.2±2.0** | **60.1±3.9** | **46.1±3.0** |

→ 모든 baseline 대비 **3-5× 향상**, novel task에도 동등한 성능 (combinatorial generalization).

### Multi-task generalization (Table 3, CLIPort suite)
| Model | Place Bowl | Pack Object | Pack Pair |
|---|---|---|---|
| Diffuser | 14.8±2.9 | 15.9±2.7 | 10.5±2.4 |
| **UniPi** | **51.6±3.6** | **75.5±3.1** | **45.7±3.7** |

### Ablation (Table 2): Frame Condition + Consistency + Hierarchy 모두 활성화 시 53.2 (Relation seen). 모두 끄면 12.4. → 세 요소가 모두 필수.

### Real-world transfer (Bridge dataset, Table 4)
| Variant | CLIP Score ↑ | FID ↓ | FVD ↓ | Success ↑ |
|---|---|---|---|---|
| No Pretrain | 24.43 | 17.75 | 288.02 | 72.6% |
| **Pretrain (14M video-text + 60M image-text + LAION-400M)** | 24.54 | **14.54** | **264.66** | **77.1%** |

→ Internet-scale pretraining으로 video plan 품질 및 surrogate task success 향상.

---

## 4. 한계 (Sec. 6)

1. **Latency**: Video diffusion이 매우 느림 (논문에서 "1분 가량"); 진보적 distillation으로 16× 가속 실험 보고했으나 여전히 real-time과 거리.
2. **Open-loop control**: Sec. 3.2에서 efficiency를 위해 open-loop 사용 → 환경 변화 대응 부족.
3. **Partial observability**: occluded 환경에서는 hallucination 위험을 명시 (Sec. 6).
4. **Inverse dynamics의 정확도** 가 video quality에 강하게 의존.

### YAML 정합성 검토
- YAML `venue: ICML 2023` → **PDF는 NeurIPS 2023** ("37th Conference on Neural Information Processing Systems (NeurIPS 2023)" line 60). **불일치 → 수정 권장**.
- YAML `dataset: Language Table, CLIPport` → PDF는 combinatorial planning (PDSketch [23]) + CLIPort + Bridge real-world (Sec. 4) 사용. CLIPort는 일치하지만 Language Table이 아닌 PDSketch 환경. **부분적 부정확**.
- `benchmarks: {}` 비어있음은 paper 결과가 표준 LIBERO/CALVIN/SimplerEnv가 아닌 자체 task이므로 **적절**.

---

## 5. 총평

**강점**: Video-as-policy 패러다임을 처음으로 정식화 (UPDP); 후속 SuSIE/HiP/RoboDreamer/UniSim 모두 본 작업의 직계. **약점**: Latency 본질적 한계, action space는 학습 단순화를 위해 7-DoF로 한정.

---

## 6. 예상 날카로운 질문

| # | 질문 | 답 (논문 근거) |
|---|------|----------------|
| 1 | Closed-loop이 아닌 open-loop인 이유? | Sec. 3.2: 비디오 diffusion이 느려 closed-loop은 비현실적. 단, MPC 형태 closed-loop도 framework 상 가능. |
| 2 | 왜 비디오 전체를 만들고 inverse dynamics로 action을 추출하나? Action을 직접 diffusion하는게 더 효율적이지 않나? | Diffuser baseline (Table 1: 9-12%)이 UniPi(53-60%)보다 훨씬 낮음. **비디오 모달리티가 cross-environment 전이에 유리**하다는 실증. |
| 3 | 14M video-text 데이터의 기여도? | Table 4: surrogate success 72.6→77.1% (+4.5pp); FID 17.75→14.54로 개선. 정량적으로는 modest이나 **novel task** 에서 큰 차이 (Fig. 8 정성 결과). |

<!-- VERIFIED: pdf -->
