# HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents (Tencent)

## 1. 배경 및 동기

- 일반 VLM은 web-scale로 훈련되어 fine-grained perception과 embodied action 양쪽에 모두 부족하다. Tencent는 real-world agent의 요구를 충족하기 위한 전용 foundation model이 필요하다고 본다(§1).
- 핵심 개선 축 두 개: (1) fine-grained visual perception (physical grounding), (2) embodied prediction/interaction/planning.
- 두 variant: **MoT-2B**(2B activated / 4B total Mixture-of-Transformers, edge 배포)와 **MoE-A32B**(32B activated / 407B total, frontier reasoning).
- 22 public benchmark suite에서 MoT-2B는 16개 benchmark에서 동급 대비 best, 평균 58.0%로 Qwen3-VL-4B(+10.2pp)와 RoboBrain2.5-4B(+8.6pp)를 상회. MoE-A32B는 67.0%로 Gemini 3.0 Pro(63.6%)를 능가(§5).
- 코드 및 모델 오픈소스: https://github.com/Tencent-Hunyuan/HY-Embodied.

## 2. 방법론

- **HY-ViT 2.0** (§2.1): 400M native-resolution ViT (임의 해상도 지원), 큰 내부 ViT로부터 distillation, tiny LLM으로 language supervision, visual reconstruction supervision. 8×8 patch를 codebook size 2K의 single discrete code로 압축하여 vision 출력을 supervise.
- **Mixture-of-Transformers (MoT)** (§2.2, Fig. 2): Language/vision 각각 non-shared QKV·FFN. Vision token은 duplicated parameter로, text token은 original로 계산. Visual tokens는 bidirectional full attention, text tokens는 causal attention — modality-adaptive mask pattern(Fig. 3). Visual next-code prediction loss를 vision branch에 auxiliary supervision으로 사용.
- **Visual Latent Tokens** (§2.3): 각 visual element 끝에 learnable latent token 삽입. Pre-training에서 더 큰 ViT의 global feature로 이 token의 출력을 supervise(Global Loss in Fig. 2). Vision-language bridge 역할.
- **Pre-training** (§3): 100M+ multimodal sample — visual perception, embodied-centric, spatial-centric, general understanding.
- **Post-training** (§4): SFT → RL → **iterative self-evolving deep-thinking** (cold-start SFT + iterative RL + rejection sampling SFT, DeepSeek R1 style) → **large-to-small on-policy distillation** (MoE-A32B → MoT-2B, Thinking Machines Lab 2025 방식).
- **VLA 확장** (§6): MoT-2B base model에 **π0/π0.5 style Action Expert module**을 부착. 먼저 5K hours of UMI data로 fine-tune (per-GPU batch 32 × 32 GPU × 200K iter, 특정 embodiment 노출 X), 이후 task별 real-robot data(300-700 episode)로 SFT.

## 3. 실험 결과

**VLM benchmarks** (Table 1, MoT-2B):

| Benchmark | HY-Embodied MoT-2B | Qwen3-VL 4B | RoboBrain 2.5 4B | MiMo-Embodied 7B |
|---|---|---|---|---|
| ERQA | **54.5** | 47.3 | 43.3 | 46.8 |
| EmbSpatial-Bench | **82.8** | 80.7 | 73.8 | 76.2 |
| RoboBench-MCQ | **49.2** | 45.8 | 44.4 | 43.6 |
| VSIBench | **60.5** | 55.2 | 41.7 | 48.5 |
| MMSI-Bench | **33.2** | 25.1 | 20.5 | 31.9 |
| SITE-Bench-Image | **62.7** | 61.0 | 57.9 | 49.9 |
| ViewSpatial | **53.1** | 41.6 | 36.6 | 36.1 |

- 22 benchmark 평균 58.0% (MoT-2B), 67.0% (MoE-A32B).
- **Table 2**: MoE-A32B가 Gemini 3.0 Pro(63.6%)를 3.4pp 상회.

**Real-world robot control** (Fig. 13, 20 trials per task):

| Task | π0 | π0.5 | **HY-Embodied-0.5 VLA** |
|---|---|---|---|
| Precision Plug-in Packing | 80% | 85% | **85%** |
| Tableware Stacking | 60% | 85% | 80% |
| Mug Hanging | 45% | 50% | **75%** |

- Mug Hanging에서 π0 대비 +30pp, π0.5 대비 +25pp의 큰 차이. 5K-hour UMI pretraining이 embodiment-agnostic representation을 구축한 것으로 해석.
- Packing과 Stacking은 π0.5와 비슷하거나 약간 밑이지만, 가장 어려운 Hanging에서 결정적 우위.

## 4. 한계 및 미해결 문제

- **LIBERO/CALVIN 등 simulation benchmark 미보고**: VLA 평가는 실제 3개 task(Packing, Stacking, Hanging)만이며 20 trials/task로 표본 크기가 작다. LIBERO나 RoboTwin 같은 표준 benchmark가 없어 다른 VLA와의 apples-to-apples 비교가 어렵다.
- **A32B variant의 VLA 평가 없음**: Paper는 MoT-2B에만 Action Expert를 부착했다. 32B MoE backbone이 VLA 성능에 주는 영향이나 edge 대비 trade-off는 미제시.
- **UMI 5K hours 데이터 의존**: UMI handheld gripper demonstration 5K hours는 비공개 내부 데이터일 가능성이 크며, 외부 연구자가 동일 pretraining을 재현하기 어렵다. Open-source 정책은 칭찬할 만하나 데이터 재현성에 공백이 있다.

## 5. 총평

- **Novelty ★★★☆☆** — MoT architecture, native-resolution ViT, iterative self-evolving post-training, large-to-small on-policy distillation 각각은 기존 연구(Liang 2024 MoT, DeepSeek-R1, Thinking Machines distillation)의 조합이지만, 이를 embodied domain에 체계적으로 통합하고 2B edge 모델에 압축한 엔지니어링 완성도가 높다. Visual latent token을 큰 ViT feature로 supervise 하여 modality bridge를 형성한 설계는 영리하다.
- **Practical impact ★★★★☆** — Tencent 스케일의 open-source release(GitHub: Tencent-Hunyuan/HY-Embodied)이며 2B edge 배포가 가능한 점, Mug Hanging에서 π0/π0.5 대비 +25~30pp gain, 22 benchmark에서 전반적 SOTA를 달성한 점에서 커뮤니티 영향력이 크다. 다만 VLA 평가가 3개 task × 20 trial로 제한적이라 robotics 쪽 impact는 VLM 쪽 대비 덜 결정적이다.

## 6. 예상 질문

- **Q1. MoT가 MoE 대비 edge deployment에 더 적합한 이유는?** A. MoT는 modality마다 fixed 파라미터(dense duplicated)로 라우팅이 없어 token-level gating overhead가 없고 총 parameter 증가(2B→4B)에도 inference latency가 사실상 두 배가 되지 않는다. 반면 MoE는 expert routing이 필요하고 activation pattern이 sparse. 2B activated + 4B total은 edge-friendly한 compromise다(§2.2 "doubles the total parameter count while introducing negligible overhead").
- **Q2. 왜 VLM benchmark가 22개 씩이나 필요한가?** A. Perception(CV-Bench, DA-2K), Embodied(ERQA, RoboBench-MCQ/Planning, ShareRobot Affordance/Trajectory, Ego-Plan2), Spatial(3DSRBench, All-Angles, MindCube, MMSI, RefSpatial, SAT, SITE-Bench×2, ViewSpatial, VSIBench, Where2Place, SIBench) — embodied agent에 필요한 perception/reasoning의 각 axis를 카테고리별로 커버하기 위함. Fig. 1과 Table 1을 통해 각 축에서 SOTA 근접임을 보인다.
- **Q3. Action Expert는 π0 그대로인가, 개선이 있는가?** A. 논문은 "the Action Expert module following the structural design of π0/π0.5"라고 명시한다(§6). 즉 구조는 π0/π0.5 답습이고 차별점은 (a) base VLM이 HY-Embodied-0.5 MoT-2B라는 점과 (b) 5K hours UMI data로 먼저 대규모 fine-tune 하는 protocol이다. Action head 자체의 architectural innovation은 아니다.
- **Q4. Mug Hanging에서 큰 gain의 원인은?** A. 저자들은 "initial fine-tuning on the extensive 5K-hour UMI dataset, combined with the underlying MoT architecture"로 귀인한다(§6). Mug Hanging은 6-DoF 정확도와 contact-rich manipulation이 필요하여 fine-grained perception + large-scale embodiment-agnostic demonstration이 결정적. UMI data 5K hours라는 규모와 MoT의 modality-specific vision capacity가 이 task에 특히 유리했다고 해석된다.

<!-- VERIFIED: pdf -->
