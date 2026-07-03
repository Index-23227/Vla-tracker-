# DAI-3D: Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization

> **한 줄 요약**: 동일한 2D grounding 포인트가 주어졌을 때 text/visual prompt로 넣으면 미미한 개선에 그치지만, depth로 3D로 리프팅해 gripper 상대 변위를 2-layer MLP로 인코딩해 action head의 AdaLN에 직접 주입하면 LIBERO-PRO에서 GR00T-N1.6이 Task Perturbation 31.2→77.5, Position Perturbation 28.1→60.2로 뛰는, "표현과 주입 방식이 진짜 병목"임을 보인 backbone-agnostic 모듈.

> **명칭 주의**: 논문 본문에 방법의 고유 명칭(acronym)이 없다. 저자들은 일관되게 "Ours" 또는 "3D via AdaLN"으로 지칭하며, DAI-3D는 제목(Direct Action-head Injection of a grounded 3D point)에서 유래한 트래커 부여 식별자다.

---

## 1. 배경 및 동기

- VLA는 대규모 vision-language 사전학습으로 자유형 지시 이해 능력을 얻지만, 테스트 시점에 두 축으로 취약: **spatial generalization**(학습과 다른 물체 위치)과 **task generalization**(익숙한 장면 + 다른 지시). LIBERO-PRO [Zhou et al., 2510.03827]가 이 취약성을 체계적으로 노출.
- 기존 대응은 외부 VLM/detector의 2D 좌표를 **language prompt**나 **visual prompt**로 전달하거나(RoboPoint, MOKA, VP-VLA, MolmoBot 계열), VLA 자체가 chain-of-thought로 spatial 출력을 내게 학습(MolmoAct, ECoT, CoA-VLA).
- 저자들의 핵심 관찰(Table 2): **동일한 oracle 2D 포인트**를 줘도 표현/주입 방식에 따라 성능이 극단적으로 갈린다. 즉 grounding 신호의 "출처"가 아니라 **표현(2D vs 3D)과 주입 경로(prompt vs action head)**가 진짜 병목.
- 직관: action 예측은 3D 물리 공간에서 일어나므로, 2D 신호는 정책이 내부적으로 2D→3D 기하를 재구성해야 하는 부담을 지운다. 3D 신호를 action head에 직접 주면 이 부담이 제거된다.

## 2. 문제 정의

- 입력: 임의의 off-the-shelf grounding 소스가 주는 2D 타깃 픽셀 (u, v) + aligned depth 이미지 D + 카메라 intrinsic K / extrinsic (R, t) + proprioception(gripper 위치 p_g).
- 목표: 사전학습된 VLA backbone과 학습 objective를 **전혀 바꾸지 않고**, 이 grounding 신호를 어떻게 표현·주입해야 spatial/task 일반화가 열리는지 규명하고 최소 형태의 모듈로 구현.
- 적용 범위: 각 sub-goal이 타깃 물체/영역과 연결되는 태스크(pick-and-place: reach-to-grasp → 물체, transport-to-place → placement 영역).

## 3. 핵심 기여

1. **통제 실험을 통한 진단**: 동일 oracle 2D 포인트 하에 2D-text / 2D-visual / 2D-AdaLN / 3D-text / 3D-AdaLN 5개 변형을 비교, "3D 표현 + action head 직접 주입"의 조합만이 일반화를 연다는 것을 분리 검증(Table 2).
2. **초경량 모듈**: 추가 구성요소는 2-layer MLP(hidden/output 1024) 하나. 기존 AdaLN linear projection을 그대로 재사용하므로 action head 내부 추가 파라미터 0.
3. **Zero-init 안정화**: MLP 2번째 layer를 zero-init하여 step 0에서 z_spatial = 0 → 사전학습된 timestep conditioning을 정확히 보존한 채 fine-tuning 시작.
4. **Backbone-agnostic 검증**: GR00T-N1.6과 π0.5 양쪽에서 대폭 개선 + Franka 실로봇에서 Qwen3-VL-4B zero-shot grounding / 소비자용 RGB-D로도 유효함을 확인.

## 4. 방법론

### 2D → 3D 리프팅 (Sec. B.1)
z = D(u, v)로 metric depth를 읽고 pinhole 모델로 카메라 프레임 back-projection: p_t^cam = z·K⁻¹[u, v, 1]ᵀ. Extrinsic 적용으로 robot base frame의 p_t = R·p_t^cam + t. Gripper 위치 p_g는 forward kinematics로 획득.

### Spatial Embedding
3D 상대 변위 d = p_t − p_g를 ReLU 2-layer MLP로 인코딩: z_spatial = MLP(d) ∈ R^{d_h} (GR00T-N1.6, π0.5 모두 d_h = 1024). 절대 좌표가 아닌 변위만 쓰는 것이 핵심 inductive bias(scene-invariant, Table 3 참조).

### AdaLN 주입 (Sec. B.2)
DiT action head의 기존 AdaLN이 γ, β = Linear(z_time)로 조건화하던 것을 γ, β = Linear(z_time + z_spatial)로 확장(element-wise 합). MLP 2번째 layer zero-init으로 초기에는 사전학습 거동과 완전 일치, 학습이 진행되며 spatial 신호가 점진적으로 반영.

### Sub-goal 전환
에피소드 시작 시 타깃 물체의 z좌표를 기록, 1cm 이상 상승하면 grasp로 판정하고 타깃을 물체 → placement 영역으로 전환(GraspVLA 방식 차용).

## 5. Action Head Category

**flow_matching**. 방법 자체는 head를 새로 만들지 않고, GR00T-N1.6 / π0.5의 기존 DiT 기반 flow-matching action head의 AdaLN conditioning 경로에 3D spatial embedding을 합산 주입한다. Fig. 2에도 "Flow Matching Action Head (Condition on Spatial Emb. via AdaLN)"로 명시. Diffusion timestep embedding 기반 AdaLN을 쓰는 어떤 DiT head에도 이식 가능하다는 점에서 모듈은 head-agnostic이다.

## 6. 실험 설정

- **벤치마크**: LIBERO(Object, Spatial 두 pick-and-place suite) + LIBERO-PRO(Task Perturbation: 지시의 타깃을 장면 내 다른 물체로 교체 / Position Perturbation: 초기 배치 변경). 태스크당 50 rollout.
- **학습 데이터**: LIBERO-Object + Spatial + LIBERO-90에서 뽑은 pick-and-place 49개 = 총 69 태스크(마스크 획득 가능성이 선정 기준). Sec. 4.4에서는 원래 20-task 설정도 재학습해 비교.
- **Grounding**: 시뮬레이션은 oracle — 시뮬레이터 segmentation mask에서 distance transform으로 최내부 픽셀 샘플링(단, 반드시 카메라 가시 2D 픽셀을 depth로 리프팅; 시뮬레이터 3D centroid 직접 사용 금지 invariant 유지). 추상 영역("접시 왼쪽")은 3D centroid를 2D로 투영 후 재리프팅.
- **학습 레시피**: GR00T-N1.6 — batch 384, 20k step, AdamW lr 1e-4 cosine(warmup 5%), 16-step action chunk. π0.5 — batch 256, 30k step, lr 5e-5, 10-step chunk. 둘 다 공식 레시피 기반.

## 7. 핵심 결과 — LIBERO / LIBERO-PRO (Table 1)

| Method | LIBERO ID Avg | Task Pert. Avg | Position Pert. Avg |
|---|---|---|---|
| OpenVLA | 86.6 | 24.1 | 0.0 |
| π0 | 97.8 | 32.2 | 5.9 |
| MolmoAct | 91.2 | 20.4 | 2.2 |
| π0.5 | 95.4 | 37.3 | 47.0 |
| **π0.5 + Ours** | **97.6** | **75.9** (+38.6) | **72.2** (+25.2) |
| GR00T-N1.6 | 93.5 | 31.2 | 28.1 |
| **GR00T-N1.6 + Ours** | **96.0** (Object 99.6 / Spatial 92.4) | **77.5** (+46.3) | **60.2** (+32.1) |

같은 데이터로 학습한 baseline 대비 in-distribution을 유지·소폭 개선하면서 perturbation에서 2배 이상. 두 backbone에서 일관 → backbone-agnostic.

## 8. 통제 비교 — 표현 × 주입 분리 (Table 2, GR00T-N1.6, 69-task)

| Variant | Task Pert. Avg | Position Pert. Avg |
|---|---|---|
| Baseline | 31.2 | 28.1 |
| w/ 2D via Text Prompt | 42.5 | 31.7 |
| w/ 2D via Visual Prompt | 40.3 | 30.4 |
| w/ 2D via AdaLN | 47.7 | 37.8 |
| w/ 3D via Text Prompt | 38.5 | 39.4 |
| **w/ 3D via AdaLN (Ours)** | **77.5** | **60.2** |

두 가지 교훈: (1) **2D 신호는 주입 방식과 무관하게 근본적 한계** — 같은 AdaLN 주입이라도 2D는 47.7에 그침(2D→3D 매핑을 정책이 내부 학습해야 하는 부담). (2) **3D도 text prompt로 넣으면 무용** — VLM은 수치 3D 좌표 해석 prior가 약해 기하 구조가 head에 도달하기 전에 소실. 즉 3D 표현과 직접 주입은 **상보적 필요조건**.

추가로 Sec. 4.4(Fig. 3): 20-task로 학습한 Ours(62.0/41.5)가 69-task로 학습한 다른 모든 변형(최고 47.7/39.4)을 능가 — **설계 선택이 데이터 스케일링보다 강한 레버**.

## 9. Ablation & 강건성

- **입력 성분(Table 3, 4-suite avg)**: p_t 단독 34.3 < p_t+p_g 53.7 < **d 단독 68.9** > (p_t, p_g, d) 59.7. 절대 좌표는 장면 간 변동(테이블 높이 등)으로 전이가 어려워, 변위 d만 쓰는 것이 scene-invariant 신호로 최적.
- **주입 경로(Table 4)**: AdaLN 68.9 vs robot state에 concat해 self-attention으로 흘리기 51.7 vs baseline 29.7 — 블록별 normalized feature를 직접 변조하는 AdaLN이 우월.
- **Depth 노이즈(Table 7)**: 학습 시 σ=0.01/0.03m Gaussian 노이즈를 넣으면 68.9→61.3/57.2로 완만히 하락하나 여전히 baseline(29.7) 대비 2배. 테스트 시점 노이즈에는 세 변형 모두 사실상 무감(예: clean-train에서 clean 68.9 vs σ=0.03 70.2).

## 10. 실로봇 결과 (Franka Panda, Tables 8-10)

8개 pick-and-place 태스크(4 layout × 2), 태스크당 40 demo, 조건당 10 rollout. Grounding은 **fine-tuning 없는 Qwen3-VL-4B**가 JSON으로 target/placement 2D 포인트를 동시 출력, depth는 RealSense D435.

| Method | In-Dist. | Task Pert. | Position Pert. |
|---|---|---|---|
| GR00T-N1.6 | 68.8 | 3.8 | 3.8 |
| w/ 2D text | 87.5 | 5.0 | 7.5 |
| w/ 2D visual prompt | 88.8 | 6.3 | 5.0 |
| **Ours** | **92.5** | **63.8** | **66.3** |

Perturbation 하에서 baseline과 2D 변형은 사실상 전멸(≤7.5), Ours만 성능 유지. Oracle이 아닌 노이즈 많은 VLM grounding + 소비자 depth 센서로도 유효 → 실용성 입증. 단 Task 3(orange→plate)는 Task Perturbation에서 0/10으로 실패 사례 존재.

## 11. 한계 & 토의

1. **Sub-goal-타깃 연결 가정**: 각 sub-goal이 물체/영역과 연결돼야 함 — wiping 같은 object-agnostic 모션은 정식화 밖(저자 제안: 2D 궤적을 3D로 리프팅해 유사 주입, future work).
2. **적절한 2D grounding 가용성 가정**: 복잡한 장면/지시에서의 robust grounding은 본 기여와 직교하는 open problem으로 남김. 시뮬레이션은 oracle mask라 grounding 오류의 영향이 과소평가될 수 있음.
3. **단일팔 한정**: dual-arm/humanoid 확장 미검증.
4. **Depth 의존**: RGB-D 센서 또는 depth 추정이 전제. 노이즈 강건성은 보였으나 depth 자체가 없는 세팅은 다루지 않음.
5. LIBERO-PRO Spatial suite의 Position Perturbation은 41.0으로 개선폭이 상대적으로 작음(+14.0) — 잔여 병목 분석은 부족.

## 12. VLA-Tracker 관점에서의 의의

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — 3D 정보의 action-head 주입 자체는 PointVLA·FALCON 계열이 선행하나, "표현 × 주입"을 oracle 통제 하에 체계 분리한 진단과 2-layer MLP + zero-init AdaLN이라는 최소 형태가 기여 |
| **Rigor** | ★★★★☆ — 동일 grounding 입력 통제, 20/69-task 스케일 비교, depth 노이즈 sweep, 입력·주입 ablation, 실로봇 3조건까지 체계적 |
| **Practical impact** | ★★★★☆ — backbone 무변경·모듈 하나로 GR00T-N1.6/π0.5 모두에서 LIBERO-PRO 30-46점대 개선; DiT+AdaLN head가 사실상 표준이 된 현 세대 VLA에 즉시 이식 가능 |

이 논문의 메시지는 "grounding을 더 잘하라"가 아니라 **"이미 있는 grounding을 올바른 좌표계(3D)로, 올바른 위치(action head)에 넣어라"**다. Table 2의 5-way 통제 비교는 프롬프트 기반 grounding 전달(RoboPoint/MOKA 스타일)이 왜 기대만큼 일반화를 못 여는지에 대한 드문 정량 답변이며, LIBERO-PRO를 주 평가축으로 삼은 점도 memorization 논란 이후의 평가 관행 전환을 반영한다. 트래커 등록 기준으로는 GR00T-N1.6 + Ours의 LIBERO Object 99.6 / Spatial 92.4가 대표 점수이나, 이 논문의 가치는 leaderboard 수치보다 진단적 발견에 있다. 코드는 출판 시 공개 예정(현재 미공개).

<!-- VERIFIED: pdf -->
