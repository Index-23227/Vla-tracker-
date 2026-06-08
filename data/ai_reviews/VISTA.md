# VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training

> **한 줄 요약**: UMI(Universal Manipulation Interface) 손잡이형 데이터로 VLA를 학습시킬 때 발생하는 두 가지 구조적 불일치(wrist-fisheye 시각 도메인 갭, 인간 수집 궤적의 물리적 비실현성)를 진단하고, **8M UMI-VQA 보조 감독 + trajectory-level 물리 검증 파이프라인 + 2-stage 공동학습**으로 해결. π0.5에서 초기화 후 RoboTwin-UMI 0.683, LIBERO-UMI 0.943, 20개 실로봇 태스크 평균 0.598로 π0.5/LingBot-VLA/Wall-X를 일관되게 능가.

---

## 1. 배경 및 동기

### 기존 연구의 한계
- **UMI / FastUMI**: 로봇 없이 인간이 손에 든 그리퍼로 데이터 수집 → scalable하지만 두 가지 미스매치
  1. **시각 grounding 갭**: wrist-mounted fisheye (~180° FoV) → 심한 radial distortion, 균일하지 않은 spatial resolution, 그리퍼 자체의 self-occlusion → 표준 perspective 이미지에 학습된 VLM 백본의 OOD
  2. **물리적 plausibility 갭**: 인간이 수집한 궤적은 타겟 로봇의 joint limit, 충돌 geometry, 컨트롤러 bandwidth를 모름 → kinematically unreachable / collision-prone / tracking-infeasible 궤적이 학습 데이터에 그대로 들어감

### 핵심 질문
- UMI 데이터로 대규모 VLA를 학습하려면 perceptual alignment와 physical validation을 어떻게 동시에 해결할 것인가?
- 보조 VQA가 항상 도움이 되는가? (놀랍게도 **standard-view VQA는 오히려 해롭다**)

---

## 2. 방법론 심층 분석

### 2.1 전체 구조
세 가지 synergistic 컴포넌트:

1. **UMI-VQA (8M)** — wrist-fisheye 시각 분포에 정렬된 VQA 보조 감독
2. **Physical Validation Pipeline** — cross-embodiment MuJoCo replay 기반 trajectory 점수
3. **Two-stage co-training** — Stage 1: VQA + 이산 action 토큰 자기회귀 / Stage 2: knowledge-isolated continuous flow-matching expert

### 2.2 UMI-VQA 구성 (Figure 3)

| 소스 | 규모 | 비고 |
|---|---|---|
| 실제 wrist-fisheye 프레임 | 3M (5 task family) | VLM이 생성 + 인간 검증 |
| RefSpatial → fisheye 변환 | 5M | **FLUX.2-dev** 이미지 편집 모델로 semantic-aware 변환 (단순 geometric warping은 부족) |

5개 sub-task: Object Grounding 842K (27.5%), Scene Understanding 406K (13.2%), Captioning 103K (3.3%), Interaction Grounding 894K (29.1%), Spatial Reasoning 824K (26.9%)

### 2.3 Physical Validation Pipeline (Figure 4)

Cross-embodiment trajectory replay system: **MuJoCo + Mink** (Kevin Zakka IK 라이브러리). 모든 궤적이 데이터 완전성 사전점검 + 다음 세 점수로 평가:

- **Trajectory Continuity s_tc** (embodiment-agnostic): 인접 waypoint 위치/각도 변위 d에 대한 3구간 piecewise. d ≤ d_min → 100, d ≤ d_max → 선형 감점, d > d_max → 지수 감쇠. translation (d_min=5mm, d_max=45mm, d_scale=100mm), rotation (1°, 9°, 20°), α=40, β=60
- **Self-collision Risk s_sr** (embodiment-conditioned): 링크-링크 최소 거리 d_col에 대한 선형 보간
- **Execution Fidelity s_ef** (embodiment-conditioned): MuJoCo IK replay 시 desired vs achieved pose 편차

전체 점수: S(ξ, e) = 100 · (s_tc/100)^w1 · (s_sr/100)^w2 · (s_ef/100)^w3, ∑w_i = 3. 기본 uniform weights.

Pre-training 시는 **느슨한 임계값**(cross-embodiment 평균 사용), downstream fine-tuning 시는 **엄격한 임계값** (target embodiment 조건).

### 2.4 두 단계 학습

| Stage | 목적 | 방법 |
|---|---|---|
| **Stage 1** | Perception 정렬 + 이산 action 표현 | FAST 토큰화된 action + UMI-VQA 답변을 동일한 next-token CE로 공동학습. L_stage1 = -E[1/|y| ∑ log p_θ(y_j \| y_<j, x)] |
| **Stage 2** | 연속 action 정교화 | **Knowledge-isolated**: Stage-1 백본 동결. 별도 flow-matching expert f_φ를 학습. a_τ = (1-τ)ε + τa, L_fm = E ||f_φ(a_τ, τ, h_θ(o_t, l, s_t)) - (a - ε)||² |

### 2.5 추론

- VISTA 모델 초기화: **π0.5 체크포인트**
- Pre-training: 8M UMI-VQA + 100K 검증된 UMI 궤적
- Delta action 예측 + proprioceptive state conditioning
- 배포: **Zenoh** 미들웨어 기반 host-satellite 분산 추론 + temporal action ensembling

---

## 3. 데이터 전략

- **하드웨어**: FastUMI Pro (Lumos Robotics) — ~600g 경량 핸드헬드 그리퍼, 메인 fisheye + 양측 보조 fisheye + depth 카메라, Vive Tracker + VI-SLAM 융합 (sub-cm, ~3mm 정확도)
- **타겟 임베디먼트**: RealMan, AC one, Galaxea R1 Pro (dual-arm)
- **Pre-training 코퍼스**: 8M VQA + 100K 검증 궤적

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| 초기화 | π0.5 체크포인트 |
| Action head | Flow matching (Knowledge-isolated, Stage 2) |
| Action 표현 | Delta + proprioceptive state |
| Tokenizer (Stage 1) | FAST |
| 학습 데이터 | 8M UMI-VQA + 100K 검증 UMI 궤적 |
| 검증 시뮬레이터 | MuJoCo + Mink |
| VQA 변환 모델 | FLUX.2-dev |
| 배포 미들웨어 | Zenoh |

---

## 5. 실험 결과 (PDF Table 1, 2, 3, 4, 5, 6, 7, 8 직접 확인)

### 5.1 Diagnostic — Wrist-fisheye가 정책 학습을 저해 (Table 1)

| Model | LIBERO Standard | LIBERO Fisheye | RoboTwin Standard | RoboTwin Fisheye | Avg Drop |
|---|---|---|---|---|---|
| π0.5 | 96.3 | 92.2 | 82.0 | 59.4 | **13.4** |
| Wall-X | 74.6 | 70.0 | 14.9 | 15.2 | 2.2 |
| LingBot-VLA | 85.3 | 81.7 | 77.6 | 49.9 | **15.7** |

### 5.2 Diagnostic — Fisheye가 VLM의 spatial reasoning 저해 (Table 2)

| Model | Original Avg | Fisheye Avg | 상대 하락 |
|---|---|---|---|
| Qwen2.5VL-3B | 0.343 | 0.303 | ↓ 11.8% |
| Qwen3VL-4B | 0.522 | 0.498 | ↓ 4.5% |
| Embodied-R1-3B-v1 | 0.437 | 0.377 | ↓ 13.7% |
| RoboBrain2.5-4B | 0.556 | 0.522 | ↓ 6.2% |
| VLASER-2B | 0.487 | 0.443 | ↓ 9.0% |

평균 절대 4.0pt, 상대 8.6% 하락.

### 5.3 Data-level — UMI-VQA 효과 (Table 3, real-robot 3 tasks × 20 trials)

| Setting | Task 1 | Task 2 | Task 3 | Overall |
|---|---|---|---|---|
| Action-only π0.5 | 45.0% | 50.0% | 40.0% | **45.0%** |
| π0.5 + Standard-view VQA | 20.0% | 20.0% | 55.0% | **31.7%** (↓) |
| π0.5 + UMI-VQA | 40.0% | 55.0% | 70.0% | **55.0%** |

핵심 발견: **standard-view VQA는 oitherwise 잘 보이는 보조 감독이지만 wrist-fisheye 정책에서는 오히려 성능 저하**. UMI-VQA만 +10pt.

### 5.4 Data-level — Physical Score 효과 (Table 4, RealMan stapler 50 demo × 20 trial)

| Subset | Continuity | Collision | Fidelity | Avg | GSR | OSR | PSR |
|---|---|---|---|---|---|---|---|
| Low-score | 100.00 | 94.69 | 39.35 | 35.50 | 0.55 | 0.00 | 0.00 |
| High-score | 100.00 | 100.00 | 99.21 | 99.21 | 0.65 | 0.65 | 1.00 |

→ Grasping은 비슷하지만 post-grasp placement에서 low-score 0% vs high-score 100% PSR.

### 5.5 Data-level — Embodiment-conditioned (Table 5)

같은 RealMan-기준 low-score 서브셋이 R1Pro에서는 OSR 0.80을 달성. **궤적 실행성은 임베디먼트마다 다름** → embodiment-conditioned filtering의 정당성 입증.

### 5.6 Model — Main simulation (Table 6, UMI-style wrist-fisheye)

| Model | RoboTwin-UMI | LIBERO-UMI | Avg |
|---|---|---|---|
| LingBot-VLA | 0.499 | 0.817 | 0.658 |
| Wall-X | 0.152 | 0.700 | 0.426 |
| π0.5 | 0.594 | 0.922 | 0.758 |
| **VISTA** | **0.683** | **0.943** | **0.813** |

π0.5 대비 +5.5pt, LingBot-VLA 대비 +15.5pt, Wall-X 대비 +38.7pt.

### 5.7 Model — Real-robot (Table 7, 20 tasks × 20 trial)

| Model | Avg Success |
|---|---|
| LingBot-VLA | 0.313 |
| π0.5 | 0.528 |
| **VISTA** | **0.598** |

### 5.8 Component Ablation (Table 8, RoboTwin-UMI)

| Variant | Success |
|---|---|
| **VISTA (full)** | **68.3** |
| w/o Stage 2, scratch expert | 52.4 |
| w/o Stage 2, π0.5 expert | 60.2 |
| w/o state | 61.9 |
| w/o delta action | 53.1 |

핵심: (1) Stage 2 expert가 단순 π0.5 expert보다 +8.1pt, scratch보다 +15.9pt — Stage 1으로 얻은 표현 위에서 학습된 expert가 더 강함. (2) Delta action 표현이 absolute action보다 +15.2pt. (3) Proprioceptive state +6.4pt.

---

## 6. Ablation 분석

핵심 ablation은 위 §5.8에서 정리. 추가로:
- **VQA source ablation** (§5.3): standard-view vs UMI-VQA → **데이터 분포가 다운스트림 시각 분포와 일치해야 보조 감독이 효과** (단순히 "더 많은 vision-language data"가 답이 아님)
- **Score-controlled subset** (§5.4): 동일 양/동일 학습으로 점수만 다르게 → physical score가 deployment success의 predictive proxy임을 증명

---

## 7. Related Work 비교

| 측면 | π0.5 | LingBot-VLA | Wall-X | VISTA |
|---|---|---|---|---|
| Wrist-fisheye 정렬 | N/A | N/A | N/A | **UMI-VQA 8M** |
| Physical validation | hard filter (UMI 원본) | hard filter | hard filter | **continuous scoring** |
| 학습 단계 | 단일 stage | 단일 | 단일 | **Stage 1 (AR) + Stage 2 (flow-matching)** |
| Knowledge isolation | X | X | X | **O (Stage 2 시 백본 동결)** |

---

## 8. Limitations

1. **LIBERO-UMI / RoboTwin-UMI는 fisheye-adapted 변형**: 표준 LIBERO/RoboTwin 수치와 직접 비교 불가
2. **모델 파라미터 수 미공개**: π0.5 초기화 + flow-matching expert만 언급, 총 규모 N/A
3. **학습 hardware/step 미공개**: paper에 명시되지 않음
4. **실로봇 20 태스크의 세부 per-task 성공률은 Appendix D**: 본문에서는 평균만 보고
5. **임베디먼트 의존성**: physical validation이 사전 정의된 robot URDF에 의존 → 새 임베디먼트마다 Mink/MuJoCo 모델 필요

---

## 9. 종합 평가

| 항목 | 평점 (5점) | 비고 |
|---|---|---|
| **혁신성** | 4.0 | UMI-VLA 적응의 두 갭을 명시화하고 각각의 처방을 분리 검증 |
| **재현성** | 4.0 | 코드/검증 파이프라인/UMI-VQA/검증 궤적/pre-trained model 공개 약속 |
| **실험 폭** | 4.5 | Diagnostic / Data-level / Model-level 3-tier + 20 real tasks + cross-embodiment |
| **이론적 깊이** | 3.5 | Flow matching, FAST tokenization, knowledge isolation 등 기존 기법 조합 |
| **실용성** | 4.5 | UMI 데이터 자산을 가진 연구자에게 즉각적 효용 |

**총평**: UMI 데이터로 대규모 VLA를 학습할 때 "왜 그냥 안 되는가"를 정량적으로 보여주고, perception(UMI-VQA)과 physics(scoring pipeline)의 두 처방을 각각 controlled subset 실험으로 검증한 것이 강점. Standard-view VQA가 오히려 해롭다는 발견은 작지만 중요한 실용적 시사점.

---

## 10. 예상 세미나 질문

> ❓ **LIBERO-UMI 94.3은 표준 LIBERO 결과와 비교 가능한가?**
> 비교 불가. LIBERO-UMI는 wrist-only fisheye observation으로 recollect한 fisheye-adapted 버전. Table 1에서 π0.5도 standard LIBERO 96.3 → wrist-fisheye 92.2로 떨어지는 변형된 분포.

> ❓ **왜 standard-view VQA가 오히려 성능을 떨어뜨리는가?**
> 저자 가설: 백본을 공유해 학습할 때 global/regular-perspective 표현으로 편향 → wrist-fisheye action 학습에 필요한 local/distorted/gripper-centric cue를 약화. 분포 일치(distribution match)가 보조 감독의 유용성을 결정.

> ❓ **Physical score가 deployment success의 진짜 원인 변수인가? (correlation vs causation)**
> Score-controlled subset 실험(Table 4)이 데이터 양/모델/학습 과정 동일하게 통제하고 점수 분포만 분리 → 강한 인과 증거. 추가로 cross-embodiment re-scoring(Table 5)으로 score-target 의존성도 검증.

> ❓ **Knowledge isolation이 필수인가?**
> Table 8의 "w/o Stage 2, scratch expert" 52.4 vs full 68.3 → Stage 1 표현 위에서 학습된 expert가 필요. "w/o Stage 2, π0.5 expert" 60.2 → 단순히 기존 expert를 갖다 쓰는 것보다 새로 학습이 더 낫다는 결과. Stage 1을 통해 fisheye-aligned 표현이 형성된 뒤 그 위에서 expert를 학습하는 게 핵심.

> ❓ **π0.5에서 초기화했는데 그럼 본질적으로 fine-tuning인가?**
> 부분적으로 맞다. 하지만 (1) Stage-1에서 FAST 이산화 + UMI-VQA 공동학습으로 backbone을 fisheye 도메인으로 적응, (2) Stage-2에서 새로운 flow-matching expert를 처음부터 학습 — 단순 fine-tune 이상이라고 주장.

---

## 11. 코드 & 재현

- **Code / Project**: https://github.com/TeleHuman/umi-vista, https://tele-umi-vista.github.io
- **공개 약속**: physical-validation pipeline, UMI-VQA dataset, validated trajectories, pre-trained model
- **검증 도구**: MuJoCo + Mink (Kevin Zakka)
- **VQA 변환 모델**: FLUX.2-dev (Black Forest Labs)
- **배포 미들웨어**: Zenoh

---

## 12. 결론

VISTA는 UMI 데이터로 VLA를 학습할 때의 두 구조적 불일치 — wrist-fisheye 시각 도메인 갭과 인간 수집 궤적의 물리 비실현성 — 을 명시화하고, 각각에 대한 데이터 수준 처방(UMI-VQA, trajectory scoring)을 controlled 실험으로 검증한 뒤, 두 단계 학습으로 통합. RoboTwin-UMI 0.683, LIBERO-UMI 0.943, 20 실로봇 평균 0.598로 π0.5 / LingBot-VLA / Wall-X를 일관 능가. UMI 자산 + 손잡이형 데이터 수집 패러다임이 VLA 시대에 어떻게 살아남을지에 대한 실용적 청사진.

---

<!-- VERIFIED: pdf -->
