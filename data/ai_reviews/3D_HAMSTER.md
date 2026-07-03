# 3D HAMSTER: Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance

> **한 줄 요약**: 계층형 VLA에서 상위 플래너(VLM)가 2D 궤적 대신 **메트릭 3D 궤적 (u, v, d)** 를 직접 출력하도록 만들어, 포인트클라우드 기반 하위 정책(3DFA)과 **동일한 3D 메트릭 공간**에서 계획-실행을 정렬. 전용 depth encoder + dense depth reconstruction loss + 2D/3D/2D→3D 혼합 데이터로 Qwen3-VL-8B를 학습해 DroidSpatial-Bench에서 Gemini-3.0-Pro·RoboBrain-2.5를 능가하고, Colosseum 평균 44.8%(2D 가이던스 대비 +6.0%p), 실제 Franka Panda 3개 태스크에서 최고 평균 성공률(80/68/62%) 달성. KAIST(DAVIAN) 연구, IROS 2026.

---

## 1. 배경 및 동기

- 계층형 VLA는 상위 VLM 플래너(System 2)가 시각 목표를, 하위 컨트롤러(System 1)가 모터 명령을 담당하도록 분리 — 플래너가 로봇 특화 액션이 아닌 시각적 타깃을 예측하므로 비-로봇 데이터(spatial reasoning, grounding, VQA)로 학습 가능해 VLM의 일반화를 보존
- 그러나 근본적인 **표현 불일치(representational misalignment)** 존재: HAMSTER(ICLR 2025) 등 기존 방식은 플래너가 **2D 픽셀 좌표** 궤적을 출력하는 반면, 최신 하위 정책(DP3, Act3D, RVT-2, 3DDA, 3DFA)은 **포인트클라우드 위 3D 메트릭 공간**에서 동작
- 2D 웨이포인트를 3D로 리프팅하려면 각 픽셀 아래 장면 표면의 depth를 샘플링해야 함 → 궤적이 자유 공간을 통과하지 못하고 장면 표면에 달라붙는 **"graffiti effect"** 발생. 컨트롤러는 의도된 경로와 장면 기하를 구분하기 어려움
- 해결책: 플래너가 처음부터 3D 좌표를 출력하면 투영이 불필요 — 궤적이 컨트롤러와 같은 메트릭 공간에 존재. 최근 VLM(Qwen3-VL의 3D bbox 예측, G2 VLM 등)이 대규모 사전학습으로 상당한 3D 공간 지식을 획득했다는 점이 이를 가능하게 하는 토대

## 2. 방법론 심층 분석

### 2.1 문제 정식화 (§III-A)
- 입력: 보정된 RGB-D 카메라의 RGB 이미지 I, depth map D, 언어 지시 l
- 상위 플래너: end-effector 궤적 τ = {(u_t, v_t, d_t)}_{t=1..T} 예측 — (u, v)는 이미지 평면 좌표(0–1000 정규화), d는 미터 단위 메트릭 depth
- 하위 정책 π_low(P, τ): 포인트클라우드 관측 P와 τ를 조건으로 T_a개 액션 청크 생성, 리플래닝 스텝마다 폐루프 실행

### 2.2 3D Trajectory Planner (§III-B)
- 백본: **Qwen3-VL-8B-Instruct** — 3D bbox 예측 등 사전학습된 3D 지식 보유. 단, 그대로는 3D 웨이포인트 시퀀스 생성에 거의 0% 정확도 (Table II 첫 행: 5cm Both 0.7%)
- **학습 데이터 (Table I)**: 8개 소스, 2개 범주
  - *3D capability data (RGB-D)*: RLBench 606K, DROID 123K, InternData-M1 1.5M — 각각 2D-only / 3D-only / 2D→3D chain-of-thought 3가지 감독 변형으로 픽셀-depth 일관 매핑 학습. RefSpatial 2.2M(생성 depth 부가, depth-aware spatial reasoning + vacant-space localization 확장). GT depth는 RLBench·DROID, InternData-M1·RefSpatial은 MoGe-2로 메트릭 depth 생성
  - *Preservation data (RGB-only)*: RoboPoint 666K, PixMo 171K, LVIS 138K, Honey-1M 749K — depth 경로를 우회해 기존 vision-language 능력이 덮어써지는 것을 방지
- **Depth encoder**: RGB feature만으로는 단일 뷰에서 메트릭 depth 복원 불가 → 별도 초기화된 depth encoder(LingBot-Depth에서 초기화)가 depth 토큰 생성, 전용 프로젝터로 LLM 임베딩 공간에 투영 후 RGB 시각 토큰과 element-wise 융합
- **Dense depth reconstruction loss**: 궤적이 텍스트 토큰이라 autoregressive LM loss만으로는 gradient가 희소 → depth 토큰 z_D를 경량 디코더 f_dec에 통과시켜 전체 depth map 복원, L_depth = ||D − D̂||₁. 총 손실 L = L_LM + λ·L_depth (λ=0.1) — 장면 수준 기하 prior로 waypoint 간 depth drift 억제
- **2단계 학습**: Stage 1(depth alignment) — RGB encoder·depth encoder·LLM 동결, depth projector + decoder만 학습. Stage 2(task fine-tuning) — 양 encoder 동결, LLM에 LoRA(rank 64), 프로젝터·디코더는 전체 학습. 두 스테이지 모두 1 epoch (lr 1e-4, warmup 0.03, batch 256, 8×H100)

### 2.3 Trajectory-conditioned 3D Low-level Policy (§III-C)
- 백본: **3DFA (3D FlowMatch Actor)** — rectified flow matching 기반 포인트클라우드 정책
- 궤적 unprojection: p_cam = d·K⁻¹[u, v, 1]ᵀ, p_world = R·p_cam + t_cam (식 3)
- **Trajectory-Scene Fusion**: 월드 좌표 궤적을 장면 포인트클라우드에 append — 가이던스와 관측이 통합된 3D 표현에 공존. 각 웨이포인트는 시간 순서에 따라 색상 코딩, 학습 가능한 modality embedding(e_traj vs e_scene)을 점 feature에 가산(식 4)해 궤적 추종/장면 이해 전략을 구분 학습
- 서브샘플링 시 우선순위 최하위 장면 점을 궤적 점으로 교체해 **가이던스 신호가 절대 소실되지 않도록 보장**
- 액션 예측: rectified flow matching velocity field + 그리퍼 BCE (식 5), 청크 길이 T_a=20. 시뮬 500k steps(태스크당 100 demos), 실제 300k steps(pick-and-place 300 / pouring 144 / button pressing 108 에피소드, 4×H100)

## 3. 실험 설정

- **RQ1 (3D 궤적 예측)**: 자체 구축 **DroidSpatial-Bench** — held-out DROID pick-and-place 148 에피소드. grasp/placement 지점이 GT의 δ∈{5, 10}cm 이내인지로 Start/End/Both 정확도 측정. 비교: Sonnet-4.6, GPT-5.2, Gemini-3.0-Pro(RGB), RoboBrain-2.5-8B(RGB). RoboTracer는 미공개로 제외
- **RQ2 (시뮬레이션 강건성)**: **Colosseum** — RLBench 확장, 14개 섭동 축(객체 외관, 장면 맥락, 카메라 포즈, 종합). HAMSTER를 따라 전면 카메라 가시 11개 태스크, 섭동당 25 에피소드. 모든 방법이 동일 데이터로 학습된 동일 3DFA 정책 공유 → 가이던스 신호 효과만 분리
- **RQ3 (실제 로봇)**: Franka Panda + 외부 RGB-D 카메라, 3개 태스크군(버튼 누르기 3버튼 / 붓기 2컵·3보울 / pick-and-place 10물체·3보울). 4개 일반화 축: 언어(미학습 동의어), 공간(상대적 언어 참조·미학습 물체 높이), 시각(새 텍스처·조명·distractor), 복합. 이진 성공 대신 **[0,1] 연속 점수**(4단계 각 25% 또는 2단계 각 50%), 총 25 rollout 평균. π0.5(동일 데모로 50k steps 파인튜닝) 추가 비교

## 4. 핵심 결과 — 3D 궤적 예측 (Table II)

| Model | Input | 5cm Both | 10cm Both |
|---|---|---|---|
| Sonnet-4.6 | RGB | 0.7 | 2.0 |
| GPT-5.2 | RGB | 2.7 | 16.2 |
| Gemini-3.0-Pro | RGB | 16.2 | 29.7 |
| RoboBrain-2.5-8B | RGB | 39.2 | 60.1 |
| Qwen3-VL-8B (base) | RGB | 0.7 | 0.7 |
| + 3D Traj. Data | RGB | 27.7 | 50.0 |
| + Depth encoder | RGBD | 42.6 | 62.8 |
| + L_depth (**full**) | RGBD | **41.9** | **65.5** |

- 프로프라이어터리 VLM은 강력한 일반 능력에도 메트릭 depth 추론 부재 — 최고인 Gemini-3.0-Pro도 5cm Both 16.2%
- 3D HAMSTER는 RoboBrain-2.5-8B를 전 지표에서 능가, 특히 End-position에서 격차 최대 (5cm: 66.2 vs 58.1 / 10cm: 82.4 vs 74.3) — 배치 성공에 depth 정확도가 가장 중요한 지점
- 컴포넌트별 기여: 데이터 (0.7→27.7), depth encoder (27.7→42.6), L_depth는 10cm End 75.0→82.4로 시퀀스 후반 depth drift를 교정 (5cm Both는 42.6→41.9로 소폭 하락하나 10cm Both 62.8→65.5 상승)

## 5. Colosseum 강건성 결과 (Table III)

| Method | None | Avg. (14축) |
|---|---|---|
| 3DFA (no guidance) | 53.8 | 36.6 |
| 3DFA + HAMSTER (2D) | 49.5 | 38.8 |
| 3DFA + 3D HAMSTER | **62.9** | **44.8** |

- **2D 가이던스는 무섭동(in-distribution) 성능을 오히려 저해** (53.8→49.5) — 2D 픽셀 좌표로의 투영이 3D 컨트롤러에 기하학적 모호성 주입. 3D 가이던스는 무섭동도 62.9%로 개선
- 3D의 최대 이득 축: 조명 +15.6%p, manipulated object +11.3%p, 배경 텍스처 +8.7%p — 2D 플래너가 암묵적 depth 추론에 의존하는 색/텍스처 특징을 직접 훼손하는 축들. MO/RO는 크기(기하)까지 변화시키는데도 3D가 우세 → 외관 불변성 이상의 기하 안정성
- 한계 조건: All Var.(전 섭동 동시)에서는 2D/3D 모두 7.2%로 수렴 (무가이던스 0.8%보다는 크게 우위) — 심한 시각 손상으로 VLM의 타깃 grounding 자체가 무너지면 가이던스 차원은 무의미

## 6. 실제 로봇 결과 (Table IV)

| Method | Button | Pouring | Pick-and-Place |
|---|---|---|---|
| π0.5 | 74 | 41 | 40 |
| 3DFA | 38 | 50 | 30 |
| 3DFA + HAMSTER | 60 | 45 | 46 |
| 3DFA + 3D HAMSTER | **80** | **68** | **62** |

- π0.5는 in-distribution에서 강력(버튼·P&P 100%)하나 분포 이동 시 급락(공간 변형에서 pouring 15%, P&P 15%) — 모놀리식 VLA의 학습 조건 과적합 확인
- 3D vs 2D 격차 최대 축: 시각 이동(버튼 100 vs 80, pouring 65 vs 35), 공간 이동(버튼 50 vs 20, pouring 65 vs 40)
- 최대 전체 이득은 pouring (68 vs 45) — 컵 기울일 때 정밀 높이 제어 요구, 3D 웨이포인트가 직접 인코딩하는 기하 요건
- Fig. 5 정성 분석: HAMSTER 2D 궤적은 View 1에서는 그럴듯하나 View 2에서 graffiti effect 노출(표면에 고정) → 실패; 3D HAMSTER 궤적은 양 시점에서 메트릭 일관, 장면 표면과 분리된 자유 경로 → 성공

## 7. 정성 분석 — 분포 외 일반화 (Fig. 4)

- DroidSpatial-Bench에서 RoboBrain 2.5·Gemini 3.0 Pro 예측은 View 1에선 합리적으로 보이나 두 번째 시점에서 큰 depth 오차 노출; 3D HAMSTER는 양 시점 모두 메트릭 일관
- 학습에 없던 실제 장면 zero-shot 평가에서 격차 확대: 베이스라인 궤적은 공간 일관성을 완전히 상실, 3D HAMSTER는 추상 언어("와인을 만드는 과일을 흰 보울에"), 클러터, 조명 변화에서도 기하학적으로 그럴듯한 경로 유지 — depth 증강 플래너의 메트릭 3D 추론이 학습 분포 밖으로 일반화

## 8. 강점

1. **문제 정의의 명료함**: "graffiti effect"라는 2D→3D 리프팅의 실패 모드를 구체적으로 규명하고, 계획-실행의 메트릭 공간 통일이라는 원리적 해법 제시
2. **컴포넌트별 기여가 깨끗하게 분리된 ablation**: 데이터/encoder/loss 각각의 정량 기여(Table II)와 가이던스 신호만 분리한 통제 비교(동일 3DFA, Table III)
3. **3중 평가 체계**: 궤적 예측(DroidSpatial-Bench) → 시뮬 섭동(Colosseum 14축) → 실제 로봇 4개 일반화 축으로 주장을 단계별 검증
4. **실용적 학습 레시피**: LoRA + 2단계 학습 + preservation data로 8B VLM의 기존 능력을 보존하며 3D 능력 주입, 각 스테이지 1 epoch
5. 프로프라이어터리 최상위 모델(Gemini-3.0-Pro) 및 동급 오픈 모델(RoboBrain-2.5-8B) 대비 우위를 8B 오픈 백본으로 달성, GitHub/HF 모델 공개

## 9. 약점 / 한계

1. **RGB-D 센서 의존**: 명시적 depth map 필요 — 저자들도 monocular depth estimation 통합을 향후 과제로 인정
2. **단일 시점 플래너**: 심한 occlusion에 취약, 멀티뷰 융합 미탐구
3. **단일 팔 탁상 태스크 한정**: 모바일 조작·양팔 협조 미검증
4. 표준 리더보드 벤치마크(LIBERO, CALVIN 등) 부재 — Colosseum·자체 벤치마크·실기 위주라 타 VLA와의 직접 수평 비교 곤란
5. 실제 로봇 평가가 총 25 rollout의 연속 점수 — 표본이 작고 이진 성공률과 직접 비교 불가
6. All Var. 조건 7.2%가 보여주듯 VLM grounding 자체가 무너지는 극한 섭동에는 3D 가이던스도 무력
7. L_depth의 이득이 지표에 따라 혼재(5cm Both는 소폭 하락) — 효과가 주로 시퀀스 후반 drift 교정에 국한

## 10. 다른 연구와의 위치

- **HAMSTER (ICLR 2025)** 의 직접 후속: 동일한 계층 구조(VLM 플래너 + 3D 컨트롤러)에서 플래너 출력을 2D→3D로 승격. Colosseum·실기에서 원조 HAMSTER를 일관되게 능가
- **RoboTracer / RoboBrain 2.5 (동시기 연구)**: 마찬가지로 (u, v, d) depth-aware 궤적을 예측하나, RoboTracer는 하위 컨트롤러와 통합되지 않은 독립 모션 플래너. 3D HAMSTER는 dense depth reconstruction으로 장면 수준 기하 이해를 강제하고 3D 궤적-포인트클라우드 정책 결합까지 폐루프 전체를 닫음
- **모놀리식 VLA (π0.5 등)** 와의 대비: in-distribution 성능은 대등하나 분포 이동 강건성에서 계층형+3D 가이던스가 우위라는 실증
- 하위 정책 계보(DP3 → Act3D → 3DDA → 3DFA)의 3D-native 흐름을 상위 플래너까지 확장한 작업으로 위치 지을 수 있음

## 11. 향후 연구 방향

- Monocular metric depth estimation(MoGe-2 등) 통합으로 RGB-D 하드웨어 의존 제거 — 학습 데이터 생성에는 이미 MoGe-2를 사용 중이므로 자연스러운 확장
- 멀티뷰 융합으로 occlusion 강건성 확보
- 모바일 조작·양팔 협조로 3D 궤적 가이던스의 일반성 검증
- All Var. 붕괴가 시사하는 VLM grounding 강건화(시각 손상 하 타깃 식별)와의 결합
- LIBERO/CALVIN 등 표준 벤치마크 평가로 커뮤니티 비교 가능성 확보

## 12. 종합 평가

계층형 VLA의 잘 알려졌지만 방치되던 2D-3D 표현 불일치를 정면으로 겨냥해, "플래너가 애초에 3D를 말하게 하라"는 단순한 원리를 데이터 믹스·depth encoder·dense reconstruction loss의 3요소 레시피로 실현한 견실한 시스템 논문. 궤적 예측→시뮬 섭동→실기 일반화의 3단 검증과 가이던스 신호만 분리한 통제 실험이 설계 논리를 잘 뒷받침하며, 특히 "2D 가이던스가 무섭동 성능마저 해친다"(49.5 vs 53.8)는 관찰은 계층형 VLA 설계에 실질적 함의가 있다. 표준 리더보드 부재와 소규모 실기 표본이 비교 가능성을 제한하지만, HAMSTER 계열의 자연스러운 진화이자 계층형 VLA의 인터페이스 표준이 (u, v) → (u, v, d)로 이동할 것임을 설득력 있게 보여준 논문. RLBench 트랙에는 Colosseum 무섭동 62.9 / 14축 평균 44.8로 등재되며, 동일 3DFA 정책 기반 통제 비교라는 점에서 절대 수치보다 가이던스 효과의 상대 비교로 읽어야 한다.

<!-- VERIFIED: pdf -->
