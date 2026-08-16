# CrossTracer: Cross-Embodiment Navigation via VLA Model Reasoning and Trace Residuals Adapting

> **한 줄 요약**: 내비게이션 계획을 정규화된 **이미지 평면 위 8개 waypoint(픽셀 공간 trace)** 라는 단일 인터페이스로 표현하고, OmniVLA를 LoRA로 미세조정한 VL-Tracer가 embodiment에 무관한 의미론적 trace를 제안하면 CE-Adapter가 로봇 ID 임베딩·FiLM·cross-attention으로 **잔차(residual) 보정**을 예측하는 2단계 구조. 학습 라벨은 사람이 아니라 CE-RRT*(panoptic segmentation → 로봇별 traversability cost map → RRT*)가 자동 생성하며, NaviTrace total score 45.68로 Gemini-2.5-Pro(35.67) 대비 +10.01점을 기록.

- **arXiv**: 2608.06688v1 (2026-08-07, cs.RO)
- **소속**: Peng Cheng Laboratory / Southern University of Science and Technology / Innovation Investment Research Institute / Soochow University
- **프로젝트 페이지**: https://lilduckkk.github.io/CrossTracer-Nav/
- **분량**: 14페이지

---

## 1. 배경 및 동기

VLA 모델은 로봇 내비게이션에 강력한 semantic prior를 제공하지만, **embodiment별 이동 제약(mobility constraint)을 무시**하는 경향이 있다. 논문의 출발점은 단순하고 명확하다. 어떤 로봇에게 의미론적으로 그럴듯한 경로가 다른 로봇에게는 물리적으로 실행 불가능할 수 있다는 것이다. 다리 달린 로봇은 거친 지형이나 작은 높이 변화를 넘을 수 있지만, 바퀴 로봇은 그것을 우회해야 한다.

기존 접근의 한계를 저자들은 두 갈래로 정리한다.

1. **End-to-end VLA 정책**: 관측과 목표를 액션/궤적으로 직접 매핑하면 semantic reasoning, embodiment 제약, 제어가 하나의 모델에 **얽혀버린다(entangle)**. 그래서 동일한 의미론적 계획을 traversability 프로파일이 다른 로봇에 재적용하기 어렵다.
2. **계층적 픽셀 공간 내비게이션(VAMOS 계열)**: VLM이 후보 경로를 제안하고 embodiment별 affordance 모델이 점수를 매겨 **선택**한다. 그러나 선택은 유한한 후보 집합 안에서만 가능하며, **모든 후보가 국소적으로 실행 불가능한 구간을 포함하면 실패**한다.

핵심 통찰은 **2D 픽셀 공간 trace가 semantic intent와 embodiment-aware physical grounding 사이의 통합 중간 표현이 될 수 있다**는 것이다. trace는 VLA가 추론한 장면 수준 구조를 보존하면서도 특정 로봇의 저수준 제어 공간에 조기 커밋하지 않는다. 동시에 이미지 내 traversability 단서와 공간적으로 정렬되어 있으므로, embodiment 조건부 모듈이 이를 정제할 수 있다.

📌 [Fig. 1 삽입] — 동일 목표에 대해 legged/wheeled 로봇이 서로 다른 실행 가능 경로를 따르는 개념도. 하단 패널은 각자의 1인칭 관측과 빨간 점선으로 표시된 픽셀 공간 계획 trace.

---

## 2. 문제 정의: 목표 조건부 trace 생성

논문은 cross-embodiment 내비게이션을 **이미지 평면에서의 goal-conditioned trace generation**으로 정식화한다.

```
입력: 1인칭 RGB 관측 I ∈ R^{H×W×3}
      목표 명세 G = {언어 지시 L, 목표 픽셀 좌표 P_g=(x_g,y_g), 또는 둘 다}
      대상 embodiment e ∈ E
출력: T_e = {w_t}_{t=1..N},  w_t = (x_t, y_t) ∈ [-1,1]^2
```

waypoint는 정규화 좌표이므로 **해상도 독립적 인터페이스**를 제공하며, `u_t = (x_t+1)/2 · W`, `v_t = (y_t+1)/2 · H`로 이미지 좌표로 변환된다. NaviTrace의 trace 표현을 따라 **고정 길이 N = 8**을 사용한다.

여기서 중요한 선긋기가 있다. trace는 저수준 제어 명령도, 완전한 3D 상태 궤적도 아니다. **의미론적 목표 이해와 하류 로봇 실행을 연결하는 중간 공간 표현**일 뿐이다.

embodiment는 학습된 임베딩 `z_e ∈ R^{D_emb}`로 표현되고, 분석·학습 감독을 위해 embodiment 조건부 traversability cost map `C_e : Ω → R_{≥0}`를 정의한다. 낮은 값일수록 해당 embodiment가 통과하기 쉬운 영역이다. **추론 시에는 cost map이 필요 없고 RGB, 목표 명세, embodiment ID만 있으면 된다**는 점이 설계상 핵심이다.

**잔차 분해**:

```
T_init = f_φ(I, G)          # embodiment-agnostic 의미론적 제안
ΔT_e   = g_θ(I, T_init, e)  # embodiment 조건부 잔차 보정
T_e    = T_init + ΔT_e
```

의미론적 제안으로부터의 과도한 이탈은 **명시적 hard bound가 아니라** trace imitation / physical cost / smoothness loss로 억제한다.

---

## 3. VL-Tracer: Vision-Language Trace Proposer

**OmniVLA 아키텍처 위에 구축**되며, 출력 인터페이스를 저수준 내비게이션 액션에서 고정 길이 2D waypoint 시퀀스로 교체한 것이 핵심 변경이다.

```
T_init = F_prop(I, G) = H_head( H_llm( H_enc(I, G) ) )
```

Fig. 2에 따르면 Stage 1의 언어 백본은 **Llama 7B**이고 그 위에 Trace Head가 붙는다.

**멀티모달 토큰 구성**:

```
X_input = [X_V ‖ X_L ‖ X_P],   X_P = MLP_pose(P_g) ∈ R^{1×D}
```

사용 불가능한 modality는 마스크 `m ∈ {0,1}^M`으로 제거된다. 학습 중 **언어와 pose 입력을 확률 p_drop = 0.3으로 무작위 드롭**하여 불완전한 목표 명세에 대한 강건성을 확보한다. 추론 시 마스크는 가용 입력에 맞춰 설정된다.

**Trace head**:

```
ŵ_t = tanh( MLP_head(h_t) ),  t = 1..N
```

tanh가 출력을 정규화 이미지 좌표로 제한한다.

**학습**: VAMOS의 내비게이션 trace 데이터로 **LoRA 미세조정**. 백본 가중치는 동결하고 LoRA 파라미터와 trace prediction head만 최적화한다.

```
L_VL = (1/N) Σ_t ‖ŵ_t − w_t^gt‖²  +  λ_smooth · (1/(N−1)) Σ_t ‖ŵ_{t+1} − ŵ_t‖²
λ_smooth = 0.01
```

VAMOS를 선택한 이유는 명시적이다. VAMOS가 VL-Tracer의 픽셀 공간 trace 인터페이스와 **일치하는 이미지 공간 내비게이션 경로 주석**을 제공하기 때문이다.

---

## 4. CE-RRT*: 자동 라벨 생성 파이프라인

embodiment별 trace를 사람이 주석하는 비용을 없애는 것이 목적이다. 동일 장면이라도 로봇 플랫폼마다 다른 경로가 필요하기 때문에 수작업 주석은 특히 비싸다.

**절차** (Algorithm 1):

1. **Mask2Former (ResNet-50 백본)** 로 panoptic segmentation mask M 획득
2. embodiment별 semantic cost 설정 `K_e = {S^e_free, S^e_soft, ρ_e}` 로 base cost map 구성

```
C^e_base(p) = 0                 if M(p) ∈ S^e_free
              ρ_e(M(p))         if M(p) ∈ S^e_soft
              C_obs             otherwise
```

3. **Euclidean distance transform**으로 clearance 반영
   - 주행 가능 영역 내부: `P^e_ext(p) = λ_ext · max(0, 1 − d^e_free(p)/d_max)` (비주행 픽셀 근접 페널티)
   - 주행 불가 영역: `P^e_int(p) = C_obs + β · min(1, d^e_obs(p)/d_int)`
4. **RRT\*** 로 이미지 평면에서 저비용 경로 탐색

```
J(P; C_e) = Σ_i d(q_i, q_{i+1}) · ω(q_{i+1}),   ω(q) = 1 + η · C_e(q)/C_max
```

**플래너 하이퍼파라미터**: goal-biased sampling 확률 p_goal = 0.15, steering step Δ_step = 25 픽셀, 탐색 반경 r = 60 픽셀, 최대 반복 N_max = 10,000. 세그먼트 상의 모든 샘플 점의 cost가 C_obs 미만일 때만 edge를 수락한다. 추출된 경로는 N개 waypoint로 균일 리샘플링 후 [-1,1]²로 정규화된다.

📌 [Fig. 3 삽입] — RGB 관측 → segmentation → embodiment-aware cost map → RRT* 픽셀 공간 trace 생성 파이프라인.

**중요**: 생성된 `T*_e`와 cost map `C_e`는 **학습 시에만** 사용된다. 추론 시 CrossTracer는 semantic segmentation도 사전 계산된 cost map도 요구하지 않는다.

---

## 5. CE-Adapter: 적응적 trace 잔차 학습

**시각 분기**는 사전학습된 ResNet에서 출발한다. embodiment ID를 학습 가능한 임베딩 `z_e ∈ R^D`로 매핑하고, 각 인코더 스테이지에서 **FiLM** 층이 affine 변조 파라미터를 예측한다.

```
F^e_i = γ_i(z_e) ⊙ F_i + β_i(z_e)
```

FiLM 층은 **항등에 가깝게 초기화**된다(γ ≈ 1, β ≈ 0). 즉 어댑터는 사전학습된 시각 표현에서 시작해 점진적으로 embodiment 조건부 변조를 학습한다. 최종 특징 맵은 flatten 후 시각 토큰 `X_v ∈ R^{L×D}`로 투영되고, robot token이 prepend된 뒤 Transformer 블록을 통과한다.

```
X^e_0 = [z_e ‖ X_v] + E_pos
X^e_vis = Transformer(X^e_0)
```

**Trace query와 cross-attention**:

```
Q_trace = MLP_trace(T_init) + E_trace
Z_trace = CrossAttn(Q_trace W_Q, X^e_vis W_K, X^e_vis W_V)
```

**세 개의 헤드**:

| 헤드 | 출력 | 역할 |
|------|------|------|
| Trace Residual Head | `ΔT_e = δ_max · tanh(MLP_res(Z_trace))` | 2D waypoint offset. δ_max가 보정 크기를 제한해 goal intent 보존 |
| Traversability Reconstruction Head (Feasibility) | `Ĉ_e = Decoder_trav(X^e_vis, {F^e_i})` | embodiment 조건부 cost map 재구성 (auxiliary) |
| Embodiment Sensitivity Head | `α_e = Softplus(MLP_sens(z_e))` | cost 항의 embodiment별 가중치 |

📌 [Fig. 4 삽입] — CE-Adapter 아키텍처. Multimodal Condition Encoder → FiLM → Fusion Tokens, Trace Projector → Trace Query, 3개 헤드 분기.
📌 [Fig. 5 삽입] — 동일 관측에 대해 wheeled/legged가 서로 다른 traversability 분포를 유도하는 Feasibility Map 비교.

---

## 6. 학습 목표와 데이터 파이프라인

**2단계 학습이고 gradient가 섞이지 않는다**는 점이 중요하다. VL-Tracer를 먼저 미세조정한 뒤 **동결**하고, CE-Adapter는 고정된 초기 trace를 정제하도록 학습된다. **CE-Adapter의 손실은 VL-Tracer로 역전파되지 않는다.**

```
L_CE = L_trace + λ_trav·L_trav + λ_cost·L_cost + λ_smooth·L_smooth
```

- `L_trace = (1/N) Σ ‖w^e_t − w*_t‖²` — 플래너 감독 참조 trace와의 L2
- `L_trav = (1/|P|) Σ_p ‖Ĉ_e(p) − C_e(p)‖²` — 시각 인코더가 희소한 waypoint 감독에만 의존하지 않고 로봇 조건부 지형 affordance를 학습하도록 유도
- `L_cost = α_e · (1/N) Σ_t C_e(w^e_t)` — 고비용 영역에 놓인 waypoint 페널티, cost map에서 bilinear sampling
- `L_smooth` — 2차 유한차분 기반 평활화

**하이퍼파라미터**: RGB 이미지와 cost map을 64×64로 리사이즈, N = 8, Adam lr 1e-4, batch size 64, `(λ_trace, λ_trav, λ_cost, λ_smooth) = (1.0, 1.0, 1.0, 0.05)`. smoothness에 작은 가중치를 준 이유는 급격한 방향 전환은 억제하되 **필요한 embodiment 조건부 보정까지 억누르지 않기 위해서**다.

**데이터/컴퓨트**:

| 모듈 | 데이터 | 하드웨어 | 시간 |
|------|--------|----------|------|
| VL-Tracer | VAMOS 내비게이션 trace 데이터 | 8× NVIDIA A100 | ~48시간 |
| CE-Adapter | CE-RRT*가 주석한 **62k** 내비게이션 이미지 | 1× RTX 4090 | ~3시간 |

CE-Adapter가 4090 한 장으로 3시간이면 학습된다는 점은 잔차 학습 설계의 실용적 강점이다.

---

## 7. NaviTrace 벤치마크 결과

각 테스트 샘플은 1인칭 RGB, 언어 지시, embodiment 타입을 제공하고 모델은 픽셀 공간 2D trace를 출력한다. 공식 평가 프로토콜의 **total score**가 주 지표다.

**Table I 주요 수치 (total score)**:

| 모델 | Open-Source | Total Score |
|------|-------------|-------------|
| Qwen3-VL-8B-Thinking | ✓ | −41.30 |
| MiMo-Embodied-8B-Thinking | ✓ | −33.55 |
| Claude Sonnet-4.5 | ✗ | 7.36 |
| Qwen3-VL-235B-Thinking | ✓ | 26.24 |
| Robobrain-2.5-8B | ✓ | 27.96 |
| Gemini-2.5-Pro | ✗ | 35.67 |
| **CrossTracer-8B (Ours)** | ✓ | **45.68** |
| CrossTracer w/o CE-Adapter | ✓ | 22.56 |
| CrossTracer w/ Goal Pose | ✓ | 63.91 |

Gemini-2.5-Pro 대비 **+10.01점(상대 28%)**, 최강 embodied 모델 Robobrain-2.5-8B 대비 **+17.72점**이다. 저자들의 해석은 "강한 시각 추론만으로는 이 벤치마크에 충분하지 않다"는 것이다.

**CE-Adapter 기여도 (ablation)**: 제거 시 45.68 → 22.56으로 **23.12점 하락**. 특히 물리적 grounding이 필요한 카테고리에서 낙차가 극적이다.

| 카테고리 | w/o CE-Adapter | Full | Δ |
|----------|----------------|------|---|
| Accessibility | −3.18 | 33.79 | +36.97 |
| Social Norms | 1.28 | 37.87 | +36.59 |
| Stationary Obstacle | 25.23 | 46.11 | +20.88 |

Accessibility와 Social Norms가 음수/1점대에서 30점대로 올라간다는 것은 CE-Adapter가 단순히 trace를 매끈하게 다듬는 게 아니라 **embodiment별 traversability 제약과 충돌할 구간을 실제로 교정**한다는 증거다.

**Embodiment 간 일관성**: bicycle 42.16, human 46.26, legged robot 46.40, wheeled robot 46.28. 편차가 작다는 것은 단일 embodiment에만 과적합되지 않았음을 시사한다. 장면 카테고리에서는 dynamic obstacle 52.93, visibility 52.49가 강하고 geometric terrain 45.87, semantic terrain 45.54로 안정적이다.

**Goal-pose 변형**: 언어 지시와 embodiment 타입에 더해 목표 픽셀 좌표를 주면 45.68 → **63.91**. 저자들이 스스로 밝히듯 **동일 입력 조건이 아니므로 직접 비교 대상은 아니며**, 픽셀 공간 인터페이스가 정밀한 목표 위치 정보를 효과적으로 활용할 수 있음을 보이는 용도다. 개선 폭은 semantic terrain(45.54 → 61.50)과 social norms(37.87 → 63.32)에서 특히 크다.

📌 [Fig. 6 삽입] — 카테고리별 성능 레이더/라인 차트.
📌 [Fig. 7 삽입] — 인도, 식생 지대, 도심 도로, 실내, 좁은 통로, 장애물, 지형 변화 시나리오의 정성적 trace 비교.

---

## 8. 실제 로봇 배치 결과

**플랫폼**: 바퀴 로봇(지형 적응성 제한)과 다리 로봇(복잡한 표면 변화와 작은 높이 변화 처리 가능). 양쪽 모두 1인칭 RGB 카메라.

**배치 구조**: 온보드 NVIDIA Jetson Orin이 카메라 관측을 WiFi로 RTX 4090 워크스테이션에 전송 → 워크스테이션이 추론 → 픽셀 공간 trace를 되돌려받아 Jetson에서 국소 waypoint로 변환 후 컨트롤러가 폐루프 실행. **저수준 컨트롤러는 방법 간 동일하게 유지**되어 비교가 trace 품질에만 집중된다.

**과제**: 실내 semantic target 2개(팬트리 카운터 뒤 흰 테이블, 베이지 소파), 2층 플랫폼의 "Caution Wet Floor" 표지판(지형 접근성 테스트), 옥외 좌회전 후 검은 쓰레기통(장거리 + 회전 행동). **플랫폼·방법당 과제별 5회 반복**.

**지표**: SR, SPL(= (1/N)Σ S_i·L^ref_i/L_i), STT(= (1/N)Σ S_i·T^ref_i/T_i). 실패 시행은 S_i = 0이므로 SPL/STT도 0. **참조 경로 길이와 시간은 human expert path 기준**이다.

**Table II 평균 (전 시나리오)**:

| 플랫폼 | 방법 | SR | SPL | STT |
|--------|------|----|----|----|
| Wheeled | OmniVLA | 0.40 | 0.37 | 0.17 |
| Wheeled | **CrossTracer** | **0.65** | **0.59** | **0.30** |
| Legged | OmniVLA | 0.45 | 0.31 | 0.27 |
| Legged | **CrossTracer** | **0.70** | **0.58** | **0.43** |

과제별로 보면 다리 로봇의 2층 플랫폼 과제에서 0.60 → 0.80으로 개선폭이 두드러진다. 저자 해석은 예측된 trace가 실행 가능한 접근 방향으로 유도할 때 로봇이 자신의 지형 적응성을 더 잘 활용할 수 있다는 것이다. 흰 테이블 과제의 다리 로봇은 0.20 → 0.60으로 3배가 된다.

📌 [Fig. 8 삽입] — robot-side / server-side 통신 파이프라인.
📌 [Fig. 9 삽입] — 동일 실내 환경에서 embodiment별로 다르게 생성되는 trace.
📌 [Fig. 10 삽입] — CrossTracer / OmniVLA 실행 경로 vs human expert 참조 경로 비교.

---

## 9. Related Work 상의 위치

저자들은 자기 위치를 네 축으로 대비시킨다.

- **VLA 내비게이션 (LeLaN, NaVILA, OmniVLA)** 대비: 이들은 멀티모달 목표 인터페이스를 개선했지만, CrossTracer는 **VLA 파생 모듈을 의미론적 픽셀 trace 제안에만 쓰고** embodiment 의존 적응은 별도 모듈에 맡긴다.
- **VAMOS** 대비: VAMOS는 이미지 공간 후보 경로를 affordance 모델이 평가·재랭킹한다. CrossTracer는 **고정 후보 집합에서 고르지 않고** trace를 연속적 정제 대상으로 취급한다.
- **Cross-embodiment 정책 (X-Nav, X-Mobility, COMPASS, CE-Nav, NavDP, FlowNav)** 대비: 이들은 주로 action/velocity/control 공간에서 적응하지만, CrossTracer는 **픽셀 공간에서 직접** embodiment 적응을 수행한다.
- **플래너 감독 (MTG, CE-Nav, COMPASS, VAMOS)** 대비: CE-RRT*는 동일한 planner-supervision 아이디어를 따르되 **학습 인터페이스가 다르다**. segmentation → 로봇 조건부 cost map → 픽셀 공간 trace라는 경로를 취한다.

---

## 10. 강점

1. **표현 선택이 문제를 정확히 풀어낸다.** 픽셀 공간 trace는 (a) VLA의 장면 구조 추론을 보존하고, (b) 특정 로봇 제어 공간에 조기 커밋하지 않으며, (c) 이미지 내 traversability 단서와 공간 정렬되어 있어 정제 가능하다. 세 성질이 동시에 필요했고 trace가 이를 모두 만족한다.
2. **Ablation의 낙차가 설득력 있다.** 23.12점 하락, 특히 Accessibility −3.18 → 33.79는 CE-Adapter가 장식이 아님을 보인다.
3. **주석 비용을 실제로 없앴다.** CE-RRT*가 62k 이미지를 자동 주석하고, 결과 모듈은 4090 한 장에서 3시간이면 학습된다. 새 로봇 추가 비용이 낮다.
4. **추론 시 의존성을 깔끔히 제거했다.** segmentation과 cost map은 학습 전용이며, 배치 시에는 RGB + 목표 + embodiment ID만 필요하다.
5. **자기 결과에 대한 정직함.** goal-pose 변형이 동일 입력 비교가 아님을 본문에서 먼저 밝힌다.
6. **실기 검증이 두 플랫폼에 걸쳐 있다.** 동일 컨트롤러로 통제한 상태에서 SR/SPL/STT 세 지표 모두 개선.

---

## 11. 약점 및 한계

1. **segmentation 의존성 (저자 자인).** 데이터 생성이 panoptic segmentation에 기대므로 segmentation 오류가 cost map과 플래너 감독 trace 품질을 직접 오염시킨다.
2. **cost map 설정이 수작업 (저자 자인).** embodiment별 `K_e = {S_free, S_soft, ρ}` 구성이 수동이며 새 플랫폼 적응 시 전문 지식이 필요하다. "cross-embodiment"라는 주장의 확장성이 여기서 제한된다.
3. **2D 표현의 근본적 한계 (저자 자인).** 돌출 장애물(overhanging obstacle)과 높이 불연속을 픽셀 평면 표현으로 다루기 어렵다.
4. **폐루프 재계획 부재 (저자 자인).** 동적 환경 대응은 future work로 남는다.
5. **실기 시행 수가 적다.** 과제·플랫폼·방법당 5회, 총 4과제. SR 0.65 vs 0.40 같은 차이의 통계적 유의성을 판단하기에는 표본이 작고 신뢰구간이 제시되지 않는다.
6. **실기 베이스라인이 OmniVLA 하나뿐.** VL-Tracer가 OmniVLA에서 파생되었으므로 이 비교는 사실상 "CE-Adapter 유무" 비교에 가깝다. Table I의 다른 강력한 베이스라인들은 실기에서 검증되지 않았다.
7. **δ_max, λ_ext, β, η, d_max, d_int, D_emb 등 여러 상수의 구체 값이 본문에 없다.** 잔차 크기를 제한하는 δ_max는 설계상 매우 중요한 값인데 수치가 제시되지 않는다.
8. **VL-Tracer 단독 성능이 낮다.** 22.56은 Robobrain-2.5-8B(27.96)나 Gemini-2.5-Pro(35.67)보다 낮다. 즉 성능의 상당 부분이 VLA 백본이 아니라 RRT* 플래너 감독을 증류한 어댑터에서 나온다. 이는 "VLA reasoning" 프레이밍에 대한 반문을 낳는다.
9. **원격 추론 구조.** RTX 4090 워크스테이션에 WiFi로 붙는 배치는 실제 자율성 주장에 제약이며, latency나 온보드 실행 가능성 수치는 보고되지 않는다.
10. **NaviTrace 단일 벤치마크.** 시뮬레이션 평가가 한 벤치마크에 집중되어 일반화 근거가 얇다.

---

## 12. 총평

CrossTracer는 **"의미론적 제안과 물리적 적응을 분리하되 공통 좌표 인터페이스로 소통시킨다"**는 아이디어를 픽셀 공간 trace로 구현한 깔끔한 연구다. VLA를 인터페이스 교체(액션 → waypoint) 후 LoRA로 미세조정하고, 그 위에 플래너가 자동 생성한 라벨로 학습한 경량 잔차 어댑터를 얹는 구성은 재현 가능성과 확장성 양면에서 실용적이다.

VLA-Tracker 등재 관점에서 이 논문은 명확히 **자체 학습된 정책 산출물**을 갖는다. frozen VLA를 추론용으로 호출만 하는 파이프라인이 아니라, (1) OmniVLA 백본 위에 trace prediction head를 새로 붙이고 VAMOS 데이터로 LoRA 미세조정한 VL-Tracer, (2) 62k CE-RRT* 라벨로 처음부터 학습한 CE-Adapter — 두 개의 학습된 산출물이 존재하며 두 단계 모두 손실 함수와 하이퍼파라미터가 명시되어 있다.

다만 냉정하게 보면 이 논문의 성능 향상 대부분은 **RRT* 플래너의 기하학적 지식을 신경망 어댑터로 증류한 결과**다. VL-Tracer 단독 22.56 → 전체 45.68이라는 구조는, VLA의 semantic prior보다 고전 플래너 감독이 더 큰 기여를 했음을 시사한다. 이는 결함이라기보다 이 논문이 실제로 발견한 것에 가깝다. 즉 **"foundation model의 시각 추론만으로는 embodiment 제약을 다룰 수 없고, 고전 플래닝의 기하학적 지식을 어떤 형태로든 주입해야 한다"**는 관찰이다. 픽셀 공간 trace는 그 주입 통로를 만드는 영리한 방법이었다.

**한 문장 요약**: 픽셀 공간 waypoint를 인터페이스로 삼아 VLA의 의미론적 제안 위에 플래너 감독으로 학습한 embodiment 조건부 잔차를 더하면, NaviTrace 45.68과 실기 SR 0.65/0.70을 얻는다 — 다만 그 성능의 무게중심은 VLA가 아니라 RRT*가 만든 라벨 쪽에 있다.

<!-- VERIFIED: pdf -->
