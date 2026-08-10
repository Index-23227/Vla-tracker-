# SkillMemo: Expert-guided Skill Memory Framework for Compositional Embodied Manipulation

> **한 줄 요약**: 로봇 조작 정책이 궤적을 하나의 monolithic 네트워크로 학습해 **재사용 가능한 스킬 구조**를 포착하지 못한다는 문제를, MoE 게이팅으로 궤적을 **암묵적(implicit)으로 latent atomic skill로 분할**(EGTS)하고 그 게이팅 프로파일을 key-value **episodic memory bank**에 저장·검색·융합(SLMA)하여 해결. LIBERO에서 π0.5-SkillMemo가 평균 **98.0%** (π0.5 96.8% 대비 +1.2, MemoryVLA 96.5% 대비 +1.5)를 달성하고, DP 백본에서도 Push-T/BlockPush/Kitchen 평균 **64.6%**로 SOTA.

- arXiv: 2608.05970v1 (2026-08-06, cs.RO)
- 저자: Changyuan Wang, Chubin Zhang, Zhenyu Wu 외 — Tsinghua University (SIGS / Dept. of Automation), NTU, Beijing Normal University
- Project Page: https://changyuanwang17.github.io/SkillMemo/

---

## 1. 배경 및 동기

### 데이터 희소성이 만드는 구조적 한계
- DP(Diffusion Policy)와 VLA 모델은 대규모 vision-language 사전학습 + embodied trajectory 파인튜닝으로 큰 성과를 냈지만, **대규모 로봇 데이터 획득 비용**이 실배포의 근본 제약.
- 불충분한 시연으로 학습하면 **monolithic policy**가 되어 재사용 가능한 행동 구조를 포착하지 못함 → 알려진 객체·수용기(receptacle)·지시가 **새로운 조합**으로 나타나는 OOD 상황에서 성능 급락.

### 기존 접근의 결핍
- 로봇 RL, motion primitive library, memory-augmented policy가 시도되었고, 그중 메모리 계열이 long-horizon 일반화의 유망한 패러다임.
- **MemoryVLA**: dual-stream Perceptual-Cognitive Memory Bank로 시각 디테일 + 의미 요약 저장.
- **HAMLET**: 상호작용 히스토리를 temporal moment token으로 인코딩하여 non-Markovian 의존성 처리.
- **공통 결핍**: 이들은 *비구조화된/전체론적(holistic)* 시각-의미 표현을 저장할 뿐, **행동적으로 유의미한 skill-level 특징을 기억하는 명시적 메커니즘이 없음**. 결과적으로 과거 경험 재사용이 비효율적이고 미지 과제에 대한 compositional generalization이 떨어짐.

### 본 논문의 주장
"기억해야 할 단위는 프레임이나 요약이 아니라 **스킬(skill primitive)** 이다." SkillMemo는 궤적을 latent atomic skill로 분해하고, 그 스킬 수준 특징을 dynamic episodic memory에 통합한다.

---

## 2. 방법론 심층 분석 (1): Expert-Guided Trajectory Segmentation (EGTS)

### 명시적 분할의 실패
궤적 τ = {(x_t, a_t)}_{t=1..T}를 고정 chunk 크기 L로 비중첩 세그먼트 {s_k}로 나누는 explicit segmentation은 잠재적 스킬 경계를 모두 포착하지 못해 **부정확한 분할**을 낳는다.

### 암묵적 분할: MoE 라우팅이 곧 경계
N개 expert {E_1,...,E_N}와 게이팅 네트워크 G로 구성된 MoE 레이어:

```
y_t = Σ_{i=1..N} g_i(h_t) · E_i(h_t)
```

- 각 expert는 **반복되는 국소 motion primitive**를 담당.
- 게이팅 네트워크 g는 시간에 따라 expert 활성화를 변조하며 **암묵적으로 시간적 분할 경계를 정의**.
- 행동 레이블 없이도 데이터 주도적(data-driven) 스킬 발견이 가능하며, 발견된 스킬이 emergent task-level 구조와 정렬됨.

### PID 기반 시너지 손실
Expert 특화와 조합성을 동시에 유도하기 위해 Partial Information Decomposition 이론에서 유도한 손실을 사용:

```
L_PID = −( I({A,B};G) − I(A;G) − I(B;G) )
```

- 임의의 expert 쌍 (A, B)에 대해, **joint 표현이 개별 표현의 합보다 더 많은 task-relevant 정보**를 담도록 강제.
- 각 expert가 상호 보완적 스킬에 특화되면서도 고차 협응이 가능 → 서로 겹치지 않는(distinct, non-overlapping) skill primitive 학습.
- 보완적으로 **expert load-balancing loss**와 **top-k routing**으로 expert collapse 방지.

📌 [Figure 2 삽입] — EGTS(MoE 라우팅으로 latent atomic skill 추출) → SLMA(게이팅 프로파일 저장·동적 프루닝·검색) 전체 파이프라인

---

## 3. 방법론 심층 분석 (2): Skill-Level Memory Architecture (SLMA)

### Memory Storage — 무엇을 Key로, 무엇을 Value로
메모리 뱅크 M을 key-value 쌍의 집합으로 구성:

**(1) Memory Key** — 스킬 세그먼트의 **시간 집약 centroid feature**
```
k_m = (1/L) · Σ_{t ∈ τ_m} h_t
```
프레임 단위 특징을 전부 저장하면 메모리가 확장 불가능(unscalable)해지므로, 세그먼트의 대표 특성만 압축 보존하여 검색 효율 확보.

**(2) Memory Value** — 게이팅 가중치 **시퀀스 전체**
```
M = {(k_m, v_m) | v_m = {g(h_t)}_{t ∈ τ_m}}_{m=1..M}
```
게이팅 가중치가 expert 활성화를 지배하고 스킬의 범주를 암묵적으로 정의하므로, 이를 **핵심 knowledge content**로 지정. 시간적 동역학 보존을 위해 세그먼트 내 모든 timestep의 게이팅을 저장.

> 💡 **설계상 핵심**: 저장되는 것이 "관측"이나 "요약"이 아니라 **"어떤 expert를 언제 얼마나 켰는가"라는 절차적(procedural) 지식**이라는 점이 MemoryVLA류와의 결정적 차이.

### Memory Retrieval — 유사도 + 신뢰도 게이트 + 융합
1. 현재 관측의 latent feature q_t와 모든 key의 **코사인 유사도** 계산: `s(q_t, k_m) = (q_t·k_m)/(‖q_t‖‖k_m‖)`
2. 상위 N개 항목 M_top = {(k_n, v_n)} 검색.
3. **Memory reliability check**: 검색 집합의 평균 유사도를 사전 정의 임계값 δ와 비교. 신뢰할 만할 때만 융합 수행 (잘못된 prior 주입 방지).
4. 게이팅 융합:
```
g'(q_t) = λ·g(q_t) + (1−λ)·(1/N)·Σ_{n=1..N} v_n[t′]
```
λ는 메모리 개입 강도를 조절하는 하이퍼파라미터, v_n[t′]는 검색된 스킬 시퀀스 내 정렬된 timestep.

### Memory Update — 유한 용량에서의 우선순위 프루닝
용량 M_max에 도달하면 세 기준으로 저효용 항목 제거:
1. **가장 오래된 timestamp** (강화되지 않은 먼 기억의 망각을 모사)
2. **한 번도 성공적으로 검색되지 않은 항목**
3. **검색 단계에서 반복적으로 신뢰도 임계값 δ를 넘지 못한 항목** (저품질/무관)

→ 메모리 다양성과 품질의 trade-off를 균형 잡아 뱅크를 compact하게 유지 (lifelong learning 시나리오 대응).

---

## 4. 구현 및 학습 세부

| 항목 | 값 |
|------|-----|
| 하드웨어 | 8× NVIDIA A6000, PyTorch |
| 배치 | GPU당 32 샘플 → global batch 256 |
| 옵티마이저 | AdamW, lr = 2×10⁻⁵ |
| 입력 | 단일 RGB 이미지 관측 + 언어 지시 |
| 출력 | 7차원 연속 액션 (6-DoF end-effector pose + gripper) |
| MoE expert 수 | **N = 5** (기본값) |
| 메모리 용량 | **M_max = 1000** skill entries |
| LIBERO 학습 | π0.5를 따라 전 task 단일 통합 정책, **30k steps** |
| 검증 | task당 **50 rollouts** 평균 성공률 |

**실로봇 셋업**: 7-DoF UR5e + Weiss WSG-50 평행 그리퍼, ORBBEC Femto Bolt RGB-D 카메라(flexible boom arm, side-view), 높이 조절 가능 이동식 리프팅 테이블(시점·높이 변화에 대한 강건성 평가 목적). 액션 공간은 6차원(3D 병진 + 3D Euler 회전) + binary gripper. task당 kinesthetic teaching으로 50 시연 수집.

---

## 5. 실험 설정

### 백본
- **DP 계열**: Diffusion Policy (저수준 visuomotor control)
- **VLA 계열**: UniAct-0.5B, UniVLA-8.5B, π0.5-3.3B

### 벤치마크
- **LIBERO** (7-DoF Franka Panda): Goal / Spatial / Object / Long 4개 suite, suite당 10개 task. Long은 long-horizon 전용.
- **DP 표준 연속제어**: Push-T, UR3 Block Push, Franka Kitchen.
- **실세계**: UR5e 5개 contact-rich task (Strawberry in Bowl, Lemon on Plate, Corn in Pot, Butter in Pot, Chip Bucket on Table), task당 40 trial.
- **Zero-shot cross-suite**: Goal/Spatial/Object 중 하나로 학습 → 나머지 suite에서 직접 평가.

---

## 6. 주요 결과

### LIBERO (Table 5) — 백본 무관 일관된 향상

| Method | Params | Goal | Spatial | Object | Long | **Avg** |
|--------|--------|------|---------|--------|------|---------|
| OpenVLA | 7.0B | 78.0 | 85.0 | 86.8 | 54.0 | 76.0 |
| TriVLA | 3.4B | 89.8 | 91.2 | 93.8 | 73.2 | 87.0 |
| CogACT | 7.6B | 90.2 | 97.2 | 98.0 | 88.8 | 93.2 |
| π0 | 3.3B | 95.8 | 96.8 | 98.8 | 85.2 | 94.2 |
| MemoryVLA | 7.3B | 96.4 | 98.4 | 98.4 | 93.4 | 96.5 |
| UniAct | 0.5B | 68.7 | 72.1 | 75.7 | 51.4 | 67.2 |
| **UniAct-SkillMemo** | 0.6B | 73.4 | 80.2 | 79.8 | 57.2 | **72.7** (+5.5) |
| UniVLA | 8.5B | 91.8 | 96.5 | 95.6 | 92.0 | 93.9 |
| **UniVLA-SkillMemo** | 8.9B | 94.5 | 98.1 | 97.8 | 93.2 | **95.9** (+2.0) |
| π0.5 | 3.3B | 98.0 | 98.8 | 98.2 | 92.4 | 96.8 |
| **π0.5-SkillMemo** | 3.6B | **99.0** | **99.4** | 98.2 | **95.4** | **98.0** (+1.2) |

- π0.5-SkillMemo가 **98.0%**로 SOTA. MemoryVLA(96.5) 대비 **+1.5**.
- 파라미터 증가는 0.1~0.4B 수준으로 매우 경제적 (π0.5: 3.3B → 3.6B).
- 가장 큰 이득은 **작은 백본**(UniAct +5.5)과 **long-horizon suite**(π0.5 Long 92.4 → 95.4, +3.0)에서 발생 — 스킬 재사용이 데이터/지평 제약을 보완한다는 가설과 정합.

### DP 백본 시뮬레이션 (Table 3)

| Model | Push-T | BlockPush | Kitchen | Avg |
|-------|--------|-----------|---------|-----|
| DP | 52.9 | 73.6 | 57.1 | 61.2 |
| SDP | 53.5 | 72.9 | 56.3 | 60.9 |
| CP | 53.2 | 73.2 | 56.0 | 60.8 |
| IMLE Policy | 53.7 | 77.2 | 57.5 | 62.8 |
| STEP | 49.7 | 76.9 | 58.1 | 61.6 |
| **SkillMemo** | **55.2** | **78.1** | **60.6** | **64.6** |

모든 벤치마크에서 전 baseline 상회. DP 대비 +3.4, IMLE Policy 대비 +1.8.

### 실세계 UR5e (Table 6, task당 40 trial)

| Model | Strawberry in Bowl | Lemon on Plate | Corn in Pot | Butter in Pot | Chip Bucket on Table |
|-------|-----|-----|-----|-----|-----|
| Diffusion Policy | 77.5 | 77.5 | 82.5 | 62.5 | 67.5 |
| **SkillMemo** | **82.5** | **80.0** | **90.0** | **75.0** | **77.5** |

어려운 task일수록 이득이 큼: Butter in Pot **+12.5**, Chip Bucket **+10.0**, Corn in Pot **+7.5**.

---

## 7. Ablation 및 메커니즘 분석

### 컴포넌트별 기여 (Table 1)

**UniAct 백본 (LIBERO)**

| EGTS | SLMA | Goal | Spatial | Object |
|------|------|------|---------|--------|
| — | — | 68.7 | 72.1 | 75.7 |
| ✓ | — | 71.1 | 78.6 | 79.3 |
| ✓ | ✓ | **73.4** | **80.2** | **79.8** |

**DP 백본**

| EGTS | SLMA | Push-T | BlockPush | Kitchen |
|------|------|--------|-----------|---------|
| — | — | 52.9 | 73.6 | 57.1 |
| ✓ | — | 53.6 | 76.5 | 59.1 |
| ✓ | ✓ | **55.2** | **78.1** | **60.6** |

→ EGTS 단독으로도 전 suite 일관 향상(암묵적 스킬 분해의 유효성), SLMA는 그 위에 특히 **조합적 추론이 필요한 task**에서 추가 이득.

### Expert 수의 영향 (Table 2, UniAct)

| N | Goal | Spatial | Object | Inference Time (s) |
|---|------|---------|--------|--------------------|
| 1 | 68.7 | 72.1 | 75.7 | 3.20 |
| 2 | 69.7 | 75.6 | 77.9 | 2.78 |
| 3 | 70.3 | 76.3 | 78.5 | 3.60 |
| **5** | **71.1** | **78.6** | **79.3** | **3.93** |
| 8 | 72.3 | 78.9 | 79.5 | 5.43 |
| 10 | 72.7 | 79.5 | 80.1 | 7.94 |

N=5 이후 **수확 체감**. N=10은 정확도 이득이 미미한 반면 추론 시간이 약 2배(3.93s → 7.94s) → **N=5를 기본값**으로 채택.

### Expert 활성화 분석 (Figure 3)
게이팅 계수의 시간적 변화가 **critical motion transition과 정렬**. "그릇을 잡기 시작"·"스토브를 켜기 시작" 같은 의미적 행동 전환 시점에서 특정 expert의 활성화가 뚜렷한 peak를 보임 → **명시적 action label이 전혀 없음에도** MoE 라우팅이 distinct latent atomic skill을 추출·특화함을 확인.

### Zero-shot cross-suite 일반화 (Table 4)

| 학습 Dataset | Method | Goal | Spatial | Object |
|--------------|--------|------|---------|--------|
| LIBERO-Goal | w/o memory | 68.7 | 71.3 | 72.5 |
| | **w memory** | **73.4** | **75.6** | **79.1** |
| LIBERO-Spatial | w/o memory | 70.3 | 72.1 | 73.9 |
| | **w memory** | **72.0** | **80.2** | **78.2** |
| LIBERO-Object | w/o memory | 69.5 | 70.8 | 75.7 |
| | **w memory** | **71.6** | **73.4** | **79.8** |

**가장 인상적인 결과**: Goal로 학습한 SkillMemo가 미지의 Spatial suite에서 **75.6%**를 기록, Spatial로 직접 학습한 in-distribution baseline **72.1%를 +3.5 상회**. 메모리 뱅크가 스킬 primitive의 조합적 재사용을 가능케 해 학습 분포를 넘어서는 일반화를 달성함을 강하게 시사.

### 실세계 조합 일반화 (Table 7, task당 40 trial)

| Unseen Task | Diffusion Policy | SkillMemo |
|-------------|------------------|-----------|
| Strawberry in Bowl (학습됨) | 77.5 | 82.5 |
| Lemon on Plate (학습됨) | 77.5 | 80.0 |
| **Strawberry on Plate** | 65.0 | **75.0** |
| **Lemon in Bowl** | 62.5 | **75.0** |
| **Strawberry in Pot** | 60.0 | **70.0** |
| **Lemon in Pot** | 57.5 | **72.5** |
| **Butter in Bowl** | 65.0 | **70.0** |
| **Butter on Plate** | 60.0 | **67.5** |

두 가지 설정: (1) Strawberry in Bowl + Lemon on Plate 공동 학습 후 crossover 조합 평가, (2) Butter in Pot을 학습에 추가 후 더 복잡한 미지 조합 평가. 미지 조합에서 DP 대비 **+5 ~ +15%p**로 격차가 in-distribution보다 오히려 커짐 — 서로 다른 학습 경험에서 atomic skill을 검색·융합하는 메커니즘이 실제로 작동함을 뒷받침.

---

## 8. 정성적 분석

Figure 5는 UR5e의 실세계 롤아웃 연속 프레임(Strawberry into Bowl, Corn into Pot, Butter into Pot, Chip Bucket onto Table)을 제시. 메모리 뱅크에서 latent atomic skill을 검색·조합함으로써 다양한 객체·수용기 조합에 대해 정밀한 grasping과 placing이 부드럽게 이어짐을 보여준다. Figure 4는 하드웨어 셋업(UR5e + WSG-50 + Femto Bolt + 이동식 리프팅 테이블)을 도해.

---

## 9. 관련 연구와의 비교

| 축 | 대표 연구 | SkillMemo의 차별점 |
|----|-----------|--------------------|
| Embodied visuomotor | DP, 3D-DP, 계층형 DP / OpenVLA, RT-2, π0, π0.5 | 백본을 대체하지 않고 **wrapper**로 작동, DP·VLA 양쪽에 동시 적용 |
| Skill learning (explicit) | RoboMatrix, Atomic Skill Library, Primitive Prompt Learning | **정적 스킬 라이브러리 없이** MoE 게이팅으로 암묵 분할 |
| Skill learning (implicit) | Long-VLA (phase-aware masking), SDP/MENTOR (MoE) | MoE 스킬 발견을 **동적 episodic memory와 결합** |
| Memory-based | MemoryVLA (dual-stream perceptual-cognitive), HAMLET (temporal moment token), RoboMemory, MAP-VLA | 비구조화·전체론적 시각-의미 표현 대신 **분해된 skill primitive를 구조적으로 저장**하고 **행동적 관련성**으로 검색 |

핵심 포지셔닝: "MoE 기반 스킬 발견 × 동적 episodic memory"의 **결합**이 본 논문의 고유 지점.

---

## 10. 강점

1. **저장 단위의 재정의**: 관측/요약이 아닌 **게이팅 프로파일(절차적 지식)** 을 저장한다는 발상이 명확하고, key(centroid feature)와 value(게이팅 시퀀스)의 역할 분리가 깔끔하다.
2. **백본 무관성 실증**: DP, UniAct-0.5B, UniVLA-8.5B, π0.5-3.3B의 4개 이질적 백본 모두에서 일관된 향상 — 우연한 튜닝 결과가 아님을 시사.
3. **파라미터 효율**: +0.1~0.4B로 π0.5 대비 +1.2, MemoryVLA(7.3B) 대비 +1.5를 절반 이하 파라미터로 달성.
4. **압도적 조합 일반화 증거**: Goal 학습 모델이 미지 Spatial에서 in-distribution baseline을 능가(75.6 vs 72.1)하는 결과는 단순 성능 개선을 넘어 메커니즘 주장을 직접 지지한다.
5. **실세계 검증의 설계**: 학습된 task와 미지 조합 task를 같은 표에서 비교(Table 7)해, 이득이 in-distribution이 아니라 조합 상황에서 커진다는 것을 명료히 보여줌.
6. **엔지니어링 현실성**: 신뢰도 임계값 δ 게이트와 3기준 프루닝으로 메모리 오염·무한 증식이라는 실전 실패 모드를 선제 처리.

---

## 11. 한계 및 논의

1. **핵심 하이퍼파라미터의 ablation 부재**: 메모리 개입 강도 λ, 신뢰도 임계값 δ, 검색 개수 N(top-N)에 대한 민감도 분석이 제시되지 않는다. 특히 λ는 "메모리를 얼마나 믿을 것인가"를 직접 통제하는 값이라 없는 것이 아쉽다.
2. **PID 손실의 실증 부재**: L_PID가 방법론의 이론적 핵심임에도 이를 제거한 ablation이 없어, 실제 기여분이 load-balancing/top-k routing 대비 얼마인지 알 수 없다.
3. **메모리 용량·프루닝의 검증 부족**: M_max=1000의 근거, 프루닝 3기준 각각의 효과, lifelong 시나리오에서의 장기 열화 곡선이 보고되지 않았다.
4. **추론 비용**: N=5에서 3.93s(UniAct 기준, Table 2)는 baseline N=1의 3.20s보다 느리며, 검색·융합 오버헤드가 별도로 분리 보고되지 않았다. 실시간 제어 관점의 지연 분석이 필요하다.
5. **π0.5-SkillMemo의 Object suite 정체**: 98.2 → 98.2로 변화 없음(포화). 상위 백본에서는 남은 헤드룸이 Long suite에 집중되어 있어, 향후 이득은 long-horizon에서만 기대 가능해 보인다.
6. **범위의 협소함**: 시뮬레이션 평가가 LIBERO + DP 3종에 국한. SimplerEnv, CALVIN, RoboTwin 등 교차 검증이 없어 일반성 주장이 제한적이다.
7. **재현성**: 프로젝트 페이지만 공개되고 코드 릴리스가 논문에 명시되지 않아, MoE 삽입 위치·게이팅 정렬(v_n[t′]의 timestep alignment) 등 구현 디테일 재현이 어렵다.

> ❓ **예상 질문**: 저장하는 것이 게이팅 분포뿐인데, 시각적으로 전혀 다른 상황에서 잘못된 스킬이 검색되면?
> **답변**: 그것이 δ 신뢰도 체크의 역할이다. 검색 집합의 평균 코사인 유사도가 δ 미만이면 융합 자체를 건너뛰고 정책의 원래 게이팅을 사용한다. 나아가 반복적으로 δ를 넘지 못한 항목은 프루닝 대상 3순위로 지정되어 뱅크에서 제거된다. 다만 δ 값 자체의 민감도가 보고되지 않은 점은 한계(§11-1).

---

## 12. 종합 평가

**기여도**: ★★★★☆ — "메모리에 무엇을 넣을 것인가"에 대해 skill-level 게이팅 프로파일이라는 구체적이고 검증 가능한 답을 제시. MemoryVLA/HAMLET 계열의 다음 단계로 자연스럽게 위치한다.

**실험 견고성**: ★★★★☆ — 4개 백본 × 시뮬레이션 3종 + LIBERO + 실로봇 8개 조합 task로 폭이 넓고, 특히 cross-suite zero-shot 결과가 메커니즘 주장을 직접 뒷받침. 다만 λ/δ/L_PID ablation 부재가 감점 요인.

**실용성**: ★★★★☆ — 백본을 갈아끼우지 않고 +0.1~0.4B로 붙일 수 있는 wrapper라는 점이 강력. 코드 미공개와 추론 오버헤드 미분석이 진입 장벽.

**총평**: 메모리 증강 VLA의 저장 단위를 "관측/요약"에서 "절차적 스킬 서명"으로 옮긴 설득력 있는 제안. LIBERO 98.0%라는 SOTA 수치보다도, **Goal로 학습한 모델이 미지 Spatial에서 in-distribution baseline을 능가**한 Table 4의 결과가 이 논문의 진짜 메시지다. 벤치마크 포화 구간에서 "점수 1.2%p"보다 "조합 일반화 +10%p"에 주목해야 할 논문. 다만 이론적 핵심인 PID 손실과 메모리 하이퍼파라미터에 대한 검증이 후속 작업으로 반드시 필요하다.

<!-- VERIFIED: pdf -->
