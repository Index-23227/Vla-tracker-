# DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving

> **한 줄 요약**: 주행 VLA가 "같은 상황에서 같은 실수를 반복"하는 문제를 정면으로 다뤄, 오라클 PDM 시뮬레이터로 걸러낸 **실패 사례만을 latent memory에 적재**하고, vision-language feature가 아니라 **도로 구조(map)와 동적 agent를 분리한 구조적 retrieval key**로 유사 실패를 찾아, **Decoupled LoRA 기반 test-time training**으로 Action Decoder만 시나리오별로 즉석 보정하여 NAVSIMv1 Navtest 92.3 PDMS (메모리 10K 확장 시 94.1) / NAVSIMv2 Navhard 47.0 EPDMS를 달성한 연구.

- **arXiv**: 2608.10413v1 (2026-08-11, cs.CV)
- **소속**: Institute of Automation, Chinese Academy of Sciences / Chongqing Chang'an Technology Co., Ltd.
- **형식**: ACM MM '26 (Rio de Janeiro), DOI 10.1145/3767308.3835233, 15 pages
- **코드**: https://github.com/ZebinX/DriveVLA-M0

---

## 1. 배경 및 동기

VLA 기반 end-to-end 자율주행은 perception–language–planning을 하나의 정책으로 통합한다는 점에서 각광받아 왔다. LMDrive가 VLM의 주행 명령 수행 가능성을 보였고, 이후 언어 모델 기반 reasoning이 복잡한 주행 시나리오 성능을 끌어올렸다.

그러나 저자들이 지목하는 결함은 명확하다.

> **기존 VLA는 과거의 실패를 활용하는 메커니즘이 없다.** 파라미터가 고정된 채 배포되므로, 이전에 실패했던 것과 유사한 상황에서 **지속적으로 같은 실수를 반복**하고, distribution shift에도 적응하지 못한다.

동기는 인지과학에서 온다. Botvinick et al.의 conflict monitoring, Brown & Braver의 anterior cingulate cortex 오류 예측 연구를 인용하며, **인간은 현재 상황을 과거의 오류와 연결(associative retrieval)하여 실패 가능성을 예측하고 행동을 조정한다**는 점을 지적한다. DriveVLA-M0는 이 메커니즘을 명시적 latent memory + test-time adaptation으로 구현한다.

📌 [Figure 1 삽입] — (a) Classic VLA (VLM → Action Decoder 단방향) vs (b) Ours (latent memory retrieve → gradient descent 주입) vs (c) Navtest/Navhard SOTA 위치

---

## 2. 핵심 문제 정의: 왜 vision-language feature로 retrieve하면 안 되는가

Embodied AI 쪽의 memory-augmented VLA(MemoryVLA, EchoVLA)는 대체로 **intermediate vision-language feature를 retrieval key로** 쓴다. 저자들은 이것이 주행에서는 부적절하다고 주장한다.

주행 planning이 의존하는 본질적 정보는 두 가지다.
1. **Dynamic information**: 주변 agent의 움직임
2. **Scene structure information**: 도로 topology

Vision-language feature는 고수준 semantic은 잡지만 도로 경계·차선·agent 배치 같은 **구조적 속성**은 흐려진다. 결과적으로 "의미는 비슷하지만 구조적으로는 전혀 다른" 사례가 retrieve되고, 이는 안전 판단에 오히려 해롭다. (이 주장은 §4의 Table 3 ablation에서 실제로 검증된다 — 언어 feature retrieval은 base model보다 **성능이 떨어진다**.)

따라서 저자들이 요구하는 두 성질은:
- **Structural retrieval**: vision-language 유사도가 아닌 dynamic + physical structure 유사도
- **Failure awareness**: base model이 실제로 못 하는 시나리오에 집중

---

## 3. 방법론: 전체 아키텍처

파이프라인은 학습 단계(Base Model + Retrieve Model 학습)와 배포 단계 2-stage로 구성된다.

**[M] Memory Generation (offline)**: base model이 학습/외부 데이터에 대해 추론 → 오라클 시뮬레이션으로 실패 판정 → 중간 표현을 latent memory에 기록
**[I] Inference with TTT (online)**: Retrieve Model이 구조적으로 유사한 실패 사례를 검색 → Decoupled LoRA로 planning head를 test-time 학습 → 보정된 궤적 출력

Algorithm 1의 흐름을 요약하면:

```
[M] Offline
  for s in D:
      tau_hat, features <- B(s)                 # base model 추론
      (F_map, F_agent)  <- R(I)                 # retrieval key
      Q(tau_hat)        <- PDM(tau_hat)         # 오라클 채점
      if Q(tau_hat) < beta:  M <- M ∪ (k, x, y) # 실패만 저장

[I] Online
  (F_lang, F_ego, T_hat) <- B_enc(I, z)
  C_map, C_agent <- top-k retrieve from M with sim > lambda
  if both empty: return tau_0                   # TTT 스킵
  init Map LoRA theta_m, Agent LoRA theta_a
  for i in 1..S: theta <- theta - eta * grad L_TTT(C_map, C_agent)
  return argmax PDMS-style aggregated score
```

📌 [Figure 2 삽입] — 전체 개요: 상단 Memory Generation(Simulation → Failure? → Add to Memory), 하단 Inference with TTT(Retrieve → Trigger → Decoupled LoRA Router A/B)

---

## 4. Base Model 상세

### 4.1 VLM Backbone
ReCogDrive를 따라 **InternVL3**를 주행 QA 데이터로 fine-tune한 것을 백본으로 사용한다. 전방 카메라 이미지 `I ∈ R^{C×W×H}`와 시스템 프롬프트 `T`를 각각 인코딩해 통합 공간으로 매핑하고, LLM의 **마지막 레이어 feature h_{-1}**을 중간 scene 표현으로 뽑는다.

문제는 크기다. `h_{-1}`은 **2800 × 1536 토큰**으로 메모리 부담이 과도하다. 이를 Q-Former 스타일 **Compress Module**(학습 가능한 쿼리 `Q_cmp ∈ R^{N×D}`, N=16, D=256)로 cross-attention 압축한다.

```
F_lang = Transformer(Q_cmp, Linear(h_-1), Linear(h_-1))
```

→ **1050배 압축**된 `F_lang ∈ R^{16×256}`.

### 4.2 Action Decoder (proposal-and-score)
score-based planning 계열(iPad, GTRS 등)을 따라 2단 구조를 쓴다.

**Trajectory Head**: ego-status를 경량 MLP로 `F_ego ∈ R^{1×D}`로 압축한 뒤, 학습 가능한 임베딩 `Q_ego ∈ R^{M×D}`와 함께 `F_lang`을 cross-attend하여 **M개의 다양한 궤적 모드 클러스터** `T_hat ∈ R^{M×8×3}` 생성.

```
F_proposals = Transformer((Q_ego + F_ego), F_lang, F_lang)
T_hat       = MLP(F_proposals)
```

**Score Head**: `T_hat`을 재인코딩해 `F_lang`과 융합, proposal당 **K개의 sub-score**(collision rate, drivable area compliance 등 안전 기준)를 예측. 최종적으로 aggregated score 최대 궤적을 선택.

**학습 (Appendix A)**:
- 1단계 VLM pre-training: RecogDrive가 수집한 12개 공개 데이터셋(Talk2Car, SUTD, NuScenes-QA, OmniDrive 등) 총 **3.1M QA pair** → 고품질 필터링 후 NAVSIM 학습용 **약 775K** 유지. Compress Module은 이 단계에서 제외.
- 2단계: **VLM을 freeze**하고 Compress Module + Action Decoder만 학습.
  - `L_total = L_traj + L_score`
  - `L_traj = min_i ||tau_hat_i - tau*||_1` (min-over-N → 다양성 확보하면서 최소 하나는 expert에 근접)
  - `L_score = -1/N Σ [s_i log(s_hat_i) + (1-s_i) log(1-s_hat_i)]`, soft target `s_i`는 PDM scorer가 부여

> 💡 **세미나 포인트**: VLM을 freeze한다는 점이 중요하다. 이 논문의 TTT는 **Action Decoder(planning head)만** 건드리며 backbone은 수정하지 않는다. 그래서 latency overhead가 26.44 ms에 그친다.

---

## 5. Retrieve Model: 구조적 검색 키의 설계

경량 **DINOv2**를 feature extractor로 쓰고 LoRA로 fine-tune하되, 정적 도로와 동적 agent를 독립 적응시키기 위해 **Decoupled LoRA**로 Map(static) / Agent(dynamic) 두 branch로 분리한다.

```
F_map, F_agent = DINO_LoRA(I)          # 두 branch 병렬 라우팅
```

각 feature map은 학습 가능한 쿼리를 가진 Transformer decoder로 집약되고, 별도 head가 이를 **occupancy grid** 출력 `M_hat_map`, `M_hat_agent`로 매핑한다.

**학습 (Transfuser의 occupancy-grid supervision 패러다임)**:
```
L_Retrieve = BCE(M_hat_map, M_map) + alpha * BCE(M_hat_agent, M_agent)
```
`alpha = 10` — agent loss의 크기가 map loss보다 훨씬 작아서 균형을 맞추기 위함(Appendix B).

Attention map 시각화(Figure 3) 결과: **map embedding은 도로 경계·차선 표식 등 topology에**, **agent embedding은 전방 차량에** 집중. 의도한 disentanglement가 실제로 나타났다.

📌 [Figure 3 삽입] — query scene vs retrieved scenes의 map/agent attention map
📌 [Figure 5 삽입] — Retrieve Model 구조 (DINO + Transformer + Query)

---

## 6. Memory Generation: 무엇을 저장하는가

각 memory case는 세 범주를 담는다.

| 범주 | 내용 | 용도 |
|------|------|------|
| `k` (Retrieval keys) | `F_map`, `F_agent` | 구조적 검색 |
| `x` (Adaptation inputs) | `F_lang`, `F_ego`, 궤적 클러스터 `T_hat` | TTT 입력 |
| `y` (Supervision targets) | expert 궤적 `tau`, 오라클 sub-score `S` | TTT 지도 신호 |

```
M = {(k_i, x_i, y_i)}_{i=1}^N
```

**실패 판정**: base model 예측 `tau_hat`을 오라클 PDM scorer로 시뮬레이션 채점, K개 안전·승차감 sub-score를 곱셈·덧셈 조합으로 집약해 `Q(tau_hat) ∈ [0,1]`을 얻고, **`Q(tau_hat) < beta`이면 실패 사례로 기록**. Appendix B에 따르면 `beta = 0.5` — NAVSIM 채점 관례상 NC/DAC가 0.5 미만이면 충돌 또는 주행가능영역 위반을 의미하기 때문이다.

**중복 제거**: 대규모 외부 데이터를 넣을 때 확장성을 위해, 기록 전 cosine similarity 기반 dedup으로 충분히 유사한 시나리오가 이미 있으면 건너뛰어 메모리 풀 크기를 관리 가능한 수준으로 묶는다.

---

## 7. Inference with TTT: 주입 메커니즘

### 7.1 Trigger
TTT는 **실패 확률이 높은 시나리오에서만** 정당화된다. 따라서 cosine similarity 기반 이진 게이트를 둔다.

```
g = 1  if  (F^T F*) / (||F||_2 ||F*||_2) > lambda,  else 0
```
`F* ∈ {F*_map, F*_agent}`는 메모리에서 검색된 feature. `C_map`과 `C_agent`가 모두 비면 TTT 없이 base 궤적을 그대로 반환한다.

### 7.2 Decoupled LoRA TTT
검색된 `{x, y}`로 Action Decoder의 LoRA를 fine-tune하되, **LoRA 가중치는 테스트 시나리오마다 재초기화**되어 시나리오 특화 적응을 보장한다. 분리 설계는 다음과 같다.

- **map으로 검색된 사례 → Map(static) LoRA branch**
- **agent로 검색된 사례 → Agent(dynamic) LoRA branch**

추론 시 두 branch가 각각 독립적인 score를 예측하고, **pathway-aware score fusion**을 적용한다.
- drivable-area compliance 같은 **도로 이해형 sub-score → Static LoRA 예측 채택**
- collision avoidance 같은 **동적 능력형 sub-score → Dynamic LoRA 예측 채택**

### 7.3 하이퍼파라미터 (Appendix B)
| 항목 | 값 |
|------|-----|
| Retrieve Model alpha | 10 |
| 실패 임계 beta | 0.5 |
| Trigger 임계 lambda | 0.9 |
| TTT optimizer | AdamW, lr 2e-4 |
| TTT steps | 3 |
| Memory 크기 | ~4K (Base) / 10K (Scale) |
| 계층적 검색 | map으로 top-k1 → agent로 top-k2, k1 = 3·k2, k2 = 3 |

> 계층적 검색은 순수한 엔지니어링 대응이다. NAVSIM에는 주변 차량·보행자가 없는 **빈 도로 구간이 많아서**, agent embedding만으로 검색하면 노이즈가 들어온다. 그래서 map으로 먼저 거른다.

📌 [Figure 4 삽입] — TTT 주입 전(상)/후(하) 궤적 분포. 주입 후 GT score density가 우측 이동 = 클러스터 품질 자체가 개선

---

## 8. 실험 결과

### 8.1 벤치마크
- **NAVSIMv1 (Navtest)**: nuPlan/OpenScene 센서 데이터 기반 non-reactive 시뮬레이션. BEV 추상화를 짧은 horizon으로 unroll. 주 지표 **PDMS** = NC, DAC, EP, TTC, C의 곱셈·가중합 조합.
- **NAVSIMv2 (Navhard)**: pseudo closed-loop 2-stage 프로토콜. **EPDMS**는 PDMS에 LK, DDC, TLC, EC를 추가하고, 인간 운전자도 같은 위반을 저지른 경우 페널티를 무효화하는 false-positive filtering을 도입. Stage 1은 초기 계획 궤적, Stage 2는 Stage 1 결과에서 분기한 사전계산 후속 장면을 채점.

### 8.2 NAVSIMv1 (Table 1)

| 구분 | Method | NC | DAC | EP | TTC | C | **PDMS** |
|------|--------|----|-----|----|-----|---|----------|
| Classic E2E | TransFuser | 97.7 | 92.8 | 84.0 | 92.8 | 100 | 83.4 |
| | DiffusionDrive | 98.2 | 96.2 | 82.2 | 94.9 | 99.9 | 88.1 |
| | GoalFlow | 98.4 | 98.3 | 85.0 | 94.6 | 100 | 90.3 |
| | iPad | 98.6 | 98.3 | 88.0 | 94.9 | 100 | 91.7 |
| | Centaur (TTT 계열) | 99.5 | 98.9 | 85.9 | 98.0 | 100 | 92.6 |
| | DriveSuprim | 98.6 | 98.6 | 91.3 | 95.5 | 100 | 93.5 |
| VLA | MTRDrive | 97.3 | 95.8 | 86.8 | 91.2 | 100 | 88.3 |
| | AutoVLA | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| | DriveVLA-W0 | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | 90.2 |
| | ReCogDrive | 97.9 | 97.3 | 87.3 | 94.9 | 100 | 90.8 |
| | ELF-VLA | 98.9 | 98.1 | 85.3 | 96.0 | 100 | 91.0 |
| | MTDrive | 97.5 | 98.2 | 90.6 | 91.8 | 99.8 | 91.1 |
| **Ours** | **DriveVLA-M0-Base** | 99.0 | 97.7 | 89.5 | 95.0 | 99.9 | **92.3** |
| | **DriveVLA-M0-Scale** | 99.1 | 98.1 | 90.2 | 98.5 | 99.9 | **94.1** |

전방 카메라 **한 대만** 쓰면서 모든 기존 VLA 계열을 앞선다. 특히 Centaur도 TTT 패러다임을 쓰는데, Base 설정에서는 92.3 vs 92.6으로 근소하게 뒤지지만 Scale 설정(94.1)에서는 DriveSuprim(93.5) 포함 표 전체를 넘어선다.

### 8.3 메모리 스케일링
SimScale로 시나리오를 합성해 latent memory를 **10K 케이스로 확장**하되 **재학습은 전혀 하지 않는다**. 새 실패 사례를 메모리에 넣고 추론 시 검색하기만 하면 되므로 92.3 → **94.1 PDMS**의 이득이 나온다. 이것이 이 논문에서 가장 실용적으로 매력적인 주장이다 — **post-training-free scaling**.

### 8.4 NAVSIMv2 Navhard (Table 2)

DriveVLA-M0-Base Stage 1 / Stage 2:

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | **EPDMS** |
|---|----|-----|-----|-----|----|-----|----|----|----|-----------|
| Stage 1 | 98.9 | 96.2 | 99.7 | 100.0 | 73.8 | 99.1 | 94.2 | 97.3 | 64.4 | |
| Stage 2 | 91.1 | 89.5 | 93.1 | 98.6 | 65.5 | 89.2 | 52.7 | 98.6 | 72.1 | **47.0** |

비교군: LTF 23.1, DiffusionDrive 28.9, GoalFlow 28.7, Mimir 34.6, DriveSuprim 44.7, GTRS-Dense 45.3 → **47.0으로 최고**. Navhard는 신설 벤치마크라 VLA 계열 보고가 거의 없어 강한 E2E baseline과 비교했다. Transfuser 대비 NC/DAC가 개선되고, scoring 기반 GTRS 대비 궤적 연속성(EC 72.1 vs 54.2)에서 앞선다.

---

## 9. Ablation 분석

### 9.1 검색 키 (Table 3) — 논문의 핵심 주장 검증
모든 ablation의 기준 base model은 **PDMS 91.0**.

| Search Type | NC | DAC | EP | TTC | C | PDMS |
|-------------|----|-----|----|-----|---|------|
| Base Model† (메모리 없음) | 98.4 | 97.1 | 87.7 | 95.2 | 97.6 | 91.0 |
| Lang | 98.0 | 97.3 | 88.1 | 93.9 | 100.0 | **90.7** |
| Map | 98.4 | 97.7 | 89.1 | 94.5 | 99.9 | 91.7 |
| Map + Agent | 98.9 | 97.7 | 89.6 | 95.0 | 99.9 | **92.3** |

> **가장 중요한 수치**: 언어 임베딩 검색은 90.7로 **base model(91.0)보다 낮다**. "vision-language space 검색은 주행에서 오히려 해롭다"는 §2의 주장을 정면으로 뒷받침한다. Map만 써도 91.7, agent를 더하면 92.3(+1.3). Map 대비 agent 추가는 NC/EP/TTC를 각각 0.5씩 올린다 — map은 topology(DAC, EP), agent는 충돌계(NC, TTC)를 담당한다는 분리 설계가 지표별로 정확히 나타난다.

### 9.2 주입 전략 (Table 4)

| Injection Type | NC | DAC | EP | TTC | C | PDMS |
|----------------|----|-----|----|-----|---|------|
| Base Model† | 98.4 | 97.1 | 87.7 | 95.2 | 97.6 | 91.0 |
| Offline (10 epochs LoRA post-training) | 98.1 | 97.5 | 88.6 | 94.2 | 99.9 | 91.2 |
| TTT Full | 99.0 | 97.8 | 89.6 | 95.1 | 99.9 | **92.4** |
| TTT LoRA (제안) | 99.0 | 97.8 | 89.5 | 95.0 | 99.9 | 92.3 |

실패 사례로 **오프라인 post-training(10 epoch)을 해도 91.2에 그친다.** 저자들은 고정된 데이터 혼합으로 학습되어 시나리오 특화 적응이 없기 때문(distribution mismatch)이라고 해석한다. 반면 TTT는 92.4/92.3. **Decoupled LoRA가 full fine-tuning과 사실상 동률(-0.1)이면서 backward latency를 55.42 → 26.44 ms로 절반 이하로 줄인다.**

### 9.3 Trigger 임계값 (Table 5, map-only 검색)

| lambda | NC | DAC | EP | TTC | C | PDMS |
|--------|----|-----|----|-----|---|------|
| 0.70 | 98.0 | 96.9 | 88.1 | 93.8 | 99.9 | 90.4 |
| **0.90** | 98.4 | 97.7 | 89.1 | 94.5 | 99.9 | **91.7** |
| 0.95 | 98.1 | 97.6 | 88.8 | 94.3 | 99.9 | 91.4 |
| 0.99 | 98.1 | 96.9 | 85.2 | 94.3 | 99.9 | 89.4 |

양방향 실패 모드가 뚜렷하다. **너무 관대하면(0.70)** 무관한 시나리오까지 끌어와 noisy supervision → NC/TTC 저하(90.4). **너무 엄격하면(0.99)** 거의 트리거되지 않고 89.4로 **base model(91.0)보다도 떨어진다**. 선택적 적응이 무차별 test-time fine-tuning보다 낫다는 설계 원칙의 근거.

### 9.4 Latency (Table 6, 단일 H20, 메모리 4,000 케이스)

| Component | Retrieve | Forward | Backward LoRA | Backward Full |
|-----------|----------|---------|---------------|---------------|
| Time (ms) | 15.19 | 30.79 | **26.44** | 55.42 |

검색 15.19 ms, forward 30.79 ms. TTT가 트리거될 때만 26.44 ms가 추가된다.

---

## 10. 비판적 검토

**강점**
1. **문제 설정이 정직하다.** "실패를 재사용한다"는 아이디어는 흔하지만, 오라클 PDM 점수로 실패를 **정량적으로 정의(beta=0.5)**하고, 그 사례만 저장하는 선택은 깔끔하다.
2. **핵심 주장을 반증 가능한 형태로 검증했다.** Lang retrieval이 base보다 낮다는 결과(90.7 < 91.0)는 저자 주장에 유리한 방향이지만, 동시에 "메모리를 잘못 쓰면 오히려 손해"라는 리스크를 스스로 드러낸 정직한 보고다.
3. **재학습 없는 스케일링.** 92.3 → 94.1을 메모리 확장만으로 얻는다는 것은 배포된 차량 fleet에서 수집한 실패를 계속 축적하는 운영 모델과 잘 맞는다.
4. **Decoupled LoRA의 지표별 분업**이 attention 시각화(Figure 3)와 ablation 지표 변화(NC/TTC vs DAC/EP)로 이중 검증됐다.

**약점 및 의문**
1. **오라클 의존성.** 메모리 구축에 시뮬레이션 PDM scorer(정답 궤적과 시뮬레이터 필요)가 반드시 있어야 한다. 실제 도로 주행 로그에서는 "실패"를 이렇게 채점할 수 없다. Fleet-scale 확장 주장과 이 요구사항 사이에 긴장이 있다.
2. **Ablation base(91.0)와 Table 1 base(92.3)의 관계.** Table 3/4의 Base Model†는 91.0인데 Table 1의 DriveVLA-M0-Base는 92.3이다. 즉 Table 1의 "Base"는 이미 메모리+TTT가 켜진 설정이고, "Scale"은 메모리 10K 설정이다. 표기가 다소 혼동을 부른다.
3. **NAVSIMv2 결과가 Base 단일 행뿐이다.** Navhard에서 memory scaling 효과가 재현되는지 보고되지 않았다. 또한 Stage 2의 LK 52.7은 눈에 띄게 낮아, 장기 horizon에서 차선 유지가 취약할 가능성이 있다.
4. **전방 카메라 단일 입력**은 효율 측면의 장점이지만, 측후방 정보가 필요한 시나리오(합류, 차선 변경)에서의 한계는 논의되지 않았다.
5. **파라미터 수 미보고.** InternVL3 기반이라고만 밝히고 정확한 모델 크기가 명시되지 않아 다른 VLA(ReCogDrive-8B 등)와의 공정 비교가 어렵다.
6. **VLM은 freeze되어 있고 TTT는 Action Decoder만 건드린다.** 따라서 "실패로부터 배운다"기보다 **"실패 사례로 planning head를 순간 캘리브레이션한다"**에 가깝다. reasoning 자체는 교정되지 않는다.

---

## 11. 관련 연구 속 위치

- **메모리 계열 대비**: MANTRA/MemoNet(궤적 예측 메모리), JARVIS-1/MemGen(에이전트 episodic memory), MemoryVLA(perceptual-cognitive memory), MTRDrive(코너케이스 프로토타입 저장), EvoVLA(자기진화)와 달리, 검색 공간을 **vision-language space가 아닌 구조적 occupancy space**로 옮긴 것이 차별점.
- **TTT 계열 대비**: TENT(entropy 최소화), TTT++, LoRA-TTT, Centaur(cluster entropy 최소화)는 **비지도 대리 목적**을 쓴다. DriveVLA-M0는 메모리가 **expert 궤적과 오라클 sub-score라는 명시적 지도 신호**를 제공한다는 점에서 더 강한 학습 신호를 갖는다.
- **실패 활용 계열 대비**: SERA, ELF-VLA, BeyondDrive, SoAD는 실패/안전 값을 쓰되 **광범위한 post-training**을 요구한다. DriveVLA-M0는 실패 맥락을 latent memory로 압축해 **오프라인 재학습 없이** test-time에 직접 적응한다.

---

## 12. 총평 및 시사점

DriveVLA-M0는 "주행 VLA에 메모리를 붙인다"는 흔한 아이디어를, **(a) 무엇을 저장할 것인가(오라클이 실패로 판정한 것만), (b) 무엇으로 찾을 것인가(도로 구조와 agent를 분리한 구조적 키), (c) 어떻게 주입할 것인가(트리거로 게이트된 decoupled LoRA TTT)** 세 질문으로 분해하고 각각을 ablation으로 방어한 잘 설계된 논문이다.

특히 **언어 임베딩 검색이 base보다 나쁘다(90.7 < 91.0)**는 결과와 **트리거가 너무 엄격하면 89.4로 붕괴**한다는 결과는, 메모리 증강이 공짜 점심이 아니며 **검색 품질과 적응 게이팅이 성능을 좌우한다**는 일반적 교훈을 준다. Embodied manipulation VLA에 메모리를 도입하려는 연구자에게도 그대로 이식 가능한 시사점이다.

한계는 오라클 시뮬레이터 의존성과 freeze된 VLM이다. 실차 로그에서 실패를 자동 라벨링하는 방법, 그리고 backbone reasoning 자체를 교정하는 확장이 후속 과제로 자연스럽다.

**VLA-Tracker 관점**: 자체 학습된 주행 VLA policy(InternVL3 백본 도메인 사전학습 + Trajectory/Score Action Decoder imitation learning)이며, frozen 정책 위의 래퍼가 아니다. 메모리·TTT는 그 위에 얹힌 추론 시 보정 메커니즘이다.

---

<!-- VERIFIED: pdf -->
