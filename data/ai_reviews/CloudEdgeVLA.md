# CloudEdgeVLA: Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization

**arXiv**: [2608.00569](https://arxiv.org/abs/2608.00569) · **발표일**: 2026-08-01 · **소속**: 홍콩과기대 광저우(HKUST(GZ)) / 산둥대학교(Shandong University) / RoboScience
**저자**: Daojie Peng, Fulong Ma, Bingtao Wang, Sheng Wang, Jun Ma
**코드**: 미공개

> **한 줄 요약**: 클라우드-엣지 VLA의 시간 어긋남(temporal misalignment)을 **스케줄링 문제가 아니라 표현 학습 문제**로 재정의한다. 클라우드 7B 백본은 지연된 관측으로부터 **천천히 변하는 task feature**를 만들고, 엣지의 경량 **Vision-Augmented Action Head**가 가장 최근에 도착한 클라우드 feature와 **현재** 로컬 시각을 융합해 행동을 낸다. 학습 시 현재 프레임과 무작위 지연 프레임을 **같은 현재 행동 타깃**으로 짝지어 supervise(paired-frame dual-path)함으로써, 지연 메타데이터·시계 정렬 없이도 40-step 균일 지연에서 LIBERO 63.8–78.0%를 유지한다(VLASH 최대 6.4%, 단일 경로 baseline 최대 3.0%).

---

## 1. 문제 정의 (Problem Statement)

수십억 파라미터 VLA를 모바일 로봇에 올리려는 순간 **시스템 차원의 충돌**이 생긴다. 의미 추론은 클라우드 GPU에서 이득을 보지만, closed-loop 제어는 네트워크 지연과 jitter에도 불구하고 **로컬에서 즉시** 반응해야 한다.

핵심은 이렇다. 클라우드가 반환한 feature는 **그 입력 이미지가 촬영된 시점의 장면**을 기술할 뿐, 행동이 실행될 시점의 장면이 아니다. 두 선택지 모두 나쁘다 — 새 feature가 올 때까지 **블로킹**하면 제어 주파수가 떨어지고, 최신 feature로 **그냥 실행**하면 시간적 불일치가 들어온다.

형식화하면: 엣지가 o_t를 보내고 클라우드는 지연 때문에 o_{t-k}를 받아 h_{t-k} = f_θ(o_{t-k}, ℓ)를 만든다. 엣지는 **stale한 h_{t-k}와 자신의 실시간 관측 o_t만으로** a_t를 내야 한다. 즉 â_t = g_φ(h_{t-k}, v_ψ(o_t))인 action head g_φ를 학습하는 문제다. 기존 multi-rate 정책(MResT, DP-VLA, HiRT, SmolVLA)은 추론 스케줄링·action chunk 보정·명시적 시간 조건화를 다루지만, **네트워크를 건너가는 표현 자체가 나이(age)를 사전에 모른 채로도 유용해야 한다**는 클라우드-엣지 고유 요구는 다루지 않는다.

## 2. 핵심 기여 (Key Contributions)

1. **비블로킹 인터페이스로서의 비동기 클라우드-엣지 VLA 정식화**: 경량 Vision-Augmented Action Head가 **최신 클라우드 feature + 현재 엣지 시각**을 시계 정렬이나 주파수 비율 α 없이 융합한다. 엣지는 특정 업데이트를 절대 기다리지 않는다.
2. **Paired-frame dual-path training**: 현재 프레임과 무작위 지연 프레임에서 계산한 클라우드 feature를 **동일한 현재 행동**에 대해 공동 supervise. 명시적 invariance loss 없이 **시간적 역할 분화(temporal specialization)**를 유도한다.
3. **행동이 아니라 표현을 보낸다**: VLASH·A2C2처럼 stale action chunk를 굴리거나 보정하는 대신, 학습된 **task 표현**을 느린 경로로 보낸다. 추론 시 지연 메타데이터가 필요 없다.
4. **지연 민감도의 분해 진단**: 백본 drift D_h, 헤드의 staleness transfer gain κ = D_a/(D_h+ε)를 정의하여 강건성이 **백본 안정화**와 **헤드 감쇠** 중 어디서 오는지 분리 측정하고, 소규모 실기 sanity check를 덧붙인다.

## 3. 방법론 (Methodology)

**System 1 / System 2 유비**(Kahneman)를 시간적 역할로 번역한다. 클라우드 백본은 **무엇을 할 것인가**(task goal, 조작 전략, 물체 의미)를 담는 느리게 변하는 맥락을, 엣지 헤드는 **지금 어떻게 할 것인가**를 담는 현재 시각을 담당한다.

**Paired frame 추출.** 각 샘플은 같은 에피소드의 연속 관측 윈도우 W_t = {o_{t-W+1}, …, o_t}를 제공하고, 여기서 현재 프레임 o_t와 지연 프레임 o_{t-d}, d ~ Uniform(1, W-1)을 뽑는다.

**Dual forward pass.** h_fresh = f_θ(o_t, ℓ), h_stale = f_θ(o_{t-d}, ℓ). 중요한 것은 **h_stale이 계산 그래프에서 detach되지 않는다**는 점 — 양쪽 모두 백본으로 gradient를 흘린다.

**Vision-augmented 예측.** 두 planning feature를 **동일한 실시간 시각 feature** z_t = v_ψ(o_t)와 융합: â_fresh = g_φ(h_fresh, z_t), â_stale = g_φ(h_stale, z_t).

**Dual-path loss.** L = (1-λ)·‖â_fresh - a_t‖₁ + λ·‖â_stale - a_t‖₁. L_fresh는 동기 동작을 학습시키고 백본에 직접적인 행동 supervision을 준다. L_stale은 헤드가 stale feature를 **실시간 시각으로 보상(compensate)**하도록 만든다 — h_stale이 현재 상태와 어긋날수록 헤드는 z_t에 더 의존해야 한다. λ는 curriculum으로 0 → λ_max까지 n_warmup 동안 서서히 올려, 백본이 먼저 강한 표현을 학습한 뒤에 지연 불변성 압력을 받도록 한다.

**배포 프로토콜.** (1) 엣지가 o_t 촬영 → (2) 비동기·비블로킹 전송 → (3) 클라우드가 (아마도 이전 o_{t-k}로부터의) h를 반환하면 h_received 갱신 → (4) 엣지가 z_t = v_ψ(o_t) 로컬 계산 → (5) â_t = g_φ(h_received, z_t) → (6) 실행. **로봇은 절대 블로킹하지 않는다.**

**Emergent representational specialization.** ∇_θ L_stale = ∇_θ‖g_φ(f_θ(o_{t-d}, ℓ), v_ψ(o_t)) - a_t‖₁. 무작위 d에 걸쳐, o_{t-d}의 **순간 상태**에만 묶인 feature는 a_t의 신뢰할 수 없는 예측자인 반면 task 정체성·목표·거친 진행도는 더 안정적이다. 따라서 목적함수는 시간적 변위에 대한 불변성을 **장려하되 수학적으로 보장하지는 않는다**(저자들이 직접 명시). 상태 민감 정보는 z_t가 담당한다.

## 4. 아키텍처 상세 (Architecture Details)

- **클라우드 백본 f_θ (System 2, latency-insensitive)**: 사전학습 VLM 위에 LoRA를 얹은 대규모 모델. h = f_θ(o, ℓ) ∈ R^{L×D}, L = T × A (chunk 길이 T × 행동 차원 A). 클라우드 서버에서만 실행.
- **엣지 시각 인코더 v_ψ (System 1 지각, latency-sensitive)**: **SigLIP-Base**, 로봇 로컬에서 실행, **학습 중 frozen**. z_t = v_ψ(o_t) ∈ R^{D_v}는 항상 **현재 관측**에서 계산되는 **지연 없는(delay-free) 신호**.
- **엣지 action head g_φ (System 1 제어)**: planning feature를 행동 차원 축으로 mean-pool → 시각 feature를 동일 잠재 공간으로 projection → concat → **residual MLP** → action chunk â_t ∈ R^{T×A}. 클라우드의 고수준(그러나 stale할 수 있는) 지시를 현재 시각 맥락으로 **정밀 모터 명령으로 번역하는 실시간 grounding 레이어**.
- **구현 베이스**: **OpenVLA-OFT** (7B OpenVLA + LoRA + 병렬 action chunk 디코딩 + 연속 L1 회귀). 수정 사항은 (1) 기존 L1 Regression Action Head를 Vision-Augmented Action Head로 교체, (2) 같은 에피소드에서 과거 프레임을 뽑는 paired-frame RLDS 로더, (3) dual L1 loss의 dual-path forward, (4) 네트워크 지연을 시뮬레이션하는 지연 평가 프로토콜.
- **행동 차원**: x, y, z, roll, pitch, yaw, gripper의 7차원 × horizon t+0…t+7 (총 56 셀).

## 5. 실험 설정 (Experimental Setup)

- **벤치마크**: **LIBERO** 4개 suite(Spatial / Object / Goal / Long), suite당 10 태스크, 태스크당 50 demonstration 에피소드.
- **평가 조건**: **무지연**(d_max=0, 동기 기준선)과 **균일 지연** k ~ Uniform{1,…,d_max}, d_max ∈ {5, 10, 15, 20, 25, 30, 40} steps. d_max > 20은 **학습 윈도우 밖**이다.
- **프로토콜**: 태스크당 50 trial의 성공률(%). CloudEdgeVLA는 시드 7·8·9의 **3회 반복**으로 평균 ± 표준편차 보고(baseline은 published point estimate).
- **베이스라인**: 단일 경로 OpenVLA / OpenVLA-OFT / UniVLA(추론 경로를 바꾸지 않고 지연 프레임을 그대로 투입), 그리고 future-state-aware 비동기 **VLASH**(동일한 균일 지연 grid에서 평가).
- **실기**: Franka pick-and-place(장난감 곰을 상자에 넣기), static / dynamic(도달 중 목표를 10 cm 이내로 외부 변위) 변형, 변형당 10 trial, 추가 RTT 0 / 400 / 1000 ms(네이티브 지연 **위에** 얹은 emulated 값).

## 6. 주요 결과 (Main Results)

**Table 1 — 균일 지연 하 LIBERO suite별 성공률 (%)**:

| 방법 | d=0 Spat./Obj./Goal/Long | d=10 | d=20 | d=40 |
|---|---|---|---|---|
| OpenVLA | 84.6 / 71.2 / 77.0 / 56.2 | 6.8 / 1.2 / 2.2 / 5.2 | 0.2 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |
| OpenVLA-OFT | 98.4 / 98.6 / 97.2 / 93.4 | 10.6 / 3.6 / 26.2 / 8.4 | 0.0 / 0.0 / 4.0 / 1.4 | 0.0 / 0.0 / 0.0 / 0.0 |
| UniVLA | 96.0 / 96.6 / 94.6 / 93.2 | 27.4 / 34.8 / 48.2 / 35.0 | 0.4 / 2.6 / 21.8 / 2.2 | 0.0 / 0.0 / 3.0 / 0.5 |
| VLASH | 97.3 / 99.6 / 96.7 / 93.5 | 60.0 / 62.8 / 56.8 / 45.6 | 5.0 / 8.2 / 28.2 / 9.2 | 0.0 / 0.4 / 6.4 / 0.0 |
| **CloudEdgeVLA** | **97.9 / 97.8 / 96.5 / 91.7** | **93.6 / 94.4 / 92.9 / 83.2** | **89.6 / 92.1 / 90.9 / 78.1** | **76.4 / 75.6 / 78.0 / 63.8** |

Macro-average(4 suite 평균)는 d=0에서 **96.0%**, d=10 **91.0%**, d=20 **87.7%**, d=40 **73.5%**로, d_max=40에서 VLASH 대비 **+71.8 pp**다.

**요약 지표(Figure 5)**: 정규화 delay AUC는 CloudEdgeVLA **90.8%** vs VLASH 32.4 / UniVLA 25.0 / OpenVLA-OFT 18.4 / OpenVLA 12.5. d_max=40에서의 **동기 성능 유지율(retention)**은 **76.5%** vs VLASH **1.8%**, UniVLA 0.9%, 나머지 0.0%.

주목할 점: **동기 정확도와 중간 지연에서의 우위는 심한 staleness 내성을 함의하지 않는다.** VLASH는 d=0에서 CloudEdgeVLA보다 오히려 높고(Object 99.6 vs 97.8) d=10에서도 45.6–62.8%로 견디지만, d=40에서 0.0–6.4%로 붕괴한다. 열화 양상은 suite 의존적이다 — d=0 대비 하락 폭이 Long 27.9 pp > Object 22.2 > Spatial 21.5 > Goal 18.5 pp로, **Long이 가장 취약하고 Goal이 가장 안정적**이다. 최악인 Long에서조차 d=40에서 최강 baseline보다 **63.3 pp 높다**.

**Table 3 — 실기 성공률 (static/dynamic, 10 trial)**:

| 방법 | RTT 0 ms | 400 ms | 1000 ms |
|---|---|---|---|
| VLASH | 100 / 90 | 60 / 30 | 0 / 0 |
| **CE-VLA** | 100 / 90 | **90 / 90** | **80 / 70** |

저자들은 이것을 하드웨어 벤치마크가 아닌 **물리적 실현 가능성 확인(feasibility check)**이라고 명시한다.

## 7. Ablation 분석 (Table 2, d_max=10 집계 성공률)

| 변형 | Success (%) |
|---|---|
| 시각 인코더 없음 (stale h만) | 31.6 |
| **SigLIP-Base (frozen, default)** | **95.1** |
| SigLIP-SO400M (frozen) | 95.8 |
| L_fresh only | 54.8 |
| L_stale only | 89.2 |
| **L_fresh + L_stale (Ours)** | **95.1** |

두 가지 명확한 결론이 나온다.

**(a) 결정적 아키텍처 요인은 "헤드가 현재 관측을 받는가"다.** frozen SigLIP-Base를 붙이는 것만으로 31.6% → 95.1%, **63.5 pp** 상승. 훨씬 큰 SigLIP-SO400M으로 바꿔도 95.8%로 **0.7 pp** 차이뿐이다. 즉 이득의 대부분은 **인코더 규모가 아니라 실시간 시각 grounding** 자체에서 오며, 이는 엣지 배포용 경량 Base 모델 선택을 정당화한다.

**(b) 지연 노출이 강건성의 주 동력이고 fresh 경로는 보완재다.** stale-only가 89.2%로 fresh-only(54.8%)보다 **34.4 pp** 높다. 공동 supervision이 95.1%로 stale-only 대비 **5.9 pp**를 추가 회복하므로, fresh 경로는 지연 노출을 대체하는 것이 아니라 보완적 grounding을 제공한다.

## 8. 메커니즘 진단 — 백본과 헤드의 분해 (Table 4, 부록 A–E)

저자들은 성공률만으로 specialization을 추론하지 않고 인터페이스 자체를 측정한다. LIBERO-Spatial 10 태스크의 **80개 공유 demonstration 상태**에서, 클라우드 백본 이미지만 d step 지연시키고 proprioception은 현재로 유지한 채:

- D_h(d) = E[1 - cos(h_t, h_{t-d})] — 백본 표현 staleness
- D_a(d) = E[‖â_t^(0) - â_t^(d)‖] — end-to-end 행동 drift
- **κ(d) = D_a(d)/(D_h(d)+ε)** — 헤드가 잔여 표현 drift를 행동으로 **전달하는 비율**

d=20에서 OpenVLA-OFT → CloudEdgeVLA(120k step): D_h **0.391 → 0.160 (−59.2%)**, κ **1.082 → 0.295 (−72.7%)**, D_a **0.423 → 0.047 (−88.9%)**, demo MAE **0.421 → 0.048 (−88.5%)**. 즉 강건성이 **인터페이스 양쪽 모두**에 국소화된다 — 더 안정적인 백본 **그리고** 덜 전달하는 헤드.

**정직한 통제 장치들**:
- **"둔감한 예측기" 반박**: d=0에서 demo MAE는 CE 0.023 vs OFT 0.020으로 CloudEdgeVLA가 **약간 나쁘고**, d=1부터 역전되어 격차가 벌어진다. 즉 지연 학습은 fresh 정확도를 조금 희생해 강건성을 얻은 것이지, 단순히 출력이 안 변하는 예측기를 만든 게 아니다.
- **태스크 편중 반박**: d=20에서 **10개 태스크 전부**에서 MAE가 낮고, 감소폭은 0.278–0.566 범위.
- **첫 스텝 편중 반박**: 억제가 **56개 horizon×dimension 셀 전부**에서 나타나며 평균 0.376(범위 0.214–0.651). x·z 병진의 평균 감소가 가장 크지만 회전·gripper도 horizon 전반에서 개선.
- **결정적 negative result (부록 E)**: 백본 나이 d_h와 엣지 나이 d_z를 **독립적으로** 변화시키면, d_h=20에서 엣지 입력을 stale → current로 바꿔도 평균 행동 drift가 0.047104 → 0.047097로, **edge-rescue fraction이 겨우 0.03%**이고 보정 벡터의 cosine alignment도 0.030에 불과하다. 저자들은 이 120k 체크포인트의 이득이 **현재 엣지 시각의 직접 수정이 아니라 백본 안정성과 헤드 감쇠에 지배된다**고 스스로 결론짓는다.

## 9. 강점 (Strengths)

1. **압도적 지연 강건성 격차**: d_max=40에서 73.5% vs VLASH 1.7% macro-average. 이 정도 규모(+71.8 pp)의 차이는 측정 잡음이나 튜닝으로 설명되지 않는다.
2. **학습 윈도우 밖으로의 외삽**: d_max > 20은 학습 분포 밖인데도 d=40에서 63.8–78.0%를 유지한다.
3. **다중 시드 보고**: 시드 7·8·9의 평균±표준편차(0.20–1.51)를 보고하여, point estimate만 제시하는 관행보다 신뢰도가 높다.
4. **추론 시 메타데이터 불필요**: 지연 값 k, 시계 동기화, 주파수 비율 α 중 어느 것도 필요 없다. 실제 네트워크에서 이 조건들은 거의 만족되지 않으므로 실용적 의미가 크다.
5. **메커니즘을 측정으로 검증**: κ라는 전달 이득 지표를 정의해 백본과 헤드의 기여를 분리하고, "둔감한 예측기" 대안 설명을 E_demo로 명시적으로 배제한다.
6. **부정적 결과를 은폐하지 않음**: 논문의 서사적 핵심(현재 엣지 시각이 stale feature를 보정한다)을 반증하는 counterfactual audit(edge-rescue 0.03%)을 부록에 그대로 싣고 메커니즘 주장을 스스로 축소한다. 드문 학술적 정직성이다.
7. **깔끔한 ablation 분업**: 시각 인코더 유무(63.5 pp)와 loss 경로(34.4 pp)가 서로 다른 축을 담당함을 명확히 보인다.

## 10. 약점 및 한계 (Weaknesses & Limitations)

1. **핵심 메커니즘 주장이 자체 진단과 상충**: 논문 제목·서론·Figure 2가 "엣지의 현재 시각이 stale feature를 grounding한다"를 내세우지만, 부록 E는 이 체크포인트에서 엣지 보정 기여가 **0.03%**임을 보인다. 그렇다면 Table 2의 63.5 pp 시각 인코더 이득은 **추론 시 보정이 아니라 학습 중 정규화 효과**로 재해석되어야 하는데, 이 재해석이 본문에 충분히 반영되지 않았다.
2. **LIBERO 시뮬레이션 단일 벤치마크**: CALVIN, SimplerEnv, RoboCasa 등 다른 벤치마크가 없다. 지연 강건성이 LIBERO의 준정적(quasi-static) 태스크 특성에 의존할 가능성을 배제할 수 없다.
3. **실기 실험의 통계적 검정력 부재**: 셀당 10 trial, 단일 플랫폼, 단일 태스크(곰을 상자에). 저자들 스스로 "descriptive evidence"라고 인정한다. 게다가 "엣지"가 **RTX 5080 워크스테이션**이라 임베디드 효율성 증거가 전혀 아니다 — 논문의 동기인 "전력·중량 제약 로봇"과 괴리가 크다.
4. **학습 비용 2배**: paired-frame 학습은 7B 백본의 **forward pass를 두 번** 요구한다. 학습 시간·GPU-시간 수치가 보고되지 않아 실제 비용을 가늠할 수 없다.
5. **동기 성능의 소폭 손실**: d=0에서 OpenVLA-OFT(98.4/98.6/97.2/93.4)보다 낮다(97.9/97.8/96.5/91.7). Long에서 −1.7 pp. 지연이 거의 없는 배포에서는 순손실이다.
6. **연결 끊김(disconnection) 미대응**: 저자들이 명시하듯 장기 단절은 task-level 맥락조차 무효화하며, timestamped feature와 로컬 fallback 정책이 여전히 필요하다. 현 설계에는 둘 다 없다.
7. **동기 vs 지연 baseline 학습 조건의 비대칭**: 단일 경로 baseline들은 지연 프레임을 **추론 경로 수정 없이** 그대로 받는다. 이들을 지연 augmentation으로 재학습한 비교가 없어, 이득 중 얼마가 "구조" 덕이고 얼마가 "지연 노출 학습" 덕인지 baseline 측에서 분리되지 않는다(자체 ablation의 L_stale-only 89.2%는 후자의 비중이 크다고 시사한다).
8. **감각 modality 제한**: frozen RGB 인코더는 depth·force·contact 단서를 놓친다. 접촉 풍부(contact-rich) 조작으로의 일반화 증거가 없다.
9. **코드 미공개**: 재현이 어렵다. λ_max, n_warmup, 윈도우 크기 W 등 핵심 하이퍼파라미터의 구체적 값과 민감도 분석도 제시되지 않는다.
10. **delay는 환경 step 단위**: 벽시계 ms로의 환산은 제어 주파수와 네트워크 파이프라인에 의존하므로, 보고된 offset을 하드웨어 독립적 latency로 읽어서는 안 된다(부록 G에서 저자가 명시).

## 11. 의의 및 향후 방향 (Significance & Future Work)

이 논문의 개념적 기여는 **latency tolerance를 학습된 인터페이스 속성으로 재정의**한 데 있다. 지배적 접근은 지연을 스케줄링·보정·명시적 시간 조건화로 다뤄왔다 — 즉 **런타임에** 푸는 문제로 봤다. CloudEdgeVLA는 이를 **학습 시점의 표현 문제**로 옮긴다: 느린 경로가 나이에 둔감한 정보만 나르도록 목적함수로 압력을 가하면, 추론 시 지연 정보 없이도 인터페이스가 작동한다. 느린 경로로 **행동 계획이 아니라 재사용 가능한 task feature**를 보낸다는 선택은 이 재정의의 자연스러운 귀결이다.

이 설계는 오래된 정보를 최신으로 만들지 않는다 — **어떤 결정이 그 정보에 의존하는지를 제한**할 뿐이다. Action chunk 재생과 달리 엣지가 매 스텝 시각 피드백 루프를 닫고, world model로 현재를 재구성하는 접근과 달리 현재 이미지를 직접 관측하지만 미관측 상태에 대한 예측은 제공하지 않는다. 이 절충은 작고 감사 가능한(auditable) 엣지 경로를 유지하면서 **클라우드 백본을 독립적으로 스케일**할 수 있게 한다는 실용적 함의를 갖는다.

향후 방향: (a) 다중 벤치마크·다중 백본으로의 일반화, (b) 진짜 임베디드 엣지 하드웨어(Jetson급)에서의 전력·지연 프로파일링, (c) 부록 E의 negative result를 정면으로 다루는 후속 — 엣지 보정이 실제로 작동하는 체크포인트/학습 레짐이 존재하는가, (d) timestamped feature와 로컬 fallback을 통한 연결 단절 처리, (e) 실제 네트워크 트레이스(jitter·패킷 손실·burst)에서의 평가, (f) paired-frame 학습의 2× forward 비용을 줄이는 방법.

## 12. 총평 (Overall Assessment)

CloudEdgeVLA는 **결과가 매우 강하고 자기 검증이 이례적으로 정직한** 논문이다. d_max=40에서 macro-average 73.5% vs VLASH 1.7%, retention 76.5% vs 1.8%라는 격차는 해석의 여지가 거의 없는 수준이며, 학습 지연 윈도우 밖으로의 외삽과 3-시드 반복이 이를 뒷받침한다. Ablation은 "현재 관측을 헤드에 주는가"가 63.5 pp를 좌우하고 "지연 노출 학습"이 34.4 pp를 좌우한다는 분업을 깔끔하게 보인다.

동시에 이 논문은 **자기 서사를 스스로 반증하는 부록**을 싣는다. Counterfactual audit이 보여주는 edge-rescue 0.03%는, 이 체크포인트의 강건성이 "엣지가 stale feature를 실시간으로 고친다"가 아니라 "지연 학습이 백본을 안정화하고 헤드가 잔여 drift를 감쇠시킨다"에서 온다는 뜻이다. 이는 방법의 효과를 부정하지 않지만 **메커니즘 설명을 상당히 바꾸며**, 제목과 Figure 2가 강조하는 System 1/System 2 서사보다 훨씬 소박한 이야기다. 리뷰어라면 이 간극을 본문에 반영하라고 요구할 만하다.

한계는 분명하다 — LIBERO 단일 벤치마크, RTX 5080을 "엣지"라 부르는 10-trial 실기 파일럿, 2× 학습 비용, 연결 단절 미대응, 코드 미공개. 그럼에도 **latency tolerance를 스케줄링이 아니라 학습된 표현 속성으로 다루자**는 프레이밍과, 그 프레이밍을 κ 같은 측정 가능한 지표로 검증하려는 태도는 클라우드-엣지 VLA 배포 연구에 유용한 기여다. 새로운 백본이 아니라 **인터페이스 설계 + 학습 목적함수**의 기여로 읽는 것이 정확하다.

<!-- VERIFIED: pdf -->
