# T2VLA: Trust Your Instincts — Confidence-Driven Test-Time RL for Vision-Language-Action Models

> **한 줄 요약**: VLA가 스스로 생성한 rollout의 **generation confidence**(discrete: 길이 정규화 mean log-prob, flow: denoising Gaussian likelihood)가 실제 성공률과 강하게 상관한다는 관찰에서 출발해, 가장 자신 있는 궤적을 **Local Pseudo-Expert**로 선출하고 상위 K=5개를 **Global Expert Pool**로 유지한 뒤, 각 rollout의 **DTW 궤적 유사도**를 intrinsic reward로 삼아 GRPO로 정책을 test-time 업데이트하는 **외부 보상 완전 무의존 self-bootstrapping RL** 프레임워크. OpenVLA-OFT LIBERO 평균 91.0% → **97.2%**, RoboTwin 2.0 평균 37.8% → **59.1%**, π0는 +24.2%p.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **온라인 RL-VLA** (SimpleVLA-RL, πRL, VLA-RL, RIPT-VLA 등): 시뮬레이터가 주는 **binary success flag** 등 외부 보상에 의존 → 성공 판정이 미리 정의된 시나리오 밖에서는 작동 불가.
- **Offline RL-VLA** (Q-Transformer, ReinboT 등): 사전 annotated reward가 필요한 정적 데이터셋에 갇힘.
- **Test-time adaptation** (V-GPS, VLA-Pilot, VLAPS, TACO): 외부 verifier·MCTS·휴리스틱에 의존하는 training-free 재랭킹 — 정책 자체는 개선되지 않음.
- **TTRL** (LLM용 test-time RL)은 majority voting으로 자기 검증하지만, 로봇 조작은 **동일 목표에 도달하는 발산적 연속 궤적**이 많아 단일 정답 기반 self-verification이 불가능.

### 핵심 질문
- **VLA는 외부 검증 없이 자신의 실행 품질을 스스로 평가할 수 있는가?**
- 평가할 수 있다면, 그 신호(confidence)를 어떻게 **안정적인 trajectory-level reward**로 변환하는가?

📌 [Figure 1 삽입] — 외부 보상(시뮬레이터/사람/외부 LLM) 기반 기존 RL vs. 모델 내부 신호만으로 자율 bootstrapping하는 T2VLA 비교.

---

## 2. 방법론 심층 분석

### 2.1 경험적 근거: Confidence–Success 상관
2,000개 OpenVLA-OFT rollout 분석(Figure 3): **mean log-prob이 높을수록 성공률이 단조 증가**. 이 상관은 LIBERO-10처럼 초기 SR 17%인 약한 정책, OpenSora world-model 관측, StarVLA, GR00T(denoising confidence)까지 유지되며(Appendix C), 학습이 진행될수록 오히려 강해짐 (r=0.58 → 0.80 → 0.86).

### 2.2 Confidence 정의
- **Discrete VLA**: $c_i^{disc} = \frac{1}{T_i}\sum_t \log \pi_\theta(a_{i,t}|s_{i,t})$ — padding 제외 유효 horizon $T_i$로 길이 정규화(짧은 degenerate 궤적 선호 방지).
- **Flow VLA**: 선택된 denoising step k에서의 전이 Gaussian likelihood를 action-chunk horizon·차원에 걸쳐 합산 후 유효 rollout 길이 $L_i$로 정규화.

### 2.3 Confidence-Driven Dual Expert Bootstrapping
- **Local Pseudo-Expert**: 언어 지시 l로 그룹핑한 배치 내에서 $\tau^*_{local,l} = \arg\max c_i$ 선출 — 즉각적 on-policy 정렬 앵커.
- **Global Expert Pool** $\mathcal{P}_l$: 역대 local expert만 입장 허용, confidence 기준 **top-K(K=5)** priority buffer. 오래된 정책의 stale 궤적은 자동 축출.

### 2.4 DTW 기반 Hybrid Similarity Reward
- 유클리드 거리는 시간축 위상 변화에 취약(유사도 0.6914) → **DTW**가 비선형 시간 왜곡으로 공간 형상 유사도를 복원(0.9460). 액션 차원별 min-max 정규화 후 누적 거리를 max(M,T)로 나눠 (0,1] 유사도로 변환.
- **동적 융합**: $r_i^{sim} = w \cdot Sim(\tau_i, \tau^*_{local}) + (1-w) \cdot \max_{\tau_p \in \mathcal{P}_l} Sim(\tau_i, \tau_p)$, 여기서 $w = clip\left(\frac{c^*_{local}-c_{min}}{c_{max}-c_{min}+\epsilon}, 0, 1\right)$ — 자신 있는 배치는 local 우선(w→1), 불확실하면 historical로 폴백(w→0).
- **KL 페널티** (β=0.02, SFT 모델 π_ref 기준)로 over-optimization 억제.

### 2.5 GRPO 최적화
그룹 정규화 advantage $A_i = (r_i - \mu)/(\sigma+\epsilon)$, clipped surrogate로 업데이트. Critic 불필요.

> ❓ **예상 질문**: Confidence를 직접 reward로 쓰면 안 되는가?
> **답변**: Appendix G에서 실험 — raw log-prob reward는 epoch 60 부근 ~90%까지 오르다가 이후 84–85%로 **붕괴**. 성공한 궤적 간의 confidence 편차 때문에 유효한 실행이 벌점을 받는 misclassification이 누적됨. 그래서 confidence는 **expert 선출(랭킹)에만** 쓰고, reward는 물리적 궤적 유사도(DTW)로 부여.

---

## 3. 데이터 전략

- **추가 시연 데이터 없음** — 전부 정책 자신의 exploratory rollout. Instruction당 N=8 trajectory 샘플링(온도 1.6/1.0).
- **SFT 초기화**: OpenVLA-OFT는 SimpleVLA-RL의 수정 아키텍처+SFT 가중치, π0/π0.5는 πRL의 SFT 가중치·MDP 재사용. 즉 baseline과 완전 동일 출발점에서 self-reward만으로 개선량을 측정.
- **극한 세팅**: 1-shot SFT(traj1) 초기화, OpenSora world-model이 합성한 관측으로만 상호작용하는 세팅까지 검증.
- **RoboTwin 2.0 domain randomization 대응**: N/G개 시드를 G회 복제하는 **group-level seed synchronization** — 그룹 내 동일 초기 상태(공정한 advantage), 그룹 간 다양성(랜덤화 이점) 동시 확보.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Advantage estimator | GRPO (critic-free) |
| Group size | 8 rollouts/task |
| Train batch size | 16 (validation 128) |
| Actor LR | 5e-6 (OpenVLA-OFT: cosine, π0/π0.5: constant) |
| Clip bounds | (0.2, 0.28) OpenVLA-OFT / 0.2 π0 계열 |
| KL 계수 | 초기 0.02 (target 0.04), entropy 0.005 |
| Rollout 온도 | 1.6 (OpenVLA-OFT), 1.0 (π0/π0.5) |
| Max env steps | 500 (OpenVLA-OFT), 480 (π0/π0.5, max 1024 tokens) |
| Expert pool K | 5 |
| 총 epochs | 300 |
| π0.5 denoising steps | 4 |

---

## 5. 실험 설계 및 평가 프로토콜

- **LIBERO** 4개 suite (Spatial/Object/Goal/Long): 성공률(%).
- **RoboTwin 2.0** bimanual 5개 task를 horizon별 분류: Short(Lift Pot, Beat Hammer, 112–130 steps), Medium(Place Empty Cup, 151–223), Long/Extra-Long(Handover Block, Stack Bowls, 283–637).
- **RoboCasa** Close Drawer (π0, 가정환경 전이 검증).
- 비교군: SFT(Octo, OpenVLA, UniVLA), 외부 보상 RL(VLA-RL, SimpleVLA-RL=oracle), test-time training(EVOLVE-VLA, learned critic 사용).
- 환경 성공률은 **모니터링에만** 기록, gradient에는 절대 미사용.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1)

| Method | Reward | Spatial | Object | Goal | Long | **Avg** |
|--------|--------|---:|---:|---:|---:|---:|
| UniVLA | None (SFT) | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| VLA-RL | Env. Success | 90.2 | 94.3 | 91.8 | 82.2 | 89.6 |
| SimpleVLA-RL (oracle) | Env. Success | 99.4 | 99.1 | 99.2 | 98.5 | 99.1 |
| EVOLVE-VLA | Learned Critic | 95.4 | 97.4 | 95.8 | 94.4 | 95.8 |
| OpenVLA-OFT (SFT) | None | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| **Ours (OpenVLA-OFT)** | **Self-Reward** | **97.7** | **99.6** | **96.1** | **95.3** | **97.2** |
| π0 (Base) | None | 65.3 | 64.4 | 49.8 | 51.2 | 57.7 |
| **Ours (π0)** | **Self-Reward** | **86.3** | **91.0** | **82.0** | **68.0** | **81.9** |
| π0.5 (Base) | None | 84.6 | 95.4 | 84.6 | 43.9 | 77.1 |
| **Ours (π0.5)** | **Self-Reward** | **94.9** | **98.4** | **91.8** | **55.1** | **85.1** |

- OpenVLA-OFT +6.2%p로 **SOTA SFT(UniVLA 95.2) 및 learned-critic TTT(EVOLVE-VLA 95.8)를 추월**, oracle RL(99.1)과의 갭을 1.9%p로 축소 — 외부 보상 0으로.
- π0에서 +24.2%p(Goal은 +32.2%p): few-shot 초기화 정책일수록 개선 여지가 큼.

### RoboTwin 2.0 (Table 2, OpenVLA-OFT)

| Task | SFT | Ours | Δ |
|------|---:|---:|---:|
| Lift Pot | 10.1 | 39.8 | +29.7 |
| Beat Hammer | 28.1 | 68.0 | +39.9 |
| Place Empty Cup | 77.3 | 84.8 | +7.5 |
| Handover Block | 33.1 | 44.1 | +11.0 |
| Stack Bowls | 40.6 | 59.0 | +18.4 |
| **Average** | **37.8** | **59.1** | **+21.3** |

- Bimanual 고차원 제어에서도 +21.3%p — DTW self-bootstrapping이 액션 차원 증가에 스케일함을 시사.

### 추가 아키텍처/환경 (Appendix Table 6)
- **GR00T**: LIBERO-Spatial 41.4→60.9 (+19.5), LIBERO-Object 58.6→**98.8** (+40.2). 선출된 expert의 성공률 96% (전체 rollout 64%).
- **π0 + RoboCasa** Close Drawer: 75.0→87.5.
- **OpenSora world-model 상호작용**: LIBERO-Spatial 61.2→63.3 — 시뮬레이터 없이 학습된 world model 위에서도 작동.

---

## 7. Ablation 분석

### Dual Expert (Table 3, LIBERO-Long)
| 구성 | SR (%) |
|------|---:|
| Local Expert Only | 94.5 |
| Global Expert Only | 93.0 |
| **Dual Expert** | **95.3** |
- Local은 최신 발견 반영이 빠르지만 배치 분산에 취약, Global은 안정적이나 적응이 느림 — 상보적.

### Expert Pool 용량 K (Figure 4)
- K=3: 93.4% (다양성 부족), **K=5: 95.3% (최적)**, K=10: 91.4% (stale 궤적이 앵커 오염). 모든 K가 base 86.5%는 상회.

### Expert Fusion 전략 (Table 4)
| 전략 | SR (%) |
|------|---:|
| Static w=0.5 | 88.3 († 조기 붕괴) |
| Max Routing | 87.5 († 조기 붕괴) |
| Adaptive Margin Fallback | 90.0 |
| Asymmetric Z-Score Gate | 91.4 |
| Sigmoid Gate | 92.6 |
| **Min-Max Scaling (Ours)** | **95.3** |
- 경직된 가중치·하드 스위칭은 100 epoch 내 policy collapse. 유계·연속 스케일링이 안정성의 핵심.

### DTW vs Euclidean (Figure 5)
- 동일 궤적 쌍에서 Euclidean 0.6914 vs DTW 0.9460 — 시간 위상 차이를 유사도 붕괴 없이 흡수.

---

## 8. 관련 연구 비교

| 방법 | 보상 출처 | 외부 의존 | 정책 업데이트 | LIBERO Avg |
|------|----------|----------|--------------|---:|
| SimpleVLA-RL | 시뮬레이터 success flag | 환경 oracle | yes | 99.1 |
| VLA-RL | 환경 성공 | 환경 oracle | yes | 89.6 |
| EVOLVE-VLA | 학습된 progress critic | 별도 foundation critic | yes | 95.8 |
| V-GPS / VLA-Pilot | 외부 verifier 재랭킹 | 외부 평가기 | no (training-free) | — |
| TTRL (LLM) | Majority voting | 없음 (단일 정답 전제) | yes | — (수학/코딩) |
| **T2VLA** | **자기 confidence + DTW 유사도** | **없음** | **yes (GRPO)** | **97.2** |

### 핵심 차이
- 외부 보상·verifier·critic이 전혀 없는 **최초급의 VLA test-time self-bootstrapping RL**.
- TTRL의 "정답 합의" 아이디어를 연속 궤적 도메인으로 이식하되, voting 대신 **confidence 선출 + DTW 정렬**로 검증 불가능성 문제를 우회.
- Discrete(OpenVLA-OFT, StarVLA)와 continuous(π0, π0.5, GR00T) 양 패러다임에 confidence 정의만 바꿔 적용되는 architecture-agnostic 설계.

---

## 9. 한계 및 미해결 문제

### 저자 명시 한계 (Appendix L)
1. **저확신 expert 누락**: flow VLA의 Gaussian likelihood 기반 선출은 성공했지만 likelihood가 낮은 궤적을 버릴 수 있음.
2. **초기화 임계값**: 1-shot LIBERO-Long에서 17.3→**11.0으로 악화** — 약한 정책은 짧은 조기 종료 궤적에 높은 confidence를 부여(Figure 13: 공간적으로 truncated된 궤적이 reward 1.000으로 expert 선출)해 early-termination을 강화하는 악순환.
3. **실기 미검증**: 전부 시뮬레이션 + world-model 상호작용. 물리 로봇 배포는 future work.

### 추가 비판점
- Test-time "RL"이지만 **GRPO로 전체 가중치를 300 epoch 업데이트** — 사실상 환경별 self-supervised fine-tuning에 가까워 진짜 on-the-fly 적응과는 거리가 있고, 연산 비용이 명시되지 않음.
- DTW 유사도는 "expert와 비슷함"을 보상하므로 **expert보다 나은 novel 전략의 탐색을 억제**할 수 있음 — expert pool 자체가 개선되는 속도에 상한이 걸림.
- LIBERO/RoboTwin은 성공 판정이 존재하는 벤치마크라, 외부 보상이 정말 불가능한 실환경에서의 이점은 아직 간접 증거.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Confidence–success 상관의 체계적 검증과, 이를 raw reward가 아닌 expert 선출+DTW 유사도로 우회한 reward 설계가 독창적. |
| **Technical depth** | ★★★★☆ — Fusion 전략 6종 비교, K sweep, raw-confidence 붕괴 분석, bootstrapping threshold 시각화까지 분석이 치밀. |
| **Experimental rigor** | ★★★★☆ — 5개 아키텍처 × 3개 벤치마크 + world-model 세팅. 다만 sim-only, 연산 비용 미보고. |
| **Practical impact** | ★★★★☆ — 성공 판정기가 없는 새 환경에서 VLA를 자율 개선시키는 경로 제시. oracle RL과 1.9%p 차이. |
| **Writing quality** | ★★★★☆ — 관찰→가설→설계→실패 모드까지 논리 전개가 명확. |

**강점**: 외부 보상 의존이라는 RL-VLA의 근본 병목을 정면 돌파. 실패 조건(1-shot Long 붕괴)까지 정직하게 보고하고 메커니즘(조기 종료 궤적의 과신)을 시각적으로 해부한 점이 신뢰를 높임.
**약점**: "test-time"이라 부르기엔 무거운 300-epoch GRPO 루프, 시뮬레이션 한정 검증, expert 모방 구조의 탐색 상한.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Confidence가 성공과 상관한다면 왜 raw confidence reward는 붕괴하는가? | 성공 궤적들 사이의 confidence 편차가 유효 실행을 벌점 처리(Appendix G: 90% 피크 후 84%로 하락). 랭킹 신호로는 유효하나 절대 스칼라 보상으로는 부적합. |
| 2 | Expert 모방이면 expert를 넘어설 수 없지 않은가? | 매 iteration 새 rollout에서 local expert가 재선출되고 pool이 갱신되므로 앵커 자체가 이동. LIBERO-Goal traj1에서 59.6→83.0의 지속 상승(Figure 10)이 실증. 단, 상승 속도는 pool 갱신 속도에 종속. |
| 3 | 1-shot Long 붕괴는 방법의 치명적 결함 아닌가? | 저자가 "prior threshold"로 명시 — 초기 정책이 task horizon을 커버할 최소 역량이 필요. Figure 13에서 truncated 궤적 과신 메커니즘까지 규명. 적용 전 초기 SR 점검이 실무 가이드. |
| 4 | GRPO 300 epoch이 "test-time"인가? | 환경 라벨 없이 배포 환경 상호작용만으로 학습한다는 의미의 test-time RL(TTRL 계보). Latency 관점의 test-time adaptation과는 구별 필요. |
| 5 | DTW가 bimanual 고차원에서도 유효한 이유는? | 차원별 min-max 정규화 + max(M,T) 길이 정규화로 차원·길이 불변성 확보. RoboTwin +21.3%p가 증거. |
| 6 | 왜 K=10이 K=5보다 나쁜가? | 초기 미숙 정책의 stale expert가 잔존해 suboptimal 앵커로 reward를 오염 (91.4% vs 95.3%). |
| 7 | Flow VLA의 denoising likelihood가 confidence로 타당한가? | GR00T 2,000 rollout에서 선출 expert 성공률 96% vs 전체 64% (Figure 7). 단 calibration이 아닌 ordinal ranking만 요구. |
| 8 | 학습 중 confidence가 과신으로 왜곡되지 않는가? | Step 0/100/200 체크포인트에서 상관이 r=0.58→0.86으로 오히려 강화 (Figure 8). 평가 horizon 내 과신 미관찰. |
| 9 | EVOLVE-VLA와의 본질적 차이는? | EVOLVE-VLA는 별도 학습된 progress critic(외부 모델) 필요, T2VLA는 정책 자신의 log-prob만 사용. 그러면서도 97.2 vs 95.8로 우위. |
| 10 | 실로봇에서는 rollout 수집 자체가 병목 아닌가? | 맞음 — sim 전용 검증이 한계. 다만 OpenSora world-model 실험(61.2→63.3)이 실환경 상호작용을 합성 관측으로 대체할 가능성을 예비 시사. |

---

## 12. 레퍼런스 및 리소스

- **논문**: [arXiv:2606.29892](https://arxiv.org/abs/2606.29892) — Trust Your Instincts: Confidence-Driven Test-Time RL for Vision-Language-Action Models (Chen, Yuan, Wang, Chen; Fudan University · Shanghai Innovation Institute · MoShen Intelligence)
- **코드**: 미공개 (2026-07 기준)
- **기반 모델/가중치**: OpenVLA-OFT (SimpleVLA-RL SFT weights), π0/π0.5 (πRL SFT weights), StarVLA, GR00T
- **핵심 선행 연구**: TTRL (arXiv:2504.16084), SimpleVLA-RL (arXiv:2509.09674), πRL (arXiv:2510.25889), EVOLVE-VLA (arXiv:2512.14666), GRPO/DeepSeekMath (arXiv:2402.03300), DTW (Rakthanmanon et al., KDD 2012)
- **벤치마크**: LIBERO, RoboTwin 2.0 (arXiv:2506.18088), RoboCasa

<!-- VERIFIED: pdf -->
