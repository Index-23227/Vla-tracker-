# AtlasVLA: Persistent World-Ego State Modeling for Vision-Language-Action Models

> **한 줄 요약**: 손목 카메라 **단 하나**만으로 동작하는 VLA를 목표로, 2D 관측을 depth로 3D에 back-projection해 voxel-hashing으로 누적하는 **4D Persistent World State Memory**와 task 진행도를 추적하는 **Ego-Working State Memory**를 결합하고, 이 둘을 step-wise로 조건화한 DiT action expert로 액션을 생성. LIBERO 5-suite 평균 **97.6%**, RLBench 6-task 평균 **70.8%**, 실세계 long-horizon **69.5%**를 wrist-only 설정에서 달성하며 multi-view 베이스라인을 역전. (arXiv:2608.06729, 2026-08-07)

- 저자: Guiyu Zhao, Longteng Guo(†), Yanghong Mei, Zilin Zhu, Yu Zhang, Bin Cao, MingMing Yu, Xingjian He, Jie Jiang, Jing Liu
- 소속: Institute of Automation, CAS (CASIA) / University of Chinese Academy of Sciences / Beijing Freedo Technology / Beihang University
- 코드: 논문 내 공개 저장소 URL 명시 없음 (open_source = false로 기록)

---

## 1. 배경 및 동기

현재 VLA는 근본적으로 **reactive paradigm**이다. 즉 순간 관측 → 액션의 반사적 매핑이며, 이는 부분 관측(partially observable) 환경과 long-horizon task에서 두 가지 병목을 만든다고 저자들은 정식화한다(§1, Fig. 1).

1. **Perception forgetting (공간적 망각)**: 손목 장착 카메라는 end-effector와 함께 움직이므로 FoV가 계속 바뀐다. "순간 FoV ≠ 실제 world state"이며, task 관련 객체·구조는 시야를 벗어나는 즉시 잊힌다. 결과적으로 "박스가 어디 있지?"를 답하지 못해 실행이 붕괴한다.
2. **Task-progress forgetting (시간적 망각)**: 다단계 task(예: 5개 sub-step의 change cubes)에서 표준 VLA는 무엇을 이미 했는지에 대한 historical context가 없어 sub-step을 반복하거나 누락한다.

기존 우회책은 3인칭/멀티뷰 카메라를 다는 것인데, 이는 heavily instrumented 셋업을 전제로 하므로 확장성이 떨어진다. 저자들의 문제 제기는 명확하다 — 인간은 omniscient 카메라 없이도 내부 world model(인지 지도)을 유지한다. 따라서 VLA도 `local observation → latent state update → persistent world state → future action`의 연속 사이클로 전환해야 한다는 것.

## 2. 방법론 심층 분석

### 2.1 문제 정식화 (§3.1)
시점 t에서 에이전트는 **손목 관측 O_t^w, proprioceptive state S_t, 언어 지시 L만** 받는다(3인칭 뷰 완전 배제). 정책 π_θ는 action chunk A_t = [a_t, …, a_{t+k-1}]를 생성하고, 각 a_t ∈ ℝ⁷는 6-DoF end-effector pose + binary gripper이다.

### 2.2 Persistent World State Memory (§3.2)
**(a) Instantaneous World State Construction**
- Frozen vision encoder로 2D visual token X_t^w 추출.
- Depth Anything v3(Lin et al. 2025)의 **streaming** 모델로 depth D_t^w 추정 (과거 프레임을 활용해 temporal depth consistency 강화).
- Extrinsic은 hand-eye calibration으로 유도: `T_ex = ψ(S_t) · T_h2e` (ψ는 EE state → pose matrix 변환). 즉 별도 pose tracking 없이 로봇 상태에서 카메라 외부 파라미터를 얻는 것이 핵심 트릭.
- Back-Projection(X_t^w, D_t^w, T_in, T_ex) → `m_t, P_t` (3D 위치를 가진 2D latent token).

**(b) Spatio-Temporal Embedding (Eq. 3)**
`m̂_t = m_t + E_spatial(P_t) + E_temporal(t)`. 둘 다 MLP로 파라미터화. 3D 좌표만으로는 spatial aliasing, 시간 정보 없이는 temporal degradation이 발생한다는 논거.

**(c) World State Update (Eq. 4-5)** — 이 부분이 방법론의 심장이다.
TSDF map integration(Newcombe et al. 2011)에 착안한 **voxel-hashing** 전략. 동일 물리 영역에 대응하는 토큰들을 voxel 단위로 confidence-weighted 가중 평균:

```
M_t(v) = [W_{t-1}(v)·M_{t-1}(v) + w_t·m_t(v)] / [W_{t-1}(v) + w_t]
w_t(v) = c_t(v),   W_t(v) = λ·W_{t-1}(v) + w_t(v)
```

여기서 confidence weight w_t는 **depth estimation confidence에서 직접 유도**된다. 즉 신뢰도 높은 depth 관측이 global memory에 더 크게 기여하고 불확실한 측정은 억제된다. 시간축으로는 최대 window W의 sliding window로 망각하되, **첫 프레임은 "permanent initialization" 규칙으로 영구 고정**한다(첫 프레임이 대체로 최적 FoV이고 초기 상태를 정확히 반영한다는 가정).

### 2.3 Ego-Working State Memory (§3.3)
- **Intent-aware Query (Eq. 6)**: 학습 가능한 intent query Q_ego ∈ ℝ^{N×d}를 VLM에 투입해 cross-attention으로 goal-oriented 정보를 응축 → Z_ego (ego-working latent token).
- **Ego-Working Memory Bank (Eq. 7)**: `M_t^ego = Cons(M_{t-1}^ego ∪ {Z_t^ego + E_temporal(t)})`. Cons(·)는 **시간적으로 인접하고 의미적으로 유사한 intent token을 병합**하는 redundancy-aware consolidation으로, 무한 성장과 semantic redundancy를 동시에 억제한다.

### 2.4 World-Ego-Guided Action Generation (§3.4)
- **Ego-Working Retrieval (Eq. 8)**: `C_t^ego = CrossAttn(Z_t^ego, M_t^ego, M_t^ego)`.
- **Ego-Guided World Retrieval (Eq. 9)**: `C_t^world = AddNorm(FFN(IntentAttn(C_t^ego, M_t, M_t)))`. 주목할 설계는 **ego context가 query가 되어 world memory를 조회**한다는 점 — 즉 "지금 내 의도에 관련된 공간만" 뽑아 spatial redundancy를 억제한다.
- **Step-wise conditioned DiT**: 표준 global conditioning을 버리고, 매 diffusion step k에서 noisy action token이 (1) ego-working attention → (2) world state attention → (3) FFN 순으로 순차 attend한다. 각 모듈은 Add&Norm이 붙은 표준 Transformer 블록이며 ×L 반복.

## 3. 데이터 전략

- **LIBERO**: task당 50개 expert demo. Spatial/Object/Goal은 각각 개별 모델 20k step, Long + LIBERO-90은 **단일 통합 모델**로 40k step 공동 학습(MemoryVLA 프로토콜 준수).
- **RLBench**: task당 100개 demo, 80k step 학습. 관측은 **128×128 단일 wrist RGB**만 사용하고 front/left/right 3인칭 뷰 전부 폐기.
- **실세계**: 7-DoF Franka + end-effector에 장착한 Intel RealSense D415 **단 하나**. 3인칭 카메라는 물리적으로 disconnect. 캐비닛·서랍·일상 물체가 놓인 unstructured tabletop. General 6 tasks + Long-horizon 4 tasks = 총 10개 task, 각 50 trial. ROS로 동기화.
- 대규모 cross-embodiment 사전학습(OXE 등)을 **직접 수행하지는 않고**, OpenVLA-7b 체크포인트를 pre-training model로 채택하는 방식으로 사전학습 비용을 우회.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Vision encoder | DINOv2 + SigLIP (frozen), 별도 spatial stream은 fine-tuned Depth Anything v3 |
| LLM | LLaMA-2 7B decoder-only, OpenVLA-7b로 초기화 |
| Action expert | DiT (CogACT 계열 영감), **~300M** 파라미터 |
| 입력 해상도 | LIBERO 224×224, RLBench 128×128 (wrist RGB 단일) |
| 학습 하드웨어 | 8× NVIDIA A100, PyTorch FSDP |
| Global batch / LR | 256 (32×8) / 2×10⁻⁵ |
| Action chunk size | 16 |
| Denoising | DDIM 10 steps, CFG scale 1.5, condition dropout p=0.1 |
| World memory 용량 | 2048 tokens, voxel size 0.025 m |
| Ego-working token 수 | 4 |
| Retrieval layer 수 | ego-working 2 layers / ego-guided world 4 layers |
| Loss | MSE noise-prediction, cosine noise schedule (Eq. 10-11) |

**LLM의 역할 재정의가 중요하다**: LLaMA-2 7B는 저수준 액션을 직접 디코딩하지 않고 **high-level intent tracker + semantic router**로만 작동하여 compact ego-working latent state를 출력한다. 실제 continuous control은 전적으로 300M DiT가 담당한다.

**Runtime (Table 7)**: latency 0.158s (MemoryVLA 0.146s 대비 +0.012s), throughput 101.3 Hz (vs 109.5 Hz), GPU VRAM 18.1 GB (vs 16.7 GB, +1.4 GB). 실세계 성공률은 78.7% vs 62.3%.

## 5. 실험 설계 및 평가 프로토콜

- **LIBERO**: 5개 suite(Spatial, Object, Goal, Long, 90), task당 50 rollout. **best validation step이 아닌 final training checkpoint**에서만 수치를 뽑아 validation bias를 제거했다고 명시 — 재현성 관점에서 좋은 선택이다.
- **RLBench**: Lou et al. 2026 프로토콜을 따라 6개 대표 task, task당 20 trial, 역시 final checkpoint.
- **실세계**: task당 50 독립 trial, 초기 객체 배치·공간 레이아웃 랜덤화. **모든 sub-goal을 순차적으로 완료하고 지정된 최종 상태에 도달한 경우에만** 성공으로 기록(부분 성공 불인정).
- 비교 대상 베이스라인은 카메라 구성을 명시적으로 3rd / 3rd+wrist / wrist의 세 조건으로 나누어 표기했다. 이 표기 자체가 이 논문의 주장을 검증 가능하게 만드는 장치다.

## 6. 실험 결과 심층 분석

### LIBERO (Table 1, 성공률 %)

| Method | Cameras | Spatial | Object | Goal | Long | 90 | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA | 3rd | 84.7 | 88.4 | 79.2 | 53.7 | 73.5 | 75.9 |
| π₀ | 3rd | 90.8 | 91.8 | 89.6 | 80.2 | – | 88.1 |
| 4D-VLA | 3rd | 93.8 | 92.8 | 95.6 | 86.5 | – | 92.2 |
| CogACT | 3rd | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| MemoryVLA | 3rd | 98.4 | 98.4 | 96.4 | 93.4 | 95.6 | 96.5 |
| π₀ | 3rd + wrist | 96.8 | 98.8 | 95.8 | 85.2 | – | 94.2 |
| OpenVLA-OFT | 3rd + wrist | 97.6 | 98.4 | 97.9 | 94.5 | – | 97.1 |
| GE-ACT | 3rd + wrist | 98.2 | 97.6 | 95.8 | 94.4 | – | 96.5 |
| CogACT | wrist | 96.4 | 95.8 | 88.6 | 86.2 | 87.4 | 90.9 |
| π₀ | wrist | 94.4 | 96.6 | 90.8 | 80.8 | – | 90.7 |
| MemoryVLA | wrist | 96.2 | 99.2 | 96.4 | 87.6 | 90.7 | 94.0 |
| **AtlasVLA (Ours)** | **wrist** | **99.4** | **99.8** | **98.2** | **94.6** | **95.8** | **97.6** |

핵심 관찰:
- 베이스라인은 3rd → wrist 전환 시 π₀ −3.5%p, MemoryVLA −2.5%p로 저하되는데, AtlasVLA는 wrist-only에서 97.6%로 **3rd+wrist를 쓰는 π₀(94.2)보다 3.4%p 높다**.
- LIBERO-Long **94.6%**는 wrist-only 제약 하 SOTA이며 wrist MemoryVLA(87.6) 대비 **+7.0%p**. 초록에서 강조하는 "+9.4%p on LIBERO-Long"은 3rd+wrist π₀(85.2) 기준 비교다.
- 다만 정직하게 보면 LIBERO-Long 94.6은 3rd+wrist OpenVLA-OFT(94.5)·GE-ACT(94.4)와 사실상 동률이다. AtlasVLA의 우위는 "동등 성능을 **카메라 한 대로** 낸다"는 데 있지, 절대 수치의 압도가 아니다.

### RLBench (Table 2, wrist-only 128×128, task당 20 trial)

| Method | Cameras | Sweep to Dustpan | Phone on Base | Umbrella Out | Frame off Hanger | Wine at Rack | Water Plants | Avg |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 3rd | 50.0 | 20.0 | 35.0 | 15.0 | 10.0 | 10.0 | 23.3 |
| CogACT | 3rd | 50.0 | 50.0 | 55.0 | 45.0 | 30.0 | 25.0 | 42.5 |
| FiS-VLA | 3rd | 55.0 | 50.0 | 50.0 | **70.0** | 55.0 | 20.0 | 50.0 |
| MemoryVLA | 3rd | 50.0 | 60.0 | 75.0 | 60.0 | **80.0** | 55.0 | 63.3 |
| π₀ | 3rd + wrist | 30.0 | 30.0 | 30.0 | **70.0** | 10.0 | 30.0 | 33.3 |
| GE-ACT | 3rd + wrist | 10.0 | 15.0 | 40.0 | 35.0 | 40.0 | 45.0 | 30.8 |
| CogACT | wrist | 40.0 | 35.0 | 50.0 | 35.0 | 20.0 | 25.0 | 34.2 |
| MemoryVLA | wrist | 40.0 | 55.0 | 65.0 | 60.0 | 60.0 | 50.0 | 55.0 |
| **AtlasVLA (Ours)** | **wrist** | **70.0** | **70.0** | **80.0** | 65.0 | 75.0 | **65.0** | **70.8** |

- 평균 70.8%로 wrist MemoryVLA 대비 **+15.8%p**, 3rd MemoryVLA 대비 **+7.5%p**. LIBERO보다 격차가 훨씬 크다 — 기하학적 복잡도가 높고 궤적 변동이 큰 환경에서 persistent world state의 효용이 실제로 더 크다는 방증.
- 단, task당 20 trial뿐이라 5%p = 1회 시행이다. 개별 task 수치는 통계적으로 매우 불안정하며, Frame off Hanger(65.0)와 Wine at Rack(75.0)에서는 베이스라인에 뒤진다. 저자가 본문에서 "consistent improvements across all tasks"라고 쓴 것은 **표와 불일치하는 과장**이다.

### 실세계 (Table 3-4, task당 50 trial)

General 6 tasks: AtlasVLA **78.7%** (Pepper on Plate 78.0 / Pepper in Box 72.0 / Stack Cubes 76.0 / Carrot on Plate 84.0 / Cube in Drawer 82.0 / Can in Drawer 80.0). 3rd+wrist π₀ 66.7, 3rd MemoryVLA 70.7, wrist MemoryVLA 62.3 → 각각 **+12.0 / +8.0 / +16.4%p**.

Long-horizon 4 tasks: AtlasVLA **69.5%** (Change Cubes 74 / Stack Cubes Order 66 / Clean Desk 68 / Pick Place Order 70). π₀ 52.0, MemoryVLA 60.5 → **+17.5 / +9.0%p**. 초록의 "17.5% in real-world long-horizon"이 바로 이 π₀ 대비 수치다.

## 7. Ablation 분석

### 핵심 모듈 (Table 5, LIBERO / 실세계 Long)

| No. | 구성 | LIBERO | Real-world Long |
|---|---|---|---|
| 1 | w/o World State Memory | 93.5 | 54.0 |
| 2 | w/o Ego-Working Memory | 95.0 | 56.5 |
| 3 | **AtlasVLA (full)** | **97.6** | **69.5** |
| 4 | w/o World State Update (naive accumulation) | 94.6 | 58.0 |
| 6 | w/o Spatial PE | 96.4 | 67.5 |
| 7 | w/o Temporal PE | 96.8 | 65.0 |
| 9 | w/o World State Conditioning | 95.2 | 61.5 |

- World State Memory 제거 시 실세계가 69.5 → **54.0**으로 붕괴(−15.5%p). Ego-Working Memory 제거는 −13.0%p. 두 메모리가 대등하게 중요하며, 어느 하나만으로는 성립하지 않는 설계다.
- 주목할 점: **LIBERO에서의 낙폭(−4.1, −2.6%p)이 실세계(−15.5, −13.0%p)보다 훨씬 작다.** 시뮬레이션은 이 논문이 겨냥한 부분 관측 문제를 충분히 스트레스하지 못한다는 뜻이며, 역으로 실세계 결과가 주장의 실질적 근거다.
- Naive accumulation(row 4)이 실세계 −11.5%p를 유발 → TSDF 스타일 voxel aggregation + sliding window가 단순 누적 대비 실질 기여.
- Spatial PE(−2.0%p)보다 **Temporal PE 제거(−4.5%p)의 타격이 더 크다.** 흥미로운 비대칭 — 저자는 언급하지 않지만, world memory에 이미 3D 좌표가 내재된 반면 시간 순서는 오직 PE로만 주어지기 때문으로 보인다.

### 하이퍼파라미터 (Table 8-9)

| Memory length | LIBERO | Real Long | | Voxel size | LIBERO | Real Long |
|---|---|---|---|---|---|---|
| 8 | 97.3 | 66.4 | | 0.01 | 96.3 | 65.7 |
| **16 (Ours)** | **97.6** | 69.5 | | **0.025 (Ours)** | **97.6** | **69.5** |
| 32 | 97.2 | **69.8** | | 0.05 | 97.2 | 64.0 |
| | | | | 0.1 | 95.9 | 58.5 |

- Memory length 32가 실세계에서 69.8로 16(69.5)보다 **높다.** 저자는 "marginal benefits"라며 16을 채택했지만, 이는 사후적 선택의 여지가 있다. 0.3%p는 50 trial×4 task = 200 trial 기준 노이즈 수준이라 어느 쪽도 유의하다 보기 어렵다.
- Voxel size는 명확한 U자형. 0.01m는 과도한 sparsity로 노이즈, 0.1m는 기하 구조 과압축으로 58.5까지 붕괴. 0.025m가 뚜렷한 최적점이며 이 민감도는 실배포 시 튜닝 부담을 의미한다.

## 8. 관련 연구 비교

| 계열 | 대표작 | AtlasVLA와의 차이 |
|---|---|---|
| Autoregressive VLA | OpenVLA, RT-2 | 연속 제어를 토큰 생성으로 정식화. 메모리 없음, 3인칭 의존 |
| Diffusion/Flow VLA | π₀, CogACT, DexVLA, DreamVLA | 강력한 action head지만 여전히 **instantaneous 관측**에 조건화 |
| 명시적 메모리 | MAP-VLA, MemoryVLA, ReMem-VLA, MEM | retrieval-based visual bank / recurrent latent query. 저자 표현으로 "**temporal caching에 치우쳐 명시적 spatial modeling이 없다**" |
| 공간 표현 | SOMA (Li et al. 2026b) | out-of-vision을 다루지만 **manipulation 이전의 정적 스냅샷**에 의존. 실행 중 장면 갱신이 없어 연속 wrist-only에 부적합 |

AtlasVLA의 포지셔닝은 "temporal memory(MemoryVLA 계열) × spatial memory(SOMA 계열)"의 교집합을 최초로 **실행 중 지속 갱신되는 4D 상태**로 통합했다는 것이다. 이 주장 자체는 관련 연구 서술과 실험 구성 모두에서 일관되게 방어된다.

## 9. 한계 및 미해결 문제

1. **정확한 hand-eye calibration + proprioception에 강하게 의존**. 전체 world memory는 `T_ex = ψ(S_t)·T_h2e`로 계산한 외부 파라미터 위에 세워진다. 캘리브레이션 드리프트나 joint encoder 오차가 있으면 voxel fusion이 잘못된 위치에 누적되며, 논문에는 **calibration 노이즈에 대한 robustness 실험이 전혀 없다**.
2. **Monocular depth 추정에 대한 의존**. Depth Anything v3의 실패 모드(투명·반사·무텍스처 표면)가 그대로 world state 오염으로 전이된다. Confidence weighting이 완충하도록 설계됐지만 이를 검증하는 실험이 없다.
3. **"Permanent initialization" 규칙의 취약성**. 첫 프레임이 최적 FoV라는 가정은 초기 배치가 나쁘거나 장면이 크게 변하는 경우(사람이 물체를 옮김) 오히려 **stale한 world state를 영구 고정**하게 된다. 동적 환경 실험이 없다.
4. **평가 규모가 작다**. RLBench는 6개 task × 20 trial, 실세계 long-horizon은 4개 task. 5%p 단위 차이는 시행 1회에 해당하므로 개별 task 비교는 신뢰하기 어렵다.
5. **본문 서술의 과장**. "consistent improvements across all tasks"(§4.3)는 Table 2에서 2개 task가 베이스라인 미달이므로 사실이 아니다.
6. **비용 보고의 불완전성**. Table 7은 MemoryVLA와만 비교하고, world memory 2048 tokens 유지에 따른 **episode 길이에 대한 비용 스케일링**은 보고되지 않았다. throughput 101.3 Hz는 action chunk(16-step)를 감안한 수치로 보이지만 산정 방식이 명시되지 않았다.
7. **코드/가중치 미공개**. 논문에 저장소 링크가 없고, voxel-hashing 융합과 consolidation은 디테일 민감도가 높아 독립 재현이 쉽지 않다.
8. **Cross-embodiment 일반화 미검증**. 모든 실세계 실험이 단일 Franka + D415 구성이며, 다른 로봇/카메라 배치로의 전이는 다루지 않는다.

## 10. 총평

**기여의 본질은 "정확도 SOTA"가 아니라 "센서 비용 절감"이다.** LIBERO 97.6%는 3rd+wrist OpenVLA-OFT의 97.1%를 겨우 넘고 LIBERO-Long은 사실상 동률이다. 그러나 그것을 **카메라 한 대**로 달성했다는 점, 그리고 그 격차가 기하학적으로 어려운 RLBench(+15.8%p vs wrist 베이스라인)와 실세계 long-horizon(+9.0%p vs MemoryVLA)에서 확대된다는 점이 이 논문의 실질이다. 로봇 배포 관점에서 3인칭 카메라 리깅 제거는 상당한 실용적 가치다.

방법론적으로는 **로보틱스의 고전(TSDF voxel integration)을 latent token 공간으로 옮긴 것**이 가장 인상적이다. depth confidence를 그대로 fusion weight로 쓰는 설계는 단순하면서 원리적이고, ego context를 query로 삼아 world memory를 조회하는 구조는 "관련 있는 공간만 본다"는 직관을 attention으로 깔끔히 구현했다. Ablation도 두 메모리가 각각 −15.5%p / −13.0%p로 대등하게 필수임을 보여 설계가 장식적이지 않음을 입증한다.

반면 취약점은 **파이프라인이 기하 추정 스택 위에 세워졌다는 점**이다. hand-eye calibration과 monocular depth 두 축 어디가 흔들려도 world memory 전체가 오염되는데, 이에 대한 robustness 실험이 하나도 없다. 여기에 코드 미공개, 소규모 평가, 본문의 과장 서술이 겹쳐 신뢰도를 깎는다. 아이디어의 방향성은 옳고 결과도 설득력이 있으나, 검증의 엄밀성은 주장 강도에 못 미친다.

**정리**: wrist-only VLA라는 명확한 문제 설정 + spatial/temporal 메모리의 원리적 통합 = 방향성 있는 좋은 논문. 다만 "SOTA 달성"보다는 "동등 성능을 더 싼 센서로"로 읽는 것이 정확하다.

## 11. 재현/확장을 위한 체크리스트

- [ ] **OpenVLA-7b 체크포인트 확보** — LLaMA-2 7B + DINOv2/SigLIP 백본의 출발점. 이것 없이는 학습 예산이 크게 달라진다.
- [ ] **Depth Anything v3 streaming 모드** — 단일 프레임 추론이 아니라 과거 프레임을 쓰는 streaming 변형이어야 temporal depth consistency가 확보된다. 이 차이를 놓치면 world memory가 프레임마다 흔들린다.
- [ ] **Hand-eye calibration T_h2e 정밀도** — 재현의 최대 리스크. 시뮬레이션에서는 정확한 값이 주어지지만 실기에서는 여기서 성패가 갈린다.
- [ ] **핵심 하이퍼파라미터 고정**: voxel 0.025 m (0.05로만 올려도 실세계 −5.5%p), memory capacity 2048 tokens, ego-working token 4개, memory length 16, action chunk 16, DDIM 10 steps, CFG 1.5, condition dropout 0.1.
- [ ] **Retrieval layer 비대칭 유지**: ego-working 2 layers vs ego-guided world 4 layers. world 쪽이 두 배 깊다.
- [ ] **Sliding window + permanent first-frame anchor 동시 구현** — 둘 중 하나만 구현하면 row 4(naive accumulation, −11.5%p) 쪽에 가까워진다.
- [ ] **학습 예산**: 8×A100, LIBERO Spatial/Object/Goal 각 20k step, Long+90 통합 40k step, RLBench 80k step.
- [ ] **평가 규약**: final checkpoint 사용(best-val 금지), LIBERO 50 rollout/task, RLBench 20 trial/task, 실세계 50 trial/task + 전체 sub-goal 완료만 성공 인정.
- [ ] **논문 미공개 항목**: sliding window 크기 W, decay λ, DiT block 수 L, intent query 개수 N — 재현 시 직접 탐색 필요.

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 답변 방향 |
|---|---|---|
| 1 | LIBERO-Long 94.6은 3rd+wrist OpenVLA-OFT 94.5와 동률이다. 그래도 SOTA인가? | 절대 수치로는 아니다. 주장은 "카메라 1대로 3rd+wrist와 동등"이며, 이 프레이밍에서만 유효하다. 격차는 RLBench·실세계에서 나타난다. |
| 2 | Hand-eye calibration이 틀어지면? | 치명적이다. world memory 전체가 잘못된 voxel에 누적된다. 논문에 calibration 노이즈 robustness 실험이 없어 이 질문은 미해결로 남는다. |
| 3 | Monocular depth가 실패하는 투명/반사 물체는? | Confidence weighting(Eq. 5)이 억제하도록 설계됐지만 검증 실험이 없다. 실세계 task 목록에도 투명 물체가 없다. |
| 4 | "Permanent first-frame anchor"는 장면이 변하면 독이 되지 않나? | 그렇다. 정적 tabletop 가정에 의존한다. 사람이 개입해 물체를 옮기는 동적 시나리오는 평가되지 않았다. |
| 5 | Ablation에서 LIBERO 낙폭이 실세계보다 훨씬 작은 이유는? | 시뮬레이션이 부분 관측 문제를 충분히 스트레스하지 못하기 때문. 이는 오히려 저자 논지("실세계가 진짜 시험대")를 강화하지만, 동시에 LIBERO 수치의 설득력을 약화한다. |
| 6 | Memory length 32가 실세계 69.8로 더 높은데 왜 16을 썼나? | 저자는 LIBERO 저하(97.2)와 "marginal benefit"을 근거로 든다. 다만 0.3%p는 200 trial 기준 노이즈이며, 사후 선택 여지가 있다. |
| 7 | Voxel 0.1m에서 58.5까지 떨어지는 민감도는 실배포에 문제 아닌가? | 그렇다. 작업 공간 스케일에 따라 재튜닝이 필요하다는 뜻이며, 논문은 스케일 적응 메커니즘을 제시하지 않는다. |
| 8 | 101.3 Hz throughput이 latency 0.158s와 모순되지 않나? | action chunk 16-step을 나눈 값으로 보인다(16/0.158 ≈ 101). 논문에 산정식이 명시되지 않아 해석의 여지가 있다. |
| 9 | "consistent improvements across all tasks"는 사실인가? | 아니다. Table 2에서 Frame off Hanger(65.0 vs FiS-VLA/π₀ 70.0), Wine at Rack(75.0 vs MemoryVLA 80.0)에서 뒤진다. |
| 10 | LLM 7B를 두고 액션은 300M DiT가 낸다면, 7B가 정말 필요한가? | 논문에 LLM 스케일 ablation이 없다. intent tracking/semantic routing만 담당한다면 더 작은 VLM으로 대체 가능한지가 자연스러운 후속 질문이다. |
| 11 | Episode가 길어지면 메모리 비용은? | world memory는 2048 token으로 상한이 있고 ego 쪽은 consolidation으로 병합되므로 이론상 bounded. 다만 episode 길이별 비용 스케일링 측정치는 제시되지 않았다. |
| 12 | Cross-embodiment로 전이되나? | 미검증. 모든 실세계 실험이 단일 Franka + D415 구성이며, hand-eye calibration 의존성을 감안하면 embodiment 전이는 비자명한 과제다. |

<!-- VERIFIED: pdf -->
