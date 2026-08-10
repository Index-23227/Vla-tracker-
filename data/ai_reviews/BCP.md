# BCP: Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution

**arXiv**: 2608.03483 (2026-08-04) · **소속**: Peking University, Microsoft Research Asia · **공동 1저자**: Weichen Xu, Zhenhua Liu · **교신**: Jiaolong Yang · **기타 저자**: Lin Luo, Yaobo Liang, Chengtang Yao, Qingyu Mei, Jian Cao, Xixin Cao, Xing Zhang, Baining Guo

## 1. 배경 및 동기

### 문제 설정
Action chunking은 최근 VLA의 표준 설계다. 모델은 길이 `H`의 action chunk를 예측하고(prediction horizon), 그중 `E`개를 실행한 뒤(execution horizon) 재관측·재계획한다. 문제는 거의 모든 chunk 기반 VLA(OpenVLA-OFT, π0, LingBot-VLA 등)가 **`E`를 에피소드 전체에 걸쳐 고정**한다는 점이다. 그 결과 replanning은 task 진행 상황과 무관한 **주기적(periodic) 스케줄**이 된다.

자유 공간 이동 구간에서는 긴 chunk를 그대로 실행하는 것이 부드럽고 안정적이다. 반면 파지·삽입 같은 정밀 구간에서는 작은 pose/contact 오차가 급격히 누적되므로, 그 단계에 진입하기 **직전에** 재관측이 필요하다. 고정 스케줄은 이를 보장할 수 없다. 재계획 경계가 임계 단계 앞에 떨어지지 않으면 로봇은 낡은(stale) chunk로 그 단계를 수행한다.

### 핵심 관찰: phase-shift 실험
저자들의 동기 실험이 논문에서 가장 설득력 있는 부분이다. 동일한 50-step 고정 horizon을 쓰되 **초기 phase `φ`만 바꾼다** — 첫 chunk에서 `φ`개만 실행하고 이후 50 step마다 재계획. 즉 horizon 길이는 완전히 동일하고 **재계획 경계가 어디에 떨어지는지만** 다르다.

RoboTwin 2.0 50개 task, LingBot-VLA 기준 (Table 1):

| | Original | Upper Bound | Lower Bound | Gap |
|---|---|---|---|---|
| SR | 88.52% | 93.65% | 82.50% | **11.30%** |

Upper/Lower는 task별 best/worst phase를 hindsight로 고른 oracle이다. **타이밍이라는 단 하나의 선택이 11.30%p를 움직인다.** Place Dual Shoes에서 `φ=32`는 낡은 gripper closure로 실패하고 `φ=23`은 경계가 조작 단계 앞에 떨어져 성공한다(Fig. 1).

Appendix Fig. A.2는 여기에 결정적 한 방을 더한다. 50개 task의 best phase 분포에 대한 chi-square uniformity test 결과 χ² = 5.00, p = 0.287로 **균등분포를 기각하지 못한다.** 즉 "좋은 phase"는 존재하지 않는다. 임계 순간은 task마다 다른 시점에 오므로, 어떤 고정 phase도 모든 task에 정렬될 수 없다. 이것이 "horizon을 hyperparameter에서 per-chunk decision으로 바꿔야 한다"는 논지의 통계적 근거다.

### 세 가지 학습상의 난점
1. **Ordinal/prefix-sharing 구조**: 40-step과 50-step 실행은 앞 40개 action을 공유하고 "40 이후에도 계속할 것인가"만 다르다. Softmax 분류기는 이를 무순서 독립 클래스로 취급해 구조를 버린다.
2. **Per-chunk supervision 불가능**: 어떤 horizon이 최적인지 관측 불가능하고, 결정들이 시간적으로 결합되어 있어(현재 horizon이 다음 관측을 결정) 유일한 ground-truth stopping label이 존재하지 않는다.
3. **Success-only reward의 붕괴**: 성공만 보상하면 정책은 "짧게 자르고 자주 재계획"으로 수렴한다. 성공률은 유지되지만 VLA 호출이 폭증한다.

## 2. 방법론 심층 분석

### 2.1 문제 정형화
후보 execution horizon 집합 `E = {e_1, ..., e_M}`, `1 ≤ e_1 < ... < e_M ≤ H`. 각 재계획 시점에서 정책 `π_θ(·|s_t)`가 `E_t`를 선택하고 로봇은 prefix `a_t^{1:E_t}`를 실행한 뒤 VLA를 다시 호출한다. **VLA는 여전히 모든 action을 생성하고, 정책은 오직 "얼마나 오래 신뢰할 것인가"만 결정한다.** Base VLA는 완전히 frozen이다.

### 2.2 Bernoulli-Continuation Head
Frozen VLA가 chunk를 예측하는 과정에서 이미 만들어진 표현을 재사용하므로 **추가 VLA forward가 없다.** 세 가지 입력:
- Visual-language token `F_t ∈ R^{N×d_v}` — 장면·지시 문맥
- 각 action step `j`의 denoised action `a_t^j ∈ R^{d_a}`
- 최종 step의 action-velocity feature `u_t^j ∈ R^{d_u}` — 생성 궤적의 motion-level 단서

`h_t^j = Proj_a([u_t^j; a_t^j])`, `F̂_t = Proj_v(F_t)`

학습 가능한 `[CLS]`를 앞에 붙여 `X_t = [[CLS], F̂_t, h_t^1, ..., h_t^H]`를 2-layer Transformer encoder에 넣고, `[CLS]` 출력을 `M-1`개 logit `z_t`로 매핑한다.

### 2.3 Bernoulli 연쇄 분해 (핵심 기여)
Softmax 대신 **순서화된 continue 결정의 연쇄**로 모델링한다. `p_t^i = σ(z_t^i)`는 `e_i`에서 `e_{i+1}`로 계속할 확률이며,

```
π_θ(E_t = e_k | s_t) = (∏_{i=1}^{k-1} p_t^i) · (1 - p_t^k),   1 ≤ k < M
π_θ(E_t = e_M | s_t) = ∏_{i=1}^{M-1} p_t^i
```

이 factorization은 chunk 실행의 nested 구조와 정확히 대응한다. 긴 horizon은 chunk가 **반복적으로 신뢰 가능하다고 판정될 때만** 도달 가능하고, 인접 horizon은 자연히 유사한 확률을 받는다. Flat categorical에는 없는 ordinal inductive bias다.

### 2.4 GRPO + Replanning-Efficiency Reward (RER)
Per-chunk label이 없으므로 trajectory-level 보상으로 RL 학습한다. GRPO를 쓰되 **한 가지 변형**이 있다: `G-1`개 adaptive rollout에 더해 **고정 horizon reference trajectory `τ_ref`를 1개 추가로 롤아웃**하여 advantage를 anchor한다.

```
Â_i = (R_i - mean(R)) / std(R),   R = {R_1, ..., R_{G-1}, R_ref}
```

`τ_ref`는 BCP 결정이 없으므로 gradient를 받지 않고 오직 기준점 역할만 한다.

RER은 성공과 효율을 결합한다. `S_i ∈ {0,1}`는 성공 여부, `C_i`/`C_ref`는 VLA 호출 수:

```
η_i = tanh(log(C_ref / C_i))
R_i = S_i · (1 + δ₊·max(η_i, 0) + δ₋·min(η_i, 0)),   δ₊ = 0.7 > δ₋ = 0.3 > 0
```

설계 의도가 명확하다. (a) 실패 궤적은 효율과 무관하게 0 — 짧고 실패하는 실행을 exploit할 수 없다. (b) 비대칭 `δ₊ > δ₋` — 절약은 크게 보상, 추가 재계획은 약하게만 페널티. 성공을 지킬 수 있다면 더 자주 재계획해도 된다. (c) **group 붕괴 방지**: 이진 성공만으로는 모든 궤적이 같은 결과일 때 advantage가 0이 되어 group이 무의미해진다. 효율 항이 all-success group 내부에 순위를 만들어 non-zero advantage를 살린다. Reference까지 포함되므로 "고정 전략은 성공했는데 adaptive가 실패한" 경우 음의 신호가 확실히 전달된다. Reference와 모든 adaptive가 전부 실패한 group만 폐기된다.

## 3. 실험 설계

- **RoboTwin 2.0**: Clean에서만 학습, Clean/Randomized 양쪽 평가. Base policy 3종 — LingBot-VLA (w/o depth), ABot-M0, ACT. 모두 prediction/execution horizon 50, 후보 `{15, 20, 25, ..., 50}` (8-way).
- **LIBERO / LIBERO-PRO**: base π0.5, prediction horizon 10, 후보 `{2, 4, 5, 6, 8, 10}` (6-way). LIBERO에서 학습해 LIBERO-PRO로 zero-shot 전이.
- **실기**: AGIBOT G1, Grasping Bottle / Hanging Mug. Task당 teleop 250 궤적 + 범용 grasping 2,000 궤적으로 SFT 40,000 step → 실기 128 궤적으로 BCP만 RL 학습. 평가는 동일 50 seed × 3 trial.

## 4. 시스템/학습 세부사항

| 항목 | RoboTwin 2.0 (LingBot-VLA) | LIBERO (π0.5) |
|---|---|---|
| Prediction horizon | 50 | 10 |
| Denoise steps | 10 (Euler ODE) | 10 (Euler ODE) |
| Num classes | 8 | 6 |
| Transformer layers / heads | 2 / 8 | 2 / 8 |
| Advantage | GRPO, group size 8 | GRPO, group size 8 |
| Clip ratio (low/high) | 0.2 / 0.28 | 0.2 / 0.28 |
| Entropy bonus | 0.05 | 0.05 |
| RER δ₊/δ₋ | 0.7 / 0.3 | 0.7 / 0.3 |
| Rollout / Eval envs | 256 / 100 | 64 / 500 |
| Training steps | 300 | 300 |
| Global batch / update epochs | 512 / 2 | 512 / 4 |
| LR / scheduler | 1e-4 / constant | 1e-4 / constant |

하드웨어: 8× A100 40GB. 구현: RLinf (VLA RL 오픈소스 프레임워크). **학습 파라미터는 16.393M** — RL로 action expert 전체를 튜닝하는 442.803M 대비 27배 작다.

## 5. 실험 결과 심층 분석

### RoboTwin 2.0 (Table 2)
LingBot-VLA 기준 Clean에서 90% 미만인 13개 low-success task와 전체 50 task 평균:

| | 13 low-success (Clean) | 13 (Rand.) | 50 tasks (Clean) | 50 (Rand.) |
|---|---|---|---|---|
| LingBot-VLA | 75.15 | 73.54 | 89.88 | 88.78 |
| **LingBot-VLA + BCP** | **86.23 (+11.08)** | **83.46 (+9.92)** | **93.94 (+4.06)** | **92.84 (+4.06)** |

Clean에서만 학습했음에도 Randomized에서 동일한 +4.06을 얻는다는 점이 중요하다. 학습 시 본 시각 조건에 overfit한 것이 아니라 **task-stage 수준의 "언제 chunk를 신뢰할 수 있는가"** 지식을 학습했다는 근거다.

Task별로는 Hanging Mug 41→87 (Clean), 36→62 (Rand.)가 압도적이다. Blocks Ranking Size 70→88, Turn Switch 60→72, Click Alarmclock 80→91, Place Object Scale 80→91 — 모두 정밀 파지/배치 단계를 포함하는 task다. 반대로 이미 96%인 Adjust Bottle은 97%로 거의 변화가 없다. **BCP의 이득은 타이밍 민감 task에 집중된다.**

다른 base policy에서도 재현된다. ACT 16.08→21.46 (13 task, Clean), ABot-M0 82.74→85.96 (Clean) / 76.76→85.40 (Randomized). Randomized에서 ABot-M0의 +8.64는 오히려 LingBot-VLA보다 크다.

### LIBERO / LIBERO-PRO (Table 3)

| Method | Spatial | Object | Goal | Long | **Avg.** |
|---|---|---|---|---|---|
| π0.5 | 98.5 | 98.7 | 98.1 | 92.5 | 97.0 |
| π0.5 + AAC | 99.1 | 99.2 | 98.0 | 95.2 | 97.9 |
| π0.5 + AutoHorizon | 99.1 | 99.2 | 97.5 | 91.6 | 96.9 |
| **π0.5 + BCP** | **99.6** | **99.8** | **99.2** | **96.2** | **98.7** |

4개 suite 모두에서 최고. AutoHorizon은 Long에서 오히려 base보다 떨어진다(92.5→91.6)는 점이 대비된다.

LIBERO-PRO (Object suite 위치 섭동, AAC 프로토콜):

| Method | ×0.2 | ×0.3 | ×0.4 | Avg. |
|---|---|---|---|---|
| π0.5 | 53.2 | 29.9 | 9.5 | 30.9 |
| π0.5 + AAC | **57.4** | 35.3 | 11.8 | 34.8 |
| **π0.5 + BCP** | 57.0 | **39.4** | **16.8** | **37.7** |

섭동이 강해질수록 격차가 벌어진다(×0.4에서 9.5 → 16.8, 1.77배). AAC 대비 우위도 ×0.2에서는 오히려 −0.4지만 ×0.4에서 +5.0이다. 환경이 어려워질수록 "낡은 chunk를 계속 믿는 비용"이 커지기 때문이라는 해석이 자연스럽다.

### 런타임 (Table 4)
Continuation head가 경량이므로 per-query 추론은 938.10 → 940.13 ms (**+2.03 ms**). VLA 호출은 5.382 → 5.614로 약간 늘지만, 실행 control step은 269.075 → 248.102로 줄어든다(성공 궤적이 길게 헤매지 않으므로). ALOHA-AgileX 50 Hz 기준 총 런타임 **10.43 s → 10.24 s로 오히려 감소**. Fig. 4의 SR-runtime trade-off에서 BCP는 평가된 모든 전략 중 최고 SR + 최저 런타임 지점을 차지한다. Uncertainty Proxy와 AAC는 chunk를 여러 개 샘플링해 entropy를 추정하므로 런타임이 크게 늘어난다.

### Execution-horizon 전략 비교 (Fig. 4)
Random, Fixed 20/30/40, Action Trigger, Uncertainty Proxy, AAC, AutoHorizon 8종과 비교. **Fixed 20은 full-chunk baseline을 이기지 못한다** — 짧은 horizon 자체가 답이 아니라는 직접적 증거다. 균등한 스케줄은 짧아져도 여전히 임계 단계와 어긋날 수 있다.

## 6. 실기 결과

AGIBOT G1, 50 seed × 3 trial:

| Task | Full-chunk | AAC | **BCP** |
|---|---|---|---|
| Grasping Bottle | 74% | - | **92%** |
| Hanging Mug | 44% | 48% | **84%** |

Hanging Mug는 mug 손잡이와 hook의 상대 위치 오차가 2 cm 이내여야 하는 task로, +40%p는 시뮬레이션 이득(+46%p in ablation)과 일관된다. Grasping Bottle은 매끄러운 병 표면 때문에 파지가 pose 오차에 민감한 경우다. 두 실패 모드 모두 "정밀 단계 직전 재관측"으로 해소되는 유형이다.

## 7. Ablation 분석

### 구성요소 분해 (Table 5, Hanging Mug / Clean)

| Method | Trainable Params | SR |
|---|---|---|
| SFT | - | 41% |
| RL (Fixed-horizon, action expert 튜닝) | 442.803 M | 79% (+38) |
| RL (Softmax Head) | 16.394 M | 78% (+37) |
| RL (BC Head) | 16.393 M | 83% (+42) |
| **RL (BC Head) + RER** | 16.393 M | **87% (+46)** |

두 가지가 명확하다. (1) **16M 헤드가 442M action expert 전체 RL 튜닝을 넘어선다** (78~87% vs 79%). (2) Softmax → Bernoulli 교체만으로 +5%p — ordinal 구조가 실제로 기여한다. RER 추가로 +4%p 더.

### 효율까지 본 ablation (Table A.3) — 이쪽이 더 흥미롭다

| Method | VLA Calls | Exec. Steps | Runtime (s) | SR |
|---|---|---|---|---|
| SFT | 13.525 | 676.250 | 26.21 | 41% |
| RL (Fixed-horizon) | 9.392 | 469.583 | 18.20 | 79% |
| RL (Softmax Head) | 12.142 | 464.708 | 20.71 | 78% |
| RL (BC Head) | 13.825 | 427.750 | 21.55 | 83% |
| **RL (BC Head) + RER** | 10.158 | 430.958 | **18.17** | **87%** |

RER 없는 BC Head는 **VLA 호출이 13.825로 SFT보다도 많다.** 성공만 보는 보상이 실제로 "짧게 자르기" 붕괴를 일으킨다는 것을 정량적으로 확인해준다. RER은 실행 step 수를 거의 유지한 채(427.750 → 430.958) 호출만 10.158로 줄여 SR과 런타임을 동시에 개선한다. 세 번째 challenge가 가설이 아니라 실측된 문제였음을 보여주는 표다.

### 하이퍼파라미터 (Table A.4, Hanging Mug)
- Transformer layer 1/2/4/6 → 82/**87**/84/83%. 2층이 최적이고 깊게 만들면 오히려 하락.
- 후보 horizon `{15,20,...,50}` 87% vs `{20,30,40,50}` **78%**. 세분화된 간격이 9%p를 만든다 — 경계를 정밀 단계 "바로 앞"에 놓을 수 있는가가 핵심임을 재확인.
- Rollout temperature 1.0 → 1.6: 87 → 82%. LR constant → cosine: 87 → 82%.

## 8. 관련 연구 비교

Adaptive horizon 계열은 크게 두 갈래다.
1. **Test-time 신뢰도 프록시**: entropy/attention 기반 (AAC, AutoHorizon, Zhu et al. 2026). 직접 최적화되지 않으며 매 step 여러 chunk 샘플링이 필요해 비싸다. Fig. 4에서 이들의 런타임 열세로 확인된다.
2. **전용 메커니즘 학습**: cross-horizon consensus (Jing et al. 2025), closed-loop verification (Wang et al. 2026d,c; Pan et al. 2026). 효과적이나 절차가 복잡하다.

BCP는 execution-horizon **분포 자체를** 단일 forward로 예측한다. 또 다른 축은 RL for VLA다. SimpleVLA-RL, πRL, π*0.6, VLA-RL, Q-chunking, AC³는 모두 **어떻게 action을 생성할지**를 개선하며 큰 모델을 업데이트한다. BCP는 직교적으로 **언제 멈출지**만 학습한다. Table 5의 442.803M vs 16.393M 비교가 이 구도를 그대로 보여준다.

## 9. 한계 및 미해결 문제

저자들이 명시한 한계는 정직하다: BCP는 VLA가 생성한 action을 **수정하지 않는다.** 따라서 base VLA가 근본적으로 잘못된 chunk를 내놓으면 horizon을 바꾸거나 재계획 시점을 옮겨도 궤적을 교정할 수 없다. BCP가 다루는 것은 **재계획 타이밍과 임계 단계의 misalignment로 인한 실패**뿐이며, action 생성 능력 자체의 오류가 아니다.

리뷰어 관점에서 추가로 보이는 문제들:
- **후보 집합 `E`가 여전히 수작업 hyperparameter다.** 고정 `E` 하나를 없애고 후보 집합 설계라는 새 hyperparameter를 도입했으며, Table A.4는 이 선택에 9%p가 걸려 있음을 보여준다.
- **Ablation이 사실상 Hanging Mug 단일 task에 집중되어 있다.** BCP 이득이 가장 큰 task이므로 구성요소 기여가 과대평가될 여지가 있다.
- **정책이 무엇을 보고 판단하는지 해석이 없다.** Continue 확률이 접촉 직전에 실제로 떨어지는지, gripper 상태와 상관관계가 있는지에 대한 분석이 없어 Fig. 3의 정성 예시에 의존한다.
- **`τ_ref` anchoring의 기여가 분리되지 않았다.** GRPO에 reference 궤적을 추가하는 것은 비용(rollout 1개 추가)이 있는 설계 선택인데 해당 ablation이 없다.

## 10. 총평

문제 제기가 이 논문의 최대 강점이다. Phase-shift 실험 + best-phase 균등성 검정(χ²=5.00, p=0.287)은 "고정 horizon은 길이가 아니라 정렬(alignment)의 문제"라는 주장을 반박하기 어렵게 만든다. 11.30%p라는 hindsight oracle gap은 이 방향의 headroom을 정량화한 숫자로 앞으로 자주 인용될 것이다.

방법론은 **의도적으로 미니멀**하다. Bernoulli 연쇄는 ordinal regression의 고전적 아이디어(continuation-ratio 모델)를 chunk 실행의 nested 구조에 정확히 대응시킨 것으로, 새롭기보다 **적합하다**. RER의 비대칭 설계와 reference anchoring은 group RL에서 all-success/all-fail group의 gradient 소실을 다루는 실용적 처방이고, Table A.3이 그 필요성을 실측으로 뒷받침한다.

가장 인상적인 수치는 성능이 아니라 비용 구조다. 16M 파라미터 헤드가 442M action expert RL 튜닝을 능가하고, +2.03 ms 오버헤드로 총 런타임을 **낮춘다.** Base VLA가 frozen이므로 어떤 chunk 기반 정책에도 붙일 수 있고, 실제로 ACT(2023년급 경량 정책)부터 LingBot-VLA, π0.5까지 4개 base에서 일관되게 작동했다. Plug-and-play 주장이 실증된 드문 사례다.

반면 기여의 폭은 좁다. Action 생성은 건드리지 않으므로 이득의 상한이 "base VLA가 이미 좋은 chunk를 만들지만 타이밍 때문에 실패하는 경우"로 제한된다. 50-task 평균 +4.06 vs 13개 low-success task +11.08의 격차가 이를 그대로 보여준다. 그래도 이 좁음은 결함이라기보다 정직한 scope 설정에 가깝다.

## 11. 🔥 예상 날카로운 질문 모음

1. **후보 집합 `E`가 새로운 hyperparameter 아닌가?** `{15,...,50}` vs `{20,30,40,50}`이 87% vs 78%를 만든다면, "horizon을 hyperparameter에서 해방시켰다"는 주장은 hyperparameter를 한 단계 옮긴 것에 불과하지 않은가? 최소 간격을 1 step까지 줄이면 어떻게 되는가?
2. **`e_1 = 15`라는 하한의 근거는?** 15보다 짧은 실행이 필요한 상황(접촉 직전 미세 조정)은 구조적으로 배제된다. π0.5에서는 `{2,4,...}`로 훨씬 짧은데, 왜 RoboTwin에서는 15가 하한인가?
3. **Bernoulli 연쇄의 이득이 정말 ordinal 구조 때문인가?** Softmax 78% vs BC 83%인데, 파라미터 수가 거의 같다(16.394M vs 16.393M). Sigmoid의 gradient 특성이나 초기화 차이가 아니라 prefix-sharing이 원인임을 어떻게 분리했는가? Ordinal softmax(누적 로짓) baseline이 없는 이유는?
4. **Reference trajectory anchoring의 기여는 얼마인가?** Group마다 rollout을 1개 더 쓰는 비용이 있는데 이를 뺀 ablation이 없다. 단순히 group size를 8로 늘린 것과 구분되는가?
5. **Ablation이 Hanging Mug 하나뿐인 것은 공정한가?** 41% → 87%로 BCP 이득이 최대인 task다. 이미 95% 이상인 task에서 각 구성요소가 여전히 기여하는지, 혹은 RER이 오히려 해가 되는지 확인이 필요하다.
6. **RER 없는 BC Head가 VLA 호출 13.825로 SFT(13.525)보다 많은 것을 어떻게 설명하는가?** 성공률은 41→83%로 올랐는데 호출이 늘었다면, 성공 궤적당 호출 밀도가 극단적으로 높아진 것이다. 이는 정책이 거의 항상 최단 horizon(15)을 고르는 degenerate 해에 가까운 것 아닌가? 선택된 horizon 분포를 보여줄 수 있는가?
7. **LIBERO-PRO ×0.2에서 AAC(57.4)에 지는(57.0) 이유는?** 약한 섭동에서는 BCP의 이점이 사라진다면, "어려울수록 유리하다"는 주장의 경계는 어디인가?
8. **Randomized 이득이 Clean과 정확히 같은 +4.06인 것은 우연인가?** 13-task 기준으로는 +11.08 vs +9.92로 다르다. 50-task에서 소수점까지 일치하는 것이 통계적으로 자연스러운가? 시드 수와 분산은?
9. **Continue 확률이 실제로 "임계 단계"에 반응하는가?** 접촉 발생 시점, gripper 개폐 시점과 `p_t^i` 하락의 시간적 상관을 보여줄 수 있는가? 없다면 학습된 것이 stage 인식이 아니라 단순한 "chunk 내 위치 prior"일 가능성은?
10. **`u_t^j` (final-step action-velocity feature)의 기여는?** Flow matching/diffusion 기반 VLA에만 존재하는 신호인데, ACT처럼 velocity field가 없는 정책에서는 무엇을 쓰는가? 입력 ablation이 없다.
11. **실기에서 128 궤적으로 RL을 학습했다는데 안전성은 어떻게 확보했는가?** 실패 궤적 수집 과정의 비용과 위험, 그리고 이 128개가 8-way horizon 정책을 학습하기에 충분하다는 근거는?
12. **World Action Model 확장 제안이 자연스러운가?** WAM은 예측된 미래 관측을 이미 갖고 있으므로 "언제 재계획할지"를 chunk 표현이 아니라 예측 오차로 판단할 수 있다. BCP의 학습된 헤드가 그 대안보다 나을 이유가 있는가?

## 12. 세미나 토론 포인트

- **"언제"와 "무엇"의 분리**: VLA 개선 노력이 대부분 action 생성 품질(무엇)에 쏠려 있는데, BCP는 실행 스케줄(언제)이라는 직교 축에서 16M 파라미터로 442M 튜닝을 이겼다. 이런 "저평가된 직교 축"이 VLA 스택에 또 어디 남아 있는가 — 관측 타이밍, 카메라 선택, chunk 길이 자체?
- **Hindsight oracle을 벤치마크 지표로**: phase-shift upper bound(93.65%)는 "타이밍만으로 도달 가능한 상한"이다. 이런 oracle gap 측정을 다른 설계 선택(해상도, denoise step 수)에도 적용해 headroom을 정량화하는 방법론으로 일반화할 수 있는가?
- **Ordinal 구조의 재발견**: continuation-ratio 모델은 통계학의 고전이다. VLA에서 "구조를 가진 이산 선택"(horizon, expert 수, denoise step 수)을 flat softmax로 다루는 관행이 얼마나 많은 성능을 버리고 있는가?
- **Reward hacking의 구체적 사례로서 Table A.3**: 성공만 보상했더니 호출이 SFT보다 많아진 현상은 RL 설계 수업의 좋은 교재다. RER의 비대칭 `δ₊ > δ₋`는 "효율은 보상하되 안전 마진은 허용한다"는 명시적 선택인데, 다른 로봇 RL 보상 설계에도 이 비대칭 원칙이 일반화되는가?
- **Frozen backbone + 경량 헤드 패러다임**: LoRA/adapter가 표현 학습에서 한 일을 BCP가 실행 정책에서 하고 있다. 이 패턴이 확산되면 VLA 배포는 "하나의 frozen 대형 정책 + 여러 개의 task별 경량 제어 헤드"로 수렴하는가?
- **평가 프로토콜 문제**: 기존 VLA 논문들은 고정 horizon 하나로 평가한다. 그런데 그 선택 하나가 11.30%p를 움직인다면, 현재의 벤치마크 순위표는 어느 정도까지 "실행 스케줄 운"을 반영하고 있는가? Horizon sweep 보고를 표준화해야 하는가?

<!-- VERIFIED: pdf -->
