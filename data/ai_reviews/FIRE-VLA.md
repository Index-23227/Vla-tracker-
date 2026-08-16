# FIRE-VLA: Failure-Informed Self-Evolution for Vision-Language-Action Models in Autonomous Driving

> **한 줄 요약**: GRPO는 rollout group 내부의 **reward 대비(contrast)**로 학습하는데, 한 group의 모든 궤적이 똑같이 나쁜 "**unresolved failure group**"에서는 실패들 사이의 순위만 매길 뿐 실패 영역을 **벗어나는 행동**을 지목하지 못한다. FIRE-VLA는 이런 group만 골라내어(low-reward × low-diversity 게이트), **같은 크기의 frozen 자기 복제본**에게만 숨겨진 미래 궤적(privileged info)을 보여주고 student가 실제로 생성한 prefix 위에서 answer token 분포를 distill한다. 라운드가 끝나면 갱신된 policy가 다음 라운드의 teacher가 되어 실패 분포와 teacher가 함께 진화한다. Qwen2.5-VL-3B SFT 체크포인트에서 출발, nuScenes 6,019 샘플에서 G=4 평균 L2를 **1.848 → 1.500 m**, evaluation-persistent 실패율을 **13.03% → 11.20%**로 낮췄다. 단일 저자(Harbin Institute of Technology), arXiv 2608.13395.

---

## 1. 배경 및 동기

### 자율주행 VLA의 RL post-training
- 자율주행 VLA는 SFT로 driving prior와 구조화된 action format을 얻고, 그 뒤 RL로 자기 자신의 on-policy 응답 분포에서 샘플·평가하며 개선한다. AutoDrive-R²가 reasoning 기반 driving VLA에 GRPO-style post-training이 유효함을 보였다.
- GRPO는 하나의 prompt에 대한 여러 completion의 reward를 정규화해 PPO식 value model 없이 학습 신호를 만든다.

### 핵심 문제 제기 — "unresolved failure group"
- 성공과 실패가 섞인 group은 유용한 랭킹을 준다. 그러나 **모든 궤적이 비슷하게 나쁜** group에서는 상대 advantage가 실패 영역 **내부의** 순위만 알려준다.
- 저자가 명시적으로 선을 긋는 지점: 이것은 "GRPO의 gradient가 사라진다"거나 "objective가 무의미해진다"는 주장이 **아니다**. 부족한 것은 **corrective information** — 즉 policy 자신의 실패 영역에 특화된 교정 신호다. (Sec. 1, Sec. 5의 "Role of failure-informed routing"에서 반복 강조)
- 기존 해법은 failure refinement(ELF-VLA)나 privileged distillation(OPSD)로 이 정보를 얻는다. 본 논문의 질문은 **더 큰 외부 teacher 없이, 같은 VLA가 자신의 미해결 실패로 자신의 후속 policy를 가르칠 수 있는가**이다.

📌 [Figure 1 삽입] — nuScenes 장면과 GRPO vs FIRE-VLA 궤적 대비, 6개 미래 BEV waypoint 인터페이스

---

## 2. 방법론 심층 분석

전체는 **한 라운드 = 4단계**로 구성된다 (Figure 2): ① failure-informed routing → ② 두 개의 context 구성 → ③ privileged same-policy teaching → ④ policy update.

### 2.1 Task 정의와 reward (Sec. 3.1)
- 입력 x_i = (전방 카메라 이미지 I_i, ego-history 설명 h_i). 숨겨진 미래 궤적 τ*_i = (p*_{i,1}, …, p*_{i,6})은 0.5초 간격 6개 waypoint.
- policy는 reasoning을 먼저 출력하고 `<answer>` span 안에 6개 ego-frame waypoint를 낸다.
- 유효 rollout의 reward (Eq. 1):

  r_i^(g) = ( 1 + (1/6) Σ_{t=1..6} ‖p_{i,t}^(g) − p*_{i,t}‖₂² )^(−1)

  무효 응답은 reward 0. 이 **bounded reciprocal**이 나중에 "reward는 GRPO가 높은데 L2는 FIRE-VLA가 좋은" 역설의 원인이 된다(§6).
- GRPO advantage는 group 내 표준화 (Eq. 2).

### 2.2 Failure-Informed Routing (Sec. 3.2, Stage 1)
group마다 reward 평균 r̄_i, 모표준편차 σ_i, valid fraction v_i를 계산하고 **배치 상대적(batch-relative) 온라인 게이트**를 적용한다 (Eq. 4):

> z_i^train = 1[ r̄_i ≤ Q^batch_0.3(r̄) ∧ σ_i ≤ Q^batch_0.3(σ) ∧ v_i ≥ 0.5 ]

- 낮은 평균 = 성능이 나쁨, 낮은 분산 = 교정 대비(contrast)가 부족함, validity ≥ 0.5 = malformed 출력이 보조 경로를 지배하지 못하게 하는 안전장치.
- 이 조건을 만족하면 **unresolved failure group**. 라우팅된 group은 GRPO + PSD, 나머지는 GRPO 단독. **GRPO는 모든 group에 항상 활성**.
- 실제 발생 빈도(Appendix B): Round 1에서 56 group(224 응답), Round 2에서 58 group(232 응답).

### 2.3 Privileged Future Information (Sec. 3.3, Stage 2)
- 라운드 k에서 student π_θk와 teacher π_θ̄k는 **동일 파라미터**로 시작하고, teacher는 frozen.
- context (Eq. 5):
  - student: c^S = (I_i, h_i, y_{i,g,<t})
  - teacher: c^T = (I_i, h_i, ⟨priv⟩ τ*_i ⟨/priv⟩, y_{i,g,<t})
- 즉 **파라미터 규모가 동일**하고, teacher의 유일한 우위는 **숨겨진 정답 궤적에 대한 접근**뿐이다. 두 branch는 동일한 multimodal 관측과 student가 생성한 prefix를 공유한다.
- 안전장치: teacher는 no-grad, student 입력에 privileged token이 섞이지 않았는지 검사, causal masking으로 미래 student token 접근 차단, **배포 시에는 student branch만 남는다**.

### 2.4 On-Policy Answer-Token Distillation (Sec. 3.4, Stage 3)
- teacher의 **top-16 token + residual tail 버킷 1개**로 압축한 분포 P^T, P^S 사용. β = 0.5로 혼합 M을 만들고 **Jensen-Shannon divergence** J를 계산한 뒤 **token당 0.05로 clip** (Eq. 6).
- mask m_{i,g,t}는 **모호하지 않은 단일 `<answer>` span 내부의 생성 token**만 표시. 즉 reasoning 부분은 감독하지 않고 **궤적 답 token만** 감독한다.
- 집계는 **answer-token 가중 평균** (Eq. 7): rollout이나 group을 동등 기여하도록 재가중하지 않는다. routing 이후에는 별도의 per-response validity 필터가 없어서, span만 명확하면 format-invalid 응답도 참여한다(잘린 응답에 여는 태그 1개·닫는 태그 없음인 안전 케이스 포함). 여는 태그 누락/중복, 닫는 태그 중복, 빈 span은 zero mask.
- 최종 actor loss (Eq. 8):

  **L_B = L_GRPO,B + λ · z_B^train · L_PSD,B,  λ = 0.1**

### 2.5 Round-Wise Self-Evolution (Sec. 3.5, Stage 4)
- π_0 → π_1 → … → π_K. 라운드 k 내부에서는 π_k의 frozen privileged copy가 같은 파라미터로 초기화된 trainable student를 가르친다.
- 갱신된 student가 π_{k+1}이 되어 **새로운 rollout 분포**를 만들고, 그 unresolved group이 다음 라운드의 감독 위치를 결정한다 (Eq. 9). 실험은 **K = 2, 라운드당 75 update**.

📌 [Figure 2 삽입] — 4단계 라운드 다이어그램 (routing / two contexts / PSD / policy update)

---

## 3. 실험 설정 (Sec. 4.1, Appendix B)

| 항목 | 설정 |
|------|------|
| 초기화 | 공통 Qwen2.5-VL-3B SFT 체크포인트 |
| FIRE-VLA | 2 라운드 × 75 update (disjoint 600-prompt 부분집합 각각) |
| Standard GRPO | 연속 150 update, 동일한 1,200-prompt 집합 |
| Prompt pool | seed 42로 비복원 추출한 1,200개, 위치로 2등분 |
| Rollout | prompt당 4개, 총 4,800 student rollout (양쪽 동일) |
| 학습 | full-parameter BF16, FSDP, 4×RTX 3090, global batch 8, lr 5e-7, wd 1e-2, warm-up 0.05, grad-norm 1.0 |
| 토큰 | prompt 3,072 / response 384, 이미지 ≤ 196,608 px |
| 기타 | vision tower 학습 가능, LoRA·GRPO KL penalty 비활성, trajectory reward가 유일한 scalar reward |

- **매칭된 것**: 초기화, unique prompt 집합, rollout 수, policy update 수. **매칭 안 된 것**: minibatch 순서, 연산량(teacher forward 추가 + 라운드 2에서 optimizer/scheduler 재시작). 저자가 스스로 "equal-compute 비교가 아니다"라고 명시.

### 평가 프로토콜
- **6,019 샘플 / 150 scene-disjoint nuScenes scene / 5,119 unique image**. RL·SFT 학습과 scene 중복 없음, SFT와 image 중복 없음.
- 두 가지 디코딩: single-sample low-temperature(n=1, T=0.2 — vLLM 특성상 **엄밀히 deterministic은 아님**)와 G=4 stochastic(T=0.8). G=4는 후보별로 reward·L2를 구한 뒤 sample 내 평균 → dataset 평균.
- 모든 비교는 **sample 단위 paired**, CI는 **scene을 cluster로 하는 10,000회 bootstrap**.

### Persistent-failure detector (중요)
- **학습용 라우팅 게이트와 별개**. 겹치지 않는 256-sample / 37-scene SFT G=4 validation split에서 **한 번만** 보정하여 임계값을 **동결**: r̄ ≤ 0.4457839238, σ ≤ 0.0747977498, v ≥ 0.5.
- 이 detector는 학습을 라우팅하지 않으며 GRPO/FIRE-VLA 어느 쪽으로도 재보정되지 않는다 — 모든 체크포인트를 공통 척도로 측정하기 위한 장치.

---

## 4. 주요 결과

### 4.1 Single-sample low-temperature (Table 1)

| Method | Reward ↑ | L2@1s ↓ | L2@2s ↓ | L2@3s ↓ | Avg. L2 ↓ |
|---|---|---|---|---|---|
| Standard GRPO | 0.6788 | 0.2135 | 0.6396 | 1.5302 | 0.6421 |
| **FIRE-VLA** | 0.6785 | **0.1997** | **0.6106** | **1.3898** | **0.6023** |

- 세 horizon 모두에서 수치상 우세, Avg. L2 0.642 → 0.602 m. 그러나 scene-clustered paired CI가 **[−0.1104, 0.0087]로 0을 포함**. 저자는 "통계적으로 결정적인 우위가 아니라 **comparable nominal planning**"이라고 신중하게 해석한다.

### 4.2 G=4 stochastic (Table 2)

| Method | Reward ↑ | Avg. L2 ↓ | Persistent ↓ | Any>10m ↓ |
|---|---|---|---|---|
| Standard GRPO | **0.6370** | 1.8478 | 784/6019 (13.03%) | 1.35% |
| **FIRE-VLA** | 0.6156 | **1.5001** | **674/6019 (11.20%)** | **0.83%** |

- G=4 Avg. L2 **18.8% 감소**, paired Δ = −0.3476 m, 95% CI [−0.7711, −0.0319].
- Persistent 실패율 **−1.83 pp**, CI [−2.55, −1.11].
- 10 m 초과 후보를 하나라도 포함한 샘플 81 → 50개, **−0.52 pp**, CI [−0.83, −0.22]. 저자는 **tail risk를 명시적으로 최적화하지 않았는데도** 나타난 효과임을 강조.

### 4.3 SFT 기준선을 포함한 전체 (Table 3)

| Mode | Method | Reward | Avg. L2 | L2@1s | L2@2s | L2@3s |
|---|---|---|---|---|---|---|
| single | Common SFT | 0.6656 | 0.6240 | 0.2050 | 0.6402 | 1.4332 |
| single | Standard GRPO | 0.6788 | 0.6421 | 0.2135 | 0.6396 | 1.5302 |
| single | FIRE-VLA | 0.6785 | 0.6023 | 0.1997 | 0.6106 | 1.3898 |
| G=4 | Common SFT | 0.5565 | 1.0038 | 0.3649 | 1.0048 | 2.3445 |
| G=4 | Standard GRPO | 0.6370 | 1.8478 | 0.8692 | 1.9219 | 3.9002 |
| G=4 | FIRE-VLA | 0.6156 | 1.5001 | 0.6953 | 1.5912 | 3.0597 |

- **불편한 진실**: 두 RL 변형 **모두** SFT보다 G=4 stochastic L2가 **나쁘다**(1.0038 → 1.85/1.50). FIRE-VLA는 GRPO가 유발한 열화를 상당 부분 완화하지만 **SFT 수준의 stochastic robustness를 회복하지는 못한다**. 저자가 Appendix A에 이를 그대로 적어둔 점은 정직하다.

### 4.4 실패 회복 (Sec. 4.3)
- 고정된 SFT-reference-persistent 484 샘플 중 GRPO는 113개(23.35%), FIRE-VLA는 122개(25.21%) 회복. Δ = +1.86 pp, CI [−1.13, 4.90] → **결정적이지 않음**.
- 즉 "현재 persistent 유병률은 낮췄지만, 원래 SFT 실패를 일관되게 더 잘 고친다는 증거는 없다".

📌 [Figure 3 삽입] — 4개 장면 정성 비교: (a) RL-induced degradation(ID 5449, SFT 1.32 / GRPO 1.85 / FIRE 1.28 m), (b) persistent failure(ID 3491, 1.73 / 1.56 / 1.25), (c) stochastic instability(ID 5643, 1.44 / 1.70 / 1.08), (d) **반례**(ID 4798, 1.04 / **0.22** / 1.18 — GRPO가 더 좋음)

---

## 5. Ablation 및 분포 분석

### 5.1 분포 분석 (Appendix A, E) — 이 논문의 백미
policy당 24,076개 후보에 대해 (GRPO / FIRE-VLA):

| 통계량 | GRPO | FIRE-VLA | 우세 |
|---|---|---|---|
| median | 0.550 | 0.598 | GRPO |
| P90 | 1.498 | 1.594 | GRPO |
| P95 | 1.890 | 2.013 | GRPO |
| P99 | 2.926 | 2.881 | 무승부 |
| CVaR95 | 25.52 | **17.72** | FIRE-VLA |
| CVaR99 | 118.58 | **79.21** | FIRE-VLA |
| Worst-of-4 | 5.57 | **4.13** | FIRE-VLA |
| Intra-sample std | 2.19 | **1.56** | FIRE-VLA |
| Any>10m | 1.35% | **0.83%** | FIRE-VLA |

- **Winsorized 평균 차이**(FIRE − GRPO)는 cap 3/5/10/20 m에서 각각 **+0.0464, +0.0440, +0.0372, +0.0233 m** (즉 cap을 씌우면 GRPO가 낫다), uncapped는 −0.3476 m.
- 결론: FIRE-VLA의 평균 L2 개선은 **일상 구간의 균일한 향상이 아니라 희귀한 severe rollout 억제**에서 거의 전부 나온다. 저자 스스로 title-level claim을 이 수준으로 낮춰 적는다.

### 5.2 Reward–L2 역설 (Appendix C)
- reward가 bounded reciprocal이라 physical error가 이미 클 때 0에 가까워지며 **saturate**한다. severe → catastrophic 구간의 차이가 scalar reward에서 압축된다.
- 6,019개 sample-level G=4 집계에서 reward와 −Avg.L2의 **Spearman 상관은 0.9793(GRPO)/0.9807(FIRE)**로 매우 높지만, **Pearson은 0.0781/0.0879**로 극히 낮다. 이 saturation 때문에 GRPO가 평균 scalar reward는 높으면서 동시에 더 무거운 극단 오차 tail을 가질 수 있다.
- 저자는 극단 궤적에 대해 파싱, 좌표 변환, waypoint 순서, 단위, fallback 동작을 감사했고 이들이 대형 오차를 설명하지 않음을 확인했다고 보고한다.

### 5.3 없는 것
- **component-wise ablation이 없다**. failure routing / privileged distillation / teacher promotion 세 요소를 분리하지 않은 end-to-end 비교뿐이며, 이는 저자도 Limitations에서 인정한다.

---

## 6. 기술적 강점

1. **문제 정의의 정밀함**: "GRPO gradient가 사라진다"는 흔한 과장 대신 "corrective information이 부족하다"로 좁혀 주장한다. 게이트도 그 논리에 맞춰 reward 평균과 분산의 **결합 통계**로 설계했다.
2. **외부 teacher 불필요**: teacher는 같은 규모의 자기 복제본이며 우위는 오직 privileged context. 더 큰 모델이나 외부 correction 생성기가 필요 없고, 배포 시 privileged branch는 제거된다.
3. **On-policy prefix 위의 감독**: student가 실제 방문하는 상태에 감독을 얹어 off-policy mismatch를 줄인다(OPSD 계열). 게다가 **answer token만** 감독해 reasoning 스타일을 강제하지 않는다.
4. **평가 위생**: 학습용 라우팅 게이트와 평가용 persistent detector를 **완전히 분리**하고, detector 임계값을 겹치지 않는 SFT split에서 한 번만 보정해 동결했다. scene-clustered paired bootstrap, scene/image 중복 차단, 고정 sample list 공개까지 — 소규모 연구치고 프로토콜이 엄격하다.
5. **정성 예시의 정직성**: Figure 3에 **GRPO가 이기는 반례(d)**를 일부러 포함하고, 후보 선별 없이 G=4 전체 rollout을 그린다(Appendix D).
6. **자기 반박적 분석**: winsorized 분석과 percentile 표로 "우리 방법이 일상 구간에서는 더 나쁘다"를 스스로 드러낸다.

---

## 7. 한계 및 비판

1. **단일 시드, 방법당 1회 학습**. 보고된 CI는 평가 표본의 불확실성만 담고 **학습 시드 변동은 측정되지 않았다**. RL post-training에서 시드 분산은 종종 방법 간 차이보다 크다.
2. **Equal-compute 비교가 아님**. FIRE-VLA는 teacher forward가 추가되고 라운드 2에서 optimizer/scheduler를 재시작한다. 후자는 그 자체로 LR 재-warmup 효과를 주므로, "라운드 구조"의 이득이 privileged distillation 때문인지 **optimizer 재시작 때문인지 분리되지 않는다**.
3. **두 라운드가 disjoint prompt 부분집합을 쓴다**. 동일 prompt를 종단적으로 추적하지 않으므로, "실패 분포가 policy와 함께 변한다"는 핵심 서사가 로그로 직접 검증되지 않는다.
4. **라우팅 빈도가 매우 낮다**. 라운드당 56~58 group(224~232 응답). 600 prompt 중 ~9.5%. λ=0.1까지 곱하면 실제 PSD 기여는 작으며, 그럼에도 나타난 tail 효과가 정말 PSD 때문인지 아니면 **정규화/노이즈 효과**인지 구분하기 어렵다.
5. **핵심 지표에서 GRPO가 이긴다**. G=4 scalar reward는 GRPO가 유의하게 높고(CI [−0.0250, −0.0178]), median/P90/P95도 GRPO 우세. FIRE-VLA의 우위는 CVaR와 >10 m 사건이라는 **극단 영역에 국한**된다.
6. **양쪽 RL 모두 SFT보다 stochastic L2가 나쁘다**. "RL post-training이 open-loop L2를 악화시킨다"는 더 근본적인 문제를 이 논문은 완화할 뿐 해결하지 못한다.
7. **Open-loop, 전방 카메라 단일 뷰**. closed-loop 안전성, 교통법규 준수, 승차감, compounding state shift를 측정하지 않는다. nuScenes open-loop L2가 실제 주행 능력의 대리 지표로서 갖는 한계는 이미 커뮤니티에서 널리 지적된 바다.
8. **작은 스케일**. 3B 모델, 1,200 prompt, 150 update, 4×RTX 3090. 방법의 확장성은 미검증이며 K=2를 넘어선 단조 개선도 보장되지 않는다(저자 명시).
9. **단일 저자, 미공개 venue**. 재현 자산은 공개한다고 밝혔으나 동료 심사를 거치지 않은 프리프린트다.

---

## 8. 관련 연구와의 위치

- **AutoDrive-R²** (ICLR 2026): driving VLA에 reasoning supervision + GRPO. FIRE-VLA는 이 objective를 **모든 group에 대해 유지**하면서, group의 **정보량**에 따라 보조 감독을 얹는다는 점이 차별점.
- **ELF-VLA** (CVPR 2026): 반복되는 주행 실패를 식별해 feedback-guided correction을 만들고 검증 후 RL에 재주입. FIRE-VLA는 **외부 correction 생성 없이** 같은 policy의 privileged copy를 쓴다.
- **OPSD (Self-distilled Reasoner)**: 하나의 LM이 student이자 privileged teacher가 되어 student prefix 위에서 분포를 맞춘다. FIRE-VLA는 여기에 **unresolved-failure routing**과 **라운드별 teacher 승격**을 결합했다.
- **DriveVLA-M0**와의 대비: M0는 frozen base 위에 retrieval + test-time LoRA를 얹는 **추론 시 보정**, FIRE-VLA는 **학습 시 policy 가중치 자체를 갱신**. 두 방법은 직교적이며 결합 가능성이 있다.
- backbone은 Qwen2.5-VL-3B-Instruct이며, **새 VLA 아키텍처가 아니라 post-training 기법**을 제안하는 논문임을 저자가 명시한다.

---

## 9. 재현성

- 코드/설정/평가 프로토콜: https://github.com/forever-free1/FIRE-VLA
- 고정 sample list, 평가 config, per-sample provenance, 스크립트 공개 예정이라고 명시.
- 하드웨어 요구가 낮다(4×RTX 3090). 3B full-parameter BF16 FSDP이므로 학계 규모에서 재현 가능한 편.
- 다만 detector 임계값(0.4457839238 / 0.0747977498)처럼 소수점 10자리로 하드코딩된 값들은 calibration split이 함께 공개되어야 의미가 있다.

---

## 10. 세미나 토론 주제

1. λ=0.1에 라우팅 비율 ~9.5%를 곱하면 PSD의 실효 기여는 매우 작다. 그럼에도 CVaR99가 118.58 → 79.21로 바뀐 것은 **인과인가 분산인가**? 시드 3개만 돌려도 판별될 문제인데 왜 하지 않았을까?
2. Round 2의 optimizer/scheduler 재시작은 privileged distillation과 **혼입(confound)**되어 있다. "GRPO 150 update를 75+75로 쪼개고 재시작만 한" 대조군이 반드시 필요하지 않은가?
3. Reward가 Spearman 0.98 / Pearson 0.08이라는 사실은, **이 reward로 RL을 하는 것 자체**가 극단 오차에 무관심하다는 뜻이다. FIRE-VLA 같은 보조 감독 대신 **reward 재설계**(예: 로그 스케일, tail-aware)가 더 직접적인 해법 아닌가?
4. teacher가 `<priv>τ*</priv>`를 문맥에 넣고 student prefix를 이어받는 구조는, 사실상 **정답을 본 채로 다음 token을 예측**하는 것에 가깝다. 그렇다면 이 PSD는 privileged distillation이라기보다 **soft-label teacher forcing**의 변형 아닌가? 두 관점의 실질적 차이는?
5. 두 RL 방법 모두 SFT보다 G=4 L2가 나쁘다. open-loop L2에서 RL이 손해라면, 이 벤치마크 위에서 RL post-training을 평가하는 것 자체가 타당한가?
6. Unresolved failure를 "낮은 평균 + 낮은 분산"으로 정의했는데, **모두 잘하는** 그룹도 낮은 분산을 갖는다. valid fraction과 30 percentile 조건으로 충분히 구분되는가? 게이트가 실제로 무엇을 잡는지 정성 분석이 없다.
7. K=2에서 멈춘 이유가 계산 자원인가, 아니면 3라운드에서 개선이 사라졌기 때문인가? 저자는 "monotonic improvement를 확립하지 않는다"고만 적는다.

---

## 11. 실무적 시사점

- **적용 조건**: GRPO 계열 RL post-training을 이미 돌리고 있고, ground-truth future(또는 임의의 privileged signal)를 학습 시점에 보유한 경우에 한해 얹을 수 있는 **저비용 add-on**이다. 구현은 (i) group 통계 기반 게이트, (ii) teacher forward 1회 추가, (iii) answer-span mask + top-k JSD 세 조각뿐.
- **기대 효과의 성격**: 평균 성능 향상이 아니라 **catastrophic rollout 억제**로 이해해야 한다. 안전이 tail에 의해 결정되는 자율주행에서는 이 방향이 실무적으로 더 의미 있을 수 있다.
- **주의**: 배포 시 privileged branch를 반드시 제거해야 하며, student 입력에 privileged token이 새지 않는지 검증 로직이 필요하다(논문도 이를 명시적으로 구현).
- **일반화 가능성**: 자율주행에 국한되지 않는다. "정답 궤적/정답 답안을 teacher context에만 넣고 student prefix 위에서 answer token만 distill"하는 패턴은 manipulation VLA나 일반 reasoning RL에도 그대로 이식 가능하다.

---

## 12. 총평

**기여도: 중.** 새로운 아키텍처도 새로운 SOTA도 아니고, 3B 모델·4×3090·단일 저자의 소규모 연구다. 핵심 결과인 single-sample Avg. L2 개선은 CI가 0을 가로지르고, G=4 scalar reward는 오히려 baseline이 이긴다. 라우팅 빈도가 라운드당 60 group 미만이라 관측된 tail 개선이 정말 제안 기법에서 왔는지 단일 시드로는 확신하기 어렵다.

**그럼에도 읽을 가치가 있는 이유는 방법이 아니라 태도다.** 저자는 (a) "GRPO gradient가 사라진다"는 손쉬운 서사를 스스로 기각하고, (b) 학습 게이트와 평가 detector를 분리해 후자를 사전 동결했으며, (c) winsorized 분석으로 "우리는 일상 구간에서 더 나쁘다"를 자진 공개하고, (d) 정성 그림에 반례를 넣고, (e) equal-compute가 아님과 단일 시드 한계를 결론에까지 반복해 적는다. Spearman 0.98 / Pearson 0.08이라는 reward saturation 진단은 이 논문에서 가장 재사용 가치가 높은 통찰이며, GRPO로 궤적 회귀를 학습하는 모든 연구가 점검해야 할 지점이다.

**한 줄 평가**: 주장은 작고 증거는 좁지만, 그 좁음을 정확히 아는 논문. 성능표보다 Appendix A/C/E를 먼저 읽을 것.

**추천 독자**: 자율주행 VLA의 RL post-training 연구자, GRPO의 저분산 group 문제를 다루는 사람, privileged/on-policy distillation 설계자, 그리고 "평균 지표가 좋아졌다"는 주장을 어떻게 정직하게 해부하는지 배우고 싶은 사람.

<!-- VERIFIED: pdf -->
