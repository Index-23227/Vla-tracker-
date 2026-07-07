# Guided Action Flow: Q-Guided Inference for Flow-Matching Vision-Language-Action Policies

> **한 줄 요약**: Frozen SmolVLA의 reverse-time flow 샘플러에 QGF 스타일 **value-gradient 속도 보정**을 주입하는 inference-time wrapper. 실제 LIBERO 롤아웃의 sparse success-to-go로 학습한 MLP critic 앙상블(K=3)을 클린 액션 추정치 â = x_t − t·v_t에서 미분해 v_guided = v_t − m·clip(g,c)/β로 샘플링 경로를 조정한다. 단일 태스크 68.0→82.0% / 82.0→86.0%, 멀티패밀리 validation 46.0→56.0%를 달성했으나 locked held-out은 65.0→67.5%로 소폭 — 저자 스스로 "feasibility study"로 규정하고 critic generalization을 핵심 병목으로 지목한 정직한 early-stage 논문.

---

## 1. 배경 및 동기

### 문제 설정
- Pretrained VLA(SmolVLA)는 많은 LIBERO 태스크를 수행하지만, 분포 이동·compounding error·다중 plausible chunk 상황에서 실패
- 표준 해법인 full fine-tuning은 비용·하드웨어·검증 문제 — 특히 consumer-GPU 환경(SmolVLA 채택 이유)에서 부담
- **모듈러 대안**: base policy는 동결하고, frozen policy의 롤아웃으로 작은 critic만 학습해 **추론 시에만** 사용

### 기존 접근과의 차별점
- Action reranker가 아님: 여러 chunk를 샘플해 고르는 게 아니라 **연속 flow trajectory 자체를 변형** — 최종 액션이 나오기 전 중간 샘플에 개입 가능
- Fine-tuning이 아님: critic gradient는 action chunk에 대해서만 사용, VLA로 backprop 안 함
- 개념적 선행 연구는 QGF (Zhou et al., arXiv 2606.11087, RL에서 flow policy의 test-time value-gradient guidance). 본 논문의 질문은 더 좁고 실증적: **QGF식 업데이트가 frozen flow-matching VLA(SmolVLA + LIBERO)에서 실제로 유용한가?**

### 표준 offline RL과 다른 점
1. 액션 생성기가 from-scratch policy가 아니라 자체 language-conditioned flow 샘플러를 가진 pretrained VLA
2. 적응 목표가 최소주의적: base policy 보존, inference trajectory만 변경

📌 [Figure 1 삽입] — frozen policy 롤아웃 → 성공/실패 에피소드 → critic 앙상블 학습 → 테스트 시 critic gradient + uncertainty gating + velocity 조정으로 closed-loop LIBERO 평가.

---

## 2. 방법론 심층 분석

### 2.1 문제 정식화
Frozen VLA π_θ가 (시각 관측, proprioception, 언어 지시) → action chunk a_{0:H−1} ∈ R^{H×d}. Critic:
$$Q_\phi(f_o, a_{0:H-1}, e_\tau) \to \mathbb{R}$$
- f_o: policy-side 관측/상태 feature (privileged simulator state 배제 — 배포 시 policy가 접근 가능한 feature만)
- e_τ: 선택적 task feature
- 타깃: 실제 LIBERO 롤아웃의 sparse success-to-go. **추론 시 gradient로만 사용**

### 2.2 SmolVLA reverse-time 컨벤션 — 부호가 핵심
Pinned SmolVLA 샘플러는 reverse-time flow:
$$x_t = t\epsilon + (1-t)a, \quad v_t = \epsilon - a$$
t=1(노이즈) → t=0(액션)으로 적분. 클린 액션 추정:
$$\hat a(x_t, v_t, t) = x_t - t v_t$$
**forward-time 샘플러에서 복사한 guidance 부호는 클린 액션 추정을 반대 방향으로 밀어버린다.** 실제 구현 컨벤션에서 부호를 유도해야 함 — 저자가 contribution으로 명시한 실무적 교훈.

또한 SmolVLA는 내부적으로 action chunk를 더 큰 최대 차원으로 패딩 → critic은 LIBERO의 물리적 액션 차원에만 적용, 패딩 velocity는 불변 (critic이 임의의 패딩 좌표를 학습/유도하는 것 방지).

### 2.3 Q-Guided Flow 업데이트
각 denoising step에서 denoiser 출력을 detach하고 â에서 critic 평가. K개 앙상블 평균:
$$\bar Q(\hat a) = \tfrac{1}{K}\sum_k Q_{\phi_k}(f_o, \hat a, e_\tau), \quad g = \nabla_{\hat a}\bar Q(\hat a)$$
norm c로 클리핑 후 disagreement gate:
$$m = \max(m_{\min}, \exp(-\alpha \sigma_Q))$$
(σ_Q: critic 값들의 per-sample 표준편차). 최종:
$$v_t^{guided} = v_t - \frac{m \cdot \text{clip}(g, c)}{\beta}$$
빼기(−)는 reverse-time 컨벤션의 귀결: v_t를 g 방향으로 줄이면 클린 액션 추정이 critic 값을 올리는 방향으로 이동.

### 2.4 하이퍼파라미터 트레이드오프
- 작은 β → 강한 guidance → base policy를 압도해 regression 유발 가능
- clip은 불안정한 critic gradient 제한, gate는 앙상블 불일치 시 guidance 축소, m_min은 불확실성이 높아도 최소한의 guidance 유지
- Best validation config: **β=2, clip norm 1.0, uncertainty scale α=10, m_min=0.1, K=3, hidden 768, depth 4, 30 epochs, VLM hidden-state task feature**

> ❓ **예상 질문**: 왜 reranking(BoN) 대신 gradient guidance인가?
> **답변**: reranking은 완성된 chunk 집합에서 고르기만 하므로 base policy가 좋은 후보를 아예 못 내면 무력. Guidance는 중간 샘플 단계에서 개입해 후보 분포 자체를 이동시킴 — diffusion/flow guidance 구조에 더 가깝다는 게 저자 논거.

---

## 3. 데이터 전략

### 롤아웃 데이터셋 구성
- Frozen SmolVLA 정책의 실제 LIBERO 롤아웃에서 horizon H의 **overlapping chunk** 추출: chunk 시작점 i의 액션 시퀀스 a_{i:i+H−1}이 critic 입력, 관측 feature는 chunk 시작 시점의 policy-preprocessed state
- 타깃은 sparse success-to-go:
$$y_i = \gamma^{j^*-i} \;(j^* = \min\{j \ge i : s_j = 1\}), \quad y_i = 0 \text{ (미래 성공 없음)}$$
첫 성공 이후 타깃은 1. 단순·저렴하지만 **near-success chunk들의 순위를 직접 매기지 못함** — critic 품질 병목의 한 원인으로 저자가 자인

### 분할 원칙
- **에피소드 단위** train/validation 분할. 랜덤 chunk 분할은 같은 롤아웃의 인접 chunk가 고도로 상관되어 있어 trajectory context 누수 → critic 품질 과대평가

### 멀티패밀리 학습 데이터 (Table V)
| 항목 | 값 |
|------|-----|
| 학습 패밀리 | LIBERO spatial + object |
| 학습 태스크 | spatial 0–4, object 0–4 |
| 학습 에피소드 | 500 (성공 332/500, 66.4%) |
| object 서브셋 성공 | 162/250 (64.8%) |

---

## 4. 시스템/학습 세부사항

- **Base policy**: 공식 `lerobot/smolvla_libero` 체크포인트, LeRobot 평가 스택 + 실제 LIBERO 환경 롤아웃 (dummy action 아님), 전 실험에서 동결
- **Critic**: flattened (관측 feature, action chunk, task feature) 위의 MLP, MSE regression, K=3 앙상블
- **Task conditioning 3안 비교**:
  1. Task-id — 같은 태스크 튜닝엔 유용하나 LIBERO 패밀리 간 이식 불가
  2. Hashed bag-of-token — 가볍고 이식 가능하지만 cross-task gain을 내기엔 너무 약함 (초기 실험)
  3. **Frozen SmolVLA VLM hidden-state** (non-padding task token에 mean-pooling) — 최강. 새 학습 가능한 텍스트 인코더 없이 base policy 자체의 언어 표현에 critic을 결속
- **런타임**: denoiser는 평소처럼 평가하되 critic 경로만 â에 대해 autograd 임시 활성화; denoiser 출력은 critic 미분 전 detach → gradient가 VLA로 역전파되지 않음

---

## 5. 실험 설계 및 평가 프로토콜

### 세 가지 실험 질문
1. Base policy가 이미 nontrivial한 성공률을 가진 태스크에서 critic gradient가 개선을 주는가? (feasibility)
2. 제한된 task family로 학습한 critic이 전이되는가, 아니면 unseen 태스크에 해로운 gradient를 주는가?
3. Task-description conditioning + 멀티패밀리 데이터가 validation을 개선하고, 그 개선이 **locked held-out test**에서 살아남는가?

### 프로토콜 (Table I)
| 프로토콜 | 태스크 | 예산 |
|----------|--------|------|
| Single-task | LIBERO spatial 1개 | 50 episodes/seed |
| Spatial-only transfer | spatial validation tasks | 60 episodes |
| Multi-family validation | spatial [5,7,8], object [6,7] | 50 episodes |
| Multi-family held-out | spatial [6,9], object [8,9] | 40 episodes |

### 평가 규율 (이 논문의 미덕)
- 하이퍼파라미터는 **validation에서만** 선택, held-out은 단 1회 평가로 잠금
- "held-out에서 β나 gate를 재조정하면 test set이 또 하나의 validation set이 되어 주장이 부풀려진다"고 명시

### Frozen SmolVLA 베이스라인 앵커 (Table II, QGF 미적용)
| 세팅 | 예산 | 성공률 |
|------|------|--------|
| LIBERO vanilla | 100 ep | 65/100 (65.0%) |
| LIBERO-Plus spatial subset | 50 ep | 39/50 (78.0%) |
| LIBERO-PRO zero-shot | 100 ep | **1/100 (1.0%)** |

LIBERO-PRO 앵커가 프레이밍상 중요: near-zero면 base policy가 "작은 국소 실수"를 하는 게 아니므로 QGF로 복구 불가 — 현재 결과는 vanilla LIBERO에서만 해석해야 함.

---

## 6. 실험 결과 심층 분석

### 메인 요약 (Table III)
| 세팅 | 평가 분할 | Baseline | QGF | Gain |
|------|-----------|----------|-----|------|
| Single-task QGF | seed 3000 | 34/50 (68.0%) | 41/50 (82.0%) | **+14.0 pp** |
| Single-task QGF | seed 4000 | 41/50 (82.0%) | 43/50 (86.0%) | +4.0 pp |
| Spatial-only transfer | validation | 32/60 (53.3%) | ≤31/60 (≤51.7%) | **≤ −1.7 pp (음성)** |
| Multi-family task-desc critic | validation | 23/50 (46.0%) | 28/50 (56.0%) | **+10.0 pp** |
| Multi-family task-desc critic | held-out test | 26/40 (65.0%) | 27/40 (67.5%) | +2.5 pp |

### 해석
1. **Guidance는 실제로 작동**: 단일 태스크 +14.0pp, 멀티패밀리 validation +10.0pp — 실제 closed-loop 롤아웃(액션 재구성 지표 아님)에서 확인
2. **Task coverage가 결정적**: spatial-only critic은 전이 실패(음성 결과). 같은 패밀리 태스크를 늘리는 것만으로 부족 — task-description feature + 멀티패밀리 데이터 + disagreement gate로 이행한 설계 근거
3. **Validation은 held-out을 과대평가**: +10.0pp vs +2.5pp (40 에피소드 중 성공 1개 추가). 이 괴리가 논문의 중심 empirical takeaway
4. 단일 태스크 결과도 β, critic 체크포인트 선택, gating에 민감 — 동일 메커니즘이 에피소드별로 gain과 regression을 모두 유발

### 가장 보수적인 해석 (저자 표현)
"Guided Action Flow가 해결된 적응 방법이라는 게 아니라, frozen flow-matching VLA가 critic-guided inference에 유용한 **인터페이스**를 노출하며, critic의 품질·calibration·task coverage가 이 인터페이스의 득실을 결정한다."

---

## 7. Ablation 분석

정식 ablation table은 없으나 논문 전체가 사실상 설계 선택의 단계적 검증:

| 설계 축 | 비교 | 결론 |
|---------|------|------|
| Task conditioning | task-id vs hashed BoT vs VLM hidden-state | task-id는 패밀리 간 이식 불가, BoT는 신호 약함, VLM hidden-state가 최강 |
| Critic 데이터 범위 | spatial-only vs spatial+object 멀티패밀리 | spatial-only는 ≤−1.7pp 음성; 멀티패밀리는 validation +10pp |
| Guidance 강도 | β 스윕 (초기) | 작은 β는 base policy를 압도해 regression; β=2 채택 |
| 불확실성 처리 | plain gradient vs 앙상블 disagreement gate | plain gradient는 critic 외삽 영역에서 과격; gate는 1차 휴리스틱으로 완화 |
| 분할 방식 | chunk-level vs episode-level | chunk-level은 누수로 critic 품질 과대평가 |
| Guidance 부호 | forward-time 복사 vs reverse-time 유도 | 부호 오류 시 클린 액션 추정이 역방향 이동 — 실측 샘플러에서 유도 필수 |

빠진 ablation: K(앙상블 크기), m_min, α의 개별 기여, gate on/off 정량 비교, H(horizon) 민감도 — 모두 미보고.

---

## 8. 관련 연구 비교

| 계열 | 대표 | Guided Action Flow와의 관계 |
|------|------|------------------------------|
| Q-guided flow (RL) | QGF (Zhou et al. 2606.11087) | 직계 선행. GAF는 이를 frozen **VLA**(pretrained SmolVLA + 언어조건 샘플러 + 실롤아웃 critic)로 이식한 empirical study |
| Inference-time Q-steering (VLA) | QPILOTS (2606.14801) | 유사 문제의식(frozen flow VLA + critic gradient). QPILOTS는 π0.5 3B 기반 + Tweedie/MFM posterior + 이론 보장 + SARSA online critic; GAF는 SmolVLA 기반 + 단순 MSE success-to-go critic + disagreement gate로 더 소박한 스케일. 상호 인용 없음(동시기) |
| Activation steering (VLA) | CTRL-STEER (2606.00269) | 같은 "frozen VLA + inference-time 개입" 패턴이지만 개입 지점이 다름 (FFN 뉴런 활성 vs flow velocity) |
| Diffusion/flow guidance | classifier(-free) guidance, Diffuser, Diffusion-QL, IDQL | 생성모델 guidance 관점의 원류 |
| Offline RL 보수성 | CQL, IQL, MOPO, EDAC | OOD value 문제의식 공유; GAF의 disagreement gate는 Q-ensemble 불확실성 계열의 경량 변형 |
| Residual RL | Johannink et al. 2018 | 개념적 유사 — 단 residual이 실행 토크가 아니라 **frozen 샘플러 내부의 value-gradient 보정** |
| Base/벤치마크 | SmolVLA, LeRobot, LIBERO/-Plus/-PRO | 평가 인프라 |

포지셔닝: 새 backbone도 tokenizer도 아닌, frozen flow-matching VLA 둘레의 **critic layer**. π0, RDT류 flow VLA에 개념상 이식 가능하나 본 논문은 SmolVLA만 검증.

---

## 9. 한계 및 미해결 문제

1. **평가 예산이 작음**: 최강 결과인 held-out이 40 에피소드, +2.5pp(성공 1개 차이). 신뢰구간·다중 seed 없음 — 저자도 "feasibility evidence"로 한정
2. **Critic이 기술적 병목**: compact policy-side feature + sparse success-to-go regression. near-success chunk 랭킹 신호 부족 → pairwise ranking/contrastive objective 필요성 자인
3. **파라미터 민감성**: β, clip, K, α, m_min이 유용한 보정 vs 유해한 이탈의 균형을 좌우. validation-only 선택으로 완화했을 뿐 민감성 자체는 잔존
4. **LIBERO-Plus/PRO 미검증**: PRO zero-shot 1%라 QGF 평가 자체가 무의미한 수준 — 더 나은 데이터 커버리지나 base 체크포인트 선행 필요
5. **실로봇 미평가**: 센서 노이즈·지연·캘리브레이션 하에서 action-chunk guidance 효과 미지수
6. Disagreement gate는 완전한 불확실성 추정이 아닌 1차 휴리스틱 — task embedding distance, action-feature k-NN, 학습형 OOD detector로 대체/보강 여지

---

## 10. 총평

### 강점
- **부호·컨벤션·분할·평가 규율 같은 실무 디테일을 명시적 기여로 다룬 정직한 논문**. reverse-time 부호 유도, episode-level split, validation-only 튜닝 + locked held-out은 이 하위분야의 좋은 위생 기준
- 음성 결과(spatial-only critic 전이 실패)를 본문 테이블에 포함 — validation +10pp가 held-out +2.5pp로 줄어드는 것까지 그대로 보고
- Consumer-GPU 지향(SmolVLA)이라는 현실적 세팅; critic만 갈아끼우는 모듈러 적응 경로 제시

### 약점
- 통계적 설득력 부족: 40~60 에피소드, seed 2개(단일 태스크), CI 없음
- QPILOTS 등 동시기 Q-steering 연구 대비 이론·스케일·베이스라인 비교(BoN, DSRL류 부재)가 모두 얇음
- "Guided Action Flow"가 개선하는 절대 성능(56%, 67.5%)이 낮아 SOTA 관점의 의미는 없음 — 가치는 전적으로 방법론적 feasibility에 있음

### 임팩트 전망
UCL 그룹의 early-stage 보고서 성격. 단독 임팩트보다는 **"frozen flow VLA + 경량 critic guidance"라는 레시피의 재현 가능한 최소 사례**로서 가치. critic objective(ranking loss), OOD-aware gating, 더 큰 평가 예산이 갖춰지면 QPILOTS류와의 정면 비교가 다음 단계.

**Bottom line**: SmolVLA급 소형 flow VLA에서도 Q-guided inference가 실 closed-loop gain을 낼 수 있음을 보인 소규모 feasibility study. 결과의 크기가 아니라 평가 규율과 병목 진단(critic generalization)의 명료함이 이 논문의 기여다.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | held-out +2.5pp는 40 에피소드에서 성공 1개 차이인데 노이즈와 구별 가능한가? | 구별 불가에 가깝고 저자도 "positive but modest"로만 서술. 논문의 주장은 held-out gain이 아니라 (a) 단일 태스크 +14pp, (b) validation→held-out 괴리 자체가 critic generalization 병목의 증거라는 것. |
| 2 | QPILOTS(2606.14801)와 뭐가 다른가? | 베이스(π0.5 3B vs SmolVLA), critic(SARSA online vs MSE success-to-go offline-rollout), 추정점(Tweedie/MFM posterior vs 단순 â = x_t − t·v_t), 이론(W2/KL 보장 vs 없음), 게이팅(없음 vs 앙상블 disagreement gate). GAF는 더 소박하지만 uncertainty gating이 독자 요소. 상호 인용 없음. |
| 3 | Sparse success-to-go가 병목이라면 왜 처음부터 ranking loss를 안 썼나? | 저자 자인: 단순·저렴해서 채택, near-success chunk 랭킹 신호 부족이 확인된 건 사후. pairwise ranking/contrastive가 future work 1순위. |
| 4 | 부호 유도가 contribution이라기엔 사소하지 않나? | 구현 실무에선 치명적: forward-time 관례를 복사하면 guidance가 정확히 역효과. pinned 구현의 x_t = tε+(1−t)a 관례를 확인하고 v_guided = v_t − m·clip(g,c)/β를 유도한 건 재현성 관점의 실질 기여. |
| 5 | Spatial-only critic 실패가 "no-op"이 아니라는 근거는? | 진단에서 task-conditioned guidance가 개별 에피소드 결과를 실제로 바꿈. 문제는 gain과 regression이 상쇄되거나 음(≤31/60 vs 32/60)으로 기움 — guidance가 안 걸린 게 아니라 critic이 unseen 태스크에서 유해한 gradient를 냄. |
| 6 | Gate가 m_min=0.1로 항상 guidance를 남기는 이유는? | 불확실성이 높다고 guidance를 완전히 끄면 개선 기회도 사라짐. m_min은 탐색적 최소 보정 유지장치. 다만 OOD가 심한 영역에선 0으로 꺼야 한다는 게 저자의 deployment 논의 — gate 설계는 미완. |
| 7 | 패딩 차원 처리 왜 중요한가? | SmolVLA는 chunk를 최대 액션 차원으로 내부 패딩. critic이 패딩 좌표에 gradient를 내면 물리적으로 무의미한 방향으로 velocity가 왜곡됨. 물리 차원만 guidance하고 패딩 velocity는 불변으로 두는 게 안전장치. |
| 8 | LIBERO-PRO 1%인데 이 방법으로 올릴 수 있나? | 불가능하다는 게 저자 입장. QGF는 base policy가 "쓸만한 후보를 내지만 신뢰도가 부족한" 영역용. near-zero면 국소 보정 대상 자체가 없음 — 더 나은 critic 데이터/커버리지/베이스 체크포인트 선행 필요. |
| 9 | 왜 BoN/reranking 베이스라인과 정량 비교가 없나? | 논문의 약점. 방법론 절에서 "reranker가 아니다"라고 구별만 하고 실험 비교는 없음. QPILOTS는 BoN(N=5/20)과 직접 비교해 우위를 보였다는 점에서 대조적. |
| 10 | 추론 오버헤드는? | 미보고. 매 denoising step마다 K=3 critic forward-backward가 추가되나 critic이 MLP(768×4)라 작을 것으로 추정될 뿐 latency 수치 없음 — 배포 주장에는 공백. |

---

## 12. 시사점 및 후속 연구 방향

- **재현 체크리스트**: (1) 베이스 샘플러의 시간 방향·부호 검증, (2) episode-level split, (3) validation-only 튜닝 + held-out 잠금 — 이 세 가지는 QGF류 VLA 연구의 최소 위생 기준으로 채택할 만함
- **Critic 개선 경로**: pairwise ranking/contrastive objective, 더 풍부한 시각-상태 표현, conservative value regularization, calibrated uncertainty loss
- **OOD-aware guidance**: 앙상블 disagreement를 task embedding distance, action-feature 최근접 거리, 학습형 OOD detector로 보강; 분포 밖에선 guidance 축소/차단
- **확장 대상**: 더 많은 LIBERO 패밀리 + 더 큰 locked 예산, LIBERO-Plus/PRO 본평가, 실로봇 (노이즈·지연 하 chunk guidance 효과)
- **커뮤니티 관점**: QPILOTS(π0.5), GAF(SmolVLA), CTRL-STEER(OpenVLA activation)로 "frozen VLA + inference-time 개입" 계열이 빠르게 형성 중. 표준 벤치마크·공통 베이스라인(BoN, DSRL)·latency 보고를 갖춘 정면 비교가 다음 마일스톤

<!-- VERIFIED: pdf -->
