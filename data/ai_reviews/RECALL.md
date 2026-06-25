# RECALL: Recovery Experience Collection for Active Lifelong Learning in Vision-Language-Action Models

> **한 줄 요약**: π0-FAST autoregressive VLA를 대상으로, INSIGHT의 token-level uncertainty 신호로 high-uncertainty state를 식별하여 recovery demonstration을 능동적으로 수집하고(active), 이를 replay/EWC로 통합하는 active continual learning 파이프라인 제안. LIBERO-10에서 baseline 59.8% → online active + full replay 72.4% (p=0.0001)로 유의미한 향상을 보였으며, new-only fine-tuning이 catastrophic forgetting을 유발함을 실증하고, replay coverage가 stability-plasticity tradeoff의 핵심임을 밝힘.

---

## 1. 배경 및 동기 (Section 1)

- RT-1, RT-2, Gemini Robotics, π0-FAST, π0.5 같은 대규모 VLA는 일반화 능력에도 불구하고 새로운 robot/env/task 분포로 배포될 때 passive imitation learning 기반 fine-tuning이 필요하다.
- 기존 절차: (i) initial state에서 demonstration 추가 수집 → (ii) fine-tune → (iii) eval → (iv) 반복. 이 방식의 문제점:
  1. 실패가 발생해야 데이터 수집이 trigger됨 (시간/안전 비용 큼).
  2. 어디서 supervision이 필요한지 가이드가 없음.
  3. 이미 잘 수행되는 redundant 구간에 demonstrator effort 낭비.
- 저자들의 가설: **high-uncertainty state에서 수집된 demonstration이 정보량이 더 크고, 따라서 fine-tuning이 더 효율적**.

---

## 2. 연구 질문 (Section 3)

- **RQ-1**: passive 대비 high-uncertainty state에서 수집한 demonstration이 더 나은 policy 성능을 내는가?
- **RQ-2**: online(첫 uncertain state 한 번) vs offline(모든 uncertain state) 수집 중 어느 쪽이 충분한가?
- **RQ-3**: autoregressive VLA를 새로 수집한 recovery data만으로 fine-tune 가능한가?
- **RQ-4**: replay/regularization이 catastrophic forgetting을 완화하면서 적응을 유지할 수 있는가?

---

## 3. 방법론 (Section 3.1, 3.2)

### Active Learning Pipeline
- **Base policy**: π0-FAST를 LIBERO에서 저자 recipe로 30k steps 학습한 final checkpoint를 πθ0로 사용. π0-FAST는 continuous action chunk를 tokenize하여 next-token prediction loss로 학습.
- **Uncertainty estimator**: INSIGHT[6]을 step마다 적용. INSIGHT는 token-level entropy, log-probability, Dirichlet 기반 aleatoric/epistemic uncertainty feature를 사용해 "help가 필요한 시점"을 예측.
  - **Strong INSIGHT**: 실제 robot 실행에서 step-wise help label로 학습 (generalization 강함).
  - **Weak INSIGHT**: LIBERO episode-level success/failure label로 학습.
- **Candidate set**: 각 rollout τi에서 T_i = {t ∈ τi | H(o_t, s_t, ℓ, πθ0) = 1}.
- **Recovery dataset 종류**:
  - **Online**: D_online = {첫 high-uncertainty state t_online = min{t : h_t = 1}} 한 곳에서 한 demonstration.
  - **Offline**: D_offline = T_i 전체에서 demonstration.
  - **Passive baseline**: task initial state에서 동일 개수의 demonstration 수집 (수집 위치만 다름, distribution/quantity 통제).
- 시뮬레이터를 해당 state로 reset해 expert recovery trajectory를 task 완료까지 기록. Fine-tuning loss: standard autoregressive cross-entropy L_CE(θ) = −Σ log p_θ(x_t | x_<t).

### Continual Learning Strategies
- **Replay 데이터 mixture**:
  - new-only (D_new), full replay (D_old ∪ D_new), LIBERO-10 replay (D_LIBERO10 ∪ D_new), targeted replay (D_collected_tasks ∪ D_new).
- **EWC**: L_EWC(θ) = L_CE(θ) + λ·Σ_j F_j (θ_j − θ_0,j)^2. 다양한 λ 값 sweep + Fisher reference dataset에서 collected tasks 제외하는 filtered-EWC variant.
- **Learning rate ablation**: 표준 schedule (warmup → α=2.5e-5 → cosine decay → 2.5e-6) vs 매우 낮은 constant LR α=2.5e-8.

---

## 4. 실험 설정 (Section 4)

- Benchmark: **LIBERO-10** (10개 long-horizon manipulation task).
- 각 policy당 task별 50 rollout × 10 task = checkpoint당 500 rollout.
- Metric: overall success (10개), collected-task success (recovery 수집된 5개), retained-task success (나머지 5개).
- Statistical test: two-sided two-proportion z-test, 유의수준 *p<0.05, **p<0.01, ***p<0.001.
- 95% Wilson CI band (line plot).
- Normalization: replay 실험은 new normalization stats 사용. new-only 실험은 old normalization을 regularizer로 사용.

---

## 5. 실험 1: Active vs Passive 수집 (Section 5, Figure 2)

- 동일 task 분포/데이터 양 매칭, full replay + 표준 LR + CE.
- **Strong INSIGHT online recovery: 72.4% overall** vs **matched passive: 60.2%** vs **baseline: 59.8%**, p=0.0001로 유의.
- Weak INSIGHT도 동일 정성적 경향.
- 결론: 단순히 demonstration을 더 모은 효과가 아니라, **high-uncertainty state로부터의 demonstration 자체가 더 informative supervision**임을 입증.

---

## 6. 실험 2: Online vs Offline (Section 6, Figure 3)

- 두 dataset 모두 Strong INSIGHT 기반.
- Online: **72.4% overall, 49.2% collected**.
- Offline: **68.4% overall, 48.4% collected**.
- Overall 차이는 통계적으로 유의하지 않음 (p=0.1659).
- 해석: 첫 신뢰할 만한 high-uncertainty state가 rollout이 발산하기 시작하는 지점일 가능성. 한 번의 corrective demonstration이 그 state와 나머지 task 양쪽을 동시에 supervise. **Online이 더 demonstration-efficient**.

---

## 7. 실험 3: New-Only Fine-Tuning (Section 7, Figure 4)

- new-only training은 old normalization으로 제약을 줘도 **catastrophic forgetting** 발생.
- Strong INSIGHT online new-only best checkpoint: **overall 28.4%, collected 0.4%, retained 56.4%**.
- 대조: replay-based online: overall 72.4%, collected 49.2%, retained 95.6%.
- 결론: recovery data는 distributionally narrow하므로 단독 fine-tuning은 prior 능력을 보존하지 못함. **Active learning만으로 환원 불가; adaptation procedure가 prior competence를 보존해야 함**.

---

## 8. 실험 4: Continual-Learning Regularization (Section 8, Figure 5)

- new-only data 위에서 CE+standard LR vs CE+constant low LR vs EWC + 다양한 λ.
- Low LR: overall 62.8%, collected 34.4%, retained 91.2%.
- EWC (λ=10^12): overall 61.4%, collected 32.0%, retained 90.8%.
- 두 방법 모두 standard new-only보다는 forgetting을 줄이지만, **full replay(72.4%/49.2%)에 미달**.
- 결론: 약한 regularization은 adapt 가능하나 잊고, 강한 regularization은 retain 가능하나 학습 못 함. **Replay가 더 우수한 stability-plasticity tradeoff** 제공.

---

## 9. 실험 5: Replay Scope (Section 9, Figure 6, 7)

- Full replay vs LIBERO-10 replay vs targeted replay (Strong INSIGHT online + CE + 표준 LR).
- LIBERO-10 replay: **overall 68.6%** (full replay 72.4% 대비 p=0.1877, 유의하지 않음).
- Targeted replay: **overall 63.2%, retained 83.6%** (LIBERO-10 replay 90.0% 대비 p=0.0345로 유의하게 retention 감소).
- Filtered-EWC를 targeted replay에 적용해도 일관된 stabilization 실패.
- 결론: **Replay coverage가 결정적**. 보존하고자 하는 behavior를 포함해야 하며, 과도하게 collected tasks로 좁히면 retention이 무너짐.

---

## 10. 결론 및 핵심 메시지 (Section 10)

다섯 가지 발견:
1. High-uncertainty recovery demonstration이 matched passive보다 더 informative (Strong/Weak INSIGHT 모두).
2. Online이 offline보다 demonstration-efficient (overall 차이는 유의하지 않음).
3. Recovery data 단독 fine-tuning은 catastrophic forgetting 유발.
4. Low LR/EWC는 forgetting을 줄이지만 stability-plasticity tradeoff에 갇힘.
5. Replay-based mixing이 가장 강력한 해법이며, replay coverage가 핵심 (LIBERO-10 replay ≈ full replay >> targeted replay).

핵심 주장: **VLA의 active learning은 active continual learning으로 정식화되어야 한다** — robot이 도움 필요 시점을 감지하는 것뿐 아니라, 그 순간에 informative experience를 수집하고 prior skill을 지우지 않고 통합하는 메커니즘이 함께 필요.

---

## 11. 한계 (Section 11)

1. **시뮬레이션 only**: LIBERO-10에 한정. 실 robot에는 sensor noise, execution variability, safety constraint, 불완전한 reset 등이 추가됨. Recovery 수집이 simulator reset에 의존하는 점도 실배포에서는 runtime intervention/자연 도달 state로 대체 필요.
2. **Help predictor 품질 의존성**: INSIGHT의 false positive(demonstration 낭비)와 false negative(유용 state 누락). 더 calibrated, risk-sensitive uncertainty 추정 연구 필요.
3. **단일 VLA family, 단일 benchmark**: autoregressive π0-FAST + LIBERO-10만 평가. Diffusion/hybrid action head VLA는 다른 uncertainty signal·state selection 필요할 수 있음.
4. **단순한 continual learning 기법만 평가**: LoRA, adapter, selective freezing, gradient projection, distillation, 재가중 등 더 정교한 방법 미시도.
5. **Task success 중심 평가**: intervention cost, demonstration count, recovery length, action smoothness, safety violation, calibration, 반복 active cycle 등은 미평가.

---

## 12. 의의 및 기여

- **첫 active continual learning framework for autoregressive VLA**: INSIGHT의 token-level uncertainty를 recovery demonstration 수집 신호로 활용한 최초 사례.
- **Demonstration efficiency의 실증**: 첫 uncertain state 한 곳의 단일 recovery만으로도 dense offline 수집과 동등한 성능 — 실배포 비용 절감 시사.
- **Forgetting의 정량화**: VLA 규모 모델에서도 LLM 문헌처럼 catastrophic forgetting이 명백히 발생함을 control된 setup으로 보임 (28.4% vs 72.4%).
- **Replay coverage의 중요성**: stability-plasticity가 단순히 regularization 강도가 아니라 **replay 데이터의 분포 coverage**에 좌우됨을 sharp하게 분리.
- **Open research agenda**: VLA가 deployment 후 평생 학습하려면 (1) 더 나은 uncertainty estimator, (2) coverage-aware replay buffer, (3) targeted plasticity 메커니즘 (LoRA/adapter 등)이 필요함을 명확히 함.

---

<!-- VERIFIED: pdf -->
