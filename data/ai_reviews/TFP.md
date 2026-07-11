# TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning

> **한 줄 요약**: π0.5 같은 반응형(reactive) VLA가 stage-dependent manipulation에서 잠재 task progress를 추적하지 못한다는 문제를 지적하고, **Liquid Time-Constant(LTC) 동역학으로 유지되는 episode-local task-progress belief**를 물리적 경과 시간(Δt)에 보정된 retention gate `k_t = exp(−Δt/τ_t)`로 갱신한 뒤, 이를 **AdaLN 변조로 flow-matching action decoder에 직접 주입**하는 경량 memory-action 프레임워크를 제안. 3.3B 파라미터로 LIBERO 96.9→**98.75%**, LIBERO-plus 91.4→**93.77%**, 실제 Galaxea A1 object swap 15→**75%**, counting pick-place 40→**90%** 달성.

---

## 1. 배경 및 동기

### 반응형 VLA의 한계
- π0.5, OpenVLA, Octo 등 VLA 정책은 현재 관측·지시·proprioceptive state만으로 다음 행동을 예측하는 **reactive** 구조가 대부분.
- **Stage-dependent manipulation**에서는 시각적으로 유사한 상태라도 잠재적 task progress와 이전 상호작용 결과에 따라 다른 행동이 필요(예: object swapping에서 같은 장면이 "첫 물체를 버퍼로", "둘째 물체를 원위치로", "종료" 중 어느 단계인지 모호).

### 핵심 주장: 메모리만으로는 부족 — dynamics-aware belief update가 필요
- 유용한 belief는 **언제 바뀌어야 하는지**도 결정해야 함: 안정적 운반·가림(occlusion) 구간에서는 task-progress 정보를 보존하고, contact·release·subgoal 전환 근처에서는 새 증거를 빠르게 반영.
- Chunked receding-horizon 제어에서는 정책 질의 간격이 실행 진행·접촉·불안정성에 따라 변하는 **물리적 시간 변수**이므로, 메모리 갱신은 discrete step index가 아니라 경과 시간에 의존해야 함.
- 기존 memory-aware 정책은 (1) history buffer 검색형(HAMLET, MemoryVLA, CDP) — 풍부하지만 명시적 task-progress state가 없음, (2) recurrent latent state형(AVA-VLA, ReMem-VLA) — 상태는 있으나 갱신이 frame/chunk 인덱스에 묶임.

📌 [Figure 1 삽입] — memory-conditioned visuomotor control의 동기 개요

---

## 2. 방법론 심층 분석: 연속시간 belief 갱신 (LTC)

### 2.1 문제 설정
정책 질의 t에서 시각 관측 I_t, 로봇 상태 s_t, **직전 질의 이후 경과 시간 Δt_t**, 언어 지시 ℓ, 이전 belief h_{t−1} ∈ R^{d_h}(실험에서 256차원)를 받음.

### 2.2 LTC 갱신식
압축 관측 표현 x_t = [φ_vision(V_t); φ_state(s_t)]에 대해:
- 후보 belief: `ĥ_t = tanh(W_h[x_t; h_{t−1}] + b_h)`
- 입력 의존 시간상수(벡터): `τ_t = softplus(W_τ[x_t; h_{t−1}] + b_τ) + ε`
- retention gate: `k_t = exp(−Δt_t / τ_t)`
- 갱신: `h_t = k_t ⊙ h_{t−1} + (1 − k_t) ⊙ ĥ_t`

Write gain `g_t = 1 − k_t`로 쓰면 `h_t − h_{t−1} = g_t ⊙ (ĥ_t − h_{t−1})`: 후보 belief가 유지된 belief와 다르고 경과 시간 의존 write gain이 허용할 때만 메모리가 변함. 벡터 τ_t 덕분에 채널별로 다른 시간 스케일 — 느린 채널은 task context를 chunk를 넘어 보존, 빠른 채널은 새 지각 증거를 흡수.

### 2.3 이론적 정당화 (Appendix C)
- **Proposition 1**: LTC 갱신은 1차 연속시간 belief relaxation ODE `dh/du = −(h − ĥ_t)/τ`의 **정확한(exact) 이산화**.
- **Proposition 2**: `k(Δ₁+Δ₂) = k(Δ₁)k(Δ₂)`라는 elapsed-time consistency를 만족하는 유일한 연속 스칼라 retention은 지수 형태 — 지수 retention이 임의 설계가 아님을 증명.
- **Proposition 3**: 후보 belief가 노이즈일 때 높은 retention k는 Var(h_t) ≈ σ²(1−k)/(1+k)로 분산을 감소 — **낮은 update gain은 memory collapse가 아니라 belief 안정화**일 수 있음.

---

## 3. 방법론 심층 분석: Belief 조건화 행동 생성

- 갱신된 belief를 decoder 조건 공간으로 투영: `m_t = W_m h_t + b_m`.
- Flow-matching timestep 임베딩 `z_τ`와 **가산 융합**: `c_t = z_τ + m_t`, 이를 **AdaLN**으로 각 decoder layer의 정규화 활성값에 feature-wise affine 변조로 주입 (DiT/FiLM 방식).
- 결과: memory-conditioned flow velocity `v_θ(a_τ, τ | I_t, s_t, ℓ, h_t)` — 메모리 토큰 cross-attention 없이 belief가 생성되는 action chunk를 직접 변조. 시각적으로 동일한 관측이라도 유지된 belief가 다르면 다른 행동 생성 가능.
- 설계 의도: cross-attention이 열등해서가 아니라, LTC 메모리가 "검색 가능한 history buffer"가 아닌 **고정 크기 time-consistent belief state**라는 역할에 맞는 단순한 인터페이스이기 때문 (Appendix B-A).

📌 [Figure 2 삽입] — TFP 전체 구조: 연속시간 belief 유지 + AdaLN 주입

---

## 4. 학습: Episode-Aware Temporal Batching (EATB)

- 문제: chunk 단위 random shuffle은 hidden-state 연속성을 깨고, full-episode BPTT는 대형 VLA에서 메모리 비용이 과도.
- 해법: B개 활성 에피소드를 병렬 샘플링해 각각 **K=8개 연속 chunk**를 unroll. 에피소드-로컬 hidden state는 저장소에서 복원되어 세그먼트를 통과한 뒤 `h̃ ← stopgrad(h_{b,K})`로 **gradient 절단 후 write-back**.
- 효과: hidden state는 학습 세그먼트를 넘어 수치적으로 지속(장기 forward memory 보존)되지만 backprop 길이는 유계. 에피소드 경계에서만 reset.
- 추론 시: **adaptive receding-horizon executor** — 정책은 항상 horizon H chunk를 예측하되, gripper 전환 경계나 고위험 구간(jerk·boundary·continuity 가중 risk R_t,r)에서 prefix를 조기 종료하고 재질의. 다음 메모리 갱신은 실제 경과 `Δt_{t+1} = E_t δt_ctrl`을 받음.

📌 [Figure 3 삽입] — EATB: 연속 chunk 학습과 episode-local hidden state 전달

---

## 5. 기존 메모리 메커니즘과의 이론적 비교

- **GRU**: 보간 형태는 유사하나 gate가 recurrent step 인덱스 기반 — Δt 입력이 없으면 같은 관측·상태에서 물리적 간격이 달라도 같은 gate. TFP는 `k_t = exp(−Δt_t/τ_t)`로 경과 시간 보정이 구조에 내장.
- **Fixed-decay memory**: τ₀ 상수인 특수 케이스. TFP는 `τ_t = τ_θ(x_t, h_{t−1})`로 증거와 belief 모두에 적응.
- **SSM (S4/Mamba)**: 연속시간 SSM도 elapsed-time 의존 전이 구현 가능 — 저자들은 LTC가 모든 SSM보다 시간-일관적이라 주장하지 않고, **비선형 입력 의존 belief fuser로서 생성 decoder를 직접 조건화**한다는 기능적 차이를 강조.
- 오버헤드: LTC 갱신 O(d_h(d_x+d_h)) + projection O(d_h d_c) + AdaLN affine — 백본 대비 미미. 주 비용은 recurrent 학습.

---

## 6. 실험 설정

- **백본**: π0.5 (3.3B)를 primary VLA 백본으로 사용. 언어 prompt + RGB 관측 2개 + proprioceptive state, 256차원 episode-local memory.
- **벤치마크**: LIBERO(일반 롤아웃 품질), LIBERO-plus(occlusion·시각 섭동 zero-shot 강건성), MIKASA-Robo ShellGameTouch(occlusion 하 belief 필터링 진단), 실제 Galaxea A1 2개 태스크(object swapping, counting pick-and-place).
- **통제 비교**: 같은 백본·데모·action horizon·optimizer·rollout protocol·**동일한 irregular-query 스케줄**(adaptive executor)에서 재현. 발표된 baseline 수치는 literature reference로 표기.
- **실기기**: single-arm Galaxea A1 + parallel gripper, wrist fisheye + 고정 RealSense D435, ROS1 Noetic joint-angle 명령.
- **통계**: LIBERO는 suite당 10 task × 50 trials = 500 rollouts, Wilson 95% CI 보고 (Appendix E).

---

## 7. 주요 결과

### LIBERO (Table I, VI)
| 정책 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 (3.3B) | 98.8 | 98.0 | 98.2 | 92.4 | 96.85 |
| OpenVLA-OFT (7.3B) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| AVA-VLA (7.3B) | 97.4 | 99.4 | 97.4 | 97.6 | 98.0 |
| **TFP (3.3B)** | **99.6** | 99.0 | **99.4** | 97.0 | **98.75** |

- 평균 +1.90pp [95% CI +0.99, +2.81], 특히 **Long +4.6pp [+1.8, +7.4]** — 장기 태스크에서 belief 유지 가설을 뒷받침.

### LIBERO-plus
- 평균 91.4 → **93.77%** (+2.37pp), Noise 85.2→88.5, Light 93.9→96.1 — 관측이 불완전할 때 belief가 행동 생성을 안정화.

### MIKASA-Robo ShellGameTouch
- **75.0%** (OpenVLA-OFT 47.0 대비 대폭 개선). 단, MemoryVLA 88.0이 있어 SOTA 주장 아닌 **진단**으로 제시 — 남은 격차는 object-centric hidden-location binding의 필요성을 시사 (Appendix D-B).

### 실제 Galaxea A1 (Table II, VII)
- Object swap: π0.5 3/20(15%) → TFP **15/20(75%)**; Counting pick-place: 8/20(40%) → **18/20(90%)**.
- 실패 유형 분석: reactive baseline은 완료한 subtask 반복·단계 착오(Stage/Repeat)가 지배적, TFP 실패는 target grounding·저수준 실행 오류로 이동 — **task-progress 추적 개선**이라는 메커니즘 해석과 일치.

---

## 8. Ablation 및 메커니즘 분석

### 8.1 Memory dynamics ablation (Table IV)
| 방법 | LIBERO Avg | LIBERO-plus Avg |
|---|---|---|
| π0.5 | 96.9 | 91.4 |
| π0.5 + GRU | 95.5 | 91.2 |
| π0.5 + SSM (S4D) | 96.625 | 90.0 |
| TFP w/o Δt | 96.9 | 90.1 |
| **TFP** | **98.75** | **93.8** |

- GRU는 spatial에선 경쟁력 있으나 object/goal/long에서 열화, S4D도 TFP 미달 — **일반적 recurrence만으로는 부족**.
- **TFP w/o Δt** (동일 LTC+AdaLN, 측정된 경과 시간 대신 상수 step): LIBERO Long·LIBERO-plus에서 뚜렷한 하락 — **실제 경과 시간 조건화 자체가 기여**함을 분리 입증. 이 ablation 설계가 논문의 가장 설득력 있는 부분 중 하나.

### 8.2 Event-sensitive belief dynamics (Table III)
- LIBERO mug-in-microwave 30개 성공 롤아웃의 224개 이벤트에서 write-gain 변화 |Δg_t|가 **이벤트 근처에서 far non-event baseline 대비 약 6배** (예: 평균 |Δḡ_t| 6.24× [4.81, 7.88]), lag ≈ −1.6~−2.1로 시뮬레이터 이벤트 라벨보다 **약간 앞서** write/retain 스케줄 조정. 이벤트 라벨은 post hoc 해석용으로만 사용(정책 입력 아님).

### 8.3 Hidden-state intervention (Table V, Appendix D-A)
- 관측·상태·지시를 고정하고 LTC hidden state만 25개 시점의 것으로 교체 → 10-step action chunk의 평균 pairwise 거리 0.0442(최대 0.1303), 시간적으로 인접한 hidden state 간 중앙값 0.0108 — belief가 행동을 **인과적으로** 변조함을 입증.

---

## 9. 정성적 분석 및 실패 모드

- Fig. 4: 롤아웃 중 LTC gain g_t의 차원별 heatmap — 일부 차원은 운반 중 안정, 다른 차원은 reaching·carrying·releasing·pushing 전환 근처에서 변화.
- ShellGameTouch 실패 4범주 (Appendix D-B): (1) occlusion 전 ball-mug 연합 인코딩 실패, (2) 대략적 task-progress는 유지하나 정밀한 object-location binding 부족, (3) belief는 맞으나 시각적으로 동일한 머그에서 grounding 실패, (4) 저수준 실행 오류.
- 저자들의 솔직한 해석: ShellGameTouch의 결정 변수는 **이산적 hidden object-location binding**(episodic/key-value memory 영역)으로, TFP가 겨냥한 연속시간 event-sensitive task-progress 동역학과는 **부분적으로만 정렬**된 메모리 레짐. Occlusion 이후엔 belief를 교정할 증거가 없어 초기 write 오류를 복구 불가.

---

## 10. 강점

1. **원리적 설계**: 지수 retention이 elapsed-time consistency의 유일해라는 증명(Prop. 2), LTC가 belief relaxation ODE의 exact discretization이라는 해석(Prop. 1), retention과 분산 감소의 관계(Prop. 3) — 메모리 설계가 이론에 근거.
2. **정교한 ablation**: TFP w/o Δt로 "메모리 추가" 효과와 "물리적 시간 조건화" 효과를 분리했고, GRU/S4D 비교로 generic recurrence 가설을 배제. 모든 통제 변형이 동일한 irregular-query 스케줄에서 평가됨.
3. **메커니즘 검증**: write-gain의 이벤트 정렬(약 6배), hidden-state 개입의 인과 효과 등 성능 수치를 넘어선 분석.
4. **통계적 엄밀성**: Wilson 95% CI와 two-proportion CI를 보고 — 포화 구간(98%+)의 개선 주장에 필수적인데 실제로 수행한 드문 사례.
5. **실기기 검증의 큰 격차**: object swap 15→75%, counting 40→90%는 memory-dependent 태스크에서의 실질적 가치를 보여줌. 진단 벤치마크의 한계(MemoryVLA 88 vs TFP 75)를 숨기지 않는 서술 태도도 신뢰를 높임.
6. **경량성**: LTC+AdaLN의 추론 오버헤드가 백본 대비 미미하고, 3.3B로 7B급 메모리 VLA들과 경쟁.

---

## 11. 한계 및 논의

1. **학습 비용**: recurrent full fine-tuning이 K=8, batch 128, 100GB+ GPU 메모리, 4×H200에서 약 80시간 — EATB로 완화했지만 여전히 무겁고, 더 효율적인 recurrent fine-tuning은 future work.
2. **이산적 binding 메모리 부재**: ShellGameTouch에서 보듯 categorical object-location binding에는 slot memory나 key-value store 같은 object-centric 모듈이 필요 — 저자들도 하이브리드를 제안.
3. **평가 범위**: 실기기는 tabletop single-arm에 한정. mobile manipulator, humanoid, dexterous hand 미검증. LIBERO 계열 + MIKASA 1개 태스크 중심이라 CALVIN·SimplerEnv 등 교차 벤치마크 부재.
4. **baseline 비교의 비대칭**: 발표된 prior 수치(HAMLET, AVA-VLA, MemoryVLA)는 동일 프로토콜 재현이 아닌 literature reference — 저자들이 명시했지만 Table I의 직접 비교 해석에는 주의 필요.
5. **π0.5 단일 백본**: OpenVLA 등 다른 백본으로의 이식성은 주장 수준이며 실험적 검증 없음.

---

## 12. 종합 평가

TFP는 "VLA에 메모리가 있는가"가 아니라 "**메모리 갱신이 조작(manipulation)의 이벤트 구조와 물리적 시간을 따르는가**"로 질문을 옮긴 연구다. LTC retention의 elapsed-time consistency 증명, exact-discretization 해석, 그리고 TFP w/o Δt ablation이 하나의 논리로 연결되어 있고, write-gain 이벤트 정렬과 hidden-state 개입 분석이 belief의 인과적 역할을 실증한다. 3.3B로 LIBERO 98.75%·LIBERO-plus 93.77%를 달성하고 실기기 memory-dependent 태스크에서 15→75%, 40→90%의 큰 격차를 보인 점, Wilson CI 기반의 엄밀한 보고까지 완성도가 높다. 다만 recurrent 학습 비용, 단일 백본·제한된 벤치마크 범위, 이산적 object binding의 미해결이 확장성의 관건이다. Action head는 LTC belief를 AdaLN으로 주입받는 flow-matching decoder로 **flow_matching**으로 분류된다.

**Score: 8.0 / 10**

<!-- VERIFIED: pdf -->
