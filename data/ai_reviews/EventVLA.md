# EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies

> **한 줄 요약**: 정책 hidden state에서 chunk-wise 미래 keyframe 확률을 예측하는 lightweight Keyframe Evidence Memory(KEM) head를 VLA에 병렬로 부착하여, 초기 frame + 단기 history(visual anchors)와 함께 일시적·상호작용 기반 시각 evidence만 sparse하게 buffer에 commit하는 end-to-end memory-augmented VLA. QwenOFT backbone 기반, RMBench 67.8%(SOTA), 신규 diagnostic suite RoboTwin-MeM 75.2%(+64.4 vs MemoryVLA-QwenOFT 10.8%), 실세계 ARX ACONE bimanual 4 task에서 평균 78.8%(+41.3 vs πMEM)로 SOTA. RoboTwin 2.0 Markovian task에서도 baseline 대비 손해 없이 +3.6/+3.6%p 향상.

## 1. 배경 및 동기

- VLA(π₀, π₀.₅, OpenVLA, X-VLA 등)는 강력하나 strict Markovian 가정에 묶여 task-critical 정보가 occlusion되거나 사라지면 실패. Push Buttons처럼 중간 상태를 기억해야 하는 non-Markovian task에서 무너짐(§1, Fig. 1a).
- 기존 memory-aware VLA 3가지 paradigm 모두 결함:
  1. **Dual-system Memory-VLAs**(MemER, Mem-0, πMEM): high-level VLM이 따로 추론 → error propagation + 높은 latency.
  2. **Recurrent**(RMT, AVA-VLA): hidden state 압축으로 fine-grained 시각 디테일 손실(information bottleneck).
  3. **Memory Buffer**(MemoryVLA, CronusVLA, LoLA, ContextVLA): 시각 정보는 보존하나 selective mechanism 없어 redundant frame이 누적, sparse key evidence가 희석되고 연산량 폭증.
- 핵심 질문: "정확히 언제, 어떤 시각 evidence를 보존해야 max success + min compute?"
- 두 가지 관찰:
  1. 많은 long-horizon task는 sparse한 historical keyframe만으로 충분 → "foundational visual anchors"(초기 frame + 단기 history window)로 충분.
  2. 그러나 cover를 잠시 열어 색을 본다든가 stick으로 가리키는 순서를 본다든가 하는 transient evidence는 anchor만으로 잡히지 않음 → KEM이 필요.
- 더불어 기존 memory benchmark(RMBench, RoboMME)는 static anchor만으로도 풀리는 경우가 많아 진정한 non-Markovian 평가에 부족 → 신규 RoboTwin-MeM 제안.

## 2. 방법론

### 2.1 문제 정식화와 Foundational Visual Anchors (§3.1)

- 표준 reactive policy: `a_t = π(o_t, l)` → non-Markovian에서 실패.
- EventVLA: 외부 sparse visual evidence memory `M_t = A_t ∪ E_t` 도입.
  - `a_t = π(o_t, M_{t-1}, l)` (Eq. 1).
- **Visual Anchors `A_t`** (Eq. 2): `A_t = {o_0} ∪ {o_{t-K},…,o_{t-1}}`. 초기 frame `o_0`는 invariant global layout 유지, 단기 sliding window는 motion/progression cue 제공.
- 그러나 rigid한 anchor만으로는 중간에 잠깐 나타나는 evidence를 못 잡음 → KEM이 dynamic하게 보완.

### 2.2 Keyframe Evidence Memory (KEM) Module (§3.2, Fig. 2)

- **구조**: 기존 action head와 병렬인 lightweight MLP head. VLA autoregressive transformer 최종 layer hidden state `h_t ∈ R^{H×d}`(action horizon H개의 query token)를 그대로 입력으로 받음 → 시각 관측과 action-conditioned query embedding이 결합돼 있어 미래 실행 plan에 대한 proactive awareness가 내재.
- **Forecast** (Eq. 3): `p̂_t = σ(KEM_mlp(h_t)) = [p̂_t^1,…,p̂_t^H] ∈ [0,1]^H`. 각 `p̂_t^i`는 미래 i-step이 task-critical keyframe일 확률.
- **이유**: step-wise classifier는 chunk 중간(t+i, 0<i<H)에서 잠깐 나타났다 사라지는 event를 놓침 → chunk-wise foresight로 "memory schedule"을 미리 작성.
- **Commit**: `p̂_t^i ≥ τ_commit`이면 raw image `o_{t+i}`를 event buffer `E_t`에 기록. FIFO, capacity `N_max`로 bounded.
- **Memory 통합** (Eq. 4): `I_input = concat([A_t, E_{t-1}, o_t])` → VLM vision encoder 입력. Self-attention이 sparse 과거 frame 간 temporal correlation을 자연스럽게 추출.

### 2.3 학습과 추론 (§3.3, Appendix A)

- **Soft label** (Appendix A.1): GT keyframe `t*` 주변 dilation radius R 내에서 raised cosine kernel로 `y_t^i = 0.5(1+cos(π|t+i-t*|/R))` smoothing → 시간적 모호성에서 noisy gradient 방지.
- **Loss** (Eq. 6-7): chunk-averaged BCE `L_kem`과 표준 action loss(regression 또는 flow-matching) `L_action`을 결합 → `L = L_action + λ·L_kem`.
- **Teacher-to-student curriculum**: annealing α가 1→0으로 선형 감소. 확률 α로 GT keyframe을 commit(teacher forcing), 1-α로 자기 prediction을 사용. 초기 안정성 + train-test distribution shift 완화.
- **Online inference 후처리** (Appendix A.2): 
  1. Threshold `τ_commit`로 candidate 추출.
  2. 1D NMS(반경 w 윈도우 내 local max만 유지) → 같은 event의 중복 commit 제거.
  3. Temporal cooldown C: 직전 commit 시각 `t_last` 대비 `(t+i)-t_last>C`만 통과 → operational sparsity 강제.
- **자동 키프레임 라벨링** (Appendix A.3): Qwen3-VL-235B로 demo 영상 + task description을 파싱해 `t*` 자동 추출 → 수작업 annotation 비용 회피.
- Base model: QwenOFT(StarVLA OFT 구현체, [46]) — π₀.₅, MemoryVLA, EventVLA 모두 비교를 위해 동일 backbone 변형 사용.

## 3. RoboTwin-MeM 벤치마크 (§4)

- RoboTwin 2.0 + SAPIEN 위에 구축, 자동 데이터 생성 + 통합 평가 pipeline.
- 8개 task. 각 task에 **n** parameter (1≤n≤5): 반드시 기억해야 하는 transient keyframe 개수. Episode당 평균 430–1544 step의 매우 긴 horizon.
- Task별 n:
  - n=1: Rearrange Blocks Hard
  - n=2: Put Back Block Hard
  - n=3: Pick Objects in Order, Pick the Unhidden Block
  - n=4: Cover Blocks Hard, Reproduce Route, Find Seal And Seal Stamp(1–4)
  - n=5: Press Button Keyframe(2–5)
- 평가 능력: (1) transient memory(cover 잠깐 열리고 닫힘), (2) sequence tracking & counting(button press), (3) in-context imitation(Reproduce Route에서 stick으로 보여준 경로 따라하기).
- 기존 RMBench/RoboMME 한계 해소: static anchor만으로 풀 수 없는 genuinely non-Markovian만 격리.

## 4. 실험 결과

### 4.1 RMBench (Table 1)

| Method | Avg(%) |
|---|---|
| DP | 5.8 |
| ACT | 5.9 |
| π₀.₅ | 10.4 |
| X-VLA | 9.8 |
| QwenOFT | 5.6 |
| MemER | 8.7 |
| Mem-0 | 42.0 |
| MemoryVLA(OpenVLA) | 19.4 |
| MemoryVLA(QwenOFT) | 41.7 |
| **EventVLA(VA only)** | **67.8** |
| EventVLA(w/o initial) | 33.7 |
| EventVLA(w/o short-term) | 23.8 |

- RMBench는 기본적으로 persistent layout + fixed motion에 의존하므로 KEM 없이 visual anchor만으로도 **67.8% SOTA**. 초기 frame과 단기 history 모두 indispensable(제거 시 23.8–33.7로 급락).

### 4.2 RoboTwin-MeM (Table 2)

| Method | Total avg |
|---|---|
| π₀.₅ | 7.8% |
| QwenOFT | 3.8% |
| MemER | 10.5% |
| Mem-0 | 0.0% |
| MemoryVLA(OpenVLA) | 4.9% |
| MemoryVLA(QwenOFT) | 10.8% |
| EventVLA(VA only) | 18.0% |
| **EventVLA(VA+KEM)** | **75.2%** |

- VA only는 18.0%에 그치지만 **VA+KEM 75.2%(+64.4 vs MemoryVLA-QwenOFT)**. 단순 history concatenation 패러다임을 완전히 분쇄.
- Per-task 하이라이트: Put Back Block Hard 93%, Pick Objects in Order 90%, Cover Blocks Hard 94%, Reproduce Route 98%.

### 4.3 RoboTwin 2.0 표준 Markovian (Table 3)

- Easy: QwenOFT 80.0% → EventVLA **83.8%** (+3.8). Hard: 78.0% → **81.6%** (+3.6).
- Memory mechanism이 reactive control에 손해를 주지 않고 오히려 보완.

### 4.4 실세계 ARX ACONE bimanual (Fig. 4, 4 tasks × 20 trial)

| Task | π₀.₅ | πMEM | **EventVLA** |
|---|---|---|---|
| Find Block Easy | 10 | 50 | **90** |
| Pick-X-Times | 0 | 30 | **60** |
| Find Block Hard | 10 | 30 | **90** |
| Pick in Order | 0 | 40 | **75** |
| **Avg** | 5.0 | 37.5 | **78.8 (+41.3)** |

- π₀.₅(reactive)는 거의 0–10%, πMEM도 multi-event task(Pick-X-Times, Pick in Order)에서 lossy compression으로 무너지는 반면 EventVLA는 60–90% 안정적.

### 4.5 Ablation (Table 2 하단)

| Variant | Total |
|---|---|
| Implicit memory bank (latent) | 24.9% |
| Hard binary label | 48.8% |
| w/o NMS | 53.4% |
| N_max=2 (대신 충분 buffer) | 32.0% |
| Chunk size 30 | 31.1% |
| Chunk size 15 | 13.6% |
| **Full (VA+KEM, soft label, NMS, N_max~, chunk 50)** | **75.2%** |

- **Raw image concat이 핵심**: latent bank로 바꾸면 75.2→24.9로 붕괴(실제 자기 ablation으로 information bottleneck 입증).
- Soft label 없으면 48.8 (predictive head 학습 불안정).
- NMS 없으면 53.4 (중복 commit로 buffer flooding).
- Buffer 작거나 chunk 짧으면 31.1–13.6 (early evidence FIFO 축출, foresight window 부족).

## 5. 한계 및 미해결 문제

1. **Bounded event buffer**(§6): N_max로 인해 10분 초과·event-dense task에서 buffer saturation과 premature eviction 위험. 저자는 hierarchical/compressed representation을 future work로.
2. **자동 라벨링이 Qwen3-VL-235B 의존**: 라벨 품질 = pipeline 품질. 특이 도메인(미세 의료, 산업 검사)에서 일반화는 미검증.
3. **벤치마크 자기 평가 위주**: RoboTwin-MeM·RMBench는 RoboTwin 계열 환경 위에서 동일 backbone(QwenOFT)으로 reproduce된 baseline과 비교 → 외부 third-party 재현 비교는 부재.
4. **LIBERO·CALVIN 등 공개 표준 점수 미보고**: SimplerEnv·LIBERO에서의 비교가 없어 mainstream VLA benchmark 위치는 미확정.
5. **Real-world 4 task 모두 ARX ACONE bimanual + relatively constrained 환경**: 모바일·outdoor·cluttered 일반 환경 검증 부족.

## 6. 총평

- **Novelty: ★★★★☆** — "memory를 언제 쓸지를 정책의 hidden state로부터 직접 예측한다"는 단순하지만 깔끔한 아이디어. Lightweight head + chunk-wise foresight + NMS/cooldown으로 구현 비용 거의 없이 huge gain. RoboTwin-MeM이라는 새 평가 axis를 같이 제시한 것도 큰 기여.
- **Practical impact: ★★★★☆** — non-Markovian 로봇 조작의 실용적 baseline. 실세계 78.8% 평균은 기존 memory-augmented 대비 +41.3pp로 압도. 다만 code/checkpoint 공개 정보 없음, project page만 약속.

memory-augmented VLA의 흐름을 "selective, foresight-driven, sparse"로 명확히 정의한 작업. MemoryVLA가 cognitive science(verbatim+gist) 비유로 정적 memory bank를 깔끔히 정리했다면, EventVLA는 "memory write를 정책의 미래 예측으로부터 emergent하게 정의한다"는 메타-원리를 보여준다. KEM head는 추가 비용 거의 없이 어떤 VLA에도 부착 가능한 plug-in 형태라 후속 연구(π₀.₆, X-VLA 등)에 빠르게 흡수될 가능성이 높음.

## 7. 예상 질문

| Q | A |
|---|---|
| KEM head가 단순 step-wise classifier가 아니라 chunk-wise prediction인 이유? | 한 chunk 내부에서 잠깐 나타났다 사라지는 event(예: cover가 잠깐 열린 frame)를 step-wise classifier는 미래 정보 없이 놓침. Chunk-wise는 미래 전체 horizon에 대한 "memory schedule"을 한 번에 작성. Ablation에서 chunk 50→30→15로 줄이면 75.2→31.1→13.6으로 급락하여 foresight window 폭의 중요성 입증. |
| Soft label이 왜 필요한가? | Physical interaction의 keyframe은 t* 전후 1–2 frame이 모두 equally valid → 단일 시점 binary가 noisy gradient 유발. Raised cosine kernel로 smoothing하면 hard label 48.8 → soft 75.2. |
| Visual anchor만으로 RMBench SOTA가 가능한 이유는? | RMBench task는 결국 fixed layout + motion style에 의존. 초기 frame이 layout, 단기 window가 motion cue 제공 → KEM 없이도 67.8% 달성. 두 anchor 중 하나만 빼도 23.8/33.7로 무너져 둘 다 필수임을 확인. |
| MemoryVLA가 QwenOFT backbone에서도 10.8%인데, EventVLA가 75.2%인 본질적 차이는? | MemoryVLA는 모든 step의 perception/cognition 토큰을 token merge로 압축해서 누적 → "blind 누적 + 압축" 패러다임. EventVLA는 정책의 미래 prediction으로 "언제 쓸지"를 결정 + raw image concat → sparse + lossless. Implicit latent bank로 바꾼 ablation(24.9%)이 raw image 보존의 중요성을 직접 증명. |
| Train-test distribution shift는 어떻게 해소? | annealing α 1→0의 teacher-to-student curriculum. 초기엔 GT keyframe으로 commit(stable), 후반엔 자기 prediction(test와 동일). |

<!-- VERIFIED: pdf -->
