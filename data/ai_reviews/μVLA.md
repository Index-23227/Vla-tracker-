# μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models

> **한 줄 요약**: 사전학습된 VLA(OpenVLA-OFT) backbone에 m개의 학습 가능한 memory token만 끼워 넣고 TBPTT로 fine-tune하여, 보조 손실/검색/계층 메모리 없이 *recurrence 자체*만의 기여를 격리 측정한 controlled isolation study. MIKASA-Robo 학습 5-태스크 평균 SR 0.42 → 0.84 (K=2), held-out matched-semantics 0.07 → 0.23, LIBERO 96.2% (m=64, K=8) 무회귀(no regression) 달성.

## 1. 배경 및 동기

- VLA 모델(OpenVLA, π₀, RT-2 등)은 현재 관측 oₜ가 충분통계라는 Markovian 가정 위에서 미래 action chunk를 예측. 부분관측(partial observability) 상황(cue가 가려지거나 episode 초기에 한 번만 등장) 에서는 근본적으로 실패.
- 기존 memory-augmented VLA(CronusVLA, MemoryVLA, TraceVLA 등)는 recurrence, retrieval, compression, auxiliary loss, hierarchical state 등을 **동시에** 도입 → recurrence 단독의 기여를 분리 불가.
- 본 논문은 "recurrence를 단일 실험축으로" 두고 다른 모든 기계장치를 제거. Recurrent Memory Transformer(RMT, Bulatov et al. 2022)의 아이디어를 environment-step 수준에 적용.
- 평가 철학: SOTA를 노리는 게 아니라 *최소 recurrence가 무엇을 할 수 있고 어디서 부족한지*의 capability envelope 보정.

## 2. 방법론

### 2.1 전체 프레임워크 (§3, Fig. 1-2)
- Backbone: OpenVLA(PrismaticVLM = SigLIP + DinoV2 + Llama-2 7B) + OFT-style parallel decoding continuous action head. LoRA r=32 fine-tune.
- 매 timestep t에서 입력: vision obs oₜ + proprioception + language ℓ + 직전 step의 memory tokens M^{t-1}.
- 출력: action chunk (â_t,…,â_{t+H-1}) 와 갱신된 memory M^t. M^t는 다음 step에 재귀적으로 전달.

### 2.2 Memory Token 삽입 위치 및 read/write (§4)
- m개 학습 가능한 memory token을 PROPRIO와 action token 사이에 삽입. t=0에서는 공유 학습 파라미터 M^{init}; t≥1에서는 직전 hidden state의 memory 위치에서 read.
- 동일한 self-attention forward pass 한 번이 memory를 *읽고* 동시에 다음 step state를 *쓴다*: M^t = h_θ(o_t, ℓ; M^{t-1})|mem positions.
- **Memory-action guard (핵심)**: 일반 mask에서는 memory가 action 토큰을 attend하므로 trivial solution M^t = ϕ(ACTION_t) (예측 action을 그대로 복사) 가 가능. context-to-action block을 zero-mask하여 memory 및 prefix가 action에 attend 불가능하도록 강제 (Fig. 3). EMA full-mask 변형이 LIBERO에서 무너지는 이유.

### 2.3 학습: TBPTT vs EMA (§4)
- **TBPTT**: K 연속 step에 걸쳐 L1 chunk loss 누적, K-step recurrent graph로 단일 backward; memory chain은 K 경계에서만 detach. K ∈ {1, 2, 8} 비교.
- **EMA write**: M^{t+1} = α M'^t + (1-α) M^t (α=0.1), 양변 모두 detach. Backward는 단일 step에 국한 → 메모리 비용 ↓, 그러나 cross-step credit assignment 없음.
- 두 변형 모두 동일 backbone, mask, dataloader, memory 위치 공유 → 차이가 오직 write rule.

### 2.4 Receding-horizon inference (§4)
- Dataloader는 매 step memory를 갱신하지만 일반 chunked execution은 H step마다 한 번만 갱신 → factor-of-H mismatch.
- 따라서 inference에서도 매 step state 갱신(chunk=1) 후 첫 action만 실행하는 receding-horizon control 사용. LIBERO에서 같은 m=64/K=8 체크포인트를 open-loop chunked로 평가하면 Long-10 95.8 → 5.4, Goal 96.6 → 35.8로 붕괴.

## 3. 실험 결과

### 3.1 MIKASA-Robo-VLA (Table 1, §5.1)
23개 환경; 5개 train + 11개 held-out matched-semantics + 7개 held-out novel-semantics. 100 deterministic episodes/환경.

| 조건 | Train 5 avg | Held-out matched 11 | Held-out novel 7 |
|---|---|---|---|
| π₀.5 | 0.46 | 0.03 | 0.00 |
| OpenVLA-OFT (memoryless) | 0.42 | 0.01 | 0.01 |
| OpenVLA-OFT† (epis. dataloader, m=0) | 0.48 | 0.07 | 0.07 |
| μVLA m=1, K=8 | 0.54 | 0.07 | 0.09 |
| μVLA m=64, K=8 | 0.57 | 0.09 | 0.09 |
| **μVLA m=64, K=2 (best)** | **0.84** | **0.23** | 0.16 |
| μVLA m=64, K=1 | 0.57 | 0.09 | 0.11 |
| μVLA m=64, EMA | 0.57 | 0.09 | 0.09 |
| μVLA m=64, EMA full-mask (no guard) | 0.57 | 0.08 | 0.10 |
| OpenVLA-OFT + 1st obs (oracle) | 0.85 | 0.24 | 0.19 |

- 단일 task RememberColor5: K=2에서 0.93 vs K=1/K=8의 0.35/0.40 → cue-recall에서 K=2 sweet spot 명확.
- Episodic dataloader 기여 +6pp, memory bandwidth (m=1 → m=64) +9pp at K=8, TBPTT length가 cue-recall의 핵심 lever.

### 3.2 LIBERO (Table 2, §5.2) - Markovian 통제 실험

| Method | Spatial | Object | Goal | Long-10 | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 93.2 |
| CronusVLA* | 90.1 | 94.7 | 91.3 | 68.7 | 86.2 |
| MemoryVLA* | 98.4 | 98.4 | 96.4 | 93.4 | 96.5 |
| **μVLA (m=64, K=8)** | 93.0 | **99.4** | 96.6 | **95.8** | **96.2** |
| μVLA (m=64, EMA) | 70.8 | 64.4 | 6.6 | 37.2 | 44.8 |

- TBPTT K=8은 OpenVLA-OFT 대비 -0.9pp로 no-regression; Object/Long-10에서 오히려 향상.
- EMA full-mask 변형 붕괴(특히 Goal 6.6)는 §2.2의 memory-action guard 부재 시 trivial action-copy solution이 학습되어 fully observable에서도 무너짐을 시사.

### 3.3 Diagnostics (§5.3, Appendices)
- **Representation dynamics (Fig. 4)**: RememberColor5에서 K=2의 1−cos(M^t, M^{t-1})은 cue 사라짐(t=4→5)과 candidate 등장(t=10) 두 phase transition에서 sharp spike → 메모리가 phase-aware 동작.
- **Noise intervention**: M^t를 i.i.d. Gaussian으로 교체하면 RememberColor5 K=2: 0.94 → 0.09, TakeItBack K=2: 0.99 → 0.21 → 메모리 채널이 실제 사용됨을 입증.
- **freeze_first**: M^1로 고정 시 RC5 0.36 유지(첫 frame cue 충분) but InterceptMedium 0.07로 붕괴(동적 cue 필요).
- **Phase-length OOD (Fig. 8)**: RC5-PhaseN에서 K=2는 in-dist(N=3) 0.93이지만 N=20에 ~75% recall 손실; K=8은 0.31-0.38 flat; EMA는 N≥20에서 <0.15 붕괴.
- **Color-swap OOD**: 5색 모두 교체에서도 K=2가 0.48로 chance 위 → 부분적으로 추상 cue encoding 학습.

### 3.4 학습 비용 (Appendix Table)
| Config | GB | hours/epoch | wall clock |
|---|---|---|---|
| OpenVLA-OFT, LIBERO | 18.83 | 36.92 | 1d0h49m |
| μVLA, LIBERO, m=64, K=8 | 22.73 | 55.62 | 10d19h12m |
| μVLA, LIBERO, m=64, EMA | 19.96 | 38.73 | 1d2h26m |

K=8 TBPTT는 ~11× cost. EMA는 OpenVLA-OFT 대비 +3% 시간만으로 학습 가능하지만 cue-recall에서 K=2에 미달.

## 4. 한계 및 미해결 문제

1. **Novel memory semantics에 약함**: 학습에서 등장하지 않은 memory 종류(Shape only, Rotate 진행 추적)에서는 K=2조차 memoryless baseline 근방(0.16). 즉 recurrence가 메모리 *capability*를 부여하지 *어떤 cue type이든 일반화*시키지는 않음.
2. **TBPTT 비용**: K=8 학습이 ~11× longer wall-clock. K=2가 best이지만 이는 cue-recall이 짧은 phase에 집중된 결과일 수 있음 — 더 긴 horizon task에서 K가 더 커야 할 가능성과 그 비용 trade-off는 해결되지 않음.
3. **Receding-horizon 강제**: Chunked execution과 train-test mismatch가 너무 커서(LIBERO Long-10 95.8 → 5.4) inference latency가 chunked open-loop 대비 H배 증가. 실시간 deployment에서는 부담.
4. **Phase-length 일반화**: K=2가 N=3(in-dist)에서 0.93이지만 N=20에서 0.23 수준으로 떨어짐. 메모리 폭 m=64로는 long horizon 누적 정보 보유 불가.
5. **EMA full-mask 붕괴**: Memory-action guard 없으면 LIBERO Goal에서 6.6%까지 떨어짐. Trivial solution 회피 메커니즘이 매우 fragile하며 일반 attention design 시 자명하게 보장되지 않음.

## 5. 총평

- **Novelty: ★★★☆☆** — 새로운 architecture를 제안하기보다 RMT 스타일 recurrence를 VLA에 적용해 *isolate*한 작업. 컨트리뷰션은 controlled experimental setup 자체.
- **Scientific value: ★★★★★** — Memory-augmented VLA의 gain 중 어디까지가 recurrence이고 어디까지가 retrieval/auxiliary loss/hierarchical state인지 그 누구도 분리하지 못했음. 본 논문이 제시한 (m, K, write rule)만 변하는 family는 이후 모든 memory VLA 연구의 baseline reference가 될 수 있음.
- **Practical impact: ★★★☆☆** — MIKASA-Robo는 합성 partial-observability benchmark이며 실세계 평가 없음. 그러나 LIBERO no-regression은 기존 OpenVLA 사용자가 minimal cost로 memory 모듈을 추가할 motivation 제공.

OpenVLA-OFT에 memory token 64개와 K=2 TBPTT만 추가하면 부분관측 manipulation에서 SR 2배 향상이 가능하고 Markovian task에서는 손실 없음을 *깔끔하게* 보인 논문. "memory는 이미 강력한, 그러나 덜 연구된 ingredient"라는 결론은 향후 retrieval/hierarchical/auxiliary 기여를 분리하려는 모든 후속 연구의 출발선.

## 6. 예상 질문

| Q | A |
|---|---|
| 왜 K=2가 K=8보다 좋은가? | Credit-assignment(긴 K) vs signal resolution(짧은 K) trade-off의 sweet spot. RememberColor5의 cue→action gap이 ~5 step이라 K=2의 2-step gradient가 충분하면서도 K=8의 graph noise를 피함. K=1은 cross-step gradient 자체가 없어 cue-recall 불가. |
| Memory-action guard 없이 EMA full-mask가 왜 무너지나? | Memory가 action token을 attend 가능하면 M^t = ϕ(ACTION_t) trivial copy solution이 학습되어 실제 환경 정보를 저장하지 않음. LIBERO Goal 6.6%는 이 self-referential collapse의 결과. |
| Held-out matched-semantics에서도 0.23 (vs 0.07)는 작아 보이는데? | 11개 환경 중 RememberColor3는 0.92, RememberShapeAndColor3x2는 0.59로 일부는 큰 transfer. 평균은 ShellGameTouch/Pick(전혀 0)이 끌어내림 — task family-level transfer는 강하나 action primitive shift에는 약함. |
| MemoryVLA보다 LIBERO 평균(96.2)이 낮은 이유? | μVLA는 단일 recurrence axis만 변경; MemoryVLA는 PCMB retrieval+consolidation+timestep PE를 모두 도입. 본 논문은 "더 잘하는 것"이 목표가 아니라 "recurrence 자체가 얼마나 기여하는지" 격리가 목표. |
| OpenVLA 외 다른 VLA에서도 통할까? | 본 논문은 OpenVLA-OFT 한정. π₀ (flow matching), GR00T 등은 token 구조와 action head가 달라 memory token 삽입과 receding-horizon 갱신이 똑같이 작동할지 검증 필요. 저자도 결론에서 제한 명시. |

<!-- VERIFIED: pdf -->
