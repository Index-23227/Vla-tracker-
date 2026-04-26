# LingBot-VLA: A Pragmatic VLA Foundation Model

> **한 줄 요약**: Ant Group/Robbyant이 9종 dual-arm 로봇 플랫폼에서 수집한 약 20,000시간 실제 데이터로 사전학습한 Qwen2.5-VL-3B 기반 MoT(Mixture-of-Transformers) + flow matching VLA로, GM-100 100태스크/3플랫폼 22,500 trial 평가에서 π0.5/GR00T-N1.6/WALL-OSS를 모두 능가하는 SOTA를 기록했다.

---

## 1. 배경 및 동기

VLA foundation model의 실제 robot performance가 사전학습 데이터 규모에 따라 어떻게 scale하는지에 대한 체계적 실증 연구가 부재했다(§1). 또한 기존 codebase는 multi-node 학습 시 I/O 병목과 통신 오버헤드로 효율이 낮다. 저자들은 (1) 20,000시간 규모의 실제 dual-arm 데이터로 scaling law를 입증하고, (2) FSDP+HSDP+FlexAttention 기반 고속 codebase로 8-GPU에서 261 samples/s 처리량(기존 대비 1.5~2.8× 가속)을 달성하는 것을 목표로 한다.

---

## 2. 방법론 심층 분석

### 2.1 MoT 아키텍처 (§4.1, Eq. 1-4)

- **VLM 백본**: Qwen2.5-VL-3B
- **Action Expert**: BAGEL 스타일 MoT — VL과 action 모달리티가 별도 transformer 경로로 처리되되 layer-wise shared self-attention으로 결합
- **Joint sequence**: Ot = [I1,I2,I3, T, st], At = [at, ..., at+T-1] (T=50 chunk length, Eq. 1-2)
- **Flow Matching loss** (Eq. 4): LFM = E[‖vθ(At,s, Ot, s) − (At − ε)‖²], blockwise causal attention(π0 따름)

### 2.2 Depth Distillation (§4.1, Eq. 5)

LingBot-Depth로부터 depth token Dt를 받아 학습 가능한 query Qt를 정렬: Ldistill = E|Proj(Qt) − Dt|. Depth는 inference 시 optional이며 spatial perception을 강화한다.

### 2.3 데이터 (§3, Fig. 2)

9개 embodiment(AgiBot G1, AgileX, Galaxea R1Lite/R1Pro, Realman Rs-02, Leju KUAVO 4 Pro, Qinglong, ARX Lift2, Bimanual Franka)에서 ~20K hours 수집. Qwen3-VL-235B-A22B로 atomic instruction 자동 annotation 후 human refinement.

---

## 3. 실험 결과 심층 분석

### 3.1 GM-100 실세계 (Table 1)

| Platform | WALL-OSS SR | GR00T N1.6 SR | π0.5 SR | Ours w/o depth | Ours w/ depth |
|---|---|---|---|---|---|
| Agibot G1 | 2.99% | 5.23% | 7.77% | 12.82% | 11.98% |
| AgileX | 2.26% | 3.26% | 17.20% | 15.50% | 18.93% |
| Galaxea R1Pro | 6.89% | 14.29% | 14.10% | 18.89% | 20.98% |
| **Average** | 4.05% | 7.59% | 13.02% | 15.74% | **17.30%** |

Depth 사용 시 π0.5 대비 SR +4.28%, PS +7.76% 개선(§5.2).

### 3.2 RoboTwin 2.0 시뮬레이션 (Table 2)

- Clean Scenes: π0.5 82.74% → Ours w/ depth **88.56%** (+5.82%)
- Randomized Scenes: π0.5 76.76% → Ours w/ depth **86.68%** (+9.92%)

### 3.3 Throughput (Fig. 4)

Qwen2.5-VL-3B-π에서 8 GPU 기준 261 samples/s, 256 GPU 기준 7,356 samples/s — StarVLA(2,644) 대비 약 2.8× 가속.

### 3.4 Scaling Law (Fig. 5)

3,000h → 20,000h로 데이터를 늘림에 따라 SR/PS가 saturation 없이 단조 증가(§5.5.1) — 실제 데이터 scaling의 첫 실증.

---

## 4. 한계

1. **CALVIN/LIBERO 미보고**: 본 논문은 GM-100, RoboTwin 2.0만 평가. YAML에 기재된 `libero` 점수와 `calvin_abc_d_avg_len=4.5`는 본 paper에서 직접 확인되지 않으며, paper의 핵심 평가는 실세계 GM-100임.
2. **Mobile/single-arm 미포함**: dual-arm tabletop 한정(§6).
3. **Depth는 LingBot-Depth 의존**: 별도 모델이 필요하며 self-contained하지 않음.
4. **GM-100 SR 자체가 낮음**: 17.30%는 절대값으로는 낮은 편 — task 난이도가 높음을 시사하지만 deployment 신뢰성에는 추가 작업 필요.

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — MoT+flow+depth distillation의 결합은 새로움 |
| Technical depth | ★★★★★ — 학습 인프라까지 체계적 |
| Experimental rigor | ★★★★★ — 22,500 trial, 3 플랫폼 controlled |
| Practical impact | ★★★★★ — 20K hours 데이터+오픈 codebase |

**강점**: 사전학습 scaling 실증, throughput 1.5~2.8× 가속, multi-embodiment 일반화. **약점**: 표준 sim 벤치(LIBERO/CALVIN) 직접 보고 부재, depth 모델 의존성.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | π0의 blockwise causal attention과 무엇이 다른가? | π0은 단일 transformer, 본 모델은 MoT로 modality-specific path 분리(§4.1). |
| 2 | 20K hours scaling이 정말 saturation 없는가? | Fig. 5에서 단조 증가 곡선 확인되나 25K+ 영역은 미검증. |
| 3 | Depth 없을 때 손실이 큰가? | Table 1에서 w/o depth(15.74%) vs w/ depth(17.30%) — 1.56%p 차이로 유의미하지만 critical은 아님. |

<!-- VERIFIED: pdf -->
