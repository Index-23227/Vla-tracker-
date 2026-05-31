# IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation

> **한 줄 요약**: VLA imitation 데이터의 본질적 multimodality — 동일한 visual-language 관측이라도 시연자의 단기 의도에 따라 후속 action chunk가 달라지는 현상 — 을 정면 해결. Qwen3-VL 4B + frozen VGGT-1B를 결합해 최근 visual history에서 compact intent representation을 인코딩하고, DiT 기반 flow-matching head로 chunk를 생성한다. AliasBench 45.8% (baseline 9.0%), LIBERO 98.6% avg, SimplerEnv 72.9%, RoboCasa 57.0%를 보고.

---

## 1. 배경 및 동기

### 문제 정의: Short-Horizon Observation Aliasing
- 로봇 imitation 데이터는 본질적으로 **multimodal**: 비슷한 visual-language 관측 뒤에 서로 다른 action chunk가 따라오는 경우가 흔함 (시연자가 어떤 단기 의도를 가지고 있느냐에 따라 분기)
- 기존 frame-conditioned VLA는 현재 관측 + instruction만으로 chunk를 추론 → partial observability 하에서 매 replanning step마다 다른 의도를 resampling → **inter-chunk conflict** 발생
- 논문 표현: "demonstrations are multimodal across episodes but locally committed within an episode"

### 핵심 질문
- **시연자의 "locally committed" intent를 모델 측에서 복원할 수 있는가?**
- **그 intent가 정말 chunk 간 consistency 향상으로 이어지는가?**
- **현재 frame은 같지만 의도가 다른 상황을 isolate 측정할 수 있는 benchmark가 필요한가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 컴포넌트 | 역할 | 파라미터/특성 |
|---------|------|--------------|
| **Qwen3-VL 4B** | 현재 frame + language context 인코딩 | trainable |
| **VGGT-1B** | 최근 visual history → geometric intent | **frozen**; camera + register tokens만 retain |
| **Gated cross-attention** | history evidence와 현재 context 융합 | Eq. 10 (논문) |
| **DiT-based flow-matching head** | Action chunk 생성 | Conditional flow-matching loss |

### 2.2 Intent Representation

- 최근 visual 관측들을 VGGT-1B에 통과시킨 뒤, 출력의 **camera token + register token만 선택적 retain** → geometric/structural cue를 압축
- 이 token들을 현재 Qwen3-VL context와 gated cross-attention으로 결합:
  - Gate는 condition-dependent — 즉 어떤 context에서는 history를 강하게 활용, 어떤 context에서는 덜
- 결과적으로 short-horizon intent가 의도적으로 **단기**로 한정 — 장기 planning과 분리

### 2.3 학습 목표

Conditional flow-matching loss:

```
L_flow = E[ || V_hat_theta(X_s, s | C_t) - (tau_t - eps) ||_2^2 ]
```

- X_s = (1-s)·eps + s·tau_t는 noise→target chunk 사이의 interpolation
- C_t는 (current obs, language, intent) 결합 조건
- DiT(Diffusion Transformer) 백본으로 chunk 단위 velocity field 예측

> ❓ **예상 질문**: 왜 VGGT를 frozen으로 두는가?
> **답변**: VGGT-1B는 multi-view geometric pre-training으로 학습된 visual geometry foundation model. Frozen으로 두어 (a) geometric prior 보존, (b) intent encoding을 별도 supervision 없이 안정화. Trainable로 풀면 chunk loss가 geometric prior를 약화시킬 위험.

---

## 3. 데이터 전략

### 학습 셋업
- 다중 benchmark에 대해 unified policy로 fine-tuning (LIBERO는 4 suite 단일 정책)
- AliasBench: RoboTwin2 위에 새로 구성한 12-task ambiguity-aware benchmark
  - 학습 환경과 평가 환경을 **matched** 시켜 일반화 변수를 제거하고 오직 aliasing 효과만 측정
  - 4 카테고리: Back-and-Forth, Crossing-Path, Bimanual, Multi-Goal

---

## 4. 실험 결과 심층 분석

### 4.1 AliasBench (저자 제안 — 12 tasks, RoboTwin2 기반)

| Category | IntentVLA | Baseline |
|----------|-----------|----------|
| Back-and-Forth | 49.3 | – |
| Crossing-Path | 74.7 | – |
| Bimanual | 17.0 | – |
| Multi-Goal | 31.3 | – |
| **Average** | **45.8** | **9.0** |
| ICC-L2 (inter-chunk inconsistency) | **0.181** | – |

- Baseline 대비 **5배 이상**의 절대 차이 → aliasing이 기존 VLA의 핵심 실패 모드임을 입증
- ICC-L2 17.6% 감소 → 인접 chunk 간 의도 일관성이 실제로 개선됨을 수치로 보임

### 4.2 LIBERO (4 suite, single unified policy)

| Suite | IntentVLA | Baseline (Qwen3-VL frame-only) |
|-------|-----------|-------------------------------|
| Spatial | 99.3 | – |
| Object | 99.7 | – |
| Goal | 98.1 | – |
| Long | 97.4 | – |
| **Avg** | **98.6** | **96.5** |

- 모든 suite에서 99% 근접 — 특히 Long 97.4%가 강력 (multimodal aliasing이 long-horizon에서 더 큰 영향)

### 4.3 SimplerEnv (WidowX)

| Task | IntentVLA | Baseline |
|------|-----------|----------|
| Put Spoon on Towel | 70.8 | – |
| Put Carrot on Plate | 66.7 | – |
| Stack Green Block | 54.2 | – |
| Put Eggplant in Basket | 100.0 | – |
| **Avg** | **72.9** | **65.3** |

### 4.4 RoboCasa-GR1 Tabletop (24 tasks)

| Subset | IntentVLA | TwinBrainVLA |
|--------|-----------|-------------|
| PnP Close (avg) | 59.7 | – |
| Novel-from-Cuttingboard | 54.8 | – |
| Novel-from-Placemat | 55.5 | – |
| Novel-from-Tray | 49.6 | – |
| Novel-from-Plate | 66.5 | – |
| **Overall** | **57.0** | **54.6** |

---

## 5. Ablation 분석 (SimplerEnv)

| Configuration | Avg Success |
|--------------|------------|
| Frame-only Qwen3-VL-GR00T | 65.3 |
| + VGGT current frame only | 64.8 |
| + History fusion, no intent token | 69.5 |
| **Full IntentVLA** | **72.9** |

### 해석
1. **VGGT를 현재 frame에만 적용**: 64.8 — 오히려 baseline 미만. 즉 VGGT geometric token이 단일 frame에서는 큰 가치가 없음
2. **History fusion 추가(intent token 없음)**: 69.5 — history 자체로 +4.2pp
3. **Intent token까지 추가**: 72.9 — 추가 +3.4pp → "intent를 명시적 token으로 구조화"한 점이 핵심 기여

> ❓ **예상 질문**: AliasBench가 저자 self-design인데 over-fit 가능성은?
> **답변**: 핵심 우려. 동일 group이 benchmark와 model을 함께 디자인 → metric이 model strength를 강조하는 방향으로 편향 가능. ICC-L2와 같은 보조 metric 도입으로 일부 완화하지만 외부 모델로의 cross-validation이 필요.

---

## 6. 관련 연구 비교

| 모델 | History 활용 | Intent 모델링 | Action head |
|------|-------------|--------------|-------------|
| OpenVLA | ✗ | ✗ | discrete token |
| π₀ | △ (state history) | ✗ | flow-matching |
| GR-1 | ✓ (frame stack) | implicit | next-frame |
| **IntentVLA** | **✓ (VGGT register tokens)** | **✓ (explicit gated)** | **flow-matching** |

핵심 차이: **intent를 explicit token으로 분리**하고, 그 효과를 isolate 측정할 benchmark까지 함께 제공.

---

## 7. 한계 및 미해결 문제

1. **AliasBench self-design**: 동일 그룹이 benchmark와 model을 같이 제안 → over-fitting 위험. External validation 필요
2. **VGGT-1B frozen ↔ 4B Qwen3-VL trainable**: 총 5B 규모 → on-device 배포 부담. 작은 backbone에서의 ablation 부재
3. **Short-horizon의 길이 정의**: history window 크기, intent의 temporal extent를 abstract 단계에서 명확히 하지 않음 (본문 확인 필요)
4. **Long-horizon planning**: intent를 short로 한정한 설계는 multi-stage 조립 같은 long-horizon task에서는 hierarchical layer 필요
5. **Bimanual 성능 저조**: AliasBench Bimanual 17.0% — bimanual coordination에서는 intent representation이 약함을 시사
6. **CALVIN, RoboTwin v1 평가 부재**: 일부 표준 benchmark는 비교 불가

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★★ — Multimodal imitation의 본질을 "short-horizon intent"로 정형화한 첫 시도 |
| Technical depth | ★★★★☆ — VGGT + Qwen3-VL + flow-matching의 통합이 체계적 |
| Experimental rigor | ★★★★☆ — 4개 benchmark에 self-proposed AliasBench까지; 단 AliasBench self-design 우려 |
| Practical impact | ★★★☆☆ — 5B 규모, on-device 부담; 그러나 component는 plug-in 가능 |
| Writing quality | ★★★★☆ — 문제 정의가 명확 |

**강점**: "VLA 데이터는 본질적으로 multimodal"이라는 직관을 측정 가능한 phenomenon(observation aliasing)으로 정형화. ICC-L2라는 새로운 chunk-consistency metric 제안. LIBERO 98.6%는 매우 강력. **약점**: AliasBench가 self-design — community-wide adoption 필요. Bimanual 17%는 method가 모든 aliasing type을 해결하지 못함을 시사.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | AliasBench baseline 9% → 45.8%는 너무 큰 격차 — fair한가? | Baseline이 frame-conditioned only이므로 aliasing에 구조적으로 취약. 차이가 클수록 phenomenon의 존재성을 입증하지만, 더 강한 history-aware baseline 비교가 필요 |
| 2 | LIBERO 98.6% vs 다른 SOTA(예: π₀ 94.4%)와 정당 비교인가? | π₀는 다른 학습 셋업 — Open X-Embodiment 사전학습 등. IntentVLA는 5B 규모로 더 크고 unified policy 학습 → 공정 비교를 위해 동일 데이터/regime 통제 필요 |
| 3 | VGGT를 frozen으로 두는 이유? Trainable이 더 낫지 않나? | Ablation에서 VGGT current frame만은 64.8% (baseline 미만). Frozen + register token selection이 핵심. Trainable로 풀면 geometric prior 훼손 위험 |
| 4 | Intent representation의 차원/구조는? | 본문 확인 필요. Camera token + register token retention만 abstract에 명시 |
| 5 | Bimanual에서 왜 17%? | 양손 coordination은 single-stream intent로 표현하기 어려운 multi-agent 구조 — 본 method의 한계 |
| 6 | ICC-L2 metric의 정의는? | Inter-chunk consistency L2 — 인접 chunk 간 action distribution distance. 17.6% 감소는 직접적 chunk-conflict 해소 증거 |
| 7 | Flow-matching vs autoregressive discrete token의 trade-off? | Flow-matching은 continuous action space에서 multimodal posterior 표현이 자연스러움. Intent conditioning과 결합 시 mode selection이 자연스러움 |
| 8 | RoboCasa 57% — TwinBrainVLA 54.6% 차이가 미미한데 method가 정말 효과적인가? | RoboCasa는 aliasing이 약한 도메인일 수 있음. AliasBench의 5× 차이가 method가 효과적인 도메인을 보여주는 evidence |

<!-- VERIFIED: pdf -->
