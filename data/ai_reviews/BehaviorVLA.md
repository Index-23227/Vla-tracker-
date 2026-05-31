# BehaviorVLA: From Abstraction to Instantiation — Learning Behavioral Representation for Vision-Language-Action Model

> **한 줄 요약**: π₀.₅ 위에 long-horizon trajectory를 압축한 *behavioral representation*을 학습하는 Visuomotor Behavior Encoder(VBE, three-stream causal Mamba) + 이를 phase 조건부 flow-matching으로 실행 가능한 행동으로 재구체화하는 Phase-conditioned Behavior Decoder(PBD)로 구성된 데이터 효율적 VLA. LIBERO 4-suite 평균 98.0%, CALVIN ABC→D 4.36, RoboTwin 2.0 Hard 58.0%, 실제 GALAXEA R1 Lite long-horizon에서 π₀.₅ 대비 +15%.

---

## 1. 배경 및 동기

### "직접 회귀(direct regression)"의 한계
- 기존 VLA는 observation → action을 직접 회귀(또는 flow-matching) — 그러나 *행동의 시간적 구조*가 매 step마다 사라지고 다시 추론됨
- Long-horizon, distribution shift, 적은 demo 조건에서 직접 회귀는 **temporal incoherence**(연속 step 간 의도 불일치)와 **데이터 비효율**을 보임
- π₀.₅, OpenVLA-OFT 같은 SOTA도 50-100 demo 미만 real-world에서는 성능이 급락

### 핵심 질문
- **추상화(behavioral prior) → 구체화(executable action)**의 2-stage 표현이 직접 회귀보다 데이터 효율과 robustness에서 우월한가?
- Long-horizon 행동을 어떻게 "phase"로 분해하고 조건부 생성에 반영할 것인가?

📌 [Figure 1] — VBE(3-stream Mamba)로 trajectory를 behavioral token으로 압축, PBD가 phase-conditioned flow-matching predictor-corrector로 action chunk 생성

---

## 2. 방법론 심층 분석

### 2.1 두 단계 표현 학습

| 단계 | 모듈 | 역할 |
|------|------|------|
| Abstraction | VBE (Visuomotor Behavior Encoder) | (vision / action / behavior) 3-stream causal Mamba가 trajectory를 behavioral prior token으로 압축 |
| Instantiation | PBD (Phase-conditioned Behavior Decoder) | behavioral prior + 현재 phase 임베딩으로 flow-matching predictor-corrector가 action chunk 생성 |

### 2.2 Causal Mamba 3-stream의 의도
- **Vision stream**: 이미지 토큰 시계열
- **Action stream**: 과거 action 시퀀스
- **Behavior stream**: 학습되는 latent behavioral token sequence
- 세 stream이 causal selective state space로 융합되어, transformer 대비 long-context에서 메모리 효율적 ↑

### 2.3 Predictor-Corrector flow matching
- 표준 flow matching은 *predictor*만 사용 (Euler)
- BehaviorVLA의 PBD는 *corrector* step을 추가하여 flow의 누적 오차를 보정 — 특히 long-horizon trajectory의 안정성 향상

> ❓ **예상 질문**: 3-stream Mamba가 왜 transformer보다 나은가?
> **답변**: Long-horizon trajectory는 수백 step에 달하므로 transformer의 quadratic attention이 메모리 병목. Mamba의 linear-time selective scan이 100-500 step trajectory를 통째로 처리 가능.

> ❓ **예상 질문**: "Phase"는 어떻게 정의되는가?
> **답변**: 논문은 phase를 동적 task-level prior alignment로 표현 — 명시적 segmentation이 아니라 trajectory 내의 진행도(progress)를 학습된 임베딩으로 표현. 명확한 phase 수나 정의식은 부재.

---

## 3. 데이터 전략

| 데이터 | 규모 | 용도 |
|--------|------|------|
| LIBERO | 4 suites | Sim 평가 |
| CALVIN | ABC→D split | Long-horizon sim 평가 |
| RoboTwin 2.0 | Hard setting, 10 tasks | Bimanual sim 평가 |
| Real GALAXEA R1 Lite | 8 tasks, 100-200 demos/task | Real 평가 |

- Pretraining 데이터셋 명시 없음 — π₀.₅ pretraining을 그대로 활용한 것으로 추정

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO 4-suite

| Suite | Score |
|-------|-------|
| Spatial | 99.2 |
| Object | 99.4 |
| Goal | 98.8 |
| Long | 94.6 |
| **Avg** | **98.0** |

- π₀.₅(96.9% 추정) 대비 +1.1%p — LIBERO는 이미 saturate되어 격차가 작음
- Long suite(94.6%)에서의 강세가 *behavioral representation*의 long-horizon 강점을 시사

### 4.2 RoboTwin 2.0 Hard (10 tasks)

| 모델 | 평균 |
|------|------|
| RDT | ~20% (추정) |
| BehaviorVLA | **58%** |

- +37.7%p over RDT — 가장 큰 향상 폭이 보고된 결과
- 단 태스크별 25-90% 범위로 분산이 큼 → 특정 태스크 카테고리에 강세

### 4.3 Real World (GALAXEA R1 Lite, 8 tasks)

| 카테고리 | 평균 |
|----------|------|
| Generalization tasks | 70% |
| Long-horizon tasks | 55% |

- π₀.₅ 대비 long-horizon에서 +15%p — 가장 인상적인 결과
- OpenVLA-OFT와 비교 시 **50% 적은 demo**로 comparable 성능 → 데이터 효율 주장의 근거

### 4.4 CALVIN ABC→D
- 평균 sequence length **4.36** — VLM4VLA(4.03), FLOWER(4.54) 대비 중간 위치
- 핵심 SOTA(Qwen-VLA 등) 직접 비교 없음 — 비교가 selective

---

## 5. Ablation 분석
- 정량 ablation table은 본문에 부재(부록 언급만 있음)
- VBE 없이 PBD만? Mamba 대신 transformer? 등 핵심 모듈 기여도 검증 부족

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **Phase 정의의 모호성**: phase가 학습 임베딩으로만 정의되어 *해석 가능성* 부족 → 정책이 왜 특정 phase에서 특정 행동을 하는지 분석 어려움
2. **Pretraining 데이터 미공개**: π₀.₅를 어떻게 활용했는지(frozen vs continued pretraining) 불분명
3. **Code 미공개**: project page만 있고 GitHub URL 부재
4. **Baseline의 selective 비교**: RoboTwin에서 RDT만 비교, Qwen-VLA 87.2(Hard) 등 더 강한 baseline 부재

### Attribution 문제
- LIBERO +1.1%p는 saturation 영역의 작은 차이 — Behavioral representation의 효과를 분리하기 어려움
- Real world +15%p는 인상적이나 baseline π₀.₅가 같은 demo로 학습됐는지 control 부족

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 3-stream causal Mamba로 behavioral prior를 학습하는 구도가 신선 |
| **Technical depth** | ★★★★☆ — VBE+PBD 2-stage가 체계적 |
| **Experimental rigor** | ★★★☆☆ — Ablation/세부 baseline 부족, CALVIN 수치 본문에 없음 |
| **Practical impact** | ★★★☆☆ — Code 미공개, real demo 효율은 매력적 |
| **Writing quality** | ★★★★☆ |

**강점**: π₀.₅ 위에 Mamba 기반 behavioral representation을 도입해 real-world long-horizon에서 +15%p, OpenVLA-OFT 대비 50% 적은 demo. **약점**: Phase의 의미 불명료, ablation 부재, code 미공개.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Phase는 명시적 단계인가, 학습 임베딩인가? | 학습 임베딩. 명시적 phase 수도 정의식도 부재 |
| 2 | VBE 없이 PBD만 쓰면? | Ablation 부재. 본 모듈의 필수성 미증명 |
| 3 | LIBERO에서 +1.1%p가 의미 있는가? | LIBERO는 saturate(98% 영역) — 통계적 유의성 약함. Long-suite(94.6) 강세가 더 중요 |
| 4 | Qwen-VLA(LIBERO 97.9, RoboTwin 87.2)와 비교하면? | 직접 비교 없음. LIBERO는 BehaviorVLA(98.0)가 약간 우위, RoboTwin Hard는 Qwen-VLA(87.2)가 훨씬 우위 |
| 5 | Predictor-Corrector flow matching의 추가 비용은? | 추론 step 수가 늘어 latency 증가 가능. 수치 미보고 |
| 6 | CALVIN 4.36이 어떤 split인가? | ABC→D로 추정되나 부록 참조 — 본문에 없는 점이 약점 |
| 7 | 데이터 효율 주장(50% demo)의 control은? | π₀.₅를 같은 50% demo로 fine-tune한 control 미명시 |
| 8 | 3-stream Mamba의 메모리 footprint는? | 미보고. Linear scan이 transformer 대비 효율적이라는 주장은 정성적 |

<!-- VERIFIED: pdf -->
