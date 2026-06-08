# DriveMA: Driving Vision-Language-Action Models with verifiable Meta-Actions

> **한 줄 요약**: Autonomous driving VLA에서 **언어-trajectory gap**을 해결하기 위해 longitudinal(stop/decel/keep/accel) + lateral(turning/lane-change) **verifiable meta-action**을 중간 인터페이스로 도입하고, trajectory-grounded auto-labeling → action-centric pretraining → meta-action conditioned SFT → **turn-level credit assignment GRPO RL** 의 4단 학습으로 Qwen3.5-2B/4B 기반에서 WOD-E2E RFS 8.060 / 8.079 (RAP 8.043 능가) 달성.

---

## 1. 배경 및 동기

### 기존 driving VLA의 한계
- LLM-driven planner (DriveGPT-4, GPT-Driver, LMDrive 등)는 자연어 reasoning과 trajectory를 직접 연결 → 두 modality 간 정합성(consistency) 검증 불가
- 단순 textual rationale → trajectory의 mapping이 **verifiable** 하지 않음 → 환각(hallucination) 위험
- RL을 적용해도 trajectory만으로 scalar reward를 산정하면 reasoning step과 planning step의 credit이 섞임

### 핵심 질문
- **언어와 trajectory 사이에 verifiable 한 intermediate symbol**이 존재하는가?
- 이 symbol을 **자동 labeling**할 수 있는가?
- Reasoning step과 planning step에 **분리된 reward**를 줄 수 있는가?

📌 [Figure 1 삽입] — DriveMA 파이프라인: 입력 x → meta-action m (decision turn) → trajectory tau (planning turn)

---

## 2. 방법론 심층 분석

### 2.1 Meta-Action 정의

**Longitudinal**: stop / decelerate / keep / accelerate
**Lateral**: keep / left turn / right turn / left lane change / right lane change 등

**Verifiability**: trajectory waypoint tau로부터 rule-based projection으로 m을 재산출 가능 → "tau는 m과 consistent한가?"를 binary로 판정 가능

### 2.2 4단계 학습 파이프라인

#### Stage 1 — Trajectory-grounded Meta-Action Labeling (자동 라벨링)
- 미래 5초 expert trajectory를 분석하여 자동으로 longitudinal/lateral meta-action 부여
- 가속도 / 곡률 / lateral offset thresholding 기반
- 결과: 77K planning sample이 (image, language, meta-action, trajectory)로 quadruple 구성

#### Stage 2 — Action-Centric Pretraining
- 240K driving VQA 샘플로 backbone (Qwen3.5)에 driving domain knowledge + meta-action vocabulary 주입
- 일반 driving QA, action-centric QA(현재 행동 식별, 미래 행동 예측) 등 포함

#### Stage 3 — Meta-Action-Conditioned SFT
- 입력 x → output (m, tau) 시퀀스를 autoregressive decoding으로 학습
- m을 먼저 emit한 뒤, m을 condition으로 tau를 emit → planning을 "decision then act"로 factorize

#### Stage 4 — Turn-Level Credit Assignment GRPO RL
- Multi-turn rollout: **decision turn** (m emit) + **planning turn** (tau emit)
- 각 turn에 **별도 reward**:
  - Decision turn: R_cons (m이 결과 tau와 consistent한가?)
  - Planning turn: R_traj (trajectory quality: ADE / collision-free / smoothness)
- Turn-level **advantage normalization** → 한 turn의 advantage가 다른 turn에 leak 되지 않음

### 2.3 Reward 설계

```
R_total = R_traj + alpha * R_cons
```

- R_traj: trajectory ADE, collision metric 등 dense signal
- R_cons: meta-action m과 실제 tau의 rule-based projection 일치도 — verifiability를 활용한 dense language-action alignment reward

> ❓ **예상 질문**: R_cons는 R_traj와 redundant 하지 않은가?
> **답변**: R_cons는 reasoning(언어 출력 m)의 consistency, R_traj는 motion quality. ablation에서 consistency가 88.50% → 98.80%로 급증 (즉 R_traj만으로는 reasoning fidelity가 보장되지 않음)

### 2.4 Turn-level GRPO의 의미

- vanilla GRPO: 전체 trajectory를 단일 sequence로 보고 advantage 평균
- DriveMA: decision turn과 planning turn의 advantage를 **분리 normalize** → reasoning step의 bad output이 planning step의 좋은 output에 의해 가려지지 않음
- 결과: vanilla GRPO 7.978 → turn-level 8.060 (+0.082 RFS)

---

## 3. 데이터 전략

| 데이터 종류 | 규모 | 용도 |
|------------|------|------|
| Driving VQA | 240K | Stage 2 (action-centric pretraining) |
| Planning samples | 77K | Stage 3 (SFT) |
| Preference samples | 479 | RL preference shaping |
| Trajectory horizon | 5s @ 1Hz (Waymo) / 4s (NAVSIM) | label space |

---

## 4. 시스템/학습 세부사항

| 단계 | Epochs/Steps | LR | Batch | 기타 |
|------|-------------|-----|-------|------|
| SFT | 1 epoch | 1e-5 | 64 | full-parameter FT |
| RL (turn-level GRPO) | 600 steps | 1e-6 | - | 8 generations/sample, KL beta=0.4 |

- Hardware: 8 x NVIDIA A800
- Backbone: Qwen3.5-2B / Qwen3.5-4B

---

## 5. 실험 결과 심층 분석

### 5.1 WOD-E2E (Waymo Open Dataset End-to-End)

| 모델 | RFS Overall ↑ | RFS Spotlight ↑ | ADE@5s ↓ |
|------|---------------|----------------|----------|
| RAP (prev SOTA) | 8.043 | 7.204 | 2.646 |
| **DriveMA-2B** | **8.060** | **7.251** | **2.616** |
| **DriveMA-4B** | **8.079** | 7.169 | 2.670 |

- WOD-E2E에서 **SOTA 갱신** — 2B 변종도 RAP 능가
- RFS Spotlight(어려운 케이스)에서 2B가 4B보다 좋음 — 과적합 가능성 시사

### 5.2 NAVSIM

| 모델 | PDMS ↑ |
|------|--------|
| RAP | **93.8** |
| DriveMA-4B | 91.2 |
| DriveMA-2B | 90.5 |

- **NAVSIM에서는 RAP에 뒤짐** (-2.6) — paper에서 인정. WOD-E2E와 NAVSIM의 평가 metric / scenario 차이로 인한 trade-off

### 5.3 Consistency

| 모델 | Language-Action Consistency |
|------|---------------------------|
| Vanilla SFT | 88.50% |
| + R_cons | **98.80%** |

- consistency reward의 dramatic 효과 → meta-action 도입의 핵심 정당성

---

## 6. Ablation 분석

| 설정 | WOD-E2E RFS |
|------|-------------|
| baseline VLM SFT only | 7.741 |
| + meta-action SFT | 7.804 (+0.063) |
| + action-centric pretraining | 7.893 (+0.089) |
| + vanilla GRPO RL | 7.978 (+0.085) |
| **+ turn-level credit assignment** | **8.060 (+0.082)** |

- **각 단계가 동등하게 기여** — 어느 하나도 dominant하지 않음, **전체 파이프라인의 합 효과**
- turn-level credit assignment는 vanilla GRPO 대비 +0.082 — 작아 보이지만 RFS 8점대에서는 결정적인 SOTA 갱신폭

---

## 7. 관련 연구 비교

| 모델 | Intermediate | Verifiability | RL credit |
|------|-------------|---------------|-----------|
| GPT-Driver | textual rationale | ✗ | - |
| DriveLM | scene graph | partial | - |
| LMDrive | language plan | ✗ | - |
| RAP | reasoning chains | partial | trajectory-level |
| **DriveMA** | **verifiable meta-action** | **✓ (rule-based)** | **turn-level (decision/planning 분리)** |

핵심 차이: (1) meta-action의 rule-based verifiability, (2) turn-level credit assignment로 reasoning과 planning의 credit 분리

---

## 8. 한계 및 미해결 문제

1. **NAVSIM 성능 부족**: PDMS 91.2 vs RAP 93.8 — 모든 driving benchmark에서 SOTA가 아닌, WOD-E2E specific 강세
2. **Meta-action vocabulary의 표현력**: longitudinal 4종 + lateral ~5종은 정상 주행에는 충분하나, urgent avoidance / parking 등 nuanced 행동 표현 한계
3. **자동 labeling 노이즈**: threshold 기반 meta-action 추출의 정확도가 명시되지 않음 — borderline trajectory의 labeling 신뢰도
4. **2B가 4B를 능가하는 sub-metric**: RFS Spotlight 2B 7.251 > 4B 7.169 — 4B가 oversize일 가능성
5. **Closed-loop 평가 부재**: WOD-E2E와 NAVSIM 모두 offline log-replay metric. CARLA / actual closed-loop simulation 부재
6. **Code unreleased**: "will be released" — reproducibility 미확정
7. **Manipulation 평가 0**: 본 모델은 driving-only specialized, manipulation VLA로의 transferability 미평가

### Attribution 우려
- ablation에서 각 단계가 비슷한 크기의 기여를 함 → "meta-action 도입의 본질적 가치"가 turn-level credit assignment 없이는 SOTA에 못 미친다는 의미. meta-action만으로는 +0.063, 전체 파이프라인 합쳐야 SOTA — 개별 component 가치 evaluation은 다소 모호

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — verifiable meta-action + turn-level credit assignment의 결합이 새로움 |
| **Technical depth** | ★★★★★ — 자동 labeling, multi-stage SFT, turn-level GRPO 등 정교한 파이프라인 |
| **Experimental rigor** | ★★★★☆ — ablation 단계별, consistency metric까지 추적 |
| **Practical impact** | ★★★★☆ — WOD-E2E SOTA, but NAVSIM 뒤처짐, code 미공개 |
| **Writing quality** | ★★★★☆ — 파이프라인이 명확 |

**강점**: "rule-based verifiable intermediate"라는 단순하지만 강력한 idea. consistency reward로 LLM의 reasoning hallucination을 dramatically 해소. **약점**: NAVSIM 부진과 closed-loop 평가 부재로 인해 production deployment 신뢰성은 미해결.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | meta-action이 충분히 expressive 한가? | longitudinal 4종 + lateral ~5종은 highway/urban driving 정상 상황에 충분. 응급 회피, 후진, parking 등은 미커버 |
| 2 | 자동 labeling의 false rate는? | threshold-based이므로 borderline (e.g., slight curve = keep or turn?) 케이스에 노이즈. 정량 보고 부재 |
| 3 | turn-level GRPO와 vanilla GRPO의 차이가 +0.082인데 의미 있는가? | RFS 8점대에서 RAP를 +0.017 능가하려면 결정적. 통계 신뢰도 보고는 부재하나 SOTA 갱신폭 |
| 4 | NAVSIM에서 왜 뒤처지나? | WOD-E2E는 RFS (route fidelity), NAVSIM은 PDMS (collision / drivable area 비중 큼). meta-action이 safety constraint를 직접 다루지 않아 PDMS에 불리 가능 |
| 5 | 2B > 4B인 sub-metric은 over-parameterization의 sign 인가? | RFS Spotlight에서 2B(7.251) > 4B(7.169). 어려운 케이스에서 4B 오버피팅 의심 |
| 6 | Manipulation VLA로 transferable 한가? | meta-action 정의 자체가 driving-specific. Manipulation은 longitudinal/lateral analog가 없어 직접 transfer 어려움 |
| 7 | Closed-loop 평가가 없는데 deployment risk는? | offline log replay만으로는 covariate shift 확인 불가. CARLA / nuPlan closed-loop가 필요한 핵심 미해결 평가 |
| 8 | KL beta=0.4의 의미와 sensitivity는? | 강한 KL anchoring → behavior policy에서 너무 멀어지지 않게 함. 0.4는 RLHF 통상값(0.04-0.1) 대비 매우 큼 — meta-action의 grammar drift 방지 의도로 추정 |

<!-- VERIFIED: pdf -->
