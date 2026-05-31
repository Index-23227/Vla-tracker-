# Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language

> **한 줄 요약**: pi0 backbone에 *decoupled 힘-위치(force-position) 하이브리드 command interface*와 *tactile token (TCN/MLP) 입력*을 결합하여 "gently/firmly" 같은 언어 부사에 따라 grip force를 동적으로 modulate하는 VLA — Tabero-VTLA. 시뮬레이션에서 LIBERO-Object trajectories를 force regime별로 replay해 데이터를 확보, 평균 grip force를 **70% 감소** (gentle 명령 시).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA (OpenVLA, pi0, RDT)는 *position-control*만 출력 → 물체와 환경 사이의 **접촉력 (contact force)**을 명시적으로 제어 불가
- "Gently pick up the egg" 같은 자연어 부사는 무시되고, **task success만 평가**되기 때문에 grip이 과도해 깨지기 쉬운 물체를 손상
- **Tactile data scarcity**: vision-language-action 데이터는 풍부하지만 *촉각 (tactile)* 신호를 포함한 데이터셋은 극히 부족

### 핵심 질문
- **VLA가 task success를 유지하면서 contact force를 언어로 modulate할 수 있는가?**
- **Open-source manipulation trajectories를 force-aware 시뮬레이션으로 *재활용*하여 tactile 데이터를 확장할 수 있는가?**

📌 [Figure 1 삽입] — Tabero-VTLA: vision + tactile + language 입력 → decoupled force-position output → admittance hybrid controller

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

- **Backbone**: pi0 (PaliGemma VLM + flow-matching action expert)
- **추가 입력**: 8-frame tactile history (20 Hz 샘플링), TCN 및 MLP tokenizer로 임베딩
- **출력**: end-effector pose **+ target applied force** (decoupled)
- **저수준**: admittance-based hybrid controller가 force command를 추적

### 2.2 Decoupled Force-Position Command Interface

| 출력 | 차원 | 역할 |
|------|------|------|
| End-effector pose | 6 (xyz + RPY) | 위치/자세 명령 |
| Target applied force | scalar | grip force command |
| Hybrid controller | fixed admittance | force feedback loop |

- **방정식 (5)-(8)**: admittance 기반 position correction + grip force feedback loop (feedforward gain + admittance correction)
- 핵심: **policy는 high-level command만 출력**, low-level controller가 force/position을 hybrid로 추적 → policy 학습이 단순화

> ❓ **예상 질문**: 왜 policy가 직접 force를 출력하지 않고 admittance controller에 위임하는가?
> **답변**: VLA가 contact dynamics를 implicit하게 배우려면 방대한 contact-rich 데이터가 필요. Admittance controller를 *고정 prior*로 두면 policy는 "target force"만 결정하면 됨 → sample efficiency 대폭 증가.

### 2.3 Tactile Tokenizers

- **TCN tokenizer**: temporal convolutional network로 8-frame history를 인코딩
- **MLP tokenizer**: 단순 비교 baseline
- Tactile token은 pi0의 cross-attention stream에 결합

---

## 3. 데이터 전략

### Tabero Dataset

핵심 아이디어: **기존 LIBERO Object trajectories를 force-aware tactile simulator에서 replay**

| Dataset | Reference force | 용도 |
|---------|----------------|------|
| Dataset A | 25% | 일반적 grip |
| Dataset B | 10% | 미세한 grip (gentle) |

- LIBERO Object의 9개 task를 두 가지 force regime으로 replay
- 언어 instruction에 "firmly/tightly" 또는 "gently/softly" 부사 부착
- **Real-world tactile data 0개** — 모든 데이터가 high-fidelity 시뮬레이션 기반

> ❓ **예상 질문**: 시뮬레이션 tactile data만으로 sim-to-real transfer가 가능한가?
> **답변**: 논문은 real-robot 실험을 수행하지 않음. 저자도 한계로 인정 — "We are also developing a real-world force-position hybrid data collection system." 따라서 *sim-only* 결과로 보아야 함.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base model | pi0 |
| Fine-tune method | LoRA |
| Steps | 50,000 |
| Batch size | 32 |
| Peak LR | 2.5e-5 |
| Tactile sampling | 20 Hz, 8-frame history |
| Hardware | 미공개 |

---

## 5. 실험 설계 및 평가 프로토콜

### 5.1 평가 메트릭 (혁신점)

기존 LIBERO는 **task success만** 평가하지만, Tabero는 두 가지를 동시에:
1. **SR (Success Rate)**: 기존 LIBERO 표준
2. **AG (Average Grip force, N)**: 접촉 시 평균 grip force — *gentleness* 측정

### 5.2 데이터셋 분할

- 9-task Tabero-Object subset (LIBERO Object의 9개) — *표준 LIBERO와 다름*
- 추가 4-task cross-platform validation

---

## 6. 실험 결과 심층 분석

### 6.1 Force Modulation (Tables 1-3)

| Dataset | 명령 | SR | AG (N) |
|---------|------|-----|--------|
| A (25% force) | firm | **0.87** | 31.3 |
| A (25% force) | gentle | 0.79 | **8.5** |
| B (10% force) | firm | 0.86 | 32.4 |
| B (10% force) | gentle | 0.52 | **3.7** |

- Dataset A에서 gentle 명령 시 SR 0.79 유지하면서 grip force가 31.3N → 8.5N (**-73%**)
- Dataset B에서는 SR이 0.86 → 0.52로 크게 감소 — 매우 낮은 force (3.7N)는 task success를 희생

### 6.2 Semantic Generalization (Table 4)

- "firmly/tightly" 같은 새 부사 → SR 0.86, AG 32.4N (firm 동작)
- "gently/softly" 같은 새 부사 → SR 0.52, AG 3.7N (gentle 동작)
- **언어 부사 일반화**에 성공 — pretrained PaliGemma의 semantic 표현이 grip force와 연결됨

### ⚠️ Real-world / 표준 LIBERO 부재

- **표준 LIBERO 4-suite 평가 없음** — 9-task subset만 보고
- **Real robot 실험 없음** — 모든 결과가 sim-only
- 이는 leaderboard에서 직접적 비교가 불가능함을 의미

---

## 7. Ablation 분석

### Tactile Tokenizer 비교
- TCN (temporal conv) vs MLP — TCN이 약간 우세 (정확한 수치는 본문 깊이 확인 필요)
- Tactile 입력 자체를 제거한 ablation의 명확한 보고는 v1에서 부족

---

## 8. 관련 연구 비교

| 모델 | Tactile 입력 | Force output | Gentle 명령 | Real robot |
|------|-------------|-------------|------------|-----------|
| OpenVLA | ✗ | ✗ | ✗ | ✓ |
| pi0 | ✗ | ✗ | ✗ | ✓ |
| TVL (Tactile-VLA prior work) | ✓ | ✗ | ✗ | △ |
| **Tabero-VTLA** | **✓** | **✓** | **✓** | **✗ (sim-only)** |

### 핵심 차이
- Force output을 *언어 부사로 control*하는 첫 large-scale 시도
- 그러나 real-world 검증 부재 — sim-to-real gap이 unknown

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Real-world 실험 전무**: 모든 결과가 high-fidelity tactile simulator 기반. Sim-to-real gap이 검증되지 않음 — 저자도 future work로 인정
2. **표준 LIBERO 미보고**: 9-task subset만 보고하므로 leaderboard 비교 불가. cross-platform validation의 4 subtask 결과는 있지만 standard suite 표준화 부재
3. **Dataset B의 SR 0.52**: 10% reference force에서 gentle 명령 시 task success가 절반 수준 — **gentle vs success trade-off가 명확**
4. **Admittance controller 의존**: low-level controller가 fixed prior. 다른 robot platform (different inertia, different gripper)에서 직접 transfer하려면 controller tuning 재실시 필요
5. **Tactile sensor type 미공개**: 어떤 tactile sensor를 모사했는지 (Digit, GelSight, vision-based, force/torque) 명확하지 않음
6. **Author affiliation 미기재**: arXiv preprint에 institutional affiliation이 누락 — peer review에서 보강 필요

### Attribution 문제
- 70% grip force 감소가 (a) decoupled force-position interface, (b) admittance controller, (c) tactile token 입력 중 무엇 덕분인가?
- Tactile input 없이 force command만 학습하는 ablation이 핵심

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 언어 부사 → grip force modulation의 첫 명시적 처리 |
| **Technical depth** | ★★★☆☆ — pi0 + tactile + admittance의 합리적 조합 |
| **Experimental rigor** | ★★☆☆☆ — Sim-only, 표준 LIBERO 미보고, ablation 부족 |
| **Practical impact** | ★★☆☆☆ — Real robot 검증 없이는 깨지기 쉬운 물체 manipulation의 실질적 효용 미증명 |
| **Writing quality** | ★★★☆☆ — affiliation 누락, 일부 architecture detail 부족 |

**강점**: VLA의 "task success only" 평가에서 벗어나 *gentleness*를 정량화한 multidimensional 평가 protocol. Open-source trajectory를 force-aware로 replay하는 데이터 생성 파이프라인이 재사용 가능.

**약점**: Real-world 검증 부재가 가장 큰 약점. Sim-to-real gap이 검증되지 않은 상태에서 "gentle manipulation"의 실질적 가치를 주장하기에는 근거 부족. 또한 표준 LIBERO를 보고하지 않아 leaderboard에서 직접 비교 불가.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 표준 LIBERO 4-suite (Spatial/Object/Goal/Long) 점수는 왜 없는가? | 9-task subset만 보고. Force-aware tactile data를 만들 수 있는 task가 LIBERO Object에 국한됨 |
| 2 | Sim-to-real transfer 검증 없이 force modulation 주장이 타당한가? | 저자도 한계 인정. 시뮬레이터 tactile signal과 real sensor의 distribution gap이 가장 큰 우려 |
| 3 | Dataset B gentle 시 SR 0.52는 사용 가능한 수준인가? | 절반이 실패 — gentle은 가능하지만 *useful* gentle인지 의문. 실제 fragile object manipulation에서는 깨지지 않아도 task 완수 못함 |
| 4 | 어떤 tactile sensor를 모사했는가? | 논문에 명시 부족. 결과의 sensor-agnostic 여부 불명 |
| 5 | Admittance controller의 inertia/stiffness parameter는 어떻게 결정되나? | fixed prior로 보이나, robot platform 변경 시 재튜닝 필요 → transfer cost 무시할 수 없음 |
| 6 | "firmly" vs "gently"의 명령이 binary 결과만 만드는가, gradient가 가능한가? | Table 4의 두 cluster만 보고 — 중간 단계 ("moderately")의 force는 검증 안 됨 |
| 7 | pi0 backbone 동결인가 LoRA fine-tune인가? | LoRA fine-tune (50k steps, bs 32, lr 2.5e-5) — backbone weight 대부분 보존 |
| 8 | Tactile token 없이 force command만으로 학습한 baseline은 어떤가? | 명확한 ablation이 보고되지 않음 — tactile feedback의 *necessity* 검증 부족 |

<!-- VERIFIED: pdf -->
