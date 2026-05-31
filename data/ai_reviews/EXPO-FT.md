# EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models

> **한 줄 요약**: 사전 학습된 pi-0.5 위에 lightweight edit policy + RedQ 10-Q-network ensemble을 얹어 action chunk와 human-in-the-loop intervention까지 지원하도록 EXPO 알고리즘을 확장한 RL fine-tuning. 8개의 실로봇 manipulation task 모두에서 평균 19.1분의 online 데이터로 30/30 perfect success를 달성, HG-DAgger(22.1/30), DSRL(19/30), HIL-SERL(5.5/30)을 큰 격차로 능가.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 일반 VLA(pi-0.5 등)는 zero-shot으로 새 task에 부분적 성공만 함 — fine-tuning이 필요
- **Supervised fine-tuning (SFT)**: 데모 수집 비용 큼, distribution shift에 취약
- **RL from scratch (HIL-SERL)**: VLA의 prior 활용 못 함, 매우 sample-inefficient
- **기존 RL fine-tuning (DSRL)**: VLA의 action chunk(temporally extended action)와 호환 어려움, intervention 통합 어려움
- 결과: **VLA를 실로봇에서 빠르게 personalize하는 표준 RL 방법 부재**

### 핵심 질문
- **Action chunk를 출력하는 modern VLA를 RL로 fine-tuning하려면 어떻게 해야 하는가?**
- **Human intervention을 RL loop에 자연스럽게 통합하면 sample efficiency가 어떻게 변하는가?**

📌 [Figure 1 삽입] — EXPO-FT: pi-0.5 base + edit policy + Q-ensemble + HIL → 19분에 perfect performance

---

## 2. 방법론 심층 분석

### 2.1 EXPO 알고리즘 확장

원 EXPO는 single-step action에 대한 RL 방법. EXPO-FT의 확장:
1. **Temporally extended actions**: VLA의 action chunk를 RL action 단위로 취급
2. **Edit policy**: Base VLA action chunk를 small perturbation으로 editing
3. **Q-ensemble**: RedQ-style 10-Q networks로 value estimation
4. **HIL intervention**: Human이 실패 직전 개입 → demo로 즉시 사용

### 2.2 Base + Edit 구조

- **Base policy**: pi-0.5 — frozen 또는 천천히 update
- **Edit policy**: 작은 MLP/Transformer로 pi-0.5 output에 perturbation 추가
- 최종 action: a = base(o) + edit_scale * edit(o, base_out)
- Edit scale: 0.05~0.2 (task-dependent)

> ❓ **예상 질문**: 왜 base를 직접 RL update하지 않고 edit policy를 두는가?
> **답변**: Base VLA의 generalist capability(language understanding 등)를 보존. RL이 base의 큰 prior를 destabilize하지 않도록. Edit policy는 작아서 sample efficient.

### 2.3 RedQ Q-Ensemble

- **10개의 Q-network** ensemble (ResNet-50 visual backbone)
- Update-to-data ratio (UTD) = 20 (각 환경 step당 20번의 Q update)
- Conservative estimate를 위해 minimum of subset
- High UTD + ensemble = sample-efficient RL의 표준 trick

> ❓ **예상 질문**: 왜 10개인가? RedQ 논문은 더 큰 ensemble도 제안하는데?
> **답변**: 10개가 compute vs accuracy trade-off의 sweet spot. 더 크면 wall-clock latency 증가, 실로봇에서는 비실용적.

### 2.4 Human-in-the-Loop Intervention

- 학습 중 사람이 관찰하다가 실패 직전 개입
- Intervention trajectory는 즉시 success demo로 buffer에 추가
- Edit policy의 supervised regularization으로도 활용

> ❓ **예상 질문**: Intervention이 그냥 imitation learning과 무엇이 다른가?
> **답변**: Intervention 데이터 + Q-function bootstrapping이 결합. Imitation은 single demo만 학습, EXPO-FT는 demo를 RL의 value propagation에 활용.

---

## 3. 데이터 전략

### Online Robot Data
- 평균 **19.1분 / task**의 online interaction
- Per-task 20~30분 budget — 매우 sample efficient
- HIL intervention이 효율적인 signal source

### Base Policy
- pi-0.5 (사전 학습, generalist)
- No additional pretraining required

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Optimizer | Adam (lr 3e-4) |
| Batch size | 64 |
| UTD ratio | 20 |
| Discount | 0.99 |
| Edit scale | 0.05~0.2 (task-dependent) |
| Q networks | 10 (RedQ) |
| Base policy | pi-0.5 (frozen or slow-update) |

---

## 5. 실험 설계 및 평가 프로토콜

8개의 실로봇 manipulation task:
- Egg flip
- String light routing
- Pool shot
- Flower insertion
- (그 외 4개)

각 task에서 **30 trials** 평가 → success 30/30 = perfect

---

## 6. 실험 결과 심층 분석

### Main Results (Real-world 8 tasks, 30 trials each)

| Method | Average Success / 30 |
|--------|---------------------|
| HIL-SERL (RL from scratch) | 5.5 |
| DSRL (action chunk RL) | 19.0 |
| Supervised FT | 20.5 |
| HG-DAgger | 22.1 |
| **EXPO-FT** | **30.0** (perfect) |

- **30/30 평균은 매우 강한 주장** — 8 task 모두에서 perfect
- 19.1분 평균 학습 시간 → real-time deployment 가능

### Task-by-task variance

- 모든 task에서 perfect → between-task 분산 없음
- 학습 시간(분)이 task별로 다를 가능성 (string light routing 같은 dexterous task는 더 길 듯)

---

## 7. Ablation 분석

### Edit scale 영향

- Edit scale = 0: base policy 그대로 → fine-tuning 효과 없음
- Edit scale = 0.05~0.2: 적절
- Edit scale 큰 값: base prior destabilize

### UTD ratio

- UTD = 1: 느린 학습
- UTD = 20: sample efficient
- UTD > 20: overfitting Q-function

### HIL intervention 기여

- HIL 제거 → 19.1분 → 50분+ 필요 (recovery from failure)
- HIL이 sample efficiency의 핵심

---

## 8. 관련 연구 비교

| 방법 | Action chunk | HIL | Q-ensemble | Real-world success |
|------|-------------|-----|-----------|-------------------|
| HIL-SERL | x | ✓ | △ | low (5.5/30) |
| HG-DAgger | △ | ✓ | x | mid (22.1/30) |
| DSRL | ✓ | x | x | mid (19/30) |
| Supervised FT | ✓ | x | n/a | mid (20.5/30) |
| **EXPO-FT** | **✓** | **✓** | **✓ RedQ** | **30/30** |

### 핵심 차이
- **Action chunk + HIL + Q-ensemble의 unique combination**
- pi-0.5 prior를 적극 활용

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **8 task만 평가**: 더 광범위한 task suite(LIBERO, CALVIN 등 simulation)에서의 sample efficiency 검증 부재
2. **Perfect success(30/30)는 의심스러울 정도로 깔끔**: 변동성 어디서 나오는가? Per-task seed variation 보고 부재
3. **Edit scale은 task-dependent hyperparameter**: Real-world 배포 시 task마다 tuning 필요 — 자동화 어려움
4. **HIL의 quantification**: 평균 19.1분 중 사람의 intervention frequency/total time은? Human labor cost가 핵심 metric인데 분리 안 됨
5. **Base policy quality dependency**: pi-0.5가 약한 task에서는 어떻게 작동? pi-0.5의 prior가 너무 약하면 edit policy로 회복 불가

### Attribution 문제
- 30/30의 perfect success가 **EXPO 알고리즘**의 결과인지, **pi-0.5의 강력한 prior** 덕분인지, **HIL의 dense signal** 덕분인지 분리 어려움
- Ablation에서 HIL 제거 시 시간 ~2.5배 증가 → HIL이 핵심이지만 algorithm contribution도 분명

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — EXPO + action chunk + HIL의 결합은 새 angle |
| **Technical depth** | ★★★★☆ — RedQ ensemble + edit policy + HIL 통합 |
| **Experimental rigor** | ★★★☆☆ — 8 real-world task, 강한 main result. 하지만 simulation 없음, ablation depth 부족 |
| **Practical impact** | ★★★★★ — 19분에 perfect performance는 강력 |
| **Writing quality** | ★★★★☆ — 명확한 motivation, 실용적 detail |

**강점**: Modern VLA(pi-0.5)에 적용 가능한 **첫 strong RL fine-tuning 방법**. **19분/task의 sample efficiency**는 실로봇 배포에서 game-changer. **약점**: Simulation 벤치마크 부재로 reproducibility 어려움, edit scale tuning 필요, HIL labor 정량화 부족.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 30/30 perfect는 너무 깔끔하다 — variance는? | Per-task seed variation 미보고. 8 task 모두 동시에 perfect는 의심스러움. Stochastic eval 부재 가능성 |
| 2 | HIL labor cost를 정량화하면? | 19.1분 중 human intervention time을 분리 보고 부재. 만약 사람이 50% 시간 개입한다면 "10분 사람 시간"으로 봐야 |
| 3 | Edit scale 0.05~0.2 task-dependent tuning은 어떻게? | Paper에서 grid search 또는 manual tuning 추정. 자동 tuning 부재 — practical bottleneck |
| 4 | Pi-0.5 zero-shot success rate가 baseline인가? | 명시적 zero-shot baseline 부재. pi-0.5 + supervised FT가 baseline(20.5/30)으로 사용 |
| 5 | DSRL이 19/30인데 EXPO-FT가 30/30인 이유? | DSRL은 action chunk RL이지만 HIL 미통합 + Q-ensemble 부재. Component 차이 누적 |
| 6 | Real-world만 평가한 이유? | "VLA RL의 실용성 강조"가 motivation. 그러나 sim benchmark가 있으면 reproducibility 향상 |
| 7 | LIBERO에서 측정하면 score는? | 미보고. Pi-0.5는 LIBERO에서 ~97% — EXPO-FT가 saturation 영역에서 의미를 갖기 어려울 수 있음 |
| 8 | Edit policy가 base의 prior를 destabilize하지 않는다는 보장은? | Edit scale 0.2 이하로 제한, base는 frozen/slow-update. KL constraint 없음 — long fine-tuning 시 drift 위험 |

<!-- VERIFIED: pdf -->
