# FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies

> **한 줄 요약**: Qwen3.5-VL-4B 백본 위에서 *세부 속성(자세·색·접근 방향) 수준의 fine-grained instruction*과 raw goal-level instruction을 controlled mixing 비율로 학습시켜, VLA 정책의 steerability(지시 가능성)를 정량적으로 끌어올린 XLANG Lab + Qwen Team의 후처리 학습 레시피 및 데이터셋 연구.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA(OpenVLA, π0, GR00T-N1)들은 "pick the red bowl"처럼 **goal-level (raw) instruction**으로만 학습 → 같은 goal을 다양한 방식(저속/고속, 위에서/옆에서, 색상 변형)으로 수행하도록 *지시*하기 어려움
- Real-world 배포에서는 "서서히 / 손잡이 쪽으로 / 빨간 쪽을 잡고" 같은 *fine-grained attribute steering*이 빈번
- Fine-grained 명령으로만 학습하면 raw goal language 일반화가 무너질 수 있음 → mixing 비율에 대한 체계적 연구 부재

### 핵심 질문
- **Fine-grained 속성 수준 instruction(pose, color, approach direction 등)을 라벨링해 학습에 섞으면 raw goal 성공률이 떨어지지 않으면서 steerability가 얻어지는가?**
- **Fine-grained:Raw mixing 비율의 최적점은?**

📌 [Figure 1] — FineVLA-Data 구조: 같은 trajectory에 raw goal + 속성별 fine-grained label

---

## 2. 방법론 심층 분석

### 2.1 StarVLA 아키텍처 베이스

FineVLA는 **StarVLA**라는 공용 VLA 골격에서 두 변형을 사용:
| 변형 | 백본 | Action Head |
|------|-----|-------------|
| StarVLA-OFT | Qwen3.5-VL-4B | predefined action token의 hidden state를 읽는 lightweight **MLP regression head** |
| StarVLA-GR00T | Qwen3.5-VL-4B (System 2) | **DiT-based flow-matching** module (System 1) — GR00T dual-system 디자인 |

> ❓ **예상 질문**: OFT(regression head)와 GR00T(flow matching) 두 변형의 trade-off는?
> **답변**: OFT는 1-step inference로 가장 빠르고, multi-mode action 분포가 약한 단점이 있음. GR00T는 ODE step이 필요해 느리지만 multimodal action에 강함. 본 논문은 두 backbone 모두에서 일관된 mixing 효과를 보여 *방법론(레시피)의 일반성*을 강조한다.

### 2.2 Fine-Grained Instruction Schema

각 trajectory에 raw goal과 다음 속성을 동시 라벨링:
- **Pose**: end-effector 자세 (예: "approach from the left")
- **Color**: 대상 색상 (예: "the red apple")
- **Approach direction**: 접근 방향
- **Speed / Trajectory style**: 동작 방식
- 기타 execution-sensitive 속성

총 47,159 trajectory가 human-verified fine-grained label을 갖는다.

### 2.3 Controlled Mixing Recipe

학습 시 fine-grained와 raw goal 명령을 batch-level에서 비율로 섞음:
- Mixing ratio $r = \text{FG} : \text{Raw}$
- 후보: 1:0 (FG only), 1:1, 1:2, 1:4, 0:1 (Raw only)
- 같은 trajectory에 어떤 label을 줄지 *확률적*으로 결정

> ❓ **예상 질문**: 같은 trajectory의 다른 label로 학습한다는 것은 결국 1 trajectory를 여러 번 보는 것 아닌가, data leakage 우려?
> **답변**: 일반적 multi-instruction augmentation과 같은 framework. Goal-conditioned policy 학습에서 정당화됨. 다만 validation/test trajectory가 train과 별도 split이라는 점이 중요(논문 부록에 명시).

---

## 3. 데이터셋

### 3.1 Pretraining Pool
- **972,247 trajectories**, **85,000 tasks**, **10 open-source robot datasets**의 통합 corpus
- Open X-Embodiment 계열 및 시뮬레이션 데이터

### 3.2 FineVLA-Data
- **47,159 fine-grained trajectories** (human-verified)
- 각 trajectory마다 속성별 label과 raw goal 둘 다 보유

> ❓ **예상 질문**: 47K fine-grained 샘플이 972K 사전학습에 비해 0.5%도 안 되는데 어떻게 큰 효과를 내는가?
> **답변**: Mixing 시 fine-grained sampling 확률이 비율에 따라 boost됨(예: 1:1 ratio에서 batch의 절반). 절대 양이 아니라 *학습 중 노출 빈도*가 결정적. Curriculum / replay 효과로 해석 가능.

---

## 4. 실험 결과

### 4.1 RoboTwin 시뮬레이션 (Table 4, AlohaMix-OFT, FG:Raw = 1:1)

| Setting | Easy | Hard |
|---------|------|------|
| **AlohaMix-OFT (FG:Raw 1:1)** | **86.8** | **82.5** |

세 가지 학습 setup(RDT-OFT, RDT-GR00T, AlohaMix-OFT)을 다양한 mixing ratio에서 측정. AlohaMix-OFT 1:1이 최고치.

### 4.2 Real-world Dual-arm Manipulation

| Condition | Goal Success |
|-----------|-------------|
| Raw-only baseline | 49.9 |
| **FineVLA (fine-grained + raw mix)** | **62.7 / 100** |

(62.7 / 100은 본문에서 *goal success / fine-grained instruction adherence*를 함께 보고하는 형태로 추정됨)

### 4.3 Steerability 향상 (real-world per-factor gain)

| Factor | Gain over raw-only |
|--------|-------------------|
| Pose | **+23** |
| Color | **+18** |
| Approach direction | **+18** |

> ❓ **예상 질문**: Color나 pose는 raw goal에도 들어 있을 텐데 왜 fine-grained label이 +18~+23을 더 주는가?
> **답변**: Raw goal "pick the red bowl"은 *bowl 색상*만 명시할 뿐, *grasp pose의 시각적 변동*은 명시하지 않음. Fine-grained label은 execution detail까지 자연어로 명세하므로 policy가 "어떻게 잡을지"의 *조건부 분포*를 학습. 이는 단순 success ≠ steerability라는 분리를 정량화.

---

## 5. 어블레이션

### 5.1 Mixing Ratio의 Inverted-U 곡선

| FG:Raw 비율 | RoboTwin 평균 성공 |
|------------|-----|
| 1:0 (FG only) | 낮음 — raw 일반화 무너짐 |
| 1:1 | **peak** |
| 1:2 | peak 근처 |
| 1:4 | 중간 |
| 0:1 (Raw only) | baseline |

핵심: **inverted-U** 곡선 — fine-grained가 너무 많으면 raw 성능 떨어지고, 너무 적으면 steerability 없음. 최적점은 1:2 ~ 1:1.

### 5.2 핵심 발견 3가지
1. **Fine-grained supervision improves goal success**: +1.4 ~ +8.1 points
2. **Inverted-U mixing optimum** at FG:Raw 1:2 ~ 1:1
3. **Largest gains on execution-sensitive attributes** (pose +23, color +18, approach +18)

> ❓ **예상 질문**: 1:1과 1:2 사이의 평탄대(plateau)는 통계적으로 유의한가?
> **답변**: 논문은 multiple seed 평균을 보고하지만 95% CI는 명시하지 않음. 다양한 backbone(OFT/GR00T)에서 동일 패턴이 재현됨이 정성적 robustness 근거.

---

## 6. 한계 및 미해결 문제

1. **Parameter count 미명시**: FineVLA 자체 전체 파라미터 수(VLM + head)가 본문에 명확하지 않음 (Qwen3.5-VL-4B 클래스로 추정)
2. **Human verification 비용**: 47K fine-grained label이 *human-verified*인데 시간/비용 보고 없음 → scale-up 비용 불명
3. **Real-world subset 작음**: dual-arm 평가가 few-task에 한정 (정확한 task 수 명시되지 않음)
4. **Steerability metric 정의 불완전**: 본문은 "factor gain" 수치를 제시하지만 측정 프로토콜(rubric, blind eval)이 부록에서 충분히 풀려있지 않음
5. **Coverage**: pose/color/approach 외 다른 속성(force, speed, trajectory style)에 대한 정량 결과는 제한적

### Attribution
- 향상이 *fine-grained label* 그 자체 때문인지, *additional supervision signal*(더 많은 텍스트 노출)인지를 완전히 분리하지 못함
- Raw instruction을 paraphrase로 augment한 baseline과의 비교가 핵심인데 어블레이션이 약함

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — fine-grained instruction tuning은 NLP에서 알려진 아이디어. VLA에 체계적으로 적용한 점이 기여 |
| **Technical depth** | ★★★☆☆ — 두 backbone(OFT/GR00T)에서 일관성 검증이 좋음 |
| **Experimental rigor** | ★★★★☆ — mixing ratio sweep, real-world per-factor breakdown이 체계적 |
| **Practical impact** | ★★★★☆ — *steerability* 개념을 정량화. 실제 배포 시 user-controllable policy 의 토대 |
| **Writing quality** | ★★★★☆ — XLANG 표준의 깔끔한 분석 |

**강점**: VLA 분야에서 "steerability"라는 차원을 정량적으로 측정하고 개선 레시피를 제시. FineVLA-Data 자체가 valuable resource. **약점**: human verification cost와 paraphrase-baseline 비교가 약하여, 진정한 기여가 *label content*인지 *추가 supervision signal*인지 분리 안 됨.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fine-grained labels 자체가 valuable한가, 단순 paraphrase augmentation으로 대체 가능한가? | 핵심 한계. 논문은 paraphrase baseline 비교가 약함. Pose/approach direction은 paraphrase로 만들기 어려운 *새로운 정보*라는 점이 부분 답변 |
| 2 | Inverted-U는 왜 발생하는가? | FG-only는 raw goal 분포에서 OOD. Raw-only는 fine-grained 조건부 분포 미학습. 둘이 같은 *trajectory data*를 공유하므로 mixing이 두 distribution을 동시에 커버 |
| 3 | RoboTwin-Easy 86.8 vs FineVLA 이전 SOTA? | RDT, π0, GR00T-N1 등의 RoboTwin-Easy는 50~80% 범위 (정확 비교는 paper Table 4). 86.8은 경쟁력 있는 수치 |
| 4 | Real-world 62.7/100의 두 숫자 해석? | 본문에서 goal success vs fine-grained adherence를 동시에 보고하는 dual-metric. 62.7은 raw-only(49.9) 대비 +12.8. 100은 instruction-following metric로 보임 |
| 5 | FineVLA-Data는 공개되는가? | 프로젝트 페이지 (finevla.xlang.ai)에 공개 예정으로 표기. 코드 release 시점은 미확정 |

<!-- VERIFIED: pdf -->
