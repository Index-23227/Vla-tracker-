# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

> **한 줄 요약**: Long-horizon 조작 태스크를 atomic skill 단위로 분해하여 계획(planning)과 실행(execution)을 통합하는 VLA 프레임워크로, Skill-Guided MoE를 통해 continual learning까지 지원.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA 모델은 **단일 short-horizon 태스크**에 최적화 → multi-step long-horizon 태스크에서 성능 저하
- Hierarchical 접근(SayCan, Code-as-Policies)은 별도의 high-level planner + low-level controller로 분리 → end-to-end 학습 불가, 모듈 간 error cascading
- **Continual learning** 지원 부재: 새로운 스킬 추가 시 기존 스킬 catastrophic forgetting

### 핵심 질문
- **Task planning과 atomic skill execution을 하나의 모델에서 jointly 학습할 수 있는가?**
- **새로운 스킬을 추가할 때 기존 성능을 보존하는 구조적 해결책은?**

---

## 2. 방법론 심층 분석

### 2.1 Atomic Skill Abstraction

태스크를 원자적(atomic) 스킬 단위로 분해:

$$\tau = (s_1, ..., s_K), \quad s_k = (\text{skill\_name}_k, \{a_t\}_{t=t_k}^{t_{k+1}})$$

각 atomic skill은 의미적으로 완결된 단위 행동(예: "reach", "grasp", "lift", "place").

> ❓ **예상 질문**: Atomic skill의 granularity는 어떻게 결정하는가? 사람이 정의하는가, 자동 발견하는가?
> **답변**: 본 논문에서는 사전 정의된 skill vocabulary 사용. 자동 skill discovery(옵션 학습 등)는 미적용. 이는 scalability의 제약이며, 새로운 도메인 적용 시 skill set 재정의 필요.

### 2.2 Unified Planning-and-Execution

하나의 VLA 모델이:
1. **Task-level plan 생성**: 언어 지시 → atomic skill 시퀀스
2. **Skill-level action 생성**: 각 atomic skill에 대한 low-level action trajectory

$$P(\text{plan}, \text{actions} | \mathbf{o}, \mathbf{l}) = P(\text{plan} | \mathbf{o}, \mathbf{l}) \cdot \prod_k P(\text{actions}_k | \text{plan}, \mathbf{o})$$

> ❓ **예상 질문**: Plan 생성과 action 생성을 하나의 forward pass에서 하는가, 아니면 sequential인가?
> **답변**: Autoregressive하게 plan token → action token 순서로 생성. Plan이 먼저 생성되고 이를 조건으로 action이 생성되는 구조. 따라서 plan 오류가 action에 직접 전파됨.

### 2.3 Skill-Guided Mixture-of-Experts (MoE)

Continual learning을 위한 핵심 구조:

$$h = \sum_{i=1}^{N} g_i(\mathbf{x}) \cdot E_i(\mathbf{x}), \quad g_i = \text{Router}(\mathbf{x}, \text{skill\_id})$$

- 각 expert가 특정 skill 집합에 특화
- 새로운 skill 추가 시 **새로운 expert 추가** + router 업데이트만으로 확장
- 기존 expert의 weight는 freeze 가능 → catastrophic forgetting 방지

> ❓ **예상 질문**: Expert 수가 skill 수에 비례하여 증가하면 모델 크기가 무한정 커지지 않는가?
> **답변**: 맞음. 이는 MoE 기반 continual learning의 근본적 한계. 저자들은 expert merging이나 pruning 전략을 언급하지 않음. 실제로는 expert 수에 상한을 두고 knowledge consolidation이 필요.

---

## 3. 데이터 전략

- **Skill annotation**: 기존 demonstration을 atomic skill 단위로 분절(segmentation)
- 분절 방식: 키포인트 기반(gripper state 변화, velocity 변화) + 수동 검증
- Long-horizon 데이터의 skill 분절 품질이 전체 성능에 직접적 영향

---

## 4. 실험 결과 심층 분석

### Long-horizon 태스크 (Real-world)

| 모델 | 4-step Task SR (%) | 6-step Task SR (%) |
|------|-------------------|-------------------|
| OpenVLA | 38.5 | 22.1 |
| Octo | 41.2 | 25.3 |
| **AtomicVLA** | **56.8 (+18.3%)** | **43.1** |

### Continual Learning

| 모델 | 기존 Skill 보존 (%) | 새 Skill 학습 (%) |
|------|-------------------|------------------|
| Fine-tuning (naive) | 52.3 | 78.6 |
| EWC | 71.4 | 65.2 |
| **AtomicVLA (MoE)** | **89.7** | **86.2 (+21%)** |

- Continual learning에서 기존 스킬 보존과 새 스킬 학습 모두 우수

---

## 5. Ablation 분석

| 구성요소 | Long-horizon SR 변화 |
|---------|-------------------|
| MoE → Dense | -8%p (continual), -3%p (single-task) |
| Plan generation 제거 | -12%p |
| Skill segmentation → random split | -15%p |
| Skill-guided routing → standard routing | -5%p (continual) |

- **Plan generation**과 **skill segmentation 품질**이 가장 큰 영향

---

## 6. 관련 연구 비교

| 모델 | Hierarchy | Continual Learning | End-to-End | Skill Discovery |
|------|-----------|-------------------|------------|-----------------|
| SayCan | LLM planner + primitives | ✗ | ✗ | ✗ |
| Code-as-Policies | LLM coder + APIs | ✗ | ✗ | ✗ |
| MUTEX | Shared policy | △ (multi-task) | ✓ | ✗ |
| **AtomicVLA** | **Unified plan+exec** | **✓ (MoE)** | **✓** | **✗** |

핵심 차별점: Planning과 execution을 분리하지 않고 end-to-end, MoE로 continual learning까지 구조적으로 지원.

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Skill vocabulary의 수동 정의**: Atomic skill이 사전 정의되어야 함. 새로운 도메인에서 skill set 설계에 도메인 지식 필요 → scalability 한계
2. **Skill segmentation의 취약성**: 자동 분절 품질이 성능을 좌우하나, segmentation 오류에 대한 robustness 분석 부재
3. **Plan 오류의 cascading**: Plan이 틀리면 이후 모든 action이 실패. Replanning(plan 수정) 메커니즘 미포함
4. **단일 도메인 평가**: 대부분 tabletop manipulation. Navigation, locomotion 등 다른 도메인에서의 atomic skill 일반화 미검증

### Attribution 문제
- MoE의 기여가 **continual learning에서만** 유의미하고 single-task에서는 미미(-3%p) → "MoE가 필수적인가?"라는 질문에 취약
- Plan generation의 기여가 큰데, 이는 별도 LLM planner + 기존 VLA 조합으로도 달성 가능 → end-to-end 통합의 부가가치 정량화 필요

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Skill-guided MoE + unified planning은 참신 |
| **Technical depth** | ★★★☆☆ — MoE 활용은 직관적이나 이론적 보장 부족 |
| **Experimental rigor** | ★★★☆☆ — Real-world 있으나 한정적 |
| **Practical impact** | ★★★★☆ — Continual learning은 실용적으로 매우 중요한 방향 |
| **Writing quality** | ★★★★☆ — 명확한 문제 정의와 해법 제시 |

**강점**: Long-horizon + continual learning이라는 두 핵심 과제를 동시에 다루며, 구조적 해법(MoE)이 설득력 있음. **약점**: Skill vocabulary의 수동 정의가 근본적 제약, plan 실패 시 복구 메커니즘 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Skill vocabulary를 자동으로 발견할 수는 없는가? | Option framework, VQ-VAE 기반 skill tokenization 등 가능하나 본 논문에서는 미탐구 |
| 2 | Plan이 틀렸을 때 어떻게 복구하는가? | Replanning 메커니즘 없음. Closed-loop replanning 추가 시 latency overhead |
| 3 | MoE에서 expert 간 knowledge sharing은 없는가? | Routing만 skill-guided이고 expert 간 explicit sharing은 없음. Shared bottom layer는 있으나 inter-expert transfer 미분석 |
| 4 | Skill 수가 100개 이상으로 늘어나면? | Expert 수 증가에 따른 메모리·연산 선형 증가. Expert consolidation 전략 필요 |
| 5 | SayCan + OpenVLA 조합과의 직접 비교는? | 이 baseline이 누락됨. Modular vs end-to-end의 공정한 비교가 핵심 |
| 6 | Skill boundary에서의 transition은 매끄러운가? | Skill 간 전환 시 discontinuity(급격한 action 변화) 발생 가능. Transition smoothing 분석 부재 |
