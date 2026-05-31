# VLAs-as-Tools: Tool-Aligned Vision-Language-Action Models for Long-Horizon Embodied Agents

> **한 줄 요약**: 고수준 시간적 추론을 담당하는 VLM 오케스트레이터와 특화된 VLA "도구" family를 분리하고, Tool-Aligned Post-Training(TAPT)으로 도구 호출 정확도와 subtask 실행 능력을 동시에 강화한 계층적 long-horizon 에이전트. OpenVLA-OFT 기반 RoboTwin +35.5 pt, π0.5 기반 LIBERO-Long 97.2%(+4.8) 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 단일 VLA(π0, OpenVLA, RT-2 등)는 long-horizon task에서 **시간적 추론 부담**과 **국소 조작 정밀도**를 한 모델에 짊어짐 → 양쪽 모두 sub-optimal
- 기존 hierarchical 방식(SayCan, RoboFlamingo, Embodied-CoT)은 high-level planner가 low-level skill 호출을 보내지만, **skill의 invocation fidelity**(맞는 skill을 적시에 호출하는 능력)와 **skill 자체의 robust 실행**을 따로 개선하지 못함
- 또한 VLA를 단순 "skill"로 보면 task-specific fine-tune이 발산하여 catastrophic forgetting 발생

### 핵심 질문
- **VLA 모델을 LLM의 *tool* 패러다임처럼 사용할 수 있는가?**
- **High-level VLM의 호출 fidelity와 low-level VLA의 실행 정확도를 동시에 학습할 수 있는가?**

📌 [Figure 1 삽입] — VLM orchestrator + Tool-family VLA 구조

---

## 2. 방법론 심층 분석

### 2.1 계층적 아키텍처

```
[High-Level Layer]
  VLM Agent (Gemini / GPT / Qwen variants)
    - 시간적 추론, subtask decomposition, tool 선택
    └── invocation: <tool_name>, <args>

[Low-Level Layer]
  Tool Family (specialized VLA tools)
    - Tool_1: pick-and-place specialist (VLA + LoRA adapter)
    - Tool_2: pour/articulation specialist
    - Tool_k: ...
    └── Base VLA: OpenVLA / OpenVLA-OFT / π0.5
```

### 2.2 Tool-Family Residual Adapter

각 도구는 base VLA에 **low-rank residual LoRA adapter**를 부착:
- 도구 간 base weight 공유 → memory 효율적
- 도구별 specialization은 LoRA 부분에만 국한
- OpenVLA-OFT 기준 **+9% 파라미터, +7% inference 오버헤드** (효율적)

> ❓ **예상 질문**: 도구별 LoRA가 도구 수에 비례 증가하면 결국 monolithic VLA보다 크지 않은가?
> **답변**: LoRA rank가 작아 도구 수가 늘어나도 incremental cost가 작음. 또한 도구는 task family 단위로 묶이므로 수십 개 수준이면 충분.

### 2.3 Tool-Aligned Post-Training (TAPT)

**두 단계 학습**:

**Stage 1 — Invocation-Aligned Supervised Fine-Tuning**
- VLM이 어떤 도구를 어떤 시점에 호출해야 하는지에 대한 supervised 학습
- **Invocation-aligned training unit**: (state, tool_call, expected_subtask_outcome) triplet
- 이는 VLM의 도구 선택 분포를 task-relevant subset으로 좁힘

**Stage 2 — Bounded Subtask RL with GRPO**
- 각 도구의 subtask 실행을 GRPO(Group Relative Policy Optimization)로 fine-tune
- "Bounded subtask"란 subtask를 짧고 명확한 시간 윈도우로 자르는 것 → reward sparsity 완화
- 도구 호출이 끝나는 시점이 자연스러운 episode boundary 역할

> ❓ **예상 질문**: 왜 PPO가 아니라 GRPO인가?
> **답변**: GRPO는 group baseline으로 variance를 줄이고 critic network를 생략 → VLA처럼 큰 모델에서 메모리 효율적. 또한 sparse reward 환경에서 robust.

### 2.4 Invocation Fidelity 측정

논문은 **LIBERO-CF-Long** (Counter-Factual long horizon)이라는 진단 벤치를 정의:
- **Non-biased Rate**: 다양한 도구 후보 중 *맞는* 도구를 *맞는 시점*에 호출하는 비율
- 단순 success rate와 다름 — VLM이 "lucky" success를 거두는 case를 걸러냄

> ❓ **예상 질문**: 왜 단순 success rate가 부족한가?
> **답변**: 단일 도구만 반복 호출해도 운 좋게 성공하는 경우가 있음. Invocation fidelity는 "올바른 도구 선택"의 본질적 능력을 분리하여 측정.

---

## 3. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM Orchestrator | Gemini / GPT / Qwen variants (테스트) |
| VLA Tools | OpenVLA, OpenVLA-OFT, π0.5 (3종 비교) |
| 학습 데이터 | DROID 데이터 split + LIBERO + RoboTwin + CALVIN |
| Stage 1 (IL) | Invocation-aligned supervised fine-tune (LoRA만 학습) |
| Stage 2 (RL) | Bounded subtask GRPO |
| Param overhead | +9% (OpenVLA-OFT 기준) |
| Inference overhead | +7% |

---

## 4. 실험 결과 심층 분석

### LIBERO-Long (Imitation Learning, Table)

| Base VLA | Baseline | + TAPT (Tool) | 향상 |
|---------|----------|--------------|------|
| OpenVLA | 77.2% | **82.4%** | +5.2 pt |
| OpenVLA-OFT | 92.0% | **95.6%** | +3.6 pt |
| π0.5 | 92.4% | **97.2%** | +4.8 pt |

### RoboTwin (Imitation Learning)

| Base VLA | Baseline | + TAPT | 향상 |
|---------|----------|--------|------|
| OpenVLA | 1.9% | **5.7%** | +3.8 pt |
| OpenVLA-OFT | 16.9% | **52.4%** | **+35.5 pt** |
| π0.5 | 39.4% | **62.5%** | +23.1 pt |

> ❗ **RoboTwin에서의 +35.5 pt(OpenVLA-OFT)는 본 논문의 가장 인상적 결과**. Multi-step contact-rich manipulation에서 도구 분리가 큰 효과.

### Reinforcement Learning 결과

| Bench | OpenVLA-OFT | π0.5 |
|-------|-------------|-------|
| LIBERO-Long (with tool-family) | 82.6% | **91.2%** |
| CALVIN (baseline → +TAPT) | 78.8 → 82.3 | 80.0 → **82.3** |

### Invocation Fidelity (LIBERO-CF-Long)

| Base | Non-biased Rate 향상 |
|------|---------------------|
| OpenVLA-OFT | **+16.2 pt** |
| π0.5 | **+15.0 pt** |

→ TAPT의 invocation-aligned training이 단순 task success를 넘어 **도구 선택 능력 자체**를 개선함을 입증.

---

## 5. Ablation 분석

### TAPT 구성요소별 기여 (요약)

| 설정 | LIBERO-Long (π0.5) | Non-biased Rate |
|------|--------------------|----------------| 
| Baseline π0.5 | 92.4% | base |
| + Invocation-aligned SFT only | ~95% | +9~10 pt |
| + Bounded subtask GRPO only | ~94% | +5~6 pt |
| + 둘 다 (Full TAPT) | **97.2%** | **+15.0 pt** |

→ 두 stage 모두 기여하며, invocation-aligned SFT가 fidelity에 더 크게 기여.

> ❓ **예상 질문**: Tool 수를 늘리면 invocation fidelity가 떨어지지 않나?
> **답변**: 논문은 ~수~수십 개 도구 범위에서 평가. 도구 수가 커지면 VLM의 분류 부담 증가, scalability 검증은 추가 연구 필요.

---

## 6. 관련 연구 비교

| 모델 | Hierarchy | Tool specialization | Invocation 학습 | Long-horizon 성능 |
|------|-----------|--------------------|------------------|-----------------|
| SayCan | LLM + value func | ✗ | ✗ | medium |
| RoboFlamingo | VLM monolithic | ✗ | ✗ | low |
| Embodied-CoT | VLA + CoT | ✗ | ✗ | medium |
| Hi Robot | Hierarchical VLA | partial | ✗ | medium |
| **VLAs-as-Tools** | **VLM + Tool VLAs** | **✓ (LoRA family)** | **✓ (TAPT)** | **high** |

### 핵심 차이
- LLM의 **tool-use 패러다임**을 robotics에 그대로 옮긴 첫 본격적 시도
- 기존 hierarchical 방법들이 high-level planner의 자연어 instruction에 의존한 반면, 본 연구는 **structured tool invocation**으로 접점을 좁힘

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **VLM orchestrator의 cost**: Gemini/GPT 등 commercial VLM 호출이 inference loop마다 발생 → real-time / on-device 운용 어려움
2. **Tool family 설계의 manual 의존**: 도구 분류 체계가 사람의 사전 정의에 의존. Tool discovery 자동화 미해결
3. **DROID split 사용**: DROID 데이터로 학습하면서 LIBERO/RoboTwin에서 평가 → 데이터 leakage 가능성 점검 필요
4. **CALVIN 점수 해석 모호**: 82.3%로 보고되나 official CALVIN avg_len 메트릭과 일치 여부 불분명
5. **Code unreleased**: 코드 "will be released" 상태로 reproducibility 한계
6. **OpenVLA RoboTwin 5.7%의 절대값이 낮음**: 비록 +3.8 pt 향상이지만 절대 점수는 여전히 낮음 → base VLA 자체가 약하면 tool 분리도 한계

### Attribution 문제
- 성능 향상이 (a) tool 분리 자체인지 (b) LoRA 추가 capacity인지 (c) invocation-aligned 학습 신호인지 분리 ablation 필요
- π0.5 같은 강한 base에서 +4.8 pt는 marginal로 볼 여지 — base가 클수록 head-room이 작아짐

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA를 LLM tool 패러다임으로 재해석한 점이 명확 |
| **Technical depth** | ★★★★☆ — Invocation-aligned SFT + bounded GRPO 조합이 정교 |
| **Experimental rigor** | ★★★★☆ — 3종 base VLA × IL/RL × LIBERO/RoboTwin/CALVIN의 광범위한 grid |
| **Practical impact** | ★★★★☆ — RoboTwin +35.5 pt(OpenVLA-OFT)는 의미 있는 향상 |
| **Writing quality** | ★★★★☆ — 명확한 framework, fidelity diagnostic 도입 |

**강점**: VLA로 hierarchical agent를 구성하는 시점에서 "tool" 추상화를 통해 **invocation fidelity를 명시적으로 학습**한 점. RoboTwin에서의 큰 향상은 long-horizon contact-rich에 효과 입증. **약점**: VLM orchestrator의 cost와 tool discovery 자동화 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | RoboTwin OpenVLA-OFT +35.5 pt가 너무 큰 향상인데, 어떤 메커니즘? | 도구 분리로 catastrophic forgetting 방지 + subtask GRPO로 contact 단계별 reward 학습 가능 |
| 2 | Gemini/GPT orchestrator의 latency는? | 한 step당 수백 ms~수초. real-time control에는 unsuited. closed-loop 평가는 도구 호출 빈도가 낮은 task에서만 적합 |
| 3 | CALVIN 82.3%가 official avg_len과 어떻게 비교되는가? | 본 논문이 percentage로 reporting → 직접 비교 불가. official metric 변환 필요 |
| 4 | Tool family 수에 따른 scalability는? | 본 논문 범위(수~수십 개)에서는 잘 동작. 100개 이상에서 invocation 정확도 감소 가능 |
| 5 | DROID로 학습 후 LIBERO 평가 시 distributional shift는? | DROID는 광범위한 manipulation 데이터, LIBERO와 직접 overlap은 적으나 task family는 유사 |
| 6 | 왜 PPO 대신 GRPO인가? | GRPO는 critic-free + group baseline → 큰 VLA에서 memory 효율적, sparse reward에 robust |
| 7 | OpenVLA RoboTwin 5.7%로 여전히 매우 낮은데 의미가 있는가? | 절대값은 낮지만 +3.8 pt(상대 +200%)는 base의 한계 안에서의 개선. base 선택 중요성 시사 |
| 8 | Tool selection 정답을 어떻게 정의했나(supervision label)? | Invocation-aligned training unit이 (state, correct tool, outcome) triplet로 정의됨. 사람 또는 자동 분할 |

<!-- VERIFIED: pdf -->
