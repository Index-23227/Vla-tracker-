# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

> **한 줄 요약**: π₀ foundation model 위에 Skill-Guided MoE (SG-MoE)를 구축하여, atomic skill 단위의 expert routing으로 long-horizon 태스크에서 π₀.₅ 대비 +18.3% real-world 향상, continual learning에서 catastrophic forgetting을 -1.3%로 최소화 (π₀.₅는 -15%).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA 모델은 **단일 short-horizon 태스크**에 최적화 → multi-step long-horizon 태스크에서 성능 저하 (π₀의 LIBERO-Long: 85.2%)
- Hierarchical 접근(SayCan, Code-as-Policies)은 별도의 high-level planner + low-level controller로 분리 → end-to-end 학습 불가, 모듈 간 error cascading
- **Continual learning** 지원 부재: 새로운 스킬 추가 시 기존 스킬 catastrophic forgetting (π₀.₅는 -15% 하락)

### 핵심 질문
- **Task planning과 atomic skill execution을 하나의 모델에서 jointly 학습할 수 있는가?**
- **새로운 스킬을 추가할 때 기존 성능을 구조적으로 보존할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 세부사항

π₀ (Gemma 기반)를 backbone으로:
- Width: 2048, MLP dim: 4096, depth: 18, heads: 8, head dim: 256
- **π₀ baseline: 3.24B parameters**
- **SG-MoE K=5: 4.17B**, K=8: 4.81B, K=12: 5.65B parameters

### 2.2 Skill-Guided Mixture-of-Experts (SG-MoE)

핵심 구조: Skill embedding → expert routing

1. **Skill embedding**: Atomic skill을 noise level $\sigma \in [0, 100]$로 매핑 후 normalized log transformation으로 고차원 벡터 변환
2. **Routing**: $w_k = \text{Router}(\mathbf{Z}_\sigma)$, top-1 expert 선택
3. **Action 생성**: $\mathbf{F}_{out} = (1-w_k) \cdot \mathbf{F}_{share}(\mathbf{x}_t) + w_k \cdot \mathbf{F}_k(\mathbf{x}_t)$

- **Shared expert**: 모든 skill에 공통 지식
- **Skill-specific experts**: 각 atomic skill에 특화된 SwiGLU MLP

> ❓ **예상 질문**: 왜 timestep-conditioned MoDE가 아닌 skill-conditioned routing인가?
> **답변**: Ablation에서 직접 비교: **SG-MoE 95.2% vs MoDE 89.5% vs Standard MoE 88.6%** (LIBERO-Long). Skill conditioning이 timestep보다 +5.7%p 우수. Skill이 action의 semantic meaning을 직접 인코딩하므로 더 효과적인 routing signal.

> ❓ **예상 질문**: Expert 수가 skill 수에 비례하면 모델 크기가 무한정 커지지 않는가?
> **답변**: K=5에서 K=12로 늘려도 LIBERO에서 성능 차이 미미하나, **latency는 71ms→160ms**로 2.3배 증가. Expert 수의 sweet spot이 존재하며, expert merging/pruning은 미탐구.

### 2.3 Atomic Skill Segmentation

End-effector trajectory의 kinematic 변화를 분석하여 자동 분절:
- Δx, Δy, Δz (위치 변화)
- Δroll, Δpitch, Δyaw (회전 변화)
- Gripper state 변화

**InternVideo2.5**로 semantic label 검증.

LIBERO skill 분포 (Table 6):

| Skill | 인스턴스 수 |
|-------|-----------|
| Pick | 2,462 |
| Place | 761 |
| Turn | 175 |
| Close | 152 |
| Open | 201 |

CALVIN: 8 skills (Rotate, Push, Move, Open&Close, Lift, Place, Turn, Stack)

---

## 3. 데이터 전략

### Real-world Setup
- **로봇**: Franka Research3
- **카메라**: 2× Realsense D435i (wrist + third-person)
- **데이터**: 550 total trajectories
  - Short-horizon: 50 trajectories/task × 5 tasks
  - Long-horizon: 100 trajectories/task × 3 tasks

### 학습 설정

| 항목 | 값 |
|------|-----|
| GPU | 8× H200 |
| Iterations | 100K (LIBERO/CALVIN), 30K (real-world) |
| Batch size | 64 |
| Peak LR | 2.5×10⁻⁵ |
| Final LR | 5×10⁻⁶ |
| Warmup | 1,000 steps |
| EMA decay | 0.999 |
| Optimizer | AdamW (grad clip 1.0) |
| Inference GPU | Single NVIDIA RTX 6000 |

---

## 4. 실험 결과 심층 분석

### LIBERO Benchmark (Table 1)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| π₀ | 96.4 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **AtomicVLA** | 96.8 | 98.0 | 96.4 | **95.2** | 96.6 |
| **AtomicVLA*** | 98.8 | 98.8 | 97.2 | **96.2** | **97.8** |

- **LIBERO-Long에서 π₀ 대비 +11%p** (85.2→96.2) — long-horizon에서 가장 큰 향상
- AtomicVLA*는 π₀.₅와 comparable하면서 Long에서 우위 (96.2 vs 92.4)

### CALVIN ABC→D (Table 2)

| 모델 | 1 Task | 2 Tasks | 3 Tasks | 4 Tasks | 5 Tasks | **Avg Len** |
|------|--------|---------|---------|---------|---------|------------|
| π₀ | 94.3 | 87.0 | 77.9 | 68.5 | 59.4 | 3.87 |
| π₀.₅ | 91.9 | 84.6 | 79.4 | 75.5 | 71.0 | 4.02 |
| **AtomicVLA** | 95.0 | 87.8 | 81.9 | 75.0 | 69.1 | **4.09** |
| **AtomicVLA*** | 94.1 | 88.7 | 85.2 | 81.7 | 77.6 | **4.27** |

- **CALVIN 4.27 avg len** — 5-task에서 77.6%는 π₀.₅(71.0%) 대비 +6.6%p

### Real-world Long-horizon (Table 3)

| 모델 | Objects in Plate | Into Drawer | Into Microwave | **Avg** |
|------|-----------------|-------------|---------------|---------|
| π₀ | 45% | 55% | 10% | 36.7% |
| π₀.₅ | 65% | 35% | 35% | 45.0% |
| **AtomicVLA*** | **75%** | **60%** | **55%** | **63.3% (+18.3%)** |

### ⭐ Continual Learning (Table 4)

| 모델 | 기존 4 tasks 평균 | 새 task (Open) | 전체 Avg | **성능 변화** |
|------|-----------------|---------------|---------|-------------|
| π₀.₅ baseline | 77.5% | - | 77.5% | - |
| π₀.₅ after CL | 62.5% | 55% | 61.0% | **-15.0%** |
| AtomicVLA* baseline | 86.3% | - | 86.3% | - |
| AtomicVLA* after CL | 87.5% | 70% | 82.0% | **-1.3%** |

- **핵심 결과**: π₀.₅는 새 skill 추가 시 기존 성능 -15% 하락, AtomicVLA*는 **-1.3%만 하락**
- SG-MoE의 새 expert 추가 + 기존 expert freeze가 catastrophic forgetting을 구조적으로 방지

---

## 5. Ablation 분석 (Table 5, LIBERO-Long)

| 방법 | Success Rate (%) |
|------|-----------------|
| π₀ baseline | 85.2 |
| + Standard MoE | 88.6 (+3.4) |
| + MoDE (timestep-conditioned) | 89.5 (+4.3) |
| **+ SG-MoE (Ours)** | **95.2 (+10.0)** |

- **SG-MoE > MoDE (+5.7%p) > Standard MoE (+6.6%p)** → Skill conditioning의 결정적 우위

### Inference Latency (Table 10)

| 설정 | Act Latency | Think Latency |
|------|------------|--------------|
| π₀ (no MoE) | 71ms | - |
| K=5 experts | 92ms | 104ms |
| K=8 experts | 126ms | 104ms |
| K=12 experts | 160ms | 104ms |

- Think (VLM reasoning) latency는 expert 수에 무관 (104ms 고정)
- Act latency가 expert 수에 비례하여 증가

---

## 6. 관련 연구 비교

| 모델 | Hierarchy | CL Support | End-to-End | Long-horizon | Params |
|------|-----------|-----------|------------|-------------|--------|
| SayCan | LLM + primitives | ✗ | ✗ | ✓ (modular) | - |
| π₀ | Flat policy | ✗ | ✓ | △ (85.2%) | 3.24B |
| π₀.₅ | Flat + subtask | ✗ | ✓ | ✓ (92.4%) | ~3B+ |
| **AtomicVLA** | **Skill-guided MoE** | **✓ (-1.3%)** | **✓** | **✓ (96.2%)** | **4.17B** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점 (저자 명시 포함)
1. **VLM reasoning fidelity 의존**: 저자들이 직접 인정 — "skill router relies on the VLM to produce accurate atomic skill abstractions, constrained by VLM's reasoning and planning fidelity." VLM이 잘못된 skill sequence를 생성하면 전체 cascade 실패
2. **여전히 많은 demonstration 필요**: 저자 명시 — "acquiring new tasks still requires collecting substantial human demonstration data." 550 trajectories는 상당한 수집 비용
3. **Skill vocabulary의 수동 정의**: Pick, Place, Turn, Close, Open 등이 사전 정의됨. 새 도메인에서 skill set 재정의 필요 → scalability 제약
4. **Expert 수와 latency trade-off**: K=5→K=12에서 latency 71ms→160ms. Real-time 요구 환경에서 제약
5. **Continual learning의 범위**: 1개 새 skill (Open) 추가만 테스트. 10개+ skill 순차 추가 시의 scalability 미검증

### Attribution 문제
- AtomicVLA의 성능이 **SG-MoE** 때문인지, **π₀ backbone의 강력함** 때문인지 분리 필요. π₀ 자체가 94.2%이므로 SG-MoE의 marginal 기여 (+2.4%p on avg, +10%p on Long)가 핵심
- AtomicVLA*의 *는 무엇인지 — think token (VLM reasoning) 추가 variant로, **SG-MoE + think의 결합 효과** 분리 필요

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Skill-guided MoE + continual learning의 구조적 해법 |
| **Technical depth** | ★★★★☆ — SG-MoE 설계 체계적, skill segmentation 자동화 |
| **Experimental rigor** | ★★★★★ — LIBERO + CALVIN + Real-world + Continual Learning |
| **Practical impact** | ★★★★★ — Continual learning은 실용적으로 매우 중요 |
| **Writing quality** | ★★★★☆ — 명확한 문제 정의와 해법 |

**강점**: Long-horizon에서 π₀ 대비 +10%p, continual learning에서 forgetting -1.3% (vs -15%)는 인상적. SG-MoE가 standard MoE, MoDE를 명확히 능가하는 ablation이 설득력 있음. Real-world 결과 포함. **약점**: VLM reasoning 의존, skill vocabulary 수동 정의, expert scaling의 latency trade-off.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Skill vocabulary를 자동으로 발견할 수는 없는가? | VQ-VAE 기반 skill tokenization, option framework 등 가능하나 미탐구. 현재는 kinematic segmentation + InternVideo2.5 |
| 2 | VLM이 잘못된 plan을 생성하면 어떻게 복구하는가? | Replanning 메커니즘 없음. 저자들이 이를 핵심 한계로 인정 |
| 3 | Expert 수 K=5가 K=12보다 효율적이면서 성능도 유사한 이유는? | LIBERO가 5종 skill로 충분히 커버됨. Skill diversity가 높은 환경에서는 K 증가가 유리할 수 있음 |
| 4 | π₀.₅ 대비 AtomicVLA*의 추가 파라미터(+~1B)가 정당화되는가? | LIBERO avg 97.8 vs 96.9 → +0.9%p로 marginal. 핵심 가치는 continual learning (-1.3% vs -15%) |
| 5 | Continual learning에서 10+ skills를 순차 추가하면? | 1 skill만 테스트. Expert 수가 누적되면 메모리와 latency 문제. Expert consolidation/merging이 필요하나 미탐구 |
| 6 | Think latency 104ms + Act latency 92ms = 196ms. 5Hz 제어가 가능한가? | Think을 매 step이 아닌 주기적으로만 실행하면 Act만 ~92ms → ~11Hz 가능. 하지만 Think 주기 최적화 미보고 |

<!-- VERIFIED: pdf -->
