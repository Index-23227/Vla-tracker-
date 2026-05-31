# BORA: Bridging Offline Reinforcement Learning and Online Residual Adaptation for Real-World Dexterous VLA Models

> **한 줄 요약**: VITRA VLA backbone에 *action-conditioned IQL-style critic* (VLM token + action chunk 입력)으로 offline RL을 수행하고, deployment 시 frozen base 위에 *lightweight MLP residual actor*를 Intervention-Driven RLPD로 online 학습 — 12-DoF dexterous hand에서 Standard 86%, Object-Unseen 70% (각각 baseline 대비 +33, +43 absolute).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Diffusion Policy / Behavior Cloning 기반 VLA**는 demos의 모드(suboptimal trajectories 포함)를 그대로 모방 → suboptimal하거나 visually overfit
- **Online RL**은 처음부터 학습하면 sample inefficient, 그러나 VLA로 directly fine-tune 시 catastrophic forgetting 위험
- **Dexterous hand (12-DoF 이상)**: action space가 고차원 → 작은 prediction error도 contact failure로 이어짐
- 기존 OpenVLA / pi0 류는 *parallel jaw gripper* 기반 → 손가락 단위 control 부재

### 핵심 질문
- **VLA에 RL을 어떻게 결합하면 BC의 약점 (visual overfitting, suboptimality)을 보완할 수 있나?**
- **Online deployment 시 base policy를 *훼손하지 않고* 환경별 correction을 학습할 수 있나?**

📌 [Figure 1 삽입] — BORA 두 단계: (1) Offline RL with action-conditioned critic, (2) Online residual adaptation with HITL

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
┌────────────── Offline Phase ──────────────┐
VITRA VLA (consistency policy action expert)
  ↓
Action-conditioned Critic Q(VLM_tokens, action_chunk, pos_embed)
  ↓ IQL expectile + BC reg + PPO-style improvement
Improved offline policy

┌────────────── Online Phase ───────────────┐
Frozen base VLA  →  A_base
                    ↓
                    + λ_res · π_res(s_prop, A_base, z_VLM)
                    ↓
                    A_final = A_base + residual
  ↑
Intervention-Driven RLPD (HITL correction + asymmetric reward)
```

### 2.2 Consistency Policy로의 교체

- VITRA의 원래 action expert는 diffusion 기반 (여러 step denoise)
- BORA는 이를 **Consistency Policy**로 교체 → 1-3 step만으로 action chunk 생성
- Deployment latency 절감 + RL training 시 backprop 비용 절감

### 2.3 Action-Conditioned Critic

Q-function의 입력:
- **VLM cognition tokens** (z_VLM): VITRA가 출력하는 high-level perception representation
- **Action chunks** (A): 정책이 생성한 action sequence
- **Position embeddings**

학습 목표:
- **IQL-style expectile value objective**: conservative value targets from sub-optimal offline data
- **Within-chunk credit propagation**: "shifted value bootstrap" (Eq. 1) — chunk 내부의 action마다 separate gradient

Loss = BC regularization + PPO-style policy improvement (pure CQL/IQL이 아닌 hybrid)

> ❓ **예상 질문**: 왜 *action-conditioned* critic인가? 표준 V(s)나 Q(s,a)와 어떻게 다른가?
> **답변**: VLA는 action *chunk* (e.g., 32-step)를 한 번에 출력. 단순 Q(s,a)는 single-step action을 가정 → chunk 내부의 credit assignment 불가. Action-conditioned chunk-level critic은 chunk 내 각 action의 contribution을 분리 평가 가능.

### 2.4 Online Residual Adaptation

수식:
$$A_{\text{final}} = A_{\text{base}} + \lambda_{\text{res}} \cdot \pi_{\text{res}}(s_{\text{prop}}, A_{\text{base}}, z_{\text{VLM}})$$

- **Base policy 동결** — pretrained capability 보존
- **π_res**: lightweight MLP, proprioception + base action + VLM tokens 입력
- **Intervention-Driven RLPD**: 인간이 ~1-2회 task당 개입 → off-policy correction 데이터
- **Asymmetric reward**: 성공/실패에 비대칭 penalty/bonus

> ❓ **예상 질문**: 왜 base policy를 fine-tune하지 않고 residual을 별도로 학습하나?
> **답변**: VITRA 같은 large VLA는 fine-tune 시 catastrophic forgetting과 distribution shift 위험. Residual은 **base의 inductive bias 보존**하면서 environment-specific correction만 학습 → "skill" vs "calibration"의 분리.

---

## 3. 시스템/학습 세부사항

| 단계 | Hardware | Updates | Batch | Data |
|------|----------|---------|-------|------|
| Offline | 8× H100 | 70,000 | 8/GPU | ~60-100 trajectories/task |
| Online | 1× RTX 4090 | up to 15,000 | - | ~10 intervention traj/iter, 1-2 interventions/task |

- Action chunk size: 32
- Hand: 12-DoF dexterous hand (DexHand021)
- Arm: Franka Emika

---

## 4. 실험 결과 심층 분석

### 4.1 Standard Setting (Table 1)

| Task | VITRA | CP Base | BORA-Offline | **BORA-Full** |
|------|-------|---------|--------------|--------------|
| Pick Plush Toy | - | - | - | **100%** |
| Pick and Place | - | - | - | **90%** |
| Open Box | - | - | - | 75% |
| Pull Tissue | - | - | - | 80% |
| Press Button | - | - | - | 85% |
| **Average** | - | - | - | **86.0%** |

- Pick Plush Toy 100% — 작은 deformable object를 dexterous hand로 grasping 성공
- Open Box 75%는 sequential manipulation에서 가장 어려운 case

### 4.2 Object-Unseen Setting (Table 2)

| Task | **BORA-Full** |
|------|--------------|
| Pick Plush Toy | 85% (-15) |
| Pick and Place | 70% (-20) |
| Open Box | 50% (-25) |
| Pull Tissue | 70% (-10) |
| Press Button | 75% (-10) |
| **Average** | **70.0%** (-16) |

- Unseen object에서 86% → 70% (-16 absolute) drop
- Open Box (50%)에서 가장 큰 generalization gap — object shape 변화에 민감

### 4.3 핵심 향상폭

- **+33% absolute** 평균 성공률 (baseline 대비)
- **+43% absolute** unseen object generalization

> ❓ **예상 질문**: Baseline의 정확한 수치는?
> **답변**: v1 HTML에서 baseline (VITRA, CP Base, Decoupled-Critic) 각각의 표 수치가 명시되지 않음. "+33% / +43%"이 평균값임은 확실하나, per-task breakdown comparison이 불명확.

---

## 5. Ablation 분석

### BORA-Offline vs BORA-Full

- BORA-Offline만 (online residual 없이) → BORA-Full로 가는 추가 향상이 있음
- 정확한 수치는 v1 HTML에서 명시적이지 않으나, online residual의 *contribution이 분리되어* 보고됨

---

## 6. 관련 연구 비교

| 모델 | Base 학습 방식 | Online 적응 | Dexterous | Real-world |
|------|---------------|------------|-----------|-----------|
| OpenVLA | BC | ✗ | ✗ (parallel jaw) | ✓ |
| pi0 | BC | ✗ | ✓ (일부) | ✓ |
| Diffusion Policy | BC | ✗ | ✗ | ✓ |
| RT-2 | BC | ✗ | ✗ | ✓ |
| **BORA** | **Offline RL** | **Residual + HITL** | **✓ (12-DoF)** | **✓** |

### 핵심 차이
- "Offline RL + Online residual" 두 단계 결합은 VLA 분야에서 거의 첫 시도
- 특히 *base policy 동결 + lightweight residual*은 catastrophic forgetting 우려를 해결하는 elegant한 방식

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **VITRA backbone 미공개**: BORA는 VITRA (Li et al., 2025)를 base로 사용하나, VITRA 자체의 backbone VLM / 파라미터 수 / 학습 데이터를 BORA 논문에서 상세히 공개하지 않음 → reproducibility 약점
2. **Benchmark 표준화 부재**: LIBERO, CALVIN, SimplerEnv 등 standard benchmark 미평가. 5-task real-world 결과만 보고 → 다른 VLA와 직접 비교 불가
3. **Baseline 수치 명시 부족**: "+33% / +43% improvement"의 baseline 절대값이 표에서 명확히 분리되지 않음
4. **HITL 비용 미정량화**: "1-2 interventions per task"의 시간/노력 cost가 정량적으로 비교되지 않음 — 진정한 sample efficiency 평가에는 human time 포함 필요
5. **DexHand021 specs 부족**: 12-DoF hand의 sensor (force/tactile), kinematics detail, control frequency 부족
6. **Generalization 다른 축**: Object-unseen만 평가. Scene-unseen, language-unseen, distractor 등 다른 generalization 축 부재
7. **Consistency Policy의 quality loss 미검증**: Diffusion → Consistency 교체로 인한 *기본 성능* 영향이 명확히 ablation 안 됨

### Attribution 문제
- 86% 성공률이 (a) Offline RL critic, (b) Consistency policy speedup, (c) Online residual, (d) HITL 중 무엇 덕분인지 분리되지 않음
- BORA-Offline-only vs BORA-Full의 명확한 numerical breakdown이 핵심 ablation

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Offline RL + Online residual의 결합은 VLA 분야 첫 사례 |
| **Technical depth** | ★★★★☆ — IQL critic + RLPD + Consistency policy 통합 |
| **Experimental rigor** | ★★★☆☆ — Real robot 5-task는 충분하나 표준 benchmark 부재 |
| **Practical impact** | ★★★★☆ — Dexterous hand + HITL은 실제 deployment에 가까운 setup |
| **Writing quality** | ★★★☆☆ — VITRA backbone detail 부재, baseline 수치 분리 부족 |

**강점**: VLA의 BC dominance에 대한 명확한 대안 — *base policy 동결 + lightweight residual*은 catastrophic forgetting을 우회. 12-DoF dexterous hand + HITL 결합은 실제 robotics deployment 시나리오에 가까움.

**약점**: VITRA backbone의 black-box 처리. 표준 benchmark (LIBERO/CALVIN/SimplerEnv) 평가 부재로 다른 VLA와 직접 비교 어려움. HITL의 human cost가 quantitative comparison에서 누락.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VITRA backbone의 상세 (VLM, 파라미터, 데이터)는? | BORA 논문에서 상세 미공개. Li et al. 2025 원 논문을 참조해야 함 — reproducibility 약점 |
| 2 | "Consistency Policy" 교체가 단독으로 성능에 미치는 영향은? | Ablation 불명확. Diffusion → Consistency가 정확도를 *낮추지* 않는다는 검증이 필요 |
| 3 | Baseline 수치가 정확히 무엇인가? | "+33% / +43%" 평균이지만 absolute baseline 수치가 표에서 명시적이지 않음 |
| 4 | HITL의 1-2 interventions/task가 실용적인가? | Plush toy 같은 단순 task는 OK, 더 복잡한 long-horizon task에서는 intervention budget이 폭발 가능 |
| 5 | LIBERO / SimplerEnv를 왜 안 했나? | Dexterous hand 환경이라 표준 benchmark가 parallel jaw 가정. 그러나 cross-comparison 부재로 다른 VLA 대비 강점이 directly verifiable하지 않음 |
| 6 | Action-conditioned critic의 within-chunk credit assignment가 실제로 학습되나? | "Shifted value bootstrap"의 효과를 ablate하면 좋음 — full chunk vs single-step Q의 비교 부족 |
| 7 | Residual scale λ_res는 어떻게 결정되나? | Hyperparameter — 너무 크면 base를 overwrite, 너무 작으면 correction 부족. Sensitivity analysis 필요 |
| 8 | 12-DoF DexHand021의 force/tactile feedback은 사용되나? | 명시 부족. Tactile 없이 vision만으로 12-DoF control은 매우 어려운 setup — sensor 사용 여부가 결과의 generality에 직결 |

<!-- VERIFIED: pdf -->
