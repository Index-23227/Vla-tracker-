# StructVLA: Beyond Dense Futures — World Models as Structured Planners for Robotic Manipulation

> **한 줄 요약**: 기존 WAM의 dense future frame 예측 대신, gripper transition과 kinematic turning point 등 **sparse structured frame**만 예측하여 plan drift를 방지하고, SimplerEnv-WidowX **75.0%** + LIBERO **94.8%** 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- Dense future prediction (GR-1, UniPi): 모든 미래 프레임을 생성 → **visual redundancy + error accumulation** → long-horizon plan drift
- Semantic goal prediction (SuSIE): subgoal image 1장만 생성 → **intermediate kinematic detail 부족**
- 두 극단 사이의 중간 지점이 필요

### 핵심 질문
- **Sparse하지만 물리적으로 의미 있는 frame만 예측하면, dense prediction의 오류 누적 없이 plan quality를 유지할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Structured Frame Selection

Kinematic cue 기반으로 task milestone이 되는 frame만 선택:
- **Gripper transitions**: open→close, close→open (접촉/릴리즈 순간)
- **Kinematic turning points**: end-effector velocity 방향이 급변하는 지점

이 frame들이 task progress의 핵심 milestone.

### 2.2 Two-Stage Training with Unified Discrete Tokens

**Stage 1**: World model이 structured frame 예측
**Stage 2**: Structured foresight → low-level action mapping

Unified discrete token vocabulary로 두 stage를 연결.

> ❓ **예상 질문**: Structured frame 수는 task마다 다른데, 가변 길이를 어떻게 처리하는가?
> **답변**: Abstract에서 상세 미공개. Padding이나 max-length truncation을 사용할 가능성이 높으나, 이것이 short vs long horizon task에서 어떤 trade-off를 만드는지 불명확.

---

## 3. 실험 결과

| 벤치마크 | StructVLA | 비고 |
|---------|----------|------|
| **LIBERO** | **94.8%** avg | 높은 수준 |
| **SimplerEnv-WidowX** | **75.0%** avg | 강력한 sim-to-real |

- Real-world deployment 결과도 있다고 주장하나 구체적 수치 미공개

---

## 4. 한계 및 미해결 문제

1. **Full paper 미접근**: ar5iv HTML이 없어 architecture 상세, ablation, training 세부사항 확인 불가
2. **Discrete token vocabulary의 구체적 설계**: Autoregressive인지 다른 방식인지 불명확
3. **Structured frame 선택의 robustness**: 다양한 task에서 kinematic cue가 항상 유효한지 미검증
4. **Real-world 수치 미공개**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Dense vs sparse planning의 중간 지점 |
| **Technical depth** | ★★★☆☆ — Abstract 수준으로만 판단 가능 |
| **Experimental rigor** | ★★★★☆ — LIBERO 94.8% + SimplerEnv 75.0% |
| **Practical impact** | ★★★★☆ — Plan drift 해결이 실용적 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM과의 차이는? | Fast-WAM은 test-time generation 제거, StructVLA는 sparse structured frame만 생성. 다른 접근 |
| 2 | Gripper transition이 없는 task (pushing 등)에서는? | Kinematic turning point로 커버 가능하나, 미세한 continuous task에서 구조적 frame이 부족할 수 있음 |
| 3 | DreamVLA의 dynamic region prediction과 비교하면? | DreamVLA는 motion pixel masking, StructVLA는 시간축에서 sparse frame. 상보적 |

<!-- VERIFIED: abstract-only -->
