# PLD: Self-Improving VLA Models with Data Generation via Residual RL

> **한 줄 요약**: Probe-Learn-Distill (PLD) 3단계 프레임워크로, residual RL로 VLA의 실패 영역을 탐색(probe)하고, hybrid rollout으로 데이터를 수집(learn)하며, 이를 다시 supervised FT로 정제(distill)하여 LIBERO 99% 달성.

---

## 1. 배경 및 동기

- BC의 한계: Demonstration에 없는 상황에서 failure → 더 많은 demo 수집 필요
- RL의 한계: 대형 VLA에 직접 적용하면 학습 불안정
- **Residual RL**: VLA를 base로 두고, 작은 "잔차(residual)" 정책만 RL로 학습

---

## 2. 방법론

### 3-Stage Pipeline

**Stage 1 (Probe)**: Residual RL actor ($\pi_{\text{res}}$)를 학습하여 VLA가 실패하는 영역 탐색
$$\mathbf{a} = \pi_{\text{VLA}}(\mathbf{o}) + \pi_{\text{res}}(\mathbf{o})$$

**Stage 2 (Learn)**: Hybrid rollout (VLA + residual)으로 성공 trajectory 수집
$$\mathcal{D}_{\text{new}} = \{(\mathbf{o}_t, \mathbf{a}_t^{\text{hybrid}})\}$$

**Stage 3 (Distill)**: 수집된 데이터로 VLA를 supervised fine-tuning
$$\pi_{\text{VLA}}^{\text{new}} = \text{SFT}(\pi_{\text{VLA}}, \mathcal{D} \cup \mathcal{D}_{\text{new}})$$

> ❓ **예상 질문**: 왜 residual RL을 직접 배포하지 않고 distill하는가?
> **답변**: Residual actor는 작고 task-specific → 일반화 능력 부족. Distill하면 VLA의 generalist 능력을 유지하면서 새로운 데이터의 이점을 흡수.

---

## 3. 실험 결과

| 모델 | LIBERO (%) | SimplerEnv (%) | Real Robot |
|------|-----------|---------------|-----------|
| OpenVLA (BC) | 76.5 | 48.7 | 52% |
| OpenVLA + naive RL | 81.2 | 52.3 | - |
| **OpenVLA + PLD** | **99** | **98+ (50%+ gain)** | **성공** |

- **LIBERO 99%**: 사실상 **saturation**
- SimplerEnv에서 50%+ 이득

---

## 4. 한계 및 미해결 문제

1. **3-stage pipeline 복잡성**: Probe → Learn → Distill의 순차 실행이 비용 높음
2. **Residual actor 설계**: 크기, 아키텍처의 hyperparameter에 민감
3. **보상 함수 의존**: RL stage에서 reward design 필요
4. **Real-world RL**: 시뮬레이션에서만 full pipeline 검증

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Residual RL + distillation의 결합 |
| **Technical depth** | ★★★★☆ |
| **Experimental rigor** | ★★★★☆ |
| **Practical impact** | ★★★★★ — VLA 자기개선의 실용적 접근 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | PLD를 반복 적용하면 (iterative distillation) 계속 개선되는가? | 이론적으로 가능하나 diminishing returns. 99%에서 추가 개선 여지 미미 |
| 2 | Residual actor 없이 직접 RL로 VLA를 fine-tune하면? | AcceRL/VLA-RL의 접근. PLD는 안정성 면에서 유리 |
