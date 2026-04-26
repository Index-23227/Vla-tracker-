# HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation

> **한 줄 요약**: 부위별 독립 action prediction의 한계를 극복하기 위해, humanoid-aligned state 표현과 Mixture-of-Experts + flow-matching action generation을 결합한 **whole-body coordinated** VLA 프레임워크.

---

## 1. 배경 및 동기

- 인간의 조작은 전신(Whole-body) 협조를 통해 이루어짐.
- 기존 VLA는 팔/몸/다리 등 부위별 action을 largely independently 예측 → 고DoF humanoid에서 불안정.
- **핵심 과제**: 고차원 humanoid action space를 통합적으로 모델링하되, cross-embodiment 전이도 가능해야 함.

---

## 2. 방법론

### Humanoid-Aligned State Representation
- Robot의 proprioceptive state를 humanoid 해부학적 구조에 맞춰 정렬된 형태로 인코딩.
- 부위 간 상관관계를 보존한 state-centric 접근 (action-centric이 아님).

### Mixture-of-Experts (MoE) for Whole-Body Coordination
- 각 expert가 특정 body-part 또는 motion mode를 전담.
- Routing을 통해 여러 expert의 출력을 coordinated whole-body action으로 통합.

### Residual-Gated Fusion + Flow Matching
- Lightweight history tokens으로 temporal visual context를 경량 주입.
- Residual-gated fusion으로 modality·time 차원 정보를 결합 후, **flow matching** head가 연속 action 생성.

---

## 3. 실험 결과

### Seen scenarios (Table 1, 7개 태스크 평균, 12 trials each)

| 방법 | Param. | Seen Avg |
|------|-------:|---------:|
| ACT | 80M | 57.1 |
| SwitchVLA | 0.3B | 40.5 |
| GR00T N1.5 | 3B | 70.2 |
| π0.5 | 3.3B | 71.8 |
| **HEX** | **2.4B** | **79.8** |

- 작은 모델(ACT 80M)이 trajectory-fitting은 강하지만 HEX가 전반적 1위.
- Mirror pose / Walking-avoid 태스크에서 **100%**.

### Long-horizon Box Convey (Table 2, 4단계)

| 단계 | ACT | π0.5 | **HEX** |
|------|----:|-----:|--------:|
| Grasp Box | 80.0 | 100.0 | **100.0** |
| Turn Around | 80.0 | 100.0 | **100.0** |
| Walk to Table | 46.7 | 73.3 | **73.3** |
| Place Box | 26.7 | 40.0 | **53.3** |

- 마지막 Place Box 단계에서 +15pp (+13pp over π0.5) → cascading error 억제 능력.

### Generalization (Fig. 6, 8개 OOD 변형 평균)

- **HEX 61.8%** vs π0.5 44.3%, GR00T N1.5 41.0%, SwitchVLA 22.4%.
- Pouring 변형에서 baseline 모두 0%인데 HEX 53.3% (visual distractor) / 55.5% (position shift).

### 실험 하이퍼파라미터

- VLM: Qwen3-VL-2B-Instruct
- UPP: 4-layer transformer, hidden 768, 50-step state horizon
- MoE: 16 routed + 2 shared experts, top-1, load-balancing 0.01
- Action head: 16-layer DiT-B, hidden 1024, 100-step chunks
- Training: 200k pretraining + 20k per-task fine-tuning, ~1K A100 GPU hours
- Latency: 73.34 ms on RTX 4090 (≈13.6 Hz)

---

## 4. 한계 및 미해결 문제

1. **MoE의 routing 안정성**: Expert balance collapse 가능성, 희귀 motion mode 누락.
2. **State-centric 대 action-centric의 trade-off**: 표현 일관성은 얻지만, 플랫폼별 state schema alignment가 필요.
3. **Flow matching latency**: 실시간 고주파 제어에서 inference speed 확보 난이도.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — State-centric + MoE + flow matching의 결합 방식이 참신 |
| **Practical impact** | ★★★★☆ — Humanoid whole-body VLA는 업계 핵심 문제 |

**강점**: Humanoid의 coordinated control이라는 실제 문제를 state representation 설계로 정면 공략. **약점**: MoE와 flow matching 모두 튜닝 난이도 높은 모듈이라 재현성 부담.

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Body-part 독립 예측 대비 실제 실패 모드의 차이? | 팔-몸통-하체 간 반력/균형 관계 위반 동작 대폭 감소 |
| 2 | MoE가 진짜 부위별로 분화되는가? | Gating 분석 필요 (abstract에 세부 제시 없음) |
| 3 | 왜 flow matching을 선택했는가? | Continuous high-DoF action 분포 모델링에 강함, diffusion 대비 sampling 빠름 |
| 4 | GR00T, Helix 등 다른 humanoid foundation과 차별점? | State-aligned 표현과 MoE routing이 핵심 차별화 |

<!-- VERIFIED: pdf -->
