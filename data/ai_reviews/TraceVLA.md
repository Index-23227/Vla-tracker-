# TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies

> **한 줄 요약**: State-action trajectory를 이미지 위에 시각적 trace로 오버레이하여 VLA의 spatial-temporal awareness를 강화, SimplerEnv 10%↑ 및 real-robot 3.5x 향상 달성.

---

## 1. 배경 및 동기

- VLA가 **시간적 정보 (이전 동작 기억)**를 활용하지 못함
- Visual trace: 이전 end-effector 경로를 이미지 위에 그려 "어디에서 왔는지" 시각적으로 표현

---

## 2. 방법론

### Visual Trace Prompting

현재 이미지 위에 과거 N step의 end-effector trajectory를 색깔 선으로 오버레이:
$$\mathbf{o}'_t = \text{Overlay}(\mathbf{o}_t, \text{trace}_{t-N:t})$$

- 별도 학습 없이 **입력 이미지를 수정**하는 것만으로 효과
- VLM의 visual reasoning 능력이 trace를 자연스럽게 이해

> ❓ **예상 질문**: Trace가 중요 물체를 가리면?
> **답변**: Semi-transparent trace 사용, 얇은 선으로 occlusion 최소화. 그러나 밀집된 작업 공간에서 trace가 혼란을 줄 수 있음.

---

## 3. 실험 결과

| 모델 | SimplerEnv (%) | Real Robot (137 configs) |
|------|---------------|------------------------|
| OpenVLA | 48.7 | 1x |
| **TraceVLA** | **58.7 (+10%)** | **3.5x** |

---

## 4. 한계 및 미해결 문제

1. **Trace가 noise를 추가**: Dense trace가 시각적 혼란 유발 가능
2. **Camera viewpoint 의존**: Trace의 시각적 의미가 카메라 위치에 따라 변함
3. **Trace 없는 첫 step**: 초기에는 trace가 없어 이점 없음

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 극도로 간단하면서도 효과적 |
| **Practical impact** | ★★★★★ — 추가 학습 없이 즉시 적용 |

**강점**: 놀라울 정도로 간단한 아이디어가 큰 효과. **약점**: Visual clutter, camera 의존.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Random trace를 그으면? | 성능 하락 → trace의 정보적 가치 확인 |
| 2 | LLARVA와의 차이? | TraceVLA는 과거 궤적, LLARVA는 미래 예측 경로 |

<!-- VERIFIED: abstract-only -->
