# SAMoE-VLA: Scene Adaptive Mixture-of-Experts VLA for Autonomous Driving

> **한 줄 요약**: Tsinghua AIR 외 5기관. **Scene-Adaptive MoE** (SA-MoE) layer가 transformer 4번째마다 FFN을 대체. OpenCLIP-ConvNext + BEVFormer + InternVL2-2B 기반 + flow-matching action head로 자율주행에 VLA 적용 — **nuScenes L2 0.29m (SOTA)**, LangAuto Driving Score 51.4 (full) / 69.5 (short).

---

## 1. 배경 및 동기

- 기존 VLA는 **manipulation 위주**, 자율주행은 BEV 표현이 핵심이라 ego-centric 입력 기반 VLA 구조와 맞지 않음.
- 자율주행에는 다양한 driving scenario (urban / highway / residential / 야간 등) 적응이 필요 → Scene-Adaptive routing이 자연스러운 해결책.
- nuScenes (open-loop) + LangAuto/LMDrive (closed-loop CARLA) 두 축에서 통합 평가 필요.

---

## 2. 방법론

### Multi-Modal Encoding
- **Vision**: OpenCLIP-ConvNext + CPFPN (multi-scale) + **BEVFormer** (BEV encoder)
- **Language**: InternVL2-2B (planning expert pre-train)
- 이전 VLM과 달리 BEV feature를 명시적으로 받아들이는 hybrid 입력.

### SA-MoE (Scene-Adaptive Mixture-of-Experts)
- Transformer **layer 4번마다 FFN을 SA-MoE block으로 대체**.
- Expert routing: scene-adaptive (BEV feature 기반).
- Conditional Cross-Modal Causal Attention으로 video / language / state간 inter-modal causality 보장.

### Action Head
- **Flow-Matching trajectory generator**: Gaussian noise → target velocity sequence interpolation.
- 자율주행 trajectory (steer / throttle / brake) 출력.
- ~3.6B params (InternVL2-2B + BEVFormer + flow head).

---

## 3. 실험 결과

### Table 1 — nuScenes Open-Loop Planning (L2 error, 1s/2s/3s)

| 모델 | 평균 L2 (m) | 3s L2 (m) |
|------|-----------:|----------:|
| VAD | 0.78 | — |
| UniAD | 0.73 | — |
| **SAMoE-VLA** | **0.29** | **0.35** |

→ 평균 L2 0.29m로 **15% 낮은 3s error** (0.35m vs 0.41m).

### Table 2 — LangAuto Closed-Loop (CARLA)

| Setting | DS | RC |
|---------|---:|---:|
| Full LangAuto | **51.4** | **63.5** |
| LangAuto-Short | **69.5** | — |

→ 7B baselines를 outperform, 3.6B로 SOTA.

---

## 4. 한계 및 미해결 문제

1. **Manipulation transfer 미검증**: 자율주행 특화로 manipulation/embodied AI에는 직접 적용 어려움.
2. **CARLA-only closed-loop**: real-world driving 검증 부재 → sim-to-real gap 미평가.
3. **BEV 의존**: BEVFormer의 multi-camera setup이 필수. Mono-camera 환경 (소형 로봇 등)에는 비현실적.
4. **3-second prediction limit**: 더 long-horizon driving (15s+, urban tour 등) 미검증.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA를 자율주행에 도입하면서 SA-MoE + BEV 통합 |
| **Practical impact** | ★★★☆☆ — nuScenes/LangAuto 양쪽 SOTA, 그러나 driving-only 적용 |

**강점**: 7B baselines를 3.6B로 outperform. SA-MoE의 sparse activation으로 효율성 확보.
**약점**: VLA-Tracker의 manipulation 중심 평가에 직접 비교 불가. Driving-specific solution.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Manipulation VLA와 driving VLA 통합 가능? | 이론적으로 가능하나 BEV vs ego-centric의 representation gap이 본질적. 두 representation 사이의 unified encoder 연구 필요. |
| 2 | SA-MoE가 매 4번째 layer에만 들어간 이유? | All-layer MoE는 학습 instability + 비용 큼. 4번째마다 sparse하게 두면 routing-FFN 다양성 + dense 안정성 trade-off 균형. |
| 3 | UniDriveVLA와 차이? | UniDriveVLA는 Qwen3-VL-based MoT이고 SA-MoE는 InternVL2 + BEVFormer + SA-MoE FFN. SAMoE-VLA가 nuScenes에서 강하나 UniDriveVLA가 Bench2Drive에서 더 강함 (78.37 vs SAMoE 51.4). |
| 4 | Flow-matching이 driving에 유리한가? | Trajectory의 continuous nature와 잘 맞음. Discretized action token (RT-2 같은) 대비 smooth control. |

<!-- VERIFIED: pdf -->
