# Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models

> **한 줄 요약**: 동결된 pi0.5 VLA의 flow-matching denoising 과정에 사용자 joystick 명령을 analytical Tweedie 기반 posterior guidance 항으로 주입하고, ResNet-18 wrist-cam 기반 phase-aware interaction detector로 접촉 단계에서만 사용자 개입을 요청하여, fine-tuning 없이 scene recovery에서 91.3% partial success를 달성하면서 NASA-TLX의 mental/physical workload를 teleop 대비 유의미하게 감소시킨 shared autonomy 프레임워크.

---

## 1. 배경 및 동기

- 보조 로봇은 (i) 사전 정의된 좁은 작업에 특화된 컨트롤러와 (ii) 일상 생활의 다양성 사이의 긴장 관계를 안고 있다.
- 최근 VLA는 open-world 일반화에 강하지만, RoboArena 기준 pi0.5 46.95%, pi0 35.25%로 zero-shot 신뢰성이 50% 미만이다.
- VLA를 재학습하면 비용이 막대하고, fine-tuning은 foundation policy의 광범위 사전지식을 좁은 specialist로 collapse시킬 위험(catastrophic forgetting)이 있다.
- 저자 관찰: 실패는 의미적(semantic) 실패가 아니라 grasp/insertion/release 같은 **국소 접촉 단계의 공간 정밀도 문제**에 집중되어 있다.
- 따라서 VLA는 동결하고 접촉 단계에서만 인간이 개입하는 shared autonomy가 자연스러운 해법.

---

## 2. 방법론

### Hybrid System Policy (Sec. 2)
시스템 정책은 binary intervention indicator I_int로 토글:
pi_sys(a|s) = (1 − I_int) pi_vla(a|s) + I_int pi_shared(a|s, u). pi_vla는 동결된 pi0.5, u는 joystick 명령. Whisper로 사용자 발화를 자연어 prompt로 전사하여 VLA에 전달.

### Interaction Detection (Sec. 3)
- ResNet-18 fθ가 wrist-camera I_wrist에서 interaction confidence p_it 예측.
- 이중 검증: I_ia = (p_it > τ_it) ∧ (|Δã_grip| > ε). 단순 visual proximity만으로는 false positive가 많아 VLA의 gripper-state 변화 의도와 결합.
- 최종 트리거: I_int = I_ia ∨ I_user, 사용자도 u ≠ 0이면 언제든 manual override 가능.

### Posterior Policy Blending via Flow Matching Guidance (Sec. 4, Appendix C)
- VLA를 action prior p(a_1)로 보고, joystick 측정 u에 대한 likelihood p(u|a_1) = N(a_1, Σ_u)를 가정.
- Bayes로 posterior p(a_1|u) ∝ p(u|a_1)p(a_1).
- Tweedie's identity로 conditional flow를 unconditional flow + guidance 항으로 분해:
  v̂(a_t, u) = v̂(a_t) + g(a_t, u), 여기서
  g(a_t, u) = ((1−t)/t)(u − â_1)^T [((1−t)^2/((1−t)^2 + t^2)) I + Σ_u]^{−1} (Eq. 6).
- 즉 **새 conditional flow 모델을 학습할 필요 없이** denoising 중 analytical 항만 추가하여 사용자 의도를 사후적으로 반영.

---

## 3. 실험 결과

### Scene Recovery Real-World Study (Sec. 5.1, N=17, Fig. 4)
- 5개 sub-task: drawer open, grape→drawer, avocado→box, marker→red cup, toothpaste→blue cup. 7분 timeout.
- **Partial success**: VLA 단독 13.7% / Direct Joystick 96.3% / **Assistron 91.3%**.
- **Completion time**: Assistron 324.5s vs Direct 305.9s (VLA는 거의 항상 timeout).
- **Active control time**: Assistron은 56.5% (joystick 41.7% + voice 14.8%), 나머지 43.5%는 자율 — teleop 대비 활성 제어 시간 약 절반.
- **Pearson r = 0.762 (p = 0.001)**: 초보 사용자일수록 Assistron의 시간 단축 효과 큼.

### Subjective (NASA-TLX & 만족도)
- 만족도: Assistron이 Quick / Easy to Use / Low Workload / Reuse에서 Direct Joystick 대비 유의미 상승; Wanted/Trust는 약간 하락.
- NASA-TLX: Assistron이 Mental, Physical effort에서 Direct Joystick보다 유의미하게 낮음 (p < 0.001). Frustration도 초보일수록 r = −0.564 (p = 0.023)로 감소.

### Policy Blending Ablation (Sec. 5.2, Fig. 6)
- 단일 task (grape→drawer)에서 Posterior(우리 방법) vs Linear blending vs Direct teleop.
- Posterior가 Direct 대비 completion time 유의미 감소, Linear/Direct 둘 다보다 trajectory length 유의미 감소.
- 핵심: Linear blending은 VLA의 multimodal action 분포를 무시해 사용자와 충돌; posterior는 latent action manifold에 일관되게 머묾.

### Interaction Detector (Sec. 5.3)
- 12k wrist-cam frames (gripper state change 전 2s window = positive).
- 224×224 입력만으로 test accuracy 81.2%, AP 84.5%.
- 정성 시각화(Fig. 7): drawer open과 marker insertion 모두에서 macro-reach → Auto, grasp/release 직전 → Assist 전환이 깔끔.

---

## 4. 한계 및 미해결 문제

1. **VLA 능력에 본질적으로 종속**: 의미적 실패(잘못된 grounding, 엉뚱한 영역으로 이동)는 local intervention으로 복구 불가. 저자도 명시.
2. **사용자 명령이 VLA action 분포에 있다는 가정**: posterior 공식 (4)이 사용자 의도가 VLA prior support 내부에 있다고 가정. OOD 사용자 행동 시 충돌, 향후 OOD detection으로 fallback 필요.
3. **단일 backbone (pi0.5)에서만 검증**: pi0, Octo 등 다른 flow/diffusion 기반 VLA로의 일반성 미확인.
4. **시뮬레이션 벤치마크 부재**: LIBERO/CALVIN/SimplerEnv 점수 없음 — 본질적으로 사용자 연구에 의존하므로 정량 비교가 제한적.
5. **Detector 일반화**: 12k frame, 단일 환경에서 학습된 ResNet-18이 다양한 조명/카메라 위치/그리퍼에서도 81.2% 유지될지 불확실.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Tweedie's identity를 통해 flow-matching VLA에 학습 없이 사용자 측정을 사후 주입하는 analytical guidance 유도가 깔끔하고, "frozen VLA + 국소 인간 개입"이라는 시스템 설계도 명확 |
| **Practical impact** | ★★★★★ — 17명 사용자 연구로 NASA-TLX의 mental/physical workload가 유의미하게 감소함을 보였고, 보조 로보틱스에 즉시 적용 가능한 패러다임 제시 |

Assistron의 진짜 기여는 두 가지다. (1) **공학적**: pi0.5를 fine-tune 없이 그대로 쓰면서도 contact-rich phase에서만 사용자를 개입시키는 phase-aware arbitration. (2) **이론적**: posterior inference를 flow matching guidance로 환원하는 analytical 유도로, learned conditional flow를 우회. 둘의 결합이 "VLA가 좀 더 똑똑해질 때까지 기다리지 말고 지금 사용 가능한 frozen VLA로 사용자 효과를 극대화하자"는 실용적 메시지를 강력하게 전달한다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 fine-tuning을 피했나? 보조 데이터로 fine-tune하면 더 좋지 않나? | Foundation policy의 broad behavioral prior가 좁은 specialist로 collapse하는 catastrophic forgetting을 막기 위해. pi0.5의 open-world generalization을 그대로 유지하면서 부족한 contact precision만 인간으로 보완. |
| 2 | Linear blending과 posterior blending이 본질적으로 다른 이유는? | Linear는 VLA 출력과 user 명령을 단순 가중합하므로 VLA action 분포가 multimodal일 때 두 mode 사이에서 충돌. Posterior는 Tweedie를 통해 user 측정에 조건화된 mode를 골라낸다(Fig. 6 trajectory length가 유의미하게 짧음). |
| 3 | Interaction detector 81.2%면 30% 이상의 phase에서 잘못 트리거된다는 뜻 아닌가? | AP 84.5%로 recall이 높음. False positive를 줄이기 위해 VLA의 gripper state change 의도(|Δã_grip| > ε)와 AND 조건을 둠. 즉 시각 신호만으로 개입을 강제하지 않음. |
| 4 | 왜 RoboArena 같은 시뮬 벤치 점수가 없는가? | 본 연구는 "사용자 효과(workload, completion)"를 측정하는 사용자 연구가 주제이므로 17명 사람 대상 NASA-TLX/satisfaction이 핵심 metric. 정량 시뮬 점수는 향후 작업. |
| 5 | u의 차원이 작을 때(joystick, low-bandwidth)도 high-DOF arm 제어가 가능한 이유는? | Posterior가 VLA의 high-DOF prior와 결합되기 때문. 사용자가 일부 차원만 지정해도 VLA가 나머지 차원을 prior로 채워주므로, 사용자는 한 번에 모든 DOF를 동시 제어하는 효과를 얻는다(논문은 이를 "controlling all DOFs simultaneously rather than sequentially"라고 표현). |

<!-- VERIFIED: pdf -->
