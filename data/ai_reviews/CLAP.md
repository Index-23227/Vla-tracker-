# CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding

> **한 줄 요약**: 사전학습 VLM을 아키텍처 변경 없이 VLA로 직접 변환할 때 발생하는 "출력 분포 불일치"를, 숫자 action token 앞에 템플릿 기반 자연어 action 설명을 붙여 causal하게 조건화함으로써 해결한다. 단일 epoch 파인튜닝만으로 2B CLAP이 LIBERO 90.8%(VLA-0 대비 +14.9pt)를 달성하고 0.8B/2B/4B open-weight 계열을 공개한다.

---

## 1. 배경 및 동기

- VLA는 사전학습 VLM의 의미 능력을 상속하지만, 대규모 robot 데이터 post-training과 아키텍처 수정(action expert, diffusion/flow head 등)이 backbone을 크게 재구성해 "VLM이 제어에 실제로 무엇을 기여하는지" 분리하기 어렵게 만든다.
- 핵심 장애물은 **출력 분포 불일치(output-distribution mismatch)**: VLM은 의미적으로 구조화된 언어를 생성하도록 사전학습되었는데, 표준 VLA 파인튜닝은 `4 12 98 3 0 0` 같은 bare numeric token만 뱉도록 강제해 사전학습 표현을 훼손한다.
- 이 문제는 경량 backbone에 의존하는 compact VLA에서 특히 민감하다.
- 기존 architecture-free 접근은 (i) bare action token 예측(VLA-0) 또는 (ii) 언어만으로 action을 대체(VLM2VLA)하는 양극단이라, 직접 실행성과 언어 분포 정렬 사이 trade-off가 남는다.

---

## 2. 방법론

### Causal Language-Action Prediction (Sec. 4)
- 하나의 autoregressive 시퀀스 안에서 **언어 설명 d를 먼저, 그 다음 숫자 action token a를** 생성한다(Eq. 2). causal attention이 숫자 token을 앞선 언어 prefix에 조건화하므로, prefix는 보조 예측이 아니라 "조건화 중간표현"이 된다.
- 학습 시 언어 설명은 ground-truth action chunk로부터 고정 템플릿 d̃ = T(ã)로 결정론적으로 생성 → 수작업 annotation 불필요. 추론 시에는 d와 a 모두 모델이 생성.
- 손실은 표준 autoregressive cross-entropy(Eq. 3), 동일한 backbone/tokenizer/output head/loss를 언어·action 양쪽에 사용. **action expert, vocabulary 확장, 아키텍처 변경 없음.**

### 템플릿 구성 (Appendix F)
- 7-DoF chunk의 per-step delta를 합산해 cm(위치)·degree(회전, 10° 단위 반올림)로 변환, 부호에 따라 방향 단어(move forward/back, tilt left/right, rotate clockwise 등) 매핑. gripper는 마지막 step 상태.
- 최종 타깃: `<think> d </think> n1 n2 ... n_{7h}`. 추론 시 `</think>`까지 stripping 후 숫자 token만 controller로 전달.

### Action masking (선택적 augmentation)
- 입력 action token 일부를 `?`로 무작위 치환(비율 ~ Uniform(0, 0.4), 40% 확률로 미적용), 원값을 예측 타깃으로 유지. 숫자 tail에만 적용해 언어 prefix 정렬을 보존. 기본 설정은 unmasked.

---

## 3. 실험 결과

### LIBERO (Table 2, 1 epoch, h=8, matched protocol)
- **CLAP 2B: Spatial 93.0 / Object 97.4 / Goal 90.8 / Long 82.0 / Avg 90.8** (VLA-0 2B 75.9 대비 **+14.9pt**).
- 0.8B: 89.6 (+13.5), 4B: 84.9 (+20.7). **비단조 스케일링** — 2B > 4B.
- 맥락 참조(gray rows, 조건 불일치): full-training VLA-0(3B) 94.7, π0.5 96.85, OpenVLA 76.5, SmolVLA 88.8.

### LIBERO-PRO OOD (Table 4)
- unmasked CLAP의 평균 OOD gain: 0.8B +5.5, 2B +11.1, 4B +10.9pt.
- 가장 큰 향상은 Spatial suite의 novel visual instance: 4B에서 unmasked +42.6pt, masked +54.4pt.

### VLABench (Fig. 3) — 파인튜닝 전 backbone 능력
- 대부분 카테고리에서 크기 클수록 개선되지만 Complex·Physics Law에서 0.8B/2B가 4B와 동급 이상. CoT 제거가 대부분 성능을 높임(긴 reasoning trace가 format 위반/loop 유발).

### Ablation (Table 3)
- action masking은 일관되지 않음: 0.8B −3.9, 2B −1.7, 4B +3.2. → 핵심 구성요소가 아닌 validation-dependent augmentation.

### Real-robot (Appendix B, UR5e, Table 6)
- 120 demo, 20 trials/조건. 2B ID/OOD 모두 60%, 0.8B ID 35%/OOD 10%. 실세계에서 capacity gap이 LIBERO보다 훨씬 큼.

### Latency (Table 7)
- CLAP 0.8B 4.23s(1.89Hz), 2B 4.31s(1.86Hz), 4B 6.01s(1.33Hz). prefix가 VLA-0 대비 +1.0s(+32%, ~149 token) 추가하나 GPU 메모리 동일.

---

## 4. 한계 및 미해결 문제

1. **단일 VLM 계열(Qwen3.5)·제한된 benchmark**: 다른 backbone·embodiment·실세계 일반화는 미검증.
2. **추론 속도**: autoregressive token 생성이 parallel decoding(diffusion/flow head)보다 본질적으로 느림. speculative decoding·quantization은 적용 가능하나 미탐구.
3. **비단조 스케일링의 원인 불명**: 2B가 4B를 앞서는 이유(데이터 양 대비 용량, 사전학습 편차 등)에 대한 mechanistic 설명이 부족.
4. **실세계 데이터 부족**: 120 demo 규모로 성공률이 시뮬레이션 대비 크게 낮아, 최소 recipe가 real-world에서 얼마나 확장되는지 미해결.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "출력 표현만 바꾼다"는 최소주의 recipe가 명료하고, 언어 prefix를 최종 policy 출력 내부에 두어 조건화하는 설계가 LAP/VLM2VLA와 차별화됨 |
| **Practical impact** | ★★★★☆ — action expert 없이 단일 epoch·8GPU 6.5시간으로 90% LIBERO 달성, 0.8/2/4B open-weight 공개로 VLM-to-VLA 전이의 통제된 분석 플랫폼 제공 |

CLAP은 새로운 모듈을 더하는 대신 "무엇을 예측하게 할 것인가"라는 출력 표현 문제에 깔끔한 답을 낸다. 언어-action prefix가 학습 효율(+14.9pt)과 OOD robustness를 동시에 끌어올린다는 점, 그리고 파라미터 수가 전이 품질을 결정하지 않는다는 비단조 관찰이 이 논문의 핵심 메시지다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 언어 prefix가 왜 학습 효율을 높이나? | prefix가 VLM의 사전학습 생성 분포에 가까운 중간 타깃을 제공해, bare integer 매핑보다 의미 구조를 보존하고 숫자 token 예측을 조건화하기 때문. |
| 2 | ECoT/reasoning VLA와 무엇이 다른가? | ECoT류는 auxiliary reasoning token·visual trace에 추가 supervision/추론 오버헤드가 큼. CLAP은 템플릿에서 결정론적으로 생성되는 짧은 고정 prefix라 annotation·자유형 reasoning이 불필요. |
| 3 | LAP과의 차이는? | LAP은 언어-action 설명을 pretraining 신호로 쓰되 최종 action 생성에 별도 action expert에 의존. CLAP은 prefix를 최종 autoregressive policy 출력 안에 직접 두고 실행 가능한 숫자 token을 유지. |

---

## 7. 아키텍처 상세

- Backbone: Qwen3.5 VLM(0.8B/2B/4B), 기존 vocabulary·output head 그대로 사용.
- Action: 7-DoF(end-effector translation·rotation·gripper), K=1000 bin 이산화, h=8 chunk를 flatten한 56개 정수 token.
- 입력: system message + 2개 카메라뷰(agentview+wrist를 224×224로 tiling) + 언어 instruction. Qwen3.5 built-in extended-thinking(enable_thinking=True) 사용, `</think>` 이후 56 정수만 controller로 전달.
- 출력 형식: `<think> 언어 설명 </think> 숫자 token`.

---

## 8. 학습 세부

- 8×H200, GPU당 micro-batch 16 → effective batch 128(DDP). 1 epoch = LIBERO 4 suite 1회 통과 ≈ 17,000 step(~2.18M sample).
- AdamW, lr 5e-6(constant), weight decay 1e-10, bf16, Flash Attention 2. Image aug: random crop(scale 0.875)·color jitter. Action masking rate 0.4(선택).
- 실세계: Qwen3.5-0.8B/2B를 LIBERO 파인튜닝 없이 사전학습 VLM 체크포인트에서 직접 120 real demo로 파인튜닝.

---

## 9. 벤치마크 위치

- **LIBERO 2B Avg 90.8%**는 robot pretraining 없이 단일 epoch로 SmolVLA(88.8)·OpenVLA(76.5)를 상회하고 full-training π0.5(96.85)에 접근.
- 동일 backbone·데이터·gradient step으로 재현한 VLA-0가 primary baseline이라, 향상분이 순수하게 언어-action prefix 효과임을 분리.
- SmolVLA 전 사이즈 비교(Table 9): CLAP 0.8B/2B가 세 SmolVLA 스케일을 평균에서 match/상회, CLAP 4B는 SmolVLA 2.25B 아래.

---

## 10. 재현성 및 공개

- 저자들은 weights와 code를 공개 예정(open-weight 0.8B/2B/4B 계열)이라 명시. 논문에 하이퍼파라미터(Table 8), 템플릿 구성(Appendix F), 평가 프로토콜(Appendix E), masking 세부(Appendix G) 상세 기재.
- 단, 본 리뷰 작성 시점 기준 실제 repo·weight 링크는 논문에 URL로 제시되지 않음("upon publication").

---

## 11. 실용적 시사점

- action expert·vocabulary 확장 없이 표준 VLM 파인튜닝 파이프라인만으로 배포 가능한 compact VLA를 만들 수 있음 → VLM 커뮤니티 발전(distillation·quantization·신규 backbone)을 재설계 없이 흡수 가능.
- 추론은 느리지만(1.3–1.9Hz) speculative decoding/quantization 여지가 큼. h=8 open-loop 실행으로 실제 제어율 확보.
- compact 영역에서 무작정 큰 모델이 답이 아님(2B 최적) → 배포 비용·성능 균형점 선택에 실질 가이드.

---

## 12. 결론

CLAP은 VLM-to-VLA 적응을 "출력 표현 변경"이라는 최소 개입으로 재정의한다. 숫자 action token 앞에 템플릿 기반 언어-action 설명을 causal하게 붙이는 단순한 아이디어가 학습 효율(+14.9pt on LIBERO 2B)과 OOD robustness를 동시에 향상시키며, 파라미터 수가 전이 품질을 결정하지 않는다는 비단조 스케일링을 드러낸다. VLA 파인튜닝을 표준 VLM 파인튜닝만큼 경량·투명하게 만들려는 방향에서 의미 있는 한 걸음이다.

<!-- VERIFIED: pdf -->
