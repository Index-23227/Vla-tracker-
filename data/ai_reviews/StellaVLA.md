# StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models

> **한 줄 요약**: 검색된 expert demonstration을 "무엇을 했는가(raw observation+action)"가 아니라 "왜 그렇게 했는가(sub-goal + 언어화된 2D/3D motion)"라는 구조화된 rationale로 변환해 in-context prefix로 주입하고, 학습 시에만 붙는 parallel spatial-language expert로 그 reasoning을 backbone에 내재화한 뒤 추론 시엔 제거하는 Qwen3-VL-4B 기반 VLA. LIBERO 98.8%, LIBERO-Plus 85.1%, VLA-Arena overall 0.63(1위, π₀.₅ 0.44)을 기록.

- **arXiv**: 2608.11671 (v1, 2026-08-12) · **저자**: StellarEdge AI Technical Team (Siyu Xu, Yunke Wang, Zijian Wang, Dihao Zhu, Chenghao Xia, Chengbin Du, Daochang Liu, Tao Huang, Chang Xu)
- **백본**: Qwen3-VL-4B-Instruct + OpenVLA-OFT 스타일 MLP action expert (L1 regression)

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA는 scene/viewpoint/object가 학습 분포를 벗어나면 성능이 급락하며, 회복하려면 데이터 재수집 + fine-tuning이 필요하다.
- In-Context Imitation Learning(ICIL)은 weight update 없이 test-time 적응을 제공하지만, 기존 방식은 검색된 trajectory를 **raw observation + continuous action** 형태로만 붙인다. 이는 표면적 모방(surface-level imitation)을 조장한다.
- 논문의 진단이 날카롭다: 구조가 없으면 정책은 demonstration을 **노이즈로 취급하고 pretrained prior로 되돌아간다**. 저자들은 이를 *behavioral inertia*라 부른다.

### 핵심 질문
- **검색된 demonstration 안의 reasoning을 어떻게 표면화(surface)해서, 정책이 motion이 아니라 expert의 사고를 모방하게 만들 것인가?**
- **그 reasoning을 넣으면서도 고주파 제어에 필요한 실시간성(autoregressive decoding 없는)을 유지할 수 있는가?**

📌 [Figure 1 삽입] — Human/XR/Robot demonstration → 구조화 → retrieval → in-context prompt → action expert + (학습 전용) spatial-language expert

---

## 2. 방법론 심층 분석

### 2.1 전체 파이프라인 3단계

| 단계 | 내용 | 비용 |
|------|------|------|
| (1) Offline structured context extraction | raw trajectory τ = {(oₜ, aₜ)} → rationale-augmented τ_rat = {(oₜ, aₜ, sₜ, l_k)} | zero human annotation |
| (2) Parallel dual-training | 검색된 prefix P_demo 조건화 + action/language 동시 최적화 | 학습 시에만 |
| (3) Asymmetric inference | language expert 제거 + prefix KV cache | 지연 거의 0 |

### 2.2 Offline Structured Context Extraction

- **Semantic segmentation via causal deduction**: off-the-shelf VLM(Qwen3-VL)이 task instruction I와 함께 연속 trajectory를 K개의 의미 단위 segment로 분해하고, 각 segment의 sub-goal을 시각 변화로부터 식별한다. "결과(물리적 실행)"에서 "원인(expert의 rationale)"을 역추론한다는 프레이밍.
- **Kinematic verbalization (language-as-action)**: 결정론적 verbaliser **Φ**가 임의의 연속 action span을 (a) workspace 3D 변위, (b) 알려진 intrinsic/extrinsic으로 image plane에 투영한 2D 변위로 텍스트화한다. 예: *"Move the end-effector by Δx=+0.05, Δy=−0.02, Δz=+0.10 and close gripper"*. 별도의 action token vocabulary를 도입하지 않는 것이 포인트.
- 최종 구조: **Semantic Rationale(sub-goal 설명)** + **Kinematic Rationale(3D movement + 2D trace)**. sub-goal 라벨 sₜ는 매 프레임에 붙고, segment-level rationale l_k는 구간 전체에 붙는다.

### 2.3 Retrieval-augmented prompt

- 학습은 **leave-one-out retrieval**: 타깃 trajectory τ_tgt를 pool에서 제외하고 나머지에서 top-1을 검색.
- 검색 신호는 **task instruction의 language embedding cosine similarity 뿐** — 시각 특징은 쿼리에 들어가지 않는다. 이것이 다른 embodiment(사람 손, XR)에서 기록된 demo를 로봇 에피소드에 검색해줄 수 있게 하는 핵심 장치.
- 입력: xₜ = P_demo ⊕ I ⊕ oₜ.

### 2.4 Parallel experts

공유 latent hₜ = f_θ(xₜ)를 두 head가 **병렬로** 읽는다:

1. **Action expert**: 2-block residual MLP → 연속 action chunk Â_t (길이 H), L1 loss.
2. **Spatial-language expert**: native autoregressive LM head → ĉₜ = (ŝₜ, m̂ₜ) = (현재 subtask, 같은 action chunk의 2D/3D 언어 렌더링), cross-entropy.

목적함수: **L = L_act(Â_t, A_t) + λ·L_lang(ĉₜ, cₜ)**, λ = 0.3.

> ❓ **예상 질문**: 이건 결국 Embodied CoT 아닌가?
> **답변**: 아니다. ECoT는 텍스트를 먼저 생성하고 그 토큰에 action이 **causally 의존**한다. 여기선 두 expert가 hₜ만 공유하고 서로 의존하지 않는다("They share the representation hₜ and nothing else"). 그래서 추론 시 language head를 통째로 떼어내도 action path가 깨지지 않는다.

### 2.5 Asymmetric inference + prefix caching

- 추론 시 spatial-language expert는 **완전 제거**. backbone 1회 forward + MLP만으로 action chunk 출력.
- P_demo는 에피소드 동안 불변이므로 KV cache를 t=1에 한 번만 계산하고 이후엔 live suffix(현재 관측)만 forward.

---

## 3. 데이터 전략

- **시뮬레이션**: LIBERO(표준 4 suite), LIBERO-Plus(7축 perturbation), VLA-Arena(11 suite, L0~L2 — 학습은 **L0 데이터만** 사용).
- **실기**: AgileX Piper 6-DOF, third-person + wrist RGB. **125 teleoperated 에피소드 / 71,702 프레임**, 4개 태스크(pen→cup 정밀 정렬, carrot→bowl pick-and-place, blocks→second drawer 후 닫기(long horizon, articulated), 그릇 3개 stacking).
- **Cross-source pool**: XR로 기록한 **human-hand 26 takes** + 동일 궤적을 Piper로 retarget한 **26 에피소드**(각 10,996 프레임). 두 데이터셋은 프레임 단위로 정렬되어 있어 **embodiment의 외형만이 유일한 변수**다 — cross-embodiment 실험 설계로서 매우 깔끔하다.
- 2D grounding point는 사람은 손목, 로봇은 flange로 통일. **off-embodiment 프레임은 오직 retrieval context로만 쓰이고, 실행 가능한 action target은 항상 실제 로봇 공간에서 나온다.**

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| 백본 | Qwen3-VL-4B-Instruct, full fine-tuning |
| Optimizer | AdamW (β=0.9/0.95, wd 1e-8) |
| LR | backbone 5e-6, action expert 1e-4, cosine, warmup 500 |
| 기타 | grad clip 1.0, bf16, DeepSpeed ZeRO-2 |
| Steps / batch | 30k steps, global batch 128 |
| Dropout | 2D gripper-path dropout 0.5; context-demo dropout 0.0(LIBERO/Plus) / 0.5(VLA-Arena, 실기) |
| Sim 세팅 | 256×256 이미지, EEF delta, chunk 길이 8, 최대 10 subgoal keyframe |
| 실기 세팅 | 640×480, 7차원 chunk 길이 16(joint delta 6 + absolute gripper width, q99 정규화), 최대 8 keyframe |
| 제어 스택 | CAN 30Hz, ROS 미사용, receding-horizon(horizon 8), One Euro filter + 25°/step jump guard |

---

## 5. 실험 설계 및 평가 프로토콜

논문이 던지는 네 가지 질문: (i) 구조화된 demo가 in-distribution과 shift 하에서 도움이 되는가, (ii) 정책이 실제로 demo를 **사용**하는가 그리고 어느 부분이 전이되는가, (iii) spatial-language supervision이 표현에 어떤 영향을 주는가, (iv) cross-embodiment context를 배포 효율 손해 없이 쓸 수 있는가.

**Matched control이 이 논문의 실험 위생에서 가장 좋은 부분이다**: `StarVLA-OFT`는 동일 backbone / 동일 action expert / 동일 데이터로 학습하되 **retrieved demo도 spatial-language supervision도 받지 않는다**. 따라서 두 컴포넌트의 합산 기여를 측정한다. 개별 기여 분리는 Table 5(고정 체크포인트에 correct/none/wrong demo 개입)로 처리.

- LIBERO: suite당 500 rollout.
- VLA-Arena: 공식 프로토콜, 베이스라인은 리더보드 수치.
- 실기: 셀당 10 rollout, 물체 배치 재랜덤화 + 방법 간 동일하게 매칭, 사람 채점.

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO (in-distribution, Table 1)

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| ACoT-VLA | 99.4 | **99.6** | 98.8 | 96.0 | 98.5 |
| AVA-VLA | 99.2 | **99.6** | 97.9 | 96.2 | 98.2 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 |
| Retrieval-VLA | 97.4 | 98.8 | 96.3 | 89.5 | 95.5 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| StarVLA-OFT (matched control) | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| **StellaVLA** | **99.6** | 99.0 | **99.6** | **96.8** | **98.8** |

matched control 대비 이득 분포가 해석에 유리하다: Object는 +0.4에 그치지만 **Goal +3.4, Long +3.0**. 현재 관측만으로는 의도한 결과나 sub-goal 순서를 결정할 수 없는 suite에서 context가 값을 한다.

### 6.2 VLA-Arena (task-level generalization, Table 2)

- Mean success: **L0 0.84 / L1 0.62 / L2 0.43**, Overall **0.63** (π₀.₅ 0.44, Evo-Depth 0.41, OpenVLA-OFT 0.39, Motus 0.34, GR00T-N1.6 0.28, LingBot-VLA 0.22).
- π₀.₅ 대비 마진이 L0 0.15 → **L1 0.24** → L2 0.17. 파라미터 업데이트 없이 타깃 태스크 demo 한 개만 주고 얻은 결과라는 점이 핵심.
- 세부 패턴이 흥미롭다: **Unseen Objects는 L0/L1/L2 모두 0.80으로 평평** — 지시 대상만 바뀔 때 절차가 그대로 전이된다는 증거. **Task Workflows는 L0 0.64 → L2 0.84로 오히려 상승** — 테스트 워크플로가 학습에서 멀어질수록 context의 상대 가치가 커진다.
- **Long Horizon은 L1/L2에서 모든 방법이 0 근처**(StellaVLA도 0.02/0.00). 저자들이 직접 인정: 고정된 prefix는 절차를 명시할 뿐 **실행 drift 후 re-plan을 못 한다**.

### 6.3 LIBERO-Plus (zero-shot robustness, Table 3)

| Method | Orig. | Cam. | Robot | Noise | Layout | Backg. | Light | Lang. | Avg |
|--------|-------|------|-------|-------|--------|--------|-------|-------|-----|
| OpenVLA | 76.5 | 1.1 | 4.1 | 19.3 | 31.6 | 25.3 | 4.4 | 26.8 | 16.0 |
| OpenVLA-OFT | 97.1 | 59.7 | 37.2 | 76.7 | 77.1 | 92.4 | 85.8 | 81.5 | 71.4 |
| π₀ | 94.2 | 15.8 | 6.6 | 79.4 | 70.4 | 78.5 | 79.6 | 61.0 | 53.8 |
| π₀-FAST | 85.5 | 66.4 | 24.8 | 75.8 | 70.3 | 67.7 | 73.0 | 63.3 | 62.5 |
| RIPT-VLA | 97.5 | 58.3 | 36.7 | 73.8 | 76.5 | 90.4 | 87.9 | 80.1 | 70.4 |
| StarVLA-OFT | 96.6 | 47.0 | 60.1 | 73.1 | 79.2 | **95.3** | **96.3** | 87.0 | 75.0 |
| **StellaVLA** | **98.8** | **70.5** | **74.8** | **92.8** | **79.3** | 95.2 | 95.7 | **95.3** | **85.1** |

matched control 대비 **+10.1점**. 이득이 큰 축은 camera viewpoint(+23.5), sensor noise(+19.7), robot initial state(+14.7), language(+8.3) — **관측이나 지시가 바뀌지만 태스크 절차는 그대로인** 축들이다.

반대로 **object layout은 +0.1에 불과**하다. 저자 해석: demonstration의 scene-specific spatial relation이 현재 장면과 더 이상 맞지 않기 때문. 즉 구조화된 context는 **절차(procedure)를 전이할 뿐 고정된 공간 궤적은 전이하지 못한다**. 이 negative result를 숨기지 않고 명시한 점은 신뢰도를 높인다.

### 6.4 실기 및 cross-embodiment

- in-distribution **85.0%**, OOD-L1 **75.0%**. L1 변형이 있는 두 태스크만 짝지어 비교하면 80.0% → 75.0%로 **5점 하락**, 반면 StarVLA-OFT는 25점, π₀.₅는 20점 하락(각각 자기 자신 대비).
- OOD-L2(학습 에피소드가 전혀 없는 first-drawer 태스크)는 **어떤 방법도 완수 못 함**. progress score(4점 만점)만 StellaVLA 1.9 vs π₀.₅ 1.5 vs StarVLA-OFT 1.1. 저자 표현대로 "partial progress이지 zero-shot 완수가 아니다".
- **Cross-source consistency (Table 4)**: 관측을 고정하고 검색 소스만 바꿨을 때 예측 action chunk의 정규화 L1 불일치가 real/human/XR 사이 **0.0014–0.0016σ**(관절각 0.02°–0.03°, gripper width 0.05mm 미만). demo를 아예 제거하면 0.0041–0.0045σ. 1,780 ID / 1,719 OOD 프레임 기준.

### 6.5 추론 지연 (Table 8)

| 구성 | Cache | Latency (ms) |
|------|-------|--------------|
| No demo | — | 64 |
| Image+Text | no | 183 |
| Image+Text | yes | **91** |
| Text-only | — | 109 |
| Action only | yes | 88 |
| + language decoding | yes | **3177** |

83 토큰의 spatial-language 출력을 실제로 디코딩하면 **약 36× 느려진다**. 이 한 줄이 "language expert를 추론에서 제거한다"는 설계 결정을 정당화하는 가장 강한 증거다. 실기 파이프라인 전체는 chunk당 ≈205ms.

---

## 7. Ablation 분석

### 7.1 Demonstration의 인과적 역할 (Table 5) — 이 논문의 백미

고정된 체크포인트에 주는 demo만 바꾼다:

| Demo | Sp. | Obj. | Goal | Long | AVG |
|------|-----|------|------|------|-----|
| Correct | 99.6 | 99.0 | 99.6 | 96.8 | **98.8** |
| None | 72.2 | 94.4 | 24.8 | 58.2 | 62.4 |
| Wrong | 72.6 | 52.0 | **0.0** | 55.0 | 44.9 |

논증이 우아하다: 정책이 context를 무시한다면 none과 wrong이 비슷해야 한다. 그런데 wrong이 none보다 **더 나쁘다**(62.4 → 44.9). 따라서 정책은 demo를 **task specification으로 능동적으로 사용**한다. Goal suite는 wrong demo에서 **0.0%**까지 떨어진다 — 유사한 물체/장면에 다른 결과를 요구하는 태스크라 demo가 의도 disambiguation에 결정적. 반면 Spatial은 none 72.2 / wrong 72.6로 거의 동일 — 현재 관측이 "어디서 상호작용할지"를 이미 알려주기 때문.

### 7.2 Spatial-language 구성요소

- 3D movement 제거: AVG 98.8 → 97.8
- 2D gripper path 제거: AVG 98.8 → **97.3**

2D가 3D의 투영에 불과함에도 제거 효과가 더 크다 → 2D는 **현재 이미지에 motion을 앵커링**하는 visual grounding 역할. 저자도 인정하듯 **semantic subtask label 단독 제거 실험은 없다**.

### 7.3 Demonstration modality (Table 7)

| Modality | Sp. | Obj. | Goal | Long | AVG | LP |
|----------|-----|------|------|------|-----|-----|
| *평가 시 변경 (고정 체크포인트)* | | | | | | |
| Image+Text (default) | 99.6 | 99.0 | 99.6 | 96.8 | 98.8 | **85.1** |
| Text-only | 99.0 | 99.8 | 99.8 | 96.6 | 98.8 | 84.4 |
| Image-only | 96.4 | 98.6 | 96.0 | 80.4 | 92.9 | 75.7 |
| *학습 시 변경* | | | | | | |
| Text-only | 98.6 | 98.6 | 99.2 | 92.8 | 97.3 | **85.0** |
| Image-only | 99.4 | 99.4 | 98.8 | 95.8 | 98.4 | 78.7 |

두 가지 통찰: (1) 평가 시 text-only가 full과 사실상 동률(98.8/84.4 vs 98.8/85.1) → **전이 가능한 정보는 대부분 구조화된 언어 표현에 담긴다**. image-only는 특히 Long에서 80.4로 붕괴. (2) 학습 시엔 trade-off가 뒤집힌다 — image-only가 in-distribution은 좋지만(98.4 vs 97.3) LIBERO-Plus에서 크게 나쁘다(78.7 vs 85.0). 시각 demo는 **appearance correspondence라는 shortcut**을 허용하고, 이 shortcut이 perturbation에서 무너진다는 해석.

### 7.4 λ (spatial-language loss weight, Table 6)

| λ | AVG | LP |
|---|-----|-----|
| 0 | 97.6 | **86.9** |
| 0.3 | **98.8** | 85.1 |
| 0.6 | 97.5 | 84.0 |
| 1.0 | 97.2 | 81.9 |

**중요한 반직관 결과**: in-distribution은 λ=0.3에서 얕은 역U자 정점이지만, **LIBERO-Plus robustness는 λ=0에서 최고(86.9)이고 λ가 커질수록 단조 감소**한다. 즉 spatial-language supervision은 OOD robustness를 단조 개선하지 **않으며**, robustness의 주된 원천은 context conditioning 쪽이다. 저자 가설: 큰 language weight가 offline annotation schema를 과대적합시킨다.

### 7.5 Context granularity (Table 9)

subgoal keyframe 3 → 98.1, 5 → 98.2, 8 → 98.3, 10(default) → 98.8. **0.7점 차이로 거의 평평**. dense trajectory replay가 아니라 **sub-goal 분해 자체**가 유용한 신호이며, 이것이 prefix를 캐시 가능한 길이로 유지시켜준다.

---

## 8. 관련 연구 비교

| 접근 | context의 역할 | StellaVLA와의 차이 |
|------|----------------|---------------------|
| Retrieval-VLA (CVPR'26) | training-free in-context 적응 | prompting만 사용, 명시적 supervision 없음. LIBERO 95.5로 낮음 |
| π₀.₇ | subtask instruction, subgoal image, metadata, control mode | 실행 **전략 지정**용. StellaVLA는 prior demonstration 자체를 구조화된 절차로 변환 |
| Qwen-RobotManip | 최근 observation-state-action chunk | 현재 에피소드의 **dynamics 적응**(execution history). StellaVLA는 명시적 task demonstration |
| ECoT / CoT-VLA / ACoT-VLA | autoregressive reasoning을 action 앞에 생성 | causal dependency → 3177ms 지연. StellaVLA는 parallel + 추론 시 제거 |
| UniVLA / LAPA (latent action) | 학습된 latent 변수로 behavior 인코딩 | VLM native semantic space 밖. StellaVLA는 VLM 어휘 그대로 사용 |
| Motus / X-op 등 cross-embodiment | shared latent action, retargeting, canonicalized action space | StellaVLA는 **표현 수준**에서 해결 — source action space 정렬 불필요, off-embodiment는 context로만 |

---

## 9. 한계 및 미해결 문제

저자들이 명시한 것과 리뷰어가 추가하는 것을 구분하면:

**저자 명시**
1. **Long Horizon L1/L2에서 0에 수렴** — 고정 prefix는 execution drift 후 re-plan 불가. 아키텍처적 한계.
2. **Object layout 변화에 취약(+0.1)** — demo의 scene-specific spatial relation이 전이되지 않음.
3. **Cross-source consistency는 single-step 예측 한정** — 작은 차이가 closed-loop에서 누적될 수 있고, 이 실험은 closed-loop invariance를 입증하지 않는다.
4. **OOD-L2 미완수** — progress 1.9/4는 부분 진전이지 zero-shot 완수가 아니다.
5. **subtask label 단독 ablation 부재**.
6. **10 rollout 셀당 노이즈 10–15점** — 개별 셀이 아닌 조건별 평균(ID 40, L1 20 rollout)으로만 결론.
7. **OOD-L1 severity가 태스크 간 보정되지 않음** — pen→cup은 색+물체 동시 치환, carrot→bowl은 색만. 태스크 **내부** 비교만 유효.

**리뷰어 추가**
8. 검색이 **instruction language embedding 단독**이므로 시뮬레이션에서는 사실상 exact task match로 축퇴한다(저자도 Appendix A에서 인정). 지시문이 open-ended한 실제 배포에서 검색 품질이 어떻게 되는지는 미검증.
9. **M=1(top-1)만 사용**. multi-demo 앙상블이나 검색 실패 시 fallback 전략 부재 — wrong demo가 no demo보다 나쁘다는 Table 5 결과를 고려하면 **검색 실패는 치명적 실패 모드**다.
10. offline annotator가 Qwen3-VL이고 policy backbone도 Qwen3-VL이다. annotation 품질과 policy 능력이 같은 모델에 묶여 있어 **오류 상관(correlated error)** 가능성이 있다.
11. 실기 데이터가 125 에피소드 / 4 태스크로 소규모. cross-embodiment 주장의 근거인 human/XR pool도 26 takes에 불과.

---

## 10. 총평

**강점**
- **실험 위생이 이 분야 평균보다 확연히 좋다.** StarVLA-OFT라는 동일 backbone/데이터/action space matched control, 고정 체크포인트에 대한 개입 실험(Table 5), 프레임 단위로 정렬된 human/XR 쌍(외형만 변수) — 인과 주장을 뒷받침하는 설계가 일관되게 깔려 있다.
- **negative result를 숨기지 않는다.** layout +0.1, Long Horizon 0, λ=0이 LIBERO-Plus 최고, image-only 학습이 ID에서 더 좋음 — 자기 주장에 불리한 수치를 본문에 그대로 싣고 해석한다.
- **"parallel + 추론 시 제거"라는 아키텍처 결정이 3177ms vs 88ms라는 숫자로 정당화**된다. Embodied CoT 계열의 실용적 병목에 대한 깔끔한 답.
- LIBERO 98.8은 이미 포화 구간이지만, **진짜 기여는 VLA-Arena overall 0.63(2위 0.44 대비 +43% 상대)과 LIBERO-Plus 85.1**이다.

**약점 / 유보**
- λ ablation이 보여주듯 **성능의 주된 원천은 context conditioning이고 spatial-language supervision은 in-distribution 정밀도에만 기여**한다. 논문 제목과 서사가 후자를 크게 다루는 것에 비하면 기여 배분이 비대칭적이다.
- retrieval이 언어 임베딩 단독이고 M=1이라 **검색 실패 모드가 그대로 노출**되어 있다(wrong demo < no demo).
- 코드/가중치 미공개, venue 미정(arXiv preprint), 기업 technical report 형식.

**한 줄 평가**: ICIL을 "궤적 모방"에서 "절차 모방"으로 옮긴 설득력 있는 작업이며, 특히 **무엇이 전이되고(절차) 무엇이 전이되지 않는지(공간 궤적, 장기 re-planning)를 실험으로 정확히 구획**한 점이 기여의 핵심이다.

---

## 11. 재현성 체크리스트

| 항목 | 상태 |
|------|------|
| 코드/가중치 | ❌ 미공개 (blog URL만) |
| 백본 | ✅ Qwen3-VL-4B-Instruct (공개) |
| Optimizer / LR / steps | ✅ Appendix A에 전부 명시 (AdamW, 5e-6/1e-4, 30k, batch 128, ZeRO-2) |
| Dropout / 정규화 | ✅ 명시 (2D path 0.5, context demo 0.0/0.5) |
| 벤치마크 프로토콜 | ✅ LIBERO 500 rollout/suite, VLA-Arena 공식, LIBERO-Plus [10] 기준 |
| 실기 재현 | ⚠️ AgileX Piper + XR 셋업 필요, 데이터 미공개 |
| Offline annotation 파이프라인 | ⚠️ Qwen3-VL 사용은 명시되나 프롬프트/필터링 세부 미공개 |
| 베이스라인 수치 출처 | ✅ 원논문/리더보드 인용 명시, matched control은 자체 학습 |

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 답변 / 논문 근거 |
|---|------|------------------|
| 1 | LIBERO 98.8은 이미 포화인데 0.3점 차이가 의미 있나? | 무의미하다고 봐야 한다. 저자도 LIBERO를 "in-distribution validation"으로만 쓰고, 실제 주장은 VLA-Arena 0.63 vs 0.44와 LIBERO-Plus 85.1 vs 75.0에 걸려 있다 |
| 2 | 정책이 demo를 정말 쓰는가, 아니면 무시하는가? | Table 5가 직접 답한다. wrong(44.9) < none(62.4) < correct(98.8). 무시한다면 wrong≈none이어야 한다 |
| 3 | λ=0이 LIBERO-Plus에서 더 좋은데(86.9 vs 85.1) spatial-language expert가 필요한가? | 논문의 가장 약한 고리. 저자 답변은 "in-distribution 정밀도(97.6→98.8)를 위해"이고, robustness는 context conditioning이 이미 제공한다고 인정한다. 즉 두 컴포넌트의 목적이 다르다 |
| 4 | Text-only demo가 full과 동률이면 subgoal 이미지는 왜 넣나? | 평가 시엔 거의 무의미(98.8/84.4 vs 98.8/85.1). 다만 **학습 시** modality는 다르다 — image 포함이 Long에서 우세(96.8 vs 92.8). 배포 시 text-only로 prefix를 줄이면 109ms → 캐시 시 91ms 대비 오히려 손해라는 점도 Table 8에 있다 |
| 5 | cross-embodiment 주장의 강도는? | 제한적이다. Table 4는 **관측 고정 single-step 예측 불일치**(0.0014–0.0016σ)만 보인다. 저자도 "closed-loop invariance를 입증하지 않는다"고 명시. 실기 결과는 mixed-source 샘플링이라 provenance를 분리하지 못한다 |
| 6 | Long Horizon이 L1/L2에서 0인 이유와 해법은? | 고정 prefix는 절차를 명시하지만 drift 후 re-plan 불가. 해법(에피소드 중 재검색, 계층적 재계획)은 논문 범위 밖 — 명확한 후속 연구 지점 |
| 7 | 검색이 틀리면? | 치명적. wrong demo가 no demo보다 17.5점 나쁘다. 그런데 검색은 instruction 언어 임베딩 단독 + top-1이고, 검색 신뢰도 추정이나 abstain 메커니즘이 없다 |
| 8 | Object layout에서 +0.1인 것이 방법론의 근본 한계인가? | 그렇다. 구조화된 context는 "무엇을 어떤 순서로"는 전이하지만 demo의 scene-specific 공간 관계는 현재 장면과 맞지 않는다. 2D trace가 visual grounding에 기여(제거 시 -1.5)한다는 점과 정확히 짝을 이루는 trade-off |
| 9 | annotator와 policy가 같은 Qwen3-VL인 것이 문제인가? | 논문은 다루지 않는다. annotation 오류와 policy 편향이 상관될 수 있고, 서로 다른 annotator VLM으로 교차 검증한 결과가 없다 |
| 10 | 실기 결과의 통계적 강도는? | 약하다. 저자 스스로 셀당 10 rollout에 10–15점 노이즈가 있다고 밝히고, 조건별 평균(ID 40, L1 20 rollout)과 paired degradation에만 근거해 결론을 낸다 |
| 11 | VLA-Arena에서 Task Workflows가 L0 0.64 → L2 0.84로 **오르는** 것은 어떻게 해석하나? | 저자 해석: 테스트 워크플로가 학습 분포에서 멀어질수록 pretrained prior가 덜 유용해지고 검색된 절차의 상대 가치가 커진다. behavioral inertia 가설과 일관된 증거 |
| 12 | 실제 배포 지연은 감당 가능한가? | 모델 측 88–91ms, 실기 파이프라인 전체 ≈205ms/chunk(16-step ≈0.53s, horizon 8). 제어 예산 안에 들어간다. 단 prefix caching이 전제이며, 에피소드 중 demo를 교체하면 이 이점이 사라진다 |

<!-- VERIFIED: pdf -->
