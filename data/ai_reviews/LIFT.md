# LIFT: Never Too Late for Force — Accelerating VLA Post-Training with Reactive Force Injection

> **한 줄 요약**: 사전학습된 vision-only VLA(pi0.5)에 "reactive action expert 복제 + causal force memory + zero-initialized cross attention"을 뒤늦게(late) 접목하여, 초기화 시점에는 원본과 output-equivalent를 유지하면서 6D 힘을 chunk 내에서 실시간 반영하고, online DAgger로 힘 분포 shift까지 흡수하는 force-aware post-training 프레임워크.

---

## 1. 배경 및 동기

- 사전학습 VLA(RT-2, Octo, OpenVLA, pi0/pi0.5 등)는 web-scale vision-language prior 덕분에 언어 조건 조작에 강하지만 **주로 vision에 의존**한다. Occlusion, 부정확한 depth, 부분 관측 등 contact 상태가 시각적으로 모호할 때 실패한다.
- 힘/토크는 사전학습 단계에서 대규모로 수집하기 어렵다(수집 비용, 로봇·엔드이펙터 플랫폼 의존, 셋업별 편차). 따라서 **force-aware post-training**이 자연스러운 다음 단계.
- 이때 3가지 질문이 생긴다: (Q1) 고주파·저차원·contact 결합된 힘을 어떻게 주입해야 빠르게 반응하고 유용한 force memory를 유지하나? (Q2) 새 force 경로가 사전학습 prior를 훼손하지 않게 어떻게 보존하나? (Q3) policy 의존적이고 offline이 커버하지 못하는 힘 분포 shift에서 어떻게 post-training을 유지하나?

---

## 2. 방법론 심층 분석

세 목표(O1/O2/O3)를 축으로 설계.

### O1. Reactive force injection
- **Reactive action expert**: 원본 pi0.5 action expert 옆에 두 번째 expert를 세우고, chunk 내 action을 **causal**하게 디코딩. contact이 들어올 때 action을 하나씩 갱신(refresh)할 수 있게 함.
- **Causal force-injected cross attention**: 최근 6D end-effector wrench를 causal force memory로 인코딩하고, latency-aligned causal mask를 가진 cross attention을 통해 **reactive expert에만** 주입. 추론이 끝나는 시점의 최신 힘을 쓰도록 정렬.
- **Cached Vision-Language Slow Context**: 느린 VLM prefix를 한 번 계산해 KV cache로 저장하고, chunk 내에서는 최신 force history만 재인코딩해 action을 재평가. 매 refresh마다 full VLM forward를 기다리지 않아 실시간 contact 반응 가능.

### O2. 사전학습 prior 보존 (초기화 등가성)
- **Output-equivalent reactive expert (O2.1)**: 원본 action expert 가중치를 복제. Shifted causal attention을 써서 각 reactive token r_i가 VLM prefix + 이후 base-action token a_{i+1:H-1} + causal reactive prefix r_0:i를 attend → 초기화 시 r_i가 원본 full-attentive expert의 a_i와 동일한 context를 받음. 그림 3의 ①③가 초기화 등가.
- **Zero-initialized cross attention (O2.2)**: force cross attention의 output projection을 0으로 초기화 → step 0에서 force residual이 정확히 0. 따라서 post-training이 nonzero residual을 배우기 전까지 힘이 원본 action output을 바꾸지 않음.

### O3. 이질적 데이터 동시 학습
- **Additive flow-matching objective (O3.1)**: base/reactive 두 stream을 같은 forward에서 flow-matching loss로 동시 학습, gradient 누적. 추론 시 두 stream 모두 계산하되 **reactive action만 컨트롤러로 전송**.
- **Selective force masking (O3.2)**: 실제 힘 데이터가 있는 샘플만 force 경로 활성화. vision-only 배치는 zero placeholder + mask로 force encoder/attention gradient 차단.
- **Equal sampling (O3.3)**: RLPD symmetric sampling을 따라 offline task-alignment : online correction = 1:1.

---

## 3. 데이터 전략

- **Stage 1 (offline, handheld)**: 힘 센서 없는 handheld iPhone 그리퍼로 10 Hz 수집. iPhone main camera RGB + ARKit SLAM pose, iPhone camera frame 기준. 에피소드 앞뒤 정적 구간 제거.
- **Stage 2 (online correction)**: Flexiv TDK bilateral master-follower 셋업(두 대의 Flexiv Rizon 4S)으로 수집. 각 timestep에 synchronized 6D wrench + RGB 저장. robot **command**를 저장(state 아님)해 정확한 제어 supervision. **human-intervention correction 데이터만** Df에 추가.
- Action은 모두 **relative action**, shared output dim(32)으로 padding.

---

## 4. 시스템/학습 세부사항

- **하드웨어**: Flexiv Rizon 4S 7-DoF cobot + Robotiq 2F-85 그리퍼, position-control 10 Hz, 내부 impedance controller가 target 추종. 내장 6D F/T 센서(>1000 Hz)를 10 Hz로 downsample·동기화. wrist-mounted iPhone RGB 10 Hz, single-camera.
- **Force encoder**: 단일 layer causal GRU + linear projection, hidden 512, 출력 dm=1024. Latency-alignment offset L=3.
- **Backbone/hyperparam (Table 1,2)**: pi0.5(PaliGemma tower + action expert), action chunk horizon 10, padded action dim 32. Batch 32, 총 30,000 step, constant lr 5e-5, AdamW(β1=0.9, β2=0.95, wd 1e-10), grad clip 1.0.
- **System (Fig.4, App.H)**: SOP 스타일 adaptive sampler(W=200, α=1.5, ω_on clip [0.2,0.8]), checkpoint를 100 step마다 inference server로 push. openpi 기반 코드.

---

## 5. 실험 결과 심층 분석

3개 실물 로봇 task(각 checkpoint당 10 rollout). 비교군 5종: pi0.5 w/ Online DAgger, LIFT w/o Reactive Force Injection(single-frame force), LIFT w/o Online DAgger, pi0.5 w/ Offline Handheld Data, LIFT(full).

- **Towel folding** (graded 0.25/0.5/0.75/1.0): pi0.5+DAgger 0→0.725(#3.1K). LIFT는 0.65를 #2.3K에서 도달, 최대 0.825(#2.8K). 흥미롭게도 **LIFT w/o Reactive Force Injection**이 이 task에선 더 빨라 0.925(#2.4K)~0.95(#2.8K). 얇은 towel의 depth 모호성을 힘이 직접 노출.
- **Book insertion** (0.5 slot 진입 / 1.0 완전 삽입): **LIFT 최고 0.6(#4.6K)** vs pi0.5+DAgger 0.4(#5.6K). 완전 삽입 후에도 push 지속 여부(bottom-out 판단)를 힘으로 학습.
- **Hanoi ring placement** (binary): LIFT 0.6(#1.7K), 최종 checkpoint에서도 0.6 유지(#2.0K). pi0.5+DAgger는 0.3(#1.4K) peak 후 0.2(#1.6K)로 하락. 10 mm 폴과의 미세 정렬을 힘 방향/크기로 보정.

전반적으로 힘이 **post-training을 가속하고 peak 성능을 높인다**(Q1).

---

## 6. Ablation 분석 (reactivity, online data)

- **Q2 reactivity (single-frame vs reactive memory)**: Book insertion에서 LIFT 0.6 vs single-frame 0.4(#5.0K,#5.3K). Bottom-out 후 계속 미는 non-Markovian 실패를 force memory가 contact phase 추론으로 해결. Hanoi에서 single-frame은 0.5(#0.7K) 후 0.1/0/0.1로 붕괴 — 순간 force noise/OOD 충격에 취약. reactive memory가 짧은 window로 평균화해 안정화.
- **Q4 online data**: **LIFT w/o Online DAgger**는 3 task 모두 저조, book insertion에선 0으로 추락. offline은 실제 실행 중 급격한 force 패턴 변화를 못 담고, online DAgger가 학습자 실패 상태에 대한 correction을 공급.

---

## 7. 관련 연구 비교

- **Force-augmented policies**: FoAR(force-aware gating diffusion), ForceMimic(force-motion capture + hybrid force-position IL), ACP(approximate compliance), FACTR, DexForce, Force Policy(interaction frame). 아키텍처적으로는 RDP(slow-fast) 및 ImplicitRDP(end-to-end causal attention)의 reactive force control에서 직접 영감.
- **Force-aware VLA**: ForceVLA, ForceVLA2, TA-VLA가 가장 근접하나, 이들은 post-training 전반에 힘을 쓰되 각 chunk를 여전히 **open-loop**로 실행. LIFT는 **late reactive injection**으로 chunk 내에서 contact을 반영하면서 원본 VLA 능력을 보존.
- **Post-training/DAgger**: HG-DAgger, SIRIUS/Sirius-Fleet, CR-DAgger(compliant residual). LIFT는 residual action head가 아니라 **zero-init cross attention**으로 힘을 주입하고 배포 컨트롤러를 position-based로 유지한다는 점에서 차별화.

---

## 8. 한계 및 미해결 문제

1. **Human correction 의존**: online DAgger가 사람 개입 correction에 의존 → 데이터 throughput 제한.
2. **Single-arm에 국한**: 평가가 단일 팔 조작에 한정. 양팔/다양한 엔드이펙터 미검증.
3. **Task별 상충 결과**: towel folding에서는 reactive memory 없는 single-frame 변형이 더 우수 → force reactivity의 이득이 task contact 특성에 따라 달라짐.
4. **플랫폼 특이성**: Flexiv Rizon 4S + 특정 F/T 센서 스택에 맞춘 post-training이라 이식성은 미평가(Limitations에서 다양한 arm/센서/엔드이펙터로의 확장을 future work로 명시).

---

## 9. 이론적/설계적 기여

- **초기화 등가성(initialization equivalence)**: 복제 가중치 + shifted causal attention + zero-init cross attention의 조합으로 augmented policy가 학습 전 **정확히** base pi0.5와 동일 출력을 보장. 구조 변경으로 인한 prior 망각을 원천 차단하는 것이 핵심 설계 논리.
- **Latency-aligned causal mask (App.B)**: force memory에 대한 causal mask b_ij^{(L)}(j ≤ i-L이면 0, 아니면 -∞)로 아직 관측 불가능한 힘을 차단하고, 추론 지연 후에도 completion time에 실행할 action을 출력하도록 window를 shift.

---

## 10. 시스템/추론 루프 (Algorithm 1)

- 매 chunk 시작 시 VLM prefix(KV) 캐시 → chunk 내 for i=0..H-1: 최신 force frame 취득 → force memory 재인코딩 → cached prefix + latency-aligned mask로 DenoiseAction → 즉시 Execute. chunk 끝에서 slow context/noise/force history refresh.
- 이 구조가 "느린 VLM 1회 / 빠른 force refresh 다회" 라는 slow-fast 분리를 실현.

---

## 11. 실무적 시사점

- 기존에 사전학습된 vision-only VLA(pi0.5류)를 **재학습 없이** 특정 로봇·센서 스택에 force-aware로 adapt하는 실용적 recipe. openpi 위에 구현되어 재현 가능성이 높음(코드 공개 예정).
- Zero-init cross attention + 가중치 복제 패턴은 다른 modality(촉각, audio 등) late injection에도 일반화 가능한 템플릿.
- Contact-rich 삽입/정렬/그랩 성공률이 낮은 배포 환경에서 online DAgger + force memory 조합이 실패 상태 회복을 유의미하게 개선.

---

## 12. 종합 평가

LIFT는 "force를 사전학습에 넣을 수 없다면, post-training에서 늦게라도 넣자"는 문제의식을 **prior 보존을 수학적으로 보장하는 초기화 등가 설계**로 정교하게 푼 연구다. Reactive action expert 복제 + causal force memory + zero-init cross attention의 3-way 결합은 힘 주입으로 인한 성능 저하 리스크를 초기화 시점에 제거하고, online DAgger가 policy-dependent force shift를 흡수한다. 실물 3-task(towel/book/Hanoi)에서 vision-only DAgger 대비 더 빠른 학습과 더 높은 peak를 보이며 OOD 일반화도 유지. 다만 human correction 의존과 single-arm 한정, task별로 reactivity 이득이 엇갈리는 점은 확장 연구가 필요하다. 아키텍처 관점에서 force-aware VLA post-training의 견고한 baseline을 제시한 기여가 크다.

<!-- VERIFIED: pdf -->
