# π0.5: A Vision-Language-Action Model with Open-World Generalization

> **한 줄 요약**: π0를 확장한 π0.5는 mobile manipulator 데이터 (~400h)와 다양한 non-mobile robot, web data, high-level subtask, verbal instruction 등 heterogeneous source의 co-training으로 **훈련에 본 적 없는 실제 가정의 침실·주방을 청소**할 수 있는 첫 end-to-end VLA. Discrete FAST 토큰으로 pre-train한 뒤 flow-matching action expert로 post-train하는 hybrid recipe가 핵심.

---

## 1. 배경 및 동기 (Section I)

- π0가 "in-distribution"에서 dexterous task를 잘 수행했다면, π0.5의 목표는 **open-world generalization** — 훈련 데이터에 없는 새 가정에서 10~15분짜리 multi-stage task를 수행 (Section I).
- 단일 source data scaling만으로는 한계: "97.6% of training examples in the first phase do not come from mobile manipulators performing household tasks" (Section I) — 압도적 다수가 다른 robot/web/HL data에서 옴.
- 핵심 가설: 인간이 직접 경험뿐 아니라 책·말·다른 task의 통찰을 종합하듯, robot도 heterogeneous source 통합이 필수.

---

## 2. 방법론 (Section IV, Figure 3)

### Two-Stage Training Recipe
1. **Pre-training (280k step)**: 모든 task를 discrete token으로 표현. Robot action도 FAST tokenizer로 discretize → standard auto-regressive transformer로 next-token prediction. α=0 in Eq.(1) — text/action 모두 cross-entropy.
2. **Post-training (80k step)**: Flow-matching action expert를 새로 attach (random init). α=10.0으로 cross-entropy + flow-matching MSE 동시 최적화. Inference 시 10-step denoising으로 continuous action 생성.

### Hybrid Action Representation (Eq. 1, Section IV-B)
$$\mathbb{E}_{D,\tau,\omega}\Big[H(x_{1:M}, f_\theta^\ell(o_t,\ell)) + \alpha \|\omega - a_{t:t+H} - f_\theta^a(a_{t:t+H}^{\tau,\omega}, o_t, \ell)\|^2\Big]$$
- 동일 모델이 텍스트(M token)와 action(H token)을 각각 출력. Attention matrix로 두 representation이 서로 attend하지 않도록 차단.

### Hierarchical Inference (Section IV-B, Figure 1)
- 매 step:
  1. **High-level**: "clean the bedroom" + observation → "pick up the pillow" subtask 생성 (autoregressive, low frequency).
  2. **Low-level**: subtask + observation → action chunk (flow matching, high frequency).

### 데이터 Source (Section IV-C, Figure 4)
- **MM** (Mobile Manipulator): ~400h, 100 가정.
- **ME** (Multi-Environment non-mobile): 다양한 가정의 정적 robot.
- **CE** (Cross-Embodiment laboratory): 단일/양팔 다양한 robot, lab 환경.
- **HL** (High-Level subtask): MM/ME/CE 데이터에 subtask annotation.
- **WD** (Web Data): CapsFusion, COCO, Cambrian-7M, PixMo, VQAv2 + indoor scene 추가.
- **VI** (Verbal Instruction): 인간이 실시간으로 robot에게 subtask command 제공하며 teleoperate (post-training only).

### 시스템 (Section IV-E, Figure 5)
- 두 개의 mobile manipulator: 4 cameras (front/back/2 wrist), 2× 6-DoF arm + parallel jaw, 1-2 DoF lift, 3-DoF holonomic base. State/action 18-19D.
- Control: 50Hz action chunking, target pose를 PD controller가 추적. **collision detection·motion planner 없음 — 완전 end-to-end**.

---

## 3. 실험 결과

### Real Home Generalization (Section V-A, Figure 7)
- 3개 실제 가정 (훈련에 없음): bedroom·kitchen cleaning task.
- "items in drawer", "laundry basket", "dishes in sink" task에서 일관된 성공률.
- Mock home 평가가 real home 평가와 거의 일치 → controlled benchmark의 타당성 입증.

### Scaling with Locations (Section V-B, Figure 8)
- 훈련 environment 수 변화: 3, 12, 22, 53, 82, 104 locations.
- 104-location 모델은 **test home data가 직접 포함된 baseline (green)과 동등** 성능 — pre-training recipe만으로 broad generalization 달성.
- Light yellow baseline (test home + 104 locations이지만 다른 co-training source 없음)이 훨씬 약함 → ME/CE/HL/WD가 결정적.

### Co-training Ablation (Figure 10, 11)
- **no ME** / **no CE** / **no ME+CE**: 모두 큰 폭 성능 하락 — cross-embodiment data가 필수.
- **no WD**: mock home 평균은 큰 차이 없으나 **OOD object language following에서 큰 하락** (Figure 11) — web data가 unseen object semantic generalization에 결정적.

### vs π0 (Section V-D, Figure 12)
- π0 / π0-FAST+Flow / π0.5 비교. π0.5가 동일 robot data·동일 train step에서 **압도적으로 우월**.
- 300k step까지 π0를 훈련해도 격차 유지 → π0.5의 우위는 단순한 train step 차이가 아니라 co-training recipe 덕분.
- "training with FAST tokens is more effective in terms of compute than pure diffusion based training" (Section V-D, Pertsch et al. 2025 인용 = pi0-FAST 논문).

### High-Level Inference (Section V-E, Figure 13)
- **Full π0.5 > human HL oracle** — 놀라운 결과로 unified model의 high-level inference가 인간 oracle보다 미묘하게 우위.
- **implicit HL** (HL data로 훈련했지만 inference 시 high-level 호출 안 함) > **no HL** — 단지 training data에 subtask가 있는 것만으로도 행동에 implicit benefit.
- **no VI** 변형: 큰 폭 하락 — VI는 전체 HL의 11%지만 효과 압도적.
- **GPT-4 zero-shot HL**: 가장 약함 — robot data 없이 zero-shot prompting은 부족.

---

## 4. 한계 (Section VI)

- **Persistent failure modes**: 익숙하지 않은 drawer handle, 물리적으로 열기 힘든 cabinet, 팔이 spill을 가리는 partial observability, drawer를 반복 여닫는 high-level distraction.
- **Prompt 복잡도 한계**: 상대적으로 단순한 prompt만 처리. 복잡한 선호/지시는 더 많은 annotation 필요.
- **Context window 제한**: 방 사이 navigation·물건 위치 기억 등 long-term memory 부재.
- **400h MM data로 부족할 수 있는 task**: 더 long-horizon은 추가 data 필요.

---

## 5. 총평

π0.5의 진정한 기여는 architecture 자체보다 **"어떤 data mixture가 진짜 generalization을 만드는가"**에 대한 체계적 ablation이다. Figure 8의 "104-location model ≈ test-home-trained model" 결과는 cross-embodiment + multi-environment data scale이 in-domain data scale을 대체할 수 있음을 시사한다. 또한 **discrete FAST pre-train → flow-matching post-train**의 hybrid recipe (Section IV-B의 Eq.1)는 "discrete token이 학습 효율적이지만 inference에 비효율"이라는 trade-off를 우아하게 해결한다. 다만 모델 weight·data 비공개로 재현이 불가능하다는 점은 학술적 가치를 약화시킨다.

---

## 6. 예상 질문

- **Q1**: discrete pre-train vs flow post-train, 왜 두 단계?
  - A: Section IV-B — discrete token은 학습 빠르고 효율적, flow는 real-time inference에 적합. 두 representation을 attention matrix로 분리하여 한 모델에 통합.
- **Q2**: 104 location 모델이 test-home 데이터로 직접 학습한 모델과 동등한 이유?
  - A: Figure 8 — cross-embodiment + multi-environment scale이 충분히 커지면 specific home 데이터의 marginal benefit이 사라짐.
- **Q3**: π0.5가 "human HL oracle"보다 좋다는 결과는 어떻게 해석?
  - A: Section V-E — unified model이 "어떤 subtask가 자기에게 실행 가능한지"를 더 잘 안다는 해석. 인간은 robot의 실패 mode를 모르고 너무 어려운 subtask를 줄 수 있음.
- **Q4**: VI (verbal instruction)이 11% data인데 큰 효과인 이유?
  - A: VI는 expert가 robot의 능력에 맞춰 task를 분해한 "고품질 supervision". HL annotation의 quality > quantity.
- **Q5**: pi0-FAST+Flow와의 차이는?
  - A: pi0-FAST+Flow도 hybrid token+flow지만 robot action data만 사용. π0.5는 추가로 HL/WD/VI co-training 포함.

<!-- VERIFIED: pdf -->
