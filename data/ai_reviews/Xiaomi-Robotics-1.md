## Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories

### 1. 한 줄 요약

Xiaomi Robotics가 발표한 Xiaomi-Robotics-1은 **100k 시간 이상의 실세계 UMI 궤적**으로 pre-training하고 약 10k 시간의 cross-embodiment 데이터로 post-training하는 2단계 recipe를 통해, **Mixture-of-Transformers(Qwen3-VL VLM + flow-matching DiT)** 구조로 데이터·모델 스케일링 법칙을 로봇 정책에 실증한 foundational VLA로, RoboCasa365 57.4%(이전 최고 46.6%)와 RoboDojo 평균 20.07(이전 최고 13.07)로 4개 시뮬레이션 벤치마크에서 SOTA를 달성했다(arXiv 2607.15330v2, 2026-07-22).

---

### 2. 배경 및 동기

- 현대 대규모 모델의 성능은 근본적으로 **스케일**(데이터·컴퓨트·모델 용량)에 의해 견인되지만, 로봇 학습은 **데이터 병목**이라는 고유한 제약에 걸림.
- 지배적 데이터 수집 방식인 real-robot teleoperation은 느리고 비싸며 hardware-bound → 확장 어렵고, 수집된 데이터는 좁은 task/환경에 집중되어 diversity가 제한됨.
- 저자들의 착안: LLM/VLM의 스케일링 궤적을 로봇에도 적용하되, **UMI(Universal Manipulation Interface) 휴대형 그리퍼**로 물리 로봇 없이도 in-the-wild 다양성 궤적을 대량 수집.
- 핵심 질문: (i) pre-training에서 데이터/모델 스케일이 효과적인가, (ii) 강한 pre-trained 모델이 post-training out-of-the-box 성능으로 전이되는가, (iii) 최소 데이터로 새 task에 적응 가능한가, (iv) 기존 foundation model 대비 우수한가.

---

### 3. 방법론

#### 3.1 문제 정식화

관측 $o_t$와 언어 지시 $l$이 주어지면 정책 $\pi_\theta$는 action chunk $a_{t:t+H}$의 log-likelihood를 최대화하도록 학습된다: $\max_\theta \mathbb{E}_{(o_t,l,a_{t:t+H})\sim D}\log\pi_\theta(a_{t:t+H}\mid o_t,l)$.

#### 3.2 Architecture (Fig. 2, Table 1)

- **Mixture-of-Transformers(MoT)**: 사전학습 VLM(**Qwen3-VL**)과 **Diffusion Transformer(DiT)**를 결합. DiT는 VLM과 동일한 layer 수를 갖지만 hidden size는 더 작아 추론 속도 향상.
- 스케일 변이(Table 1): **2B(28층, VLM 2.1B + DiT 470M = 총 2.6B)**, **5B(36층, 4.4B + 604M = 5.1B)**, **10B(36층, 8.8B + 1.5B = 10.5B)**.
- **DiT flow-matching**: $\tilde a^\tau = \tau a + (1-\tau)\epsilon$, $\epsilon\sim\mathcal N(0,I)$, timestep $\tau$는 $u\sim\text{Beta}(1.5,1)$에서 $\tau=(1-u)\cdot0.999$로 샘플(노이즈 큰 구간에 가중). adaLN으로 timestep 주입. 추론시 5-step Euler 적분($\Delta\tau=0.2$).
- **VLM Choice-Policy 보조 supervision**: VLM 프레임 내에서 로봇 state를 MLP로 토큰화 후 action/score query 토큰을 붙여 **K개 후보 action chunk와 score**를 예측. Winner-takes-all(L1 최소 후보만 action loss)로 $L_{Regression}=\lVert\hat a^*-a\rVert_1+\sum_k\lVert\hat s_k-s_k\rVert_2^2$, 여기서 $s_k=\lVert\hat a_k-a\rVert_1$.
- **핵심 트릭**: DiT가 VLM의 action-related 토큰 KV cache에 attend하면 성능 저하(단순 copy shortcut) → 이 토큰들을 **DiT attention에서 제외**하고 언어/시각 표현에만 attend하도록 제약.

#### 3.3 Pre-training (100k+ 시간)

- UMI 그리퍼 + egocentric 카메라로 가정/상업/산업/사무실/야외까지 대규모 환경에서 궤적 수집.
- **Auto-labeling 파이프라인**: 궤적을 등길이 세그먼트로 분할 후 **Qwen3.5-27B**로 그리퍼·상호작용 객체의 state transition을 캡션. Producer-consumer 파이프라인(CPU 워커가 클립 절단, 클라이언트가 수백 개 captioning 요청 동시 유지)으로 100k 시간을 **약 2주 만에** 라벨링.
- 목적함수: $L = L_{Flow} + L_{Regression} + \lambda L_{NTP}$ ($\lambda=0.1$), VL 데이터로 co-training해 VLM 능력 보존. VL:UMI 샘플 비율 1:9. 샘플당 4개 flow-matching timestep을 샘플·packing해 DiT 비용 amortize.

#### 3.4 Post-training (~10k 시간)

- 목적 이중화: (1) UMI 그리퍼 action → 로봇 embodiment action 전이, (2) state-transition 설명 → 인간이 쓰는 **imperative 지시**로 언어 조건 전환.
- 데이터: 7,200시간+ 인하우스 mobile manipulator/dual-arm 로봇 데이터, 1,000시간+ human-annotated UMI 데이터, open-source(Bridge V2, RT-1, DROID). Qwen3.5로 human-segmented 클립에 지시 라벨.
- Arm action은 현재 상태 기준 **relative delta EE-pose**, EE frame orientation을 UMI/로봇 간 통일해 하드웨어 무관하게 일관 값. Mobile base는 base velocity + waist delta. 이질 embodiment는 **통일 action vector + 결손 차원 masking**으로 처리.
- 동일 목적함수(Eq. 1), 샘플 비율 VL:open-source:UMI-instr:in-house = 0.5:0.5:0.5:8.5.

---

### 4. 실험 결과

#### 4.1 Pre-training 스케일링 (Fig. 5)

- **데이터 스케일링**(5B, 약 20k 시간의 12.5/25/50/100%): 데이터 증가 시 validation action MSE 단조 감소. 12.5%/25%는 overfitting(감소 후 재상승), 50%/100%는 단조 감소하며 20k에서 가장 가파른 하강.
- **모델 스케일링**(2B/5B/10B, 동일 20k): 모델 커질수록 정밀도 향상, 단 데이터 스케일 대비 격차는 덜 뚜렷 → billions 규모에서 **데이터 양이 1차 병목**임을 시사.

#### 4.2 Post-training Out-of-the-Box (Fig. 7, 8)

미학습 환경/객체에서 shoe storage, bag packing, table organization, sofa tidying 4개 task 평가.

- **데이터 스케일 전이**(5B): action pre-training 없음 26% → 12.5% 데이터 53% → 100% 데이터 **75%**로 단조 상승. Contact-rich task(shoe tidying)에서 특히 큰 이득(0% → 75%). 50%→100%에서도 +6%p로 saturation 징후 없음.
- **모델 스케일 전이**: 2B 61% → 5B 75% → **10B 79%**. Shoe tidying 58%→75%→**92%**. sofa tidying만 5B/10B가 80%/77%로 비슷.

#### 4.3 Downstream Fine-tuning (Fig. 10)

held-out 4개 신규 task(phone packing 양손 협응, laundry loading 장기 mobile, printer refilling 변형체, box packing 언어 grounding).

- **Low-data(<10h/task, 총 36h)**: 평균 SR **75%**, 평균 progress **90%**. π0.5(40% SR / 66% progress), Xiaomi-Robotics-0 대비 크게 우세.
- printer refilling: best baseline 20% → **70%**(변형체 조작). laundry loading: **80% SR / 96% progress**(장기 robustness). box packing은 모든 방법이 데이터 충분 시 100% 도달.
- High-data(144h)에서도 우위 유지 → 대규모 pre-training + 정교한 post-training 정렬 덕에 **데이터 효율** 우수.

#### 4.4 Simulation Benchmarks (Tab. 2-5)

| 벤치마크 | Xiaomi-Robotics-1 | 이전 최고 |
|---|---|---|
| **RoboCasa** (24-task avg SR) | **74.5%** | World2Act 72.6% |
| **RoboCasa365** (avg) | **57.4%** | ABot-M0.6 46.6% |
| RoboCasa365 atomic / comp-seen / comp-unseen | 80.2 / 57.1 / **32.1** | 79.4 / 48.3 / 14.9 |
| **VLABench** (avg SR / PS / IS) | **59.1** / **70.3** / 69.9 | ERVLA 53.2 / 65.9 / **70.4** |
| **RoboDojo** (avg score / SR) | **20.07** / **13.93** | Hy-Embodied-0.5 13.07 / 8.80 |

- RoboCasa365 Composite-Unseen에서 14.9→**32.1**로 최대 향상 → task 조합 일반화 강력.
- VLABench cross-category/texture에서 각각 +6.0/+15.2%p로 unseen 객체·시각 교란에 강건. 단 IS는 ERVLA에 근소 열세.
- RoboDojo 5개 차원 중 4개 1위. history 관측을 쓰지 않아 memory 차원만 열세(7.81 vs Hy-Embodied 13.37).

---

### 5. 다른 모델과의 비교

- **π0.5(Physical Intelligence)**: 동일 flow-matching 계열이나 Xiaomi-Robotics-1은 UMI 대규모 pre-training + MoT 구조로 downstream low-data 75% vs π0.5 40%, RoboCasa365 57.4 vs 16.9로 큰 격차.
- **Xiaomi-Robotics-0(자사 전작, arXiv 2602.12684)**: RoboDojo 6.93 → 20.07로 대폭 향상, async fine-tuning recipe 재사용.
- **Qwen-RobotManip**: 같은 Qwen 계열 backbone + flow-matching DiT를 쓰지만, Qwen-RobotManip은 alignment/cross-embodiment 표현에, Xiaomi-Robotics-1은 **100k+ 시간 스케일링과 auto-labeling**에 방점.
- **ABot-M0.6 / GR00T-N1.6 / RLDX-1**: RoboCasa365에서 모두 하회, 특히 unseen 조합에서 우위.

---

### 6. 강점

1. **스케일 실증**: pre-training 데이터/모델 스케일이 validation error뿐 아니라 **post-training out-of-the-box 실로봇 성능으로 직접 전이**됨을 정량 확인 → 로봇 스케일링 법칙의 강한 증거.
2. **Auto-labeling 파이프라인**: 100k 시간을 수동 세그먼트 없이 2주 만에 state-transition 캡션으로 라벨 → 데이터 병목의 실질적 해법.
3. **UMI 기반 데이터 수집**: 물리 로봇 없이 in-the-wild 다양성 확보, teleoperation 한계 우회.
4. **아키텍처 디테일**: DiT의 action-token attention 배제로 copy shortcut 방지, VLM Choice-Policy로 수렴 가속 — 실험적으로 근거 있는 선택.
5. **폭넓은 검증**: 4개 시뮬 벤치 SOTA + 4개 OOTB task + 4개 downstream task + 10분 장기 suitcase packing까지 커버.

---

### 7. 약점 및 의문점

1. **데이터/컴퓨트 비공개성**: 100k 시간 UMI 데이터·인하우스 로봇 데이터는 비공개 → 재현 불가능, 학계 대비 불공정 비교 소지.
2. **모델 스케일 이득 제한적**: 2B→10B 격차가 데이터 스케일 대비 작음 → 현재 데이터 분포가 이미 billions 용량으로 포화, 모델 스케일링의 marginal value 불투명.
3. **Memory 차원 열세**: history 관측 미사용으로 RoboDojo memory에서 열세 → 아키텍처가 명시적 memory를 다루지 못함.
4. **VLABench IS 열세**: intention score는 ERVLA에 뒤짐 → 암묵적 의도 이해에서 개선 여지.
5. **Open-source 미완**: "code and checkpoints will be released"만 명시, 현재 미공개 → 검증 어려움.
6. **UMI→로봇 gap의 정량 분석 부족**: gripper orientation 통일·masking으로 정렬했으나 embodiment gap이 남긴 오차의 세부 ablation 미흡.

---

### 8. 핵심 기여 (저자 contribution)

1. 100k+ 시간 실세계 UMI 궤적으로 학습한 foundational VLA로, **pre-training 스케일이 post-training 실로봇 성능으로 전이**됨을 실증.
2. VLM 기반 **state-transition auto-labeling 파이프라인**과 producer-consumer 인프라로 대규모 궤적을 저비용 라벨링.
3. Qwen3-VL VLM + flow-matching DiT의 **MoT 구조**(Choice-Policy 보조 supervision, action-token attention 배제)로 4개 시뮬 벤치 SOTA 및 데이터 효율적 downstream 적응 달성.

---

### 9. 후속 연구 방향 (저자 미래 작업 포함)

- **데이터 스케일 지속 확대**: 데이터가 1차 병목이므로 더 크고 다양한 궤적 수집이 성능의 열쇠.
- 명시적 **memory/history 모델링** 통합으로 state-dependent task 개선.
- Room-level 장기 mobile manipulation(10분 suitcase packing) 확장.
- RL/online improvement와의 결합, 더 강력한 차세대 VLM backbone 채택.
- Auto-labeling 캡션 품질과 action 학습 간 상관 정밀 분석.

---

### 10. 실험 재현·검증 체크리스트

- [ ] 100k 시간 UMI 데이터 및 인하우스 로봇 데이터 공개 여부(현재 비공개).
- [ ] Code / checkpoint 공개 시점 및 라이선스.
- [ ] RoboCasa365 50개 평가 task 구성(18 atomic + 16 seen + 16 unseen composite) 재현.
- [ ] VLABench CoT 라벨링 50% 확률 학습 설정의 영향 ablation.
- [ ] RoboDojo memory 차원에서 history 관측 추가 시 성능 변화.
- [ ] 2B/5B/10B 각 변이의 학습 step·컴퓨트 예산 공개.

---

### 11. VLA-Tracker 관점에서의 의의

- **스케일링 축의 대표 레퍼런스**: RhinoVLA(edge deployment 축), Qwen-RobotManip(alignment 축)과 함께, Xiaomi-Robotics-1은 **데이터 스케일(100k+ 시간)** 축의 대표 대형 테크리포트.
- Qwen3-VL backbone + flow-matching DiT 조합은 PaliGemma 중심에서 **Qwen 계열로의 이동**을 가속(Qwen-RobotManip, RhinoVLA와 합류).
- **UMI 기반 대규모 데이터**는 teleoperation 의존 흐름에 대안을 제시 → Tracker의 데이터 수집 방법론 분류에 새 카테고리.
- RoboCasa365 / RoboDojo / VLABench 등 **최신 대규모 벤치마크**의 SOTA 앵커로서, 후속 모델 비교 기준점 제공.

---

### 12. 결론

Xiaomi-Robotics-1은 **"로봇도 스케일링 법칙을 따른다"**는 명제를 100k+ 시간 실세계 UMI 궤적과 자동 라벨링 인프라로 실증한 대형 테크리포트다. Qwen3-VL VLM과 flow-matching DiT를 결합한 MoT 구조에 VLM Choice-Policy 보조 supervision과 DiT action-token attention 배제 같은 실험적 디테일을 더해, pre-training 데이터·모델 스케일이 post-training out-of-the-box 실로봇 성능(2B 61%→10B 79%)으로 직접 전이됨을 보였다. 결과적으로 RoboCasa365 57.4%(이전 46.6%), RoboDojo 20.07(이전 13.07)을 포함해 4개 시뮬 벤치에서 SOTA를 달성하고, <10h/task low-data downstream에서 75% SR로 π0.5(40%)를 크게 앞선다. 데이터·코드 비공개와 모델 스케일 이득의 제한이라는 한계가 있으나, **데이터 스케일링을 로봇 정책 성능으로 번역한 설득력 있는 증거**로서 VLA 스케일링 연구의 중요한 이정표로 평가된다.

<!-- VERIFIED: pdf -->
