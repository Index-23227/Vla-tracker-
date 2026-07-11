# WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving

> **한 줄 요약**: 기존 driving VLA가 world cognition의 부재 또는 파편화(semantic 예측만 하고 generative 진화는 못 함)로 인해 **reactive driving**에 머문다는 문제를 지적하고, (1) semantic level에서 3D perception 기반 agent token + world head로 세계 상태·미래 동역학을 명시적으로 인지하며 game-theoretic Game-CoT 추론을 수행하고, (2) generative level에서 **ADDT(Aligned Decoupled Diffusion Transformer)**가 joint multi-agent trajectory를 합성하는 dual-level world cognition 프레임워크를 제안. NAVSIM v1에서 SOTA **PDMS 92.9**, NAVSIM v2에서 **EPDMS 85.9** 달성.

---

## 1. 배경 및 동기

### 기존 driving VLA의 세 가지 한계
- **3D 공간 인지 부족**: 대부분 2D image feature에 의존해 주변 도로 참여자에 대한 구조화된 3D 표현이 없음 → 정밀한 spatial reasoning과 ego planning에 제약.
- **불충분한 world cognition**: 세계 상태 표현과 미래 동역학(주변 agent의 의도) 예측이 미흡. 일부 연구(UniDrive-WM, DriveVLA-W0 등)가 VLM hidden state나 future image generation으로 world cognition을 도입했으나, world modeling을 보조적 semantic task로만 취급하고 **generative-level world evolution**을 무시 → "fragmented world foresight". ego와 주변 agent 간 상호작용적 joint trajectory를 world-generative 관점에서 합성하지 못함.
- **전략적 social reasoning 부재**: 기존 reasoning은 정적 scene description 위주로, 능동적 사회적 상호작용에 필요한 game-theoretic 'if-what' imagination이 없음.

### 핵심 질문
> Semantic-level forecasting과 generative-level evolution을 아우르는 포괄적 world cognition을 VLA에 부여해 proactive driving을 가능하게 할 수 있는가?

Fig. 1은 VLM 활용의 네 가지 패러다임(autoregressive text action / cognitive encoder + action decoder / fragmented world foresight / 본 논문의 dual-level world cognition)을 대비시킨다.

📌 [Figure 1 삽입] — E2E 자율주행에서 VLM 활용 4가지 패러다임 비교

---

## 2. 방법론 심층 분석: VLM Backbone (Semantic Level)

### 2.1 입력 및 기반 모델
- 입력: 6개 surround-view 카메라 이미지, navigation instruction(예: 'turn right'), ego state S = {속도 v, 가속도 a, 2초 과거 궤적 T_hist (2 Hz 샘플링)}.
- 백본: **InternVL3-2B** — 300M InternViT vision encoder + Qwen2.5 LLM.

### 2.2 3D Spatial Perception
- Multi-view camera feature를 **BEVFormer**의 off-the-shelf BEV encoder로 lift → F_BEV.
- **TrackFormer**(UniAD 계열)가 learnable agent query Q_agent와 F_BEV 간 cross-attention으로 dense BEV feature를 **sparse agent-centric token** T_agent (N_a개, 검출된 agent 수)로 변환. 각 agent token은 공간 위치와 기하 특징을 인코딩.

### 2.3 Unified World Cognition and Reasoning
- Vision(T_vision), text(T_text), agent(T_agent) token을 sequence 차원으로 concat해 LLM에 입력 (Eq. 1).
- 출력 hidden state를 **두 가지 기능적 역할로 decouple**:
  - **Cognition role (O_agent)**: 전용 **world head**로 라우팅되어 주변 agent의 현재 3D perception + 미래 trajectory prediction을 디코딩 → 명시적 semantic-level world cognition.
  - **Reasoning role (O_vision, O_text)**: language modeling head로 textual response 생성. Game-CoT paradigm으로 학습되어 명시적 game-theoretic reasoning 과정을 출력.

---

## 3. 방법론 심층 분석: ADDT (Generative Level)

### 3.1 동기: 단일 네트워크 diffusion의 최적화 딜레마
- 단일 DiT에서 low-frequency 추상 의미의 인코딩과 high-frequency 연속 디테일의 디코딩이 충돌 → 자율주행에서는 복잡한 multi-agent 상호작용 모델링과 정밀 trajectory 생성 간 tension으로 발현.
- 해법: **condition encoder + generation decoder로 분리된 decoupled 구조** (총 16 DiT block, 각 8 block).

### 3.2 Condition Encoder
- 입력: joint multi-agent action noise x_t ∈ R^{N_m×H×3} (N_m: 최대 agent 수, H: planning horizon)에 embedded noise action, 과거 ego action τ_his, average-pooled VLM token F̄_VLM을 concat한 fused representation F_at (Eq. 2).
- Timestep t와 ego state S는 **AdaLN modulation**으로 주입(물리적 kinematics guidance), full-sequence VLM token F_VLM = [O_vision, O_text, O_agent]은 **cross-attention**으로 주입(고수준 semantic prior).
- 출력: semantic self-condition feature z_t.

### 3.3 Representation Alignment
- Condition encoder의 i번째 DiT block 중간 feature h_i를, GenAD 방식으로 pre-train된 **trajectory VAE encoder의 latent scene representation r\***와 정렬: L_align = 1 − cos(r\*, h_φ(h_i)) (Eq. 3, learnable projection MLP h_φ).
- 효과: z_t가 실제 scene dynamics에 grounding되고, 인접 denoising timestep 간 **local consistency**가 유지되는 정규화 역할 → 적은 denoising step으로도 안정적 생성. 정렬은 6번째 encoder block에서 추출.

### 3.4 Generation Decoder
- Encoder와 동일 구조의 8 DiT block이지만 high-frequency 기하 디테일 복원에 전념. Timestep t와 self-condition feature z_t를 AdaLN으로 주입해 semantically aligned denoising 수행: x_{t−1} = Decoder(F_at, t, z_t, F_VLM) (Eq. 4).

📌 [Figure 3 삽입] — ADDT의 decoupled encoder-decoder 구조와 representation alignment

---

## 4. Game-CoT Reasoning Annotation

- **Qwen3-VL-Plus** 기반 자동 annotation pipeline으로 4단계 구조화 reasoning 생성: (1) scene description, (2) critical object analysis, (3) game-theoretic reasoning, (4) payoff evaluation.
- Game-theoretic 단계는 교통 상호작용을 **Stackelberg game**으로 정식화 — ego가 leader, 주변 agent가 follower. 'if-what' imagination으로 후보 ego action을 열거하고 follower들의 대응을 추론, payoff 단계에서 안전성·효율성을 평가해 최적 전략 결정.
- Hallucination 억제를 위해 **GT action을 guiding hint**로 주입 → 관측된 scene context에서 최종 GT action까지의 명시적 인과 사슬을 재구성하도록 강제.
- 결과물: NAVSIM 기반 **85k 고품질 Game-CoT annotation** 데이터셋.

---

## 5. 4단계 학습 파이프라인

| Stage | 내용 | 세부 |
|---|---|---|
| 1. 3D Perception Pre-Training | BEV encoder + TrackFormer 최적화 | UniAD 방식 detection head, focal loss(분류) + L1(3D box), NAVSIM 1 epoch |
| 2. VLM SFT | VQA + world cognition | 158k 공개 driving VQA(DriveLM, CODA-LM, LingoQA, nuScenes-QA, NuInstruct, DriveGPT4)로 1 epoch pre-train 후, 170k NAVSIM-tailored(85k trajectory VQA + 85k Game-CoT)로 world head와 3 epoch joint fine-tuning. L = L_LM + λ_world·L_world (Eq. 5-6) |
| 3. ADDT SFT | VLM freeze, DDPM 학습 | L2 denoising + λ_align·L_align (Eq. 7-8), agent별 weight mask W(α_ego > α_surr)로 ego 정확도 우선, 200 epochs |
| 4. RFT | **DiffGRPO** (diffusion 특화 GRPO) | RL policy loss + behavior cloning loss로 policy collapse 방지 (Eq. 9), 10 epochs, group size 6 |

- RFT 보상: r_i = r_PDMS − λ_surr·L_L1(τ_surr) — ego 주행 품질(PDMS)과 주변 agent 궤적 예측 정확도를 결합한 joint reward.
- 전체 학습: **4x NVIDIA A100 40GB**.

---

## 6. 실험 설정

- **벤치마크**: NAVSIM v1 / v2 (navtrain 1,192 scenes 학습, navtest 136 scenes 평가). 카메라 입력만 사용.
- **지표**: v1 — NC(무과실 충돌), DAC(주행가능영역 준수), TTC, Comfort, EP(ego progress), PDMS. v2 — DDC, TLC, LK, HC, EC 추가한 EPDMS.
- v1 결과는 4단계 전체 학습 후, v2 결과는 3단계 SFT까지만 수행 후 평가.

---

## 7. 주요 결과

### NAVSIM v1 (Table 1)
| Method | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---|---|---|---|---|---|
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE (camera+lidar) | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ReCogDrive-2B | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| AutoVLA-3B | 99.1 | 97.1 | 97.1 | 100 | 87.6 | 92.1 |
| LatentVLA-3B | 98.9 | 98.2 | 95.2 | 100 | 88.2 | 92.4 |
| **WCog-VLA-2B** | **99.4** | **98.8** | **98.5** | 100 | 87.1 | **92.9** |

- 카메라 입력만으로 camera+lidar 기반 WoTE 대비 **+4.6 PDMS**. Generalist QwenVL2.5/InternVL3-8B(fine-tuned) 대비 +9.6. RL-refined ReCogDrive/AutoVLA 대비 최소 +0.8, 3B LatentVLA 대비 +0.5 — 2B로 더 큰 모델을 능가.
- 안전 지표에서 특히 우수: NC 99.4, TTC 98.5 — 주변 agent의 미래 의도를 선제 예측해 능동적 회피가 가능하기 때문.

### NAVSIM v2 (Table 2)
- 3단계 SFT만으로 **EPDMS 85.9** SOTA (DiffusionDrive 84.3 대비 +1.6). NC 98.8, TTC 98.2로 안전 지표 최고.

---

## 8. Ablation 분석

- **4단계 학습 (Table 3)**: Stage 2만 = 84.4 → +Stage 1(3D perception) 85.5 → +Stage 3(ADDT, text→연속 궤적 전환) 89.3 (+3.8) → +Stage 4(RFT) **92.9** (+3.6). 모든 단계가 필수적.
- **Dual-level world cognition (Table 4, 3-stage SFT 기준)**: baseline 86.5 → semantic current 87.0 / future 87.2 / 둘 다 88.1 → generative만 87.4 → **semantic + generative 결합 시 89.3으로 시너지 도약**. Semantic forecasting과 generative evolution의 결합이 핵심.
- **ADDT (Table 5)**: VLM text 출력(reasoning 포함) 85.5 / 9.896 s 대비, 5-step ADDT는 **89.3 / 0.106 s (10.7배 가속)**. 20-step 표준 DiT(SDT, 88.5) 대비 5-step ADDT가 +0.8 PDMS에 3.7배 빠름. 5→20 step 증가 시 +0.3에 불과 — alignment 덕분에 denoising step 수에 둔감.
- **VQA 데이터 (Table 6)**: trajectory VQA만 86.7 → +공개 driving VQA 88.2 / +Game-CoT 87.5 → 셋 결합 89.3.
- **3D perception (Table 7)**: 없으면 86.0, 있으면 89.3 (+3.3) — 명시적 3D 인지의 기여가 큼.

---

## 9. 정성적 분석

- **vs ReCogDrive (Fig. 5)**: 복잡한 도심에서 ReCogDrive는 과도하게 보수적으로 느린 차선에 갇히는 반면, WCog-VLA는 전방 저속 버스를 인지하고 차선을 변경해 human GT와 근접한 효율적 주행.
- **Generative world cognition (Fig. 6)**: 교차로에서 baseline은 마주 오는 차량에 대한 interactive foresight 없이 ego-only 궤적을 생성해 수동적 감속. WCog-VLA는 joint multi-agent 궤적으로 상대 차량의 직진을 명시적으로 예측하고 자신 있게 좌회전 — proactive maneuver의 실증.
- **Semantic world cognition (Fig. 7)**: world head가 디코딩한 3D perception과 미래 궤적 예측이 GT와 밀접히 일치.

---

## 10. 강점

1. **개념적 완결성**: "semantic forecasting + generative evolution = dual-level world cognition"이라는 프레임이 명확하고, ablation(Table 4)이 두 레벨의 시너지(88.1/87.4 → 89.3)를 직접 입증.
2. **효율적 diffusion 설계**: representation alignment로 5 denoising step만으로 20-step 표준 DiT를 능가 — 실시간성(0.106 s)과 성능을 동시 확보한 실용적 기여.
3. **Game-CoT 데이터셋**: Stackelberg game 기반 'if-what' reasoning supervision 85k는 social driving reasoning의 공백을 메우는 자원. GT-hint 기반 인과 사슬 재구성으로 hallucination 억제 설계도 합리적.
4. **작은 모델로 SOTA**: 2B 모델이 카메라만으로 3B/8B 및 camera+lidar 베이스라인을 능가. 4x A100 40GB라는 비교적 소박한 compute도 재현성 측면에서 긍정적.
5. **RFT까지 포함한 완전한 파이프라인**: DiffGRPO + joint reward(ego PDMS + 주변 agent 예측)로 imitation 한계를 넘는 탐색을 수행, +3.6 PDMS의 큰 기여.

---

## 11. 한계 및 논의

1. **정적 세계 요소 미반영**: 저자들도 인정하듯 semantic cognition이 agent에 집중되어 도로 기하·맵 토폴로지의 미래 진화는 다루지 않음.
2. **NAVSIM 중심 평가**: open-loop 계열 벤치마크(NAVSIM v1/v2)만 평가. Bench2Drive 같은 closed-loop CARLA 평가나 실차 검증이 없어 누적 오차·분포 이동에 대한 강건성은 미확인.
3. **Game-CoT의 검증 한계**: Qwen3-VL-Plus 자동 annotation 품질에 대한 인간 평가가 본문에 없고, reasoning 텍스트가 실제 planning에 인과적으로 기여하는지(Table 6에서 CoT 단독 기여 +0.8)는 상대적으로 작음.
4. **코드 미공개**: 논문에 코드/체크포인트 공개 언급이 없어 재현은 서술에 의존.
5. **v2에서 RFT 미적용**: NAVSIM v2 결과가 3-stage SFT 기준이라 RFT의 일반화 효과는 v1에서만 확인됨.

---

## 12. 종합 평가

WCog-VLA는 driving VLA의 "reactive → proactive" 전환을 **dual-level world cognition**이라는 일관된 설계 철학으로 달성한 연구다. 3D agent token 기반 semantic world head, Stackelberg game 기반 Game-CoT, 그리고 VAE latent 정렬로 denoising step을 극적으로 줄인 ADDT가 유기적으로 결합되며, 각 구성요소의 기여가 ablation으로 깔끔하게 분리 입증된다. 2B 카메라 전용 모델로 NAVSIM v1 PDMS 92.9 / v2 EPDMS 85.9 SOTA를 달성한 점, 특히 안전 지표(NC 99.4, TTC 98.5)에서의 우위는 world cognition 접근의 실질적 가치를 보여준다. 다만 closed-loop/실차 검증 부재와 코드 미공개는 아쉬운 지점. Action head는 VLM hidden state를 조건으로 joint multi-agent 궤적을 denoising하는 DiT 구조로 **diffusion**으로 분류된다.

**Score: 8.0 / 10**

<!-- VERIFIED: pdf -->
