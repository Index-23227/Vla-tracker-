# DriveTeach-VLA: Teaching Vision-Language-Action Models What to See and Where to Look

> **한 줄 요약**: 기존 자율주행 VLA의 VQA/CoT 기반 학습이 **text-centric**이어서 attention이 공간적으로 grounding되지 않는다는 문제를 지적하고, (1) Grounding DINO bbox-augmented image self-distillation로 "무엇을 볼지"를 가르치는 **DVD pretraining**, (2) expert BEV 궤적을 pinhole camera model로 이미지 평면에 투영한 **2D-TGP**로 "어디를 볼지"를 조건화하는 dual-model(TGP-Prompter + TGP-Planner, 둘 다 Qwen2.5-VL-3B) 프레임워크. NAVSIM navtest **90.4 PDMS**, nuScenes L2 0.30m/collision 0.12%로 SOTA 달성. ECCV 2026.

---

## 1. 배경 및 동기

### 기존 AD-VLA의 한계
- 최신 AD-VLA들은 3단 파이프라인(VQA pretraining → CoT-SFT IL → GRPO RL)을 따르지만, 앞 두 단계의 supervision이 본질적으로 **text-centric** — semantic QA에 집중할 뿐 planning에 필요한 spatial/behavioral 지식을 주입하지 못함.
- 결과적으로 모델은 "보이는 것을 명명(name what is seen)"할 뿐, **무엇을 봐야 하고(what to see) 어디를 봐야 하는지(where to look)**를 배우지 못함.
- Figure 1의 attention map 비교: 순수 텍스트 traffic-knowledge SFT를 거친 Qwen2.5-VL baseline은 attention이 장면 전체에 흩어져 spatial grounding이 없음. 저자들은 이를 Attention Mass(AM) 지표로 정량화 (Section 4.3, Analysis 1).

### 핵심 질문
- VQA pair 없이 vision encoder에 driving-specific perceptual prior를 주입할 수 있는가?
- MLLM의 intrinsic 2D grounding 능력을 활용해 feasible driving region을 명시적 spatial guidance로 줄 수 있는가?

📌 [Figure 1 삽입] — Qwen2.5-VL SFT baseline vs DriveTeach-VLA의 autoregressive decoding attention map 비교

---

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Dual-Model)
- **TGP-Prompter**: 카메라 이미지 C, ego-state S, instruction L_TGP를 입력받아 2D-TGP(이미지 평면 궤적 좌표 텍스트) 추정 — Eq. 4.
- **TGP-Planner**: TGP-Prompter의 weight로 초기화되어 traffic prior를 상속. C, S, [L; P̂_I]를 조건으로 step-by-step CoT reasoning 후 BEV 궤적을 텍스트 waypoint로 autoregressive 생성.
- 두 모델을 분리한 이유: 2D-TGP 생성과 BEV 궤적 예측은 입력이 유사해도 최적화 목표가 달라 단일 모델에서는 **instruction confusion** 발생 (Table 8에서 검증).

### 2.2 2D-TGP (Preliminary)
- BEV 궤적 T_w = {(x_t, y_t, ψ_t)}를 카메라 intrinsic K, extrinsic [R|t]의 pinhole projection π로 이미지 평면 (x^I_t, y^I_t)에 투영 (Eq. 1). 좌표 시퀀스를 텍스트화한 것이 2D-TGP P_I.
- AD에서는 궤적점이 지면(z=0)에 있다고 가정하므로 π가 가역 — Eq. 2의 π⁻¹로 2D-TGP에서 유일한 BEV 위치를 복원 가능(ψ_t는 별도 예측/보존).
- 2D 좌표는 MLLM이 natively 해석 가능하므로 MLLM의 intrinsic grounding 능력과 driving task를 정렬.

### 2.3 DVD Pretraining (TGP-Prompter, "what to see")
- **Bbox-augmented self-distillation**: Grounding DINO로 car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, barrier, traffic element/light 등 traffic-critical 객체 검출 → bbox를 raw image에 overlay한 C_bbox 생성. Teacher ViT는 C_bbox, student ViT는 raw C를 입력받고(둘 다 동일 MLLM ViT로 초기화), feature map을 K개 블록(2×4=8)으로 분할해 블록 평균 간 **Smooth-L1 block-wise alignment loss** L_distill (Eq. 6). Teacher는 DINO식 EMA(0.996) 업데이트.
- **2D-TGP SFT**: MLLM decoder는 ground-truth 2D-TGP 궤적을 예측하도록 cross-entropy L_TGP로 학습 (Eq. 7).
- 총 손실 L_DVD = L_distill + λ_TGP·L_TGP, λ_TGP=0.1 (Eq. 8). VQA pair에 의존하지 않고 텍스트 supervision 없이 traffic-relevant visual cue를 내재화.

### 2.4 TGP-guided Learning (TGP-Planner, "where to look" → "how to act")
- Poutine을 따라 Qwen2.5-VL-72B로 pseudo-label한 4-task CoT: (1) Critical Object Detection, (2) Natural Language Explanation, (3) Meta-Behavior Selection, (4) **2D-TGP keypoint 8개를 조건으로 한** 4초 미래 궤적 예측.
- CoT-SFT 후 AutoVLA를 따라 GRPO RL: rollout 궤적을 PDMS(Eq. 9: NC·DAC 곱 × EP/TTC/C 가중평균, w={5,5,2})로 보상. CuriousVLA의 data filtering/rollout 방식 채택.
- **Teacher forcing**: 학습(SFT/GRPO) 중에는 ground-truth 2D-TGP를 조건으로 공급, 추론 시에는 TGP-Prompter가 생성한 2D-TGP를 사용 (정보 누출 방지).

📌 [Figure 2, 3 삽입] — 전체 아키텍처 및 DVD/2D-TGP scheme

---

## 3. 학습 파이프라인 요약

| 단계 | 대상 | 데이터/설정 |
|------|------|------------|
| (i) DVD pretraining | TGP-Prompter | 1 epoch, block 2×4, λ_TGP=0.1, EMA 0.996 |
| (ii) CoT-SFT | TGP-Planner (Prompter로 초기화) | 6 epochs, Qwen2.5-VL-72B CoT 주석 |
| (iii) GRPO-RL | TGP-Planner | 180 update steps, group size 8, PDMS 보상 |

- 공통: AdamW (lr 4e-5, wd 0.05), cosine schedule, warm-up 0.10, batch 16, 8×H100.
- CuriousVLA를 따라 step-wise normalized text trajectory 사용. nuScenes에서는 공정 비교를 위해 CoT/GRPO 미사용.

---

## 4. 실험 설정

- **NAVSIM (navtest)**: 비반응형 closed-loop, 4초 미래 궤적, intention-changing 시나리오. PDMS(NC/DAC/TTC/C/EP) 보고 (RecogDrive 프로토콜).
- **nuScenes**: open-loop, 3초 horizon 평균 L2 error와 collision rate. ST-P3 및 UniAD 두 metric 프로토콜 모두 보고.
- 비교 대상: end-to-end(UniAD, TransFuser, Hydra-MDP, DiffusionDrive, WoTE, ASSCG)와 VLA 계열(ReCogDrive, ImagiDrive-S, AutoVLA, CuriousVLA 등).

---

## 5. 주요 결과

### NAVSIM navtest (Table 1)
| Method | Base VLM | NC | DAC | TTC | C | EP | PDMS |
|--------|----------|----|----|-----|---|----|------|
| ReCogDrive | InternVL3-8B | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| AutoVLA | Qwen2.5-VL-3B | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| CuriousVLA* | Qwen2.5-VL-3B | 97.7 | 95.9 | 97.2 | 98.2 | 89.2 | 88.9 |
| **DriveTeach-VLA** | Qwen2.5-VL-3B | 98.5 | 96.9 | 97.9 | 98.2 | 88.5 | **90.4** |

- **90.4 PDMS를 single inference pass로 달성** — VLA 계열 최고, 3B 백본으로 8B ReCogDrive(89.6) 상회. (Human 94.8, 최고 end-to-end ASSCG 91.4)

### nuScenes open-loop (Table 2)
- ST-P3 metric: **L2 0.30m / Collision 0.12%** (Impromptu VLA 0.33/0.13, EMMA 0.32 대비 최고).
- UniAD metric: **L2 0.60m / Collision 0.31%** — 양 프로토콜 모두 1위.

---

## 6. Ablation 분석

### 점진적 ablation (Table 5, Navsim)
- Qwen2.5-VL-3B 단독 84.8 → VQA+CoT 86.4 → **DVD**+CoT 87.1 → +**2D-TGP** 88.2 → +**GRPO** 90.4.
- DVD가 conventional VQA pretraining을 대체하며 우위 (Planner 기준 87.1 vs 86.4). TGP-Prompter 단독(2D-TGP를 Eq. 2로 BEV 변환)도 87.3/88.2 달성.

### 하이퍼파라미터 (Table 3, 4)
- λ_TGP=0.1이 최적 (87.6 PDMS): 작으면(0.05) trajectory cue 활용 부족, 크면 보조 task 과다 강조.
- 블록 분할 **2×4가 최적** (87.6) vs 4×7 (87.1) vs patch-by-patch (86.5) — bbox 주변 문맥까지 포착하는 넓은 receptive field가 유리, patch 단위는 bbox 밖 patch가 신호를 희석. w/o DVD는 86.2.

### DVD 강건성 (Table 7)
- bbox random drop/jitter 20% 노이즈에도 86.8/87.1로 w/o DVD(86.2)보다 우위 유지, 40% 심한 노이즈에서는 역효과(85.3/86.6).

### Train-test gap (Table 9, Figure 5)
- 추론 시 predicted 2D-TGP(90.4) vs ground-truth 누출(90.8) — **gap 0.4 PDMS에 불과**. 2D-TGP L2 error 구간별 PDMS는 완만하게 감소 → 중간 수준 오차에 강건.

### Dual-model 설계 & 효율 (Table 8, 10)
- Prompter-Planner dual (88.2) > Prompter 단독 (87.3) > 단일 모델 multi-turn QA (87.0) — 단일 모델의 두 instruction 동시 학습은 어려움. 대신 latency 3.03s/메모리 17.2GiB (H100)로 2배 비용.
- L20 GPU 런타임: DriveTeach 3.18s < AutoVLA action-token 3.95s < AutoVLA text-waypoint 7.65s — 2D-TGP의 시각적 spatial 강화가 **CoT reasoning token 소모를 줄여** dual-model임에도 더 빠름.

---

## 7. Attention 분석 (Analysis 1)

- **Attention Mass (AM)**: waypoint token → visual patch attention을 평균한 map A에서 Grounding DINO bbox 내부 attention 합을 총 box 면적으로 정규화 (Eq. 10).
- Navtrain 1k scene 샘플: Qwen2.5-VL-3B baseline AM 2.28×10⁻² (std 3.12) vs TGP-Planner **3.19×10⁻²** (std 3.77) — DVD가 traffic-critical object로의 attention을 실질적으로 강화함을 정량 검증 (Table 6).

---

## 8. 정성적 분석

- Figure 1: baseline의 분산된 attention 대비 DriveTeach-VLA는 신호등·차량 등 traffic-critical 객체에 명확히 집중.
- Figure 4: predicted/GT 2D-TGP 시각화 — 좌회전 등 driving behavior와 2D-TGP가 강하게 연동되며, feasible driving region을 이미지 위에서 직관적으로 표시.

---

## 9. 관련 연구 대비 위치

- 궤적 생성 두 갈래(텍스트 waypoint autoregression vs 별도 diffusion/MLP planner) 중 **텍스트 waypoint autoregressive** 노선 — MLLM native 능력 탐구가 목적.
- AutoVLA(action token + GRPO), ReCogDrive(diffusion planner + RL), Poutine(CoT 설계), CuriousVLA(exploration)의 계보를 이으며, 차별점은 **VQA pretraining을 vision-centric DVD로 대체**하고 IL/RL 전 단계에 2D-TGP spatial guidance를 주입한 것.

---

## 10. 강점

- "text-centric supervision → spatially ungrounded attention"이라는 문제 진단이 AM 지표로 정량 검증되어 설득력 높음.
- **VQA pair 없이** 순수 시각 self-distillation로 driving prior 주입 — 라벨링 비용 측면의 실용적 기여.
- 2D-TGP는 MLLM의 intrinsic 2D grounding과 자연스럽게 정합하며, z=0 가정하의 가역 투영으로 검증 가능성까지 확보한 우아한 설계.
- Teacher forcing train-test gap이 0.4 PDMS로 작음을 직접 검증하는 등 ablation이 매우 철저 (λ_TGP, 블록 크기, detector 노이즈, dual vs single, oracle gap, 효율).
- 3B×2 구성으로 8B 단일 모델 대비 우위, CoT token 절약으로 AutoVLA보다 빠른 추론. 코드 공개.

---

## 11. 한계 및 논의

- **Dual-model 설계로 latency/메모리 2배** (3.03s, 17.2GiB) — 실시간 주행(통상 ≥10Hz 요구)과는 거리가 큼. 0.3Hz 수준의 추론 속도는 학술 벤치마크 맥락에서만 유효.
- DVD가 Grounding DINO 검출 품질에 의존 — 40% 수준의 심한 노이즈/도메인 시프트에서는 오히려 성능 저해 (Table 7).
- Teacher forcing으로 학습된 Planner가 Prompter의 대오차 2D-TGP에는 여전히 취약 (Figure 5의 고오차 구간 PDMS 하락).
- nuScenes에서는 CoT/GRPO 미적용 설정이라 full pipeline의 open-loop 일반화는 미검증. front-view 단일 카메라 중심 설계.
- NAVSIM에서 end-to-end 계열 최고치(ASSCG 91.4)에는 미달.

---

## 12. 종합 평가

DriveTeach-VLA는 AD-VLA 학습의 근본 문제 — 텍스트 중심 supervision이 planning에 필요한 spatial grounding을 만들지 못한다는 점 — 를 명확히 진단하고, "what to see"(DVD 시각 증류)와 "where to look"(2D-TGP 투영 조건화)이라는 두 직교적 해법으로 정면 공략한 잘 설계된 연구다. VQA pretraining의 대체재로서 bbox-augmented self-distillation을 제시한 점, MLLM의 2D 좌표 해석 능력을 driving guidance로 전환한 2D-TGP, 그리고 oracle gap·detector 노이즈·dual-model 필요성까지 검증한 촘촘한 ablation이 돋보인다. NAVSIM 90.4 PDMS(VLA 최고)와 nuScenes 양 프로토콜 1위, 게다가 CoT token 절약으로 dual-model임에도 AutoVLA보다 빠르다는 효율 분석까지 완성도가 높다. dual-model의 배포 비용과 실시간성 한계는 남지만, "vision-grounded spatial guidance가 reasoning token 의존을 줄인다"는 통찰은 AD를 넘어 로봇 VLA 전반에 시사점이 있다. Action head는 2D-TGP 조건 하 텍스트 waypoint를 autoregressive 디코딩하는 구조로 분류됨.

**Score: 9.5 / 10**

<!-- VERIFIED: pdf -->
