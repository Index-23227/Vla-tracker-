# Qwen-RobotManip: Alignment Unlocks Scale for Robotic Manipulation Foundation Models

> **한 줄 요약**: Qwen3.5-4B VL 백본 + flow-matching DiT action expert에 *cross-embodiment alignment*(80-D canonical state-action, camera-frame delta-pose action, in-context policy adaptation)를 결합해, OXE·RoboMIND·DROID·EgoDex·VITRA 등에서 합성한 ~38,100시간 데이터를 흡수하고 LIBERO/RoboTwin SOTA와 LIBERO-Plus·RoboCasa365·EBench·RoboTwin-IF/XE 등 OOD 벤치마크 전반을 휩쓴 Qwen 팀의 generalist VLA foundation model.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 한계
- VLA 모델들이 LIBERO/RoboTwin 같은 *in-domain* 벤치마크에서 95%+로 포화되는 반면, 카메라 viewpoint·robot state·clutter가 흔들리는 OOD에서는 급격히 무너진다.
- π0.5, GR00T-N1.x, Cosmos-Policy 등 cross-embodiment 시도는 *shared architecture*나 *embodiment token*으로 끝났을 뿐, *같은 물리 동작이 데이터셋마다 수치적으로 다르게 표현되는 본질적 alignment 문제*를 풀지 못함.
- 즉 **데이터 다양성만 늘려도 alignment가 없으면 시너지가 아니라 간섭**이 된다 (논문이 “alignment is a prerequisite for data scaling”이라고 강하게 주장).

### 핵심 질문
- **언어 모델처럼 robotic manipulation도 alignment + scale로 generalization을 얻을 수 있는가?**
- **proprietary 데이터 수집 없이 open-source robot data + human ego-video만으로 진정한 generalization VLA를 만들 수 있는가?**

📌 [Figure 1 삽입] — 38,100h 데이터, 3-축 alignment, 4 영역 generalization 결과 요약

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 컴포넌트 | 구성 | 역할 |
|---------|-----|------|
| VL backbone | Qwen3.5-4B (early-fusion VL, dynamic-resolution ViT) | 멀티뷰 vision + instruction + 구조화 prompt + history 토큰을 함께 인코딩 |
| Action expert | Diffusion Transformer (10 blocks, $D_{act}=768$, 12 heads) | flow-matching으로 80-D state-action chunk 디노이즈 |
| Cross-attention | 짝수 block은 visual token, 홀수 block은 language token에 alternating cross-attn | spatial vs. linguistic grounding 분리 |
| 추론 | Beta(1, 1.5) timestep + 4-step Euler integration | low-latency real-time 제어 |

> ❓ **예상 질문**: DiT가 10블록·768차원이면 0.5B 이하 수준으로 보이는데, π0.5(0.3B 액션 헤드)·Qwen-VLA(1.15B DiT)와 비교해 너무 작지 않은가?
> **답변**: Qwen-RobotManip은 *backbone에 지능을 몰아주고 action expert는 모달리티 변환에 한정*하는 디자인 철학. 4-step Euler로 inference가 가벼워지고, alignment·co-training이 작은 expert로도 충분한 표현력을 제공한다는 가정. ablation은 expert size scan을 제공하지 않아 sweet-spot은 미검증.

### 2.2 80-D Canonical State-Action Representation

각 arm을 29-D 블록(joint 7 + EEF pose 9 [pos 3 + 6D rot] + gripper 1 + dex hand 12)으로 표현하고, **per-dimension binary mask**로 미사용 차원의 gradient를 차단:
- single-arm Franka는 한쪽 arm 블록만 채우고 나머지는 zero+mask
- ALOHA는 양쪽 arm 채움
- dexterous hand 보유 모델은 hand-joint 12-D 추가 활용
- 남는 22-D는 mobile-base velocity 등 future extension용 reserve

### 2.3 Camera-Frame Delta-Pose Action

핵심 통찰: **시각적으로 비슷해 보이는 모션은 카메라 프레임에서도 수치적으로 비슷해야 한다.**
- 절대 base-frame 좌표는 데이터셋마다 base 정의가 달라 interference 발생
- 식 (5): $a_p = \begin{pmatrix} {}_c^eR\,{}_e^{e^*}R\,{}_c^eR^\top & {}_c^eR\,{}_e^et_{e^*} \\ 0 & 1 \end{pmatrix}$ — relative EEF rotation을 camera frame에 conjugate
- Camera Positional Encoding (CaPE)을 keys/queries/values/outputs 모두에 적용 → world-frame origin이 attention dot-product에서 algebraically 소거되어 *상대 카메라-EEF 포즈만* 남음
- 카메라 캘리브레이션 가능 여부를 binary flag로 conditioning → camera-frame delta 모드 ↔ base-relative 모드 스위치

> ❓ **예상 질문**: camera-frame delta가 RoboTwin-XE에서 23.9% (joint 14.5%, π0.5 eef 7.5%)로 3.2× 향상시킨 결과는 정말 *alignment* 덕인가, 아니면 단순히 EEF가 morphology-invariant라서인가?
> **답변**: 같은 모델이 base-frame eef는 표시 안 했지만, π0.5(eef) 7.5%와 비교해 동일 action space에서도 3× 차이가 난다는 것은 *camera-grounding + CaPE 자체*가 효과적임을 시사. 다만 표가 "joint vs eef"만 비교하므로 base-frame eef ablation이 빠진 점은 약점.

### 2.4 In-Context Policy Adaptation

- 한 episode 내 최근 H개의 *(observation, state, K-step action chunk)* 튜플을 컨텍스트로 주입
- 시각 관찰은 backbone visual encoder에 prepend, 상태·액션은 별도 MLP로 hidden space에 투영 (각 chunk에 temporal embedding + slot embedding)
- 효과: LIBERO-Plus Robot 차원에서 75.5 → **83.9** (+8.4), RoboTwin-C2R Hard 62.6 → **69.4** (+6.8)
- 핵심: parameter update 없이 *intra-episode kinematic prior*로 새로운 robot/scene에 즉시 적응

### 2.5 Dual-Stream Co-training

- VLA 데이터 배치와 VL 데이터 배치를 **분리(mutually exclusive batch)**해서 교차 학습
- VL 믹스 ~28M 샘플: VQA, spatial grounding, OCR, STEM, multilingual, 그리고 *embodied chain-of-thought (ECoT)*, egocentric video understanding, 2D trajectory prediction 같은 embodied-centric 데이터
- 목적: action prediction pressure로 VLM의 perception·reasoning이 catastrophic forgetting되는 것을 막음 → RoboTwin-IF (instruction following) 72.2 vs. π0.5 49.6에서 효과 확인

📌 [Figure 3 삽입] — 전체 아키텍처, 5스테이지 신호 필터링 + 3개 cross-modal 체크

---

## 3. 데이터 전략

### 38,100시간의 구성

| 구분 | 출처 | 시간 |
|------|------|------|
| Robot single-arm | OXE (Fractal/Bridge/BC-Z), DROID, RH20T | 3,808h |
| Robot dual-arm | AgiBotWorld-Beta, RoboCOIN, RDT, RoboMIND | 6,744h |
| Robot mobile/humanoid | InternData-A1, Galaxea Open-World | 868h |
| Human ego-hands | EgoDex (732h), VITRA (247h Ego4D+EPIC), EgoVerse (954h) | 1,933h |
| **Human-to-Robot 합성** | 15-platform retargeting | **24,808h** |

### Human-to-Robot 파이프라인

1. **Action retargeting**: thumb + virtual finger(0.7·index + 0.3·middle)로 가상 gripper 정의 → position·width·orientation 추출 (식 1-2)
2. **Visual alignment**: SAM3 hand mask → ProPainter inpainting → MuJoCo IK로 15개 morphology 각각의 base placement grid search → Depth Anything v3로 occlusion-aware composite (식 4)
3. **Speed alignment**: EgoDex 60%, EgoVerse 45%, VITRA 25% 로 다운샘플링 → robot 속도 분포와 매칭

### 5-Stage 신호 필터링 + 3 Cross-modal Check

| 단계 | 목적 | 인상적 사례 |
|------|------|-------------|
| S1 Sudden change | residual + 2nd/3rd diff threshold | InternData-A1은 collision 시 에피소드 전체 폐기 |
| S2 State-Action 시차 | cross-correlation + directional agreement | **RoboMIND UR 데이터 81% 제거** |
| S3 Extreme value | $[q_{01}-\alpha\Delta, q_{99}+\alpha\Delta]$ 밖 제거 | gripper는 bimodal로 예외 |
| S4 FK 일관성 | Pinocchio FK vs. logged EEF | TCP offset·rotation rep 자동 수정 |
| S5 Base/orient 정렬 | per-dataset rotation correction | 통일 world frame |
| C1 Instruction | 3-stage VLM 평가 + multi-expert adjudication | 의미 misalignment 제거 |
| C2 Video-state | URDF 투영 mask vs. SAM3 mask IoU | 미달 시 camera param 최적화 또는 제거 |
| C3 Video quality | black/blurred/static 제거 | gripper closure key frame은 보존 |

> ❓ **예상 질문**: 진짜 *open-source-only*인가? Qwen 팀의 proprietary 데이터가 슬쩍 들어간 것은 아닌가?
> **답변**: 논문은 robot data, ego-video 모두 출처를 명시했고 proprietary 수집 없음을 강조. 단, VL co-training 믹스에는 “proprietary data, open-source datasets... carefully synthesized embodied-centric data”라고 적어 일부 proprietary VL 데이터는 포함 — VLA action data는 open-source이지만 보조 VL 데이터는 그렇지 않다.

---

## 4. 실험 결과

### 4.1 In-distribution 벤치마크 (Table 3)

| Model | LIBERO | RoboTwin-Easy | RoboTwin-Hard |
|-------|-------|--------------|--------------|
| π0.5 | 97.6 | 82.7 | 76.8 |
| StarVLA | 98.0 | 85.7 | 87.3 |
| Abot-M0 | 98.6 | 86.1 | 85.1 |
| Being-H0.7 | 99.2 | 90.2 | 89.6 |
| Qwen-RobotManip-scratch | 98.2 | 88.7 | 88.4 |
| **Qwen-RobotManip** | **99.1** | **93.4** | **92.5** |
| **Qwen-RobotManip-Context** | **99.2** | **93.7** | **94.0** |

→ LIBERO는 거의 포화. RoboTwin은 Being-H0.7 대비 Easy +3.5 / Hard +4.4 으로 상대적 gap 큼.

### 4.2 OOD 일반화 (Tables 4–7)

| Benchmark | π0.5 | Qwen-RobotManip | Qwen-RM-Context |
|-----------|------|-----------------|-----------------|
| LIBERO-Plus (7-축 평균) | 84.4 | 89.0 | **91.4** |
| RoboTwin-C2R Hard | 47.9 | 62.6 (joint) | **69.4** |
| RoboCasa365 (total) | 16.9 | **35.9** | 33.8 |
| EBench (SR / Score) | 27.1 / 41 | **45.6 / 60** | 43.6 / 59 |
| RoboTwin-IF (avg) | 49.6 | **72.2** | 72.0 |

가장 인상적인 것은 **RoboCasa365 Composite-Unseen 14.9% vs. 차선 RLDX-1 5.4% (≈3× 향상)** 와 **RoboTwin-IF Pick-Diverse 79 / Place-Relative 57 / Operate-Mic-Drawer 42 등 instruction grounding 전 항목에서의 +20~35 점프**.

### 4.3 Zero-shot Cross-Embodiment (Table 9, RoboTwin-XE)

AgileX ALOHA에서만 학습 → 다른 embodiment에 zero-shot:

| Action space | ARX-X5 | UR5-WSG | Franka | Total |
|--------------|--------|---------|--------|-------|
| π0.5 (joint) | 24.6 | 2.2 | 0.9 | 9.2 |
| π0.5 (eef) | 11.5 | 10.0 | 1.1 | 7.5 |
| **Qwen-RM (joint)** | 37.6 | 4.1 | 1.8 | 14.5 |
| **Qwen-RM (eef)** | **42.9** | **22.8** | **5.9** | **23.9** |

→ Camera-frame eef는 ARX(visually similar)에서 가장 좋고, Franka(7-DOF + 큰 reach)에서 가장 약함 — *시각·운동학적 유사도에 비례*.

### 4.4 RoboChallenge Table30-v1 Generalist Track (Table 14)

| Model | Avg SR | Process Score |
|-------|--------|---------------|
| π0_generalist | 9 | 20.22 |
| GR00T-MULTI | 15.33 | 32.29 |
| π0.5_generalist | 17.67 | 31.27 |
| DM0_generalist | 37 | 48.43 |
| **Qwen-RobotManip (Lira)** | **45** | **59.83** |

→ **1위, DM0 대비 +8% SR / +11.4 process score** (20% relative). 특히 ALOHA bimanual 8 태스크 평균에서 40% vs. π0.5 21.2%, *pour fries into plate*는 유일하게 30% 성공.

### 4.5 Real-World CobotMagic ALOHA

| Setting | π0.5 | StarVLA | Qwen-RobotManip |
|---------|------|---------|-----------------|
| In-domain 7 tasks | 42.9% | 20.0% | **88.6%** |
| OOD 4 tasks (clutter / lighting / left-right) | 37.5% | 0.0% | **87.5%** |

특히 *banana-on-towel* 디스코 라이팅(9/10), *left-right-bowl-stacking*(10/10) 같은 cluttered + 좌우 reference 태스크에서 π0.5는 1/10에 그쳐 generalization gap이 극명.

### 4.6 ARX few-shot + cross-embodiment skill transfer (Table 13)

CobotMagic 6K + ARX 130 demo만으로 fine-tune → 4가지 unseen ARX 태스크에서:
- w/o UnifiedSpace: 7.5%
- w/o UnifiedEEF: 12.5%
- **Full**: 55.0% (**4× 이상 향상**)

→ canonical state-action + camera-frame EEF *둘 다* 필수임을 강하게 입증.

---

## 5. 비판적 분석

### 강점
- 데이터·alignment·평가 세 축을 모두 새로 정의. 특히 *human-to-robot 24,808h 합성*은 ego-video를 robot-grade 데이터로 끌어올린 첫 대규모 시도.
- Open-source-only 데이터만으로 π0.5/GR00T를 **OOD axis 전 항목에서 능가** — 데이터 alignment의 ROI가 대단히 큼.
- *RoboTwin-XE, RoboTwin-IF* 같은 새로운 OOD 벤치마크 제안 → 분야 evaluation 표준 끌어올림.
- Real-world ID 88.6%, OOD 87.5% — 시뮬→실 gap이 매우 작음.

### 약점 / 미흡한 점
1. **LIBERO sub-suite 점수 미공개**: spatial/object/goal/long 별 점수 없이 99.1만 보고 → 다른 모델과 fine-grained 비교 한정.
2. **Compute / latency transparency 부족**: Qwen3.5-4B + DiT inference latency, GPU 시간, 4-step Euler의 실제 control Hz가 모두 미보고.
3. **Action expert size ablation 없음**: 10블록·768차원이 sweet spot인지, scaling law 가 적용되는지 미검증.
4. **Camera 캘리브레이션 의존성**: camera-frame action은 intrinsics + extrinsics 필요. real-world에서 캘리브레이션 정확도가 떨어질 때 성능 손실은 정량화 안 됨 (auxiliary flag fallback은 있으나 quality 측정 없음).
5. **In-context adaptation의 메모리/추론 비용**: H개 history chunk 추가하면 backbone 입력 길이가 늘어 inference latency 증가 — 정량 보고 없음.
6. **합성 데이터 품질 검증**: 24,808h 중 IK 실패·압축 artifact가 학습에 미친 영향은 ablation 미제공.

### Attribution
- 성능 향상이 (a) Qwen3.5-4B backbone, (b) 80-D canonical + masked loss, (c) camera-frame delta + CaPE, (d) in-context adaptation, (e) human-to-robot 24,808h 중 **무엇이 가장 결정적인가?** Table 13(ARX cross-embodiment)에서 UnifiedSpace/UnifiedEEF 둘 다 큰 효과는 보였으나, **다른 OOD 축에서는 각 요소를 분해한 ablation이 없음**.

---

## 6. 한계 및 미해결 문제

1. **Franka에서 5.9%** — 7-DOF + larger reach가 camera-frame alignment만으로는 부족. 더 다양한 morphology pretraining이 필요할지, 아니면 alignment 자체의 한계인지 미해결.
2. **EBench Long-Horizon 29.9%, RoboCasa Composite-Unseen 14.9%** — 절대 수치는 여전히 낮음. *foundation* model이라기엔 long-horizon planning이 부족.
3. **언어 generalization 한계**: RoboTwin-IF Operate-Mic-Drawer 42% — 가장 어려운 spatial language grounding은 여전히 미해결.
4. **Real-world 평가 규모**: ALOHA에서 7+4 태스크는 환영할 만하지만 *수십 가지 contact-rich/deformable* 태스크에는 미치지 못함.
5. **Open-source 라이센스/체크포인트**: GitHub repo 존재하지만 학습 체크포인트, VL 데이터 라이센스 명시는 본문에 없음.

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — alignment-first 프레임워크, 24,808h human-to-robot 합성, 6개 OOD 벤치마크(2개 신규)는 분야 패러다임 전환 |
| **Technical depth** | ★★★★★ — canonical representation + CaPE + in-context adaptation + dual-stream co-training의 체계적 통합 |
| **Experimental rigor** | ★★★★★ — 7개 simulation + 4개 real-world platform, 500+ tasks, RoboChallenge 1위 |
| **Practical impact** | ★★★★★ — open-source-only 데이터만으로 π0.5/GR00T 능가, real-world 88.6%/87.5% |
| **Writing / transparency** | ★★★☆☆ — sub-suite 점수, latency, expert ablation 미공개 |

**강점**: alignment + scale 동시 달성, open-source-only 데이터로 SOTA, real-world OOD 87.5%. **약점**: ablation 분해 부족, latency/compute transparency 부족, Franka·long-horizon은 여전히 미해결.

### Score: **8.5 / 10**

---

## 8. 후속 연구 방향

1. **Action expert scaling law**: 0.5B → 3B → 10B로 키울 때 in-domain vs. OOD trade-off가 어떻게 바뀌는지 연구.
2. **Camera-frame action의 캘리브레이션 robustness**: noisy intrinsics/extrinsics에 대한 ablation.
3. **In-context adaptation으로 zero-shot fine-tuning**: 새로운 robot에서 한 episode만 demo로 받아 policy를 바로 adapt — Franka 5.9% 끌어올릴 수 있는가?
4. **Bimanual + dexterous hand combined**: dex hand 12-D는 canonical에 들어있으나 실험은 parallel gripper 중심. 5-finger hand 평가가 필요.
5. **Long-horizon 향상**: ECoT 데이터를 더 길게 (multi-minute) 만들고 in-context history H를 늘려 RoboCasa Composite-Unseen·EBench Long-Horizon 끌어올리기.
6. **Human ego-video 직접 학습 vs. retargeted robot data**: 어느 쪽이 더 효율적인가? 후자가 24,808h로 우세하지만 데이터 만드는 비용은 큼.

---

## 9. VLA-Tracker 관점 비교

| 모델 | LIBERO | RoboTwin (Easy/Hard) | LIBERO-Plus | Cross-Embod. |
|------|-------|---------------------|-------------|--------------|
| π0.5 | 97.6 | 82.7 / 76.8 | 84.4 | 7.5 |
| Qwen-VLA (Alibaba 4월 모델) | 97.9 | 86.1 / 87.2 | – | – |
| ACE-Ego-0 | – | RoboTwin v2 90.87 | – | – |
| Being-H0.7 | 99.2 | 90.2 / 89.6 | 84.8 | – |
| **Qwen-RobotManip** | **99.1–99.2** | **93.7 / 94.0** | **89.0–91.4** | **23.9** |

같은 Qwen 팀의 5월 Qwen-VLA보다 모든 축에서 향상되었고, 특히 *cross-embodiment*와 *OOD*에서 격차가 두드러진다.

---

## 10. 재현성 / 실험 환경

- **Hardware (실험)**: AgileX ALOHA / Franka / UR / ARX 실로봇 + Isaac Sim (EBench), MuJoCo (human-to-robot IK), RoboTwin/RoboCasa365/LIBERO-Plus 시뮬레이션
- **Training**: 명시적 GPU 수·시간 미보고 (~38,100h 데이터 + ~28M VL 샘플 규모로 보아 대규모 cluster 필요)
- **Inference**: 4-step Euler integration. 절대 Hz는 미보고. action chunk 길이 K (eq. 9) 명시되어 있으나 본문에 구체값 부재.
- **Code/checkpoint**: github.com/QwenLM/Qwen-RobotManip 공개 페이지 존재. 본문 작성 시점에서 weights 공개 여부 미명시.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|----------------|
| 1 | LIBERO 99.1과 Being-H0.7 99.2 차이가 사실상 noise level인데 SOTA라 부를 수 있나? | LIBERO 절대 수치는 포화. 가치는 OOD axis (LIBERO-Plus +6.6 over Being-H0.7, RoboTwin-IF +22.6 over π0.5, RoboTwin-XE +16.4 over π0.5)에 있음. |
| 2 | Human-to-Robot 합성 24,808h가 진짜 robot 데이터만큼 valuable한가? Scratch model이 LIBERO 98.2까지 가는데 의미가 크나? | LIBERO는 saturated여서 차이 없어 보임. 하지만 OOD에서 scratch 78.3 vs. pretrained 89.0 (LIBERO-Plus), scratch 22.6 vs. 62.6 (RoboTwin-C2R Hard) — 합성 데이터의 가치는 in-domain이 아니라 OOD에서 드러난다. |
| 3 | CaPE는 ZeroNVS/GTA에서 빌려온 trick인데, 진짜 novel한 부분은? | 단순 CaPE 채택이 아니라 *camera-frame delta action representation* + *키/쿼리/값/출력 모두에 CaPE 적용* + *embodiment-type embedding 결합*이 통합 디자인. 본 논문이 처음으로 manipulation에 적용. |
| 4 | RoboTwin-XE Franka 5.9%는 너무 낮은데 실용성 있는가? | Zero-shot이라는 점을 감안하면 π0.5의 1.1% 대비 5.4× 향상. 실용은 아니지만 alignment의 *방향성*이 옳다는 증거. 실제 배포에는 few-shot adaptation 필수. |
| 5 | Dual-stream co-training이 catastrophic forgetting을 막는다지만, VLM의 일반 ability(MMMU, MMBench)는 얼마나 보존되는가? | 본문 ablation에 없음. RoboTwin-IF 72.2가 간접 증거지만, 일반 VL 벤치마크 수치 미공개는 큰 약점. |
| 6 | In-context history H 개수가 늘면 backbone 입력이 매우 길어지는데 latency 영향은? | 본문 미보고. KV cache 활용은 가능하지만 visual encoder는 매 chunk 재계산이 필요해 보임. real-time control Hz 명시가 없는 게 결정적 약점. |
| 7 | 80-D canonical에 22-D reserve가 너무 많지 않은가? | mobile base velocity + future dexterous hand + tactile 등을 위한 여유분으로 보임. 현재는 zero+mask로 비용 없음. |
| 8 | π0.5와 비교해 정말 *데이터가 본질적인가 alignment가 본질적인가*? Scratch 결과는 데이터에 가깝고 alignment ablation은 ARX cross-embod에만 있음. | LIBERO-Plus에서 scratch 78.3 (alignment+데이터) vs. full 89.0 — alignment+pretraining이 +10.7. ARX에서 UnifiedSpace/EEF 제거 시 55→7.5 — alignment 단독으로 거의 모든 신호. 둘 다 필수이며 *데이터를 alignment로 가공해야 효력 발생*이 정답. |

---

## 12. 결론

Qwen-RobotManip은 *"VLA scaling을 위한 alignment-first 패러다임"*을 처음으로 풀스택으로 구현한 작업이다. 80-D canonical representation + camera-frame delta action + in-context policy adaptation의 세 가지 alignment 메커니즘이 ~38,100시간(open-source-only)을 진정한 학습 신호로 변환해, LIBERO/RoboTwin SOTA뿐 아니라 LIBERO-Plus, RoboCasa365, EBench, RoboTwin-IF, RoboTwin-XE 같은 OOD 평가 전반을 휩쓸고 RoboChallenge Table30-v1 generalist 1위, real-world ALOHA OOD 87.5%까지 도달했다.

분야적 함의는 두 가지다.
1. **데이터 다양성만으로는 부족하다** — 데이터 alignment가 prerequisite. 본 논문은 이를 정량적으로 입증한 첫 사례.
2. **평가 표준이 바뀌어야 한다** — in-domain LIBERO 99%로 모델을 평가하는 시대는 끝났고, OOD 축(perturbation, instruction, cross-embodiment)이 새 north star.

남은 과제는 (a) action expert scaling law, (b) camera 캘리브레이션 robustness, (c) Franka·long-horizon 한계 돌파, (d) latency/compute transparency다. 그럼에도 **open-source robotics community가 proprietary 데이터 없이 만들 수 있는 generalist VLA의 새 상한선을 정의했다**는 점에서 분야적 마일스톤이다.

<!-- VERIFIED: pdf -->
