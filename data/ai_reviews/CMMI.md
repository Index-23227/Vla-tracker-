# CMMI: A Collaborative Multi-Modality Interaction for VLA-based End-to-End Autonomous Driving

> **한 줄 요약**: 기존 driving VLA가 주행을 VQA 문제로 다루고 이종 센서 간 상호작용이 약하다는 문제를 지적하고, Janus-1.5B 기반 VLA에 (1) **Affinity-Guided Optimal Transport(AGOT)**로 카메라↔LiDAR 양방향 상호작용, (2) **Distribution-Consistent Modality Transfer(DCMT)**로 이종 분포 정렬, (3) **Discrete Flow Matching 기반 multi-trajectory 계획 + 인지 기반 궤적 최적화**를 결합. NAVSIM **PDMS 92.2 / EPDMS 87.0**, Bench2Drive **DS 77.45 / SR 56.00**, nuScenes 평균 L2 0.30 m.

---

## 1. 배경 및 동기

- **교차 모달 불일치·이질성**: RGB와 LiDAR는 같은 장면의 물리적으로 일관된 관측이어야 하지만, 기존 early/late fusion은 구조적 의미가 없고 휴리스틱하며 이론적 근거가 약함.
- **VLA의 3D 인지 한계**: 3D 사전학습 부재로 좌표–객체 의미 정렬이 약하고, 숫자를 자릿수 단위로 처리하는 토크나이즈 때문에 연속 waypoint 정밀도가 떨어짐.
- **명시적 안전 추론 부재**: 기존 E2E-AD는 충돌·차선 이탈 같은 위험을 암묵적으로만 다루어 long-tail 상황에서 취약.
- 대부분의 VLA 주행 모델이 주행 결정을 질문–답변 형태로 생성해 신뢰할 수 있는 동작 결정 메커니즘이 부족.

---

## 2. 문제 정식화

- Multi-input-multi-output 멀티태스크: 보조 과제로 **3D 객체 검출 + BEV 맵 분할**, 주 과제로 **ego 궤적 예측**.
- 입력: 전방 이미지 I, LiDAR 포인트 클라우드 P, 변환 행렬, 자연어 navigation prompt.
- 출력: τ = {(x_t, y_t)}_{t=1}^{T_f}, T_f = 8.
- 2단계: (1) 상호작용 모델을 검출·분할로 학습하며 융합된 카메라 visual token과 LiDAR BEV token 생성, (2) 사전학습 모델을 언어 가이드 궤적 예측으로 fine-tune.

---

## 3. 방법론: Multi-Modality Interaction

### 3.1 Affinity-Guided Optimal Transport (AGOT)
- 주(main)·보조(auxiliary) 모달 간 유사도 측정으로 **양방향 transport plan**을 구성해, 동일 의미 내용은 교환하고 모달 고유 정보는 보존.
- 이산 최적수송(Kantorovich 완화)의 결합 분포 γ를 모달 간 대응으로 사용. 학습 목표: 정규화 loss L_reg + OT loss L_ot.

### 3.2 Distribution-Consistent Modality Transfer (DCMT)
- 이종 모달 feature를 통일된 잠재 분포로 정렬해 교차 모달 상호작용 강화. 학습 목표: L_unf + L_consist.
- 언어 prompt와의 상호작용(prompts interaction)도 포함.

---

## 4. 방법론: Multi-Trajectory Planning & Refinement

### 4.1 Multi-modal Multi-Trajectory Planning
- **WAM-Flow** 방식의 Discrete Flow Matching(DFM) 플래너를 확장: 숫자 토큰 임베딩으로 ego 상태 토크나이즈 및 waypoint 디토크나이즈.
- Janus 토크나이저에 수치 grounding 토큰 **20,001개** 추가, 이미지 384×384 → 576 visual token, BEV token 196개.
- 후보 궤적 **5개**, 각 8 waypoint.
- 계획 loss: **match**(최소 한 궤적이 GT에 부합) + **diversity**(denoising step 간 궤적 붕괴 방지) + **consistency**(denoising 과정에서 오차가 단조 감소하도록 coarse-to-fine 유지).

### 4.2 Perception-Oriented Trajectory Refinement
- GRPO 기반 RL(WAM-Flow 등)의 안전 제약이 암묵적이고 학습 데이터 의존적이라는 비판에서, **CHOMP에서 영감받은 해석 가능한 목적함수 최적화**를 도입 — **추가 정책 학습 없음**.
- Γ(τ̃) = Γ_prior(DFM 분포를 prior로, 고확률 주행 manifold 근처 유지) + Γ_risk(3D 검출 기반 detection risk cost map + BEV 분할 기반 semantic risk cost map) + Γ_smooth.
- 각 궤적(16차원 변수)을 최적화 후 최선 궤적 선택.

---

## 5. 실험 설정

- **데이터셋**: NAVSIM(계획, open-loop), Bench2Drive(CARLA closed-loop, 44개 상호작용 시나리오), nuScenes(검출·분할·open-loop 계획), Argoverse 2 Sensor(검출·분할).
- **지표**: NAVSIM PDMS/EPDMS, Bench2Drive DS/RC/IS/SR + 세부 위반 점수, nuScenes L2/Collision/Intersection, mAP/NDS/mIoU.
- **구현**: 8× A6000 48GB, 계획 학습 4개 순차 단계, 인지 사전학습 nuScenes 20 epoch 후 NAVSIM에 보조 감독(HD 의미 맵 CE + 3D box L1)으로 fine-tune, AdamW(weight decay 0.01), 기반 VLA backbone **Janus-1.5B**.

---

## 6. 주요 결과: 궤적 계획

### NAVSIM navtest PDMS (Table 1)
| Method | Backbone | Input | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---|---|---|---|---|---|---|---|
| DiffusionDrive | – | 3×Cam+LiDAR | 98.2 | 96.0 | 94.8 | 100 | 82.2 | 88.1 |
| GoalFlow | – | 3×Cam+LiDAR | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| SafeDrive | – | 3×Cam+LiDAR | 99.5 | 99.0 | 97.2 | 84.3 | 100 | 91.6 |
| AutoVLA | Qwen2.5-3B | 3×Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| WAM-Flow | Janus-1.5B | 1×Cam | 99.2 | 98.3 | 97.0 | 99.7 | 82.3 | 90.3 |
| **Ours** | Janus-1.5B | Cam+LiDAR | 99.0 | 98.5 | 98.3 | 100 | 88.6 | **92.2** |

### NAVSIM EPDMS (Table 2)
- **87.0 EPDMS** (NC 99.0, DAC 98.5, EP 88.6, TTC 98.3, DDC 99.0, LK 97.6, EC 98.7) — GaussianFusion 85.0 대비 +2.0, TransFuser 77.8 대비 +9.2.

### Bench2Drive closed-loop (Table 3)
- **DS 77.45, RC 88.23, IS 0.90, SR 56.00** — VLR-Driver(75.01 / 86.09 / 0.87 / 50.00) 대비 전 core 지표 개선. 보행자 충돌 0.55(vs 0.72), 차량 충돌 2.82(vs 2.83).

### nuScenes open-loop (Table 4)
- 평균 L2 **0.30 m**(1s 0.14 / 2s 0.27 / 3s 0.50), 평균 충돌률 **0.21%**, 평균 intersection 1.27% — SpaceDrive(0.32 / 0.23 / 1.27), OmniDrive(0.33 / 0.30 / 3.00) 대비 우수 또는 동률.

---

## 7. 주요 결과: 장면 인지

- **nuScenes 3D 검출 (Table 5)**: val mAP 73.7 / NDS 77.2, test mAP 74.0 / NDS 78.0 — BEVFormer 대비 val +2.5 mAP / +4.0 NDS.
- **Argoverse 2 검출 (Table 6)**: mAP 43.5 (GeoFormer 41.7 대비 +1.8), 차량 79.0, 보행자 75.9, 트럭 26.3.
- **nuScenes BEV 분할 (Table 7)**: 평균 mIoU 71.1 (BEVFusion 62.7 대비 +8.4).
- **Argoverse 2 BEV 분할 (Table 8)**: 평균 IoU 63.8 (CMGFA 63.3 대비 +0.5).

---

## 8. Ablation 분석

- **계획 loss (Table 9)**: match만 86.9 → +diversity 87.5 → +consistency **92.2** PDMS.
- **VLA backbone (Table 10)**: Qwen2.5-3B 91.8 / LLaVA-7B 92.1 / Janus-1.5B 92.2 — backbone 비의존.
- **궤적 refinement 목적항 (Table 11)**: 전체 92.2; prior 제거 88.0, detection risk 제거 85.1, semantic risk 제거 85.6, smooth 제거 91.5.
- **상호작용 방식 (Table 12)**: BEVFusion 90.3 PDMS / 85.7 EPDMS, LiDAR-only 91.0 / 86.4, w/o DCMT 92.0 / 86.9, w/o prompt interaction 90.1 / 86.0, 전체 92.2 / 87.0.
- **AGOT·DCMT 구성요소 (Table 13)**: L_ot 제거 시 mAP 71.3, PDMS 91.1; L_reg 제거 시 mAP 72.0, PDMS 91.0; L_consist 제거 시 PDMS 91.8; L_unf 제거 시 PDMS 91.7.
- **속도–성능 (Figure 7)**: 1-step denoising 90.1 PDMS @ 11.5 samples/s, 5-step 92.23 PDMS @ 3.0 samples/s (WAM-Flow 2.0 samples/s).

---

## 9. 관련 연구와의 위치

- **융합 방식**: flatten fusion(TransFuser 등)과 BEV fusion(BEVFusion, BEVFormer)과 달리, 의미 인식 transport plan으로 "같은 의미는 교환, 고유 정보는 보존"하는 상호작용.
- **E2E-AD 다중 궤적**: VADv2(anchor scoring), DiffusionDrive(truncated diffusion), DriveSuprim(coarse-to-fine) 계열 대비, VLA 내 DFM 다중 궤적 + 인지 기반 사후 최적화.
- **Driving VLA**: EMMA·DrivingGPT(언어 생성형 계획), dual-system(VLM + diffusion planner), WAM-Flow(DFM) — 본 논문은 WAM-Flow 계열을 LiDAR 상호작용과 안전 최적화로 확장.

---

## 10. 강점

1. **폭넓은 평가**: open-loop(NAVSIM v1/v2, nuScenes)와 closed-loop(Bench2Drive), 인지(nuScenes, Argoverse 2)까지 네 개 데이터셋에서 일관된 개선.
2. **해석 가능한 안전 최적화**: 검출·분할 결과로 만든 risk cost map으로 궤적을 사후 정제 — 추가 정책 학습 없이 안전 제약을 명시화.
3. **체계적 ablation**: loss 항, backbone, refinement 목적항, 상호작용 방식, OT/DCMT 구성요소를 모두 분해.
4. **Backbone 이식성**: Qwen2.5-3B, LLaVA-7B, Janus-1.5B에서 91.8~92.2로 안정.
5. **코드 공개 예정**: 프로젝트 페이지(GitHub S-JingTao/CMMI) 제공.

---

## 11. 한계 및 논의

1. **비교 공정성**: NAVSIM Table 1에서 LiDAR를 쓰는 본 방법을 1×Cam VLA(WAM-Flow 등)와 같은 "VLA-based" 그룹에서 비교 — 입력 모달 차이가 개선의 상당 부분일 수 있음(LiDAR-only counterpart 91.0이 WAM-Flow 90.3보다 높음).
2. **Refinement 의존성**: detection/semantic risk 항 제거 시 PDMS가 85.1/85.6까지 하락 — 최종 성능이 학습된 정책보다 추론 시 최적화에 크게 의존하며, 인지 오류에 민감(저자도 인정).
3. **연산 오버헤드**: OT 기반 상호작용이 dense 토큰에서 추가 비용 발생(저자 언급), 5-step 기준 3.0 samples/s.
4. **실험 세부 불명확**: "4개 순차 학습 단계"의 구체적 구성, 학습 epoch/learning rate 등이 본문에 충분히 기술되지 않음. Table 1에서 SafeDrive의 Comf./EP 값이 뒤바뀐 듯한 표기 등 표 정리 품질 이슈.
5. **Bench2Drive 비교 대상 제한**: 비교 결과가 모두 VLR-Driver 논문에서 인용되어, 최신 강력한 closed-loop 방법들과의 비교가 부족.

---

## 12. 총평 및 예상 질문

- **총평**: WAM-Flow 계열 DFM driving VLA에 최적수송 기반 카메라–LiDAR 상호작용과 인지 기반 궤적 최적화를 더해, 계획·인지 양쪽에서 폭넓게 SOTA급 수치를 보고한 시스템 논문이다. 다만 성능 이득의 상당 부분이 LiDAR 추가와 추론 시 최적화에서 오는 것으로 보여, "VLA 정책 자체"의 기여를 분리한 분석이 더 필요하다.
- **예상 질문**
  - Q1. Refinement 없이 DFM 정책 단독의 PDMS는 얼마인가(Figure 6 외 수치)?
  - Q2. 동일 LiDAR 입력을 WAM-Flow에 단순 융합(BEVFusion식)했을 때와의 차이(Table 12의 90.3)가 OT 상호작용의 순수 기여인가?
  - Q3. Bench2Drive에서 refinement의 실시간 연산 비용은 어느 정도이며, 인지 오류가 큰 장면에서 실패 양상은?
  - Q4. 다중 카메라(3×/6×Cam) 입력으로 확장 시 추가 이득이 있는가?

<!-- VERIFIED: pdf -->
