# MobileVLA-R1 2.0: RL-Enhanced Reasoning for Mobile Robot Control

> **한 줄 요약**: NaVILA(LLaMA3-8B) 기반 VLA가 CoT를 생성하고, 관측과 추론 hidden state를 cross-attention으로 읽는 "reasoning-conditioned action decoder"가 이동 속도(Vx, Vy, ω)와 이산 행동 primitive α를 직접 예측하도록 한 MobileVLA-R1의 저널 확장판. 다중 granularity CoT SFT 후 오프라인 GRPO로 학습해 R2R-CE SR 69.8 / RxR-CE SR 73.1, QUARD 평균 0.76, G1 휴머노이드 실물 모바일 조작 full-task 56.7%(MobileVLA-R1 46.7%).

- **arXiv**: 2609.06251v1 (2026-09-05, cs.CV), IEEE TPAMI 투고
- **소속**: Peking University, South China University of Technology, National University of Singapore
- **코드**: https://github.com/AIGeeksGroup/MobileVLA-R1-2.0
- **전작**: MobileVLA-R1 (ECCV 2026) — 텍스트 출력의 결정적 파싱으로 명령 추출

---

## 1. 배경 및 동기

ECoT, CoT-VLA 등 추론형 VLA는 사람이 읽을 수 있는 추론을 만들지만, 추론과 실행 가능한 저수준 제어 사이의 연결은 약하다. 전작 MobileVLA-R1은 생성된 텍스트에서 규칙 기반으로 제어 명령을 파싱했기 때문에 추론 표현과 물리 행동 사이에 학습 가능한 인터페이스가 없었다. 이동 로봇은 부분 관측·긴 horizon 속에서 이동과 조작을 함께 조율해야 하므로, 추론(System-2) → 행동 결정(System-1) → 로봇별 제어기(System-0)로 이어지는 명시적 인터페이스가 필요하다는 것이 동기다.

## 2. 문제 정의

- 입력: RGB, depth, point cloud 관측과 자연어 지시.
- 출력: `<think>` 추론 r_t, `<answer>` 텍스트 u_t, 그리고 태스크 수준 물리 행동 ā_t = [V_x, V_y, ω, α] (α는 상호작용·자세·스킬 전환 등 이산 primitive).
- 관절 명령이 아닌 태스크 수준 목표를 예측하고, 이를 로봇별 고정 저수준 제어기가 실행.

## 3. 방법

- **MobileVLA-CoT 데이터 엔진**: R2R, RxR, QUARD에서 Gemini-2.5-Flash로 episode(18K)/step(78K)/navigation(38K) 수준 CoT를 생성하고, 형식·행동 일관성·안전·의미 4단계 반자동 검증으로 168K → 134K로 필터링.
- **멀티모달 백본**: NaVILA 초기화 LLaVA 스타일; RGB/DepthAnything V2/Point Transformer V3 인코더는 동결, projection과 LoRA만 학습.
- **Reasoning-conditioned action decoder**: locomotion·behavior용 학습 query 2개가 [관측 hidden; 추론 hidden]에 cross-attention(1층, 8 heads, d=4096) → 회귀 헤드(V_x, V_y, ω), softmax 헤드(α).
- **SFT**: L = L_CoT + λ_loc·L1(속도) + λ_beh·CE(primitive), 레이블 유무 마스크 사용. 먼저 episode+nav CoT로 장기 추론 정렬, 이후 step CoT로 decoder와 공동 학습.
- **오프라인 GRPO**: 입력당 8개 출력 샘플링, 고정된 decoder로 행동을 얻어 이동 보상(정규화 명령의 cosine), 행동 보상(exact match), 형식 보상으로 group-relative advantage 계산, KL 정규화. 환경 상호작용 없음.

## 4. 구현 세부

- LoRA r=16, α=32; SFT 3 epoch, 4×H20, AdamW lr 2e-4, cosine.
- GRPO: λ_mov=1.0, λ_beh=1.0, λ_fmt=0.2, lr 1e-6, β=0.04, clip 0.2, 1K step, H20 1장.
- 배포: 8B 모델과 decoder는 원격 H20에서 실행, 로봇은 센싱·전처리·저수준 제어 담당. 종단 지연 205–245 ms(약 4.1–4.9 Hz).

## 5. 평가 프로토콜

- **VLN-CE**: R2R-CE/RxR-CE val-unseen, NE/OS/SR/SPL/nDTW.
- **QUARD**: 6개 과제(Distinguish, Go-to, Go-avoid, Go-through, Crawl, Unload) 과제당 25 에피소드.
- **Unitree Go2**: Workspace/Corridor/Outdoor × Simple/Complex, 총 160 에피소드.
- **Unitree G1**: Tabletop/Shelf-Cabinet/Cluttered 각 4과제 × 10 trial = 120 에피소드. G1 데이터로는 전혀 학습하지 않으며, 조작 동작(reach/grasp/lift/place)은 제어기 쪽 고정 루틴이 수행.

## 6. 실험 설계의 요점

전작과 같은 데이터·백본에서 "결정적 파싱 → 학습형 decoder" 교체의 효과를 Table 8/9로 분리했다. G1 실험은 정책이 무엇을 할지(α)만 결정하고 어떻게 조작할지는 고정 루틴이 담당하므로, 휴머노이드 조작 능력 자체라기보다 태스크 수준 의사결정의 embodiment 전이를 평가하는 설정이다.

## 7. 주요 결과

**Table 2 – VLN-CE val-unseen**

| Method | R2R NE↓ | OS | SR | SPL | RxR NE↓ | SR | SPL | nDTW |
|---|---|---|---|---|---|---|---|---|
| NaVILA | 5.22 | 62.5 | 54.0 | 49.0 | 6.77 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 4.98 | 64.2 | 56.9 | 51.9 | 6.22 | 52.9 | 46.0 | 61.9 |
| CorrectNav | 4.24 | 67.5 | 65.1 | 62.3 | 4.09 | 69.3 | 63.3 | 75.2 |
| MobileVLA-R1 | 4.05 | 69.7 | 68.3 | 65.2 | 3.92 | 71.5 | 66.8 | 76.1 |
| **MobileVLA-R1 2.0** | **3.86** | **71.2** | **69.8** | **66.9** | **3.71** | **73.1** | **68.5** | **77.6** |

**Table 3 – QUARD**: 0.95 / 0.92 / 0.77 / 0.72 / 0.66 / 0.56, 평균 0.76 (MobileVLA-R1 0.70, MoRE 0.60, QUART 0.44). 어려운 과제(Unload +0.12)에서 향상 폭이 크다.

**Table 4 – Go2**: Complex SR Workspace 0.94(전작 0.91), Corridor 0.91(0.86), Outdoor 0.98(0.96); NE는 Workspace 1.23 → 1.12, Corridor 1.23 → 1.11. 160 에피소드 중 실패 10건(Table 5).

**Table 6 – G1 (%)**

| Method | Tabletop Nav/Manip/Full | Shelf Nav/Manip/Full | Cluttered Nav/Manip/Full | Full Avg |
|---|---|---|---|---|
| NaVILA | 75.0/63.3/47.5 | 67.5/59.3/40.0 | 57.5/43.5/25.0 | 37.5 |
| MobileVLA-R1 | 80.0/71.9/57.5 | 72.5/65.5/47.5 | 65.0/53.8/35.0 | 46.7 |
| **2.0** | 85.0/79.4/67.5 | 77.5/74.2/57.5 | 70.0/64.3/45.0 | **56.7** |

**절제**
- Table 9: SFT+파싱 R2R SR 58.0 → SFT+decoder 63.1 → GRPO+파싱 68.3 → GRPO+decoder 69.8 (GRPO의 단독 기여가 더 큼).
- Table 8: decoder 조건 — 관측만 68.8, 추론만 69.3, 둘 다 69.8.
- Table 10: CoT 없음 64.0 → episode 65.5 / step 66.0 / nav 66.7 → multi-granularity 68.3.
- Table 11: 보상 구성 요소 모두 포함 시 68.3이 최고, 단일 보상 중에선 behavior 보상이 가장 강함(61.9).
- Table 12: decoder 구조 Mean Pooling+MLP 68.7 → Dual-Query Dual-Head 69.8.

## 8. Related Work 상의 위치

- NaVILA·StreamVLN·CorrectNav 같은 VLM 기반 VLN과, QUART·MoRE 같은 사족 보행 VLA를 하나의 추론형 프레임워크로 묶는다.
- ECoT/CoT-VLA/ACoT-VLA의 추론–행동 결합 흐름에서, 추론 hidden state를 직접 읽는 학습형 decoder와 GRPO 보상을 결합한 점이 차별점.

## 9. 강점

1. 전작 대비 변경점(파싱 → decoder)을 통제된 절제로 분명하게 분리.
2. 시뮬레이션(VLN-CE, QUARD)과 두 종류의 실물 로봇(Go2, G1)을 모두 평가하고 실패 유형·지연까지 보고.
3. 보상 구성, CoT granularity, decoder 구조 등 설계 선택에 대한 폭넓은 분석.
4. 코드 공개.

## 10. 약점 및 한계

1. 전작 대비 향상 폭이 VLN-CE에서 1.5–1.7pt로 작고, 분산·반복 실험 정보가 없다.
2. G1 조작은 고정 루틴이 수행하므로 "휴머노이드 모바일 조작" 성과는 주로 태스크 수준 선택 능력을 반영한다.
3. 8B 백본을 원격 GPU에서 돌려야 해 온보드 배포가 불가능하고 네트워크 의존성이 있다.
4. GRPO가 오프라인 데이터셋 기반이라 실제 closed-loop 보상을 활용하지 않는다.
5. Table 9의 "GRPO+파싱" 값(68.3/65.2)이 전작 MobileVLA-R1 수치와 동일해, 전작 재현인지 새 실험인지 본문만으로는 구분이 어렵다.

## 11. 재현 및 확장 아이디어

- 시뮬레이터 상호작용 기반 온라인 GRPO/PPO로 closed-loop 보상 활용.
- α를 이산 primitive 대신 연속 조작 목표(예: 말단 자세)로 확장해 진짜 end-to-end 모바일 조작 평가.
- 경량 백본·양자화로 Jetson 온보드 추론.
- CoT 생성 교사 모델(Gemini 외)에 대한 민감도 분석 확대.

## 12. 총평

MobileVLA-R1 2.0은 추론형 이동 로봇 VLA에서 "텍스트 파싱"을 "추론 hidden state 기반 학습형 decoder"로 바꾸고 GRPO와 결합해, VLN-CE·QUARD·실물 Go2/G1 전반에서 일관된 향상을 보였다. 향상 폭은 크지 않지만 실험 범위와 절제가 충실한 저널 확장판이다.

**한 문장 요약**: NaVILA 기반 8B VLA에 다중 granularity CoT SFT, 추론 조건부 action decoder, 오프라인 GRPO를 결합해 R2R-CE SR 69.8 / RxR-CE SR 73.1, QUARD 0.76, G1 full-task 56.7%를 달성한 이동 로봇 추론 VLA.

<!-- VERIFIED: pdf -->
