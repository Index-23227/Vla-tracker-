# Towards Zero-Shot Transfer Across Embodiments For Driving VLAs (BEV-Forcing)

> **한 줄 요약**: Qwen 3.5 2B를 LoRA로 driving VLA로 적응시키면서, 마지막 층 이미지 hidden state를 SimpleBEV teacher의 차량 occupancy map으로 감독하는 저용량 보조 head(**BEV-Forcing**)를 붙여 카메라 rig 간 공통 공간 인터페이스를 주입한다. 적은 rig로 학습할 때 zero-shot transfer가 크게 좋아지지만(KITScenes LongTail test MMS 4.62 → 5.15), 학습 rig 다양성이 늘면 이득이 줄어든다는 점을 정직하게 보고한다.

---

## 1. 배경 및 동기

- Driving VLA는 대부분 단일 데이터셋(단일 카메라 rig)으로 학습·평가되며, 보지 못한 데이터셋/카메라 구성으로의 **zero-shot 전이**는 거의 평가되지 않는다.
- 저자들은 이를 **semantic transfer**(같은 rig, 다른 장면)와 **geometric transfer**(다른 rig)로 구분한다. VLM 백본은 새 이미지의 내용을 의미적으로 이해하더라도, 3D 공간 이해는 소수의 rig에서 암묵적으로 학습되므로 전이되지 않는다.
- 로봇 조작에서는 다중 embodiment 사전학습이 이 문제를 완화했지만, 자율주행에서는 여러 공개 데이터셋을 합쳐도 in-dataset 성능이 항상 오르지는 않는다.
- 핵심 가설: 모델이 여러 rig의 멀티뷰 이미지로부터 **통일된 공간 표현(BEV)**을 학습하면 zero-shot 전이가 좋아진다.

## 2. 문제 설정

- 입력: 현재 시점 전방 3대 카메라 이미지, 과거 ego 궤적(4 Hz, 16 step), 고수준 ego intent, 자연어 프롬프트.
- 출력: 1 Hz 미래 waypoint를 **텍스트 좌표 쌍**으로 출력 → cubic spline으로 목표 샘플링 주기로 재샘플.
- 평가 축: (a) 학습에 포함된 rig에서의 in-dataset 성능(WOD-E2E), (b) 보지 못한 rig로의 zero-shot 전이(Physical AI 1k clip 검증셋, KITScenes LongTail test).

## 3. 핵심 아이디어 — BEV-Forcing

- Spatial Forcing(VGGT 특징으로 이미지 임베딩 정렬)에서 영감을 받았으나, 타깃을 3D foundation model 특징 대신 **전문 BEV 모델(SimpleBEV)이 출력한 차량 occupancy map**으로 바꾼다. 저자들은 BEV occupancy가 rig에 걸쳐 더 균일하고 task에 정렬된 인터페이스라고 가정한다.
- LiDAR나 GT 지도가 필요 없다 — SimpleBEV는 학습에 쓰이지 않은 데이터셋에서도 합리적인 occupancy map을 내므로 WOD-E2E처럼 카메라만 있는 데이터셋에도 적용 가능.
- 보조 head는 추론 시 제거 → **추론 비용 0**.

## 4. 아키텍처

- 백본: Qwen 3.5 2B (natively multimodal VLM). 전체 fine-tuning보다 **LoRA가 더 좋았음**(텍스트 행동 표현이 사전학습과 가까워 catastrophic forgetting 회피가 중요, "Actions as Language" 관찰과 일치).
- BEV head (Eq. 1–3):
  - 층 ℓ(백본 마지막 층)의 이미지 토큰 임베딩 f_I^(ℓ) ∈ R^{N_I×D}를 MLP로 d 차원에 투영
  - 무작위 초기화된 N_X×N_Y 개 BEV query가 투영된 이미지 임베딩에 cross-attention
  - Linear layer로 각 grid 위치의 occupancy logit 출력
- **저용량 설계가 핵심**: full transformer block 대신 cross-attention 한 층 수준으로 제한해, 공간 정보가 head가 아니라 이미지 임베딩 자체에 담기도록 강제.

## 5. 학습 목표

- L = L_NTP + λ·L_BEV
  - L_NTP: 텍스트 궤적(또는 VQA 답)의 next-token prediction
  - L_BEV: SimpleBEV occupancy target에 대한 binary cross-entropy, 양성 셀에 가중치 w+ 부여(클래스 불균형 보정)
- VQA 샘플(nuScenes-QA)에서도 이미지 hidden state를 BEV head에 통과시켜 L_BEV 계산.

## 6. 데이터 및 학습 세부사항

- Planning: WOD-E2E, NAVSIM, nuScenes, Physical AI(계산 제약으로 6천 clip 부분집합; 5천 학습/1천 검증, 학습 부분은 Fig. 3 비교용으로만 사용).
- VQA: nuScenes-QA (전체 서라운드 카메라, 단어 하나 정답).
- 데이터셋별 인터페이스를 하나의 contract로 통일해 조합 학습.
- LoRA rank 16, lr 1e-4 cosine decay, 1 epoch, global batch 64, A100 4장. 이미지 저해상도화 시 카메라 calibration도 함께 변환. Teacher occupancy map은 고해상도 입력으로 추출.

## 7. 주요 결과 — zero-shot 전이 (Table 1, KITScenes LongTail test)

| Method | MMS ↑ | L2 (mean) ↓ |
|---|---|---|
| UniAD | 3.24 | 10.90 |
| DMAD | 3.51 | 10.04 |
| Alpamayo 1.5 기반† | 4.31 | 3.17 |
| VLA & Kinematics 기반† | 4.31 | 3.57 |
| Gemini 3 Pro | 4.61 | 2.99 |
| Ours [W+N] | 4.62 | 2.63 |
| **Ours [W+N] (+BEV)** | **5.15** | **2.48** |

- WOD-E2E + NAVSIM 두 rig만으로 학습한 2B 모델이 Gemini 3 Pro, Alpamayo 1.5 기반 제출보다 높은 MMS.
- KITScenes 검증셋(내부 재현 MMS): WOD-E2E만 학습 시 3.90 → 4.64(+BEV)로 향상. 그러나 WOD-E2E+NAVSIM+nuScenes-QA에서는 BEV 없음 5.13 > BEV 있음 4.84로 **역전**.
- Physical AI 1k (Fig. 3): WOD-E2E 단독 학습 시 BEV로 ADE 10.1% 감소. 데이터셋을 추가할수록 이득이 baseline에 수렴.

## 8. 주요 결과 — in-dataset (Table 2, WOD-E2E test)

| Method | RFS ↑ | ADE ↓ |
|---|---|---|
| IRL-VLA | 7.890 | 2.823 |
| Poutine (SFT-only) | 7.909 | 2.941 |
| Poutine | 7.986 | 2.742 |
| NTR | 8.046 | 2.638 |
| DriveMA-2B | 8.075 | 2.662 |
| Ours [W] | 7.873 | 2.898 |
| Ours [W] (+BEV) | 7.902 | 2.891 |
| Ours [W+N+nSQA] | 7.939 | 2.833 |
| Ours [W+N+nSQA] (+BEV) | 7.902 | 2.938 |

- 단일 rig 학습에서는 BEV가 소폭 개선, 다중 rig 학습에서는 test에서 오히려 소폭 악화. 전체 수준은 유사 구조인 Poutine SFT-only와 비슷하며 RL을 쓰는 SOTA(DriveMA, NTR)에는 못 미친다.
- 검증셋(RFS 라벨): WOD-E2E 단독 FDE 5.946 → 5.831, RFS 7.961 → 7.971. 최선 구성 BEV 5.687 FDE / 8.119 RFS vs baseline 5.851 / 8.063.
- **언어 데이터의 중요성**: nuScenes-QA 정확도가 base Qwen 3.5 2B 29.2% → WOD-E2E 학습 11.1% → +NAVSIM 7.0%로 붕괴. nuScenes-QA(학습 샘플의 약 10%) 공동학습 시 FDE 6.019 → 5.851로 회복, QA 정확도 60.2%.

## 9. Ablation (Table 3, 4, Fig. 5)

- **BEV head 용량** (Table 3, W+N 소규모 학습): cross-attention only가 full transformer block보다 WOD-E2E(ADE 2.376 vs 2.441, FDE 5.806 vs 5.959)와 PhAI-1k(ADE 1.725 vs 1.805, FDE 5.073 vs 5.308) 모두에서 우수 → 저용량 head가 임베딩에 공간 정보를 더 밀어넣는다는 가설 지지.
- **BEV-Forcing vs Spatial Forcing** (Table 4, W+N+nSQA): WOD-E2E ADE/FDE 2.319/5.687 vs 2.341/5.766, PhAI-1k 1.783/5.192 vs 1.818/5.264 — BEV 타깃이 양쪽 rig에서 근소 우위.
- **강건화 기법과의 상호작용** (Fig. 5): 이미지 augmentation, calibration-as-input(zero-init α 스케일)은 BEV-Forcing과 함께할 때만 zero-shot 전이를 개선. 다만 향상폭은 "modest"하다고 저자 스스로 기술.

## 10. 관련 연구 대비 위치

- Spatial Forcing(로봇 조작, VGGT 특징 정렬)의 주행 버전이자 타깃 교체. VGGDrive, LaST-VLA, DynVLA 등 3D/동역학 특징 주입 계열 중 "카메라만, 추가 추론 비용 없음" 쪽에 위치.
- BEVDriver 등 BEV를 입력으로 넣는 방식과 달리 **감독 신호로만** 사용.
- Impromptu VLA, 123D 등 주행 데이터 통합 흐름과 맞물려, 기법 평가 시 **학습 rig 수를 변수로 두어야 한다**는 방법론적 주장을 편다.

## 11. 강점과 한계

**강점**
- 추론 비용 0, 라벨(지도/LiDAR) 불필요, 구현이 단순.
- 부정적 결과(데이터 다양성이 늘면 이득 소멸, 다중 rig test에서 RFS 하락)를 숨기지 않고 핵심 메시지로 제시.
- 언어 능력 붕괴와 VQA 공동학습 회복을 정량화.
- 코드 및 선택 clip 공개.

**한계**
- 절대 성능이 SOTA(DriveMA-2B 8.075, NTR 8.046 RFS)보다 낮고, 핵심 주장인 다중 rig 설정에서 BEV의 효과가 사라지거나 역전됨.
- KITScenes 비교 대상 일부(†)는 test set 제출명으로부터 추정한 설명이라 공정 비교가 불확실.
- Physical AI 결과가 대부분 Fig. 3(그래프)로만 제시되어 수치 검증이 제한적. KITScenes 검증 MMS는 내부 재현 지표.
- Teacher가 차량 occupancy만 제공 — 차선/주행 가능 영역 미포함(저자도 future work로 언급).
- 1 epoch, 단일 시드 수준으로 보이며 분산 보고 없음; 개선폭(예: RFS 0.03)은 노이즈 범위일 수 있음.
- 개루프 평가만 수행, 폐루프 주행 미검증.

## 12. VLA-Tracker 관점 평가

- **등록 판정: ACCEPTED** — Qwen 3.5 2B를 LoRA SFT로 직접 학습한 자체 driving VLA 정책이며 WOD-E2E test, KITScenes LongTail test에 정량 결과 보고.
- 표준 조작 벤치마크(LIBERO/CALVIN 등)는 없으므로 추적 대상 랭킹에는 들어가지 않고, 주행 전용 블록(kitscenes_longtail_zero_shot, wod_e2e_test_*)으로만 기록.
- 대표 수치: KITScenes LongTail test MMS 5.15 / L2 2.48 (W+N, +BEV), WOD-E2E test RFS 7.902 / ADE 2.891 (W, +BEV).
- 트래커 관점의 가치: "보조 공간 감독은 데이터 다양성과 부분적으로 대체 관계"라는 관찰은 조작 VLA의 Spatial Forcing류 기법 평가에도 시사점이 있다.

<!-- VERIFIED: pdf -->
