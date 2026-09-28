# PhaseLoRA: Control-Regime-Conditioned Low-Rank Adaptation for Continuous-Action Vision-Language-Action Policies

> **한 줄 요약**: π0.5의 action expert에 붙는 LoRA를 "롤아웃 내내 고정된 업데이트"가 아니라, 경량 GRU 라우터가 매 action-chunk 질의마다 예측하는 두 개의 제어 서술자(fine-control tendency P, event/boundary intensity E)로 **왼쪽 인자 B를 조건화**해 업데이트 방향 자체가 시간에 따라 바뀌게 만든 PEFT 기법. 파라미터 수를 맞춘 High-rank LoRA 대비 LIBERO 평균 56.7 → 68.9(+12.2), 실물 Piper 4개 과제 평균 50.8 → 69.2(+18.4).

- **arXiv**: 2608.15285v1 (2026-08-15, cs.RO)
- **소속**: Tsinghua University
- **코드**: https://github.com/Grinffin/PhaseLoRA (공개 예정으로 표기)
- **백본**: π0.5 (openpi `pi05_base` 공개 체크포인트)

---

## 1. 배경 및 동기

사전학습된 VLA가 커지면서 다운스트림 적응은 점점 LoRA 같은 PEFT에 의존한다. 그런데 기존 PEFT는 과제·지시·예시 단위로 어댑터를 고르거나 생성할 뿐, **한 번 정해진 업데이트를 롤아웃 전체에 동일하게 적용**한다. 조작 롤아웃은 접근 → 접촉 전이 → 파지 → 운반 → 정밀 배치처럼 성격이 다른 제어 구간(control regime)을 지나가며, 같은 과제·같은 지시 안에서도 사전학습 정책에 필요한 보정이 달라진다. 저자들은 이 "within-trajectory control heterogeneity"를 PEFT에서 빠져 있던 적응 축으로 규정한다.

## 2. 문제 정의

- 정책: a_{t:t+H-1} = f_θ(x_t), x_t는 관측·언어·상태.
- 표준 LoRA: W' = W + BA, 모든 질의에서 동일한 ΔW.
- 목표: 백본은 거의 고정하고, LoRA의 효율(rank ≤ r)은 유지하면서 ΔW가 **롤아웃 내 제어 국면에 따라 변하게** 하기. 수동 phase 라벨, 힘/촉각 센서, 전체 용량 expert 분기 없이.

## 3. 방법: PhaseLoRA 파라미터화

- action expert의 각 적응 레이어에서 ΔW_t = B_t A.
- B_t = B_0 + P̂_t·B_P + Ê_t·B_E + (P̂_t Ê_t)·B_PE. 오른쪽 인자 A는 공유(공통 입력 사영), 왼쪽 인자만 서술자 조건.
- 임의의 (P̂, Ê)에서 여전히 rank ≤ r이며, 단순 스칼라 재스케일이 아니라 **업데이트 방향**이 바뀐다.
- 서술자 쌍은 한 forward pass의 모든 PhaseLoRA 레이어가 공유하고, VLM 쪽 LoRA는 표준(시간 불변) 그대로 둔다.
- 초기화: B_P, B_E, B_PE = 0, (B_0, A)는 표준 LoRA 초기화 → 학습 시작 시점엔 일반 LoRA와 동일.
- 파라미터 비용: 레이어당 r·d_in + 4r·d_out (표준 대비 +3r·d_out) + 라우터.

## 4. 약지도 서술자와 라우터

- **Fine-control tendency**: m_t = ‖a^xyz‖ + λ_r‖a^rot‖ + λ_g|g_t − g_{t−1}| (λ_r=0.5, λ_g=0.1). 움직임이 작을수록 정밀 제어로 간주 → P_t = σ(−m̃_t).
- **Event/boundary intensity**: jerk j_t = ‖a_t − 2a_{t−1} + a_{t−2}‖ → E_t = σ(j̃_t).
- 궤적별 median/MAD 로 robust 정규화. 정확한 phase 라벨이 아니라 "약한 프록시"임을 명시.
- **라우터**: VLM prefix 임베딩의 masked mean pooling(2048-d) + 최근 6개 chunk × 첫 5 실행 action = 30 step 이력(스텝당 23-d: action 7, 1차 차분 7, 2차 차분 7, gripper 상태·변화) → 128-d 사영 → 1-layer GRU(128) → 두 개의 sigmoid 헤드.
- 손실: L = L_policy + 0.5·Huber(P̂,P; δ=0.1) + 0.2·BCE(Ê,E) + 1e-5 Σ‖B_P‖²+‖B_E‖²+‖B_PE‖².

## 5. 구현 세부

- π0.5: VLM gemma_2b_lora, action expert gemma_300m_lora, 224×224, LIBERO에서 base + left wrist 두 뷰(세 번째 슬롯은 0).
- action dim 32, LIBERO H=10 중 5 step 실행 후 재계획; 실물은 H=50 중 10 step 실행.
- LoRA rank: VLM 48 / action expert 96 (High-rank 기준선은 80 / 160 으로 파라미터 매칭).
- AdamW, lr 5e-5(10k warmup 후 상수), batch 32, 30k step, bf16, 단일 RTX 5090 기준 suite당 약 24 GPU-hour, 보고된 LIBERO 실험 전체 약 2160 GPU-hour.

## 6. 실험 설정

- **LIBERO**: 4개 표준 suite 각각 suite별 정책을 따로 PEFT. 과제당 10 trial × 3 seed.
- **실물**: AgileX Piper 팔, 손목·외부 RGB 두 대, proprio/힘/촉각 입력 없음. 4개 과제(공 옮기기, 큐보이드 제거, 빨간 큐브 집기, 사과 놓기), 과제당 100 demo, 10 trial × 3 seed.
- **비교 기준선**: LoRA, 파라미터 매칭 High-rank LoRA, DoRA, LoRA-MoE, LoRA-SP — 모두 같은 레이어 집합·데이터·스케줄.

## 7. 주요 결과

**Table 1 – LIBERO (%, 3 seed 평균)**

| Method | #Params | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|---|
| LoRA | 2.70% | 60.7 | 41.3 | 32.7 | 18.7 | 38.3 |
| High-rank LoRA | 4.42% | 71.0 | 73.3 | 55.3 | 27.0 | 56.7 |
| DoRA | 4.42% | 72.7 | 74.0 | 57.3 | 29.7 | 58.4 |
| LoRA-MoE | 4.47% | 73.3 | 70.7 | 58.0 | 31.0 | 58.3 |
| LoRA-SP | 4.42% | 76.3 | 72.0 | 56.0 | 32.3 | 59.2 |
| **PhaseLoRA** | 4.41% | **85.3** | **88.0** | **64.0** | **38.3** | **68.9** |

표준 LoRA 대비 +30.6, High-rank LoRA 대비 +12.2.

**Table 2 – 실물 (%, 3 seed × 10 trial)**

| Method | Move | Remove | Pick | Put | Avg |
|---|---|---|---|---|---|
| High-rank LoRA | 60.0 | 53.3 | 63.3 | 26.7 | 50.8 |
| **PhaseLoRA** | 66.7 | 70.0 | 90.0 | 50.0 | **69.2** |

**Table 3 – LIBERO-Spatial 절제**: random descriptor 66.3, scalar-gated 73.7, 프록시 감독 제거 78.7, P만 80.7, E만 81.0, PE 상호작용 제거 82.3, 전체 85.3. → 임의의 시간 변조나 스칼라 게이팅으로는 이득이 재현되지 않는다.

**Table 4 – 업데이트 방향 분석**: 인접 스텝 Frobenius 코사인 거리 d_t와 |ΔÊ|의 Spearman ρ가 0.709–0.743으로 |ΔP̂|(0.439–0.502)보다 높다. 상위 10% d_t 스텝은 gripper 전이 ±5 스텝 이웃에서 2.52–3.10배 농축.

## 8. Related Work 상의 위치

- VLA PEFT(OpenVLA-OFT, LoRA-SP의 적응적 rank 할당) 계열과 달리 rank가 아니라 **업데이트 방향을 시간 조건화**.
- HINT·HyperLoRA·Conditional LoRA Generation 같은 조건부 어댑터는 과제/예시 단위, PhaseLoRA는 롤아웃 내부 질의 단위.
- BehaviorVLA·Mag-VLA 등 phase-conditioned decoder와 달리 디코더가 아니라 어댑터를 조건화하고, 명시적 phase 라벨·힘 센서를 쓰지 않는다.
- CO-RFT·VLA-RFT 같은 RL 후학습과는 직교(학습 신호 vs 어댑터 파라미터화).

## 9. 강점

1. **정직한 용량 통제**: 파라미터 매칭 High-rank LoRA, 그리고 random-descriptor/scalar-gating 대조군으로 "파라미터가 늘어서"와 "아무 시간 변조나 효과"라는 대안 설명을 직접 배제.
2. **라벨 불필요**: action/gripper 시퀀스 통계만으로 약지도 → 어떤 continuous-action 데이터셋에도 바로 적용 가능.
3. **해석 가능성**: d_t 피크가 파지·운반·배치 전이에 몰린다는 정량 분석(Table 4)으로 메커니즘을 뒷받침.
4. 3 seed 평균±표준편차 보고, 컴퓨트 예산까지 명시.

## 10. 약점 및 한계

1. **절대 성능이 낮다**: π0.5 전체/표준 파인튜닝이 LIBERO에서 90%대 후반을 보고하는 것과 비교하면 68.9는 크게 낮다. 30k step·suite별 학습·rank 제약 등 PEFT 설정 자체가 약한 영역이라, "PEFT끼리의 상대 비교" 이상으로 해석하기 어렵다. 전체 파인튜닝 기준선이 표에 없다.
2. 절제가 LIBERO-Spatial 한 suite에만 있고, 실물 비교는 High-rank LoRA 하나뿐.
3. 서술자가 action 통계에서 유도되므로, action 크기·jerk가 국면을 잘 반영하지 못하는 과제(예: 일정 속도의 접촉 작업)에서는 부적합할 수 있다고 저자 스스로 인정.
4. 실물은 단일 로봇·4개 단순 pick/place 과제, 과제당 30 trial.
5. 추론 시 라우터가 자기 과거 출력 이력을 입력받으므로 오류 누적 가능성에 대한 분석은 없다.

## 11. 재현 및 확장 아이디어

- 동일 조건에서 **full fine-tuning 및 표준 π0.5 LIBERO 레시피**와의 비교를 추가해 PEFT 격차를 정량화.
- 서술자를 힘/촉각 신호나 VLM이 예측한 subtask 경계로 확장해 contact-rich 과제에 적용.
- VLM 쪽 LoRA까지 서술자 조건화했을 때의 효과, 서술자 차원을 2개 이상으로 늘리는 스케일링.
- 다른 백본(GR00T, OpenVLA-OFT 등 regression/diffusion head)으로 일반화 검증.

## 12. 총평

"LoRA 업데이트를 롤아웃 안에서 움직이게 하라"는 단순하지만 새 축을 제시하고, 파라미터 매칭 및 대조군 실험으로 그 효과가 용량이나 무작위 변조가 아니라 **제어 국면에 정렬된 방향 변화**에서 온다는 것을 설득력 있게 보인다. 다만 비교가 PEFT 기준선들끼리의 저성능 영역(LIBERO 평균 38–69)에서 이루어져, 실제 π0.5 파인튜닝 관행에서도 이 이득이 유지되는지는 열린 질문이다.

**한 문장 요약**: 두 개의 약지도 제어 서술자로 LoRA 왼쪽 인자를 조건화해 π0.5 PEFT에서 파라미터 매칭 LoRA 대비 LIBERO +12.2, 실물 +18.4를 얻은 깔끔한 연구 — 단, 전체 파인튜닝 대비 위치는 보여주지 않는다.

<!-- VERIFIED: pdf -->
