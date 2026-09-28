# Robo-Dopamine 2.0: History-Conditioned and OOD-Aware Process Reward Modeling for Robotic Manipulation

> **한 줄 요약**: Qwen3-VL-8B 기반 pairwise 진행도 보상 모델(GRM)에 (1) 롤아웃 이력/정렬된 성공 참조 패널 조건화와 (2) 양·강건·음 브랜치를 갖는 OOD-aware signed progress 공간, (3) Signed-Hop 커리큘럼을 더한 process reward model. 동결된 보상 모델로 OpenVLA-OFT를 GRPO로 RL 학습해 RoboTwin 2.0 3개 과제 평균 86.8%(300 epoch, sparse 1,000 epoch 85.8%), 실물 ConRFT 반복 삽입 71/80.

- **arXiv**: 2608.15680v1 (2026-08-16, cs.RO)
- **소속**: 북경대(멀티미디어 정보처리 국가중점실험실), EvoPhys AI, BUPT, Tencent, 인민대, 중산대
- **코드**: 논문에 명시 없음
- **백본**: 보상 모델 Qwen3-VL-8B / 정책 OpenVLA-OFT(시뮬), ConRFT(Octo-Small + consistency policy, 실물)

---

## 1. 배경 및 동기

사전학습 VLA를 RL로 다듬으려면 조밀하고 신뢰할 수 있는 보상이 필요하다. sparse 성공 보상은 탐색이 비효율적이고, 수작업 dense 보상은 과제 특화·취약하다. 기존 로봇 보상 모델(Robo-Dopamine 1의 GRM 등)은 정적인 전/후 관측 쌍만 보므로 (a) 접촉 이벤트·반복 국면·순서 바뀐 서브골 때문에 같은 모습이 다른 진행도를 뜻하는 **temporal aliasing**, (b) 가림/방해물처럼 진행을 유지하는 변화와 실제 실패(파지 실패, 잘못된 물체)를 구분하지 못하는 **OOD 실패 의미론** 문제를 가진다.

## 2. 문제 정의

- pairwise 인터페이스: GRM이 두 상태 (s_i, s_j)와 과제 지시를 받아 **부호 있는 상대 진행도**를 예측.
- 목표: 이 인터페이스를 유지하면서 (i) 실행 맥락(이력)을 조건으로 넣고, (ii) 강건 변형과 실패를 통합 진행도 공간에 배치해, 오프라인 순서 일관성과 다운스트림 RL 성능을 모두 높이기.

## 3. 방법

- **History-conditioned pairwise reward**: 표준 질의는 같은 에피소드의 전문가 이력, 합성 OOD 질의는 phase 정렬된 성공 참조 패널(편집된 끝점은 그대로 유지), 온라인 질의는 관측된 롤아웃 prefix로 패널(예: 3×3)을 구성해 입력.
- **OOD-aware signed progress space**: positive(정상 진행), robustness-preserving(진행 유지 변형), negative(실패) 브랜치로 상태를 조직. 실패·복구까지 하나의 pairwise 학습 인터페이스로 표현. 음·강건 상태는 공유 source index를 통해 성공 궤적과 연결.
- **Signed-Hop Curriculum**: 먼저 큰 Hop 쌍으로 전역 순서·실패/복구 기하를 배우고, 이후 작은/0 Hop 쌍으로 국소 보정. 이 단계에서 큰 Hop 쌍을 일정 비율(25%) replay.
- **RL 보상 형태**: r̃_t = r_t + γ_RL Φ̂_{t+1} − Φ̂_t (potential-based shaping), 정책 최적화 중 GRM은 동결.

## 4. 데이터와 OOD 궤적

- RoboTwin 계열 시뮬레이션과 AgiBot World 실물 궤적 기반.
- 성공 궤적의 특정 프레임을 편집해 음(실패)·강건(가림·방해물) 상태를 만들고, 원본 상태·편집 관측·카메라 사영·시간 맥락의 provenance를 기록.
- 5개 평가 패밀리: in-distribution, temporal-memory, OOD-negative, OOD-robust, OOD-temporal. 지표는 trajectory-level visual order consistency(VOC).

## 5. 구현 세부

- 보상 모델: Qwen3-VL-8B, 400K signed pair(200K × 2 단계) + QA200K 보조 혼합, 8×H100에서 약 20시간.
- 시뮬 RL: 과제별 OpenVLA-OFT SFT 체크포인트, RLinf의 GRPO, 병렬 환경 128, 그룹 8, γ=1, LoRA BF16 + FSDP, head 카메라 RGB + 14-D proprio → 25-step chunk. 과제 horizon 200/150/400, lr 1e-4/2e-4/2e-4.
- 실물 RL: 듀얼 Franka, ZED 카메라, ConRFT(동결 Octo-Small + consistency policy), 단일 RTX 4090.

## 6. 실험 설정

- 오프라인: 5개 패밀리 VOC, 60 에피소드 과제 완료 분류 정확도(범용 VLM과 비교), 패널·OOD·QA 절제, Signed-Hop replay 비율.
- 시뮬 RL: RoboTwin 2.0의 place_empty_cup, place_container_plate, handover_block에서 2×2 요인 설계(C00 통제, C01 이력, C10 OOD, C11 둘 다) vs RLinf sparse.
- 실물 RL: 네 구멍 블록을 페그에 K=4회 연속 삽입하는 Repeated Insert Square.

## 7. 주요 결과

**Table 1 – VOC**: Robo-Dopamine 2.0 패널 입력 평균 0.986 (ID .991 / Temp .994 / Neg .994 / Rob .958 / OOD-T .991). 같은 체크포인트 정적 입력 0.967, GRM-8B-Pro 정적 0.958. OOD-robust가 0.906 → 0.958로 가장 크게 개선.

**과제 완료 분류(Fig. 2, 60 에피소드)**: 96.1% vs GPT-5.5 81.1%, Claude-Opus-4.8 79.4%, Gemini-3-Pro 77.2%, Qwen3.5-397B 76.1%, RoboBrain 2.0 61.7%.

**Signed-Hop**: 400K 예산에서 25% replay 0.9872 vs 동일 샘플 풀 셔플 0.9858, replay 없음 0.9866.

**Table 3 – RoboTwin 2.0 RL (%)**

| 조건 | Cup | Plate | Hand. | Avg |
|---|---|---|---|---|
| RLinf Sparse (1,000 ep) | 94.2 | 95.0 | 68.2 | 85.8 |
| C00 통제 | 77.5 | 82.1 | 57.0 | 72.2 |
| C01 이력 | 79.5 | 88.6 | 57.1 | 75.1 |
| C10 OOD | 85.2 | 89.5 | 67.1 | 80.6 |
| **C11 Full** | **96.2** | 94.6 | **69.7** | **86.8** |

**Table 4 – 실물 반복 삽입**: C11 71/80 삽입, 15/20 전체 성공 vs Event Sparse 48/80, 8/20 (200 vs 300 온라인 에피소드). 이력 패널이 가장 큰 기여(C00→C01, C10→C11 각각 +21 삽입).

## 8. Related Work 상의 위치

- VLM 기반 보상(RL-VLM-F, VICtoR 등)과 Robo-Dopamine 1의 GRM 계열을 잇되, 끝점 중심·성공 중심 설계를 **시간 맥락 + 부호 있는 OOD 감독**으로 확장.
- 순환 RL·이력 인지 정책처럼 메모리를 정책에 넣는 대신 **보상 층**에 넣는다는 점이 차별점.
- 정책 쪽은 RLinf/GRPO, ConRFT 등 기존 RL 파이프라인을 그대로 사용하며, 보상만 교체.

## 9. 강점

1. **통제된 요인 설계**: 정책 초기화·환경·옵티마이저를 고정한 2×2(OOD × 이력) 설계로 각 요소의 RL 기여를 분리.
2. **matched-pool 대조군**: 커리큘럼 효과를 데이터 구성과 분리하는 셔플 대조군.
3. **실물 검증**: 시각적으로 같은 상태가 반복되는 과제를 골라 이력의 필요성을 직접 보여 줌.
4. 300 epoch로 sparse 1,000 epoch 성능을 넘는 샘플 효율.

## 10. 약점 및 한계

1. **정책 기여가 아닌 보상 기여 논문**: 정책은 기존 OpenVLA-OFT/ConRFT이고, RoboTwin 평가도 3개 과제·과제별 정책이라 VLA 리더보드 수치와 직접 비교할 수 없다.
2. place_container_plate에서는 sparse 기준선(95.0)이 C11(94.6)보다 높고, 평균 차이(86.8 vs 85.8)도 작다. 시드 반복이 없다("각 조건 1회 학습").
3. VOC 수치들이 0.95–0.99 영역에서 소수점 셋째 자리 차이를 다투므로(예: 0.9872 vs 0.9858) 통계적 유의성이 불분명.
4. 과제 완료 분류의 범용 VLM 비교는 60 에피소드 규모이며 프롬프트 조건이 모델마다 공정한지 알기 어렵다.
5. 실물은 단일 과제, 20회 평가.

## 11. 재현 및 확장 아이디어

- RoboTwin 2.0 전체 50개 과제, 다중 시드로 RL 이득 검증.
- flow-matching VLA(π0/π0.5)나 diffusion 정책 RL에 동일 보상을 적용해 정책 계열 의존성 확인.
- 패널 구성 비용(시각 토큰 예산)과 추론 지연이 온라인 RL 처리량에 미치는 영향 분석.
- 음 브랜치 합성을 실패 롤아웃 자동 수집과 결합한 self-improving 루프.

## 12. 총평

Robo-Dopamine 2.0은 "보상 모델도 이력을 봐야 한다"는 명확한 가설을 통제된 오프라인·RL 실험으로 뒷받침한 process reward 연구다. 이력 패널이 반복 국면 과제에서 큰 차이를 만든다는 점은 설득력 있지만, 산출물의 중심은 정책이 아니라 보상 모델이며 정책 성능 비교는 소규모 과제 집합에 한정된다.

**한 문장 요약**: 이력 조건 + 부호 있는 OOD 진행도 공간을 갖춘 8B 보상 모델로 OpenVLA-OFT GRPO를 RoboTwin 2.0 3과제 평균 86.8%, 실물 ConRFT 반복 삽입 71/80까지 끌어올린 보상 설계 논문.

<!-- VERIFIED: pdf -->
