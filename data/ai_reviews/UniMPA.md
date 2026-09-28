# UniMPA — A Unified Memory-Prediction-Action Model via Action-Grounded Transition Modeling

**arXiv**: 2609.11875 · **기관**: Harbin Institute of Technology (Shenzhen), Nanyang Technological University (S-Lab) · **날짜**: 2026-09-10

## 1. 한 줄 요약
π0.5에 World Expert(미래 잠재·선택적 픽셀 예측으로 감독되는 전이 토큰), 양방향 Visual-Action / Action-Visual 메모리 뱅크, 검색된 행동 프로토타입으로 흐름 소스를 이동시키는 Prototype-Biased Flow를 결합한 3-시스템 VLA로, π0.5 학습 에폭의 25~50%만으로 LIBERO 98.6, LIBERO-Plus 85.3, RoboTwin 2.0 Hard(11과제) 58.2를 달성.

## 2. 문제 설정
- 관측→행동 학습의 "전이 실현 가능성 격차(transition realizability gap)"를 세 가지로 분해:
  1. **전이 모호성**: 비슷한 관측이 서로 다른 조작 위상(예: 접기 시작/계속/재파지/놓기)에 대응.
  2. **예측–실행 불일치**: 그럴듯한 미래 이미지가 물리적으로 실현 가능한 행동을 보장하지 않음.
  3. **경험–실현 불일치**: 과거에 실행 가능했던 행동 패턴도 현재 장면 기하에서는 조정이 필요.
- 기존 예측 기반(WAM)과 메모리 기반(MemoryVLA) 방법은 이를 분리된 보조 장치로 다뤘다는 문제 제기.

## 3. 핵심 아이디어
- **anticipate–ground–refine**: 예측이 의도를 해소하고 검색 질의를 만들고, 메모리가 실행 가능한 시각–행동 증거로 이를 접지하며, 검색된 경험이 행동 사전(prior)이 되어 현재 문맥에서 정제된다.
- **Persistent-Selective Future Prediction**: V-JEPA 2 잠재 미래를 지속적으로 감독하고, 잠재 변화 점수나 이동/회전/그리퍼 변화 지표로 켜지는 Trigger Gate가 전이 결정적 순간에만 픽셀 디코딩을 활성화.
- **양방향 메모리**: 시각 키로 행동 인지 전이 경험을, 행동 이력 키로 시각적으로 접지된 행동 프로토타입을 검색.
- **Prototype-Biased Flow**: 소스를 N(λ_p·P_a,t, I)로 이동해 역사적으로 지지되는 행동 다양체 근처에서 흐름 정제 시작.

## 4. 아키텍처
- 백본: π0.5(PaliGemma-2B VLM + 311M Gemma 액션 전문가).
- World Expert: 액션 전문가와 같은 18층 트랜스포머(hidden 1024, 8 heads, FFN 4096), 0 초기화 전이 쿼리 16개. VLM·World·Action 세 스트림이 블록 마스크로 결합된 교차 어텐션.
- 잠재 예측 대상: 뷰당 16 토큰·1408차원의 동결 V-JEPA 2 특징. 픽셀 예측: 64 토큰, 4층 디코더로 256×256 복원(학습 전용).
- 메모리: 512차원, 2층 stateful Mamba(시각·행동 시간 값), 8-head 교차 모달 융합, 코사인 유사도 τ=0.1, 최대 인과 이력 32, 조대→세밀 시간 검색.
- 추론 시 잠재/픽셀 헤드와 시각 검색 경로는 제거, World Expert와 Action-Visual 메모리는 유지. 10 Euler step.

## 5. 학습 목표
- Stage 1(메모리): 미래 이미지/행동 재구성 L_rec + 검색 시뮬레이션 L_ret(소프트 검색된 값이 올바른 미래를 예측하도록) + 교차 모달 페어링(가중 0.5).
- Stage 2(정책): L_UniMPA = L_act + λ_lat·L_lat + λ_pix·L_pix (LIBERO 1.0, 기타 0.01), L_act는 프로토타입 편향 소스 기반 flow matching.

## 6. 학습 절차
- Stage 1: 10 에폭, AdamW lr 1e-4, truncated BPTT 8.
- Stage 2: 메모리 동결, 정책 미세조정 20~30k step, lr 5e-5(VLABench 5e-6), 워밍업 5~10k, bf16, 4~16× NVIDIA Pro6000.
- 학습 예산: LIBERO는 π0.5 에폭의 25%, RoboTwin·VLABench는 50%.

## 7. LIBERO / LIBERO-Plus (Table II, III, V)
| 방법 | Spatial | Object | Goal | Long | Avg | LIBERO-Plus Avg |
|---|---|---|---|---|---|---|
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 73.6 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | 71.4 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 | 59.0 |
| MemoryVLA | – | – | – | – | 96.5(Origin) | 70.2 |
| **UniMPA** | 99.6 | 99.8 | 98.8 | 96.0 | **98.6** | **85.3** |
- LIBERO-Plus 섭동별: Camera 75.7, Robot 73.0, Language 84.6, Light 96.8, Background 94.5, Noise 90.9, Layout 87.3, 원본 대비 하락 13.3(최소).
- 스위트별 LIBERO-Plus: Spatial 91.0, Object 89.7, Goal 84.3, Long 76.5.

## 8. RoboTwin 2.0 Hard / VLABench (Table IV, VI, XXI)
- Clean 학습 → Randomized(Hard) 제로샷, 11과제 평균 58.2(π0.5 대비 +18.5, HALO 대비 +25.7), 11개 중 8개 1위. Move Playingcard Away 46 vs 14, Place Object Scale 38 vs 20, Place Phone Stand 37 vs 7.
- 부록 44과제 확장 평균 35.84%.
- VLABench 6과제 평균 44.0(π0.5 대비 +4.3): Painting 38.0, Book 59.1, Drink 51.0, Tube 41.7, Condiment 52.0, Flower 22.0.

## 9. 실제 로봇 (Table VII, VIII)
- **GALAXEA R1 Lite**(21과제·7 suite·과제당 25회): TSR/CSR 77.7/86.3 vs π0.5 65.3/75.6, OpenVLA-OFT 45.7/58.1. 장기·복구 suite G에서 72.0/83.3(π0.5 대비 +20.0/+14.9), 동적 suite F 70.7/79.2.
- **AgileX Cobot Magic**(7과제·25회): 평균 TSR 74.9 / CSR 86.4 vs π0.5 62.3/74.9 (+12.6/+11.5). Conveyor Interception에서 76.0 vs 40.0으로 격차 최대.

## 10. 어블레이션 (Tables IX–XIV, LIBERO Avg / AgileX RW TSR)
- 예측: latent만 95.3/64.6, pixel만 96.9/68.6, dense pixel 97.3/70.3, 제안 98.6/74.9.
- 트리거: random 96.0/65.1, action-only 97.0, latent-only 97.4.
- 메모리: 제거 94.5/61.1, Visual-Action 뱅크 제거 96.5, Action-Visual 뱅크 제거 97.1.
- 시간 검색: 시간 모델링 제거 95.9, 고정 창 96.7.
- 메모리 사전학습: 현재 상태 재구성 96.1/62.9.
- 행동 사전: 최근접 행동 복사 91.3/52.0(가장 큰 하락), Action Proposer 제거 96.2, 조건으로만 사용 97.0.

## 11. 강점과 한계
**강점**
- 예측·메모리·행동을 하나의 전이 인터페이스로 묶은 설계와, 모든 구성요소에 대한 시뮬레이션+실로봇 동시 어블레이션.
- 더 적은 학습 예산으로 LIBERO-Plus·RoboTwin Hard 등 OOD 설정에서 큰 폭 개선.
- 두 종류 양팔 로봇, 총 28과제의 대규모 실로봇 평가.

**한계**
- 3-스트림 구조 + 이중 메모리 + Mamba 인코더 + V-JEPA 2로 시스템 복잡도와 메모리 저장 비용이 크며, 추론 지연·처리량 수치가 보고되지 않았다.
- RoboTwin 헤드라인은 11과제 부분집합(Hard만)이며, 44과제 확장에서는 평균 35.84%로 떨어진다.
- 비교 기준선 다수가 타 논문 보고값이라 프로토콜 차이가 섞여 있고, 실로봇 기준선은 저자 재현.
- 메모리 뱅크가 과제별/벤치마크별 궤적으로 구축되어 새로운 환경에서의 검색 커버리지 문제가 남는다.

## 12. VLA-Tracker 관점 평가
π0.5를 확장한 새 구조를 2단계로 직접 학습한 자체 정책이므로 ACCEPTED, 액션 생성은 프로토타입 편향 소스를 쓰는 flow matching이라 `action_head_category: flow_matching`. LIBERO 4개 스위트·보고 평균 98.6과 LIBERO-Plus 섭동별·스위트별·평균 85.3을 `benchmarks.libero`에 넣었다. RoboTwin 2.0은 Hard 전용 11과제 부분집합이므로 표준 `robotwin_v2`에 넣지 않고 `robotwin_v2_hard_11task`(58.2)와 `robotwin_v2_hard_44task`(35.84)로 분리했다. Table IV의 "Place BF"는 부록 Table XXI 대조로 Place Burger Fries(40%)임을 확인했다. 실로봇은 GALAXEA(21과제)를 `real_world`, AgileX를 별도 블록으로 두었다.

<!-- VERIFIED: pdf -->
