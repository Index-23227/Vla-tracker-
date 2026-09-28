# 3DWay: Generalizing Robot Manipulation via 3D Consistent Waypoints

> **한 줄 요약**: NVILA VLM을 fine-tune해 두 시점 이미지에서 서로 일관된 2D waypoint를 텍스트로 예측하게 하고, 카메라 파라미터로 삼각측량해 3D TCP waypoint를 얻는 중간 표현 방법. 이를 top-down 전략으로 바로 실행(3DWay-TD)하거나 π0 fine-tuning 시 상태처럼 주입(3DWay-augmented π0)한다. RLBench 10-task unseen 64.0%(π0.5 18.8%), VLABench 평균 37.7%(π0.5 31.3%), 실물 기본 과제 π0 21.7% → 65.8%.

- **arXiv**: 2609.08224v1 (2026-09-08, cs.RO)
- **소속**: Tsinghua University, ETH Zürich, UC Berkeley
- **코드**: https://github.com/ziqin-h/3DWay (공개 예정)
- **백본**: NVILA 2B/8B/15B (기본 15B), 통합 정책은 π0

---

## 1. 배경 및 동기

VLM의 세계 지식을 조작 정책에 활용하려고 2D 궤적(RT-Trajectory, HAMSTER 등)을 중간 표현으로 쓰는 연구가 많지만, 2D 궤적은 깊이 모호성이 있고, depth로 역투영한 2.5D 궤적은 depth 노이즈에 민감하며 자유 공간(물체 표면이 아닌 점)의 위치가 유일하게 정해지지 않는다. 저자들은 VLM의 출력 형식(텍스트 좌표)을 유지하면서도 명시적 3D 동작 의도를 표현하는 방법을 찾는다.

## 2. 문제 정의

- 다시점 이미지 I, 내부/외부 파라미터 K, T, 지시 ℓ로부터 3D waypoint W_3D를 생성하는 Φ를 학습.
- 직접 3D 좌표를 예측하는 대신, 각 시점의 2D 투영 W_2D를 예측(다시점 일관성 요구)한 뒤 삼각측량 Tri(Ŵ_2D, K, T)로 3D를 복원.

## 3. 방법

- **데이터 처리**: RLBench, DROID, RH20T(약 290 태스크, 134k 궤적)에서 TCP 궤적을 계산하고 RDP 알고리즘으로 단순화한 뒤 두 시점에 투영, 그리퍼 열기/닫기 토큰 포함. 개방 어휘 검출기 기반 자동 점수로 저품질 데이터 필터링.
- **VLM 학습**: 먼저 RoboPoint로 1 epoch 점 예측 능력을 강화하고, 처리된 데이터로 10 epoch full fine-tune. 좌표를 텍스트로 표현해 표준 언어 모델링 손실로 학습(별도 regression/action head 없음).
- **직접 실행(3DWay-TD)**: top-down 파지와 waypoint 기반 모션 계획/IK로 실행.
- **Adaptive waypoint-guided integration**: 현재 말단 위치와의 거리로 선택한 waypoint 윈도우(N=2)를 로봇 상태에 concat해 π0를 in-distribution fine-tune. 구조 변경 없음.

## 4. 구현 세부

- batch 256, cross-entropy, AdamW lr 1e-5, cosine; 8×A100에서 15B도 2일 이내.
- 카메라 2대만 사용, 기본 waypoint 생성기는 NVILA-15B.
- 기준선(OpenVLA-OFT, π0, π0.5)은 공식 프로토콜로 in-distribution fine-tune.

## 5. 평가 프로토콜

- **RLBench 10-task (회전 불변 과제 부분집합)**: seen/unseen(테스트 과제 시연 제외), 3 run × 25 rollout. **표준 RLBench 멀티태스크 프로토콜이 아님.**
- **VLABench**: Select Toy/Fruit/Painting/Poker/Mahjong 5과제(각 500 demo), ID/교차 범주/상식/의미 지시/미지 텍스처 5개 차원, 설정당 50 trial.
- **RLBench few-shot 통합**: 5과제 × 10 demo로 π0 fine-tune.
- **실물**: AgileX PIPER에서 기본 5과제(과제당 20 demo) + 시각 일반화·의미 서술 확장 과제, 과제당 24 rollout. KUKA iiwa14로 embodiment/카메라 변화 평가.

## 6. 실험 설계의 요점

3DWay-TD 실험은 waypoint 표현 자체의 실행 가능성과 일반화를, π0 통합 실험은 데이터가 적을 때 foundation VLA의 일반화를 얼마나 보완하는지를 본다. 저자들도 3DWay-TD는 회전 불변의 단순 과제에 한정된다고 명시한다.

## 7. 주요 결과

**Table 1 – RLBench 10-task / VLABench**

| Method | RLBench Seen | Unseen | VLABench ID | CC | CS | SI | UT | Avg |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 8.8 | – | 0.8 | 0.0 | 0.8 | 0.0 | 0.4 | 0.4 |
| π0 | 39.6 | 12.0 | 33.0 | 31.8 | 22.6 | 16.8 | 25.2 | 25.9 |
| π0.5 | 61.5 | 18.8 | 39.2 | 40.0 | 19.4 | 25.8 | 32.0 | 31.3 |
| **3DWay-TD** | **84.8** | **64.0** | 49.0 | 44.2 | 23.4 | 24.3 | 47.4 | **37.7** |

**Table 2 – RLBench 5과제 few-shot π0 통합**: 표준 fine-tune(10 demo) 17.6 → 3DWay-augmented zero-shot 생성기 38.1 → few-shot 생성기 46.1. 표준 fine-tune 30 demo 31.2, 50 demo 46.7로, 20% 데이터로 50-demo 기준선과 비슷한 수준.

**실물(본문)**: 기본 과제 평균 π0 21.7% → 3DWay-augmented π0 65.8%; 확장 과제에서 표준 π0는 6.7%·2.5%, 3DWay-augmented는 두 과제군 모두 50% 초과.

**Table 3 – 절제(RLBench unseen, 3DWay-TD)**: NVILA-15B 64.0, 8B 62.8, 2B 50.1; 카메라 외부 파라미터 작은/큰 변화 61.6/58.4; 다시점 일관성 없이 시점별 독립 학습 22.0.

**Table 4 – 실물 embodiment/카메라**: PIPER 2외부 43.9, KUKA 2외부 45.0, KUKA 손목+정면 45.5, 손목+좌측 44.7 — 로봇·카메라 구성에 강건.

## 8. Related Work 상의 위치

- HAMSTER, RT-Trajectory(2D 궤적), MOKA, A0(depth 역투영 2.5D)의 계보에서 다시점 일관성 + 삼각측량으로 3D를 얻는 방향.
- SpatialVLA, BridgeVLA, Evo-0처럼 VLA 내부에 3D를 주입하는 방식과 달리, VLM 출력 형식을 그대로 두는 명시적 중간 표현.

## 9. 강점

1. VLM의 텍스트 좌표 출력 패러다임을 유지해 사전학습 지식 손실을 줄이면서 3D 정보를 얻는 간단한 재정식화.
2. 다시점 일관성 제거 시 64.0 → 22.0으로 급락하는 절제가 핵심 아이디어를 강하게 뒷받침.
3. 카메라 외부 파라미터 변화, 다른 로봇·카메라 구성에서의 강건성 검증.
4. 적은 데이터에서 π0 일반화를 크게 개선하는 실용적 통합 방식.

## 10. 약점 및 한계

1. 병진 waypoint만 예측하고 회전을 모델링하지 않아, 3DWay-TD는 회전 불변 과제에 제한(저자 명시).
2. RLBench 평가는 자체 선정 10과제 부분집합이라 표준 RLBench 결과와 비교 불가.
3. 보정된 카메라 두 대를 전제하므로 가림이 많은 환경에서의 강건성이 불확실.
4. 실물 확장 과제의 3DWay-augmented 수치는 그림으로만 제시되고 본문에는 "50% 초과"로만 서술.
5. waypoint 생성(VLM 1회 추론)과 π0 실행의 결합이 느슨해, 실행 중 waypoint 재생성이나 폐루프 보정은 다루지 않는다.

## 11. 재현 및 확장 아이디어

- SE(3) 자세까지 포함한 waypoint 예측으로 회전 의존 과제 확장.
- 자기 보정(self-calibrating) 삼각측량으로 카메라 보정 의존성 완화.
- π0.5 등 다른 foundation VLA와의 통합, 실행 중 waypoint 재계획.
- 표준 RLBench 18과제·LIBERO 등 공인 벤치마크에서의 비교.

## 12. 총평

3DWay는 "다시점에서 일관된 2D waypoint를 텍스트로 예측하고 삼각측량한다"는 단순한 아이디어로 2D/2.5D 궤적 표현의 모호성을 해결하고, 직접 실행과 π0 통합 모두에서 뚜렷한 일반화 향상을 보였다. 회전 미모델링과 비표준 평가가 한계지만, 데이터가 부족한 배포 환경에서 foundation VLA를 보완하는 중간 표현으로 가치가 크다.

**한 문장 요약**: NVILA를 fine-tune해 다시점 일관 2D waypoint를 예측·삼각측량한 3D waypoint로 RLBench 10-task unseen 64.0%, VLABench 37.7%를 달성하고, π0에 주입해 실물 기본 과제 성공률을 21.7%에서 65.8%로 끌어올린 방법.

<!-- VERIFIED: pdf -->
