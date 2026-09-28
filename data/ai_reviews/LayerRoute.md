# LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies

> **한 줄 요약**: 행동 모듈의 각 층이 현재 행동 상태를 질의로 삼아 여러 깊이의 VLM 은닉 상태를 가중 혼합해 읽고(Layer Mixture Router), 이전 행동 층의 상태를 토큰 단위로 다시 읽는(Action-State Reread) 표현 접근 인터페이스. π0.5에 붙여 LIBERO 96.9 → 98.2(Long 92.4 → 96.0), StarVLA-π에 붙여 95.7 → 98.0(Long 88.4 → 95.6).

- **arXiv**: 2609.06079 (v1 2026-09-05, v2 2026-09-17, cs.AI)
- **소속**: 칭화대, MBZUAI, NTU, 저장대, Rutgers
- **트래커 헤드라인**: π0.5+LayerRoute (StarVLA-π 변형은 별도 블록)

---

## 1. 배경 및 동기

VLM의 은닉 표현은 깊이에 따라 국소 질감·기하(얕은 층)에서 추상적 언어 정렬 의미(깊은 층)로 변한다. 조작 과제는 파지 위치·정렬처럼 세밀한 시각 단서가 필요한 단계와 지시 접지·하위 목표 선택처럼 의미 단서가 필요한 단계가 섞여 있다. 그런데 기존 VLA는 각 행동 층을 미리 정해진 VLM 층(최종층 readout, 고정 층 매핑, prefix/suffix 동일층 결합)에만 연결한다. 또한 행동 모듈 내부에서도 이전 층 상태는 residual로만 간접 전달되어 후반 층이 초기 중간 상태를 직접 재참조할 수 없다.

## 2. 문제 정의

VLM 은닉 상태 {V_n}을 캐시해 두고, 행동 층 s에서의 컨텍스트를 정적 readout Φ_s^static({V_n}) 대신 현재 행동 상태에 의존하는 Φ_s({V_n}, a^(s))로 바꾸는 readout 문제로 정식화한다. 동시에 행동 모듈 내부의 이전 상태 {A_0, A_j1, ..., A_cur}에 대한 명시적 재접근을 허용한다.

## 3. 방법

- **Action-Conditioned Layer Mixture Router**: 현재 행동 상태와 각 후보 VLM 층을 유효 토큰에 대해 풀링 → 층별 LayerNorm + 선형 사상으로 256차원 라우터 공간에 투영 → 스케일드 내적 softmax로 깊이 가중치 α_n^(s) 계산 → 같은 토큰 위치의 VLM 상태를 Σ α_n V_n으로 혼합해 cross-attention의 key/value로 사용. 샘플별·층별 하나의 분포(토큰 공유).
- **Action-State Reread**: 층별 학습 질의 q_A가 각 행동 토큰 위치에서 초기·보존된 이전·현재 상태를 RMSNorm 후 내적 softmax로 가중 혼합해 FFN 입력으로 전달. 질의는 공유지만 토큰별 가중치는 달라짐.
- **StarVLA-π 구현**: Qwen3-VL 층 [5,11,17,23,29,35] 6개 상태를 캐시, 36층 DiT의 18개 cross-attention 층 모두에 라우터, 36개 블록 모두에 reread. 파라미터 +0.31%.
- **π0.5 구현**: 18개 action expert 층마다 비공유 8-head action-to-VLM cross-attention 어댑터를 native joint attention 앞에 두고 잔차 업데이트, reread는 층 {2,5,8,11,14,17} 이후 상태를 보존. 파라미터 +3.87%.

## 4. 데이터

- LIBERO 4 suite.
- SimplerEnv: Bridge와 RT-1의 LeRobot 변환본을 동일 확률로 혼합해 학습, WidowX VM과 Google Robot VM/VA로 평가.
- RoboCasa-GR1: 24개 데이터셋 혼합, 24개 과제.
- 실물: Franka 팔, 과제당 50개 시연(큐브 PnP, 지우개로 보드 닦기, 빗자루로 테이블 청소).

## 5. 구현 세부

- 8×H200, 기기당 배치 32(전역 256), AdamW, weight decay 1e-8, 5K warmup.
- StarVLA-π: 베이스 lr 2.5e-5, 행동/라우팅 lr 1e-4, cosine → 1e-7. π0.5: lr 5e-5 고정.
- 학습 step: LIBERO 30K / SimplerEnv 40K / RoboCasa-GR1 100K, 종료 체크포인트로 평가.
- VLM, 행동 모듈, 라우터, reread를 모두 함께 미세조정.
- 지연 증가: StarVLA-π +10.03%, π0.5 +25.04%.

## 6. 실험 설정

- LIBERO: 과제당 50회(총 2,000 에피소드), 시드 7.
- SimplerEnv: StarVLA 공식 프로토콜대로 전체 평가기를 5회 반복해 평균, WidowX 과제당 24 에피소드/반복.
- RoboCasa-GR1: 과제당 50 롤아웃(1,200/모델).
- 비교는 각 백본의 원래 정적 인터페이스와 동일 프로토콜로 재평가한 쌍 비교.
- 절제: 구성요소(Table 6), 라우팅 정책(Table 7), VLM 백본(Table 5), 라우팅 세분도(Supp. Table 10), 라우팅 진단(Figure 2–5).

## 7. 주요 결과

**Table 2 – LIBERO (%)**

| 모델 | Spatial | Object | Goal | Long | Avg4 |
|---|---|---|---|---|---|
| StarVLA-π | 98.8 | 99.6 | 95.8 | 88.4 | 95.7 |
| StarVLA-π+LayerRoute | 98.6 | 99.2 | 98.4 | 95.6 | 98.0 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **π0.5+LayerRoute** | 99.0 | 98.4 | 99.2 | 96.0 | **98.2** |

**Table 3 – SimplerEnv (%)**: π0.5 VM/VA/GR Avg/WidowX 73.7/67.5/70.6/57.1 → +LayerRoute 76.1/69.3/72.7/62.7. StarVLA-π 75.1/69.0/72.1/60.8 → 76.8/70.6/73.7/63.3.
WidowX 과제별(π0.5+LR, Supp. Table 15): spoon 59.2, carrot 68.3, stack 54.2, eggplant 69.2.

**Table 4 – RoboCasa-GR1 24과제 macro avg**: π0.5 37.0 → 42.5, StarVLA-π 43.9 → 45.1.

**Table 5 – VLM 백본 민감도(Avg4)**: Qwen3-VL 95.7→98.0, Qwen2.5-VL 95.0→96.2, MiMo-Embodied 94.9→98.2, Cosmos-Reason2-2B 95.4→97.4.

**Table 6 – 구성요소**: π0.5 기준 router only 97.3, reread only 95.5, 둘 다 98.2. StarVLA-π는 96.4 / 95.9 / 98.0.

**Table 7 – 라우팅 정책(StarVLA-π, Avg4/Long)**: uniform 94.2/92.2, forced deep 96.4/94.2, action-independent query 97.2/94.8, current-state route 98.0/95.6.

**라우팅 분석**: 정규화 엔트로피 1.000 → 0.805(20K step), 초기 행동 층은 얕은 VLM 층, 후반 층은 중간·깊은 층 선호(MDS 0.205). Spatial은 얕은 층, Object는 중간 층, Goal은 깊은 층, Long은 가장 넓은 혼합.

**Table 8 – 실물(Franka, 과제당 50회)**: π0.5 72/54/26 → +LayerRoute 80/58/32.

## 8. Related Work 상의 위치

OTTER, CogVLA, VLA-Cache가 시각 토큰 선택/재사용을, DeeR-VLA·MoLe-VLA·Mixture-of-Depths가 계산 깊이를, DenseFormer·Attention Residuals·Hyper-Connections가 백본 내부 층간 결합을 다룬다면, LayerRoute는 **VLM–행동 경계에서 어떤 깊이의 표현을 읽을지**를 행동 상태에 조건화해 결정한다는 점이 새롭다. π0/π0.5의 prefix-suffix 공동 어텐션, StarVLA의 고정 cross-attention 매핑을 일반화한 인터페이스로 볼 수 있다.

## 9. 강점

1. 두 개의 이질적 VLA(교차 어텐션 DiT와 joint attention flow expert)와 네 개 VLM 백본, 세 개 시뮬레이션 벤치마크 + 실물에서 일관된 이득.
2. 원 백본을 동일 프로토콜로 재평가한 쌍 비교이며, SimplerEnv는 5회 반복 평균으로 분산을 줄임.
3. uniform/forced-depth/action-independent 대조군으로 "행동 상태 조건화" 자체의 기여를 분리.
4. 라우팅 분포가 suite 성격(공간 vs 목표)과 일치하는 해석 가능한 패턴을 보여줌.
5. 파라미터 오버헤드가 작다(StarVLA-π 0.31%).

## 10. 약점 및 한계

1. LIBERO는 이미 포화 영역이라 Avg4 차이(1.3–2.3pp)는 작고, 학습 시드 반복이 없다(SimplerEnv 반복은 같은 체크포인트의 평가 반복).
2. π0.5 경로는 파라미터 +3.87%, 지연 +25%로 결코 가볍지 않다.
3. π0.5 단독에서 reread only는 오히려 하락(96.9 → 95.5)해, 두 모듈의 상호작용이 충분히 설명되지 않는다.
4. RoboCasa-GR1 절대 성능(42.5%)은 다른 GR1 보고치보다 낮은 편이며 재평가 프로토콜 차이 여부가 불분명.
5. 실물은 단일 팔 3과제, 과제당 50회로 제한적이고 π0.5 경로만 평가.

## 11. 재현 및 확장 아이디어

- 토큰별(공간적) 깊이 라우팅으로 확장해 세밀한 영역과 의미 영역을 다른 깊이로 읽게 하기.
- 라우터를 사전학습 단계부터 넣어 대규모 교차 임바디먼트 데이터에서의 효과 측정.
- π0.5 경로의 지연을 줄이기 위한 캐시 층 수 감소(6 vs 18 층 절제에서 6층이 더 좋았음) 및 어댑터 경량화.
- LIBERO-Plus, RoboTwin 2.0 등 강건성·양팔 벤치마크로 검증.

## 12. 총평

LayerRoute는 "행동 계산이 VLM의 어느 깊이를 읽어야 하는가"라는 그동안 암묵적으로 고정되어 있던 설계 축을 명시적 학습 대상으로 만든 논문이다. 여러 백본과 벤치마크에서 일관되고 통제된 이득, 해석 가능한 라우팅 패턴이 설득력을 준다. 다만 포화된 LIBERO에서의 작은 차이, π0.5 경로의 적지 않은 지연 비용, 단일 시드 학습은 감안해야 한다.

**한 문장 요약**: 행동 상태로 조건화된 VLM 깊이 혼합 라우팅과 행동 상태 재참조를 π0.5·StarVLA-π에 붙여 LIBERO 98.2/98.0, SimplerEnv WidowX 62.7/63.3, RoboCasa-GR1 42.5/45.1로 일관되게 개선한 VLA 표현 접근 인터페이스.

<!-- VERIFIED: pdf -->
