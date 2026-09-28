# WorldAgen: Unified State-Action Prediction with Test-Time World Model Training

> **한 줄 요약**: 하나의 Transformer 백본에 정책(agent) 헤드와 과제 무관 월드모델 헤드를 함께 두고, 배포 시 무작위 탐색 궤적으로 월드모델 손실만 LoRA로 업데이트하는 Test-Time Training(TTT)을 수행해 CALVIN 평균 길이 3.87 → 3.93, LIBERO-10 75.5% → 79.0%를 얻은 GR-1 계열 VLA.

- **arXiv**: 2609.08162v1 (2026-09-08, cs.AI)
- **소속**: Northwestern University
- **프로젝트**: https://worldagen.github.io

---

## 1. 배경 및 동기

GR-1, Seer, PAD 등 상태-행동 공동 예측 VLA는 미래 관측을 함께 예측함으로써 데이터 효율과 장면 이해를 높였지만, 모두 정적 데이터셋으로 사전학습된 뒤 배포 시에는 고정된다. 새로운 물체 배치, 조명, 물리 특성 같은 분포 변화가 오면 내부 동역학 표현을 적응시킬 수단이 없다. 저자들은 "배포 시점에 VLA가 환경 이해를 능동적으로 갱신할 수 있는가"를 핵심 질문으로 둔다.

## 2. 문제 정의

POMDP로 정식화하고 롤아웃을 궤적 단위 U_i = (G_i, O_i, S_i, A_i, A_p, O_p)로 나눈다. 정책 p_a(Â_i | G_i, O_i, S_i, h^a_i)는 지시문을 보고 행동 청크를, 월드모델 p_w(Ô_i | O_i, S_i, Â_i, h^w_i)는 지시문 없이 다음 관측 청크를 예측한다. 목표는 (1) 두 과제를 하나의 백본에서 정보 누출 없이 공동 학습하고, (2) 테스트 시 라벨 없는 탐색 데이터로 월드모델 쪽만 적응시켜 간접적으로 행동 예측을 개선하는 것이다.

## 3. 방법

- **공유 백본 + 두 헤드**: Qwen3 구조 Transformer(24층, 12헤드). 행동 자리표시자 A_p와 관측 자리표시자 O_p를 0으로 초기화해 삽입하고, 해당 위치의 출력을 각각 행동/이미지 디코더로 보낸다.
- **Mixed Unidirectional Attention Mask**: local mask는 현재 관측·상태·지시문·A_p가 같은 단위의 실제 행동 A_i를 보지 못하게 하고, global mask는 월드모델 헤드(O_p)가 지시문 G_i를 보지 못하게 해 과제 무관 동역학 모델을 만든다. 전체적으로 엄격한 인과 구조.
- **궤적 분할/청킹**: 관측 청크 길이 n, 행동 청크 길이 m을 분리하고, m개 구간에서 n개 관측을 균등 부분표본해 중복 프레임을 줄인다(Algorithm 1).
- **사전학습**: L = L_a + λ·L_o, teacher forcing.
- **2단계 추론**: 행동 먼저 예측해 A_p를 덮어쓰고, 이어서 다음 관측을 예측.
- **TTT**: (1) 새 환경에서 무작위 탐색 롤아웃 수집, 지시문은 "no-lang" 토큰으로 대체, (2) 관측 손실 L_o만으로 공유 백본을 LoRA 미세조정하고 정책 헤드는 동결.

## 4. 데이터

- CALVIN: 34개 과제, 4개 환경(A–D), 5개 연속 지시 평가. 논문 본문과 부록 어디에도 ABC→D인지 ABCD→D인지 명시되지 않는다(비교 기준선 수치는 ABC→D 문헌값과 일치하는 것으로 보이나 논문은 밝히지 않음).
- LIBERO: 부록에서 LIBERO-90으로 사전학습하고 LIBERO-10으로 평가한다고 설명. 부록 Table 11에서 Spatial/Object/Goal/10 네 suite 수치도 보고.
- TTT 데이터: CALVIN은 학습셋에 등장한 34개 지시문으로 60프레임 무작위 탐색 후 과제당 6회 균등 샘플링(34×6=204 궤적), LIBERO는 장면별 6개 시드 × 6 샘플.

## 5. 구현 세부

- 인코더: MAE 사전학습 ViT-B(정적 + 손목 카메라), Perceiver Resampler, CLIP ViT-B/32 텍스트 인코더, 상태/행동은 MLP(팔 6-D와 그리퍼 분리).
- 디코더: 이미지는 ViT 인코더 구조 + 선형층 패치 예측, 행동은 MLP로 7-D, 그리퍼는 0.5 임계값 이진화.
- 규모: 총 370M, 학습 파라미터 120M. 4×H100에서 CALVIN 약 60시간, LIBERO 약 5시간 사전학습.
- 청크 설정: CALVIN은 이미지 청크 1, 행동 청크 5, 궤적 길이 16; LIBERO는 1/3/7.
- TTT: LoRA를 q/k/v/o 및 gate/up/down proj에 적용. CALVIN rank 128, LIBERO rank 64, 단일 epoch 단일 step, AdamW lr 0.005, weight decay 0.01. RTX 4090에서 CALVIN 과제당 약 8분, LIBERO 장면당 약 2분.

## 6. 실험 설정

- CALVIN: RoboFlamingo, SuSIE, GR-1, 3D Diffuser Actor, CLOVER, Seer, Seer-Large와 비교(5연속 성공률, 평균 길이).
- LIBERO-10: MT-ACT, MVP, MPI, OpenVLA, Seer와 과제별 비교.
- 절제: 월드모델 유무(Table 3), LoRA rank(Table 4), TTT 데이터량(Table 5), 가우시안 노이즈(Table 6), 청크 길이(Table 7), LoRA vs 전체 미세조정(Table 8), 설정별 TTT(Table 9), 백본(Table 10).

## 7. 주요 결과

**Table 1 – CALVIN (%)**

| 모델 | 1 | 2 | 3 | 4 | 5 | Avg. Len |
|---|---|---|---|---|---|---|
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| CLOVER | 96.0 | 83.5 | 70.8 | 57.5 | 45.4 | 3.53 |
| Seer | 93.0 | 82.4 | 72.3 | 62.6 | 53.3 | 3.64 |
| Seer-Large | 92.7 | 84.6 | 76.1 | 68.9 | 60.3 | 3.83 |
| WorldAgen | 96.3 | 87.7 | 76.8 | 67.3 | 59.1 | 3.87 |
| **WorldAgen-TTT** | 96.6 | 88.5 | 78.5 | 68.7 | 60.5 | **3.93** |

**Table 2 – LIBERO-10 평균 성공률**: MT-ACT 41.0, OpenVLA 54.0, MVP 68.2, MPI 77.3, Seer 78.7, WorldAgen 75.5, **WorldAgen-TTT 79.0**. TTT 후에도 "Put both pots on stove"(45.0), "Pick book and place it in back"(50.0) 등은 Seer보다 낮다.

**Appendix Table 11 – LIBERO suites**: Spatial 93.0 / Object 79.5 / Goal 91.0 / LIBERO-10 79.0 (평균은 보고되지 않음).

**절제**
- 월드모델 제거 시 CALVIN 3.87 → 2.96, LIBERO 75.5% → 46.5%(Table 3; 본문은 78.0%로 서술해 표와 불일치).
- LoRA rank 16~256 간 차이 0.02 이내(3.918~3.930).
- TTT 데이터 6 → 204에서 3.871 → 3.928, 340에서 3.917로 소폭 하락.
- LoRA TTT 3.93 vs 전체 미세조정 TTT 3.85(오히려 기준 3.87보다 하락).
- 가우시안 노이즈(std 0.1) 하에서도 CALVIN 3.90, LIBERO 78.0.
- 백본: GPT-2 3.32/75.0 vs Qwen3 3.87/75.5.

## 8. Related Work 상의 위치

GR-1/Seer(예측 역동역학), PAD, UniPi 등 "미래 관측 예측이 행동을 돕는다"는 계열에 속하며, 구조적으로는 Seer 설정(24층/12헤드)을 거의 그대로 따른다. 차별점은 월드모델을 사전학습 보조 목적이 아니라 배포 시 자기지도 적응 신호로 쓰는 점으로, 비전/NLP의 TTT(Sun et al. 2020, LoRA-TTT)를 VLA의 월드모델 헤드에 이식한 형태다. 파라미터를 전혀 바꾸지 않는 in-context 적응(RICL, ICI-VLA 등)과는 반대 방향이다.

## 9. 강점

1. 과제 무관 월드모델을 attention mask 하나로 분리해, 라벨 없는 탐색 데이터로도 공유 표현을 갱신할 수 있게 한 설계가 간결하다.
2. TTT 비용이 작다(RTX 4090, 수 분, 약 200 샘플).
3. 월드모델 유무 절제(2.96 → 3.87)가 공동 학습의 효과를 명확히 보여준다.
4. LoRA vs FFT, rank, 데이터량 등 TTT 하이퍼파라미터 분석이 비교적 충실하다.

## 10. 약점 및 한계

1. TTT 이득 자체는 작다(CALVIN +0.06, LIBERO-10 +3.5pp). rank 절제의 변동폭(0.02)과 비교하면 시드 분산 대비 유의성이 불분명하며, 다중 시드 결과가 없다.
2. CALVIN 분할(ABC→D 여부)이 명시되지 않아 비교 해석이 어렵다.
3. LIBERO 월드모델 절제 수치가 표(75.5)와 본문(78.0)에서 다르고, Table 11이 TTT 적용 여부를 밝히지 않는다.
4. CALVIN TTT는 "학습셋에 등장한 34개 지시문"으로 탐색하고 테스트 환경의 첫 샘플 장면에서 적응한다. 테스트 환경 접근을 전제하므로 다른 방법과의 비교는 공정성 측면에서 조심해야 한다.
5. "Simulator to Real World"는 가우시안 노이즈 주입뿐이며 실제 로봇 실험은 없다.
6. 한 가지 모델 크기와 구조만 평가(저자 스스로 명시).

## 11. 재현 및 확장 아이디어

- TTT 효과를 여러 시드와 CALVIN ABC→D / ABCD→D 양쪽에서 명시적으로 측정.
- 탐색 정책을 무작위가 아닌 불확실성 기반 탐색으로 바꿔 샘플 효율 비교.
- 행동 헤드까지 함께 적응시키는 joint TTT(저자 제안)와 망각 방지 정규화 결합.
- 실제 로봇에서 조명/물체 변경 시나리오로 TTT 이득 검증.

## 12. 총평

WorldAgen은 Seer 계열 공동 상태-행동 예측 모델에 과제 무관 월드모델 헤드와 배포 시 LoRA 기반 월드모델 적응을 더한 깔끔한 아이디어를 제시한다. 공동 학습 자체의 효과는 크게 나타나지만, TTT가 더하는 이득은 작고 통계적 검증이 부족하며 CALVIN 분할 등 평가 세부가 불명확하다. "월드모델을 적응 신호로 쓴다"는 방향성의 초기 증거로 보는 것이 적절하다.

**한 문장 요약**: 공유 Transformer에서 정책과 과제 무관 월드모델을 attention mask로 분리해 공동 학습하고, 배포 시 무작위 탐색 데이터로 월드모델만 LoRA 적응시켜 CALVIN 3.93, LIBERO-10 79.0%를 보고한 370M 규모 VLA.

<!-- VERIFIED: pdf -->
