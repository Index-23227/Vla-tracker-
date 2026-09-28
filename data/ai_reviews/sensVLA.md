# sensVLA: Spatially-Grounded Vision-Language-Action Model for Autonomous Wheel Loader

> **한 줄 요약**: Qwen3-VL-2B(LoRA)와 flow-matching 액션 전문가를 결합하고, 라이다 BEV 특징을 VLM이 아닌 액션 전문가의 전용 cross-attention 경로로 직접 주입해 휠로더 실데이터에서 종방향 속도 RMSE를 카메라 전용 기준선 대비 줄이고(로딩 구간 0.996 → 0.717 m/s) 카메라 손실 시 성능 저하를 완화한 중장비용 VLA.

- **arXiv**: 2609.17021v1 (2026-09-15, cs.CV)
- **소속**: sensmore GmbH (Berlin, Germany)
- **코드**: Not stated in the paper

---

## 1. 배경 및 동기

휠로더 자율화는 비정형 지형 주행과 구동계·유압(암/버킷) 동시 제어가 필요하며, 파일 면까지의 거리, 버킷-지면 간격 같은 계량적 공간 정보가 핵심이다. RGB 토큰만으로는 이런 기하 정보가 부족하고, 실기 데이터는 대형 VLM을 전부 미세조정하기에는 적다. 저자들은 "BEV 특징을 네트워크의 어디에 넣어야 하는가"를 핵심 질문으로 둔다.

## 2. 문제 정의

입력: 전/후방 RGB, 짧은 자연어 과제 프롬프트, 현재 고유수용 상태(종속도, 요레이트, IMU 3개의 6채널 등 s∈R^20)와 10스텝 상태 이력, 0.5초 누적 전/후방 라이다. 출력: 6차원 행동(종방향 속도, 조향, 차체 좌표계 변위 dx/dy, 암 속도, 버킷 속도)의 10스텝 청크(stride 5, 25 Hz 기준 2초 지평).

## 3. 방법

- **VLM 경로**: Qwen3-VL-2B-Instruct. 전/후방 이미지를 동결된 ViT로 토큰화해 이어붙이고, 언어 디코더는 앞 14개 블록만 사용(후반 층은 사전학습 목적에 특화되어 제어에 기여가 적다는 논리). 마지막 8개 디코더 층에 LoRA(r=16, α=16). 상태는 MLP로 투영해 토큰으로 추가.
- **상태 이력**: 3단 합성곱 인코더 → Perceiver resampler(8 토큰).
- **BEV 경로**: 사전학습·동결된 PointPillars의 다중 스케일 특징을 업샘플·결합해 BEV 맵을 만들고, 32개 BEV 토큰으로 압축해 액션 전문가에 전용 cross-attention으로 전달.
- **액션 전문가**: 첫 층은 양방향 self-attention, 이후 causal self-attention + 이질적 cross-attention(VLM 컨텍스트 / 근사 항등으로 초기화된 게이트 잔차를 가진 BEV 토큰 / 둘의 연결). Flow matching 속도 회귀(t∼Beta(2,5), 성분별 가중 MSE), 추론 시 10스텝 midpoint ODE.

## 4. 데이터

독점(proprietary) 휠로더 실데이터: 라이다, RGB, IMU를 제어 타임스탬프에 동기화. 파일 적재(loading)와 지하 터널 주행(driving) 두 시나리오. 학습 200K 청크, 검증 40K 청크이며 녹화 시퀀스 단위로 분할해 시간적 누수를 막았다. 데이터 공개 여부는 언급되지 않는다.

## 5. 구현 세부

- 액션 전문가: hidden 384, 8 heads, dropout 0.2. Sec. IV-C는 "4-layer"(양방향 1 + causal 3)라 하지만 구현 세부는 "5 transformer layers (1 bidirectional + 4 causal)"로 서로 불일치한다.
- AdamW, 학습률 1e-5(전문가·BEV 헤드), 3e-6(BEV resampler), 1e-6(LoRA), cosine, warm-up 5%, weight decay 1e-3, bf16, 30 epoch, 유효 배치 32.
- 첫 epoch 동안 LoRA 동결(전문가 워밍업), BEV 토큰 드롭아웃, 상태 이력 스텝/전체 드롭아웃, EMA 검증.
- 기준선: 동일 구조·학습에서 BEV 경로만 제거한 카메라 전용 모델.
- 전체 파라미터 수와 학습 하드웨어는 Not stated in the paper.

## 6. 실험 설정

모든 평가는 검증셋에 대한 오프라인 open-loop 스텝별 물리 RMSE다. 과제군(loading/driving)별 층화, 2초 지평 최종 변위 오차, 입력 제거(img-blur, no-state-hist, no-bev, no-cam) 강건성 실험을 수행한다. 폐루프 실기 주행 결과는 보고되지 않는다.

## 7. 주요 결과

**Table I – 스텝별 RMSE (낮을수록 좋음)**

| 모델 | 전체 dx+dy (m) | 전체 v_x (m/s) | Loading dx+dy | Loading v_x | Driving dx+dy | Driving v_x |
|---|---|---|---|---|---|---|
| Baseline | 0.107 | 1.134 | 0.106 | 0.996 | 0.107 | 1.281 |
| **sensVLA** | **0.101** | **0.886** | **0.097** | **0.717** | **0.105** | **1.054** |

- 본문: 전체 dx RMSE 9.52 → 8.98 cm(−5.7%), loading dx 9.27 → 8.40 cm(−9.4%), loading v_x −28.0%, driving v_x −17.7%. 조향과 dy는 "statistically identical".
- loading 최종 변위 오차 0.538 → 0.521 m(−3.1%).

**Table II – 입력 제거 강건성 (z-정규화 평균 RMSE, anchor 대비 배율)**

| 입력 | Baseline | sensVLA |
|---|---|---|
| full | 1.017 (1.00×) | 1.022 (1.00×) |
| img-blur | 1.06 (1.04×) | 1.022 (1.00×) |
| no-state-hist | 1.175 (1.16×) | 1.090 (1.07×) |
| no-bev | — | 1.206 (1.18×) |
| no-cam | 2.378 (2.34×) | 1.704 (1.67×) |

카메라 제거 시 v_x RMSE가 기준선은 1.13 → 9.05 m/s, sensVLA는 0.89 → 2.47 m/s.

## 8. Related Work 상의 위치

PointPillars, BEVFusion, TransFuser 등 자율주행 BEV 인식 계열과 RT-2, OpenVLA, π0 등 VLA 계열을 잇는다. π0식 VLM + flow-matching 액션 전문가 구조를 따르되, BEV를 인식 스택 안에 두거나 VLM 토큰으로 넣지 않고 액션 전문가에 직접 연결한 점이 차별점이다. 기존 휠로더 연구의 모듈형 적재 파이프라인과 달리 end-to-end 제어를 지향한다.

## 9. 강점

1. "기하는 액션 전문가로, 의미는 VLM으로"라는 명확한 설계 가설과 이를 뒷받침하는 no-bev / no-cam 대칭 실험.
2. 중장비라는 드문 실세계 도메인의 실데이터 규모(200K 청크)에서 검증.
3. 카메라 손실 시 저하 배율 2.34× → 1.67×로 센서 장애 내성 개선을 정량 제시.
4. 데이터 부족 상황을 위한 단계적 PEFT 학습 레시피를 구체적으로 기술.

## 10. 약점 및 한계

1. 전부 오프라인 open-loop RMSE이며 폐루프 실기 성공률·안전 지표가 없다. VLA의 실제 제어 성능을 판단하기 어렵다.
2. 전체 dx+dy 개선은 0.107 → 0.101 m로 작고, 저자도 "aggregate parity"라 표현한다. 다중 시드나 신뢰구간이 없다.
3. 기준선이 자체 ablation 하나뿐이며, 다른 VLA나 BEV 기반 정책과의 비교가 없다.
4. 액션 전문가 층 수 서술이 4층/5층으로 불일치한다.
5. 데이터·코드 비공개, 파라미터 수와 하드웨어 미기재로 재현성이 낮다.
6. 언어 지시의 역할(다양한 프롬프트, 과제 전환)에 대한 분석이 없다.

## 11. 재현 및 확장 아이디어

- 실제 휠로더 폐루프 적재 사이클 성공률, 버킷 충전률, 사이클 시간 측정.
- PointPillars를 전문가와 공동 학습(저자 제안)하고 장기 시간 BEV 융합.
- BEV를 VLM 토큰으로 넣는 변형과의 직접 비교로 "어디에 넣을까" 질문에 대한 통제 실험 보강.
- 공개 오프로드/건설 데이터셋에서의 재현.

## 12. 총평

sensVLA는 라이다 BEV를 액션 전문가에 직접 연결하는 단순하고 합리적인 설계로, 중장비 VLA에서 공간 그라운딩의 가치를 보여 준다. 특히 카메라 손실 시 강건성 향상은 실용적으로 의미가 있다. 다만 결과가 전부 독점 데이터의 오프라인 RMSE이고 기준선이 자체 ablation뿐이어서, 실제 제어 성능에 대한 증거로는 초기 단계의 짧은 보고로 보는 것이 적절하다.

**한 문장 요약**: Qwen3-VL-2B + flow-matching 액션 전문가에 라이다 BEV 전용 cross-attention 경로를 더해 휠로더 실데이터에서 로딩 구간 종속도 RMSE 28% 감소와 카메라 손실 강건성 향상을 보인 중장비용 VLA.

<!-- VERIFIED: pdf -->
