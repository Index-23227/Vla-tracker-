# GeomVLA: Unifying Scene, Motion, and Action in 3D

> **한 줄 요약**: Florence-2 VLM 특징을 깊이·카메라 보정으로 로봇 기준 3D 좌표에 올리고, SpatialTrackerV2 의사 라벨로 학습한 **3D Scene Trajectory Denoiser**의 중간 motion 토큰을 3D flow-matching 행동 디노이저에 주입해 "장면–미래 운동–행동"을 하나의 metric 3D 프레임에 묶은 1.2B VLA. 로봇 행동 사전학습 없이 CALVIN ABC→D 4.624, LIBERO 평균 98.4%, RoboTwin2.0 50과제 Easy/Hard 78.6/76.2%, ALOHA2 실물 8과제 평균 62.5%.

- **arXiv**: 2609.13812 (PDF 스탬프: v2, 2026-09-15, cs.RO)
- **소속**: Carnegie Mellon University, NVIDIA
- **학회**: Not stated in the paper
- **코드**: Not stated in the paper (프로젝트 페이지 ziyin-xiong.github.io/geomvla.io 만 명시)
- **백본**: Florence-2 (공개 초기화), 로봇 행동 사전학습 없음

---

## 1. 배경 및 동기

대부분의 VLA는 2D 이미지 위에서 동작하지만 조작은 3D 기하·접촉·운동 추론을 요구한다. 3D VLA 계열(SpatialVLA, 4D-VLA, PoseVLA 등)은 정적 3D 장면 표현과 행동 생성의 정렬에 집중했고, 미래 예측을 중간 추론으로 쓰는 계열(TraceVLA, FlowVLA, LaMP, WorldVLA 등)은 주로 픽셀/이미지 공간에서 미래를 예측한다. 저자들은 "장면 이해, 운동 추론, 제어가 서로 다른 기하 공간에 있다"는 불일치가 핵심 병목이라고 보고, **미래 운동 추론은 지각·운동 예측·제어가 같은 기하를 공유할 때 가장 유용하다**는 가설을 세운다.

## 2. 문제 정의

- 입력: 다시점 RGB-D {(I_v, D_v)}, 언어 지시 ℓ, 고유감각 p.
- 출력: 행동 청크 A = (a_1, …, a_T) — 엔드이펙터 이동/회전 델타 + 이진 그리퍼 상태 (선택적으로 관절각).
- 목표: 장면 토큰, 예측된 점 운동, 카테시안 행동 토큰을 **모두 로봇 베이스 프레임**에서 표현하고, 예측된 궤적을 open-loop 계획으로 실행하는 대신 **잠재 기하 추론 신호**로만 사용.

## 3. 방법

**3.1 3D 장면 특징**: Florence-2가 RGB와 지시를 패치 토큰·언어 임베딩으로 인코딩하고, 패치 중심 깊이를 쌍선형 보간해 카메라 내·외부 파라미터로 베이스 프레임 3D 위치를 부여(lifted scene tokens). 고유감각은 현재 EE 포즈에 고정된 학습 토큰.

**3.2 3D Scene Trajectory Denoiser**: 전면 카메라에서 20×20 그리드(N=400) 점을 3D 앵커로 올리고, H=15 미래 스텝의 스텝별 3D 변위 증분 F ∈ R^{N×H×3}를 예측. 인코더는 kNN 기반 국소 기하 인식 어텐션(상대 위치 bias), 디코더는 Morton(Z-order) 곡선 직렬화 윈도우 공간 어텐션 + 시간 어텐션 + [V_g; E] 교차 어텐션을 번갈아 쓰는 flow-matching 모델. 손실은 motion-weighted rectified flow(가시성 마스크·움직임 saliency 가중, 정적 점에도 최소 가중치 유지).

**3.3 3D Action Denoiser**: 행동 추론 시 trajectory denoiser를 **τ=0의 순수 노이즈에서 한 번만** 평가해 중간 디코더 텐서를 얻고, 학습된 시간 어텐션 쿼리로 앵커당 motion 토큰 z_m^i로 풀링 → 기하 인식 교차 어텐션 한 층(0 초기화 스칼라 잔차)으로 장면 토큰에 융합. 이후 3D RoPE를 쓰는 트랜스포머 블록이 [motion-aware 장면; 언어]에 교차 어텐션, 행동 궤적 자기 어텐션, adaLN-FFN(flow time, 고유감각)을 반복하고, rectified-flow 속도 + 그리퍼 BCE로 학습.

**3.4 관절각 변형(GeomVLA-JA)**: EE 행동 특징(한 번의 속도장 평가 후), motion-aware 장면, 언어를 조건으로 하는 관절각 디노이저를 추가, 같은 rectified-flow 목적.

## 4. 학습 절차

- **1단계**: VLM을 얼린 채 SpatialTrackerV2 의사 라벨로 trajectory denoiser 학습(비디오 3배 시간 다운샘플, 15스텝 = 원본 45프레임).
- **2단계**: 행동 디노이저 + 사전학습된 trajectory denoiser + VLM을 **행동 손실만으로** 공동 파인튜닝(궤적 손실은 유지하지 않음).
- 벤치마크 시연 데이터만 사용. 2단계 학습 시간: CALVIN 4×L40S 약 15시간, LIBERO 8×A100-40GB 약 25시간, RoboTwin2.0 5과제 설정 8×A100 약 48시간.
- 벤치마크별 하이퍼파라미터(Table 4): 행동 디노이저 hidden 960/912/912/256, 16 공유 어텐션 층, action horizon 10/10/16/8, 회전 표현 Euler/axis-angle/6D/6D.

## 5. 실험 설정

- **CALVIN ABC→D**, **LIBERO** 4 suite, **RoboTwin2.0**(50과제 전체 Easy/Hard 각 50 rollout + SimpleVLA-RL을 따른 5과제 부분집합).
- **실물**: ALOHA2의 오른쪽 ViperX300S 팔, Azure Kinect DK 2대 + 손목 RealSense D405. 8개 과제(3개는 50 demo, 5개는 20 demo), 모든 변형은 8과제 멀티태스크 정책 하나로 학습, 과제당 20 trial. π0.5는 공개 base 체크포인트에서 LoRA 파인튜닝(관절각 출력).

## 6. 주요 결과

**Table 1 – 시뮬레이션 비교**

| Method | Params | CALVIN | LIBERO Avg | RoboTwin 5과제 Avg |
|---|---|---|---|---|
| π0.5 | 3B+ | 3.964 | 96.8 | 81.2 |
| OpenVLA-OFT | 7B | 4.279 | 97.1 | 34.4 |
| FLOWER | 0.95B | 4.480 | 97.0 | – |
| X-VLA | 0.9B | 4.430 | 98.1 | 47.8 |
| SimpleVLA-RL | 7B | – | 99.1 | 76.4 |
| PoseVLA | 3B+ | – | 96.0 | 85.6 |
| LaMP | 5B | – | 98.3 | – |
| 2D-JA | 1B | 4.195 | – | 66.2 |
| GeomVLA-JA | 1.3B | 4.508 | – | 81.6 |
| GeomVLA w/o motion | 1.2B | 4.508 | 93.5 | 70.0 |
| **GeomVLA** | 1.2B | **4.624** | 98.4 | 84.8 |

- LIBERO 세부: Spatial 98.6 / Object 99.8 / Goal 97.6 / Long 97.6. 저자 표현으로 "RL 파인튜닝 없는 방법 중 최고 평균".
- RoboTwin2.0 50과제(Appendix Table 5): GeomVLA Easy 78.6 / Hard 76.2 vs 전체 파인튜닝 π0.5 75.9 / 75.7.
- 2시드 안정성: CALVIN 4.624 / 4.609, LIBERO 98.4 / 98.0.

**Table 3 – 실물 ALOHA2 (성공/20)**: GeomVLA 평균 62.5%(Stack blocks 19, Uncap 20, Place bottles 17, Stand bottle 17 등), GeomVLA-JA 59.4%, w/o motion 44.4%, π0.5 29.4%, 2D-JA 28.8%. GeomVLA는 w/o motion을 8과제 모두에서, π0.5를 7과제에서 앞선다(Insert marker는 π0.5 9/20 > GeomVLA 5/20).

## 7. 절제 연구 (CALVIN ABC→D, Table 2)

| 변형 | 점수 |
|---|---|
| 2DGeomVLA w/o motion | 4.411 |
| 2DGeomVLA (2D 운동 + 2D 행동) | 4.437 |
| 2DGeomVLA LaMP-style ((u,v)+깊이) | 4.462 |
| GeomVLA w/ 2D motion (이미지 공간 운동 + 3D 행동) | 4.085 |
| GeomVLA w/ denoised trace | 4.047 |
| GeomVLA w/ denoised latent | 4.334 |
| GeomVLA w/ partial denoising (τ=0.1) | 4.484 |
| GeomVLA encoder-only | 4.443 |
| GeomVLA w/o motion | 4.508 |
| GeomVLA w/ arbitrary frame | 4.596 |
| **GeomVLA** | **4.624** |

- 미래 운동 추론은 2D(4.411→4.437)와 3D(4.508→4.624) 모두에 도움.
- **기하 불일치가 가장 해롭다**: 이미지 공간 운동 + 3D 행동은 4.085로 완전 2D 모델보다도 낮음.
- 완전히 디노이즈된 궤적/잠재보다 **초기 노이즈 상태의 한 번 평가 특징**이 가장 좋음 — 확정 궤적보다 "어떻게 변할지"의 초기 표현이 유용.
- 임의 강체 프레임에서도 4.596으로 유지 → 이득은 로봇 베이스 프레임 자체가 아니라 **일관된 metric 기하**에서 온다.
- 부록: 예측 horizon H=32로 늘리면 4.272로 하락, 같은 크기의 무작위 초기화·동결 모듈로 대체하면 4.443 → 파라미터 수가 아닌 학습된 운동 특징의 효과. 지시를 바꾸면 궤적 끝점 오차가 CALVIN +3.40cm, LIBERO +5.44cm, RoboTwin +1.64cm 증가(언어 조건 의존).
- 추론 지연(L40S, 10-action 청크): 280.0 ms vs w/o motion 219.0 ms, 27.9% 오버헤드.

## 8. Related Work 상의 위치

- 3D Diffuser Actor, 3D FlowMatch Actor, Act3D(같은 CMU 그룹) 계열의 3D 정책을 VLM 의미 특징과 결합한 연장선.
- SpatialVLA·4D-VLA·PoseVLA·GeoVLA 등 3D VLA는 정적 장면 표현에 초점, GeomVLA는 **미래 운동까지 같은 3D 프레임**에 둔다.
- TraceVLA·FlowVLA·LaMP·RoboFlow4D 등 운동 유도 VLA는 이미지 공간에서 추론하거나 plan-then-act, GeomVLA는 3D 잠재 운동을 **행동 디노이저의 조건**으로만 사용.

## 9. 강점

1. **세밀한 통제 절제**: 2D/3D 운동 × 2D/3D 행동, 잠재 추출 시점, 좌표 프레임, 용량 대조군까지 체계적으로 분리해 "기하 일관성"이라는 주장을 직접 검증.
2. **사전학습 없이 경쟁력**: 로봇 행동 사전학습 없이 1.2B로 CALVIN SOTA, LIBERO 98.4, RoboTwin 50과제에서 전체 파인튜닝 π0.5 상회.
3. 관절각 출력(GeomVLA-JA vs 2D-JA)으로 행동 공간을 맞춘 비교를 추가해 3D 표현의 효과를 출력 파라미터화와 분리.
4. 2시드 재현, 지연 측정, 시연 수·과제 정의 등 세부 공개가 충실.

## 10. 약점 및 한계

1. **깊이·카메라 보정 의존**: 저자 스스로 캘리브레이션 오류가 기하 표현을 해친다고 인정. 단일 RGB 환경에는 바로 적용 불가.
2. RoboTwin 5과제 비교는 학습 데이터가 다름(전체 GeomVLA는 50과제, 변형들은 5과제) — 논문도 "fully controlled ablation이 아님"을 명시.
3. 절제는 대부분 CALVIN 한 벤치마크에서만 수행.
4. 벤치마크 규모 데이터만 사용해 대규모·교차 embodiment 일반화는 검증되지 않음. 실물은 단일 팔·과제당 20 trial.
5. RoboTwin 50과제에서 Blocks Ranking, Lift Pot, Handover Block 등 일부 과제는 π0.5보다 크게 낮다(예: Lift Pot 32/18 vs 88/88).
6. 코드/가중치 공개 여부는 논문에 명시되지 않음.

## 11. 재현 및 확장 아이디어

- 대규모 교차 embodiment 데이터로 사전학습했을 때 3D 운동 토큰의 이득이 유지되는지 검증.
- SpatialTrackerV2 의사 라벨 품질(가림·깊이 오류)에 대한 민감도 분석, 다른 3D 트래커 대체.
- 추정 깊이(단안 depth foundation model)로 캘리브레이션 의존 완화.
- 기하 추론과 언어 기반 계획/계층적 분해의 결합(저자 제안).

## 12. 총평

GeomVLA는 "미래 예측이 도움이 되느냐"가 아니라 "미래 예측을 **어떤 기하 공간에서** 하느냐"가 중요하다는 점을 설득력 있게 보여준다. 이미지 공간 운동 + 3D 행동이 완전 2D 모델보다 나쁘다는 결과는 특히 교훈적이다. 사전학습 없는 1.2B 모델로 세 시뮬레이션 벤치마크와 실물에서 강한 수치를 내지만, RGB-D·정확한 보정이라는 전제와 CALVIN 중심 절제는 일반화 범위를 제한한다.

**한 문장 요약**: 장면·미래 3D 점 운동·행동을 하나의 metric 3D 프레임에 두고, 궤적 디노이저의 초기 잠재를 행동 flow 디노이저 조건으로 쓰는 1.2B VLA — CALVIN 4.624, LIBERO 98.4, 실물 62.5%.

<!-- VERIFIED: pdf -->
