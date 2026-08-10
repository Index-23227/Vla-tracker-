# SAM3D-VLA: SAM3D 교사 모델을 활용한 객체 중심 3D 표현 정렬 VLA

*arXiv:2607.25912 · University of Hong Kong / SIAT-CAS / HUST / Beihang / Infiforce · 2026-07-28*

## 1. 개요 (Overview)

SAM3D-VLA는 π0 아키텍처 위에 구축된 **객체 중심(object-centric) 3D 표현 정렬 프레임워크**다. 학습 시에만 동결된 SAM3D를 교사(teacher)로 사용하여 과제 관련 객체의 3D shape/layout prior를 π0의 중간 시각 특징에 증류하고, 추론 시에는 원래의 RGB-language-to-action 파이프라인을 그대로 유지한다. 즉 depth, point cloud, mask, SAM3D, 추가 3D 모듈이 배포 시점에 전혀 필요 없다. LIBERO 평균 99.1%, CALVIN ABC→D 평균 길이 4.11을 달성했다.

## 2. 문제 정의 (Problem Statement)

기존 VLA 대부분은 2D RGB 관측과 action 예측 손실만으로 학습되어, 대상 객체의 형상·자세·크기·공간 배치 같은 3D 속성을 포착하지 못한다. 이는 가림(occlusion), 자세 변화, 클러터, 정밀 배치 과제에서 명확한 한계로 드러난다. 이를 보완하려는 선행 연구는 (a) depth/point cloud/RGB-D 같은 **명시적 3D 입력**을 정책에 추가하거나(PointVLA, BridgeVLA, GeoVLA, 3D-CAVLA), (b) 공간 정보를 아키텍처·action space·표현 수준 감독으로 **암묵적으로** 주입한다(SpatialVLA, Spatial Forcing). 그러나 전자는 입출력 인터페이스를 바꾸고 테스트 시 3D 센서를 요구하며, 후자는 대체로 **scene-level(장면 전역)** 공간 표현에 머문다. 한편 visual grounding 계열(ReconVLA, RoboGround, ECoT, GraspVLA)은 감독이 2D 이미지 공간에 국한되어 대상 객체의 3D 형상/배치 prior를 제공하지 못한다. 저자들의 질문: **원래 추론 파이프라인을 보존하면서 RGB 기반 VLA에 객체 중심 3D 지식을 주입할 수 있는가?**

## 3. 핵심 방법 (Method)

**SAM3D 교사 특징 추출.** SAM3D는 단일 이미지에서 동작하므로, 멀티뷰 관측 I_t ∈ R^(B×V×3×H×W)를 뷰 차원으로 flatten하여 (BV)×3×H×W로 만들고 서브태스크별 객체 마스크도 동일하게 flatten한다. 동결된 SAM3D 교사에 (이미지, 마스크) 쌍을 넣어 마지막 transformer block에서 dense 객체 중심 3D 특징 T_t ∈ R^(BV)×L_T×D_T를 얻는다.

**공간 재샘플링.** SAM3D와 π0의 토큰 해상도가 다르므로 교사 시퀀스를 2D grid로 reshape하고, global token이 있으면 제거한 뒤 bilinear interpolation으로 학생 시각 토큰 해상도(H_S×W_S = L_S)에 맞춘다. 뷰 축으로 재조립하여 T̄_t ∈ R^(B×(V·L_S)×D_T)를 얻어 π0 시각 토큰과 공간적으로 정렬한다.

**정렬 손실.** 학생 측에서는 선택된 Gemma 레이어의 중간 시각 특징 S_t = h_t^(m) ∈ R^(B×(V·L_S)×D_S)를 뽑아 projection P_φ로 SAM3D 특징 공간에 사영한다. 서브태스크 객체 마스크로부터 토큰 수준 마스크 m_t ∈ {0,1}를 만들어 **대상 객체 토큰에만** 감독을 적용하고, 양쪽 특징을 ℓ2 정규화한 뒤 masked normalized MSE를 계산한다:

> L_align = MSE( Norm(T̂_t)[M_t], Norm(T̄_t)[M_t] )

정규화 덕분에 방향성 정렬을 유도하고 SAM3D–π0 간 스케일 차이에 강건하다. 최종 목적함수는 **L = L_action + α·L_align** 이다.

**Frozen-representation probing.** 정책 학습 후 VLA backbone을 전부 동결하고, 정렬에 쓰인 동일한 중간 특징 S_t 위에 2층 MLP probe P_ω만 학습시켜 재샘플링된 SAM3D 타깃을 예측한다(L_probe, 동일 마스크 사용). probe 성능이 좋을수록 SAM3D류 객체 중심 3D prior가 학습된 표현에서 더 잘 복원 가능하다는 뜻으로, π0 베이스라인과 비교해 메커니즘을 검증한다.

## 4. 아키텍처 (Architecture)

π0 [Black et al. 2024]를 그대로 계승한다. SigLIP vision encoder + Gemma language model로 구성된 VLM backbone(Mixture of Transformers 구성)이 멀티뷰 RGB 이미지 I^1..n_t, 언어 명령 ℓ, 로봇 상태 q_t를 토큰화하고, H_t = f_VLM(·)의 hidden representation이 연속 action expert를 조건화한다. Action expert는 조건부 flow matching으로 학습된다: 노이즈 ϵ ~ N(0,I), timestep τ ∈ [0,1], A^τ_t = τA_t + (1−τ)ϵ에 대해 velocity field v_θ가 (A_t − ϵ)를 예측하도록 L2 손실로 최적화되며, 추론 시 τ=0→1 적분으로 action chunk를 생성한다. 학습 시에만 SAM3D 교사, adapter/projection, probe가 붙는다.

## 5. 학습 설정 (Training Setup)

서브태스크 인식 데이터 파이프라인이 핵심이다. 상위 명령을 LLM(GPT-4o)으로 서브태스크로 분해하고, 각 프레임·카메라 뷰마다 서브태스크 관련 객체를 Grounding DINO / YOLOv12로 grounding한 뒤 SAM2로 이진 마스크를 생성한다. 생성된 (이미지, 마스크) 쌍이 동결 SAM3D 교사에 입력되어 정렬 타깃이 된다. 실환경 학습은 8×NVIDIA H100 GPU에서 수행했다. 배포 시 서브태스크 분해·검출·분할·SAM3D 모듈 전부 제거된다.

## 6. 벤치마크 결과 (Benchmark Results)

**LIBERO (Table 1)** — Spatial 99.2 / Object 99.7 / Goal 99.1 / Long 98.4 / **Average 99.1%**로 전체 최고 성능. 비교군: π0 94.2, UniVLA 95.2, OpenVLA-OFT 97.1, SpatialVLA 78.1, GeoVLA 97.7, 3D-CAVLA 98.1, Spatial Forcing 98.5. 특히 **LIBERO-Long에서 98.4%**로 가장 큰 격차(Spatial Forcing 96.0, GeoVLA 96.6)를 벌린다. LIBERO-Long 각 과제가 두 개의 순차 서브태스크로 구성되어 단계마다 대상 객체가 바뀌기 때문에, 서브태스크별 마스크 감독이 직접적으로 작동한다는 해석이다.

**CALVIN ABC→D (Table 2, 500 rollouts)** — 1/5~5/5 성공률 96.2 / 89.1 / 80.5 / 73.6 / **71.6%**, **평균 길이 4.11**로 최고. 비교군: ReconVLA 3.95, UniVLA 3.80, OpenVLA 3.27, CLOVER 3.53, GR-1 3.06. 체인이 길어질수록 격차가 커져(5/5에서 71.6 vs 64.1) 장기 지평 일관성이 개선됨을 보인다.

## 7. Ablation 및 메커니즘 검증 (Analysis)

본 논문의 주요 메커니즘 검증은 별도 ablation table이 아니라 **frozen-representation probing 실험**이다. VLA를 동결한 채 probe만 학습했을 때, SAM3D 정렬 모델이 원래 π0보다 객체 중심 3D 타깃을 더 정확히 예측한다 — 즉 정렬 손실이 실제로 3D prior를 표현 안에 심었음을 시사한다. 또한 masked normalized MSE 설계(ℓ2 정규화 + 객체 토큰 마스킹)는 스케일 불변성과 과제 무관 배경 토큰 억제라는 두 가지 목적을 동시에 수행한다.

## 8. 서브태스크 인식 감독 (Subtask-aware Supervision)

Fig. 4가 보여주듯 "Place the food on the green plate and assemble a sandwich"는 "Pick up the green plate and put it on the desk" → "Pick up the bread and put it on the plate" → "Pick up the bacon..." 식으로 분해되고, 각 단계가 서로 다른 대상 객체 마스크와 연결된다. 이것이 단순 scene-level 3D 감독과 구별되는 핵심 설계로, 조작 단계에 따라 감독 신호가 이동하도록 만든다.

## 9. 실환경 실험 (Real-World)

AgileX Piper-X 플랫폼(전방향 바퀴 3개, 6-DoF 양팔, 손목 카메라 2개 + 헤드 카메라)에서 요리·꽃꽂이·블록 쌓기 3개 시나리오를 평가했다. 표준 설정(ST)과 가림/방해물/위치 변화가 포함된 occlusion 설정(OC) 두 조건으로 나눈다. 평균 성공률은 π0 대비 **ST 50.2% → 65.2%**, **OC 21.3% → 44.3%**로 개선되며, 장기 과제에서 격차가 특히 크다(Cook 15→40 ST / 5→26 OC, Flower 30→62 / 13→45, Stack 23→54 / 10→38). 다만 기본 과제 중 Open drawer는 ST에서 68%→62%로 오히려 하락한다.

## 10. 강점 (Strengths)

(1) **추론 파이프라인 무변경** — 성능 향상을 위해 테스트 시 depth·point cloud·mask·3D 모듈을 요구하지 않아 배포 비용이 π0와 동일하다. (2) **scene-level이 아닌 object-centric 감독**이라는 명확한 차별점과, 이를 뒷받침하는 LIBERO-Long/CALVIN 5-step 결과. (3) probing 실험으로 "성능이 올랐다"를 넘어 "표현이 실제로 바뀌었다"를 검증한 점. (4) 시뮬레이션(LIBERO/CALVIN)과 실환경(occlusion 설정 포함) 양쪽 평가.

## 11. 약점 및 한계 (Weaknesses & Limitations)

저자들이 스스로 인정하듯 학습 파이프라인이 자동 생성된 서브태스크 주석과 객체 마스크에 의존하므로 분해·grounding·segmentation 오류가 노이즈 감독으로 전파된다. SAM3D가 단일 이미지 기반이라 심한 가림, 투명 물체, 불량 시점에서 신뢰도가 떨어진다. 실환경 평가는 단일 로봇 플랫폼의 tabletop 과제로 한정된다. 리뷰어 관점의 추가 지적: α 민감도, 정렬 레이어 m 선택, α=0 대비 정량 ablation table이 본문에 제시되지 않아 각 구성요소의 기여를 분리하기 어렵고, probing 결과도 수치 테이블 없이 서술 수준에 머문다. LIBERO 99.1%는 포화 구간이라 0.6%p 차이(Spatial Forcing 98.5)의 통계적 유의성 논의가 없다. 코드도 미공개다.

## 12. 총평 (Assessment)

판정: **ACCEPTED** — π0 기반의 완결된 정책 모델(policy)이며 flow matching action expert로 연속 action을 출력한다. Spatial Forcing이 개척한 "implicit 표현 정렬" 노선을 scene-level에서 **object-centric + subtask-aware**로 정제한 자연스럽고 잘 실행된 후속 연구다. 아이디어의 독창성은 점진적이지만, LIBERO 99.1 / LIBERO-Long 98.4 / CALVIN 4.11이라는 SOTA 수치와 추론 비용 무증가라는 실용적 장점이 명확하다. Ablation의 부재가 가장 아쉬운 부분이다.

<!-- VERIFIED: pdf -->
