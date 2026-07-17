# See like a Robot: Robot-Centric Pointmaps for Vision-Language-Action Models

## 1. 개요 (Overview)
VLA 모델은 시각 관측과 언어 지시로부터 로봇 행동을 예측하지만, 행동은 로봇 자신의 3D 좌표계에서 정의되는 반면 관측은 대개 카메라 좌표계에서 이루어진다. 본 논문은 이 "관측-행동 프레임 불일치(frame mismatch)"를 지적하고, 관측을 로봇 프레임의 3D 기하로 직접 표현하는 **로봇 중심 포인트맵(robot-centric pointmap)** 입력을 제안한다. KAIST AI와 Holiday Robotics 공동 연구(arXiv:2607.11498, 2026-07-13).

## 2. 문제 정의 (Problem)
고정 시점에서는 정책이 단일 관측→행동 매핑을 암기하면 되지만, 대규모 데이터셋이 다양한 카메라 시점의 데모를 aggregate하면 정책은 여러 시점을 로봇 프레임 행동으로 일반화해야 한다. RGB나 depth는 모두 카메라 프레임에 묶여 있어 이 부담을 정책에 전가한다.

## 3. 핵심 아이디어 (Key Idea)
포인트맵은 각 픽셀이 해당 장면점의 **로봇 프레임 3D 좌표(XYZ)** 를 담는 이미지다. RGB-D를 카메라 프레임에서 3D로 lift한 뒤 카메라-로봇 변환(Rc, tc)을 적용하고, 추가로 현재 엔드이펙터 위치로 re-center한다. 같은 물리적 점은 시점과 무관하게 동일한 로봇 프레임 좌표를 받으므로 시점 불변성을 얻는다.

## 4. 방법론 (Method)
- 포인트맵 구성: Eq.(1) 카메라 프레임 lift → Eq.(2) 로봇 base 프레임 변환 → Eq.(3) 엔드이펙터 중심화.
- 융합: 포인트맵을 RGB 인코더와 동일 구조의 별도 SigLIP 타워(RGB 인코더로 초기화)로 인코딩한 뒤, RGB 토큰에 **element-wise 덧셈**으로 융합(zc = fθ(Ic) + gϕ(PcEE)). 추가 토큰 없음, point-cloud 전용 인코더/voxel 불필요.

## 5. 아키텍처 (Architecture)
사전학습된 2D VLA 백본을 그대로 사용. π0.5와 SmolVLA 두 백본에 적용. RGB/포인트맵 각각 별도 vision tower, 융합 후 LLM과 action expert(flow matching)로 전달. 통제 실험에서는 base PaliGemma 체크포인트 기반 π-스타일 아키텍처를 scratch action expert와 함께 사용.

## 6. 설계 선택 분석 (Design Choices)
- RQ1: 카메라 정보(Plücker) 조건화(28.7)보다 로봇 프레임 기하 **사전 계산** 포인트맵(34.7)이 우수 (RGB 27.9 기준).
- RQ2: 이미지형 포인트맵 + add 융합(34.7)이 point cloud(PTv3 concat 32.8, MLP 24.2) 및 concat 융합(30.7)보다 우수.
- RQ3: 엔드이펙터 중심화가 base 중심화보다 우수(고정 36.9 vs 34.7), 랜덤 시점에서 하락폭 -0.3 vs -2.0.

## 7. 실험 설정 (Experimental Setup)
RoboCasa 24 atomic tasks(5 카테고리), 태스크당 50 데모·50 평가 에피소드. 시뮬레이션은 π0.5, SmolVLA 백본 fine-tune. 실제 로봇은 Franka Research 3 + RealSense D405(손목 고정)/D435i(외부, 3 학습 배치+1 held-out) 구성.

## 8. 주요 결과 (Results)
RoboCasa(Table 4, 고정 시점): **π0.5 + pointmap 62.9** (RGB π0.5 55.3, +7.6), SmolVLA + pointmap 41.4 (37.2, +4.2). 카테고리별 Doors 90.0 / Drawers 90.0 / Coffee 58.0 / Pick-and-place 52.8 / Turn 53.4. 최강 baseline인 KYC(59.1), PointVLA(57.3), GeoVLA(57.1), OC-VLA(56.3), FP3(42.8)을 모두 상회.

## 9. 강건성 및 실제 로봇 (Robustness & Real Robot)
학습 시점 변동 증가 시 RGB는 34.5→24.9(-9.6) 하락하나 pointmap은 37.6→35.8(-1.8)로 안정. 실제 로봇(Table 5): seen 시점 78.3 vs RGB 73.3(+5.0), unseen held-out 시점 66.7 vs 55.0(+11.7)로 이득 확대. DP3(63.3 seen / 48.3 unseen) 대비도 우수.

## 10. 기여 (Contributions)
(1) 사전학습 VLA에 point-cloud 인코더 없이 로봇 프레임 3D 기하를 주입하는 단순 메커니즘 제안, (2) 포인트맵 입력 설계 선택(사전계산·이미지형·EE 중심화)에 대한 통제된 분석 제공.

## 11. 한계 (Limitations)
포인트맵 주입 위치와 action expert·사전학습 recipe의 상호작용 미탐구. Point cloud 비교는 단일 샘플링 예산. 학습·평가 시 **정밀한 카메라 intrinsics/extrinsics 캘리브레이션 필요**. 카메라 개수·FoV 변화는 미포함.

## 12. 총평 (Assessment)
"행동이 정의되는 좌표계에서 관측을 표현하라"는 원칙을 이미지형 구조를 유지하며 구현한 실용적 연구. 최소 아키텍처 변경으로 다중 백본(π0.5, SmolVLA)에서 일관된 향상과 시점 강건성을 보여 3D-aware VLA의 유력한 입력 설계로 평가된다. 코드 미공개와 캘리브레이션 의존이 채택의 제약.

<!-- VERIFIED: pdf -->
