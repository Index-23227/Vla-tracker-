# SMILE — Smooth Motion for Improved Long-Horizon VLA Execution

- arXiv: 2608.29432 (v1 2026-08-29)
- 소속: Stony Brook University, Salesforce AI Research (Jongwoo Park, E-Ro Nguyen, Kanchana Ranasinghe, Cristina Mata, Xiang Li, Michael S. Ryoo)

## 1. 한 줄 요약

SMILE은 VLA의 action expert가 원시 행동 시퀀스 대신 **B-spline 계수**를 denoise하도록 바꾸는 "아키텍처 보존형" 행동 인터페이스다. 디코딩된 행동이 시간적으로 결합되어 매끄러워지므로, 한 번의 모델 호출로 더 긴 고정 실행 구간 H를 안전하게 실행할 수 있다. SMILE-Evo1은 LIBERO 98.0%(Evo1 92.7%), SMILE-VPP는 CALVIN ABC→D 평균 길이 4.42(VPP 4.33)를 달성하면서 행동당 지연도 줄였다.

## 2. 문제 설정

Action chunking에서 호출당 실행하는 행동 수 H를 늘리면 추론 비용이 분산되지만, 원시 chunk에 포함된 고주파 진동·방향 불일치·이상치가 open-loop 구간 동안 누적되어 성공률이 떨어진다. 그림 2(a)에서 SmolVLA와 Evo1 모두 H가 커질수록 LIBERO 성공률이 감소한다. 핵심 질문은 "생성된 행동 시퀀스를 얼마나 믿을 만하게 만들어야 더 긴 H를 쓸 수 있는가"이다.

## 3. 핵심 아이디어

평활성을 더 큰 백본이나 별도 플래너가 아니라 **행동 표현 자체**로 부과한다. 각 chunk를 이동(K×3)·회전(K×3)·그리퍼(K×1)의 B-spline 계수로 이루어진 macro-action으로 표현하고, 고정 basis로 F-스텝 native 행동을 복원한다. 이웃 행동들이 소수의 계수로 공동 결정되므로 개별 예측 이상치에 덜 민감하다.

## 4. 아키텍처

기존 VLA 백본, 조건 경로, 주 denoising 블록은 그대로 두고, action-side 입력 projection(노이즈 섞인 계수 토큰을 받도록)과 출력 head(계수 head)만 교체한다. 디코딩: 누적 이동 P=ΦΘ_tr, 누적 회전 R=ΦΘ_rot, 그리퍼 G=clip(ΦΘ_grip). 이동은 누적 경로의 유한 차분, 회전은 SO(3)에서 인접 상대 회전 log(exp(R(τ−1))⁻¹exp(R(τ))), 그리퍼는 절대값. 기본 설정은 K=8 제어점, 3차(P=3) spline. 적용 대상은 VLM 중심 SmolVLA(0.4B)·Evo1(0.8B), 예측-시각형 VPP(1.8B), 픽셀-모션 조건형 DAWN(1.2B)으로, 서로 이질적인 네 가지 action expert다.

## 5. 학습과 추론

계수 타깃은 각 베이스라인과 동일한 전문가 궤적에서 시작 스텝마다 다음 F개 행동을 누적 궤적으로 변환한 뒤 B-spline basis에 최소제곱 투영해 얻고, 데이터셋 수준 통계로 정규화한다(에피소드 경계에는 validity mask). 베이스 모델의 원래 손상 과정과 목적(flow matching이면 velocity, diffusion이면 noise)을 계수에 그대로 적용하며 세 모달리티 가중치는 모두 1이다. 추론 시에는 denoise된 계수를 같은 basis로 디코딩해 처음 H개를 실행한다(H는 롤아웃 내에서 고정). 실로봇에서는 각 쌍을 동일한 공식 사전학습 체크포인트에서 같은 스텝 수만큼 파인튜닝했다.

## 6. 주요 결과 — LIBERO (Table I)

| 모델 | H | Spatial | Object | Goal | Long | All | ms/행동 | 크기 |
|---|---|---|---|---|---|---|---|---|
| SmolVLA | 1 | 78.8 | 91.4 | 82.8 | 42.8 | 74.0 | 243.9 | 0.4B |
| **SMILE-SmolVLA** | 10 | 84.8 | 95.0 | 86.8 | 63.6 | 82.6 | 24.2 | 0.4B |
| Evo1 | 14 | 88.4 | 94.2 | 98.2 | 89.8 | 92.7 | 8.5 | 0.8B |
| **SMILE-Evo1** | 16 | 97.2 | 99.6 | 96.8 | 98.2 | **98.0** | 8.0 | 0.8B |
| OpenVLA-OFT | 8 | 97.7 | 98.0 | 96.1 | 96.1 | 96.8 | 22.3 | 7.7B |
| GR00T N1.7 | 8 | 97.2 | 96.8 | 97.8 | 93.4 | 96.3 | 14.7 | 3B |

SMILE-SmolVLA는 H를 1→10으로 늘리며 10.1× 속도 향상과 +8.6pt, 특히 Long 42.8→63.6을 얻는다. SMILE-Evo1은 Goal만 소폭 하락(98.2→96.8)하고 나머지 suite는 모두 향상된다.

## 7. 주요 결과 — CALVIN ABC→D (Table II)

| 모델 | H | 1 | 2 | 3 | 4 | 5 | Avg.Len | ms/행동 |
|---|---|---|---|---|---|---|---|---|
| DreamVLA | 1 | 0.982 | 0.946 | 0.895 | 0.834 | 0.781 | 4.44 | 193.2 |
| DAWN | 10 | 0.978 | 0.916 | 0.813 | 0.752 | 0.641 | 4.10 | 32.0 |
| **SMILE-DAWN** | 15 | 0.972 | 0.904 | 0.827 | 0.769 | 0.711 | 4.18 | 22.7 |
| VPP | 10 | 0.965 | 0.909 | 0.866 | 0.820 | 0.769 | 4.33 | 19.1 |
| **SMILE-VPP** | 15 | 0.962 | 0.925 | 0.885 | 0.846 | 0.798 | **4.42** | 13.0 |

이득은 주로 후반 subtask에서 나온다. SMILE-VPP는 DreamVLA보다 0.02 낮지만 행동당 지연은 14.9× 짧다.

## 8. 실로봇 결과 (Table III)

xArm7, 3인칭+손목 카메라, 6개 물체 × {plain, cluttered} = 12 과제, 과제당 20회. Plain: Evo1 26.7%→SMILE-Evo1 78.3%, VPP 82.5%→SMILE-VPP 98.3%. Cluttered: 8.3%→56.7%, 70.8%→88.3%. Cluttered에서 hit rate는 45.0%→22.5%(Evo1), 25.0%→11.7%(VPP)로 줄고, drop rate는 두 SMILE 변형 모두 0.0%다. 실행 구간은 Evo1 6→16, VPP 10→15로 늘어 각각 2.7×, 1.5× 속도 향상.

## 9. 평활성 분석과 어블레이션

같은 H=10에서 SmolVLA와 SMILE-SmolVLA를 비교하면 non-boundary 가속도 78.6%, velocity sign-change rate 42.3%, global 가속도 18.6%, boundary 가속도 20.5% 감소(그림 7). 감소폭이 chunk 내부에서 훨씬 크므로, 개선의 주된 원인은 chunk 경계 평활화가 아니라 intra-chunk jitter 억제다. Table IV: P=3에서 K=6/8/10 → 80.05/82.55/80.75, K=8에서 P=3/5/7 → 82.55/81.90/81.75. 저차·적당한 제어점 수가 최적이며 복잡도를 높이면 오히려 평활성 편향이 약해진다. 그림 9에서는 여러 H(SmolVLA 6/10/15, Evo1 14/16/18)에서 SMILE이 일관되게 우세하다.

## 10. 강점

- 네 가지 이질적인 action expert(flow matching·diffusion, VLM 중심·예측 비디오·픽셀 모션)에 동일하게 적용되어 범용성이 설득력 있다.
- 모델 크기를 유지한 채 정확도와 행동당 지연을 **동시에** 개선해 accuracy–efficiency frontier를 실질적으로 이동시킨다.
- 동일 H 조건의 평활성 지표로 메커니즘(intra-chunk jitter 감소)을 직접 검증했다.
- 실로봇에서 drop/hit/wrong-object 같은 세부 실패 유형까지 보고한다.

## 11. 한계와 의문점

- LIBERO 헤드라인(SMILE-Evo1)과 CALVIN 헤드라인(SMILE-VPP)이 서로 다른 베이스 모델이다. 한 "모델"이라기보다 인터페이스 기법이다.
- B-spline 행동 표현 자체는 BEAST, ABPolicy, B-spline policy 등에서 이미 탐구되었고, 차별점은 주로 "기존 VLA에 아키텍처 보존형으로 끼워 넣어 고정 H를 늘린다"는 실험적 초점이다.
- H는 롤아웃 내 고정이며, 적응형 horizon과의 결합은 다루지 않는다.
- 실로봇 Evo1 베이스라인 성공률(plain 26.7%)이 매우 낮아 개선폭이 과장되어 보일 여지가 있다. 과제당 20회로 표본도 작다.
- 시뮬레이션 결과의 시드 수·표준편차는 보고되지 않는다.

## 12. VLA-Tracker 관점 평가

각 베이스 action expert의 계수 head를 새로 학습(파인튜닝)해 자체 정책을 만들고 정량 결과를 보고하므로 수록 대상이다. 트래커의 `libero` 블록에는 SMILE-Evo1(98.0, 논문 보고 평균), `calvin` 블록에는 SMILE-VPP(4.42)를 기록했으며 두 블록의 베이스 모델이 다르다는 점을 eval_condition에 명시했다. SMILE-SmolVLA, SMILE-DAWN, 원시 베이스라인, 실로봇, spline 어블레이션은 별도 블록으로 분리했다. 순위 비교 시 이 항목은 "특정 백본의 성능"이 아니라 "행동 표현 교체의 효과"로 읽는 것이 적절하다.

<!-- VERIFIED: pdf -->
