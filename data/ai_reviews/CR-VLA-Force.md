# CR-VLA-Force: Learning Control-aware Compliance VLA Model for Robust Contact-rich Robotic Manipulation

> **한 줄 요약**: π0에 힘 이력 인코더와 MoE 기반 cross-modal fusion expert를 붙여 자세와 feedforward 힘을 함께 예측하게 하고, 이를 500 Hz 가변 강성 admittance 제어기가 추종하는 slow–fast 계층형 force-aware VLA(본문 명칭 CC-VLA). UR5e 접촉 과제 6개 조건 평균 성공률 89.2%(ForceVLA 73.2%, π0 47.3%), 칠판 닦기 힘 추종 오차 5.52%.

- **arXiv**: 2609.05832v1 (2026-09-05, cs.RO)
- **소속**: Huawei(CloudRobo), South China University of Technology, Ola Dimensions, The Hong Kong Polytechnic University
- **백본**: π0 (PaliGemma + flow-matching action expert)
- **참고**: 저장소의 CR-VLA(Continuous Reasoning, 2606.00229)와는 무관한 별개 논문.

---

## 1. 배경 및 동기

ForceVLA, TA-VLA처럼 F/T 신호를 VLA에 넣은 연구는 있었지만, 대부분 위치 제어 기반이라 action chunk를 실행하는 동안 과도한 접촉력이 생겨도 즉시 보정하지 못한다. VLA 추론 주기(수 Hz)와 접촉 제어에 필요한 주기(수백 Hz)의 차이, 그리고 순간 힘 한 샘플만으로는 접촉 상태를 알기 어렵다는 점이 핵심 문제다. 또한 시각과 힘을 처음부터 함께 학습하면 정보량이 큰 시각 모달리티가 지배해 힘 신호가 무시되기 쉽다.

## 2. 문제 정의

- 관측 O_t = {다시점 RGB, 상태 s_t (TCP 자세·그리퍼 폭·말단 힘, 13차원), 힘 이력 F_t (h×6)}와 언어 L이 주어질 때 action chunk A_t = {p, g, f}_{t:t+H−1}을 flow matching으로 예측.
- 예측된 자세 x_d와 힘 F_d를 임피던스 모델 K_d(x − x_d) + D_d ẋ + F_d로 실행해, 힘–위치를 동시에 추종하는 것이 목표.

## 3. 방법

- **Historical force-sequence encoder**: 힘 이력을 길이 P 조각으로 나눠 공유 MLP로 임베딩하고 시간 위치 인코딩을 더한 뒤, 학습 가능한 force 토큰이 cross-attention으로 전체를 요약.
- **Cross-modal fusion expert (sparse MoE, 4 experts)**: ForceVLA의 FVLMoE 라우팅을 차용해 vision-language 특징과 실시간/이력 힘 특징을 융합, action expert에 residual branch로 연결.
- **다단계 학습**: Stage I에서 π0를 block-wise attention mask로 fine-tune해 시각·언어·proprio 정렬, Stage II에서 fusion expert를 추가하고 flow-matching 손실로 공동 최적화해 힘 기반 보정을 "시각 정책의 residual"로 학습.
- **VLA-guided adaptive compliance controller (VG-ACC)**: action chunk를 SO(3) slerp로 고주파 보간하고, Resilient Propagation에서 착안한 부호 기반 강성 갱신(K_t = K_{t−1} − α⊙sign(ΔF_err), α ∝ |F_err|) + 저역통과 + [K_min, K_max] 포화, 임계 감쇠 D = 2ζ√K, 안정성 조건 기반 강성 변화율 제한.
- **Shared teleoperation**: 보상 가상 임피던스로 환경을 더 부드럽게 느끼게 해 게임패드/3D 마우스로도 과도한 접촉 없이 시연 수집.

## 4. 구현 세부

- UR5e + 손목 RealSense D435 + 측면 카메라 + UMI 유사 그리퍼 + 6축 F/T 센서.
- VLA는 1초에 1회 추론, action horizon 40, 힘은 0.1초 간격 샘플링, 제어 500 Hz, 강성 400–2500 N/m.
- 시연: 버튼·창문·칠판 과제 각 50개, 플러그 삽입 100개.

## 5. 평가 프로토콜

- 4개 과제: 비상정지 버튼 누르기(PB), 충전 플러그 삽입(PI), 회전창 열기(OW), 일정 힘 칠판 닦기(WB).
- 조건: PB-base, PB-OOD(6 cm 낮춤), PI-base, PI-pro(손목 카메라에서 소켓이 안 보이는 근접 시작), OW-base, WB-base/WB-OOD. 조건당 24–25 trial.
- 기준선: Diffusion Policy, π0, π0.5, π0 + 힘을 상태에 단순 concat, ForceVLA.

## 6. 실험 설계의 요점

성공률 외에 칠판 닦기에서 40 N 목표 힘에 대한 상대 추종 오차를 별도로 측정해 "힘 제어 정밀도"를 평가한 점이 특징이다. 75 N 초과 시 보호 정지가 걸리는 안전 조건도 함께 기록한다.

## 7. 주요 결과

- **전체 성공률(본문)**: CC-VLA 89.2% vs DP 31.3%, π0 47.3%, π0.5 44.7%, π0+F 60.2%, ForceVLA 73.2%. (과제별 수치는 Fig. 6에만 제시)

**Table 1 – 칠판 닦기 힘 추종 오차(%, 낮을수록 좋음)**

| | DP | π0 | π0.5 | π0+F | ForceVLA | CC-VLA |
|---|---|---|---|---|---|---|
| WB-Base | 83.83 | NA | 76.12 | 28.10 | 35.57 | **5.52** |
| WB-OOD | 79.33 | 81.20 | 82.19 | 38.15 | 52.33 | **8.78** |

π0와 π0.5는 과도한 접촉력으로 보호 정지가 발생했다.

**Table 2 – 제어기 절제(WB-Base 오차 %)**: VG-ACC 없음 17.73, 저강성 고정 7.92, 고강성 고정 13.30, 가변 강성 5.52.

**모듈 절제(본문)**: 다단계 학습으로 PB-base 성공률 +12%, 오압(false-pressing) 비율 28% → 8%; PI-base 76%(+8%). 힘 이력 인코더로 PI-pro 72% → 92%.

## 8. Related Work 상의 위치

- ForceVLA(MoE 융합), TA-VLA(힘을 상태/목표로 사용)를 잇되, 힘을 **예측 목표이자 feedforward 제어 입력**으로 삼고 명시적 compliance 제어기와 결합했다.
- Adaptive Compliance Policy, Reactive Diffusion Policy, ImplicitRDP 같은 diffusion 기반 slow–fast 정책을 대규모 VLA 쪽으로 확장한 형태.

## 9. 강점

1. VLA 예측과 고주파 가변 임피던스 제어를 결합해, 추론 지연에도 과도한 접촉력을 제어기 차원에서 억제.
2. 힘 추종 오차라는 제어 관점 지표를 보고해 성공률만으로는 드러나지 않는 품질 차이를 보여 줌(5.52% vs ForceVLA 35.57%).
3. 다단계 학습·힘 이력 인코더·제어기에 대한 개별 절제.
4. 시연 수집(shared teleoperation)부터 배포까지 시스템 전체를 다룸.

## 10. 약점 및 한계

1. 과제별 성공률은 그림에만 있어 표 수치로 검증하기 어렵고, 전체 평균만 본문에 제시.
2. 실험이 단일 로봇(UR5e)·4개 과제·조건당 약 25 trial로 규모가 작다.
3. OOD는 높이 6 cm 이동 등 부분적 변화에 한정.
4. 강성 갱신 규칙이 휴리스틱이며 파라미터 민감도 분석이 없다.
5. 코드/모델 공개 계획이 명시되어 있지 않고, 제목(CR-VLA-Force)과 본문 명칭(CC-VLA)이 달라 혼동 여지가 있다.

## 11. 재현 및 확장 아이디어

- 다른 π0 계열(π0.5, GR00T)에 같은 fusion expert와 제어기를 이식해 일반성 검증.
- 방향별 이방성 강성 예측을 VLA 출력으로 학습(Adaptive Compliance Policy와 비교).
- 손가락 수준 그리퍼 힘 제어와 결합(저자들이 제시한 향후 과제).
- 시뮬레이션 접촉 벤치마크에서 재현 가능한 비교 제공.

## 12. 총평

CR-VLA-Force(CC-VLA)는 force-aware VLA와 고전적인 가변 임피던스 제어를 결합해, 접촉이 많은 과제에서 성공률과 힘 추종 정밀도를 함께 끌어올린 시스템 논문이다. 제어기와 VLA 학습 양쪽의 설계가 실용적이지만, 평가 규모가 작고 과제별 수치가 그림에만 있다는 점이 아쉽다.

**한 문장 요약**: π0에 힘 이력 인코더와 MoE 융합 expert를 다단계로 학습시켜 자세+feedforward 힘을 예측하고, 500 Hz 가변 강성 제어기로 실행해 UR5e 접촉 과제 평균 89.2%, 힘 추종 오차 5.52%를 달성한 compliance VLA.

<!-- VERIFIED: pdf -->
