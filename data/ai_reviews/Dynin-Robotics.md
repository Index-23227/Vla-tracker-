# Dynin-Robotics — Omnimodal Unified Diffusion Vision-Language-Action Model

- arXiv: 2609.13053 (v1, 2026-09-11)
- 저자/소속: Hoeun Lee, Jaeik Kim, Jusang Oh, Jinhyeok Kim, Geon Choi, Hyeonggeun Kim, Jaeyoung Do — AIDAS Lab, Seoul National University
- 코드: https://github.com/AIDASLab/Dynin-Robotics · 모델: https://huggingface.co/snu-aidas/Dynin-Robotics

## 1. 한 줄 요약

옴니모달 masked-diffusion 모델 Dynin-Omni를 로봇 행동 토큰으로 확장해, 하나의 모델이 Policy·World Modeling(행동 조건 다음 관측 예측)·Goal-State Prediction·Task Understanding을 마스킹 패턴만 바꿔 수행하도록 학습한 통합 VLA. LIBERO 평균 98.1%, zero-shot LIBERO-Plus 73.0%, Franka Research 3 실로봇 4조건 평균 78.4%.

## 2. 문제 설정

VLM 기반 정책(언어 조건화에 강함)과 비디오/월드모델 기반 정책(시각 예측 활용)은 과제·지시문 형태에 따라 강점이 다르다. 저자들은 VLABench 2과제 진단에서 π0.5가 SelectFruit에, Mimic-Video가 InsertFlower Track 4에 상대적으로 강하고, 지시문을 무작위 문자열로 바꾸면 π0.5는 16~30%p 떨어지지만 Mimic-Video는 거의 변하지 않음을 보인다(Table 4). 이를 근거로 언어 조건화와 시각 예측을 하나의 공유 모델 안에서 함께 다루는 것을 목표로 한다.

## 3. 핵심 아이디어

지시문·관측·목표 이미지·행동을 하나의 타입화된 궤적 토큰 시퀀스로 두고, objective 토큰으로 어떤 구간을 보이게 하고 어떤 구간을 복원할지 지정한다. 같은 모델이 (a) 행동만 디코딩, (b) 행동+다음 상태 동시 디노이징, (c) 목표 상태 예측 후 정책 디코딩(기본값), (d) 월드모델로 행동 후보 재순위, (e) 목표 유도 + 동시 디노이징, (f) 목표 유도 + 후보 재순위의 6가지 추론 조합을 지원해 test-time scaling을 구현한다.

## 4. 아키텍처

- 백본: Dynin-Omni(단일 양방향 Transformer, 공유 임베딩, 통합 범주형 예측 헤드). 텍스트 토크나이저와 MAGViT-v2 이미지/비디오 토크나이저는 동결.
- 행동 토큰화: 학습된 토크나이저 없이 차원별 균일 binning(Stage 1은 256 bin, Stage 2는 과제별 선택, VLABench는 32 bin). 공통 7D EEF 인터페이스(3D 이동, 3D 회전, 그리퍼 절대값).
- 센서 토큰 블록은 예약만 되어 있고 보고된 실험에서는 비활성. 음성 블록도 상속되었으나 비활성.
- 파라미터 수는 논문에 명시되지 않음.

## 5. 학습과 추론

- Stage 1: Dynin-Omni에서 출발해 48개 OXE 데이터셋(1,332,985 궤적, 65,217,081 transition)으로 연속 사전학습. 타깃 전용 masked-diffusion 손실 + Policy 목적에만 expected-bin MAE 보조항(β=0.2). 두 구간 21.8k + 5.8k = 27.6k optimizer update(220.8k micro-step), 두 번째 구간 가중치 (λPO, λWM, λTU, λGP) = (1, 0.2, 0.05, 0.2). AdamW peak LR 2e-5, 32-rank 실행.
- Stage 2: 도메인별(LIBERO, VLABench, 실로봇) SFT. Policy-only 또는 4목적 혼합. 시연 메타데이터(action-chunk jerk, 궤적 길이, 성공 여부)를 텍스트로 조건화.
- 선택적 온라인 RL: 백본과 토크나이저는 동결하고 경량 remasking 컨트롤러(개수·위치 헤드)만 PPO로 학습해 성공률과 디노이징 비용을 절충.
- 가속: dInfer 기반 block-parallel 디코딩, 컨텍스트 캐싱, CUDA Graph로 B200에서 effective TPS 9.221 → 91.236(BL7, 9.89×) → 268.834(BL35, 29.15×).

## 6. 주요 결과 — LIBERO / LIBERO-Plus (Table 2, 3)

| Spatial | Object | Goal | Long | 평균 |
|---|---|---|---|---|
| 98.9 | 99.8 | 97.8 | 95.8 | 98.1 |

X-VLA(98.1)와 동률, ABot-M0(98.6)·Cosmos Policy(98.5)·LingBot-VA(98.5)보다 낮다. LIBERO 체크포인트를 추가 학습 없이 LIBERO-Plus에 적용한 결과: Camera 59.8, Robot 48.2, Language 85.0, Light 83.5, Background 84.6, Noise 78.2, Layout 71.8, 평균 73.0(ABot-M0 81.6, OpenVLA-OFT 71.4).

## 7. 실로봇 결과 (Table 5, Franka Research 3)

| 방법 | Fruit PnP | Cube Sort | Cube Stack | Color-Ordered Stack | 평균 |
|---|---|---|---|---|---|
| π0.5 | 96.5 | 83.5 | 63.5 | 64.0 | 76.9 |
| GR00T-N1.6 | 91.0 | 78.0 | 59.5 | 63.5 | 73.0 |
| Cosmos Policy | 98.0 | 87.0 | 67.5 | 42.0 | 73.6 |
| Mimic-Video | 96.5 | 84.5 | 64.0 | 39.5 | 71.1 |
| **Dynin-Robotics** | 97.5 | 85.5 | 62.0 | **68.5** | **78.4** |

평균 우위(π0.5 대비 +1.5%p)는 주로 색 순서가 지정된 쌓기에서 나온다. 개별 조건 3개는 Cosmos Policy가 더 높다.

## 8. 어블레이션 (VLABench, 70k Stage-2 step)

- OXE 연속 사전학습(Table 10): 없으면 ID 3.26 / OOD 0.70, 있으면 49.61 / 47.28.
- 목적 혼합(Table 11): Policy-only 47.15 / 33.88 → 전체 4목적 49.61 / 47.28 (OOD +13.40%p, gap 13.27 → 2.33).
- 추론 조합(Table 12): 행동만 45.8 / 41.4, 기본값 (c) 45.6 / 38.2, (e) 49.6 / 47.3, (f) 48.9 / 47.7(TPS 4.828로 약 절반). 목표 유도나 동시 디노이징을 단독으로 쓰면 OOD가 오히려 떨어지고, 둘을 결합해야 이득이 난다.
- Action bin(Table 13): 16/32/64 bin → ID 27.84 / 49.61 / 32.90.
- 보조 손실 스케일(Table 15): α=2에서 ID 38.05 / OOD 36.70으로 크게 하락.

## 9. 월드모델·목표 예측 품질 (Table 6, 7, DROID 검증셋, Stage-1)

다음 프레임 예측 PSNR 20.79 / SSIM 0.573 / LPIPS 0.299로 학습 모델 중 PSNR·SSIM 최고지만, 단순 last-frame 복사(31.07 / 0.953 / 0.013)보다 모든 지표에서 낮다. 목표 상태 예측은 BAGEL이 세 지표 모두 앞선다. 오프라인 행동 Acc@0.1 0.3327, MAE 0.2109.

## 10. 강점

- 네 가지 목적을 하나의 토큰 인터페이스로 묶고, 추론 시 조합을 바꾸는 방식으로 test-time scaling을 체계적으로 비교했다(체크포인트 고정/학습 레시피 고정을 분리한 설계).
- 사전학습·목적 혼합·추론 조합·bin 수·손실 스케일까지 폭넓은 어블레이션과, 지연 시간/성공률 트레이드오프(Table 14: BL7 383.6 ms에서 97.7%)를 함께 보고한다.
- 코드와 모델을 공개했다.

## 11. 한계

- 기본 추론 경로 (c)는 VLABench에서 행동만 디코딩보다 OOD가 낮은데도 LIBERO 보고에 사용되었고, VLABench에서 좋은 (e)/(f)는 LIBERO에 적용되지 않았다.
- 월드모델 예측이 frame copy보다 못하며, Task Understanding은 정성 평가만 있다.
- 기본 디코더 지연이 청크당 3.8초로 매우 크고, 가속 후에도 BL7 383.6 ms.
- VLABench 결과는 2과제뿐이고, 실로봇은 단일 플랫폼·단일 작업공간. 파라미터 수와 Stage-2 학습 스텝(LIBERO) 등 일부 세부가 명시되지 않았다. 교차 모델 표는 인용·재현 수치가 섞여 있다고 저자들이 밝힌다.

## 12. VLA-Tracker 관점 평가

Dynin-Omni에서 출발해 OXE 연속 사전학습과 도메인별 SFT로 자체 정책을 학습했으므로 수록 대상이다. 행동 헤드는 이산 토큰 masked diffusion이라 discrete_diffusion으로 분류했다. 트래커에는 LIBERO 4 suite와 논문 보고 평균 98.1, zero-shot LIBERO-Plus 7개 범주와 평균 73.0을 libero 블록에, 실로봇 4조건과 평균 78.4를 real_world 블록에 기록하고, VLABench 어블레이션·디코딩 가속·DROID 예측 품질은 별도 블록으로 분리했다. LIBERO 순위표에서는 X-VLA와 동률인 상위권이며, masked-diffusion 통합 모델(MMaDA-VLA 98.0, UD-VLA 92.7) 중에서는 가장 높다.

<!-- VERIFIED: pdf -->
