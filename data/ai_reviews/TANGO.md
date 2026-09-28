# TANGO: Humanoid Navigation in Cluttered Environments with a Whole-Body Vision-Language-Action Model

> **한 줄 요약**: 휴머노이드 내비게이션을 2D 경로 계획이 아닌 **전신(whole-body) 행동 생성** 문제로 보고, 자연어 지시와 egocentric RGB로부터 29-DoF 관절 공간 행동을 직접 예측하는 VLA. 완전 시뮬레이션 데이터(Plan-Edit-Track 파이프라인, 64,633 궤적)로만 학습해 VLNVerse-unseen SR 52.89%, 장애물 증강 장면 SR 43.75%·충돌률 9.90%를 달성하고 Unitree G1에 zero-shot 배포.

- **arXiv**: 2609.09158v1 (2026-09-08, cs.RO)
- **소속**: UC Berkeley, Peking University, Tsinghua University, HKU, Princeton University
- **백본**: Qwen2.5-VL-7B (InternVLA-N1 가중치로 초기화) + flow-matching MM-DiT action expert + SONIC tracker
- **프로젝트**: https://tango-vla.github.io

---

## 1. 배경 및 동기

어수선한 실내에서 휴머노이드가 이동하려면 팔 위치, 몸통 조정, 보폭 조절 등 연속적인 기하 인지 전신 적응이 필요하다. 기존 VLN은 2D waypoint나 이산 행동을 예측해 전신 실행 가능성을 표현하지 못하고, GR00T-N1.6, Ψ0, WholeBodyVLA 같은 휴머노이드 VLA는 하체를 고수준 명령으로 분리해 통과 가능성 추론이 어렵다. RL 기반 충돌 회피(HumanoidPF)는 장기 언어 지시 내비게이션으로 확장하기 어렵다.

## 2. 문제 정의

지시 ℓ, 전면·하향 카메라 RGB 시퀀스, 29-DoF 관절 proprioception q_t가 주어질 때 horizon H의 전신 action chunk를 예측한다. 각 행동은 원하는 관절각 q_d ∈ ℝ²⁹와 base 6D 회전 r_b ∈ ℝ⁶이며, 이를 저수준 motion tracker가 추종한다.

## 3. 시뮬레이션 데이터 생성: Plan-Edit-Track (PET)

- **환경 증강**: VLNVerse(263 장면)·SAGE-3D(1,000 장면)에서 Gemini 2.5 Flash로 저품질 장면을 걸러 578개 원천 장면을 확보, 측면(좁은 통로)·바닥(넘기)·머리 위(숙이기) 장애물 삽입.
- **Plan**: 장애물 인지 비용의 A* 경로, 좁은 통로 검출 시 90° heading 변경으로 옆걸음 유도, SONIC planner로 자연스러운 보행 생성. 2 Hz 렌더링 후 Gemini로 VLN 지시문 생성.
- **Edit**: humanoid potential field 기반 유도력을 SoftMimic식 pseudo-force로 신체 링크에 적용, 바닥 장애물에 대해 발 착지·스윙 높이 조정 → 팔 회피, 넘기, 웅크리기 동작.
- **Track**: SONIC tracker로 물리적 실행 가능성과 무충돌을 검증해 실패 궤적 폐기. 단, 감독 신호로는 추종 결과가 아닌 계획·편집된 참조 동작을 사용해 사람다운 움직임 보존.
- 총 64,633 궤적, 생성 비용 211 RTX PRO 6000 GPU-hour (Table 6).

## 4. 모델 구조: Triple-System

- **System-2**: Qwen2.5-VL-7B, InternVLA-N1 가중치로 warm-start. 전면·하향 뷰를 세로로 쌓아 한 프레임으로 입력, BATS(Budget-Aware Token Sampling)로 긴 이력 샘플링, 최근 프레임에 더 세밀한 grid pooling.
- **System-1**: flow-matching MM-DiT action expert. 안정적 회귀를 위해 첫 프레임 기준 상대 yaw와 chunk 단위 평면 변위 (Δx, Δy, Δψ) 보조 목표를 추가. 학습 시 RTC(real-time chunking)로 확정된 prefix를 조건으로 나머지를 inpaint.
- **System-0**: 기성 SONIC tracker가 참조 동작을 고주파 관절 명령으로 변환.

## 5. 학습

VL backbone과 action expert를 end-to-end로 1 epoch, lr 1e-5로 공동 학습. VideoQA 샘플과 co-training하여 일반 지식 유지: L = L_CE + 20·L_FM.

## 6. 배포

- 시뮬레이션: MuJoCo에서 tracker와 물리를 돌리고 IsaacSim의 디지털 트윈을 순간이동시켜 사실적 관측을 얻는 teleportation 시스템.
- 실제: 서버(RTX PRO 6000)에서 VLA를 0.5초마다(2 Hz) 추론해 30 Hz 기준 15 step chunk 생성 → 로봇으로 전송, 50 Hz로 리샘플 → Jetson Orin NX의 SONIC이 약 200 Hz로 제어. 네트워크 지연 약 20ms.

## 7. 주요 결과

**VLNVerse (Table 1, fine-grained val)** — 베이스라인은 teleportation, TANGO만 물리 제어:

| 방법 | Seen SR | Seen SPL | Unseen NE | Unseen SR | Unseen SPL |
|---|---|---|---|---|---|
| RDP | 47.28 | 41.69 | 3.75 | 48.60 | 42.72 |
| InternVLA-N1 | 51.56 | 34.37 | 4.09 | 45.56 | 34.98 |
| Uni-NaVid | 51.88 | 39.72 | 3.97 | 45.00 | 39.42 |
| **TANGO** | **54.69** | 40.18 | **3.72** | **52.89** | 40.18 |

SR은 최고, SPL은 RDP보다 약간 낮다(저수준 tracker의 보수적 우회 때문으로 해석).

**장애물 증강 VLNVerse-unseen (Table 2)**: TANGO NE 4.01 / SR 43.75 / SPL 31.83 / CR 9.90. 최강 baseline InternVLA-N1 + HumanoidPF(LiDAR 추가 사용)는 SR 41.88 / SPL 29.49 / CR 15.81.

**실제 Unitree G1 (Table 3, 설정당 15 trial)**: 단거리 2D 12/15, 장거리 2D 8/15, 어수선한 3D 10/15 (InternVLA-N1 + Unitree WBC: 11/15, 6/15, 6/15). 시행당 평균 충돌 0.40 / 1.07 / 0.73.

## 8. Ablation

- **행동 공간 (Table 4, VLNVerse-unseen)**: Ours-2D는 teleport에서 SR 45.74지만 Unitree 제어기로 물리 실행 시 26.67로 급락. 전신 TANGO는 52.89 유지 → 전신 행동 예측의 이점.
- **핵심 요소 (Table 5, 증강 unseen)**: RTC 제거 시 SR 10.94(−32.81%p), CR 14.60; motion editing 제거 시 SR 36.25, CR 20.60; SONIC → ScaleBFM 교체 시 SR 40.94, CR 9.10으로 다른 tracker와도 호환.

## 9. 강점

- 내비게이션과 전신 제어를 하나의 VLA에서 29-DoF 관절 공간으로 통합한 최초 시도라는 명확한 기여.
- 인간 모션캡처 없이 확장 가능한 PET 데이터 파이프라인(211 GPU-hour로 6.4만 궤적)과 실제 데이터 없는 zero-shot sim-to-real.
- RGB만으로 LiDAR를 쓰는 모듈식 baseline보다 낮은 충돌률.
- 데이터 파이프라인·데이터셋·체크포인트·배포 시스템 공개 예정.

## 10. 약점 및 한계

- Table 1 비교는 TANGO(물리 실행)와 baseline(teleport) 조건이 달라 엄밀한 동일 조건 비교가 아니다(저자도 명시).
- 실제 평가는 설정당 15회로 작고 baseline이 하나(InternVLA-N1 + Unitree WBC)뿐이다.
- VLA가 2 Hz로 원격 서버에서 동작해 네트워크 의존적이며, 동적 장애물 대응은 다루지 않는다.
- SPL이 RDP보다 낮아 경로 효율은 개선 여지가 있다.

## 11. 재현 및 확장 아이디어

- 공개 예정인 PET 파이프라인으로 다른 휴머노이드(H1, Figure 등)용 데이터 생성.
- 조작(loco-manipulation) 과제로 확장해 문 열기·물체 운반을 포함한 전신 VLA 구축.
- 온보드 경량화(증류, 양자화)로 서버 의존 제거 및 동적 장애물 대응을 위한 고주파 재계획.
- 실제 소량 데이터 co-training으로 SPL(경로 효율) 개선.

## 12. 총평

TANGO는 휴머노이드 VLN을 "어디로 갈지"에서 "몸을 어떻게 움직여 지나갈지"로 확장한 의미 있는 작업이다. 완전 합성 데이터로 학습한 전신 VLA가 VLNVerse에서 최고 SR과 증강 장면에서 가장 낮은 충돌률을 보이고 실제 G1에 zero-shot 전이된 것은 인상적이다. 평가 조건 불일치와 소규모 실제 실험은 한계이지만, 전신 인지 내비게이션 VLA의 기준점이 될 만한 논문이다.

<!-- VERIFIED: pdf -->
