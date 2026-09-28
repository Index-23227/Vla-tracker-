# TrAct: Bridging Robot Control and Visual Prediction with Visual Tracks

> **한 줄 요약**: π0.5의 액션 헤드를 확장해 액션 청크와 2D 포인트 트랙을 함께 생성하는 VLAT 정책, 트랙 조건 비디오 월드 모델(SVD + ControlNet, TWM), VLM 보상 모델(VLAC)을 묶어 "후보 샘플링 → 트랙 조건 롤아웃 → 보상 기반 선택"을 수행하는 프레임워크. LIBERO 평균 98.3%(π0.5 96.8%), 저자 제안 LIBERO-INTEGRAL 0.27 → 0.55, 실물 Franka 미학습 과제 0.49 → 0.76.

- **arXiv**: 2608.24101 (v1 2026-08-25, v3 2026-09-01)
- **소속**: Stanford University, University of Michigan — Zhi Cao, Howard Ji, Kevin Zhang (공동 1저자), Kuangzhi Ge, Li Fei-Fei, Jiajun Wu, Huang Huang
- **프로젝트 페이지**: https://lol-png.github.io/tract/
- **백본**: π0.5 (VLAT), Stable Video Diffusion (TWM), InternVL2 기반 VLAC

---

## 1. 배경 및 동기

월드 모델로 후보 행동의 결과를 예측해 최선을 고르는 방식은 정책의 폐루프 강건성을 높일 수 있지만, 정책과 월드 모델을 잇는 인터페이스가 보통 **로봇 액션**이다. 액션은 embodiment 특화적이고 저차원이며 그 시각적 효과는 장면 기하·접촉에 크게 의존하므로, 액션 조건 비디오 모델은 액션을 무시하고 성공을 "환각"하기 쉽다. 반면 2D 포인트 트랙은 이미지 공간에서 어떤 점이 어떻게 움직이는지를 직접 기술해 조밀한 공간 가이드가 되고, embodiment에 비교적 무관하다. TrAct는 트랙을 제어와 예측 사이의 공유 인터페이스로 쓰자는 제안이다.

## 2. 문제 정의

- 입력: 관측 o_t(agent-view + wrist-view), 언어 지시 l.
- VLAT: π_θ(o_t, l) → {(a_i, τ_i)}_{i=1..K}, a_i ∈ R^{H×d_a}, τ_i ∈ R^{N×H×2}.
- TWM: v̂_i = p_φ(o_t, τ_i), VLAC: i* = argmax R_ψ(v̂_i, l). 선택된 a_{i*}를 실행.

## 3. 방법

- **VLAT**: π0.5 flow-matching 액션 헤드를 수정해 액션과 트랙을 함께 생성, L = L_flow(a) + λ·L_flow(τ). 트랙 포인트는 그리퍼 메시의 고정 3D 오프셋 7점(로봇 포즈·캘리브레이션으로 투영) + wrist-view 5×5 그리드 배경점(CoTracker 추적). agent-view 헤드는 그리퍼 점만, wrist-view 헤드는 그리퍼+그리드 점을 출력하는 비대칭 설계.
- **통합 트랙 슬롯**: 로봇 데이터·인간 egocentric 데이터·카메라 구성이 달라도 고정 슬롯 레이아웃 + 마스크로 같은 토큰 공간에 표현. EgoDex는 트랙 헤드만, 로봇 데이터는 트랙+액션 헤드를 감독.
- **TWM**: SVD에 ControlNet 분기를 붙여 트랙을 공간 제어 맵으로 렌더링(agent-view 빨강, wrist-view 파랑 채널).
- **AWM(비교용)**: 액션을 MLP로 CLIP-L/14 공간에 임베딩하고 view 임베딩과 함께 cross-attention 주입.
- **추론**: Temperature-Scaled Resampling으로 다양한 후보 생성(시뮬 K=20, 실물 K=16), 각 트랙으로 롤아웃, VLAC 점수 최대 후보의 액션 실행.

## 4. 학습 절차

- VLAT 사전학습: DROID 76K + EgoDex 150K 궤적(1:2), 30K step, batch 64, H100 4장.
- 월드 모델(TWM/AWM) 사전학습: DROID 76K, 30K step.
- LIBERO 미세조정: 2K 에피소드, VLAT 5K step(16-step 액션·트랙 청크), 월드 모델 25K step, VLAC 5K step.
- 실물 미세조정: 400 demo(15Hz, 4과제), π0.5·VLAT 50K step(batch 32), 월드 모델 25K step, VLAC 4K step.
- 부록 TrAct+: DROID 76K + BridgeData V2 60K + EgoDex 전체(~300K), 60K step, H100 8장.

## 5. 데이터

- 사전학습: DROID, EgoDex(인간 손 7점×2 + 배경 25점), 추가 실험에서 BridgeData V2(CoTracker로 트랙 추출).
- 시뮬레이션: LIBERO 4개 suite, 저자 구성 **LIBERO-INTEGRAL**(20과제: LIBERO-PRO의 swap/object/task, LIBERO-Plus의 camera/robot-init 최고 난도 각 2과제 + LIBERO-10의 UR5 교체 버전 10과제).
- 실물: Franka Panda, agent+wrist 카메라, 학습 4과제 400 demo → 미학습 5과제 × (학습 배경 / 다른 배경) 평가.

## 6. 실험 설정

- 비교 변형: π0.5, VLAT(선택 없음), VLAT+AWM(액션 조건 월드 모델 선택), TrAct(트랙 조건 월드 모델 선택), 실물에서는 π0.5+VLAC 추가.
- 시뮬레이션: 과제당 10 롤아웃(표준 LIBERO는 suite당 100 에피소드), 단일 시드(INTEGRAL은 부록에서 3 시드 재평가).
- 실물: 과제·배경 조합당 10 에피소드.
- 비디오 품질: 100개 생성 영상으로 PSNR/SSIM/LPIPS/FID/FVD.

## 7. 주요 결과

**Table 2 – 표준 LIBERO (%)**

| Method | Long | Goal | Object | Spatial | Avg |
|---|---|---|---|---|---|
| π0.5 | 94.0 | 96.0 | 98.0 | 99.0 | 96.8 |
| VLAT | 94.0 | 98.0 | 100.0 | 100.0 | 98.0 |
| VLAT+AWM | 94.0 | 98.0 | 100.0 | 100.0 | 98.0 |
| **TrAct** | 94.0 | **100.0** | 99.0 | **100.0** | **98.3** |

**Table 3 – LIBERO-INTEGRAL (성공률)**

| Method | Swap | Object | Task | Camera | RobotInit | Cross-Emb. | Avg |
|---|---|---|---|---|---|---|---|
| π0.5 | 0.40 | 0.35 | 0.15 | 0.45 | 0.45 | 0.17 | 0.27 |
| VLAT | 0.50 | 0.40 | 0.35 | 0.45 | 0.55 | 0.42 | 0.44 |
| VLAT+AWM | 0.65 | 0.50 | 0.50 | 0.45 | 0.55 | 0.44 | 0.49 |
| **TrAct** | **0.65** | **0.60** | **0.50** | **0.60** | **0.65** | **0.50** | **0.55** |

3 시드 재평가(Table 6): TrAct 0.547 (95% CI [0.53, 0.56]) vs VLAT+AWM 0.490 ([0.47, 0.51]) — 구간 비중첩. 후보 수 K=5/10/20에서 0.52/0.54/0.55(Table 7).

**Table 1 – 비디오 예측 품질**: TWM이 네 설정 모두 다섯 지표에서 AWM보다 우수. 시뮬 agent-view PSNR 15.12 → 24.51, LPIPS 0.438 → 0.106; 실물 wrist-view LPIPS 0.372 → 0.225, FVD 246 → 140.

**Table 4 – 실물 Franka, 미학습 5과제 × 2배경 평균**: π0.5 0.49, π0.5+VLAC 0.65, VLAT 0.55, VLAT+AWM 0.66, **TrAct 0.76**. 다른 배경의 캐비닛 닫기에서 AWM 0.4 vs TrAct 0.7.

**Table 8 – TrAct+(더 큰 사전학습)**: 가장 어려운 UR5 5과제 평균 π0.5 0.06, TrAct 0.32, TrAct+ 0.38.

## 8. Related Work 상의 위치

- 액션/언어/멀티모달 조건 embodied 비디오 월드 모델(UniSim, Pandora 등)과 달리 트랙으로 조건화.
- 예측된 트랙·광류를 정책 중간표현으로 디코딩하는 계열과 월드 모델 조건으로 쓰는 계열을 결합하되, 예측된 미래를 VLM 보상으로 과제 지시와 대조해 평가하는 점이 차별점.
- 월드 모델 기반 추론 시 향상(inference-time enhancement via predictive world modeling), VLMPC, VLAC의 폐루프 선택 흐름을 flow-matching VLA에 적용.

## 9. 강점

1. **명확한 가설과 통제 비교**: 동일 VLAT 후보에 대해 트랙 조건(TWM) vs 액션 조건(AWM)만 바꿔 비교해 인터페이스 효과를 분리.
2. 비디오 품질·폐루프 성공률 두 축 모두에서 일관된 우위, 특히 배경 변화(시각 도메인 이동)에서 격차가 커진다.
3. 트랙 공동 예측만으로도(VLAT) π0.5 대비 강건성 향상(INTEGRAL 0.27 → 0.44).
4. 인간 egocentric 데이터를 트랙 감독으로 흡수하는 통합 슬롯 설계, 사전학습 확장 시 추가 향상(TrAct+).
5. 3 시드 신뢰구간 보고로 INTEGRAL 결과의 통계적 신뢰도 보강.

## 10. 약점 및 한계

1. **추론 비용**: 매 결정마다 K개(최대 20) 비디오 확산 롤아웃 + VLM 점수화가 필요하지만 지연/처리량 수치가 본문에 보고되지 않는다.
2. **평가 규모가 작음**: LIBERO는 과제당 10 롤아웃·단일 시드, INTEGRAL은 과제당 10 에피소드(성공률이 0.1 단위), 실물은 조합당 10 에피소드.
3. **자체 벤치마크 의존**: LIBERO-INTEGRAL은 저자가 과제를 선별(카테고리당 2과제)해 구성했으며, LIBERO-PRO/Plus 전체 결과는 없다.
4. 표준 LIBERO 향상(96.8 → 98.3)은 포화 영역이며 Long suite는 모든 변형이 94.0으로 동일.
5. VLAT·월드 모델·보상 모델 세 개를 각각 학습·미세조정해야 해 파이프라인이 무겁다. 코드 공개 여부 미명시.

## 11. 재현 및 확장 아이디어

- 롤아웃 길이·해상도를 줄인 경량 TWM이나 잠재 공간 월드 모델로 지연 절감, K 대비 비용 곡선 보고.
- LIBERO-PRO/LIBERO-Plus 전체 스위트와 SimplerEnv, RoboTwin 등 공용 벤치마크에서 검증.
- 트랙 예측 헤드를 인간 비디오만으로 사전학습한 뒤 새로운 embodiment에 zero-shot 이식하는 실험.
- VLAC 대신 학습된 가치 함수 또는 진행도 모델로 선택기를 교체해 선택 품질 비교.

## 12. 총평

TrAct는 "정책과 월드 모델 사이를 액션이 아닌 시각 트랙으로 잇자"는 명확한 아이디어를 통제된 AWM 비교로 설득력 있게 검증한다. 트랙 공동 예측 자체가 VLA의 강건성을 높이고, 트랙 조건 월드 모델은 비디오 예측과 행동 선택 모두를 개선한다. 다만 추론 비용이 크고 평가가 소규모·자체 구성 벤치마크 중심이라 실용성과 일반성 판단에는 추가 검증이 필요하다.

**한 문장 요약**: π0.5 기반 액션+트랙 공동 예측 정책(VLAT)과 트랙 조건 비디오 월드 모델·VLM 보상 선택을 결합해 LIBERO 98.3%, LIBERO-INTEGRAL 0.55(π0.5 0.27), 실물 미학습 과제 0.76(π0.5 0.49)을 달성한 월드 모델 기반 의사결정 프레임워크.

<!-- VERIFIED: pdf -->
