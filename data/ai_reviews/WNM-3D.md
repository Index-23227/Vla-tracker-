# WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN

- arXiv: [2608.07267](https://arxiv.org/abs/2608.07267) (2026-08-07, cs.AI)
- 저자: Yuehao Huang†, Yunzi Wu†, Xiaotao Zhang†, Xinhai Li‡, Jiankun Dong, Jiajun Lv, Chi Zhang, Chenjia Bai*, Yong Liu*, Xuelong Li* (†Equal Contributions, ‡Project Leader, *Corresponding Authors)
- 소속: Institute of Artificial Intelligence (China Telecom), Zhejiang University, Tongji University, Shanghai Jiao Tong University

---

## 1. 한 줄 요약

연속 VLN(vision-language navigation)에서, 관측 히스토리로부터 뽑은 **기하 인지(geometry-aware) 씬 토큰**을 미래 시점 예측과 행동 생성 **양쪽이 공유하는 조건(prefix)** 으로 넣은 생성형 world-action 모델. GN-Bench Seen에서 SR 81.3% / SPL 78.3%, Unseen에서 SR 46.8% / SPL 43.5%를 기록해, 동일 백본에 RGB 히스토리 prefix만 쓴 통제 변형(WNM-2D)과 기존 VLN 정책들을 모두 앞선다.

## 2. 문제 정의

최근 VLN은 사전학습 VLM을 행동 생성 정책(VLA)으로 적응시키는 방향으로 수렴하고 있다. 이들은 지시 이해·랜드마크 그라운딩에 강한 의미 사전지식을 물려받지만, **행동 예측만으로 최적화되기 때문에 자기가 낸 행동에 따라 관측이 어떻게 변해야 하는지를 명시적으로 모델링하지 않는다**. 연속 내비게이션에서는 모든 행동이 시점을 바꾸고, 바뀐 시점이 다음 결정의 증거가 되므로 이 누락이 치명적이다. 행동 지도만으로는 이 폐루프 관측-행동 진화에 아무런 예측적 제약을 걸지 못한다.

생성형 world model은 상호작용에 따른 관측 변화를 예측하는 상보적 관점을 제공한다. 특히 world-action model(WAM)은 미래 관측과 실행 가능한 행동을 결합된 예측 시스템 안에서 함께 모델링한다. 그러나 연속 VLN용 기존 WAM들은 **관측 히스토리로부터 추론된 기하 인지 표현에 미래 뷰·행동 결합 생성을 조건화하지 않는다**. RGB 히스토리나 비디오 잠재는 국소 외형과 시간적 맥락은 보존하지만, 누적된 관측 히스토리를 지속적인 기하 씬 컨텍스트로 통합할 명시적 메커니즘이 없다. 결과적으로 예측 궤적과 그에 대응하는 시각 전이 사이의 대응 관계가, 특히 큰 시점 변화에서 과소 제약된다.

저자들의 핵심 질문: 단안 관측 히스토리는 에이전트가 움직이면서 같은 환경을 여러 시점에서 담고 있는데, **여기서 복원한 기하 정보를 미래 뷰 예측과 행동 생성 양쪽의 공유 컨텍스트로 어떻게 넣을 것인가?**

## 3. 제안 방법 개요

WNM-3D는 DreamZero의 joint video-action flow 백본 위에 두 가지를 얹는다.

1. **동결된 feed-forward 기하 인코더 VGGT-Omega**: 단안 egocentric RGB 히스토리에서 교차 시점 특징 추출.
2. **학습 가능한 3D Scene-to-Token Adapter**: 그 특징을 world-action DiT의 은닉 폭·토큰 레이아웃에 맞는 **고정 길이 토큰 시퀀스**로 변환.

이 토큰들은 denoising 스트림 앞에 prepend되고, block-causal attention을 통해 모든 미래 video-action 블록에 계속 보인다. 즉 씬 컨텍스트가 joint denoising 전 과정에서 두 모달리티를 동시에 안내한다. 고정 토큰 인터페이스 덕분에 upstream 씬 표현과 downstream world-action 생성기가 분리된다.

비교용 통제 변형 **WNM-2D**는 백본·block-causal attention·예측 타깃·3단계 학습 절차를 전부 공유하되, 기하 유래 prefix를 백본 native VAE 인코딩 RGB-히스토리 prefix로 바꾼 것이다. 두 변형 모두 prefix 길이가 450 토큰(9x5x10)으로 동일하다.

## 4. 문제 정식화와 Block-Causal Attention

재계획 스텝 t에서 에이전트는 지시 l과 K개 단안 egocentric RGB 관측 히스토리 H_t를 받는다. 모델은 B개 미래 시각 블록 Y_t(각 N_v 프레임)과 시간 정렬된 행동 블록 A_t(각 N_a 행동)를 예측한다.

행동은 I_t의 카메라 자세에 고정된 로컬 내비게이션 프레임에서 정의된 **연속 샘플 간 증분** a_j = (dx_j, dy_j, dpsi_j)이다. 평행이동 성분은 이 프레임의 고정 좌표축을 공유하고, dpsi는 평활화된 궤적의 연속 접선 헤딩에서 계산된다. 시뮬레이터 자세는 정책에 노출되지 않고 오직 행동 타깃 구성·실행·보상 계산에만 쓰인다.

깨끗한 히스토리 조건 prefix C_t는 두 변형이 다르게 만든다. WNM-3D는 C_t = T_phi(E_geo(H_t))로, WNM-2D는 native VAE 경로로. 병렬로 현재 프레임과 언어 지시는 백본 native cross-attention 경로로 G_t = [E_img(I_t); E_l(l)]로 인코딩된다.

**Block-causal 마스크**는 직렬화된 토큰 위치가 아니라 **시간 블록 단위로** 인덱싱된다. prefix는 자기 자신만 attend, block-zero 시각 그룹은 prefix와 자기 자신, 블록 b의 시각·행동 토큰은 prefix + block zero + 블록 b까지의 모든 video-action 그룹을 attend한다. 즉 **블록 안에서는 시각-행동이 양방향, 블록 간에는 causal**이다.

## 5. Joint World-Action Flow Matching

DreamZero의 joint flow-matching 정식화를 채택해 시각·행동 변수를 각자의 연속 공간에서 공유 DiT로 모델링한다. 선형 conditional-flow 경로 x_sigma = (1-sigma) x_data + sigma eps, u_sigma = eps - x_data를 쓰고, 목적함수는

L_WA = E[ MSE_w(u_hat_v, u_v) + lambda_a MSE_{w,Ma}(u_hat_a, u_a) ]

여기서 MSE_w는 flow-timestep 의존 가중치를 블록별로 적용하고, MSE_{w,Ma}는 zero-padding된 행동 차원을 마스킹한다. lambda_a = 1(시각·행동 동일 계수). 미래 시각 그룹과 정렬된 행동 그룹은 **같은 flow timestep sigma_b를 공유하되 노이즈는 독립 샘플링**된다.

행동은 flow matching 전에 s_a = 4로 스케일하고 학습셋 분위수로 정규화한 뒤 폭 d_a = 32로 zero-padding된다. 추론 시 앞 3차원에만 역변환을 적용한다.

## 6. 3D Scene-to-Token Adapter

동결 VGGT-Omega가 L개 패치 특징 그리드(K x H_s x W_s x d_l)를 내놓으면, 어댑터가 4단계로 고정 길이 prefix를 만든다.

**(a) 인코더 레벨 융합**: 각 선택 레벨을 공통 폭 d_s로 정규화·투영. 선택된 그리드들이 동일한 history-height-width 격자를 공유하므로 추가 resampling 없이 같은 위치에서 정렬된다. 경량 게이팅 네트워크가 위치 적응적 가중치를 레벨 간에 소프트맥스로 할당해 융합 source memory F_t를 만든다.

**(b) 쿼리 형성**: source memory를 타깃 격자 T_c x H_c x W_c로 adaptive pooling해 씬 base B_t를 얻고, 학습된 슬롯 Q_slot과 구조 임베딩 E_struct(전역·시간·행·열·타깃 좌표 Fourier·쿼리 타입)를 더해 Q_0_t 초기화.

**(c) Anchored deformable resampling (R=2층)**: pooled base는 거친 컨텍스트는 보존하지만 세밀한 증거를 버릴 수 있다. 각 타깃 쿼리에 source 격자 위 정규 앵커를 대응시키고, 쿼리 의존 오프셋과 집계 가중치를 예측해 앵커 주변 소규모 이웃을 검색한다. 전체 source memory에 대한 global cross-attention 없이 세밀한 특징 집계가 가능하다.

**(d) Factorized 시공간 정제 (2블록) + 출력**: 각 블록은 시간 슬라이스 내 spatial self-attention → 각 공간 위치에서 temporal self-attention → MLP residual. detail head가 정제된 쿼리를, coarse head가 pooled base를 residual로 투영한다.

**최종 설정(Table 4)**: 백본 초기화 Wan2.2-TI2V-5B, K=33 히스토리 프레임, world-action 입력 해상도 160x320, 기하 인코더 입력 512x512, B=4, N_v=8, N_a=8(총 32 프레임 / 32 행동). 선택 VGGT-Omega 블록은 5, 12, 18, 24(L=4), d_s=512, source memory 33x32x32, 타깃 격자 9x5x10, prefix 길이 450, resampling head 8 x point 8, **학습 가능 어댑터 파라미터 23.97M**.

## 7. 3단계 폐루프 학습

**Stage I - Offline A* SFT**: A* expert planner가 생성한 16K 지시 조건 궤적으로 L_WA 최적화. 시뮬레이터가 같은 궤적을 따라 미래 시각 연속체를 렌더링. 20 epochs. WNM-3D는 어댑터와 백본을 함께 학습하고 VGGT-Omega는 동결.

**Stage II - Closed-loop DAgger-SFT**: Stage-I 정책을 같은 16K 태스크에서 롤아웃하고, 정책이 방문한 상태마다 A* expert가 교정 행동을 제공하며 시뮬레이터가 그에 대응하는 시각 연속체를 렌더링(궤적 일관 시각-행동 지도). WNM-2D는 약 691K 청크, **WNM-3D는 약 633K 청크**. 5 epochs.

Stage I/II 공통: AdamW, lr 1e-5, cosine + 5% warmup, weight decay 1e-5, grad clip 1.0, BF16, **16x H100**, global batch WNM-2D 256 / WNM-3D 192(기하 인코딩 때문에 축소).

**Stage III - Closed-loop DanceGRPO**: Stage-II 체크포인트에서 시작, 약 96K 정책 생성 시뮬레이션 레코드 사용. 16-step 샘플러의 적격 denoising 전이를 K_s = 4개 stratum {0,1,2}, {6}, {10}, {13,14,15}으로 분할하고, 각 stratum에서 전이 하나를 뽑아 **초기 잠재와 tau_k 외 모든 외생 SDE 증분을 공유하는 counterfactual 2-branch**를 생성한다. 조건당 2K_s = 8개 완전 롤아웃. 보상은 완성된 롤아웃에서 평가하되 gradient는 개입된 전이에서만 replay.

쌍별 rank advantage(Eq. 10)는 두 branch의 불편 표준편차로 정규화되므로 비동률 advantage가 ±1/sqrt(2)에 근접한다 — 즉 **cardinal 보상 차이가 아니라 쌍 내부 순서를 보존하는 rank 신호**다.

보상은 3 스트림: **visual**(pyramid-averaged SSIM 0.60 + Charbonnier 재구성 0.25 + 시간 일관성 0.15에 degradation factor D_deg를 곱한 base + flow-action 일관성 보너스), **navigation**, **stopping**. 시각 보상은 r_v로, 청크 c의 내비·정지 보상은 r_a_c로 라우팅되는 **modality-routed surrogate**(정확한 joint transition likelihood ratio가 아님을 저자들이 명시). 최종 목적 L_III = L_v + lambda_act (L_nav + lambda_stop L_stop), lambda_act = 0.25, lambda_stop = 0.50, 내비 청크 가중치 (8,4,2,1)/15로 초기 청크 강조(receding-horizon이 A_t,1만 실행하므로). 시각 SDE 계수 eta_v = 0.70, 행동 eta_a = 0.20, clipping eps_v = 5e-4 / eps_a = 0.01, lr 5e-6, 1,500 updates, CFG scale 5.

**Receding-horizon 추론**: 매 재계획 스텝에서 I_0~I_t 사이를 균등 샘플링해 H_t를 구성하고 prefix를 재계산. 첫 행동 블록 A_t,1만 실행한 뒤 새 관측으로 재계획. **예측된 시각 잠재는 RGB로 디코딩되지도, H_t에 편입되지도 않는다**(즉 world 예측은 학습·보상 신호로만 기능).

## 8. 실험 설정

**벤치마크**: GN-Bench 공식 Seen / Unseen 스플릿(각 **1,000 / 5,000 에피소드**). 지표는 NE(낮을수록 좋음), OS, SR, SPL(높을수록), TL. 평가자는 반환된 첫 8개 행동까지 실행 후 재계획하며, 그 블록의 총 평행이동이 0.15m 미만이면 STOP을 발행한다. 각 체크포인트는 에피소드당 1회, 추론 시드 1140으로 평가. **다중 시드 표준편차나 부트스트랩 신뢰구간은 계산하지 않는다**(저자 명시).

**베이스라인**: CMA, NaVid, UniNaVid, InternNav(S2), GN-BAE 및 GN-Matrix SFT 변형. WNM-2D는 아키텍처 통제군.

**Flow-action 일관성**: 640개 near-goal STOP 스냅샷 고정 세트에서, 실행 블록의 누적 XY 변위와 시간 정렬 생성 프레임에서 추론된 카메라 모션을 비교. 9프레임을 64x128 그레이스케일로 바꿔 8쌍의 DIS forward/backward flow를 계산하고, 4x6 격자 median flow + p50/p75/p90 통계 descriptor를 480개 GT 클립으로 학습한 ridge regressor로 (dx, dy, dpsi)에 매핑. flow-valid ratio는 두 변형 모두 0.9984 이상.

## 9. 주요 결과

**Table 1 — GN-Bench 폐루프 내비게이션** (관측 공간: Depth / BEV / FPV)

| Method | Obs | Seen TL | Seen NE↓ | Seen OS↑ | Seen SR↑ | Seen SPL↑ | Unseen TL | Unseen NE↓ | Unseen OS↑ | Unseen SR↑ | Unseen SPL↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CMA | Depth+FPV | 2.9 | 8.3 | 15.7 | 12.5 | 11.9 | 3.0 | 8.1 | 19.6 | 15.5 | 14.9 |
| NaVid | FPV | 3.4 | 7.9 | 20.1 | 14.6 | 12.8 | 3.5 | 7.7 | 20.3 | 14.5 | 12.8 |
| UniNaVid | FPV | 4.6 | 7.9 | 22.2 | 15.0 | 12.5 | 5.2 | 7.8 | 20.7 | 12.8 | 10.3 |
| InternNav(S2) | FPV | 3.6 | 7.4 | 23.1 | 18.8 | 17.5 | 3.7 | 7.2 | 26.7 | 22.1 | 20.3 |
| NaVid† | FPV | 2.7 | 7.4 | 19.4 | 18.8 | 18.8 | 2.7 | 7.1 | 23.8 | 23.1 | 23.0 |
| UniNaVid† | FPV | 3.7 | 7.2 | 24.1 | 22.5 | 21.9 | 5.8 | 7.5 | 23.1 | 20.8 | 20.2 |
| InternNav(S2)† | FPV | 2.9 | 7.1 | 22.5 | 22.4 | 22.4 | 2.9 | 6.9 | 24.9 | 24.0 | 23.7 |
| GN-BAE | Depth | 5.2 | 4.9 | 48.9 | 46.4 | 44.7 | 5.0 | 5.6 | 43.6 | 38.9 | 37.3 |
| GN-BAE | Depth+BEV | 5.2 | 4.3 | 59.3 | 58.6 | 58.6 | 4.0 | 5.8 | 40.2 | 38.5 | 38.2 |
| WNM-2D | FPV | 7.9 | 2.8 | 80.7 | 75.6 | 72.9 | 7.9 | 4.8 | 52.6 | 45.9 | 42.8 |
| **WNM-3D** | FPV | 8.1 | **2.0** | **87.2** | **81.3** | **78.3** | 7.7 | **4.4** | **54.1** | **46.8** | **43.5** |

(† = GN-Matrix 데이터셋으로 SFT된 변형)

- Seen에서 BEV를 추가로 쓰는 최강 선행 방법(GN-BAE Depth+BEV) 대비 **SR +22.7%p, SPL +19.7%p**.
- Unseen에서 최강 FPV-only 선행 베이스라인 대비 **SR +7.9%p, SPL +6.2%p**.
- 두 WNM 변형 모두 추론 시 **단안 FPV RGB만** 사용하며, WNM-3D는 depth·BEV·명시적 metric map 없이 RGB 히스토리에서 내부적으로 기하 토큰을 유도한다.

**기하 조건화 효과(WNM-3D vs WNM-2D)**: 동일 백본·동일 3단계 레시피에서 Seen **+5.7 SR / +5.4 SPL**, Unseen **+0.9 SR / +0.7 SPL**. 저자들은 "Seen에서는 명확한 이점, 씬 수준 분포 변화에서는 더 작은 양의 이득"이라고만 쓰고, **Seen→Unseen 저하 완화라는 더 강한 주장은 현재 결과가 지지하지 않는다**고 스스로 못박는다.

## 10. 어블레이션과 Flow-Action 일관성

**Table 2 — 학습 커리큘럼 어블레이션**

| Model | A* SFT | DAgger | DanceGRPO | Seen NE↓ | Seen OS↑ | Seen SR↑ | Seen SPL↑ | Unseen NE↓ | Unseen OS↑ | Unseen SR↑ | Unseen SPL↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WNM-3D | ✓ | | | 4.4 | 52.0 | 49.6 | 49.1 | 5.3 | 42.6 | 39.7 | 38.9 |
| WNM-3D | ✓ | ✓ | | 2.3 | 86.1 | 80.6 | 77.9 | 4.7 | 53.1 | 45.7 | 41.9 |
| WNM-3D | ✓ | | ✓ | 5.8 | 40.2 | 39.4 | 38.5 | 5.8 | 38.7 | 35.6 | 33.8 |
| WNM-3D | ✓ | ✓ | ✓ | 2.0 | 87.2 | 81.3 | 78.3 | 4.4 | 54.1 | 46.8 | 43.5 |
| WNM-2D | ✓ | | | 5.2 | 40.3 | 38.8 | 38.6 | 5.6 | 37.5 | 35.9 | 35.6 |
| WNM-2D | ✓ | ✓ | | 2.9 | 82.3 | 71.9 | 67.0 | 5.0 | 52.8 | 42.7 | 38.3 |
| WNM-2D | ✓ | | ✓ | 5.6 | 35.6 | 34.0 | 33.7 | 5.9 | 34.7 | 32.8 | 32.4 |
| WNM-2D | ✓ | ✓ | ✓ | 2.8 | 80.7 | 75.6 | 72.9 | 4.8 | 52.6 | 45.9 | 42.8 |

- **DAgger가 압도적 기여**: WNM-3D Seen SR 49.6 → 80.6, SPL 49.1 → 77.9. Unseen SR 39.7 → 45.7, SPL 38.9 → 41.9.
- **DanceGRPO는 추가 정제**: Stage II 대비 WNM-3D Seen +0.7 SR / +0.4 SPL, Unseen +1.1 SR / +1.6 SPL. WNM-2D는 같은 경향이되 더 큰 폭(Seen +3.7 SR / +5.9 SPL, Unseen +3.2 SR / +4.5 SPL).
- **DAgger 없이 DanceGRPO를 직접 적용하면 성능이 오히려 떨어진다**: WNM-3D Seen SR 49.6 → 39.4, Unseen 39.7 → 35.6. 저자 가설은 Stage-I 정책이 expert 상태 밖에서 좁고 오류가 많은 롤아웃 분포를 만들며, group-relative 최적화는 그 제한된 support에서 뽑힌 후보만 순위 매길 수 있어 후보들이 일률적으로 나쁘거나 변별력이 없으면 rank 신호가 무의미해진다는 것. 다만 **within-group 행동 다양성·보상 분산의 직접 측정은 하지 않았다고 스스로 인정**한다.

**Table 3 — Flow-action 일관성 (고정 near-goal STOP 세트, GT 모션 크기 약 2.3553e-4)**

| Model | Stage | S_flow-act ↑ | E_motion ↓ | R_action ↑ |
|---|---|---|---|---|
| WNM-2D | I | 0.3174 | 0.0300 | -0.2194 |
| WNM-2D | II | 0.3507 | 0.0364 | -0.1601 |
| WNM-2D | III | 0.3609 | 0.0289 | -0.1440 |
| WNM-3D | I | 0.3325 | 0.0278 | -0.1982 |
| WNM-3D | II | 0.3664 | 0.0332 | -0.1425 |
| WNM-3D | III | **0.3781** | **0.0255** | **-0.1211** |

WNM-3D가 세 체크포인트 모두에서 flow-action 점수 우위(+0.0151 / +0.0157 / +0.0172)와 모션 오차 우위(-0.0022 / -0.0032 / -0.0034), 그리고 모든 단계에서 더 높은 action reward를 보인다. 흥미로운 패턴은 **DAgger가 행동 품질과 일관성은 크게 올리지만 near-goal STOP 세트에서 모션 오차를 일시적으로 악화**시키고(0.0278 → 0.0332), **DanceGRPO가 이를 Stage-I 수준 아래로 되돌리면서 일관성과 보상을 더 올린다**(0.0255)는 것이다.

## 11. 한계와 비판적 검토

- **단일 벤치마크·단일 시드**: 평가가 GN-Bench 하나에 한정되고, 저자들이 명시적으로 "다중 시드 표준편차나 부트스트랩 CI를 계산하지 않는다"고 밝힌다. Unseen에서 WNM-3D vs WNM-2D 격차가 0.9 SR / 0.7 SPL에 불과한데 5,000 에피소드 단일 시드 점 추정만으로는 이 차이가 유의한지 판단하기 어렵다. 기하 조건화의 핵심 이득이 사실상 Seen에 집중되어 있다는 점은 저자들도 인정한다.
- **GN-Bench는 저자 그룹의 벤치마크**: GN-Bench와 A* expert 파이프라인은 같은 저자진의 GN0(arXiv:2606.03682)에서 온다. 학습 데이터(16K A* 궤적)와 평가 환경이 동일 생태계에서 나오므로, 외부 VLN 표준(R2R-CE, RxR-CE 등)에서의 검증이 없다는 점이 아쉽다. 표의 NaVid/UniNaVid/InternNav 수치도 GN-Bench로 옮겨온 재평가라 원 논문 성능과 직접 비교되지 않는다.
- **TL의 해석**: WNM 변형들의 궤적 길이(7.9~8.5)가 베이스라인(2.7~5.2)보다 현저히 길다. SPL이 높으므로 비효율적 배회는 아니지만, 짧은 TL의 베이스라인들은 조기 STOP으로 실패하는 경향이 강했을 가능성이 크다. Stage-II에서 TL이 4.9 → 8.5로 뛰는 것도 DAgger가 "일단 더 오래 움직이게" 만든 효과가 섞여 있을 수 있다.
- **World 예측이 실제로 쓰이지 않는다**: 추론 시 예측된 시각 잠재는 디코딩되지도, 히스토리에 편입되지도 않는다. 즉 미래 뷰 생성은 **학습 시 보조 신호**로만 작동하며, foresight 기반 후보 평가 같은 명시적 활용은 없다. "world model"의 이득이 표현 학습 정규화에 가까운 것인지, 진짜 예측적 계획인지 구분하는 실험이 없다.
- **어댑터 내부 어블레이션 부재**: 3D Scene-to-Token Adapter는 레벨 융합 게이팅, anchored deformable resampling(R=2), factorized 시공간 정제(2블록), detail/coarse dual head 등 여러 설계 요소로 구성되는데, 이들 각각의 기여를 분해하는 어블레이션이 없다. 비교는 오직 "기하 prefix vs VAE prefix"의 이분법뿐이다. VGGT-Omega 블록 선택(5/12/18/24)이나 타깃 격자 9x5x10의 근거도 제시되지 않는다.
- **Flow-action 일관성 지표의 범위 제한**: 저자들이 직접 밝히듯 이 평가는 실행 블록의 XY 모션과 near-goal 상태에만 국한되며, yaw 일관성이나 전체 미래 뷰 충실도, 시뮬레이터 렌더링과의 지각적 일치를 측정하지 않는다. 절대 수치(S_flow-act 0.38)도 낮고 R_action은 여전히 음수여서, 개선폭(+0.017)의 실질적 의미를 해석하기 어렵다.
- **S_Pyr-SSIM은 표준 MS-SSIM이 아니다**: 보상에 쓰인 pyramid-averaged SSIM은 스케일별 SSIM의 산술평균이며 저자들이 "표준 곱셈형 MS-SSIM 지수로 해석하면 안 된다"고 각주를 단다. 보상 설계가 상당히 heuristic하고(D_deg의 0.10/0.25 캡, 다수의 임계값), 이 하이퍼파라미터 민감도 분석이 없다.
- **비용**: 16x H100에서 20 epochs SFT + 5 epochs DAgger + 1,500 GRPO updates, 조건당 8개 완전 world-action 롤아웃. DanceGRPO의 최종 기여(Seen +0.7 SR)에 비하면 Stage III의 비용-효익이 좋아 보이지 않는다. 실기 로봇 배치나 추론 지연(Hz) 보고도 전혀 없다.
- **코드/가중치 미공개**: 논문에 GitHub·프로젝트 페이지 링크가 없다.

## 12. 시사점

이 논문의 가장 재현 가치가 높은 발견은 아키텍처가 아니라 **학습 순서**다. "DAgger 없이 DanceGRPO 직행 → 성능 하락(SR 49.6 → 39.4)"은 생성형 정책에 RL 스타일 정제를 붙일 때의 일반 원칙을 시사한다. **group-relative 최적화는 그룹 내부에 의미 있게 변별되는 후보가 존재할 때만 신호를 만든다.** 오프라인 SFT 정책의 롤아웃 support가 좁으면 rank 신호는 노이즈일 뿐이며, 먼저 expert 교정으로 support를 넓혀야 한다. VLA에 GRPO 계열을 도입하려는 연구자라면, "SFT → RL" 대신 "SFT → on-policy 교정 → RL"을 기본 레시피로 놓을 근거를 여기서 얻을 수 있다.

두 번째로, **고정 길이 토큰 인터페이스로 표현과 생성기를 분리한 설계**가 실용적이다. VGGT-Omega를 동결한 채 23.97M 어댑터만 학습해 450 토큰 prefix를 만들고, 그 길이를 WNM-2D의 VAE prefix와 정확히 일치시켰다. 덕분에 "prefix 내용만 바꾼" 깨끗한 통제 실험이 가능했고, upstream 기하 모델을 교체해도 downstream DiT를 건드릴 필요가 없다. 이 패턴은 depth·semantic·촉각 등 다른 조건 모달리티에도 그대로 이식 가능하다.

세 번째로, 결과의 온도가 정직하다는 점을 짚어둘 만하다. 저자들은 Seen 이득(+5.7 SR)과 Unseen 이득(+0.9 SR)의 비대칭을 숨기지 않고, "Seen→Unseen 저하가 줄었다"는 매력적인 서사를 스스로 기각한다. 뒤집어 말하면 **기하 조건화의 현재 이득은 상당 부분 학습 씬에 대한 기억·정합에서 오며, 새로운 씬으로의 일반화는 아직 열린 문제**라는 뜻이다. 히스토리 기반 기하 토큰이 미지 환경에서도 작동하려면 씬 무관(scene-agnostic) 기하 표현 학습이 별도로 필요해 보인다.

<!-- VERIFIED: pdf -->
