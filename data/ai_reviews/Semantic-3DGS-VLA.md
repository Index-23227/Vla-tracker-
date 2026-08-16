# Semantic-3DGS-VLA: Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting

- **arXiv**: [2608.10756](https://arxiv.org/abs/2608.10756)
- **소속**: 홍콩과기대(광저우) HKUST-GZ, Midea Group
- **발표**: ACM Multimedia 2026 (MM '26, Rio de Janeiro), 2026-08-11 arXiv 공개
- **저자**: Huosen Ou, Dongni Song, Yuncong Wang, Tao Zhou, Yiding Ji

---

## 1. 한 줄 요약

가정 환경 모바일 매니퓰레이션에서, 손목 카메라 4장으로 즉석에서 만든 **로컬 Semantic-3D Gaussian Splatting** 필드를 능동 시점 선택·언어 기반 3D 위치추정·장애물 추론·베이스 자세 준비·VLA 조건화가 **공유하는 단일 인터페이스**로 쓰고, 그 3D 시맨틱을 diffusion action expert의 **마지막 5개 블록에만** 주입해 사전학습된 행동 prior를 보존하는 프레임워크.

## 2. 문제 설정

논문이 겨냥하는 실패 모드는 두 가지다.

1. **2D 외형 의존**: 많은 VLA가 여전히 2D appearance cue에 크게 의존해, 부분 가림·클러터·외형만 같은 distractor(예: 태블릿에 띄운 사실적인 바나나 사진)에서 grounding이 무너진다.
2. **몸체 준비 실패**: 타깃을 정확히 찾아도 모바일 베이스가 팔 workspace 대비 나쁜 위치에 있으면 실패한다. 장기 과제나 타깃 높이가 바뀔 때 특히 그렇다.

범위는 명시적으로 **open-vocabulary 타깃 grounding + few-shot 매니퓰레이션**이지, 임의 스킬의 zero-shot 획득이 아니다. 테스트 물체는 held-out 인스턴스이고, embodiment-specific 정책은 **태스크당 실제 시연 10개**로 적응시킨다.

## 3. 시스템 개요

언어 명령 *l* 수신 후 4단계로 동작한다: (1) 능동 로컬 다시점 관측 → (2) Semantic-3DGS 구축 및 open-vocabulary 3D 위치추정 → (3) reachability-aware 베이스 재배치 → (4) Semantic-3DGS 조건부 VLA 매니퓰레이션.

플랫폼은 **Unitree Go2 Edu** 4족 로봇(기립/웅크림 모드), Unitree 4D L1 LiDAR, 온보드 Jetson Orin NX, 그리퍼 근처 RGB 카메라가 달린 **Alicia-D 6-DoF 팔**이다. 팔 관절 목표(그리퍼 포함)는 ROS로 30 Hz 스트리밍된다. **연산 분리**가 명확한데, Semantic-3DGS 인식과 VLA 추론은 오프보드 RTX 4090에서, 저수준 제어와 베이스 자세 정책은 온보드에서 돈다.

## 4. 능동 다시점 Semantic-3DGS

### 4.1 능동 시점 획득

첫 손목 이미지에서 타깃 구문을 뽑아 language relevance map을 만들고, SAM 마스크와 거친 VGGT 지오메트리를 결합해 대략적인 3D 타깃 support를 얻는다. 후보 손목 자세는

    J(v) = λ_cov·C_sem(v) + λ_par·C_par(v) − λ_move·C_move(v)

로 점수화된다. C_sem은 타깃 support의 기대 시맨틱 커버리지, C_par는 상보적 시점 보상, C_move는 불필요한 손목 이동에 대한 페널티다. 항들은 결합 전 정규화되고 **IK-feasible 후보만** 남으며, 다음 시점은 argmax로 정해진다. 센싱 동안 베이스는 고정이고, 표준 IK 컨트롤러가 4장이 모일 때까지 손목 카메라를 움직인다. **시점 계획에 VLA 정책은 쓰지 않는다.**

### 4.2 지오메트리 초기화와 시맨틱 증류

사전학습 **VGGT**가 카메라 파라미터와 dense geometry를 예측해 로컬 Gaussian 필드를 초기화한다. 각 뷰에서 CLIP·DINOv2 feature map과 SAM 마스크를 뽑고, mask-aware average pooling으로 CLIP feature를 정제한 뒤 렌더링된 Gaussian feature를 코사인 정렬로 최적화한다:

    L_feat = Σ (1 − cos(F̂_C, F̃_C)) + λ_D · Σ (1 − cos(F̂_D, F_D)),  λ_D = 0.1

각 Gaussian은 g_k = (μ_k, Σ_k, c_k, α_k, f^C_k, f^D_k)로, 기하·색·불투명도에 CLIP 정렬 feature와 DINO feature가 붙는다.

### 4.3 Open-Vocabulary 3D 위치추정

타깃 구문은 frozen CLIP 텍스트 인코더로 e+가 되고, 일반 negative prompt들과 함께 softmax로 Gaussian별 언어 관련도 s_k를 준다. support K = {k | s_k > δ}에 대해 물체 위치는

    p_obj = Σ_{k∈K} w_k μ_k / Σ_{k∈K} w_k,   w_k = α_k · trace(Σ_k)^(−1)

로 추정한다. 즉 **불투명하고 좁게 퍼진 Gaussian에 더 큰 가중치**를 준다. 6D pose는 선택된 support에 PCA를 적용해 얻고, 템플릿이 있으면 ICP로 정제한다. 명시적 3D support 덕분에 단일 외형 단서 의존이 줄고, 타깃과 주변 장애물 기하가 동시에 노출된다.

## 5. Reachability-aware 베이스 자세 제어

물체 위치를 베이스 프레임으로 옮긴 뒤 pre-manipulation 자세를 정의한다:

    x* = x_obj − d_x,  y* = y_obj − sign(y_obj)·d_y,  ψ* = atan2(y_obj, x_obj)

d_x = 0.35 m, d_y = 0.20 m이고, 높이 모드는 z_obj < 0.30 m이면 crouch, 아니면 stand로 고른다. 실제 다리 관절 residual과 stand/crouch 스위치는 **PPO 정책**이 출력하며, Isaac Sim 기반 **Isaac Lab**에서 domain randomization과 함께 학습된다. 논문은 시뮬레이션에서 학습되는 것은 **베이스 자세 정책뿐**임을 명시한다.

## 6. Semantic-3DGS 조건부 VLA 정책 (핵심 기여)

**DexVLA 계열** 정책 위에 Qwen2-VL 백본과 **ScaleDP** action expert를 쓴다. 명령 *l*과 Gaussian 필드 G로부터 네 가지 단서를 모은다: (1) 언어 조건부 타깃 heatmap H_t, (2) 장애물 인지 occupancy 단서 O_t, (3) per-Gaussian 시맨틱 필드의 3채널 PCA 렌더링 P_t, (4) 타깃 상대 pose 벡터 r_t = [p^B_obj, h_t] ∈ R⁴.

    z_img = E_img(concat(I_t, H_t, O_t, P_t)) ∈ R^128
    z_pose = E_pose(r_t) ∈ R^128
    z_sem = z_img + z_pose

그리고 **Late-Block Semantic Injection**: B = {L−4, L−3, L−2, L−1, L} 즉 **마지막 5개 diffusion 블록에만**

    h_ℓ ← h_ℓ + A_ℓ(Proj(z_sem)),  ℓ ∈ B

Proj는 zero-initialized, A_ℓ는 경량 MLP 어댑터다. **VLM 백본과 모든 사전학습 diffusion 블록은 frozen**이고, 시맨틱 인코더·projection·late-block 어댑터·embodiment-specific action head만 학습된다. 태스크당 실제 시연 10개, action chunk 15 스텝, 최근 관측 프레임 2장을 쓴다.

논문의 주장은 명확하다. 조기 주입은 사전학습 prior를 손상시키므로, 3D 시맨틱은 늦게 넣어야 연속 행동 prior를 보존하면서 공간 grounding을 더할 수 있다는 것이다.

## 7. 실험 프로토콜

기본적으로 설정당 **실제 로봇 30 trial**(초기 물체 자세 랜덤화)이고, 장기 과제와 클러터 banana-to-bowl은 **방법당 50 trial**로 확장된다. 성공은 시간 예산(로컬 멀티태스크 60 s, 장기 과제 180 s) 안에 **명령 전체를 안전하게 완수**해야만 인정된다. 50-trial 연구에는 성공 횟수, 시간 표준편차, **95% Wilson 신뢰구간**이 추가로 보고된다.

**공정한 베이스라인 적응**을 상당히 신경 썼다. 모든 베이스라인이 동일 embodiment, 동일한 태스크당 10개 원격조작 시연을 쓰고, train/test split·손목 카메라 설정·행동 인터페이스·내비게이션 waypoint·베이스 제어 보조·초기 장면 분포·평가 예산을 고정한다. 각 방법은 자기 고유 인식 표현을 유지하므로, 비교는 주로 **grounding·매니퓰레이션 표현의 차이**를 반영한다.

## 8. 주요 결과

**장기 과제 (Table 1, 50 trial)** — 서랍 열기 → 바나나 꺼내기 → 서랍 닫기 → 검은 의자로 이동 → 놓기:

| 방법 | 성공 | 95% CI | 평균 시간 (s) |
|---|---|---|---|
| DexVLA | 14/50 (28%) | [17.5, 41.7] | 178.6 ± 18.4 |
| PointVLA | 20/50 (40%) | [27.6, 53.8] | 161.8 ± 17.6 |
| Ours w/o Base-RL | 11/50 (22%) | [12.8, 35.2] | 204.1 ± 21.3 |
| **Ours (full)** | **30/50 (60%)** | [46.2, 72.4] | **140.7 ± 15.9** |

Base-RL을 빼면 22%로 떨어져 **베이스라인들보다도 낮다**. grounding만 정확해도 자세가 팔 workspace에 맞게 준비되지 않으면 소용없다는 뜻이다.

**높이 적응 (Table 2)** — 플랫폼 높이 오프셋 30/60/75 cm: full은 80% / 78% / 75%로 거의 평평하게 유지된다. DexVLA는 48/33/23%, PointVLA는 58/46/35%로 급락하고, Base-RL 없는 변형은 **모든 오프셋에서 0%**다.

**Photo deception (Table 3)** — 실제 바나나를 태블릿의 사실적 바나나 사진으로 대체:

| 방법 | 실물 성공 ↑ | 사진 오파지율 ↓ |
|---|---|---|
| DexVLA | 78% | 76% |
| PointVLA | 80% | 0% |
| Ours Single-View | 76% | 70% |
| **Ours (full)** | **88%** | **0%** |

RGB-only DexVLA와 단일 시점 ablation은 평면 화면에 크게 속고, PointVLA와 full은 완전히 거부한다. full은 거부에 더해 실물 성공률도 올려 위치추정 자체가 안정적임을 시사한다.

**클러터 banana-to-bowl (Table 4, 50 trial)**:

| 방법 | 성공 | 95% CI | 충돌 없음 | 오파지 | 평균 시간 (s) |
|---|---|---|---|---|---|
| DexVLA | 13/50 (26%) | [15.9, 39.6] | 44% | 32% | 27.9 ± 3.1 |
| PointVLA | 23/50 (46%) | [33.0, 59.6] | 62% | 20% | 30.7 ± 3.4 |
| Ours Single-View | 26/50 (52%) | [38.5, 65.2] | 70% | 18% | 29.5 ± 3.2 |
| **Ours (full)** | **37/50 (74%)** | [60.4, 84.1] | **88%** | **6%** | 33.2 ± 3.6 |

능동 다시점이 단일 시점 대비 성공 52% → 74%, 충돌 없음 70% → 88%, 오파지 18% → 6%. PointVLA 대비 **+28 %p**.

**Few-shot 멀티태스크** — full 평균 **81.7%**, PointVLA 64.0%, DexVLA 37.7% (본문 서술값; 태스크별 막대는 Figure 8에만 있어 여기서는 기록하지 않는다).

## 9. Ablation 분석

**클러터 30-trial 구성요소 ablation** (성공률):

| 변형 | 성공률 |
|---|---|
| VGGT point-map 변형 | 58% |
| CLIP/DINO 시맨틱 제거 | 60% |
| 장애물 occupancy 단서 제거 | 65% |
| All-block 시맨틱 주입 | 68% |
| **Full** | **74%** |

논문은 같은 ablation들에서 충돌 없음 비율과 오파지율도 함께 나빠진다고 언급하며, 따라서 이득이 능동 센싱이나 베이스 재배치만으로 생긴 것이 아니라고 주장한다(수치는 미제시).

**주입 깊이 (Table 5)**:

| 변형 | 평균 성공 | chunk 지연 |
|---|---|---|
| All-block | 75% | 175 ms |
| **Late-block (5)** | **82%** | **80 ms** |

성공과 지연 양쪽에서 late-block이 이긴다. Figure 11의 block sensitivity 추세는 더 이르거나 더 넓은 개입이 사전학습 행동 prior를 더 강하게 교란함을 보인다(그림 전용이라 수치 미기록).

**런타임 프로파일 (Table 6)**: 4시점 손목 이동+촬영 16.00 s, VGGT pose/depth 초기화 0.62 s, Semantic-3DGS feature 갱신 1.21 s, 시맨틱 렌더링+위치추정 0.34 s, VLA action-chunk 추론 0.08 s/chunk, ROS/WiFi 통신 0.05 s/chunk, 전체 클러터 태스크 33.2 ± 3.6 s. 논문은 **일회성 비용과 chunk당 비용을 산술적으로 더해 태스크 시간을 추정하지 않는다**고 못박고, 전체 wall-clock은 능동 센싱 시작부터 배치 완료까지 독립 측정했다고 밝힌다. 능동 다시점은 클러터 태스크에 약 **3.7 s**를 더하고(33.2 vs 29.5 s), 그 대가로 성공 +22 %p, 충돌 없음 +18 %p를 얻는다.

## 10. 강점

- **표현의 재사용이 진짜 설계 원리다.** 동일한 object-centric Semantic-3DGS가 능동 시점 스코어링, Gaussian 수준 언어 위치추정, 장애물 인지 렌더링, PCA 시맨틱 토큰화, 타깃 상대 pose 조건화, zero-init late-block 적응에 모두 쓰인다. 기존 모듈의 느슨한 결합이 아니라는 주장에 근거가 있다.
- **평가 위생이 좋다.** 50 trial + Wilson CI, 시간 표준편차, 동일 시연 예산·내비게이션 보조 고정, 런타임 3단계 분리 보고까지 실제 로봇 논문치고 이례적으로 성실하다.
- **Base-RL ablation이 시스템 논지를 직접 입증한다.** 22%(장기)와 0%(높이 전환)라는 극단적 붕괴는 "인식만으로 부족하다"는 주장에 대한 가장 강한 증거다.
- **Late-block 주입의 trade-off 근거가 정량적이다.** 성공 82% vs 75%, 지연 80 ms vs 175 ms로 두 축 모두에서 우세하다.
- **Photo deception 실험 설계가 영리하다.** 3D grounding의 가치를 단일 숫자(오파지 0% vs 70~76%)로 드러내는 저비용 진단이다.

## 11. 약점과 의문점

- **표준 시뮬 벤치마크가 전혀 없다.** LIBERO·CALVIN·SimplerEnv·RoboTwin 어느 것도 보고되지 않아, 다른 VLA와의 교차 비교가 불가능하고 leaderboard 통합이 어렵다.
- **오프보드 RTX 4090 의존.** Semantic-3DGS와 VLA 추론이 워크스테이션에서 돌고, Jetson Orin NX는 저수준 제어만 담당한다. 온보드 자율성 주장은 할 수 없다.
- **16 s의 사전 센싱 비용.** 런타임의 압도적 지배 항이 손목 4시점 이동/촬영이다. 논문도 quasi-static 가정을 명시하지만, 동적 상호작용으로의 확장 경로는 제시되지 않는다.
- **Few-shot 태스크별 수치와 block sensitivity가 그림 전용.** 81.7%라는 평균만 본문에 있고 태스크별 분해는 검증 불가다.
- **베이스라인이 두 개(DexVLA, PointVLA)뿐.** π0/π0.5, OpenVLA 계열, GaussianGrasper 같은 3DGS 파지 계열과의 직접 비교가 없다.
- **"Avg. success" 정의 불명.** Table 5의 82%/75%가 어느 태스크 집합의 평균인지, 클러터 ablation의 74%/68%와 왜 다른지 명시되지 않는다.
- **Semantic-3DGS 자체가 frozen foundation model 조합.** VGGT, CLIP, DINOv2, SAM 모두 사전학습·동결이고 학습되는 것은 어댑터·인코더·action head·PPO 정책이다. 기여의 상당 부분이 시스템 통합에 있다는 점은 인정해야 한다.

## 12. 종합 평가 및 시사점

**VLA-Tracker 편입 판정: ACCEPTED.** 이 논문은 frozen 모델을 호출만 하는 grounding 파이프라인이 아니다. (a) 실제 시연 10개로 학습되는 embodiment-specific action head와 late-block 어댑터·시맨틱 인코더라는 **자체 학습 정책 산출물**이 있고, (b) Isaac Lab에서 PPO로 학습한 베이스 자세 정책이 별도로 존재한다. DexVLA 계열 diffusion expert 위의 파라미터-효율 적응이라는 점에서 학습량은 제한적이지만, 산출물의 성격은 명백히 정책이다.

기술적 시사점은 세 가지다. 첫째, **주입 위치가 하이퍼파라미터가 아니라 설계 결정**이라는 점을 정량적으로 보인다. 사전학습 diffusion prior를 건드리지 않고 후단에서만 조건화하면 성능과 지연 모두 개선된다는 결과는 다른 3D-conditioned VLA에도 이식 가능한 교훈이다. 둘째, **모바일 매니퓰레이션에서 병목은 팔 정책이 아니라 몸체 준비**일 수 있다는 것을 Base-RL ablation의 극단적 붕괴로 보여준다. 셋째, **task-driven 로컬 표현**이 dense full-scene 최적화나 persistent global mapping보다 실용적일 수 있다는 방향성이다 — 4장의 이미지, 2초 남짓의 필드 구축, 실패 시 갱신이라는 값싼 루프.

한계도 분명하다. 표준 벤치마크 부재로 이 시스템의 절대적 위치를 가늠할 수 없고, 오프보드 GPU와 16초 센싱 오버헤드는 배포 시나리오를 좁힌다. 저자들이 밝힌 향후 방향(더 가벼운 온보드 표현, 적응적 시점 계획, 더 넓은 미지 물체 일반화)은 정확히 이 한계들을 겨눈다.

<!-- VERIFIED: pdf -->
