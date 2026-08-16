# ω-0 (OMEGA-0): A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation

> **한 줄 요약**: 휴머노이드 가사 태스크에서 이동과 조작을 분리하지 않고 **동시(concurrent) loco-manipulation**을 학습하기 위해, 미래 영상을 픽셀로 생성하는 대신 **압축된 미래 관측 임베딩**만 예측하는 경량 world-model 신호를 diffusion 기반 전신 액션 생성과 결합. Qwen3-VL-2B 기반 whole-body VLM prefix + V-JEPA 인코더 + 0.45B Action DiT가 SONIC 컨트롤러 호환 66차원 전신 action latent를 직접 출력하며, 실제 Unitree G1에서 11개 가사 태스크 단일 정책 성공률 **81.8%** (ψ-0 44.5, DiT4DiT 43.6, GR00T-N1.7 22.7 대비). 40.3시간 멀티모달 가정 데이터셋 **ω-HOME** 공개. NTU MARS Lab · PKU · BAAI · HKUST(GZ).

---

## 1. 배경 및 동기

- 가정 환경의 휴머노이드는 "이동 후 조작"이 아니라 **움직이면서 조작**해야 함 — 큰 테이블을 닦으려면 발을 옮기고 상체를 기울이며 접촉을 유지해야 하고, 세탁기 하단 칸에 옷을 넣으려면 리칭·굽힘·균형·손 제어가 동시에 일어남
- 기존 VLA 다수는 **팔 중심(arm-centric)** — end-effector나 팔 액션만 예측 (π 시리즈, RT 계열, Diffusion Policy). 모바일 매니퓰레이션은 워크스페이스를 넓히지만 베이스와 팔을 여전히 별개 모듈로 취급
- 최근 휴머노이드 VLA(ψ-0, OpenHLM, GR00T)도 진전을 이뤘으나, 대개 **locomotion / balance / manipulation을 분해**하는 실용적 절충에 의존. 이 분해는 "먼저 이동, 그다음 정지 상태 조작"에는 효과적이지만 스텝·상체 조정·리칭·접촉 유지가 동시에 요구되는 태스크에서는 한계
- World Action Model(WAM)은 미래 시각 동역학을 추가 감독으로 써 태스크 진행도를 학습하는 자연스러운 대안. 그러나 대부분 팔 중심이며, 휴머노이드 WAM(DiT4DiT 등)은 **비디오 동역학을 액션 예측의 주 경로**로 삼음
- 저자들의 문제 제기: 실제 휴머노이드에서 시각 관측은 노이즈·로봇 몸통/도구에 의한 가림·이동 중 시점 변화로 오염됨. 액션 생성이 예측 비디오 궤적에 강하게 의존하면 그 궤적의 시간적 불일치가 **급격한 전환·머뭇거림·전신 불안정**으로 증폭됨. 또 픽셀 충실도 향상이 제어 성능으로 직결되지 않음
- 핵심 질문: **"미래 예측을 비디오 생성 타깃이 아니라 전신 액션 학습을 위한 압축된 예측 신호로 쓸 수 있는가?"**

## 2. 방법론 심층 분석

전체 구조는 3단계 학습 파이프라인 + 실기 배포로 구성.

**Stage 1 — Whole-Body Action VLM 사전학습**
- 사전학습 VLM의 출력 공간은 이산 언어 토큰용이라 고차원 연속 휴머노이드 액션을 직접 예측하기 어렵다는 문제 인식
- 전신 궤적 `a_{t:t+H} ∈ R^{H×d_a}`에 **whole-body FAST tokenizer**를 학습 (`L_tok = ||â − a||₁`)해 이산 액션 토큰 `c_{1:N}`으로 변환
- Qwen3-VL-2B-Instruct를 입력 `x = [e_v, o_t^v, ℓ]` (view token + 관측 + 언어)에 대해 `p_θ(c_{1:N} | ·)` next-token loss로 파인튜닝. 시점을 구분하는 **learnable view token**이 ego/exo를 명시적으로 분리
- 결과 hidden representation을 이후 단계의 **action-aware semantic prior**로 사용

**Stage 2 — Human-to-Humanoid Action-Latent 사전학습 (핵심)**
- V-JEPA에서 영감을 받아 **reconstruction-free 미래 임베딩 예측** 목적함수 채택. Ground-truth 미래 시각 latent는 frozen **Wan encoder**로 추출: `y^v_{t+1:t+K} = E_Wan(o^v_{t+1:t+K})`, 손실 `L_video = ||h_v − y||²₂`
- Prefix condition: `p = [f_vlm, f_ℓ, r_v, f_t^v]` (Stage-1 VLM 특징, T5 텍스트 특징, view token, frozen V-JEPA 2.1 시각 특징)
- 두 종류 learnable query — **motion query** `q_m` (개수를 action chunk size에 일치시켜 각 query가 한 미래 스텝에 대응) 와 **video query** `q_v` (미래 시각 latent 토큰)
- **Prefix-guided dual-query attention** (3 STEP): ① 각각 self-attention → `p̃, q̃_m, q̃_v` ② 두 query가 prefix에 cross-attention → `q̄_m, q̄_v` ③ **motion query가 video query에 cross-attention** `h_m = CrossAttn_mv(q̄_m, q̄_v)` — 예측된 시각 동역학을 액션 표현에 주입해, 액션 브랜치가 자기 행동이 유발할 미래 관측을 고려하도록 유도
- **Token-specific RoPE**: 시각 prefix 토큰에 2D RoPE(공간 패치 좌표), 미래 video query에 3D RoPE(시공간), action query에 1D 시간 RoPE. 텍스트/view/VLM 요약 토큰은 비공간 조건 토큰으로 위치 인코딩 미부여
- 액션 브랜치: `c_dit = Φ_cond([h_m, f_ℓ, f_s])`로 융합 후 **Action DiT (0.45B)** 가 `z_τ = √ᾱ_τ z₀ + √(1−ᾱ_τ)ε`를 **x0-prediction** (`L_action = ||ẑ₀ − z₀||²₂`)으로 디노이징. 추론은 DDIM
- 총 목적: `L_stage2 = L_action + λ_video · L_video`. V-JEPA / Wan / action VLM은 **frozen**, joint predictor·state encoder·condition fusion·Action DiT만 최적화
- **SONIC 시뮬레이션 리플레이**로 인간 데이터 접지: 공개 human video-action 데이터는 로봇 고유 proprioception이나 action latent를 제공하지 않으므로, SONIC이 각 모션을 시뮬에서 추종하며 대응 전신 action latent와 로봇 상태를 생성. 무술 동작 등 물리적으로 추종 불가한 궤적은 필터링 (Motion-X에서 특히 중요)

**Stage 3 — 실기 데이터 파인튜닝**
- **RTC(real-time chunking)** 를 학습 시점에 도입: prefix 길이 M을 무작위 샘플링해 노이즈 액션 시퀀스의 앞 M 프레임을 clean latent로 치환(`z̃^{1:M}_τ = z₀^{1:M}`)하고, 손실은 비-prefix 구간에만 계산 `L_RTC = ||ẑ₀^{M+1:H} − z₀^{M+1:H}||²₂`
- `L_stage3 = L_RTC + λ_video · L_video`. Clean prefix가 시간적 앵커 역할을 해 인접 chunk 간 불연속을 줄임

**상태·액션 표현 (Appendix A)**
- 로봇 상태 `s_t = [q_pos, q_hand, r^6D_root]` — **47차원**. IMU 쿼터니언은 antipodal 이중 표현(q, −q)의 불연속 문제 때문에 **연속 6D 회전 표현**으로 변환하며, IMU 선가속도·각속도는 사용하지 않음
- 액션은 **66차원** = 컨트롤러 인터페이스가 산출하는 64차원 전신 action latent + 좌/우 손 명령 스칼라 2개([0,1], 1=완전 파지)
- 정규화 분리: 64차원 latent는 mean-std, 로봇 상태 및 손 명령 2차원은 min-max (경계 의미 보존)

**배포 (Appendix B)**
- 단일 forward pass ≈ **0.14초 (>7 Hz)**. Chunk 길이 **H=25** 예측 후 앞 **K=8**만 실행하고 새 이미지 취득 (receding horizon)
- 텍스트 명령은 에피소드 시작 시 1회만 토크나이즈해 재사용, 시각 토큰·상태는 매 스텝 갱신
- RTC-style warm start: 이전 chunk의 미실행 구간을 다음 디노이징의 prefix 초기화로 사용. 추가로 **overlap blending** `a_j = (1−α_j)a_prev + α_j a_next`, `α_j = (j+1)/(O+1)` 로 chunk 경계 고주파 점프 제거
- 미래 시각 latent 브랜치는 **실시간 제어에 불필요** — 배포 시 액션 latent만 실행. 정성 시각화가 필요할 때만 별도 video decoder로 디코딩

## 3. 실험 설정

- **로봇/컨트롤러**: Unitree G1 휴머노이드 + SONIC 저수준 전신 컨트롤러. 모든 실기 롤아웃은 학습된 정책이 자율 실행 (텔레오퍼레이션·모션 리플레이·스크립트 개입 없음)
- **학습 자원**: 3단계 모두 8× NVIDIA H100
- **공개 데이터**: ARCTIC(ego+exo), Xperience-10M(수동 필터링), Motion-X(3인칭 exo 이해 강화). SMPL-H/SMPL-X → 통일 SMPL 변환, z-up 정규화, **zero-yaw 정규화**(`ψ_norm_t = ψ_t − ψ_0`)로 초기 헤딩 정렬
- **실기 학습 데이터**: 11개 가사 loco-manipulation 태스크, 태스크당 약 200 시연, 총 **2,220 궤적**. 태스크별 개별 정책이 아니라 전부 혼합해 **단일 일반 모델** 파인튜닝
- **평가 지표 3종** (태스크당 10회 독립 시행, 모든 방법 동일 프로토콜, 모두 단일 멀티태스크 정책)
  - **Success Rate**: 평가 horizon 내 전 목표 완수한 시행 비율 (`SR = N_success/10`)
  - **Score**: 태스크를 사전 정의 subtask로 분해, 완료 subtask당 1점. 전체 태스크 스위트 최대 **41점**. 초기 실수가 이후 subtask 득점을 막지 않는 **비-prefix 척도**
  - **Task Progress**: 첫 실패/비가역 이탈 전까지 진행 비율 `m/n` (완주 시 1.0). Score와 상보적으로 **순서대로 이어진 실행**을 강조
- **태스크 스위트 11종** (Table 1): 사과→바구니(유일하게 하체 미개입), 선반 정리, 침대 옷→바구니, 수건→세탁기, 테이블 닦기, 바닥 mop, 다양한 높이 쓰레기→손에 든 통, 사과→서랍+무릎으로 서랍 닫기, 침대 쓰레기 쓸어 담고 돌아서 통에 버리기, 세탁기에서 옷 꺼내기, 냉장고에서 음료 꺼내기

## 4. 핵심 결과 — 실기 종합 평가 (Table 2)

| 방법 | Success Rate (%) | Score (max 41) | Task Progress (%) |
|---|---|---|---|
| ACT | 8.2 | 10.6 | 32.4 |
| Diffusion Policy | 15.5 | 14.8 | 40.6 |
| π-0.5 | 27.3 | 20.9 | 52.8 |
| InternVLA-M1 | 31.8 | 21.8 | 55.6 |
| EgoVLA | 25.5 | 18.6 | 49.1 |
| GR00T-N1.7 | 22.7 | 19.7 | 49.8 |
| ψ-0 | 44.5 | 23.6 | 59.6 |
| Fast-WAM | 37.1 | 22.3 | 57.8 |
| DiT4DiT | 43.6 | 23.1 | 61.0 |
| **ω-0 (Ego)** | **79.1** | **35.8** | **88.7** |
| **ω-0 (Omni)** | **81.8** | **36.7** | **90.3** |

- 최강 베이스라인 ψ-0(44.5) 대비 **+37.3%p**, 비디오 생성 기반 WAM인 DiT4DiT(43.6) 대비 **+38.2%p** — 격차가 매우 큼
- 고전 IL(ACT 8.2, DP 15.5)은 단기 chunk는 생성하나 장기 전신 협응에서 붕괴. VLA 계열(22.7~31.8)은 사전학습 표현의 이점은 있으나 액션 인터페이스가 통합 전신 제어용으로 설계되지 않음
- ω-0_Omni > ω-0_Ego: Omni는 **locomotion 비중이 큰 5개 태스크**(침대 옷→바구니, 수건→세탁기, mop, 높이별 쓰레기 수거, 사과→서랍+무릎 닫기)에 exocentric 관측과 exocentric 미래 latent 감독을 사용하고 나머지는 ego 사용. 1인칭은 실행 시점 지각과 정렬되지만 로봇 전신 이동·변위를 충분히 드러내지 못함

## 5. ω-HOME 데이터셋과 사전학습 효과 (Table 3)

- **ω-HOME**: 40.3시간, **4,827 에피소드**, **24 태스크**, 30 Hz. 에피소드마다 6개 동기화 모달리티 — ego RGB(로봇 탑재 카메라, 배포 입력과 동일), exo RGB-D(ZED), proprioception, 전신 SMPL 모션, 컨트롤러 호환 action latent, 언어 명령
- 8개 상위 능력군: 물체 회수, 표면 청소, 가전 상호작용, 용기 이송, 의류 처리, 수납 정리, 모바일 매니퓰레이션, 도구 기반 바닥 작업

| 변형 | ω-HOME 사전학습 | SR (%) | Score | Progress (%) |
|---|---|---|---|---|
| ω-0 Ego | ✗ | 79.1 | 35.8 | 88.7 |
| ω-0 Ego | ✓ | 80.4 | 36.9 | 89.7 |
| ω-0 Omni | ✗ | 81.8 | 36.7 | 90.3 |
| ω-0 Omni | ✓ | 82.4 | 37.5 | 91.2 |

- **태스크 누출 방지**를 위해 11개 다운스트림 태스크를 ω-HOME 사전학습 풀에서 제외한 뒤 나머지를 Stage 2에 투입. 그럼에도 일관된 향상(+1.3/+0.6%p SR)이 나타나 데이터셋이 다운스트림 시연 자체를 넘는 실기 전신 visual-action prior를 제공함을 시사
- 다만 향상 폭 자체는 **1%p 내외**로 크지 않아, 데이터셋의 가치 주장에 비해 정량 근거는 다소 약함

## 6. Ablation (Table 4)

| 변형 | State | VLM Prefix | Video Query | RTC | Encoder | SR (%) | Score | Progress (%) |
|---|---|---|---|---|---|---|---|---|
| w/o Robot state | ✗ | ✓ | ✓ | ✓ | V-JEPA | 60.9 | 29.8 | 75.6 |
| w/o VLM prefix | ✓ | ✗ | ✓ | ✓ | V-JEPA | 66.4 | 31.7 | 79.8 |
| w/o Video query | ✓ | ✓ | ✗ | ✓ | V-JEPA | 64.5 | 30.6 | 77.9 |
| w/o RTC | ✓ | ✓ | ✓ | ✗ | V-JEPA | 71.8 | 33.4 | 84.1 |
| Wan as encoder | ✓ | ✓ | ✓ | ✓ | Wan | 63.6 | 30.9 | 77.3 |
| **Full ω-0 Ego** | ✓ | ✓ | ✓ | ✓ | V-JEPA | **79.1** | **35.8** | **88.7** |
| **Full ω-0 Omni** | ✓ | ✓ | ✓ | ✓ | V-JEPA | **81.8** | **36.7** | **90.3** |

- **로봇 상태 제거가 가장 치명적** (−18.2%p). 동일 관측·명령이라도 현재 자세·헤딩·균형·손 형상에 따라 필요한 액션이 달라지므로, 회전·스텝·굽힘·이동 중 접촉 유지 태스크에서 특히 타격
- **Video query 제거 −14.6%p** — 미래 시각 latent 예측이 단순 보조 손실이 아니라 실제 제어 성능의 핵심 기여임을 보여주는 논문의 중심 근거
- **VLM prefix 제거 −12.7%p**, **Wan을 현재 이미지 인코더로 대체 −15.5%p** (생성용 latent가 제어용 표현으로는 V-JEPA보다 열등)
- **RTC 제거 −7.3%p** — 상대적으로 작지만, 인접 chunk 연속성이 실기 부드러움에 기여

## 7. 일반화 — 미래 latent 예측의 효과 (Table 5)

| 설정 | Video Query | SR (%) | Score | Progress (%) |
|---|---|---|---|---|
| Cross-object (3 태스크, max 13) | ✗ | 66.7 | 7.6 | 63.3 |
| | ✓ | **83.3** | **11.8** | **90.8** |
| Cross-scene (2 태스크, max 6) | ✗ | 15.0 | 0.5 | 15.0 |
| | ✓ | **79.5** | **5.5** | **91.7** |
| Human Data Transfer (1 태스크, max 3) | ✗ | 20.0 | 1.2 | 20.0 |
| | ✓ | **60.0** | **2.2** | **74.6** |

- **Cross-scene에서 격차가 압도적** (15.0 → 79.5, +64.5%p). 새 방 레이아웃·가구 배치에서 video query 없는 변형은 사실상 붕괴
- Cross-object 태스크: 선반에서 배 집기, 다른 색 옷 세탁기에서 꺼내기, 냉장고에서 와플 꺼내기. Cross-scene: 다른 방 침대에서 옷 집기, 다른 장면 테이블로 걸어가 닦기. Human transfer: 인간 시연만으로 학습 후 옷장 문까지 걸어가 닫기
- 해석: 미래 latent 예측 브랜치가 태스크 진행도·물체-장면 상호작용·전신 동작 결과에 대한 **future-aware 표현**을 학습시켜 전이 가능한 액션 표현을 만든다는 것. 본 실험이 논문 주제를 가장 강하게 뒷받침

## 8. 강점

- **문제 정의가 선명하고 타당**: "비디오 생성 충실도 ≠ 제어 성능"이라는 관찰과, 이동 중 시점 변화·가림이 많은 실제 휴머노이드에서 비디오 궤적 의존이 오류를 증폭한다는 진단이 설득력 있음. V-JEPA식 latent 예측으로의 전환이 자연스러운 귀결
- **통합 전신 인터페이스**: SONIC 호환 66차원 latent를 직접 출력해 하체/상체/손을 한 번에 생성 — 상체만 예측하고 하체는 AMO에 맡기는 ψ-0 대비 구조적 차별점이 명확하고, 그 차이가 결과(44.5 → 81.8)로 나타남
- **엔지니어링 디테일의 밀도**: 쿼터니언 이중 표현 회피를 위한 6D 회전, latent/상태 분리 정규화, zero-yaw 정규화, RTC warm start + overlap blending, 텍스트 토큰 캐싱 — 실기 배포를 실제로 해본 팀의 흔적이 촘촘함
- **Ablation이 주장과 정확히 대응**: video query 제거가 in-distribution에서 −14.6%p, cross-scene에서 −64.5%p로 나타나 "미래 예측은 일반화에 특히 기여"라는 서사가 데이터로 뒷받침됨
- **단일 정책 강조**: 11개 태스크를 태스크별 정책·태스크별 액션 헤드·별도 locomotion 모듈 없이 하나의 모델로 수행. 베이스라인도 동일하게 단일 멀티태스크로 통일한 프로토콜은 공정함
- **데이터셋 기여**: 40.3시간·4,827 에피소드·6모달리티 동기화 가정 휴머노이드 데이터는 이 도메인에서 희소한 자원

## 9. 약점 / 한계

- **표준 벤치마크 부재**: 모든 수치가 자체 정의 11개 실기 태스크·자체 subtask 채점(최대 41점) 기반. LIBERO/CALVIN/SimplerEnv 같은 공개 리더보드가 없어 **외부 재현·교차 비교가 사실상 불가능**. VLA-Tracker 관점에서는 real_world 트랙에만 등재 가능
- **베이스라인 이식의 공정성 의문**: π-0.5·InternVLA-M1·EgoVLA·GR00T-N1.7 모두 액션 헤드를 휴머노이드 액션 공간에 맞춰 저자들이 개조/확장했고, InternVLA-M1은 백본을 얼렸으며, GR00T는 RTC 없이 표준 순차 chunk 추론으로 평가. 이런 조건에서 나온 22.7~31.8%가 각 모델의 실제 상한인지 확인할 수 없음. 특히 RTC 부재는 ω-0 자체 ablation에서 −7.3%p로 측정된 요소
- **표본 크기**: 태스크당 10회 시행 × 11 태스크 = 110 롤아웃. 일반화 실험은 더 작아 human transfer는 **단일 태스크**에 불과해 60.0% vs 20.0%의 통계적 신뢰도가 낮음
- **컨트롤러 의존성**: 성능의 상당 부분이 SONIC의 추종 능력에 귀속될 수 있으나, 컨트롤러를 교체한 대조 실험이 없어 "정책 기여 vs 컨트롤러 기여"의 분해가 불가능. 인간 데이터 접지도 "SONIC이 추종 가능한 모션만" 남기는 필터링에 의존하므로 사용 가능한 인간 데이터 분포가 컨트롤러에 의해 제약됨
- **추론 속도 7 Hz**: 30 Hz로 수집된 데이터에 비해 정책 갱신 주기가 느림. H=25 예측 후 K=8 실행이라는 receding horizon과 blending으로 보완하지만, 빠른 외란 대응 여지는 제한적
- **미공개**: 프로젝트 페이지만 존재하고 코드·가중치·ω-HOME의 실제 배포 여부는 논문 시점에 확인되지 않음. `open_source: false`로 기록
- **λ_video 민감도 부재**: 논문의 핵심 하이퍼파라미터인 미래 latent 손실 가중치에 대한 스윕이 없음. FAST tokenizer 어휘 크기, chunk 길이 H, DDIM 스텝 수도 미보고
- **Wan 인코더 ablation의 혼동 가능성**: "Wan as encoder"는 현재 이미지 인코더 교체 실험인데, Wan은 동시에 미래 latent의 ground-truth 공급원으로도 쓰임. 두 역할의 분리가 서술상 다소 압축적

## 10. 다른 연구와의 위치

- **ψ-0 (Wei et al., 2026)** 대비: ψ-0는 ego human video 사전학습 + 실기 post-training이라는 단계 전략은 공유하나, 상위 정책이 주로 상체/팔-손 액션을 예측하고 하체는 AMO 컨트롤러가 담당하는 **구조적 분리**. ω-0는 전신 latent를 단일 인터페이스로 예측 — 이 논문이 강조하는 핵심 대비축
- **DiT4DiT (Ma et al., 2026)** 대비: video DiT의 중간 디노이징 특징으로 action DiT를 조건화하는 비디오 생성 기반 WAM. ω-0는 비디오 생성 경로 자체를 제거하고 latent 예측만 남김. 43.6 vs 81.8은 "생성 없는 예측"이라는 설계 선택의 근거로 제시됨
- **Fast-WAM (Yuan et al., 2026)** 대비: 학습 시 비디오 co-training, 추론 시 명시적 비디오 생성 제거라는 점에서 철학이 가장 가깝지만 주로 팔 중심 매니퓰레이션·타월 접기에서 평가됨. 37.1%는 팔 중심 WAM이 전신 태스크로 곧장 옮겨지지 않음을 보임
- **V-JEPA 계보**: reconstruction-free 미래 임베딩 예측이라는 표현 학습 아이디어를 로봇 제어 정책의 보조 목적함수로 이식한 사례. JEPA류 표현이 실기 제어 성능으로 이어짐을 실증한 드문 결과
- **GR00T-N1.7 / π-0.5** 대비: 대규모 일반 로봇 파운데이션 모델이라도 액션 인터페이스가 전신 휴머노이드용이 아니면 loco-manipulation에서 크게 뒤처짐을 보여주는 사례로 기능
- **SONIC / GMT / UniTracker / Humanoid-GPT** 계보: 저수준 전신 추종 컨트롤러 발전이 상위 정책의 "액션 latent 인터페이스"를 가능하게 한 전제. ω-0는 이 인터페이스 위에 VLA를 올린 구조

## 11. 향후 연구 방향

- **컨트롤러 교체 실험**: SONIC 외 GMT·UniTracker·Humanoid-GPT 등으로 저수준을 바꿔가며 latent 인터페이스의 이식성과 정책/컨트롤러 기여 분해 검증
- **공개 벤치마크 이식**: 시뮬 휴머노이드 loco-manipulation 벤치마크(예: RoboCasa 계열 humanoid 확장)에서의 수치 제공으로 외부 비교 가능성 확보
- **λ_video 및 미래 지평 K 스윕**: 미래 latent 예측 강도와 예측 길이가 제어/일반화에 미치는 trade-off 정량화
- **인간 데이터 스케일링 법칙**: SONIC 필터를 통과하는 인간 모션 비율과 최종 성능의 관계 — 현재 human transfer는 단일 태스크 실증에 그침
- **추론 가속**: distillation·consistency model로 DDIM 스텝을 줄여 7 Hz → 20~30 Hz 도달 시 외란 대응력 개선 기대
- **실패 모드 분석**: per-trial progress 표(Appendix C, Tables 6–16)가 제공되므로 어떤 subtask 경계에서 무너지는지 체계적 분류 가능. 균형 상실 vs 파지 실패 vs 언어 오해의 분해가 다음 개선점을 지목할 것
- **양팔·도구 접촉 힘 감각**: 현재 상태는 관절 위치와 root 자세뿐 — 접촉 힘/토크 피드백 통합이 mop·wipe 같은 접촉 풍부 태스크에 유효할 가능성

## 12. 종합 평가

"미래 예측을 비디오로 만들지 말고 latent로만 쓰라"는 단일 원칙을, 휴머노이드 동시 loco-manipulation이라는 가장 까다로운 실기 세팅에서 끝까지 밀어붙인 시스템 논문. prefix-guided dual-query attention과 token-specific RoPE(2D/3D/1D)로 미래 시각 query와 액션 query를 연결한 설계는 우아하며, SONIC 호환 66차원 전신 latent라는 통합 인터페이스 선택이 "상체는 정책·하체는 컨트롤러"라는 기존 절충을 실제로 넘어섰음을 81.8% vs ψ-0 44.5%로 보여준다. 특히 video query ablation이 in-distribution −14.6%p에서 cross-scene −64.5%p로 확대되는 패턴은 latent world modeling의 가치가 성능보다 **일반화**에 있다는, 이 분야에서 자주 주장되지만 드물게 측정되는 명제를 실기 데이터로 뒷받침한다. 반면 모든 수치가 자체 정의 태스크·자체 채점표 위에 서 있고 베이스라인은 저자들이 휴머노이드 액션 공간으로 개조한 버전이며(GR00T는 RTC 없이 평가), 태스크당 10회·일반화 설정은 1~3 태스크에 불과해 절대 격차의 크기(+37%p)는 액면 그대로 받기 어렵다. 코드·가중치·ω-HOME 미공개 상태에서 40.3시간 데이터셋의 기여도 아직 약속에 가깝다. VLA-Tracker에는 표준 벤치마크 점수가 없어 real_world 트랙(SR 81.8 / progress 90.3)으로만 등재되며, 리더보드 순위보다는 **휴머노이드 전신 액션 인터페이스와 latent 예측형 WAM의 설계 참조점**으로 읽는 것이 적절하다.

<!-- VERIFIED: pdf -->
