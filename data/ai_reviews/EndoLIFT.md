# EndoLIFT: Language-Disambiguated Latent-Conditioned Rectified Flow for Bidirectional Endoscopic Control

> **한 줄 요약**: 내시경의 전진(삽입)과 긴급 후퇴가 "거의 같은 영상에서 정반대 축 방향 행동"을 요구하는 문제를 *intent aliasing*으로 정식화하고, 언어 지시로 모드를 고르며 32-D 변분 궤적 잠재(VTL)로 rectified-flow 행동 생성을 조건화한 PaliGemma 2 기반 VLA. 폐루프 팬텀 실험에서 VTL 없는 동일 모델 대비 전체 성공률 +30%p(학습 대장 팬텀 70→100%, 미관측 폐·위 팬텀 70→100%), ex-vivo 돼지 기관 10/10.

- **arXiv**: 2608.20478v1 (2026-08-20, cs.RO)
- **소속**: The Chinese University of Hong Kong, The Sixth Affiliated Hospital of Sun Yat-sen University
- **코드**: 논문에 명시 없음

---

## 1. 배경 및 동기

위장관 내시경은 본질적으로 양방향이다. 목표 부위까지 전진한 뒤 철수하며 관찰하고, 생리 이상이나 구두 명령 같은 외부 신호로 조기 후퇴가 필요할 수도 있다. 요청 시점에 영상은 거의 변하지 않았는데 필요한 축 방향 행동은 반대가 된다. 관측만으로는 모드가 식별되지 않으므로, MSE 회귀는 두 모드를 평균하거나 우세 모드를 따르고, 생성형 정책도 "어느 모드를 실행할지"에 대한 증거가 없다.

## 2. 문제 정의

- o_t = (I_t, s_t): RGB 내시경 영상과 직전 정규화 행동.
- 모드 m_t ∈ {N(정상 항법), R(긴급 후퇴)}, 행동 청크 Ã_t ∈ R^{H×3}(축 방향, 상하 굽힘, 좌우 굽힘), a_fwd < 0 전진, > 0 후퇴.
- 동일 관측 이웃이 두 모드 모두에 등장하면 a*_MSE = q·v_R − (1−q)·v_N 로 평균화 → 조건부 비식별성.
- 해법: 언어 변수 ℓ_t ∈ {forward navigation, urgent retraction}로 p_θ(A_t | I_t, s_t, ℓ_t)를 학습.

## 3. 방법 개요

- 외부 트리거 감독기(생리 임계 모니터 + 2단계 음성 경로)가 두 개의 표준 지시 중 하나를 선택. 생리·음성 신호는 신경망 입력이 아니다.
- 하나의 공유 VLA가 영상·지시·직전 행동을 받아 3축 연속 청크를 생성.
- 지시가 바뀌면 현재 청크를 폐기하고 즉시 재계획.

## 4. 모델 구조

- **VLM prefix**: google/paligemma2-3b-pt-224 + LoRA. 프롬프트에 영상, 활성 지시, 행동 채널 정의, 다음 청크 요청을 넣고 마지막 hidden sequence P_t를 prefix로 사용.
- **Rectified-flow action expert**: 폭 1024, 8 블록, 16 헤드. 상태/행동 토큰 self-attention + VLM prefix cross-attention + FFN. AdaRMSNorm이 flow 시간 τ와 32-D VTL z로 조건화.
- **VTL**: 학습 시 posterior q_φ(z | P_t, s_t, A)가 전체 목표 청크를 풀링해 z를 샘플(reparameterization), 추론 시 z ~ N(0, I).
- **손실**: L = L_flow + β(s)·L_KL^FB, 차원당 0.5 nat free-bits, β는 500 step 동안 선형으로 1e-3까지 증가.
- **추론**: 20 Euler step으로 τ: 1→0 적분 → 32×3 청크, 최대 8개 실행 후 재계획. 호스트/모터 루프 20 Hz, VLA는 비동기 추론.

## 5. 학습 데이터와 절차

- 기본 모방 데이터 44,942 프레임, 185 에피소드(90/5/5 에피소드 분할, seed 42). 지시 문자열 4종(전진 3종 + 후퇴 1종).
- 이후 1,200 프레임으로 2,000 step의 지시 조건화 단계: 전진·후퇴 각 40개 paraphrase + 즉석 교란 + 20% 표준 앵커 혼합.
- **ModeFlag-LCRF** 참조: 동일 구조에서 지시 텍스트를 제거하고 1-bit 모드 플래그를 임베딩. EndoLIFT 체크포인트에서 시작해 frozen-teacher 행동 청크(대장 29,672 프레임)로 2 epoch 파인튜닝.

## 6. 실험 설정

- 로봇: 모터 구동 급송 장치(축 방향) + 2축 원위 굽힘, 480×480 팁 카메라.
- 폐루프: 대장 팬텀(학습 도메인), 폐·위 팬텀(미관측). 방법·조건 셀당 10회, 5개 방법 총 300회. Clopper–Pearson 95% CI, Fisher exact test.
- Ex-vivo: 돼지 기관에서 인간 트리거 후퇴 10회.
- 기준선: DINOv2 + MLP, Qwen3-VL + MLP, Qwen3-VL + GR00T(flow head), EndoLIFT w/o VTL(latent_dim=0 매칭 절제).

## 7. 주요 결과

**Table VI – 대장 팬텀(학습 도메인) 성공률 (%)**

| Method | Nav. | Retr. | Overall |
|---|---|---|---|
| DINOv2 + MLP | 0 | 0 | 0 |
| Qwen3-VL + GR00T | 0 | 0 | 0 |
| Qwen3-VL + MLP | 40 | 0 | 20 |
| EndoLIFT w/o VTL | 100 | 40 | 70 |
| **EndoLIFT** | **100** | **100** | **100** |

**Table VIII – 미관측 폐·위 팬텀 (%)**: EndoLIFT 네 조건 모두 100, 전체 100. w/o VTL은 후퇴 폐 30 / 위 50, 전체 70. Qwen3-VL + MLP 전체 32.5. 후퇴 합산 20/20 vs 8/20 (p = 4.51×10⁻⁵).

**Table V – 오프라인 방향성**: 항법 방향 정확도 0.565 vs w/o VTL 0.453 (+11.1%p), 후퇴 목표에서 잘못된 전진 0.010 vs 0.059 (83% 감소).

**Table III – 지시 추종 정확도(IFA, %)**: 44개 미관측 표현 전체 82.8 (w/o VTL 78.2, Qwen3-VL + GR00T 63.1, Qwen3-VL + MLP 59.7). 키워드/TF-IDF 라우터 + 표준 정책 파이프라인은 53.2 / 54.8.

**Table II – 텍스트 vs 1-bit**: IFA 85.0 vs 68.8, canonical paired flip 70.0 vs 37.5.

**Table VII – Ex-vivo**: 10/10 성공, 트리거→지시 전환 445±126 ms, 전환→첫 후퇴 54±61 ms(약 1 제어 스텝), 트리거 후 축 우세 스텝 1,127/1,127 모두 후퇴, 인간 개입 0.

## 8. Related Work 상의 위치

- EndoVLA(프롬프트 추적), BiliVLA(항법 + RL), EndoWAM(월드-액션 모델) 등 내시경 VLA가 과제·단계 조직에 지시를 쓰는 반면, EndoLIFT는 지시를 "동일 관측에서 반대 축 모드를 고르는 인과 변수"로 좁게 사용한다.
- π0 계열의 flow action expert에 VAE식 궤적 잠재를 결합한 형태로, 다중모달성 표현과 모드 선택을 분리해 분석한다.

## 9. 강점

- 문제(intent aliasing)의 정식화가 명확하고, 동일 관측·동일 노이즈 시드 개입 실험으로 "언어가 모드를 고른다"는 주장을 직접 검증한다.
- VTL 유무만 다른 매칭 절제와 Fisher exact test로 기여를 통계적으로 분리한다.
- 지시 민감도(Table IV)와 제어 성능이 다른 성질임을 솔직하게 보고(w/o VTL이 더 민감하지만 성공률은 낮음).
- 미관측 해부 구조 팬텀과 ex-vivo 조직까지 폐루프로 검증하고 지연을 정량화.

## 10. 약점 및 한계

- 셀당 10회 시행으로 신뢰구간이 넓다(10/10의 CI 69.2–100%). 저자도 인정.
- 오프라인 항법 방향 정확도 0.565는 절대값이 낮아, 폐루프 성공이 재계획에 크게 의존함을 시사.
- 과제가 "전진/후퇴 두 모드"로 한정되어 있고 실제 임상 전 과정 자율성은 주장하지 않는다(Level 2 과제 자율성).
- 힘 측정이나 조직 안전성 지표가 없다.
- 기준선이 자체 구현이며 공개 코드가 명시되지 않았다.

## 11. 재현 및 확장 아이디어

- 삽입–관찰 워크플로의 더 많은 모드(retroflexion, 랜드마크 검사)로 지시 공간 확장.
- 힘/형상 센싱을 상태 입력에 추가해 안전 제약 학습.
- VTL을 모드별 사전분포(조건부 prior)로 바꿔 모드 선택과 궤적 다양성의 결합 분석.
- 더 큰 반복 폐루프 연구와 동기화된 영상·행동·트리거·힘 기록.

## 12. 총평

의료 로봇이라는 좁은 도메인이지만, "관측으로 식별되지 않는 모드를 언어로 명시하고, 궤적 잠재로 실행 품질을 높인다"는 두 기여를 깔끔하게 분리해 검증한 논문이다. 표본 수는 작지만 매칭 절제와 통계 검정이 탄탄하고, 결과(후퇴 성공 40→100%)가 뚜렷하다.

**한 문장 요약**: PaliGemma 2 + VTL 조건 rectified-flow VLA로 내시경 전진/긴급 후퇴의 intent aliasing을 해결해 팬텀·ex-vivo 폐루프에서 100% 성공을 보인 연구.

<!-- VERIFIED: pdf -->
