# AnchorVLA: Bridging Discrete Decisions and Continuous Trajectories for Vision-Language-Action Planning

> **한 줄 요약**: 자율주행 VLA 플래닝에서 좌표 단위 waypoint 토큰 대신 "궤적 패턴 앵커"(K=100 k-means 코드북)를 LLM의 행동 어휘로 도입하고(DAAR), 선택된 앵커가 정의하는 잔차(residual) 공간에서 flow matching으로 연속 궤적을 생성(DARF)하여, Bench2Drive closed-loop에서 SOTA Success Rate 77.28 / Driving Score 89.92를 달성한 계층적 결정-앵커 VLA 플래너.

---

## 1. 배경 및 동기

### 기존 VLA 주행 플래너의 두 패러다임과 한계
- **Planning-head 기반** (ORION, SimLingo 등): LLM/VLM 특징을 독립적인 플래닝 헤드의 조건으로만 사용 → 연속 궤적 공간이 VLA 추론에 의해 **약하게만 제약**되어, 국소 기동(maneuver)이 고수준 결정과 일관되게 정렬되기 어려움
- **Full-trajectory autoregressive** (AutoVLA, LinkVLA 등): 궤적을 좌표/waypoint 토큰으로 이산화하여 순차 생성 → 각 토큰이 **저정보밀도의 기하학적 점**에 불과해 긴 시퀀스, 이산화 오차 누적, 의미-행동 정렬 약화, 느린 추론(LinkVLA-AR 기준 361ms 추가 지연)

### 핵심 질문
- **고수준 언어 추론과 저수준 연속 궤적 실행 사이의 추상화 간극(abstraction gap)을 어떻게 메울 것인가?**
- LLM의 강점인 이산적 추상화·의미적 결정 능력을 살리면서도 연속 생성의 유연성을 유지할 수 있는가?

## 2. 방법론 심층 분석

### 2.1 전체 구조: 결정-후-실행 분해

궤적 분포를 앵커에 대해 분해:

$$p(\tau|x) = \sum_{k=1}^{K} p(a_k|x)\, p(r_k|x, a_k), \quad \tau = a_k + r_k$$

- $a_k$: 궤적 패턴 앵커(완전한 국소 모션 패턴 — 차선 유지, 제동, 양보, 회전, 추월 등)
- $r_k$: 앵커 중심 좌표계에서 정의되는 연속 잔차
- 앵커는 단순 조건이 아니라 **잔차 좌표계와 국소 행동 생성 공간을 정의하는 명시적 결정 인터페이스**

### 2.2 VLA 백본
- SimLingo 백본 기반: InternVL2-1B (비전: InternViT-300M, 언어: Qwen2-0.5B-Instruct)
- 입력: 언어 프롬프트, 전방 카메라 이미지 타일, 내비게이션(GPS target point를 MLP로 토큰 임베딩 투영)
- 백본은 변경하지 않고 **플래닝 인터페이스만 교체**

### 2.3 DAAR (Decision-as-Anchor Representation)
- 학습 데이터 궤적의 k-means 클러스터링으로 **K=100 궤적 패턴 코드북** 구축
- GT 궤적과 각 앵커의 평균 L2 거리 기반 **soft anchor target** (top-N 최근접 앵커에 온도 γ softmax 가중치) → 패턴 공간의 국소 연속성 보존
- 두 가지 결정 모델링 방식:
  - **Query-based**: 학습 가능한 쿼리를 시퀀스에 추가, MLP 헤드로 앵커 분류 (LLM next-token 인터페이스 미사용)
  - **Autoregressive**: LLM 어휘에 K개 앵커 특수 토큰 추가(앵커↔토큰 1:1 매핑), control token 위치의 logits를 앵커 토큰 부분집합으로 제한하여 분포 산출. 좌표 토큰과 달리 **각 토큰이 완전한 국소 궤적 패턴**을 표현
- 추론 시 top-M=6 후보 앵커 선택으로 행동 수준 다중성(multimodality) 보존

### 2.4 DARF (Decision-Anchored Residual Flow)
- 각 후보 앵커에 대해 잔차 목표 $r_{gt}^k = \tau_{gt} - a_k$를 설정하고 **앵커 정의 잔차 공간에서 flow matching**: $z_t^k = (1-t)\epsilon + t r_{gt}^k$
- GT에 가장 가까운 매칭 앵커에만 L1 flow loss 적용(matched-anchor supervision) → 여러 행동 가설 유지
- **confidence branch**: 앵커-컨텍스트 적합도 점수 $s_k$ 예측(cross-entropy로 매칭 앵커 감독), 추론 시 최종 궤적 선택
- 디코더: 속도(velocity) 분기와 신뢰도 분기의 **분리된 2-branch 구조**. 속도 분기는 time-feature affine modulation + 앵커 positional bias 쿼리로 멀티모달 컨텍스트에 cross-attention; 신뢰도 분기는 $z_t$, $t$를 사용하지 않고 앵커 특징만으로 컨텍스트 적합도 평가
- 소수의 Euler step으로 잔차 적분: 앵커가 좋은 참조를 제공하면 잔차 목표가 전체 궤적보다 콤팩트 → **적은 스텝의 flow 적분이 용이**

## 3. 실험 설정

- **벤치마크**: Bench2Drive (CARLA 기반 closed-loop), 지표: Driving Score(DS), Success Rate(SR), Efficiency, Comfort, Multi-Ability
- **학습 데이터**: SimLingo 세팅 준수 — expert 궤적, instruction-conditioned dream 궤적, 혼합 언어 감독(VQA, driving commentary, no-language)
- **Expert**: PDM-Lite
- **학습**: 2단계 — (1) DAAR 앵커 결정 모델 15 epoch (8×A100, batch 16), (2) 백본 동결 후 DARF 15 epoch (4×A100, batch 32)
- 앵커 K=100, 후보 M=6, flow step 2

## 4. 주요 결과

### Bench2Drive Closed-loop (Table 1)
| Method | Expert | DS | SR(%) | Efficiency | Comfort |
|---|---|---|---|---|---|
| SimLingo | PDM-Lite | 85.07 | 67.27 | 259.23 | 33.67 |
| LinkVLA | PDM-Lite | **91.01** | 74.55 | 255.84 | 34.62 |
| BridgeDrive | PDM-Lite | 87.99 | 74.99 | 236.49 | 20.98 |
| **AnchorVLA** | PDM-Lite | 89.92 | **77.28** | 251.14 | 28.94 |

- **SR 77.28로 전체 비교 대상 중 최고** (closed-loop 태스크 완수 신뢰성 직접 측정)
- DS는 LinkVLA에 근소하게 뒤지나(89.92 vs 91.01) SR은 74.55→77.28 개선; BridgeDrive 대비 DS/SR 모두 우위
- Comfort 28.94는 일부 baseline보다 낮음 — 유연한 생성적 정제와 궤적 평활성 간 트레이드오프 존재

### Multi-Ability (Table 2)
| Method | Merging | Overtake | Brake | Give-Way | Traffic-Sign | Mean |
|---|---|---|---|---|---|---|
| LinkVLA | 60.00 | 80.00 | **93.33** | 50.00 | 83.68 | 73.40 |
| BridgeDrive | **69.92** | 66.67 | 90.00 | 50.00 | **89.47** | 73.15 |
| **AnchorVLA** | 65.00 | **81.11** | 90.00 | 50.00 | 85.00 | **74.22** |

- **평균 74.22로 최고**, 특히 능동적 상호작용 시나리오인 Overtake에서 81.11로 최고

## 5. Ablation 분석

### 결정 모델링 방식 (Table 3, flow step 2 고정)
| Decision | 추가 지연 | DS | SR(%) |
|---|---|---|---|
| LinkVLA-AR (full-traj AR) | 361 ms | 89.57 | 73.18 |
| None (전체 앵커 → DARF) | – | 87.49 | 70.91 |
| Query-based | 33 ms | 88.67 | 73.81 |
| **Autoregressive** | 64 ms | **89.92** | **77.28** |

- 명시적 앵커 결정 자체가 이득(None→Query +2.9 SR), LLM 토큰 공간에서의 AR 앵커 예측이 추가 이득(+3.47 SR)
- **full-trajectory AR 대비 지연 361ms→64ms (약 5.7배 절감)하면서 DS/SR 모두 상회** — 이득이 단순 앵커 사용이 아니라 고수준 결정과 앵커 예측을 동일 LLM 토큰 공간에서 긴밀히 결합한 데서 옴

### Flow matching 형식 (Table 4, AR 결정 고정)
| Formulation | 추가 지연 | Flow Step | DS | SR(%) |
|---|---|---|---|---|
| Deterministic residual | 8 ms | 1 | 86.74 | 70.45 |
| Full-trajectory flow | 12 ms | 2 | 88.34 | 72.73 |
| **DARF (residual flow)** | 18 ms | 2 | **89.92** | **77.28** |

- 결정론적 잔차 회귀는 다중 가능 정제를 평균화하여 불충분; 동일 flow step에서 잔차 공간 flow가 전체 궤적 공간 flow보다 우수 (수송 경로가 짧아 제한된 스텝 적분에 유리)

### 내비게이션 모달리티 (Table 5)
- GPS target point(DS 89.92/SR 77.28) vs 내비게이션 명령(DS 90.26/SR 76.82) — 거의 동등, 입력 형태에 강건

## 6. 기존 연구와의 비교

| 축 | Planning-head (SimLingo, ORION) | Full-traj AR (AutoVLA, LinkVLA) | AnchorVLA |
|---|---|---|---|
| LLM 역할 | 조건 인코더 | 좌표 토큰 순차 생성 | 행동 수준 앵커 토큰 결정 |
| 토큰 정보 밀도 | – | 낮음(좌표점) | 높음(국소 모션 패턴 전체) |
| 연속성 | 유지(약한 제약) | 이산화 오차 | 잔차 flow로 유지 |
| 추가 지연 | 낮음 | 높음(361ms) | 낮음(~64+18ms) |

- VADv2의 planning vocabulary, DiffusionDrive의 anchored diffusion과 개념적 친연성이 있으나, **앵커 선택을 LLM의 next-token 인터페이스에 통합**하고 잔차 공간 flow matching과 결합한 점이 차별점
- confidence 기반 후보 선택은 DiffusionDrive의 denoising decoder 선택 전략을 따름 (부록 B에 명시)

## 7. 강점

1. **명확한 문제 정의와 구조적 해법**: 추상화 간극을 "결정(이산 앵커) + 실행(연속 잔차)"으로 분해하는 수식화(Eq. 2)가 깔끔하고, 실제 운전의 decision-then-execution 구조와 부합
2. **효율-성능 동시 개선**: full-trajectory AR 대비 지연 5.7배 절감하면서 SR 최고치 달성
3. **철저한 ablation**: 결정 모델링(4종), flow 형식(3종), 내비게이션 모달리티까지 각 설계 선택의 기여를 분리 검증
4. **soft anchor target**: hard label 대신 top-N 최근접 앵커 soft 분포로 패턴 공간의 국소 연속성 보존 — query/AR 두 방식에 통일된 감독
5. **정직한 한계 보고**: Comfort 트레이드오프, 실패 사례(앵커 오예측의 오류 전파) 분석 포함

## 8. 한계 및 비판적 검토

1. **앵커 오류 전파**: DARF는 선택된 앵커의 행동을 "정제"할 뿐 "번복"하지 못함 → 앵커가 off-route거나 언어 지시와 불일치하면 최종 궤적도 잘못됨 (부록 A.2에서 저자 스스로 인정, reflection 메커니즘을 향후 과제로 제시)
2. **DS는 SOTA 아님**: LinkVLA(91.01)에 뒤짐 — SR 중심 서사이며, 종합 주행 품질에서는 여전히 격차 존재
3. **Comfort 저하**: 28.94로 LinkVLA(34.62)·SimLingo(33.67) 대비 낮음 — 생성적 정제의 평활성 비용
4. **단일 벤치마크**: Bench2Drive(CARLA)만 평가. nuScenes/NAVSIM 등 실데이터 open-loop나 실차 검증 부재
5. **코드북 고정성**: K=100 k-means 앵커가 학습 데이터 분포에 종속 — 분포 외 기동이나 새로운 도로 위상에서의 일반화 미검증; K, M, γ에 대한 민감도 분석 없음
6. **소형 백본**: Qwen2-0.5B 기반 — "LLM 추론 능력" 주장 대비 실제 언어 추론의 기여가 CoT 토큰을 통해 얼마나 되는지 분리 검증 없음
7. **오픈소스 여부 불명**: 코드 공개 언급 없음

## 9. 예상 질문 및 답변

> ❓ **K=100 앵커로 충분한가? 앵커 수를 늘리면?**
> **답변**: 논문에 K 민감도 분석이 없다. K가 크면 앵커당 잔차가 작아져 flow가 쉬워지지만 분류가 어려워지고 AR 어휘가 커지는 트레이드오프가 예상된다. "None" ablation(전체 앵커를 DARF에 투입, 70.91 SR)이 시사하듯 결정 모델의 선별이 핵심이므로, K 증가가 단조 개선을 보장하지 않을 것.

> ❓ **로봇 조작(manipulation) VLA에도 이전 가능한가?**
> **답변**: 원리적으로 가능 — 조작 스킬(잡기, 밀기, 들기)도 패턴 클러스터링이 가능하다. 다만 주행 궤적은 2D(T×2)로 저차원인 반면 조작은 고차원 관절/EE 공간이라 k-means 코드북의 품질과 잔차 콤팩트성이 유지될지는 별도 검증 필요.

> ❓ **anchored diffusion(DiffusionDrive)과의 본질적 차이는?**
> **답변**: DiffusionDrive는 앵커를 diffusion의 초기화/조건으로 쓰는 비-VLA 플래너인 반면, AnchorVLA는 앵커 선택 자체를 LLM의 next-token 예측으로 수행하여 언어 추론(CoT)과 행동 결정을 동일 토큰 공간에서 결합한다. Table 3에서 이 결합이 query-based 대비 +3.47 SR의 실질적 이득임을 보였다.

> ❓ **flow step 2로 충분한 근거는?**
> **답변**: 잔차 목표가 전체 궤적보다 콤팩트해 수송 경로가 짧다는 논리(4.4절). 동일 2 step에서 full flow(72.73 SR) 대비 DARF(77.28 SR)가 우수함이 간접 증거이나, step 수 스윕은 제공되지 않았다.

## 10. 향후 연구 방향

- **Reflective anchor selection**: 언어 추론·내비 의도·장면 맥락·후보 앵커 간 일관성을 재평가해 잘못된 앵커를 수정하는 반성 메커니즘 (저자 제안)
- Comfort 개선을 위한 smoothness-aware 목적함수/제약을 DARF에 통합 (저자 제안)
- 실데이터(NAVSIM, nuScenes) 및 실차 검증, 코드북의 적응적/계층적 구성
- 조작 도메인으로의 확장: 스킬 수준 앵커 + 잔차 flow의 일반 VLA 적용

## 11. 세미나 토론 주제

1. 앵커 코드북(이산 결정)과 잔차 flow(연속 실행)의 분해는 로봇 조작 VLA의 "discrete skill token + continuous refinement" 설계와 어떻게 수렴/분기하는가?
2. DARF가 앵커를 번복하지 못하는 구조적 제약 — 계층적 플래닝에서 하위 레벨에 얼마나 "거부권"을 주어야 하는가?
3. SR과 DS 중 무엇이 closed-loop 주행 품질의 더 나은 대리 지표인가? Comfort 하락은 수용 가능한 비용인가?
4. 0.5B 규모 LLM에서 "언어 추론이 행동 결정을 이끈다"는 주장은 어디까지 유효한가?

## 12. 총평

AnchorVLA는 VLA 주행 플래닝의 핵심 병목인 "언어 추론 ↔ 연속 궤적" 추상화 간극을, 행동 수준 궤적 패턴 앵커라는 중간 인터페이스로 해소한 잘 설계된 연구다. 좌표 토큰 AR의 지연·이산화 문제와 planning-head의 약한 결합 문제를 동시에 겨냥했고, ablation이 각 설계(명시적 앵커 결정, LLM 토큰 공간 통합, 잔차 공간 flow)의 기여를 설득력 있게 분리했다. Bench2Drive SR 77.28은 의미 있는 SOTA이나, DS는 LinkVLA에 뒤지고 Comfort 저하와 단일 시뮬레이션 벤치마크 평가는 한계다. 앵커 오류 전파라는 계층 구조 고유의 취약점을 스스로 분석하고 reflection을 후속 과제로 제시한 점은 성실하다. "이산 결정 + 연속 잔차 생성"이라는 설계 패턴 자체는 주행을 넘어 일반 VLA 액션 헤드 설계에 시사점이 크다.

**평점 근거 (8.0/10)**: 명확한 문제 인식과 우아한 분해(+), SR SOTA와 대폭의 지연 절감(+), 철저한 ablation(+); DS 비-SOTA와 Comfort 트레이드오프(-), 단일 벤치마크·코드 미공개·하이퍼파라미터 민감도 부재(-).

<!-- VERIFIED: pdf -->
