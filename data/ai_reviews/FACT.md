# FACT: Demystifying When and Why VLAs Fail in Contact-Rich Tasks and How to Fix Them

> **한 줄 요약**: 접촉이 많은(contact-rich) 조작에서 VLA가 실패하는 이유를 **정밀도 실패(precision failure)** 와 **힘 실패(force failure)** 라는 인과적으로 독립된 두 모드로 분해하고, 전자는 flow-matching **노이즈 스케줄 교체(Beta → Logit-Normal, m=1.5)**, 후자는 **시간 인식 힘 주입(time-aware force injection)** 으로 각각 처방한 뒤 둘을 합쳐 **FACT**를 제안. 5개 실물 접촉 과제에서 평균 성공률 **66.0%**로, 최고 선행 베이스라인 ForceVLA(40.5%)·TA-VLA(37.5%) 및 pi0.5 베이스라인(39.0%)을 25%p 이상 상회 (실물 롤아웃 약 2,500회).

- arXiv: 2608.01402v1 (2026-08-02, cs.RO)
- 저자: Carlota Parés-Morlans, Nils Kuhn, Isabel Liu, Alberta Longhini, Jeannette Bohg — Stanford University (IPRL Lab)
- Project Page: https://stanford-iprl-lab.github.io/fact/

---

## 1. 배경 및 동기

### 접촉이 많은 조작이 여전히 미해결인 이유
- VLA는 물체 회수, 셔츠 접기 같은 자유 공간(free-space) 조작에서 큰 진전을 이뤘지만, **커넥터 삽입·정밀 조립**처럼 지속적인 힘 조절이 필요한 영역은 여전히 취약하다.
- 접촉 과제는 (a) 부품 기하·재질·표면 컴플라이언스에 따라 접촉력이 변하는 **연속적 force-regulated interaction**을 요구하고, (b) 정밀 보정이 가장 중요한 순간에 하필 **커넥터 몸체가 포트를 가려 시각 피드백이 붕괴**한다.

### 선행 연구의 두 가지 암묵적 가정
힘 피드백을 붙이는 계열(ForceVLA의 전용 fusion MoE, TA-VLA의 토크 인식, 학습 시 보조 감독을 쓰는 FACTR/CRAFT 계열)은 공통적으로:
1. **접촉 격차를 "센싱 문제"로만 프레이밍**한다 — 학습 절차 자체가 원인일 가능성을 검토하지 않음.
2. 힘을 넣을 때도 **시각·고유수용감각 옆에 그냥 이어붙인다(naive concatenation)** — 힘 신호 고유의 통계적 성질을 활용하지 않음.

### 본 논문의 주장
"실패의 근본 원인을 먼저 진단하라." 저자들은 접촉 실패가 **하나가 아니라 두 개의 인과적으로 분리된 모드**임을 보이고, 각 모드에 정확히 대응하는 개입을 설계한다.

---

## 2. 실패 모드 진단 (1): 정밀도 실패 — flow-matching 학습 불균형

플러그 삽입에서 힘 크기 |F|의 시간 프로파일을 보면 성공 궤적은 접근(|F|=0) → 접촉·정렬 탐색(|F|>0) → 소켓 진입(|F|≈0) → 완전 착좌 시 급격한 힘 상승(t1)의 패턴을 갖는다. **정밀도 실패**는 소켓 입구에서 정렬이 어긋나 플러그가 림을 누르며 t2에서 힘이 상승·포화하는 형태로 나타난다.

원인은 센싱이 아니라 **학습 신호 배분**이다.
- flow-matching 정책은 노이즈 레벨 τ를 어떤 스케줄에서 샘플링해 학습하는데, 접촉 보정에 해당하는 작고 정확한 액션은 **저노이즈 영역(τ < 0.2)** 에서 생성된다.
- 그러나 널리 쓰이는 Beta(1.5, 1.0) 스케줄은 SmolVLA 기준 τ < 0.2에 **전체 gradient 신호의 8.9%만** 배분한다 (Figure 3a).
- 즉 delta collapse(접촉 구간에서 액션 델타가 작아짐)와 training starvation(그 구간의 학습 신호 부족)이 겹쳐, 정책은 **정확히 필요한 곳에서 undersupervised** 상태가 된다.

📌 핵심: 이 불균형은 데이터·아키텍처 변경 없이 **노이즈 스케줄만으로 완전히 교정 가능**하다.

---

## 3. 실패 모드 진단 (2): 힘 실패 — 힘 신호의 구조적 특성

**힘 실패**는 플러그가 소켓에 제대로 들어갔는데도 정책이 완전 삽입 직전에 멈춰 t3의 착좌 힘 상승이 관측되지 않는 형태다. 힘 센서가 있어도 이런 실패가 나는 이유는 힘 신호의 세 가지 성질 때문이다.

1. **접촉 희소성(contact sparsity)** — 힘은 대부분의 자유공간 스텝에서 거의 0이고 짧은 접촉 구간에서만 정보를 갖는다. 단순 concatenation 시 목적함수가 0에 가까운 샘플에 지배되어 gradient가 **힘 입력을 아예 무시하는 방향**으로 편향된다.
2. **시간 구조(temporal structure)** — 순간 측정치는 현재 상호작용 상태를, 최근 이력은 그에 이르게 한 동역학(과도응답, 누적 힘 상승)을 인코딩한다. 둘 중 하나를 빠뜨리면 과제 관련 접촉 정보가 소실된다.
3. **민감도 변조(sensitivity modulation)** — 자유공간에서는 힘을 사실상 무시해야 하고, 접촉 시에는 작은 편차에도 보정이 촉발되어야 한다. 즉 힘은 **입력이 아니라 변조자(modulator)** 여야 한다.

---

## 4. 처방 (1): Logit-Normal 노이즈 스케줄

post-training 단계에서 Beta 스케줄을 Logit-Normal(LN) 스케줄로 교체한다.

```
f_T(τ) = 1/(s√(2π)) · 1/(τ(1-τ)) · exp( -(logit(τ) + m)² / (2s²) )
τ = σ(s·z - m),  z ~ N(0, 1)
```

- Esser et al.(SD3)의 LN 스케줄을 가져오되, 위치 파라미터를 이미지 생성용 **m=0이 아니라 m=1.5**로 두어 확률 질량을 τ=0 쪽으로 이동.
- 결과적으로 τ < 0.2 구간에 Beta 대비 **6배의 gradient 신호**를 배분.
- **파라미터 증가 0, 추가 데이터 0, 아키텍처 변경 0** — 모든 flow-matching VLA에 drop-in 적용 가능.

---

## 5. 처방 (2): 시간 인식 힘 주입 (Time-Aware Force Injection)

세 가지 설계 요소가 §3의 세 성질에 1:1 대응한다.

**(a) Contact state — 민감도 변조 대응.** 최근 센서 윈도우 f_t ∈ R^(H_w×6)를 mean-pooling해 6차원 요약 f̄_t를 얻고, 2층 MLP를 거쳐 레이어별 스케일 변조량으로 투영:
```
Δγ(f̄_t) = W_γ · φ_force(f̄_t) ∈ R^d
h_l = (γ_l(τ) + Δγ(f̄_t)) · RMSNorm(h_{l-1}) + β_l(τ) + h_{l-1} g_l(τ)
```
Δγ는 **모든 레이어에 공유**되며 단일 forward pass로 전 레이어의 민감도를 조정한다 (AdaRMSNorm).

**(b) Contact history — 시간 구조 대응.** 직전 H=30 스텝(≈2초)의 F/T 판독값을 공유 causal TCN 인코더로 각각 인코딩해 토큰으로 만들어 action expert 입력 앞에 prepend. 접촉의 *궤적*에 대해 추론 가능.

**(c) Contact gating — 접촉 희소성 대응.** 임계값 δ(=0.5 N) 미만으로 접촉이 감지되지 않는 스텝에서는 힘 인코딩 경로의 **gradient를 차단**하여, 인코더가 무의미한 0 근처 판독값에 과적합되는 것을 방지.

두 처방을 합친 것이 **FACT (Force-Aware Contact-rich manipulation via Timestep modulation)** 이다.

---

## 6. 구현 및 학습 세부

- 하드웨어: Franka Research 3 + Robotiq 그리퍼, 손목 장착 **Bota SensONE** 6축 F/T 센서. 외부 Realsense D435 + 손목 ZED Mini(둘 다 15 Hz). F/T는 400 Hz 취득 후 one-euro filter 처리 → 제어 스텝당 H_w = ⌈400/15⌉ = 27개 원시 판독값.
- 제어기: 15 Hz 정책 업데이트와 1 kHz 관절 토크 루프를 분리한 **2-rate operational-space control**. 궤적 생성 → Cartesian impedance PD (a_u = a_des − k_p(x−x_des) − k_v(v−v_des)) → f = Λ(q)a_u, τ_u = J^T(q)f.
- 학습: 과제당 **100개 원격조작 시연**(Haply Inverse 3 햅틱 장치, 조작자에게 힘 피드백 제공). 목표 위치는 32×20 cm 표면에서, 홈 위치는 5 cm 큐브 내에서 균등 샘플링.
- 모든 방법(베이스라인 포함)이 **동일한 pre-trained pi0.5 체크포인트에서 LoRA로 20,000 스텝** 파인튜닝. 액션 공간은 delta end-effector pose로 통일.
- 추가 파라미터: FACT ≈2.2M (causal TCN ≈0.2M — 4 dilated block, hidden 64; AdaRMS conditioning head ≈2.1M, zero-init γ projection) vs ForceVLA ≈45M(대부분 LIMoE), TA-VLA ≈2.1M.

---

## 7. 실험 설정

**5개 실물 접촉 과제**:
| 과제 | 유형 | 난점 |
|---|---|---|
| Plug insertion | 정밀도 critical | 2핀 플러그, 커넥터 몸체가 구멍을 가림, 마찰 끼워맞춤 극복 위해 지속 힘 필요 |
| USB insertion | 정밀도 critical | USB-A(NIST 조립 보드), 삽입 중 포트 완전 가림 |
| Button push | 힘 critical | 힘 임계 도달 시 잠김; 최대 힘으로 밀면 실패하도록 시연 수집 |
| Board erasing | 힘 critical | 스트로크 내내 일정 접촉 유지, 과소=잔여물 / 과대=표면 손상 |
| Key insertion | 힘 critical | 완전 가림 + **길이가 다른 시각적으로 동일한 열쇠** + Gaussian blur(σ=2)로 시각 단서 차단, hard-stop 힘 서명 인식 필요 |

**프로토콜**: 방법·과제당 **40회 독립 롤아웃**, 60초 타임아웃 내 완수 시 성공. 전체 조건 합산 약 **2,500회 실물 롤아웃**. 유의성은 pi0.5 베이스라인 대비 **Fisher's exact test**.

**베이스라인**: pi0.5 원본, ForceVLA, TA-VLA (후자 둘은 원래 pi0 기반이지만 **동일 pi0.5 백본으로 재구현**하여 통제 비교).

---

## 8. 주요 결과

**Table 1 (pi0.5 백본, SR %)** — 좌→우: plug / USB / button / board / key / All

| Method | Plug | USB | Button | Board | Key | All |
|---|---|---|---|---|---|---|
| pi0.5 | 30.0 | 37.5 | 12.5 | 100.0 | 15.0 | **39.0** |
| pi0.5 + LN | 50.0 | 47.5 | 57.5 | 87.5 | 37.5 | **56.0** |
| **FACT** | 57.5 | 47.5 | 75.0 | 90.0 | 60.0 | **66.0** |
| ForceVLA (pi0.5) | 32.5 | 37.5 | 12.5 | 77.5 | 42.5 | **40.5** |
| TA-VLA (pi0.5) | 30.0 | 25.0 | 20.0 | 97.5 | 15.0 | **37.5** |

- **LN 단독 효과**: plug +20 pp (p=.055), USB ~+10 pp, button +45 pp (p<.001), key +22.5 pp (p=.020). 추가 데이터·파라미터 없이 얻은 이득.
- **힘 주입 추가 효과**: button +17.5 pp, key +22.5 pp (둘 다 pi0.5 대비 p<.001). 반대로 plug/USB에서는 **통계적으로 유의한 이득 없음** — 이는 정밀도 과제의 실패가 "힘 피드백 부재"가 아니라 "denoising 시간 부족"임을 역으로 증명한다.
- **Board erasing은 이상치**: 모든 방법이 포화(77.5~100%). 지속적 표면 접촉은 compliant operational-space 제어기가 이미 처리하며, 드문 실패는 시각 정렬 오차로 마크 일부만 지우는 경우.

**Table 4 (pi0 백본)**: FACT_pi0 = plug 70.0 / key 50.0 / button 60.0 로 pi0(32.5/45.0/47.5), ForceVLA_pi0(55.0/42.5/42.5), TA-VLA_pi0(30.0/27.5/27.5)를 모두 상회. 백본 간 전이 확인. 흥미롭게 pi0.5는 힘 critical 과제에서 +10 pp 이상 우세, pi0는 정밀도 과제에서 +12.5 pp 우세.

---

## 9. Ablation 및 메커니즘 분석

**Table 3 — 힘 컴포넌트 제거 (plug / key / button)**

| Method | Plug | Key | Button |
|---|---|---|---|
| FACT | 57.5 | 60.0 | 75.0 |
| w/o gradient threshold | 42.5 | 27.5 | 57.5 |
| w/o current reading | 60.0 | 35.0 | 50.0 |
| w/o history | 30.0 | 20.0 | 12.5 |

- **힘 이력이 가장 중요** — button −62.5 pp, key −40 pp. 시간 통합이 힘 critical 과제의 핵심.
- **순간 판독값**은 상대적으로 기여가 작지만 key insertion(−25 pp)에서는 hard-stop 피크가 이력만으로는 얻을 수 없는 완료 신호를 제공.
- **contact gating**은 힘 critical 과제에서 필수적(key −32.5 pp).
- 흥미롭게 w/o current reading에서 plug가 오히려 60.0으로 소폭 상승 — 정밀도 과제에서 순간 힘 변조가 필수적이지 않음을 시사.

**Table A.1 — 노이즈 치환 대조 실험 (F/T를 i.i.d. Gaussian noise로 대체하여 재학습)**

| Method | Plug | Key | Button |
|---|---|---|---|
| FACT | 57.5 | 60.0 | 75.0 |
| FACT force→noise | 40.0 (p=.090) | 5.0 (p<.001) | 17.5 (p<.001) |
| ForceVLA | 32.5 | 42.5 | 12.5 |
| ForceVLA force→noise | 40.0 | 15.0 (p=.006) | 17.5 |
| TA-VLA | 30.0 | 15.0 | 20.0 |
| TA-VLA force→noise | 22.5 | 25.0 | 12.5 |

이 실험이 이 논문에서 가장 날카로운 부분이다. 입력 차원과 경로를 유지한 채 과제 종속적 힘 정보만 제거했을 때, FACT는 힘 critical 과제에서 붕괴(key 60→5)하지만 **TA-VLA는 유의한 변화가 없다** — 즉 TA-VLA의 단일 토큰 압축 이력은 실제로 힘 정보를 추출하지 못하고 있으며, 그 성능 이득의 상당 부분은 **추가 입력에 의한 정규화 효과**에 불과했다는 뜻이다.

**Table 2 — LN의 범용성**: 다른 힘 아키텍처에 LN만 얹어도 개선. TA-VLA 30.0/15.0/20.0 → 47.5/32.5/42.5, ForceVLA 32.5/42.5/12.5 → 57.5/30.0/30.0 (plug/key/button). Beta 스케줄 불균형이 **선행 연구 전반의 공통 병목**임을 시사.

**Appendix F — m 스윕**: m ∈ {−1.5, −0.5, 0.5, 1.5}를 s=1 고정으로 plug insertion에서 스윕(각 40 롤아웃). m이 커져 τ<0.2로 신호가 이동할수록 성공률이 급격히 상승 (수치는 Figure A.3b에만 제시).

---

## 10. 관련 연구와의 비교

**Table A.2 — 힘 증강 VLA 아키텍처 비교 (모두 pi0.5 + LoRA)**

| | FACT | ForceVLA | TA-VLA |
|---|---|---|---|
| 힘 인코더 | 공유 causal TCN + 2층 MLP | Linear proj. + LIMoE | 2층 MLP |
| 주입 지점 | Action Expert (AdaRMS + 토큰) | VLM/AE 브리지 (MoE) | Action Expert (1 토큰) |
| 시간 인코딩 | H=30 윈도우 × 27 스텝 (≈2s) | 없음 | 10 프레임 (≈2s) |
| 접촉 게이팅 | grad. threshold (δ=0.5 N) | 없음 | 없음 |
| 추가 파라미터 | ≈2.2M | ≈45M | ≈2.1M |

ForceVLA는 파라미터를 20배 더 쓰면서도 시간 문맥이 전혀 없고, TA-VLA는 2초를 단일 토큰으로 압축해 시간 구조를 사실상 파괴한다. FACT는 **더 적은 파라미터로 더 나은 표현 구조**를 택했다는 것이 핵심 대비점이다.

---

## 11. 강점과 한계

**강점**
- 성능 향상보다 **진단의 인과 구조**가 먼저 온다. "정밀도 vs 힘"이라는 분해가 ablation(정밀도 과제에서 힘 주입 무효, 힘 과제에서 LN+힘 모두 유효)으로 실증된다.
- LN은 **파라미터/데이터/아키텍처 변경 0의 무료 개선**이며 타 baseline·타 백본에 모두 전이됨이 확인됨.
- 노이즈 치환 대조군이라는 **엄밀한 반증 실험** 설계 — 힘 이득이 정규화 아티팩트인지 진짜 힘 활용인지 구분.
- 시뮬레이션 없이 약 2,500회 실물 롤아웃 + Fisher's exact test라는 이 분야로서는 이례적인 통계적 엄밀성.

**한계 (저자 명시 포함)**
- **단일 로봇 플랫폼, 고정된 손목 F/T 센서 배치**. 다른 기구학 구조나 센서 위치로의 일반화 미검증.
- **LN은 flow-matching 액션 헤드 전용** — autoregressive나 diffusion 정책에는 직접 적용 불가.
- 시간 인식 힘 주입은 RMS 스케일로 변조되는 transformer 레이어를 전제 — 다른 아키텍처는 개조 필요.
- 5개 과제뿐이며 board erasing은 사실상 포화되어 유효 비교 과제는 4개. 더 다양한 기하·재질에서의 검증 필요.
- 정밀도 과제(plug p=.012, USB p=.249)의 효과 크기가 힘 과제 대비 약하고, USB는 어떤 방법으로도 50%를 넘지 못한다 — 서브밀리미터 정렬 문제가 완전히 해결되지 않았음을 시사.
- 코드·STL은 "공개 예정"이며 리뷰 시점 기준 미공개.

---

## 12. 종합 평가

**참신성**: ★★★★☆ — LN 스케줄 자체는 SD3에서 차용했고 AdaRMS 변조도 새롭지 않다. 그러나 "접촉 실패 = 저노이즈 영역 학습 기아"라는 **진단 자체가 이 논문의 진짜 기여**이며, 이는 지금까지 아무도 명시적으로 짚지 않았다.

**엄밀성**: ★★★★★ — 노이즈 치환 대조군, Fisher's exact test, 두 백본 전이, 3중 ablation, m 스윕까지. 실물 로봇 논문 중 상위권의 실험 위생.

**실용성**: ★★★★☆ — LN은 오늘 당장 어떤 flow-matching VLA에도 한 줄로 적용 가능한 사실상 무료 개선. 힘 주입은 F/T 센서가 있어야 하지만 ≈2.2M 파라미터로 끝난다.

**총평**: 이 논문의 가장 중요한 문장은 66.0%라는 헤드라인 숫자가 아니라, **정밀도 과제에서는 힘 주입이 통계적으로 무의미했다**는 결과다. 지난 2년간 이 분야는 "접촉이 안 되면 힘 센서를 붙이자"는 단일 처방을 반복했는데, 이 논문은 그 처방의 절반이 잘못된 병에 처방되고 있었음을 보인다. 나아가 Table A.1은 기존 힘 증강 VLA(특히 TA-VLA)가 힘을 **실제로는 쓰고 있지 않았다**는 불편한 증거를 제시한다. 벤치마크 점수 경쟁이 아니라 **음성 대조군(negative control)을 설계하는 문화**가 VLA 연구에 필요하다는 점을 보여준 사례로 읽어야 할 논문이다. 다만 단일 플랫폼·5개 과제라는 좁은 증거 기반과 flow-matching 한정이라는 적용 범위는 후속 검증을 요구한다.

<!-- VERIFIED: pdf -->
