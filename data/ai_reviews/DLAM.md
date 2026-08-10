# DLAM: Distributional Latent Actions with Temporal Constraints

> **한 줄 요약**: Latent action을 결정론적 "점"이 아니라 **대각 가우시안 분포**(mean + dimension-wise variance)로 표현하고, 등간격 triplet에 대해 **normalized composition**과 **reversal** 제약을 평균과 분산 모두에 부과. 인접 transition이 중간 프레임을 공유하는 의존성은 **학습된 공유 상관계수 ρ**로 처리. 다운스트림에서는 encoder를 freeze하고 **posterior mean만** π0의 joint flow matching 보조 타깃으로 전달 → MetaWorld MT50 **87.6%**, LIBERO **99.0%**, 실세계 4개 태스크 **73.8%**.

- arXiv: 2607.27138v1 (2026-07-29)
- 소속: Zhejiang University / Amap, Alibaba Group / Nanjing University / SJTU / SUAT / XJTU / Chery Auto

---

## 1. 배경 및 동기

### Action label 희소성
- VLA는 (vision, language, **action**) 트리플이 필요하지만 action label 수집 비용이 지배적.
- Action-free video는 물리적 변화의 관측을 풍부하게 제공 → transition prior의 대체 공급원.

### 기존 LAM의 한계
1. **Reconstruction-only LAM** (Genie, LAPA, UniVLA 계열): 미래 프레임 예측에는 도움되지만 카메라 모션·배경 dynamics·appearance 변화를 인코딩할 수 있어 **control과의 관련성이 약함**. 예측적이지만 robot action과의 joint generation에는 부적합.
2. **Structured LAM** (AC-LAM, ALAM, RotVLA): composition/reversal/inverse/cycle 제약을 추가하지만 각 transition을 여전히 **결정론적 점**으로 표현. 각 관계식이 단일 추정치 하나에만 작용하므로, 국소적으로 추론된 transition의 잔차 오차가 **재귀적 장기 합성에서 누적·증폭**될 수 있음.

### 핵심 질문
> Transition을 분포로 표현하면, 시간적 제약이 평균뿐 아니라 **분산까지 감독**하여 더 안정적인 latent dynamics를 얻을 수 있는가?

📌 [Figure 1] LAM(구조 없음) → ALAM(결정론적 점 합성/역전) → DLAM(분포 공간 제약)

---

## 2. 방법론 심층 분석

### 2.1 Distributional latent-action pretraining

등간격 triplet $(O_a, O_b, O_c)$, $b-a = c-b = k$ 를 샘플링. Forward set $T_{fwd} = \{(a,b), (b,c), (a,c)\}$, backward pair $(b,a)$ 는 reversal 감독용. **action / language / proprioception 라벨 불필요**.

Relational encoder $E_\phi$ 가 두 프레임을 K개의 learnable query로 처리해 $[\mu^j_i, \tilde{\ell}^j_i]$ (둘 다 $K \times d$) 출력:

$$q^j_{i,\kappa} = \mathcal{N}\big(\mu^j_{i,\kappa},\ \mathrm{diag}((\sigma^j_{i,\kappa})^2)\big), \quad \ell = \mathrm{clip}(\tilde\ell, \ell_{min}, \ell_{max}),\ \sigma = \exp(\tfrac12 \ell)$$

- **Reconstruction**: source-conditioned decoder $D_\omega(O_i, \mu^j_i)$ 로 target 프레임 복원. **posterior sample을 쓰지 않고 mean만** 사용 → mean이 관측된 시각적 변화에 grounding.
- **Prior**: factorized standard Gaussian $\mathcal{N}(0, I_d)$ 로 KL 정규화, free-nats floor 적용.

### 2.2 Temporal constraints — mean과 variance에 동시 부과

**Normalized composition** (등간격 $k+k \to 2k$ 관계에 1회 적용):

$$\bar\mu^c_a = \frac{\mu^b_a + \mu^c_b}{\sqrt2}, \qquad (\bar\sigma^c_a)^2 = \frac{(\sigma^b_a)^2 + (\sigma^c_b)^2}{2} + \rho\,\sigma^b_a \odot \sigma^c_b$$

- $\rho = \rho_{max}\tanh(r)$, $r$ 은 **학습되는 스칼라**, $0 < \rho_{max} < 1$ 로 $|\rho| < 1$ 보장. 예제·토큰 슬롯·latent 차원에 걸쳐 **공유**.
- $\rho = 0$ 이면 독립 분산 전파로 환원되고, $1/\sqrt2$ 계수가 단위 분산을 보존.
- 저자들은 이를 **associative composition law가 아니라 pairwise normalized relation**이라고 명시적으로 한정.

**Reversal**: 평균은 부호 반전, 분산은 보존.

$$\mathcal{R}[\mathcal{N}(\mu, \mathrm{diag}(\sigma^2))] = \mathcal{N}(-\mu, \mathrm{diag}(\sigma^2))$$

두 번 적용하면 원래 posterior로 복귀. 정확한 역원이 아니라 **forward-backward 일관성 관계**로 사용.

**Discrepancy**: $D(q, q') = (\|\mu - \mu'\|_F^2 + \lambda_\ell \|\ell - \ell'\|_F^2)/(Kd)$

$$\mathcal{L}_{DLAM} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{prior}\mathcal{L}_{prior} + \lambda_{comp}\mathcal{L}_{comp} + \lambda_{rev}\mathcal{L}_{rev}$$

### 2.3 Transfer to world action modeling

- Reconstruction decoder는 **폐기**, transition encoder는 **freeze**.
- 각 뷰 $m$(third-person / wrist)에 대해 $H$개 연속 프레임 쌍에서 $(\mu^m_h, \ell^m_h)$ 추출. **$\mu$만 다운스트림 타깃**, log-variance는 policy로 전달되지 않음.
- Joint flow matching: $\mathcal{L}_{transfer} = \lambda_u \mathcal{L}^u_{FM} + \sum_m \lambda_m \mathcal{L}^m_{FM}$
- Inference 시 두 스트림을 모두 생성하지만 **robot action 스트림만 실행**. 미래 프레임은 학습 타깃 구성에만 사용.

> ❓ 왜 variance를 downstream에 안 넘기는가?
> 저자 스스로 "분산은 calibrated uncertainty가 아니라 **shared encoder에 대한 보조 학습 신호**"라고 못박음. 두 endpoint가 모두 관측되어 있으므로 미래에 대한 불확실성 해석이 성립하지 않음.

---

## 3. 데이터 전략

- **11개 action-free robot-video 데이터셋**, 총 **6.27M 샘플**. 주로 Open X-Embodiment + CALVIN.
- 가용 샘플 비중 vs 정규화 샘플링 확률 (outer/inner ring, Figure 4):

| 소스 | 샘플 수 (비중) | sampling weight |
|------|---------------|-----------------|
| Fractal | 3,182,825 (50.7%) | 150 |
| CALVIN | 1,795,025 (28.6%) | 200 |
| BridgeData V2 | 579,207 (9.2%) | 50 |
| TOTO | 236,632 (3.8%) | 5 |
| RoboTurk | 134,496 (2.1%) | 10 |
| TACO-Play | 126,298 (2.0%) | 5 |
| AutoLab UR5 | 74,883 (1.2%) | 5 |
| VIOLA | 60,987 (1.0%) | 3 |
| JaCo-Play | 52,715 (0.8%) | 20 |
| Cable Routing | 15,223 (0.2%) | 20 |
| NYU Door | 15,221 (0.2%) | 5 |

- CALVIN이 28.6% 점유에 가중치 200으로 **정규화 확률 42.3%** — 사실상 최대 기여.

---

## 4. 시스템/학습 세부사항

| 항목 | Pretraining | Policy transfer |
|------|-------------|-----------------|
| 하드웨어 | 64 × AMD MI308X | (미명시, per-device batch 32) |
| Epoch | 57 | - |
| Optimizer | AdamW | AdamW |
| LR | 1e-4 (peak) | 5e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Per-device batch | 64 | 32 |
| Loss weights | $\lambda_{rec}=1$, $\lambda_{prior}=0.005$, $\lambda_{comp}=\lambda_{rev}=0.05$, $\lambda_\ell=0.1$ | $\lambda_u, \lambda_m$ |

- Backbone: **π0** = PaliGemma-2B (VLM) + Gemma-300M (action expert). 업데이트 대상은 policy backbone, action expert, projection layer뿐.
- **모든 controlled variant가 동일한 visual tokenizer / Transformer capacity / decoder / data order / training budget** 사용 → 비교 공정성 확보.

---

## 5. 실험 설계 및 평가 프로토콜

세 층위 평가:
1. **표현(representation)**: scale-normalized composition/reversal residual.
2. **재구성(reconstruction)**: direct vs cumulative decoding, PSNR/SSIM/LPIPS.
3. **정책(policy)**: MetaWorld MT50, LIBERO, 실세계 Piper 6-DoF.

Temporal probe (length-aware): $C_h(z_{1:h}) = \frac{1}{\sqrt h}\sum_{i=1}^h z_i$ — $h=2$ 에서 DLAM 학습 규칙과 일치. $h>2$ 는 associativity가 아니라 **per-length consistency** 측정.

Scale 민감도 제거를 위한 대칭 상대오차:
$$R_{sym}(x,y) = \frac{\mathrm{mean}|x-y|}{\frac12(\mathrm{mean}|x| + \mathrm{mean}|y|) + \epsilon}$$

- Held-out window는 11개 데이터셋에서 $k=10$ 프레임 단위로 샘플, 재구성은 5k까지 / 시간 관계는 10k까지. **3k 이상은 직접 감독되지 않은 OOD 영역**으로 명시.

---

## 6. 실험 결과 심층 분석

### 6.1 MetaWorld MT50 (Table 1)

| Method | Type | Size | Easy | Med | Hard | V-Hard | **Avg** |
|--------|------|------|------|-----|------|--------|---------|
| RT-2 | AR | 7B | 75.5 | 35.3 | 30.7 | 15.2 | 39.2 |
| RoboTron Mani | AR | 4B | 85.5 | 67.7 | 76.7 | 81.0 | 77.7 |
| GR-1 | VA | 0.2B | 76.6 | 35.3 | 46.0 | 44.0 | 50.5 |
| PAD | VA | – | 81.8 | 65.1 | 56.7 | 87.2 | 72.7 |
| Evo-1 | VA | 0.8B | 89.2 | 76.8 | 77.2 | 79.2 | 80.6 |
| π0.5 | FM | 3B | 68.2 | 37.3 | 41.7 | 28.0 | 43.8 |
| π0 | FM | 3B | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | FM | 2B | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| π0+ALAM | LA | 3B | 89.3 | 83.6 | **85.0** | 82.0 | 85.0 |
| **π0+DLAM** | LA | 3B | **90.3** | **84.8** | 84.0 | **91.3** | **87.6** |

- π0 대비 **+39.7점**, 결정론적 ALAM 대비 **+2.6점**.
- 최대 이득은 **Very-Hard 82.0 → 91.3 (+9.3)**. Hard tier에서는 ALAM에 0.99점 뒤짐 — 유일한 열세 항목.

### 6.2 LIBERO (Table 2)

| Method | Type | Size | Spatial | Object | Goal | Long | **Avg** |
|--------|------|------|---------|--------|------|------|---------|
| OpenVLA | AR | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | AR | 7B | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| DreamVLA | VA | 0.4B | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| OneWM-VLA | VA | 3B | 98.2 | 99.6 | 99.0 | 95.1 | 98.0 |
| GR00T N1 | FM | 2B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | FM | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | FM | 3B | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LAPA | LA | 7B | 87.4 | 91.2 | 90.0 | 65.4 | 83.5 |
| UniVLA | LA | 9B | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| JALA | LA | 3B | 96.0 | 98.2 | 97.4 | 96.0 | 96.9 |
| π0+ALAM | LA | 3B | 99.2 | 99.6 | 99.0 | 94.4 | 98.1 |
| **π0+DLAM** | LA | 3B | **99.6** | **99.8** | **99.6** | **97.1** | **99.0** |

- 4개 suite 전부 개선. 최대 이득은 **LIBERO-Long +2.7 (94.4 → 97.1)** — 장기 태스크에서 분포적 제약의 효과가 가장 큼. Spatial/Object/Goal은 이미 포화 구간.

### 6.3 재구성 및 시간 관계 (Figure 5)

- 3k–5k span 평균 PSNR: **direct 29.14 dB**, **cumulative 22.40 dB** → ALAM 대비 각각 **+3.45 dB**, **+1.17 dB**.
- LPIPS: direct **-45.6%**, cumulative **-26.1%**. SSIM도 일관 개선.
- 3k–10k의 미감독 구간에서 DLAM residual은 낮게 유지·완만히 증가, ALAM은 뚜렷이 증가. 단 저자는 "평가된 span 상의 거동이지 임의 horizon에 대한 일반 보장은 아니다"라고 한정.

### 6.4 실세계 (Figure 6, Piper 6-DoF)

- insert cylinder / insert cube / arrange flowers / hang cup 4개 태스크 전부 최고 성공률.
- 평균 **73.8%** vs π0+ALAM 63.8%, π0.5 53.8%, π0 40.0%. **각 태스크마다 ALAM 대비 정확히 10점 개선**이라고 기술.

---

## 7. Ablation 분석 (Table 3)

| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | $R^\mu_{comp}$↓ | $R^\mu_{rev}$↓ | MetaWorld Avg↑ |
|---------|-------|-------|--------|------|------|------|
| No temporal relations | 21.236 | 0.7894 | 0.1755 | 1.2071 | 1.0314 | 76.6 |
| Matched mean-only (σ=1, ρ=0) | 22.109 | 0.8057 | 0.1644 | 1.1808 | 1.0371 | 82.1 |
| Learned variance (ρ=0) | 22.084 | 0.8067 | 0.1637 | 1.1777 | 1.0341 | 85.3 |
| **Full DLAM** | **22.400** | **0.8086** | **0.1623** | **1.1662** | **1.0010** | **87.6** |

해석:
1. **Normalized mean constraint가 재구성 이득의 대부분을 회수** (21.236 → 22.109 dB, +0.873 중 전체 1.164의 75%). 성공률도 76.6 → 82.1.
2. **Learned variance는 재구성을 거의 안 올리지만**(22.109 → 22.084, 오히려 미세 하락) **downstream 성공률은 +3.2점(82.1 → 85.3)**. → 분산이 재구성이 아니라 **정책 전이에 유리한 표현**을 만든다는 증거.
3. **상관 인지 합성(ρ 학습)이 모든 지표 최고**. 특히 $R^\mu_{rev}$ 가 1.0341 → 1.0010 으로 눈에 띄게 개선.
4. 시간 제약 전무 대비 총 **+11.0점**.

---

## 8. 관련 연구 비교

| 계열 | 대표 | Transition 표현 | 제약 |
|------|------|----------------|------|
| Reconstruction-only LAM | Genie, LAPA, UniVLA, AdaWorld, CLAM | 점 (discrete/continuous) | 없음 |
| Structured LAM | AC-LAM, ALAM, RotVLA | 결정론적 점 | composition / reversal / inverse / cycle |
| Stochastic video prediction | SV2P, SVG | 잠재 변수 | 다중 미래 모델링 목적 |
| **DLAM** | - | **대각 가우시안** | **평균 + 차원별 분산 모두에 composition/reversal** |

- Stochastic video prediction과의 결정적 차이: DLAM의 분산은 **미래 다양성 모델링이 아니라** 관측된 두 프레임 사이 transition에 대한 **추가 학습 신호**.
- ALAM과의 직계 관계: 동일 저자 그룹(Zuojin Tang 등), 동일 π0 transfer 프로토콜, 동일 사전학습 mixture. 사실상 **ALAM의 분포화 확장**이며 controlled comparison이 가능하도록 설계됨.

---

## 9. 한계 및 미해결 문제

저자 명시 한계:
1. **국소 제약만 존재** — 등간격 triplet에 대한 pairwise normalized relation일 뿐, 장기 horizon 일반화는 미해결.
2. **Variance objective가 near-constant 해로 붕괴할 수 있음** — 분산이 상수로 수렴하면 사실상 mean-only로 퇴화.
3. **분산은 calibrated uncertainty가 아님** — downstream이 mean만 쓰므로 보조 신호에 그침.
4. **공유 ρ의 표현력 한계** — 문맥·차원 의존적 의존성을 놓칠 수 있음 (모든 예제/슬롯/차원에 스칼라 하나).

리뷰어 관점 추가 한계:
5. LIBERO는 99.0%로 사실상 포화 — 이 벤치마크로는 더 이상 변별이 어려움.
6. 실세계 평가가 4개 태스크·단일 로봇(Piper)에 한정.
7. π0 이외 백본(OpenVLA, GR00T 등)으로의 전이 검증 부재.
8. 64 × MI308X × 57 epoch 사전학습은 재현 장벽이 높고, 코드/가중치 미공개.

---

## 10. 총평

- **기여의 성격**: 새로운 백본이나 실행 인터페이스 없이, latent action 표현의 "타입"을 점 → 분포로 바꾼 **표현 수준의 최소 침습적 개선**. 실용성이 높다.
- **가장 설득력 있는 근거**: Ablation에서 learned variance가 **재구성은 안 올리면서 성공률만 +3.2점** 올린 부분. 분산이 단순 정규화 효과가 아니라 policy-relevant structure를 만든다는 분리된 증거.
- **가장 약한 고리**: ρ가 전역 스칼라 하나라는 점, 그리고 분산이 결국 버려진다는 점. "분포로 표현했지만 결국 mean만 쓴다"는 구조는 우아함 대비 최종 이득 경로가 간접적.
- **재현성**: 손실 가중치·LR·batch·epoch·데이터 mixture 가중치까지 공개되어 수치 재현 정보는 충실하나 코드/체크포인트는 미공개.
- 평점: 방법론적 신규성 중상, 실험 통제 우수, 벤치마크 포화로 인한 헤드룸 제약.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 답변 |
|---|------|------|
| 1 | Downstream에서 분산을 안 쓸 거면 왜 분포로 모델링하나? | 분산은 shared encoder에 추가 gradient 신호를 주는 auxiliary 역할. Ablation에서 재구성은 동일한데 성공률만 82.1 → 85.3으로 올라 이 경로가 실증됨. |
| 2 | $1/\sqrt2$ 정규화의 근거는? | ρ=0이고 두 입력이 독립 표준 가우시안일 때 단위 분산을 보존. 스케일 폭주를 막는 정규화 장치. |
| 3 | Composition이 associative하지 않다는 게 문제 아닌가? | 저자도 pairwise normalized relation이라고 한정. $k+k \to 2k$ 에 1회만 적용하며, $h>2$ probe는 associativity가 아니라 per-length consistency 측정. |
| 4 | ρ를 모든 차원·슬롯에 공유하는 게 과도한 단순화 아닌가? | 저자가 한계로 인정 (context/dimension-dependent dependency 미포착). 다만 파라미터 1개로 $R^\mu_{rev}$ 를 1.0341 → 1.0010 개선. |
| 5 | Variance가 상수로 붕괴하면? | 저자 명시 한계. 실제로 붕괴하지 않았다는 직접 증거(분산 통계)는 논문에 제시되지 않음 — 검증 필요 지점. |
| 6 | Reconstruction에 sample 대신 mean만 쓰는 이유는? | 두 endpoint가 모두 관측된 상태이므로 미래 불확실성 샘플링이 의미 없고, mean을 관측된 변화에 직접 grounding하는 게 목적. |
| 7 | 3k 이상 span 결과는 신뢰할 수 있나? | 직접 감독되지 않은 OOD 영역으로 논문에서 음영 표시. 32 클립 paired bootstrap 95% CI 제시하나 일반 보장은 주장하지 않음. |
| 8 | MetaWorld Hard tier에서 ALAM에 진 이유는? | 논문 미설명. 84.0 vs 85.0으로 1점 이내이며 Very-Hard의 +9.3 이득이 이를 압도. |
| 9 | LIBERO 99.0%는 포화 아닌가? | 맞음. 유의미한 차이는 Long suite(+2.7)뿐이며 나머지는 오차 범위. MetaWorld와 실세계가 실질 변별력을 제공. |
| 10 | 사전학습 mixture에서 CALVIN 비중이 과한 것 아닌가? | 샘플 28.6%에 가중치 200으로 정규화 확률 42.3%. CALVIN이 LIBERO/MetaWorld와 도메인이 다르므로 편향 우려는 있으나 모든 baseline이 동일 mixture라 비교는 공정. |

---

## 12. 세미나 토론 포인트

1. **분포 latent의 "올바른" 용법**: mean만 전이하는 현 설계 대신, policy가 분산을 조건으로 받아 exploration이나 chunk length를 조절하면 어떤가? 분산이 calibrated가 아니라는 저자의 주의를 어떻게 우회할 것인가.
2. **ρ의 일반화**: 스칼라 → 차원별 벡터 / 문맥 조건부 네트워크로 확장 시 이득 대비 과적합 위험은?
3. **장기 horizon**: $k+k \to 2k$ 를 재귀적으로 여러 번 적용하는 multi-scale 학습(예: $2k+2k \to 4k$)을 추가하면 OOD span 성능이 개선될까, 아니면 오차 누적만 심화될까?
4. **Variance collapse 진단**: 학습 중 $\sigma$ 의 엔트로피/분산 통계를 모니터링하는 프로토콜이 필요. 현 논문은 이를 제시하지 않음.
5. **백본 독립성 검증**: OpenVLA(autoregressive)나 diffusion policy 계열에 동일 encoder를 붙이면 이득이 유지되는가? flow matching 특유의 joint generation 구조에 의존하는 이득일 가능성.
6. **ALAM 대비 비용**: 분포화로 인한 추가 학습 비용(파라미터·시간)이 +2.6점(MetaWorld) / +0.9점(LIBERO)에 값하는가? 실세계 +10점은 값한다고 볼 수 있는가.
7. **Real-world 일반화**: 4개 태스크·1개 로봇을 넘어 다중 embodiment에서도 10점 격차가 유지될지.

<!-- VERIFIED: pdf -->
