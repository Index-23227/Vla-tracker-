# StableVLA: Towards Robust Vision-Language-Action Models without Extra Data

> **한 줄 요약**: 학습 데이터에 없는 시각적 disturbance에 취약한 기존 VLA 문제를, 정보이론(Information Bottleneck) 기반의 lightweight Fused IB-Adapter로 해결. Qwen2.5-0.5B 백본 위에 <10M 파라미터만 얹어 ImageNet-C 19종 corruption 하에서 7B-급 SOTA에 견주는 robustness를 달성한 ICML 2026 논문. LIBERO clean 평균 96.65%, CALVIN 4.17 avg-len, 실로봇(Astribot S1)에서 corruption 대비 성능 저하가 OpenPi-0.5(-30 to -42pp) 대비 절반 이하(-14 to -18pp).

---

## 1. 배경 및 동기

### 기존 연구의 한계
- 모든 가능한 시각적 disturbance(조명, occlusion, sensor noise, weather artifact)를 학습 분포에 담는 것은 비현실적
- 최신 SOTA VLA(OpenVLA, OpenPi 등)는 학습 분포 밖의 corruption에 극심한 성능 저하 — 저자들의 systematic study가 이 robustness gap을 정량적으로 폭로
- 기존 augmentation은 데이터 수집·다양성 부담이 큼 → "추가 데이터 없이" robustness를 끌어올리는 구조적 해법이 필요

### 핵심 질문
- **시각 표현에서 task-relevant 정보는 유지하면서 분포 밖의 noise만 선택적으로 제거할 수 있는가?**
- **0.5B 같은 작은 백본으로도 7B 급 VLA의 robustness curve에 도달할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

- **Vision encoder**: DINO-SigLIP (224px), fine-tuning 동안 frozen
- **LLM backbone**: Qwen2.5-0.5B
- **Pre-training**: LLaVA-LVIS4V-LRV로 vision-language alignment (Prismatic VLM 프로토콜)
- **Action head**: Fused IB-Adapter — 표준 MLP pathway와 IB pathway의 dual-pathway 결합

### 2.2 Fused IB-Adapter 설계

세 가지 메커니즘이 핵심:

| 컴포넌트 | 역할 | 수식적 정의 |
|---------|------|------------|
| Subspace covariance (Gram matrix) | 채널 간 종속성 포착 | G_h = Q_h^T K_h, head별 d×d |
| Sigmoid gating | 채널별 독립 노이즈 억제 | 학습 가능한 σ(·) — 채널 경쟁(softmax) 회피 |
| Two-layer MLP (GELU) | Value token 비선형 변환 | spectral gate로 modulation |

최종 출력:

```
Z = MLP(X) + tanh(λ) · IB-Adapter(X)
```

- λ는 robustness signal 주입 강도 제어
- Stochastic Pathway Dropout p_drop으로 task별 MLP/IB 비중 조정 (예: LIBERO-Long은 p_drop≈0, LIBERO-Object/CALVIN은 p_drop≈0.3)
- 추가 파라미터 **< 10M** (백본 0.5B 대비 ~2%)

> ❓ **예상 질문**: 왜 softmax 대신 sigmoid 게이팅인가?
> **답변**: Ablation에서 Fused IB-Adapter(softmax)는 LIBERO 62.8% / CALVIN 0.46으로 sigmoid 변형(79.1% / 2.13) 대비 극심한 열화. Softmax는 채널 간 경쟁을 강제해 독립적 노이즈 채널 억제와 충돌. Sigmoid는 channel-wise 독립 Bernoulli latent 가정과 부합.

### 2.3 학습 프로토콜

- Pre-training: LR 2e-5, batch 64, λ=0.3, p_drop=0.0
- Fine-tuning: task-dependent p_drop (Long은 0에 가깝게, Object/CALVIN은 0.3 부근)
- Corruption augmentation **없음** — zero-shot robustness 평가

---

## 3. 데이터 전략

- **외부 corruption 데이터 미사용**: ImageNet-C 등은 평가에만 사용, 학습에는 노출되지 않음
- Pre-training: LLaVA-LVIS4V-LRV (vision-language alignment)
- Fine-tuning: LIBERO, CALVIN 표준 데이터셋
- 실로봇: Astribot S1로 4개 태스크(Pick and Place, Throw Basketball, Pour Water, Pack Doll) 수집

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO Clean (Table 1, % 성공률)

| Suite | StableVLA | VLA-Adapter | Δ |
|-------|-----------|-------------|---|
| Spatial | 96.2 | – | – |
| Object | 98.8 | – | – |
| Goal | 98.0 | – | – |
| Long | 93.6 | – | – |
| **Avg** | **96.65** | – | – |

### 4.2 LIBERO 고난도 corruption (severity 3–5 평균)

| Suite | StableVLA (S3-S5 avg) | vs VLA-Adapter |
|-------|----------------------|----------------|
| Spatial | 90.1 | +6.8pp |
| Object | 82.1 | +22.0pp |
| Goal | 83.1 | +22.0pp |
| Long | 61.3 | +22.0pp |

- 14× 작은 백본임에도 7B-scale SOTA의 robustness profile에 근접
- 평균 합성 corruption에서 **+35.2%** improvement (논문 주장)

### 4.3 CALVIN (ABC→D, average completed tasks of 5)

| Method | Clean | Avg Corrupted |
|--------|-------|---------------|
| **StableVLA** | **4.17** | **2.13** |
| VLA-Adapter | 4.14 | 1.44 |

- Clean에서는 baseline과 유사하나 corruption에서 +0.69 차이 → "stability under shift"가 본질적 기여

### 4.4 실로봇 — Astribot S1 (corruption별 평균 성공률 하락)

| Method | Δ (corruption-average drop, pp) |
|--------|--------------------------------|
| **StableVLA** | **-14.2 ~ -18.4** |
| VLA-Adapter | -25.0 ~ -49.2 |
| OpenPi-0.5 | -30.0 ~ -41.7 |

- "Pick and Place" 예시: clean 80% → corruption 평균 62.5% (Δ -17.5pp), OpenPI-0.5는 100% → 69.9% (Δ -30.1pp)
- 핵심 주장: 31.7pp 향상(특정 corruption type 기준)

---

## 5. Ablation 분석

### 5.1 어댑터 아키텍처 (Table 3)

| Configuration | LIBERO Avg | CALVIN Avg |
|--------------|-----------|-----------|
| IB-Adapter only | 76.0% | 1.44 |
| **Fused IB-Adapter (sigmoid)** | **79.1%** | **2.13** |
| Fused IB-Adapter (softmax) | 62.8% | 0.46 |

- Dual-pathway(MLP+IB) 설계와 sigmoid gating 두 요소가 모두 결정적
- Softmax 변형의 붕괴는 **채널 경쟁이 noise filtering과 양립 불가**임을 강하게 시사

### 5.2 p_drop의 task 민감도

- LIBERO-Long: p_drop ≈ 0 (MLP pathway 유지 → 정밀 spatial control 보존)
- LIBERO-Object / CALVIN: p_drop ≈ 0.3 (IB pathway 강조 → robustness 우선)
- Task별 적정 p_drop을 manual로 조정 — automated scheduling은 미구현

---

## 6. 관련 연구 비교

| 모델 | 백본 크기 | Robustness 전략 | Extra Data |
|------|----------|----------------|-----------|
| OpenVLA | 7B | – | OXE |
| OpenPi-0.5 | 4B+ | – | OXE |
| VLA-Adapter | 7B | Adapter (no IB) | – |
| **StableVLA** | **0.5B** | **Fused IB-Adapter** | **None** |

- 14× 작은 backbone + extra data 없음 → SOTA robustness curve와 견줄 만한 성능

---

## 7. 한계 및 미해결 문제

1. **p_drop의 manual 조정**: task별로 적절한 dropout 비율이 다르며 automated 적응 메커니즘 부재 — 새 도메인에서는 hyperparameter sweep 필요
2. **Clean 성능의 한계**: LIBERO clean 96.65%, CALVIN 4.17은 SOTA(예: π₀의 4.5+ 영역)에는 미달. 목표는 robustness 보존이지 clean ceiling이 아님
3. **백본 ceiling**: 0.5B의 한계로 인해, 7B 백본에 IB-Adapter를 부착했을 때의 추가 이득은 별도 검증 필요
4. **Corruption distribution**: ImageNet-C 외의 더 현실적인 disturbance(조명 변화의 미세 변화, motion blur 시간적 패턴 등)에 대한 평가는 제한적
5. **Theoretical bound 부재**: "Information Bottleneck"이라는 명명에도 불구하고 explicit MI lower/upper bound 도출이나 β-VAE식 스케줄링은 abstract method가 아님 — 본질적으로는 architectural inductive bias

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — VLA에 IB regularization을 architectural form으로 가져온 점이 독창적 |
| Technical depth | ★★★★☆ — Subspace covariance + dual-pathway + sigmoid의 조합이 체계적 |
| Experimental rigor | ★★★★★ — 시뮬레이션 2종 + 실로봇 + 4 종 corruption family를 모두 커버 |
| Practical impact | ★★★★★ — 0.5B 백본 + <10M 어댑터 → on-device 배포에 매우 매력적 |
| Writing quality | ★★★★☆ — robustness 측정 프로토콜이 명확 |

**강점**: "추가 데이터 없이 robustness"라는 실용적 화두를 정면 돌파. 14× 작은 백본의 한계를 architectural inductive bias로 보상한 점이 인상적. **약점**: Clean 성능 ceiling(LIBERO 96.65, CALVIN 4.17)은 최상위 7B 모델 대비 여전히 격차. 또한 task-specific p_drop tuning은 deployment 부담.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 왜 IB라고 부르는가? Mutual information bound가 명시적으로 등장하는가? | 명시적 MI 추정은 없음. Subspace covariance + sigmoid gating이 IB의 "compress noise, preserve task-relevant" 정신을 architectural로 구현한 inductive bias |
| 2 | 7B 모델에 IB-Adapter를 붙이면? | 본 논문 미실험. 14× 작은 백본을 7B-급에 견주는 게 핵심 메시지이므로 의도적 selection 가능성 |
| 3 | Softmax가 왜 그렇게 망가지는가? (62.8% vs 79.1%) | Softmax는 채널 간 경쟁 → 독립적 noise channel 억제와 양립 불가. Sigmoid는 channel-independent Bernoulli latent와 부합 |
| 4 | p_drop을 task별로 손으로 맞춰야 하나? | 그렇다. Long은 ~0, Object/CALVIN은 ~0.3. Automated scheduling은 future work |
| 5 | Corruption training 없이 정말 zero-shot 되는가? | ImageNet-C 19 type에서 그렇다고 주장. 단 실로봇 augmentation은 mild geometric/photometric만 사용 |
| 6 | OpenPi-0.5와 14× 비교가 공정한가? | 백본 크기는 14×지만 OpenPi-0.5는 OXE 사전학습 + 더 큰 데이터. 결과는 적은 자원으로 견주는 의미가 있지만 "동일 조건 비교"는 아님 |
| 7 | 실로봇 4 task, corruption 4종은 통계적으로 충분한가? | per-task drop 14-18pp는 일관성 있으나, trial 수가 제한적이라 분산 추정이 필요 |
| 8 | CALVIN clean 4.17은 SOTA가 아니다 — robustness 향상이 ceiling 희생을 정당화하는가? | 4.14(VLA-Adapter)→4.17은 clean에서도 미세 향상. Corruption에서 1.44→2.13 차이가 본 논문의 진짜 기여 |

<!-- VERIFIED: pdf -->
