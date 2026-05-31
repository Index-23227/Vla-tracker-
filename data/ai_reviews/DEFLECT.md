# DEFLECT: Delay-Robust Execution via Flow-matching Likelihood-Estimated Counterfactual Tuning for VLA Policies

> **한 줄 요약**: Async VLA 추론의 prediction-execution misalignment를 해결하기 위해, **fresh/stale action pair**를 frozen reference policy로 생성하고 **implicit flow-matching likelihood**를 surrogate로 한 DPO objective(+ SFT anchor)로 fully offline post-training — π₀.₅ base를 LIBERO와 Kinetix에서 fine-tune하여 **high-delay(5-7 step) 영역에서 +6.4% SR**을 추가로 끌어올린 HKUST(GZ)·SYSU의 delay-robust refinement 기법(arXiv 2605.19294, 2026.05).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 정책은 보통 **action chunk**(여러 step의 action sequence)를 한 번에 예측 → robot이 chunk를 실행하는 동안 다음 chunk를 비동기로 계산(**async inference**)
- 문제: chunk는 추론 시작 시의 observation $o_t$를 conditioning으로 생성되지만, 실제 실행은 robot이 이미 여러 step 이동한 state에서 일어남 → **prediction-execution misalignment**
- 추론 latency가 5-7 control step 정도되면 naive async는 거의 실패: "naive async collapses from 89% to under 1% as inference delay grows to 7 control steps"
- VLASH 같은 기존 방법은 robust하나 여전히 high-delay에서 성능 저하

### 핵심 질문
- **Delay 자체를 label로 활용하여 fully offline로 정책을 개선할 수 있는가?**
- **Flow-matching 정책에서 explicit likelihood 없이 preference optimization이 가능한가?**

---

## 2. 방법론 심층 분석

### 2.1 핵심 아이디어: Counterfactual Fresh/Stale Pair

기존 데이터(state-action trajectory)에서:
- **Fresh action**: 현재 state에 대해 추론한 action (chosen, $a_w$)
- **Stale action**: 과거 state에서 추론한 action을 현재까지 가져온 것 (rejected, $a_l$)
- 둘은 **frozen reference policy**로 생성 — 추가 demonstration 불필요

이 pair는 "현재 state에 대해 fresh가 stale보다 낫다"는 preference signal — DPO objective의 chosen/rejected가 됨.

> ❓ **예상 질문**: stale action이 실제로 항상 나쁜가? 약간의 delay는 무해할 수도?
> **답변**: 작은 delay에서는 거의 동일 — DPO margin이 작아 update가 미미. Large delay에서만 강한 신호 → 자연스러운 curriculum.

### 2.2 Implicit Flow-Matching Likelihood Surrogate

DPO는 두 action의 log-likelihood 비교가 필요. 그러나 flow matching은 **explicit density 없음**.

해결: per-example flow-matching loss(예측 velocity와 target velocity 간의 MSE)를 **negative log-likelihood proxy**로 사용.

$$\text{score}(a) \approx -\mathcal{L}_{\text{FM}}(a) = -\mathbb{E}_{\tau, \epsilon}\bigl[\|v_\theta(a^\tau, o) - u(a^\tau \mid a)\|^2\bigr]$$

이 surrogate를 DPO formula에 plug-in.

### 2.3 최종 손실

$$\mathcal{L}(\theta) = \lambda_{\text{SFT}} \cdot \mathcal{L}_{\text{FM}}(\theta; \text{expert chunk}) + \lambda_{\text{DPO}} \cdot \mathcal{L}_{\text{DPO}}(\theta)$$

- $\lambda_{\text{SFT}} = 1.0$ (anchor — 핵심)
- $\beta = 1.0$ (DPO temperature)
- $\lambda_{\text{DPO}}$: Kinetix 0.02 / LIBERO 0.1

> ❓ **예상 질문**: SFT anchor 없이 DPO만으로는?
> **답변**: Appendix H ablation — "removing the SFT anchor causes catastrophic collapse". DPO의 unconstrained nature가 정책을 drift시킴. SFT가 trust region 역할.

---

## 3. 데이터 및 학습

### 3.1 Setup

| Benchmark | Base policy | Trainable params | Steps |
|-----------|------------|-----------------|-------|
| Kinetix | VLASH-Kinetix 33M | 100% (full fine-tune) | — |
| LIBERO | π₀.₅ (~3.6B) | 19% (action expert only) | 200 steps |

- **VLM frozen on LIBERO** — 빠른 post-training 가능(200 step만으로)
- Training delays $d \in \{0,1,2,3,4\}$ → test에서 $d \in \{0,...,7\}$로 일반화

### 3.2 Optimizer

- AdamW + cosine schedule
- 1000-step warmup
- 추론은 base policy와 동일 (no architectural change)

---

## 4. 실험 결과 심층 분석

### 4.1 Kinetix (Table 1)

| Metric | π₀.₅ baseline | VLASH | **DEFLECT** | Δ vs VLASH |
|--------|--------------|-------|------------|-----------|
| Avg SR ($d=0$-7) | ~50? | 79.4 | **83.3** | +3.9 |
| Avg SR ($d=5$-7) | ~1? | 67.1 | **73.5** | **+6.4** |

- Naive async는 $d=7$에서 < 1% SR
- VLASH는 robust하나 high-delay에서 정체
- DEFLECT는 high-delay regime에서 가장 큰 gain

### 4.2 LIBERO ($\pi_{0.5}$ scale)

- 4 suite × 500 episode 평가
- $d=1$에서는 +0.2(거의 차이 없음) — 작은 delay는 baseline도 잘 처리
- $d=7$에서는 **+4.6 avg** — high-delay에서만 확연한 향상
- Suite별 breakdown은 Appendix F (Spatial / Object / Goal / Long 모두 포함)

> ❓ **예상 질문**: 200 step training만으로 충분한가?
> **답변**: 핵심 정당화: "gains are cleanly attributable to the DPO objective itself" (단순 추가 SFT가 아닌 DPO가 효과). 작은 step 수가 오히려 attribution 명확화.

### 4.3 Real-World (Bimanual Arm)

| Task | VLASH | **DEFLECT** |
|------|-------|------------|
| Conveyor-I | 86.7% | **96.7%** |
| Conveyor-II | 83.3% | **90.0%** |
| Whack-a-Mole (moles/30s) | 10.4 | **13.6** |

- Conveyor task: moving target tracking — delay에 가장 민감
- Whack-a-Mole: 빠른 reaction이 필요한 dynamic task — DEFLECT가 30s 동안 31% 더 많은 mole 격파

### 4.4 Ablation

- **SFT anchor 제거**: catastrophic collapse (Appendix H)
- **$\lambda_{\text{DPO}}$ sweep** {0, 0.01, 0.05, 0.1, 0.2, 0.5}: Figure 11에서 sensitivity 분석. 너무 크면 unstable
- **Counterfactual pair 구성**: naive(stale image + stale state) vs VLASH-style(mixed context) — 두 variant 모두 standard에 미달

---

## 5. 강점 및 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Implicit flow-matching likelihood + counterfactual pair는 신선 |
| **Technical depth** | ★★★★☆ — DPO를 flow matching에 잘 adapt |
| **Experimental rigor** | ★★★★☆ — Kinetix/LIBERO/real-world 모두 평가 |
| **Practical impact** | ★★★★★ — Inference 변경 없이 deployment 즉시 적용 가능 |
| **Open access** | ★★★☆☆ — Anonymous repo 공개 |

**강점**: Inference 변경 없이 post-training만으로 delay robustness를 향상시키는 drop-in 솔루션. Flow matching에 implicit likelihood로 DPO 적용한 첫 사례 중 하나. **약점**: π₀.₅ 등 특정 base policy에 dependency, base policy의 quality가 ceiling 결정.

---

## 6. 한계 및 미해결 문제

### 6.1 방법론
1. **Implicit likelihood surrogate의 theoretical validity**: per-example FM loss를 −log p(a) proxy로 쓰는 것은 heuristic. Calibration이 정확하지 않을 수 있음
2. **Fresh action의 quality에 의존**: counterfactual pair가 reference policy로 생성 → reference가 약하면 fresh도 약함 → DPO signal noisy
3. **Hyperparameter sensitivity**: $\lambda_{\text{DPO}}$가 task별로 다름(Kinetix 0.02 vs LIBERO 0.1) — 새로운 task에서 tuning 필요

### 6.2 평가
1. **Baseline coverage 제한**: 주로 VLASH와 naive π₀.₅과 비교. 다른 delay-robust 방법(asynchronous chunk re-planning 등)과의 직접 비교 부재
2. **LIBERO per-suite 결과는 appendix에 있음** — main text에서는 avg만 강조하여 suite별 trade-off 가시성 부족
3. **Real-world는 3 task만** — 더 다양한 robotic task에서 generality 검증 부족

### 6.3 일반화
1. **Flow matching 정책에 specific** — diffusion policy / autoregressive VLA에는 직접 적용 불가
2. **Online RL과의 비교 부재**: DEFLECT는 offline이나, online RL(예: PPO)이 같은 delay-robustness를 더 잘 달성할 수도

---

## 7. 관련 연구 비교

| 방법 | Online RL | Inference 변경 | Flow matching 호환 | LIBERO d=7 SR |
|------|----------|---------------|-------------------|--------------|
| Naive async | — | — | ✓ | ~1% |
| VLASH | ✗ | ✓ (mixed context) | ✓ | baseline |
| Online RL (e.g. PPO) | ✓ | ✓ | ✗ (대부분) | — |
| **DEFLECT** | ✗ (offline) | **✗ (no change)** | ✓ | **baseline +4.6** |

핵심 차이:
- **유일하게 fully offline + inference 변경 없음** + flow matching 호환
- Drop-in upgrade로 기존 async VLA stack에 즉시 적용

---

## 8. 예상 날카로운 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Per-example FM loss를 log-likelihood로 쓰는 정당성은? | Heuristic. ELBO도 아니고 lower bound도 아님 — 그러나 실험적으로 효과적. Theoretical 분석 부족 |
| 2 | SFT anchor 제거시 collapse — 그렇다면 DPO의 contribution은? | Appendix H에서 SFT only vs SFT+DPO 비교 — DPO가 추가 +4.6(d=7). SFT가 stability, DPO가 delay-specific signal |
| 3 | Training delay d∈{0..4}만으로 test d=7에 일반화? | Implicit curriculum. Delay 증가에 따라 stale action이 점점 더 나빠지므로 학습된 preference가 자연스럽게 외삽 |
| 4 | π₀.₅ 외에 다른 base policy에 적용된 결과는? | 미보고. Generalization across base policies는 unknown |
| 5 | LIBERO에서 200 step만 학습 — over-fitting 우려? | 학습 데이터가 expert demo + policy-generated pair로 다양 → over-fitting 위험 낮음 |
| 6 | Anonymous repo 외에 official release 있는가? | 현재 anonymous(under review). Camera-ready에서 공개 예상 |
| 7 | Bimanual real-world setup의 robot platform은? | "Bimanual arm setup"으로만 기술, 구체적 hardware 미명시 |
| 8 | $\lambda_{\text{DPO}}$가 Kinetix 0.02 vs LIBERO 0.1로 5x 차이 — 왜? | Task의 action variance와 reward landscape 차이. Kinetix는 contact-rich, LIBERO는 manipulation — domain마다 optimal DPO weight 다름 |
| 9 | Online RL과 비교시 sample efficiency 우위는? | Online RL은 robot 실측이 필요하나 DEFLECT는 fully offline → 안전·비용 우위. SR 절대값 비교는 없음 |
| 10 | Counterfactual stale은 어떤 시간 window에서 생성? | Training delay d∈{0..4} 범위에서 random sample. 실제 deployment delay distribution과 매칭되도록 |

<!-- VERIFIED: pdf -->
