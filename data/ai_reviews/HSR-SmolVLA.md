# HSR-SmolVLA: Per-Group Error, Not Total MSE - Fine-Tuning VLA Models for 11-DoF Mobile Manipulation

> **한 줄 요약**: SmolVLA(450M)를 Toyota HSR의 11-DoF(arm 5 + gripper 1 + head 2 + base 3) 액션 공간에 맞춰 두 단계(109k HSR 사전학습 + 3,971-에피소드 pick-up fine-tuning)로 fine-tuning하여 HSR-SmolVLA를 만들고, 동시에 "aggregate MSE는 checkpoint 선택의 신뢰 가능한 신호가 아니며 per-joint-group MSE(arm/gripper/head/base)가 실 로봇 성능을 더 잘 추적한다"는 점을 60회 실 로봇 시험(Mann–Whitney p ≤ 0.010)으로 입증한 ICRA 2026 워크숍 paper.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 한계
- π₀ / π₀.₅, SmolVLA, OpenVLA 등 대부분의 VLA는 **고정 6~7-DoF Cartesian 팔**에 맞춰 사전학습되었으나, Toyota HSR(Human Support Robot)는 **11-DoF**(팔 5 + 그리퍼 1 + 머리 2 + 휠 기반 베이스 3)를 제어해야 한다 — 대부분의 사전학습 코퍼스에 부재한 조합.
- 11차원이 단일 MSE로 합산되면, **easy-to-predict 관절(예: 머리)이 hard-to-predict 관절(예: 모바일 베이스 또는 그리퍼)을 마스킹**한다. Aggregate MSE에서 경쟁력 있어 보이는 checkpoint가 실제 로봇에서 deployment-critical group이 underfit인 채로 실패한다.

### 핵심 질문
- **이질적(heterogeneous) 액션 공간**에서 어떤 offline metric이 실 로봇 성공률을 가장 잘 예측하는가?
- 어느 joint group이 학습-time bottleneck이고 어느 group이 deployment-time bottleneck인가? 둘이 같은가?

📌 [Figure 1 삽입] — SmolVLA(450M) end-to-end 파이프라인: head/hand 480x640 카메라 + 언어 + 11D joint state → SmolVLM2-500M(frozen) → Flow-matching action expert(trained) → 50-step x 11D 액션 청크(129 ms).

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

- **Backbone**: SmolVLA(450M). SmolVLM2-500M(VL encoder) **frozen**, Flow-matching action expert만 학습.
- **Native 11-DoF 출력**: SmolVLA가 configurable action dimensionality를 지원하므로 아키텍처 수정 없이 11로 설정.
- **출력 형식**: 50 미래 스텝 x 11D 액션 청크, 129 ms 추론.
- **비교 모델**: π₀.₅(3.3B; PaliGemma2 3B + diffusion expert ~300M). LoRA는 제공된 checkpoint와 호환되지 않고 full fine-tuning은 24 GB VRAM 초과 → 저자는 **expert-only fine-tuning**(backbone freeze + 액션 헤드만 학습)을 사용.

### 2.2 Per-Joint-Group Error 분해 (Eq. 1)

```
MSE_g = (1/|g|) * sum_{j in g} (a_hat_j - a_j)^2,  g in {arm, gripper, head, base}
```

11D 액션 벡터를 4개 functional group으로 분해하여 각 group의 평균 MSE를 보고. 이것이 본 논문의 **유일한 신규 평가 도구**이자 핵심 주장 — "어느 group이 ceiling을 결정하는지가 model과 regime에 따라 달라진다".

> ❓ **예상 질문**: Group 분해 자체는 trivial한데 왜 contribution이 되는가?
> **답변**: 저자는 "VLA 문헌은 aggregate MSE 또는 task success만 보고하고 joint group별 systematic decomposition은 없다"고 명시한다. 본 논문은 (i) per-group이 deployment ranking과 일치한다는 실증, (ii) "어느 group이 결정적인지가 regime마다 다르다"는 비자명한 발견(SmolVLA에서는 base, π₀.₅ expert-only에서는 arm)을 제공한다.

### 2.3 두 단계 SmolVLA 학습 (HSR-SmolVLA 생성 절차)

- **Phase 1 (사전학습)**: 109,269 에피소드 HSR-only subset(531 tasks; AIRoA / ICRA 2026 워크숍 dataset), 20k step, batch 32.
- **Phase 2 (task fine-tuning)**: 6-task HSR pick-up release(3,971 에피소드), 50k step, batch 32, 5k마다 checkpoint 저장.
- **HSR-SmolVLA = B32 40k checkpoint**: total MSE 1.61 x 10^-3 (-80.2% vs. Phase-1 끝의 8.12)로 Phase 2 학습 궤적의 최솟값.

### 2.4 π₀.₅ Expert-Only Fine-Tuning

- 80k step의 워크숍 baseline을 시작점으로 PaliGemma2 백본(~3B) freeze, expert head(~300M)만 최대 5k step 학습.
- "메모리적으로 실제 사용 가능한 유일한 전략"이라고 명시.

> ❓ **예상 질문**: 이 논문이 새로운 모델 제안 paper인가, 분석 paper인가?
> **답변**: **둘 다**다. 명시적 deliverable은 HSR-SmolVLA(B32 40k checkpoint)와 "per-group error를 checkpoint selection 신호로 쓰라"는 가이드라인. 신규 architecture는 없고, SmolVLA의 native 11D 출력을 활용한 fine-tuning pipeline + per-group analysis가 contribution.

---

## 3. 데이터 전략

| Phase | Dataset | 규모 | 용도 |
|------|---------|-----|-----|
| Phase 1 | AIRoA 10k / ICRA 2026 워크숍 HSR-only subset | 109,269 ep / 531 tasks | SmolVLA action expert 사전학습 |
| Phase 2 | 워크숍 6-task pick-up release | 3,971 ep | 태스크별 fine-tuning |
| π₀.₅ baseline | 워크숍 사전학습 80k / 100k | (다중 embodiment + 모바일 베이스 포함) | 비교 baseline |

핵심 비대칭: SmolVLA는 109k의 **HSR-only** subset만, π₀.₅는 "다양한 embodiment + 모바일 베이스 제어 포함" 더 큰 corpus로 사전학습. 저자는 "controlled study가 아닌, 실무자가 실제로 손에 넣을 수 있는 두 checkpoint를 비교한 것"이라고 명시.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 1 x NVIDIA RTX 3090 (24 GB) |
| Batch size | 32 |
| Phase 1 steps | 20k |
| Phase 2 steps | 50k (HSR-SmolVLA = 40k checkpoint) |
| Backbone | SmolVLM2-500M (frozen) |
| Action expert | Flow matching, 50-step x 11D 청크 |
| 추론 chunk | 129 ms |
| π₀.₅ baseline | 80k 워크숍 checkpoint |
| π₀.₅ expert-only | 최대 5k step, 백본 freeze |

---

## 5. 실험 설계 및 평가 프로토콜

### 5.1 Offline Evaluation
- 10개 held-out 에피소드, 에피소드당 20 frame uniform sampling → **총 200 frame**.
- 각 frame에서 head/hand 카메라 + state + 언어 instruction으로 single-step 추론.
- per-group MSE와 total MSE 계산.

### 5.2 Real-Robot Validation
- Toyota HSR, **60 trials (모델당 20)**, 머그컵 + Cheez-It 크래커 상자 픽업 태스크.
- 4점 rubric: **4 = 완전 성공, 3 = 객체 접촉 but lift 실패, 2 = 접근 but grasp 실패, 1 = 접근조차 못함**.
- 통계 검정: Mann–Whitney U test.

📛 평가 범위: **단일 embodiment(HSR), 단일 태스크 유형(pick-up), 통제된 초기 조건**.

---

## 6. 실험 결과 심층 분석

### 6.1 SmolVLA Per-Group Trajectory (Table I)

| Checkpoint | Arm | Gripper | Head | Base | **Total** |
|-----------|-----|---------|------|------|----------|
| Pretrained 20k | 2.67 | 53.34 | 0.02 | 7.54 | 8.12 |
| B32 5k | 1.52 | 11.20 | 0.01 | 7.73 | 3.82 |
| B32 15k | 1.04 | 6.65 | 0.01 | 3.92 | 2.15 |
| B32 20k | 1.07 | 4.64 | 0.01 | 5.56 | 2.43 |
| **HSR-SmolVLA (B32 40k)** | **0.88** | **3.77** | **0.01** | **3.18** | **1.61** |
| B32 50k | 1.21 | 6.06 | 0.01 | 4.55 | 2.34 |

- **Gripper +92.9% 개선**(53.34 → 3.77), **Base +57.8%만**(7.54 → 3.18) — Base가 마지막까지 수렴 안 되고 ceiling을 결정.
- 40k 이후 50k에서 다시 악화 → **overfitting regime 진입**.
- Head MSE가 거의 0인 것은 dataset 특성(pick-up 태스크에서 객체가 초기 시야 안에 있어 head 재정렬 거의 불필요) 때문이라고 저자가 명시.

### 6.2 π₀.₅ Baseline vs. Expert-Only (Table II)

| Checkpoint | Arm | Gripper | Head | Base | **Total** |
|-----------|-----|---------|------|------|----------|
| Baseline 80k | 0.30 | 4.31 | 0.05 | 1.85 | **1.04** |
| Baseline 100k | 0.33 | 5.46 | 0.22 | 1.91 | 1.21 |
| Expert-only 1k | 0.48 | 2.59 | 0.06 | 1.87 | 0.97 |
| **Expert-only 3k** | 0.59 | **2.21** | 0.06 | **1.72** | **0.95** |
| Expert-only 4k | 0.54 | 2.20 | 0.09 | 1.79 | 0.95 |
| Expert-only 5k | 0.52 | 4.21 | 0.07 | 1.89 | 1.15 |

- **Expert-only 3k가 lowest total MSE(0.95)**, 그리고 base error도 줄어듦(1.85 → 1.72). 그러나 **arm은 0.30 → 0.59로 악화**.
- 100k baseline은 80k보다 나쁨 → 사전학습된 모델조차 overfitting.

### 6.3 Real-Robot (Table III) — 핵심 결과

| 모델 | Total MSE | n1 | n2 | n3 | n4 | **Mean** |
|------|----------|----|----|----|----|---------|
| **π₀.₅ 80k (baseline)** | 1.04 | 0 | 0 | 0 | **20** | **4.0/4** |
| Expert-only 3k | **0.95** (lowest) | 0 | 0 | 5 | 15 | 3.75/4 |
| HSR-SmolVLA (40k) | 1.61 | 0 | 2 | 6 | 12 | 3.5/4 |

- **π₀.₅ 80k: 20/20 완전 성공**.
- Expert-only 3k는 total MSE가 가장 낮음에도 **3.75/4**, p = 0.010으로 baseline에 유의하게 패배.
- HSR-SmolVLA는 3.5/4 (12 lifts / 20), p = 0.001.
- Expert-only 3k vs. HSR-SmolVLA는 p = 0.128 → **통계적으로 구분 불가**.
- 핵심 관찰: deployment ranking과 일치하는 유일한 offline metric은 **arm-group MSE**(0.30 < 0.59 < 0.88). Total MSE(0.95 < 1.04 < 1.61)는 baseline을 가장 나쁘게 잘못 예측한다.

> ❓ **예상 질문**: HSR-SmolVLA failure pattern은?
> **답변**: 저자가 정성적으로 보고 — "20cm 떨어진 위치에서 gripper를 닫아 empty space를 grasp." SmolVLA의 0.88 arm MSE(vs. π₀.₅의 0.30)가 grasp calibration 오류로 직결.

---

## 7. Ablation 분석

본 논문에는 전통적 ablation table은 없지만, 다음 비교가 ablation 역할:

### 7.1 학습 길이 ablation (SmolVLA)
- B32 40k에서 1.61로 최저, 50k에서 2.34로 다시 상승. → **3,971 에피소드 fine-tuning에서 ~40k가 sweet spot**.

### 7.2 학습 길이 ablation (π₀.₅ expert-only)
- 3k가 sweet spot(0.95). 5k에서 gripper MSE가 4.21로 급등 → **overfitting**.
- "Both SmolVLA (40k 이후) and π₀.₅ expert-only (3k 이후) enter overfitting regime"가 deployment guideline #3.

### 7.3 Group별 수렴 속도 (Fig. 2의 log-scale 곡선)
- Gripper: 가장 빠르게 수렴 (10배 빠름).
- Base: 가장 느리게 수렴, total MSE의 ceiling 설정.
- Head: 거의 0 유지 (dataset bias 영향).
- Arm: 중간 속도.

---

## 8. 관련 연구 비교

| 모델 | Architecture | 평가 metric | 11-DoF native | 본 paper와의 차이 |
|------|-------------|------------|-------------------|------------------|
| OpenVLA (7B) | 7-DoF Cartesian token | aggregate MSE / success | ✗ | 고정 팔 전용 |
| RT-2 | 다중 7-DoF | success | ✗ | 고정 팔 전용 |
| π₀ / π₀.₅ | Flow-matching expert | success | △ (multi-embodiment) | 본 paper의 비교 baseline |
| **SmolVLA(450M)** | Flow-matching, configurable D | aggregate MSE / success | **✓** | 본 paper의 base model |
| MoManipVLA | 고정 팔 VLA + 모바일 어댑터 | success | ✗ (retraining 없이 adapt) | 비교 군 |
| **HSR-SmolVLA (본 연구)** | **SmolVLA + native 11D + per-group selection** | **per-group MSE + 4-pt rubric** | **✓** | per-group을 deployment proxy로 검증 |

핵심 차이: 본 연구는 **새 architecture 제안이 아닌, evaluation methodology(per-group MSE) + 실용적 11-DoF fine-tuning pipeline**.

---

## 9. 한계 및 미해결 문제

### 저자가 명시한 limitations
1. **60 trials만**: 통제된 초기 조건의 pick-up 단일 태스크. 다른 태스크 / embodiment 일반화는 future work.
2. **Frame-level metric**: Offline MSE가 trajectory-level 누적 오차를 못 잡음.
3. **Controlled architecture study 아님**: SmolVLA vs. π₀.₅는 서로 다른 사전학습 corpus → 단순 architecture 비교가 아닌 "실무자가 실제로 얻을 수 있는 두 checkpoint" 비교.
4. **두 fine-tuned model 간 ordering 불가**: Expert-only 3k vs. HSR-SmolVLA는 p = 0.128로 통계적으로 구분 안 됨.

### 추가 미해결 문제
5. **Loss balancing 미적용**: 저자가 인정 — uncertainty weighting [Kendall+18], GradNorm [Chen+18]을 적용하면 small-magnitude group(arm)을 보호하면서 base/gripper를 개선할 수 있을 것. **본 paper는 진단까지만 하고 치료는 future work**.
6. **Head MSE의 신뢰성**: 데이터셋 bias(객체가 시야 안)이라고 저자가 솔직히 인정 — head group의 낮은 MSE는 일반 결론 아님.
7. **Per-group metric의 generality**: arm이 deployment proxy라는 발견이 다른 태스크(예: navigation-heavy)에서도 성립하는지 불명.
8. **다른 embodiment 일반화**: HSR 외 다중 embodiment에서 검증 안 됨.

### Attribution 우려
- HSR-SmolVLA의 약점(0.88 arm MSE)이 SmolVLA architecture 한계인지 데이터 부족(3,971 에피소드)인지 분리 불가.
- Expert-only 3k가 baseline에 패배한 게 "arm 악화" 때문인지 단순 distribution drift인지의 인과 분리 부족.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★☆☆☆ — Per-group MSE 분해 자체는 trivial. 신규성은 "이게 deployment proxy로 쓸만하다"는 실증과 model별 bottleneck group이 다르다는 발견에 있음. |
| **Technical depth** | ★★★☆☆ — 신규 architecture/loss 제안 없음. 진단까지만, 치료(loss balancing)는 future work로 미룸. |
| **Experimental rigor** | ★★★★☆ — 60 trials + Mann–Whitney 통계 검정, total/per-group/real-robot 세 축의 일관된 ranking 비교 등 ICRA 워크숍 paper치고 견고. |
| **Practical impact** | ★★★★☆ — "Total MSE만 보지 마라"는 가이드라인은 실무자에게 직접적 가치. HSR-SmolVLA 자체도 single-GPU(RTX 3090)로 재현 가능. |
| **Writing quality** | ★★★★☆ — 명확한 두 단계 narrative(SmolVLA: base bottleneck → π₀.₅ expert-only: arm bottleneck). |

**강점**: ICRA 워크숍 paper의 scope를 명확히 인지하고 그 안에서 통계적으로 검정 가능한 실험을 설계. 실 로봇 60 trials는 워크숍 paper 기준 견고하며, "total MSE가 lowest인데 robot에서 패배"라는 counterintuitive case를 깔끔히 격리.
**약점**: 신규 architecture/loss 없음. Per-group 분해는 누구나 할 수 있는 단순 metric. Future work로 미룬 uncertainty weighting / GradNorm을 실제로 적용했다면 paper의 contribution이 훨씬 커졌을 것.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Per-group MSE 분해 자체는 trivial한데 contribution이 뭔가? | "VLA 문헌은 aggregate MSE / success만 본다"는 gap을 채움 + 어느 group이 결정적인지가 regime마다 다르다는 비자명한 발견(SmolVLA = base, π₀.₅ expert-only = arm) |
| 2 | Expert-only 3k가 lowest total MSE인데 robot에서 지는 이유? | Arm group이 0.30 → 0.59로 악화(딴 group이 dominate해서 capacity가 arm에서 빼앗김). 5/20 trial에서 gripper는 닫혔지만 lift 실패가 정성적 증거. |
| 3 | SmolVLA의 base group이 끝까지 수렴 안 하는 이유? | SmolVLA의 LeRobot 사전학습이 거의 고정 팔이라 base 노출 없음. 109k Phase 1 + 3,971 Phase 2도 부족. π₀.₅는 multi-embodiment + 모바일 base 사전학습이라 base error가 1.85로 처음부터 낮음. |
| 4 | Real-robot 60 trials면 통계적으로 충분한가? | Baseline vs. fine-tuned는 p ≤ 0.010으로 충분. 두 fine-tuned model 간(p = 0.128)은 부족. 저자가 명시적으로 한계 인정. |
| 5 | Head MSE가 거의 0인 게 generalizable한가? | 안 됨. Pick-up 태스크에서 객체가 초기 시야 안이라 head 재정렬 거의 불필요 — 저자 명시. Navigation-heavy 태스크에선 다른 group이 결정적일 것. |
| 6 | Loss balancing(uncertainty weighting, GradNorm)을 왜 본 paper에서 안 했나? | 저자가 명시적으로 future work. 본 paper는 진단까지만 — "이질적 group이 한두 자릿수 magnitude 차이라 unweighted objective가 small-magnitude group을 희생시킨다"는 메커니즘만 제시. |
| 7 | LoRA 못 쓴 이유는? | "Provided π₀.₅ checkpoint의 architecture와 incompatible." Full fine-tuning은 24 GB VRAM 초과 → expert-only가 유일한 메모리적 선택지. |
| 8 | HSR-SmolVLA vs. expert-only 3k 중 누가 deploy해야 하나? | 통계적으로 구분 안 됨(p = 0.128). 둘 다 baseline 80k보다 못함 → "건드리지 마라"가 본 paper의 실용적 결론. |
| 9 | π₀.₅ 100k이 80k보다 나쁜 이유? | 사전학습된 모델조차 overfitting. 저자는 "scale and diversity of pretraining"이 중요하다고 결론(scale만으로는 부족). |
| 10 | Per-group MSE가 항상 arm-discriminative한가? | 아니다 — SmolVLA에서는 base가 bottleneck, π₀.₅ expert-only에서는 arm이 discriminative. **Regime별로 dominant group을 먼저 식별하는 게 deployment guideline #2**. |

<!-- VERIFIED: pdf -->
