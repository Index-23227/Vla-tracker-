# Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack

> **한 줄 요약**: 4B MoT VLM(Hy-Embodied-0.5)에 370M flow-matching action expert와 compact temporal-spatial memory encoder를 결합하고, 10K시간 fingertip-UMI 데이터로 사전학습한 뒤 FlowPRO(critic/reward-free preference RL)와 cubic-Bézier 비동기 추론으로 마무리한 **end-to-end VLA 풀-스택**. RoboTwin 2.0에서 90.9/90.1% (Clean/Randomized) SOTA, Dobot X-Trainer 4개 정밀 조작 과제에서 RPRO 후 99/99/98/94% 성공률을 달성.

---

## 1. 배경 및 동기

- 작년 한 해 동안 VLA 연구가 폭증했으나, "배포 가능한 generalist robot"은 단일 모델만으로 만들 수 없고 데이터 수집 ~ 실세계 배포에 이르는 **풀 스택**의 공동 설계가 필요하다는 문제의식.
- 저자들은 데이터 측면 3가지 결합 문제를 지적: (i) 전통적 master-slave 원격조작은 햅틱 피드백 부재로 정밀 조작 불가, (ii) UMI 같은 hand-held rig는 SLAM 기반이라 fingertip-level force를 못잡음, (iii) cross-embodiment gap은 motion space + control dynamics + perception viewpoint의 3중 격차.
- 모델/배포 측면에서도 (i) discrete-token autoregressive는 빈도·정밀도 한계, (ii) flow-matching VLA(π0류)는 일반 VLM 백본이라 spatiotemporal reasoning이 부족, (iii) imitation은 last-mile dexterity가 약하고 reward-based RL은 contact-rich에서 brittle, (iv) high-frequency closed-loop 배포가 first-class 설계 대상이 아니라는 4중 병목.

---

## 2. 방법론

### 2.1 백본 (Sec. 2.2)
- Hy-Embodied-0.5-MoT (4B, edge 최적화). HY-ViT 2.0 native-resolution ViT + Mixture-of-Transformers (vision/text별 비공유 QKV·FFN, 공유 self-attention).
- Visual token에는 frame-내 양방향 attention, language token에는 causal attention.

### 2.2 Dual-Tower Flow-Matching Action Expert (Sec. 2.3)
- VLM tower와 action-expert tower 분리; 공유 self-attention으로 결합.
- 토큰 시퀀스 3-block: [It, ℓ], [st], [noisy action]. Block-내 양방향, block-간 causal. State block의 KV는 캐싱.
- 손실: L_fm(θ) = E ‖v_θ(A^τ_t, o_t) − (ε − A_t)‖² (Eq. 3). τ는 high-noise로 skewed Beta. Auxiliary co-training은 NTP loss.
- 추론: 10-step forward-Euler (δ=0.1), prefix KV는 캐시 → action token만 재계산.

### 2.3 Compact Memory Encoder (Sec. 2.4)
- ViT 블록에 L층마다 temporal causal attention을 끼워넣음 (Eq. 4-5). 동일 QKV·W_O 공유, sinusoidal temporal encoding(e(0)=0)으로 **추가 학습 파라미터 0**.
- Upper layer에서 과거 frame 토큰을 drop → single-frame과 동일한 토큰 수를 backbone에 전달.
- K=1이면 정확히 기존 Hy-ViT로 환원 → pre-trained weight 직접 로드 가능.

### 2.4 Delta-Chunk Rel-EE 표현 (Sec. 2.1, 5.1)
- 한 팔당 10-D: xyz(3) + 6-D continuous rotation + gripper(1). 모든 chunk는 chunk-시작 EE 기준의 **상대** 변위.
- 배포 시 W·T_Gt를 곱해 절대 world pose 복원 후 IK. Humanoid는 고정 chassis frame W·T_C를 캐시 + 24-D head/torso는 휴리스틱.

### 2.5 Hy-UMI-10K 데이터 (Sec. 3.1)
- Fingertip UMI gripper (Changingtek CTAG2F90 기반) + 외부 mocap → sub-mm 6-DoF 라벨, head RGB-D + IR 동기화. 6-D F/T 센서 옵션.
- 1M+ episode, 10K시간, 70 task, 6개 장면 가족 (Laundry 28.5% / Kitchen 19.2% / Personal Care 13.8% / Dexterous 10.4% / Storage 10.0% / Cleaning 5.7%).

### 2.6 사전학습 & SFT (Sec. 3.2-3.3)
- Pre-train: K=1, 3 views @ 224×320, H=50 @ 10 Hz, 200K steps, batch 1024, lr 5e-5, AdamW + bf16. Action expert는 hidden 2048→1024, FFN 6144→2048로 축소(370M).
- SFT: K=6 (5개 history + 현재), 50 Hz, H=50(real)/H=20(sim), 60K steps, batch 32, lr 2.5e-5.
- Track A (intra-embodiment, Dobot X-Trainer 원격조작) vs Track B (cross-embodiment, UMI만으로 JAKA K1 / Astribot S1 배포).

### 2.7 FlowPRO (Sec. 4)
- 3가지 설계 원칙: (P1) 실패 직접 활용, (P2) reward/critic 모델 전무 (flow-matching log-likelihood proxy로 implicit reward 계산), (P3) symmetric proximal regularizer로 reward 절대값 anchoring.
- Implicit reward: r_θ(s,a) = β/2 · (ℓ_ref(s,a) − ℓ_θ(s,a)) (Eq. 7). PRO pairwise + proximal regularizer (Eq. 8) + SFT term (Eq. 9: L_RPRO = λ_PRO L_PRO + λ_SFT L_SFT).
- 데이터 파이프라인: 원격조작 intervention-and-rollback → (τ_w, τ_l) 쌍. Smooth Interpolation으로 dense (s, a_w, a_l) 합성 (cubic Bézier 위치 + Slerp 회전 + linear gripper).
- Contrastive gradient cancellation: a_w=a_l일 때 ∇L_con=0 → SFT 샘플을 같은 loss로 안전하게 mix 가능.
- 배치 mix: 1라운드 80/20(D^k_pref / D_SFT), 2+라운드 70/15/15(D^k / D^<k / D_SFT).

### 2.8 비동기 배포 (Sec. 5)
- Producer-consumer 런타임: inference thread가 buffer B를 smoothed action으로 overwrite, execution thread는 servo rate로 pop.
- Latency-aware cubic-Bézier stitcher: 신선한 chunk의 stale prefix K=⌈N/α⌉ 절단 → 내부 connection point f_c (γ로 선택) → 4-점 cubic Bézier B(t) (Eq. 17)로 history tangent와 future tangent를 일치, C¹-continuous 전이. SLERP(회전) + linear(gripper).

---

## 3. 실험 결과

### 3.1 RoboTwin 2.0 (Sec. 6.1, Table 1)
- 50개 task, task당 100 rollout 평균.
- **HyVLA-0.5: Clean 90.9 / Randomized 90.1** — 두 setting 모두 SOTA.
- 비교: π0 65.9/58.4, π0.5 82.7/76.8, ABot-M0 81.2/80.4, Qwen-VLA 86.1/87.2, LingBot-VLA 86.5/85.3, starVLA 88.2/88.3, Motus 88.7/87.0, **JoyAI-RA 90.5/89.3** (차순위).
- 차이: vs π0 +25.0/+31.7, vs π0.5 +8.2/+13.3, vs JoyAI-RA +0.4/+0.8.
- Ablation: memory encoder 제거 → 88.8/88.6, UMI pre-training까지 제거 → 88.1/87.9 (시뮬-vs-egocentric 도메인 갭에도 불구하고 일관된 이득).

### 3.2 실세계 SFT (Sec. 6.2)
- **Track A (Dobot X-Trainer 4-task)**: Insert Bottles, Fold & Store Glasses, Set the Table, Zip Up Pen Case. UMI pre-training이 sub-cm 정밀 구간(temple folding, zipper pinch)의 예측을 sharpen해 end-to-end 성공률 상승.
- **Track B (UMI-only cross-embodiment)**: JAKA K1 "Put Away the Accessory" (hair tie를 cell-tight slot에 넣기) + Astribot S1 "Clean Up the Table" (paper cup 분리수거). target-robot 원격조작 0개로 π0/π0.5 대비 상당한 향상 — UMI prior가 형태가 다른 로봇에 살아남음을 입증.
- **Unitree G1 force task**: 2개 box 중 가벼운 것을 식별해 앞 바구니에 넣음. ~2M 파라미터 TCN encoder + MLP projector를 action expert에 추가 + UMI tip-F/T 신호로 SFT. spatial memory 무관, force 비교가 필수.

### 3.3 FlowPRO 강화학습 (Sec. 6.3, Table 2)
- 같은 SFT checkpoint π_ref에서 K=3 라운드, 3 seed × 100 rollout/task.
- Bottle / Cap / USB / Zip (SR%):
  - DAgger: 93 / 88 / 86 / 83
  - π0.6*: 95 / 95 / 95 / 89
  - **RPRO: 99 / 99 / 98 / 94** (모든 task 1위, CT도 최소)
- 해석: positive-only DAgger 대비 negative 활용으로 per-state push-away gradient, advantage-conditioning π0.6* 대비 contrastive loss로 preference signal을 직접 주입.

---

## 4. 한계 및 미해결 문제

1. **Mocap 의존성**: 10K시간 라벨 정밀도가 외부 광학 mocap cage에 의존하므로 in-the-wild 확장이 어렵다. 저자들도 exoskeleton 기반 수집을 차기 방향으로 명시.
2. **시뮬 도메인 갭**: UMI pre-training의 RoboTwin 2.0 이득이 88.1→90.9로 +2.8점에 그침. 합성 렌더링과 egocentric real의 시각 분포 차이가 큼.
3. **Cross-embodiment 평가 범위**: Track B는 JAKA K1·Astribot S1 각 1 task에 그쳐 일반화 주장의 N이 작다.
4. **FlowPRO 데이터 단위**: 원격조작 intervention-and-rollback이 인간 operator-in-the-loop를 요구 → 완전 autonomous loop가 아니라 실세계 데이터 비용이 남아있음.
5. **Zero-shot 미평가**: 저자 스스로 "현재 데이터 규모는 zero-shot generalization 주장에 불충분"이라 명시.
6. **Force 모달리티**: 1 task의 qualitative 검증에 그침. Tactile-conditioned policy의 정량 벤치는 향후 과제.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 개별 컴포넌트(MoT VLM, flow matching, UMI, DPO 계열, async deployment)는 익숙하나, "정밀 UMI 데이터 + delta-chunk rel-EE + RPRO preference RL + Bézier stitching"을 하나의 deploy-able 스택으로 묶어낸 시스템적 통합이 강함. RPRO의 proximal regularizer + contrastive gradient cancellation은 신선. |
| **Practical impact** | ★★★★★ — RoboTwin 2.0 SOTA + 4개 정밀 실세계 task RPRO 후 ≥94% + UMI-only cross-embodiment 성공 + 오픈소스(Github/HF). Tencent 산업 인프라와 결합되면 deployable generalist에 매우 근접. |

요약하면, HyVLA-0.5는 "VLA를 하나의 모델이 아니라 데이터 ~ 배포의 풀 스택으로 본다"는 명제를 실험적으로 입증한 보고서다. 특히 reward-free offline RL인 FlowPRO가 contact-rich precision task의 last-mile을 메우는 실용적 도구로 안착할 가능성이 있다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Memory encoder가 추가 파라미터 0인데 정말 작동하나? | Sinusoidal e(0)=0 + 공유 QKV·W_O로 K=1일 때 identity로 환원 → pre-trained Hy-ViT weight를 그대로 로드, K>1에서 점진적으로 temporal 정보를 흡수. RoboTwin ablation에서 +2.1점 기여 입증. |
| 2 | RPRO가 plain Flow-DPO보다 좋은 이유는? | DPO 계열은 implicit reward 절대값이 발산 가능 → reward hacking. RPRO의 symmetric proximal regularizer는 r_θ(s,a)=0에서 최소화·|r|에 대칭 증가 → 절대값 anchoring. 또 contrastive gradient cancellation으로 SFT 샘플을 같은 loss에 안전히 mix. |
| 3 | UMI-only로 Track B(JAKA/Astribot)가 작동하는 핵심은? | Rel-EE delta-chunk 표현이 embodiment-specific 운동학을 deployment 시점으로 미루고, world frame composition (Eq. 10-11)으로 fixed-arm/humanoid 모두에 동일한 학습 인터페이스 유지. UMI prior가 정밀 구간 행동분포를 sharpen해주는 게 ablation에서 일관되게 관찰됨. |
| 4 | Asynchronous Bézier stitching이 학습 trick과 무엇이 다른가? | Training-free, policy-agnostic, plug-and-play. RTC류는 별도 refinement module을 학습해야 하지만, 본 방법은 history tangent + future tangent를 4-점 cubic Bézier로 결정해 C¹ continuity를 명시적으로 보장. Cartesian/joint-space 모두 적용 가능. |

---

## 7. 핵심 수식

- **Flow-matching loss** (Eq. 3): L_fm(θ) = E ‖v_θ(A^τ_t, o_t) − (ε − A_t)‖².
- **Auxiliary NTP** (Eq. 2): L_ntp(θ) = −E Σ_j log p_θ(y_j | c, y_<j).
- **Implicit reward** (Eq. 7): r_θ(s,a) = β/2 · (ℓ_ref(s,a) − ℓ_θ(s,a)), ℓ는 per-sample flow-matching regression loss.
- **RPRO 손실** (Eq. 8-9): L_PRO = −E[log σ(r_θ(s,a_w)−r_θ(s,a_l)) + ½ Σ_{a∈{a_w,a_l}} (log σ(r_θ(s,a)) + log σ(−r_θ(s,a)))]; L_RPRO = λ_PRO L_PRO + λ_SFT L_SFT.
- **Bézier stitch** (Eq. 13-17): B(t) = (1−t)³P0 + 3(1−t)²t P1 + 3(1−t)t² P2 + t³ P3; P1=P0+λ d̂_hist, P2=P3−λ d̂_fut, λ=σ‖P3−P0‖.

---

## 8. 아키텍처 다이어그램 요약

```
[Multi-view RGB × K frames]           [Language: "Insert the RAM"]    [State s_t (rel-EE)]   [Noisy action A^τ_t]
        |                                       |                              |                         |
     HY-ViT 2.0 (native-res)               Text Tokenizer                     MLP                       MLP
        |                                       |                              |                         |
  Vision MoT QKV/FFN  <-- shared joint-attention --> Language MoT QKV/FFN  <-- shared --> Action Expert MoT (370M)
                            (block-wise causal: [I,ℓ] | [s] | [A^τ])
                                            |
                              Flow-matching velocity field v_θ
                                            |
                          10-step forward Euler  →  delta-chunk action A_t (H=50)
                                            |
                          Bézier stitch + async exec  →  servo @50 Hz
```

---

## 9. 데이터 분포 시각화 (Hy-UMI-10K)

- **Task families (10K h)**: Laundry 3,025h · Kitchen 2,040h · Personal Care & Misc 1,465h · Dexterous/Tool-use 1,110h · Storage & Organization 1,065h · Cleaning 610h · Other 1,315h.
- **Skill primitives (h)**: Bimanual Fold & Stack 3,945 · Precision Placement 3,065 · Spatial Organization 1,495 · Sequential Retrieval 1,045 · Surface Wiping 438 · Articulated 251 · Constrained Insertion 209 · Dexterous Assembly 61.
- **Object categories (h)**: Deformable Textiles 3,050 · Tableware 2,060 · Small Rigid 1,475 · Precision Instruments 1,120 · Rigid Containers 1,075 · Cleaning Implements 1,051 · Packaged Goods 674.

→ Bimanual fold/stack과 precision placement에 시간이 집중되어 있어, Track A의 Fold & Store Glasses와 Zip Up Pen Case에 대한 prior 효과를 설명함.

---

## 10. 비교 분석: π0 / π0.5 / π0.6* 와의 차별점

| 축 | π0 (2024) | π0.5 (2025) | π0.6* (2025) | **HyVLA-0.5** |
|---|---|---|---|---|
| VLM 백본 | PaliGemma | PaliGemma | PaliGemma + advantage 조건화 | **Hy-Embodied-0.5-MoT 4B (embodied-native, native-res ViT)** |
| Action 모델 | Flow matching | Flow matching + 개방형 일반화 | Advantage-conditioned regression | Flow matching + **compact memory encoder** + **delta-chunk rel-EE** |
| 사전학습 데이터 | 다 embodiment 텔레옵 | 텔레옵 + 인터넷 | 텔레옵 | **10K h fingertip-UMI + mocap** |
| Post-training | SFT/DAgger | SFT | Advantage-conditioned RL (reward model 필요) | **FlowPRO (critic/reward-free, RPRO loss)** |
| 배포 | 표준 chunk | 표준 chunk | 표준 chunk | **Async + Bézier C¹ stitch** |

핵심 차별점: (a) embodied-native MoT 백본, (b) sub-mm UMI 사전학습, (c) reward-free preference RL, (d) training-free C¹ chunk stitching.

---

## 11. 재현 시 점검 사항

- **공개 자산**: tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla / github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA / huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI / Hy-Embodied-0.5-VLA-Data (2,000h 서브셋 공개 예고).
- **사전학습 비용**: 200K step × batch 1024 (~2×10⁸ sample) on bf16 — 정확한 GPU·시간은 보고서가 명시하지 않음.
- **SFT 데이터량**: Dobot 4-task 합 18h, JAKA 1.2h, Astribot 1.5h, Unitree 2.2h. 200-400 demos/task로 매우 가볍다 → UMI prior가 강함을 시사.
- **하이퍼파라미터**: H=50 (real) / H=20 (sim, stride 3), K=6 memory frames, 10-step flow Euler, λ_PRO/λ_SFT 정확한 값은 본문 미명시.
- **배포 파라미터**: α (truncation ratio), γ (connection index ratio), σ (tangent length) — 하드웨어별 튜닝 필요. acceleration limit과 servo rate에 따라 조정.

---

## 12. 결론 및 향후 방향

HyVLA-0.5는 단일 모델의 점수 경쟁이 아니라 **"deployable generalist robot stack"**이라는 시스템 명제로 VLA 연구를 한 단계 끌어올린 보고서다. 4가지 축 — (i) 고정밀 UMI 데이터 인프라, (ii) MoT 백본 + flow-matching expert + compact memory + delta-chunk rel-EE 아키텍처, (iii) critic-free RPRO offline RL, (iv) C¹-continuous async deployment — 가 각자 다른 병목을 공략하며 동일한 학습 인터페이스를 보존한다.

저자들이 명시한 향후 과제: (1) **mocap-free 고정밀 데이터** (exoskeleton) 및 라벨 노이즈가 pre-training에 미치는 marginal value 연구, (2) **시각 augmentation** (UMI egocentric vs robot-mounted camera 격차), (3) **deployment-time RL**과 결합한 실행 효율 향상, (4) **zero-shot 평가 프로토콜** 정립 — π0.7류의 emergent behavior를 측정하는 방법론.

요컨대, HyVLA-0.5는 RoboTwin 2.0 SOTA와 실세계 4개 task ≥94% RPRO 결과를 넘어, "VLA 모델 한 줄"이 아니라 **데이터-아키텍처-RL-배포**가 합쳐진 stack을 일관된 인터페이스로 묶는 것이 generalist 로봇으로 가는 길임을 명확히 한 작업이다.

<!-- VERIFIED: pdf -->
