# CLEAR: Closed-Loop Reinforcement Learning at Scale for End-to-End Autonomous Driving

> **한 줄 요약**: SimLingo 스타일 VLA(InternVL3-1B)를 3.1M 전문가 샘플로 IL pretraining한 뒤, pretrained waypoint prior 주변의 **residual waypoint policy**를 PPO로 closed-loop 학습 — 시뮬레이터(32×V100, 64 CARLA 환경)와 learner(8×H100)를 분리한 **heterogeneous pipeline**으로 100M 샘플까지 RL을 스케일링하여, CARLA longest6 v2에서 DS 39.89/SR 25%(IL 베이스라인 전원 SR 0%), Bench2Drive에서 DS 86.8/SR 69.5%로 SotA를 달성한 Qualcomm AI Research의 자율주행 RL-VLA 시스템(arXiv 2607.02841, 2026.07).

---

## 1. 배경 및 동기

- E2E 자율주행(E2E-AD)용 VLA 정책들(EMMA, ORION, SimLingo 등)은 대부분 **imitation learning(IL)** 기반 — logged expert trajectory에 대한 L2 등 distance metric만 최적화
- **Open-loop 학습 ↔ closed-loop 추론 간 distribution shift**: 자기 행동의 결과로 상태 분포가 바뀌는 closed-loop 환경에서 IL 정책은 오차가 누적되며 실패 (longest6 v2에서 모든 IL 방법이 SR 0%)
- 기존 closed-loop RL 연구(PlanT, CaRL 등)는 **privileged planning** — 완벽한 perception GT(3D box, BEV seg, HD map)를 입력으로 요구 → 실제 배포 조건과 불일치
- ReCogDrive 등 open-loop RL은 여전히 closed-loop에서 부족
- 핵심 질문: **비특권(vision-only) VLA 정책을 closed-loop RL로, 그것도 대규모로 finetuning할 수 있는가?**

## 2. 방법론 심층 분석

### 2.1 전체 구조 (2-stage)

```
Stage 1 (IL pretraining):
  multi-camera image + ego speed + navigation + language prompt
    → InternVL3-1B (InternViT-300M + Qwen2.5-0.5B, LoRA)
    → action queries q_p, q_w → MLP ψ_p, ψ_w
    → path waypoints P̂_t, speed waypoints Ŵ_t   (smooth L1 + VQA loss)

Stage 2 (closed-loop RL):
  P̂_t 을 m=4 anchor 위치에서 보간 → base lateral offsets ŷ
  Ŵ_t → base target speed v̂
  z_t = MLP([h_t, ŷ¹..ŷᵐ, v̂])
  a_t = [Δy¹..Δyᵐ, Δv] ~ π_θ(·|z_t)          ← PPO가 최적화
  최종 waypoint = clip(ŷ+Δy), clip(v̂+Δv)
  deterministic controller g → (steering, merged throttle/brake)
```

### 2.2 Residual Waypoint Policy (핵심 기여 1)

- Direct-control 정책(steering/throttle/brake 직접 출력)은 waypoint 기반 pretraining 표현과 **misaligned** → pretrained 지식을 버리게 됨
- 대신 pretrained VLA를 **trajectory prior**로 취급, RL은 anchor별 lateral residual Δy와 speed residual Δv만 학습 (clipping으로 bounded)
- Controller는 environment transition의 일부로 취급 → controller를 통한 gradient 불필요
- RL 중 vision encoder와 LLM은 **전부 freeze**, autoregressive language head도 비활성화 (compute 제약)

### 2.3 Heterogeneous Finetuning Pipeline (핵심 기여 2)

- 문제: CARLA는 GPU rendering이 불안정(특히 docker), 서버 1개당 ~6GB VRAM → H100(80GB)에서 VLA와 동거 시 **resource contention**으로 스케일링 불가
- 관찰: CARLA/Unreal은 구형 GPU(V100)와 구버전 드라이버(≤535)에서 오히려 원활
- 해법: **시뮬레이터는 32×V100(GPU당 2개, 총 64 환경), learner는 8×H100** — autossh 터널로 통신
- 결과: 100M 샘플, total-batch 16384 / mini-batch 4096의 DD-PPO 업데이트 가능, 학습 ~6일

### 2.4 보상 설계

- r_t = RC_t·(∏p_t) − P: CaRL의 discounted Route Completion + soft penalty factor p_t ∈ [0,1] + hard penalty P(충돌, 신호위반 등 → episode 종료)
- "simple reward"만으로 SotA — reward engineering 최소화

## 3. 데이터 및 학습

| 항목 | Stage 1 (IL) | Stage 2 (RL) |
|---|---|---|
| 데이터 | SimLingo 데이터셋 ~3.1M 샘플 @ 4fps | CaRL의 절차적 생성 CARLA 루트 |
| 하드웨어 | 8×H100 | 8×H100 (learner) + 32×V100 (64 CARLA) |
| 학습량 | 14 epochs, global batch 96 | 100M 샘플, batch 16384/4096 |
| 최적화 | AdamW, lr 3e-5 cosine, wd 0.1, LLM은 LoRA | PPO (DD-PPO), ~6일 |

## 4. 실험 결과 심층 분석

### 4.1 CARLA longest6 v2 (Table 1, 비특권 planner)

| Method | Mode | DS↑ | SR(%)↑ | RC(%)↑ |
|---|---|---|---|---|
| UniAD | IL | 4.64 | 0.00 | 8.78 |
| SSR | IL | 6.38 | 0.00 | 10.02 |
| ORION | IL+RL | 11.13 | 0.00 | 15.98 |
| SimLingo (InternVL2-1B) | IL | 13.11 | 0.00 | 19.55 |
| InternVL3-1B (IL only) | IL | 18.43 | 0.00 | 21.17 |
| **CLEAR (InternVL2-1B)** | IL+RL | 37.44 | 25.00 | 41.91 |
| **CLEAR (InternVL3-1B)** | IL+RL | **39.89** | **25.00** | **47.24** |

- 모든 IL 방법이 SR 0% (긴 루트 + 고속 배경 트래픽 + 복합 시나리오)에서 RL finetuning만으로 DS 2.2배, SR 0→25%
- 백본 무관하게 개선(InternVL2/3 모두) → 프레임워크의 일반성

### 4.2 Bench2Drive (Table 2)

- **CLEAR (InternVL3-1B): DS 86.8, SR 69.5%, Efficiency 275.4, Comfort 25.7, Multi-Ability mean 69.8%**
- SimLingo(DS 85.1/SR 67.3) 대비 개선, HiP-AD(DS 86.8, 단 †)와 동률 DS에 더 높은 SR
- Multi-Ability: Merging 57.5 / Overtaking 66.7 / E-Brake 85.0 / Give Way 70.0 / T.Sign 69.8
- 단, Overtaking(66.7)은 HiP-AD(84.4), T.Sign(69.8)은 SimLingo(82.5)에 뒤짐 — 전 능력 균일 우위는 아님

### 4.3 nuScenes zero-shot (Table 3)

- 실세계 데이터를 전혀 학습하지 않고 nuScenes val에서 **avg L2 0.49m로 최고** (Senna 0.59, ORION 0.69), collision rate 0.30%로 경쟁력 — sim-only 학습의 실세계 전이 가능성 시사

### 4.4 Ablations

- **Action space** (Table 4): waypoint residual (DS 39.89) vs direct control (DS 33.91, SR 19.44) → residual formulation이 pretrained 지식 활용에 핵심
- **Scaling** (Table 5): 10M/batch 1024-256 → DS 12.72 (수렴 실패, IL보다 나쁨), 40M/4096-1024 → 25.31, 100M/16384-4096 → 39.89 — **RL 스케일 자체가 성능의 결정 요인**

## 5. 한계 및 미해결 문제

- **Sim2Real 미검증**: 전 실험이 CARLA 시뮬레이터 내 (nuScenes는 open-loop L2뿐)
- 병렬 CARLA 환경 수(64)가 여전히 스케일링 병목 — CARLA rendering stack의 불안정성
- CaRL 보상 차용: metric의 global optimum과 일치하는 보상이 없는 경우의 reward design은 미해결
- longest6 v2 SR 25%는 여전히 낮은 절대치 — 4개 중 3개 루트는 실패
- RL 중 LLM/vision encoder freeze — 표현 자체는 개선되지 않고 residual만 학습

## 6. 관련 연구 비교

| 연구 | 접근 | CLEAR와의 차이 |
|---|---|---|
| SimLingo | IL only, vision-only VLA | CLEAR의 pretraining 기반; closed-loop RL 부재 |
| CaRL | 특권 RL planning (BEV GT 입력) | CLEAR는 비특권(vision-only); 보상 설계 차용 |
| RAD | 3DGS 시나리오 내 RL | 제한된 스케일; CLEAR는 100M 샘플 |
| AlphaDrive / AutoVLA | GRPO로 reasoning/planning 개선 | open-loop 중심; CLEAR는 closed-loop PPO |
| ReCogDrive | open-loop RL | closed-loop 검증 부족 (B2D DS 71.4 vs 86.8) |
| Raw2Drive | aligned world model RL | DS 71.4로 CLEAR에 열세 |

## 7. 강점

- IL 사전지식을 보존하는 residual waypoint 정식화 — 단순하지만 ablation으로 명확히 검증
- 시스템 엔지니어링 기여(heterogeneous pipeline)가 재현 가능한 수준으로 구체적 (드라이버 버전, GPU 배치, autossh까지)
- 3개 벤치마크(longest6 v2, Bench2Drive, nuScenes) + 2개 백본에서 일관된 개선
- RL 스케일링 법칙을 driving VLA에서 실증 (10M→100M 단조 개선)

## 8. 약점

- 알고리즘적 novelty는 제한적 — residual RL과 PPO는 기존 기법의 조합
- Bench2Drive 개선 폭(SimLingo 대비 DS +1.7)은 longest6 v2 대비 marginal
- 코드/모델 미공개 (Qualcomm 산업 연구), 시드/분산 미보고
- Comfort 지표는 오히려 SimLingo(33.7)보다 하락(25.7) — RL이 급격한 조작 유발 가능성

## 9. 재현성 평가

- 데이터(SimLingo 공개 데이터셋), 백본(InternVL3-1B 공개), 시뮬레이터(CARLA), 보상(CaRL 공개) 모두 접근 가능하나 **코드 미공개**
- 필요 자원: 8×H100 + 32×V100 6일 — 학계 재현은 고비용
- 하이퍼파라미터(lr, batch, anchor m=4, LoRA config)는 비교적 상세히 보고

## 10. 파급 효과 및 후속 연구 방향

- "IL pretrain → closed-loop residual RL" 레시피는 조작(manipulation) VLA에도 이식 가능한 일반 패턴
- 시뮬레이터-learner 분리 pipeline은 CARLA 외 Isaac/ManiSkill 기반 RL에도 적용 가능
- 후속 방향: Sim2Real transfer, LLM unfreeze RL, 언어 reasoning과 residual policy의 결합, metric optimum이 없는 보상 설계

## 11. 세미나 토론 포인트

1. Residual policy의 clipping 범위(y_min/max, v_min/max)가 성능에 미치는 민감도는? (미보고)
2. m=4 anchor는 어떻게 선정되었나 — 더 긴 horizon anchor의 효과는?
3. LLM freeze 상태에서 얻은 개선이 "VLA finetuning"인가 "prior 위 경량 RL head 학습"인가?
4. nuScenes collision rate(0.30)는 UniAD(0.31)과 사실상 동률 — zero-shot 우위 주장은 L2에 국한
5. 100M 샘플 이후에도 스케일링이 지속되는가? (saturation 곡선 미제시)

## 12. 예상 날카로운 질문

| # | 질문 | 답변/코멘트 |
|---|---|---|
| 1 | SR 25%면 여전히 75% 실패인데 SotA 의미는? | 상대적 SotA — 기존 비특권 방법 전원 SR 0%. 절대 성능은 한계로 인정 |
| 2 | Direct control ablation이 불리하게 설계된 것 아닌가? | 동일 config로 학습했다고 명시하나, control 정책용 튜닝은 별도 수행 안 함 |
| 3 | Heterogeneous pipeline의 통신 latency가 on-policy PPO에 미치는 영향은? | 미분석. SSH 터널 기반이라 rollout 수집 지연 가능 |
| 4 | InternVL2→3 교체 효과와 RL 효과의 분리는? | Table 1이 분리 제공: 백본 교체 +5.3 DS, RL +21.5 DS — RL 기여가 압도적 |
| 5 | Bench2Drive에서 HiP-AD(†)와의 공정 비교인가? | HiP-AD는 별도 조건(†) 표기; CLEAR는 비특권 vision-only 유지 |
| 6 | 보상이 route completion 중심인데 comfort 하락은 필연 아닌가? | 그렇다고 볼 수 있음. Comfort 항이 보상에 없음 (soft penalty에 일부만 반영) |

<!-- VERIFIED: pdf -->
