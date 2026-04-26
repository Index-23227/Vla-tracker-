# DMW: Drive My Way — Preference Alignment of Vision-Language-Action Model for Personalized Driving

> **한 줄 요약**: SimLingo(InternVL2-1B) backbone에 사용자 임베딩과 GRPO 기반 강화 미세조정을 결합한 개인화 자율주행 VLA. 30명 운전자의 PDD 데이터셋으로 학습하고 LLM이 생성한 style-aware reward 가중치로 보정하여, Bench2Drive closed-loop에서 SimLingo 대비 일관된 스타일 적응을 달성한 CVPR 2026 논문.

---

## 1. 배경 및 동기

- 기존 end-to-end 자율주행은 일반화된 안전·효율 목표만 최적화 → 운전자 개개인의 장기 습관(공격성, 보수성)과 단기 자연어 의도("I'm in a rush")를 동시에 반영하지 못함.
- 사전 정의 모드(sport/comfort/eco)나 LLM 기반 명령 해석(Talk2Drive)은 단순 시나리오에만 한정되며 장기 선호 미반영.
- 핵심 질문: VLA가 **장기 사용자 임베딩 + 단기 자연어 명령**을 동시에 받아 안전·효율을 유지하면서 스타일 변별이 가능한가?

---

## 2. 방법론 심층 분석

### 2.1 VLA Backbone
- **SimLingo** (InternVL2-1B 기반: InternViT-300M + Qwen2-0.5B)을 채택. 카메라·내비게이션·언어 입력으로 waypoint를 예측하고 PID로 throttle/brake/steer 변환.

### 2.2 사용자 임베딩 학습
- 장기 선호 인코더 fp(·): DeBERTaV3 텍스트 처리 + projection head.
- 경로 처리기 fb(·): multi-head self-attention (이미지·ego state·action 시퀀스 입력).
- **InfoNCE contrastive learning** (식 1)으로 같은 운전자의 profile-behavior 임베딩을 가깝게, 다른 운전자는 멀게 정렬.

### 2.3 Preference Alignment via GRPO
- Group Relative Policy Optimization으로 정책 미세조정. 각 입력당 4개 응답을 샘플링하여 normalized group advantage 계산.
- **Residual decoder**: vision/language/target/user-embedding/motion/residual query token을 LM에 통합 → MLP + categorical head로 (속도 변화, 조향 변화) 두 잔차 출력. PID 베이스 액션과 합산.
- 행동 다양성 보강: 가장 비유사한 운전자 u의 action 통계로 augmented action ãᵐ_t = (āᵐ/āᵘ)·aᵐ_t 생성하여 reward 다변화.

### 2.4 Style-Aware Reward Adaptation
- R = wₛ·R_safety + wₑ·R_efficiency + w_c·R_comfort (식 2).
- **GPT-5가 시나리오·명령으로부터 (wₛ, wₑ, w_c, β_safety, v_pref, β_lat, β_long)을 추론**하고 expert review로 보정.
- TTC 기반 안전, |v−v_pref| 기반 효율, 가속도 임계 기반 안락.

📌 [Figure 5] LLM이 "Let's be patient" 같은 명령을 (Safety 0.5 / Efficiency 0.3 / Comfort 0.2) 가중치로 매핑하는 예시.

---

## 3. 실험 결과

### 3.1 Bench2Drive Closed-loop (Table 1)
| Method | Style | DS | SR | Efficiency | Comfort |
|--------|-------|----|----|-----------|---------|
| SimLingo | Aggressive | 78.56 | 65.83 | 247.60 | 18.61 |
| SimLingo | Conservative | 78.18 | 65.56 | 238.77 | 26.99 |
| StyleDrive | Aggressive | 75.68 | 60.89 | 256.71 | 16.79 |
| **DMW-Vanilla** | Aggressive | **82.19** | **70.97** | 253.10 | 15.86 |
| **DMW** | Aggressive | 79.50 | 67.36 | **281.56** | 21.62 |
| **DMW** | Conservative | **82.72** | **71.56** | 237.06 | **34.62** |

- DMW의 Aggressive 명령 시 efficiency +18.77% (vs SimLingo +3.70%, StyleDrive +6.00%) — 스타일 민감도 대폭 향상.
- Conservative DS(82.72)가 SimLingo Conservative(78.18)를 능가 → 안전성 손실 없이 개인화.

### 3.2 Long-term Alignment (Table 3)
- ID 운전자(D1, D2): DMW Alignment Score = 0.92 (vs MORL-PD 0.42–0.58).
- OOD(D3, D4): 0.83 (vs MORL-PD 0.25–0.33). User study 평점도 7.8–8.7 (MORL-PD 3.5–6.2).

### 3.3 Ablation (Table 4)
- AAP(Adaptive Average Pooling) 제거 시 DS 80–87, AS 0.25–0.67로 큰 폭 하락 → 임베딩 표현력의 중요성.

---

## 4. 한계

1. **CARLA 시뮬레이터 한정**: PDD가 Logitech G-series로 수집된 CARLA 데이터로, 실제 차량 검증 부재(저자도 future work로 언급).
2. **운전자 풀 30명**: 인구통계·문화 다양성 제한, 25명 ID/5명 OOD.
3. **GPT-5 의존**: reward 가중치 생성에 외부 LLM + expert review 파이프라인 필요 → 재현·결정성 제약.
4. **Bench2Drive만 평가**: nuPlan, NAVSIM 등 다른 폐루프 벤치마크 미사용.
5. **Comfort metric 해석**: Aggressive에서 comfort 21.62로 SimLingo Aggressive(18.61)보다 *높음* → 일관된 의미 부여 어려움.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — VLA + RL preference alignment의 결합 |
| Technical depth | ★★★★☆ — 체계적 두 단계(임베딩+GRPO) |
| Experimental rigor | ★★★☆☆ — Bench2Drive 단일 + user study |
| Practical impact | ★★★★☆ — 인간 중심 자율주행 방향 제시 |
| Writing | ★★★★☆ |

**강점**: 장단기 선호의 분리, LLM 기반 동적 reward 가중치, ID/OOD 분리 평가. **약점**: 실차 미검증, LLM 의존, 단일 시뮬레이터.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | GRPO가 PPO 대비 장점은? | Group normalized advantage로 sparse reward 환경에서 안정. SimLingo backbone에서 LoRA 미세조정 가능. |
| 2 | OOD AS 0.83은 5명 기준 — 통계적 유의성? | n=5는 작음. 하지만 MORL-PD 0.25–0.33과 격차가 커 효과 크기는 명확. |
| 3 | LLM이 reward 가중치를 "잘못" 줬을 때 안전 가드는? | Expert review 단계 + style별 사전 정의 upper bound로 클리핑. 그래도 LLM 환각에 취약. |
| 4 | SimLingo 대신 더 큰 backbone 쓰면? | 미검증. InternVL2-1B 선택은 closed-loop latency 고려한 것으로 보이며, scaling 분석 없음. |
| 5 | Comfort가 Aggressive에서 더 높게 나오는 현상은? | Comfort는 binary indicator 합산이므로 임계치 완화 시 더 자주 만족. 명목적 metric 한계. |

<!-- VERIFIED: pdf -->
