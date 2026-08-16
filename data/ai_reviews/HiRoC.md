# HiRoC: Beyond Flat Policies — Hierarchical Post-Training for Embodied Agents in Robotic Manipulation

> **한 줄 요약**: Qwen2.5-VL-3B 플래너가 전역 지시를 서브골로 분해하고, OpenVLA-OFT 실행기를 서브골 조건부 SFT 후 계층적 GRPO로 온라인 RL 튜닝하여 LIBERO 4개 스위트 평균 93.5%(Long 98.0%)를 달성한 계층적 post-training 프레임워크.

## 1. 배경 및 동기

- 기존 VLA post-training은 대부분 **flat policy** 형태로, 관측 + 전역 지시문을 곧바로 저수준 행동으로 매핑한다 (§Introduction).
- 실제 로봇 태스크는 본질적으로 long-horizon·multi-stage이며, 전역 지시만 조건으로 주면 정책이 "지금 어느 단계인가"를 식별하기 어렵다. 서브골이라는 의미적 앵커가 없으면 미세한 실행 편차가 누적되어 실패로 이어진다.
- 계층적 접근(RT-H, VLA-OS)은 태스크 분해를 도입했지만 오프라인 시연 기반 지도학습에 머물러 온라인 상호작용으로 실행 능력을 개선하지 못한다. 반대로 최근 RL 기법(SimpleVLA-RL, VLA-RL, TGRPO)은 여전히 flat policy를 최적화한다.
- 논문이 제시하는 세 가지 도전 과제: (i) 플래너가 실행 가능한 서브골과 적절한 전이를 생성해야 함, (ii) 전역 지시로 사전학습된 실행기와 세분화된 서브골 사이의 **distribution misalignment(DM)** 로 인한 심각한 cold start, (iii) 궤적 단위 보상만이 아니라 중간 서브골 진척도를 정책 최적화에 반영해야 함.

## 2. 방법론

### 2.1 전체 구조
HiRoC는 **고수준 플래너 + 저수준 실행기** 2모듈로 구성된다. MDP `⟨s, a, P, r, s′, γ⟩`에서 상태공간은 `S = O × V^m`(RGB 이미지 × 토큰 공간), 보상은 성공 1 / 실패 0의 sparse terminal reward. 플래너가 `l_t`를 생성하고 실행기 `π_θ(a_t | o_t, l_t)`가 행동을 낸다.

### 2.2 플래너 SFT (Eq. 1)
- 데이터: **VLA-OS(Gao et al. 2025)** 의 멀티모달 데이터를 LlamaFactory 호환 포맷으로 재구성. 라벨 공간 통일, 모호·중복 항목 제거, 충돌 샘플을 "즉시 다음 서브골 예측" 목적에 맞게 재주석 → `D_plan`.
- 목적함수: 서브골 토큰열에 대한 autoregressive cross-entropy `L_SFT_plan(w_p)`.
- 플래너는 Qwen2.5-VL-3B를 LoRA로 파인튜닝하며, 이후 **실행기 최적화 동안 frozen**.

### 2.3 Distribution Misalignment 완화 SFT (Eq. 2)
- 각 궤적을 서브골 단위 샘플 `(s_i, l_i, a_i)`로 분해(`a_i = ⟨a_1,…,a_m⟩`은 액션 청크) → `D_exe`.
- 실행기가 "전역 지시" 대신 "현재 서브골"에 조건화된 행동열을 예측하도록 사전 SFT하여 cold start를 완화한다.

### 2.4 계층적 GRPO (Eq. 3–12)
- 동일 태스크/초기상태에서 N개 궤적을 샘플링, 궤적 보상 `R_i`를 그룹 내 표준화하여 **task-level advantage** `A_task_i = (R_i − μ_g)/(σ_g + ε)` (Eq. 4–5).
- 롤아웃 중 기록된 서브골 구조로부터 서브태스크 점수 `S_i`를 산출, 동일하게 표준화하여 **subgoal-level advantage** `A_sub_i` (Eq. 6–7).
- 최종 advantage `Ã_i = w_t A_task_i + w_s A_sub_i` (Eq. 8), 가중치는 궤적당 평균 서브골 수 `K`와 궤적 길이 `L`로 적응적 결정: `w_t = (K+L)/L`, `w_s = K/(K+L)` (Eq. 9–10).
- PPO 스타일 clipped actor loss (Eq. 11–12). 액션 토큰 확률비 `ρ_{i,t,j}(θ)`를 재계산하고, 이진 마스크 `m_{i,t,j}`로 유효 액션 토큰만 최적화에 기여시킨다.

### 2.5 플래너를 고정하는 이유 (Remark)
sparse 궤적 보상은 계획/실행 간 credit assignment가 모호하고, 플래너를 계속 갱신하면 서브골 분포가 이동하여 SFT로 확립한 planner–executor 정렬이 깨지므로 RL 단계에서는 실행기만 최적화한다.

## 3. 실험 설정

- 벤치마크: **LIBERO** 의 Goal / Spatial / Object / Long 4개 스위트(각 10 태스크). 일반화 평가는 7종 perturbation을 추가한 **LIBERO-Plus**.
- 플래너 Qwen2.5-VL-3B(LoRA), 실행기 OpenVLA-OFT. 실행기는 스위트별 소량 시연으로 먼저 SFT → 서브골 조건부 SFT → RL 튜닝.
- 하이퍼파라미터(Appendix): GRPO 그룹 크기 8(본문) / 워커당 64 벡터화 환경, 정책 업데이트당 16 롤아웃 에폭, 궤적당 512 env step, 액션 청크 길이 8, lr 2×10⁻⁵, BF16 FSDP. 플래너는 20 policy call마다 재계획. 학습 에폭은 Spatial/Object/Goal 200, LIBERO-10 150.
- 평가: 태스크당 50 에피소드, 에피소드 최대 512 step, 8개 결정론적 시드(GPU 워커당 1개). 전 실험 8×NVIDIA H200.

## 4. 주요 결과

**LIBERO (Table 1, success rate %)**

| Method | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 53.7 | 79.2 | 76.5 |
| OpenVLA*-Full | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| SmolVLA | 93.0 | 94.0 | 77.0 | 91.0 | 88.8 |
| ThinkAct | 88.3 | 91.4 | 70.9 | 87.1 | 84.4 |
| TGRPO | 90.4 | 92.2 | 81.0 | 59.2 | 80.7 |
| MolmoAct | 87.0 | 95.4 | 77.2 | 87.6 | 86.6 |
| World-Env | 87.6 | 86.6 | 57.8 | 86.4 | 79.6 |
| GRAPE | 88.5 | 92.1 | 57.2 | 83.1 | 80.2 |
| VAL-OS-A-S | 87.0 | 96.5 | 92.7 | 66.0 | 85.6 |
| VLA-RL | 90.2 | 91.8 | 59.8 | 82.2 | 81.0 |
| **HiRoC (Ours)** | **95.6** | 96.0 | 84.4 | **98.0** | **93.5** |

- 초록 기준 평균 **10.06%** 향상. Long 98.0%가 가장 두드러지며, 논문은 Long/Spatial에서 이득이 큰 이유를 "서브골 분해가 더 유리한 반면 Object/Goal은 시각 그라운딩·목표 이해 의존도가 높기 때문"이라 설명. 실제로 Goal(84.4)은 VAL-OS-A-S(92.7)·OpenVLA*-Full(90.6)보다 낮다.
- **LIBERO-Plus (Figure 3)**: OpenVLA, OpenVAL*-One, WorldVLA와 비교해 모든 perturbation 유형에서 최고 일반화 성능이라고 보고(수치는 그림으로만 제시).
- **Ablation on Object (Table 3)**: w/o local GRPO 95.20% / w/o global GRPO **4%** / w/o planner 92.60% / HiRoC **96%**. global GRPO가 주 역할을 하고 local GRPO가 보조적으로 개선, 플래너 제거 시 성능 저하 + 학습 곡선의 심한 진동.
- **플래너 5-fold 교차검증 (Table 2, Subgoal / Episode / Avg.)**: HiRoC-Planner는 Short 94.92 / 91.55 / 93.24, Medium 95.20 / 84.01 / 89.61, Long 95.09 / 80.40 / 87.75, Mix 95.07 / 85.32 / 90.20. 비교 대상 RoboBrain2-3B와 RoboBrain2-7B는 Episode 지표가 전 구간 0.00으로 사실상 실패(예: RoboBrain2-7B Long 0.98 / 0.00 / 0.49).
- **Training scheme 분석 (Figure 5 좌)**: 두 GRPO의 손실을 직접 합산하는 방식보다 advantage 수준에서 결합하는 방식(HiRoC 채택)이 Object에서 우수.
- **DM 분석 (Figure 4)**: DM 학습 없이는 초기 성공률이 크게 낮고 이후 RL로도 최종 성능이 만족스럽지 않음.
- **실세계 (Figure 7, Appendix)**: JoySim에서 JoyRA-0.1을 베이스 VLA로 사용, 플래너는 Qwen2.5-VL-3B SFT, 실행기는 Flow-SDE 기반 RL로 시뮬레이션에서만 학습 후 추가 파인튜닝 없이 실로봇에 zero-shot 배포. "수정액을 상자 2층에 넣기" 태스크에서 approach–grasp–transport–place 성공(정량 수치 미제시).

## 5. 강점

- 계층 구조를 "SFT로만" 쓰지 않고 온라인 RL과 결합한 점이 명확한 차별점. TGRPO의 step-level advantage와 달리 **의미 있는 서브골 세그먼트**를 advantage 신호로 사용한다.
- DM 문제를 명시적으로 진단하고 서브골 조건부 SFT라는 저비용 해법을 제시, ablation(w/o G = 4%)으로 각 구성요소의 역할을 분리해 보였다.
- 가중치 `w_t`, `w_s`를 서브골 개수와 궤적 길이로 적응적으로 정한 점은 태스크별 서브골 수 편차를 자동 보정한다.

## 6. 한계 및 미해결 문제

1. **Goal 스위트 열세**: 84.4%로 여러 베이스라인 대비 낮다. 서브골 분해가 만능이 아니며, 목표 이해 중심 태스크에서는 오히려 추가 조건이 노이즈일 수 있다.
2. **플래너 frozen**: 저자 스스로 credit assignment 문제로 플래너를 고정했다고 밝히며, end-to-end 학습을 future work로 남겼다. 플래너 오류는 실행기가 교정할 수 없다.
3. **LIBERO-Plus 수치 부재**: 일반화 주장을 뒷받침하는 값이 Figure 3에만 있어 정량 재현·비교가 어렵다.
4. **실세계 실험의 빈약함**: 단일 태스크, 정량 성공률 없음. 게다가 시뮬레이션 학습이 JoyRA-0.1 + Flow-SDE로 LIBERO 파이프라인(OpenVLA-OFT + 토큰 GRPO)과 다른 구성이라 주 기여와의 연결이 느슨하다.
5. **플래너 학습 데이터 의존성**: VLA-OS 주석에 의존하므로 서브골 라벨이 없는 새 도메인으로의 확장 비용이 명시되지 않았다.
6. **비교 공정성**: 베이스라인 수치의 출처·평가 조건이 본문에 정리되어 있지 않고(Appendix 참조), 표의 Rank 열 해석도 다소 모호하다.

## 7. 재현성

- 코드/체크포인트 공개 언급 없음. 하이퍼파라미터는 Appendix에 비교적 상세(lr, 청크 길이, 롤아웃 에폭, 시드 수)하나, 플래너 데이터 정제 규칙과 서브태스크 점수 `S_i`의 구체적 산출식이 본문에 명시되지 않아 계층적 advantage 재현에 불확실성이 있다.

## 8. 관련 연구와의 위치

- **vs. SimpleVLA-RL / VLA-RL**: 동일하게 outcome-reward GRPO를 쓰지만 flat policy. HiRoC는 서브골 계층을 추가.
- **vs. TGRPO**: TGRPO는 trajectory + step 이중 advantage. HiRoC는 step 대신 **의미 단위 서브골**을 사용.
- **vs. VLA-OS / RT-H**: 계층 분해는 공유하나 지도학습에 머무름. HiRoC는 여기에 온라인 RL을 얹었고, 데이터도 VLA-OS에서 재사용.
- **vs. World-Env / GRAPE**: 환경 동역학·보상 모델링 계열. 논문은 이들이 태스크를 순차 서브골로 명시 분해하지 못한다고 주장.

## 9. 실무적 시사점

- 이미 OpenVLA-OFT류 SFT 체크포인트가 있다면, (a) 서브골 주석 데이터 구축 → (b) 서브골 조건부 SFT → (c) 계층 GRPO 순의 증분 파이프라인으로 적용 가능하다.
- long-horizon 태스크 비중이 높은 워크로드일수록 이득이 크고, 단일 스테이지 pick-place 위주라면 추가 복잡도 대비 이득이 작을 수 있다.
- 20 policy call마다 재계획하는 설계는 3B VLM 추론이 매 스텝 들어가지 않게 해 실시간성 부담을 낮춘다.

## 10. 총평

- **Novelty: ★★★☆☆** — 계층 분해와 GRPO 각각은 기존 요소이나, 둘의 결합 지점(DM 완화 SFT + advantage 수준 융합)을 체계적으로 다룬 점이 기여.
- **Rigor: ★★★☆☆** — LIBERO 정량 결과와 ablation은 설득력 있으나 LIBERO-Plus·실세계는 정성 수준.
- **Practical impact: ★★★★☆** — 기존 VLA 체크포인트 위에 얹는 post-training 레시피로 재사용성이 높다.

"평면 정책 + RL"이 지배적인 현 흐름에서 계층 구조를 RL과 어떻게 결합해야 하는지에 대한 실용적 레시피를 제시했다. 다만 플래너를 얼려둔 채로는 계층화의 상한이 플래너 품질에 묶이므로, 저자들이 예고한 end-to-end 학습이 실제 후속 기여가 될 것이다.

## 11. 예상 질문

| Q | A |
|---|---|
| 왜 플래너를 RL로 함께 학습하지 않았나? | sparse 궤적 보상 하에서 계획/실행 간 credit assignment가 모호해 플래너 최적화가 불안정하고, 플래너 갱신이 서브골 분포를 이동시켜 SFT로 확립한 정렬을 깨기 때문(Remark). |
| local GRPO만으로는 안 되나? | Table 3에서 w/o G(global 제거)는 4%로 붕괴. 최종 성공 신호가 주 학습 동력이고 서브골 신호는 보조적. |
| 서브골 조건부 SFT가 왜 필요한가? | 실행기는 전역 목표로 사전학습되어 세분화된 서브골 분포와 어긋난다(DM). Figure 4에서 DM 학습 없으면 초기 성공률이 크게 낮고 최종 성능도 열위. |
| 두 advantage를 손실 합산이 아니라 advantage로 합치는 이유는? | 전체 궤적과 서브태스크 궤적이 서로 다른 최적화 목적을 유도해 손실 직접 합산은 suboptimal 수렴 위험이 있고, Eq. 9의 성분별 가중이 이 불일치를 완화(Figure 5 좌). |
| Goal 스위트에서 왜 상대적으로 약한가? | Goal/Object는 시각 그라운딩과 목표 이해 의존도가 높아 서브골 분해의 이득이 작다고 논문이 설명. |

## 12. 핵심 수치 요약

- LIBERO 평균 **93.5%** (Spatial 95.6 / Object 96.0 / Goal 84.4 / Long 98.0), 평균 향상 **10.06%**.
- Object ablation: full 96% vs w/o global GRPO 4% vs w/o planner 92.60% vs w/o local GRPO 95.20%.
- 플래너 5-fold 교차검증 Mix 평균 **90.20** (RoboBrain2-7B 0.33).
- 학습 자원: 8×H200, lr 2×10⁻⁵, BF16 FSDP, 액션 청크 8, 궤적 512 step, 평가 태스크당 50 에피소드.

<!-- VERIFIED: pdf -->
