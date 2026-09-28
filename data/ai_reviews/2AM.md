# 2AM: Grounding Agent-Side Memory as Guidance for Steerable Action Models in Long-Horizon Manipulation

> **한 줄 요약**: Task memory는 전적으로 멀티모달 Agent(Qwen3.8-27B)가 보유하고, 에피소드 상태가 없는 RGB 기반 Action Model(Qwen3-VL-4B + flow-matching expert)이 모든 동작을 수행하도록 분리한 뒤, 둘 사이를 "subtask 언어 + 선택적 2D grasp/place/move 힌트"라는 compositional steering contract로 연결. LIBERO-Mem에서 completion 76.29%, relaxed SR 63.00%, strict SR 11.83%(최고 공개 baseline completion 14.8%, 정렬된 π0 재현 70.79%).

---

## 1. 배경 및 동기

- Long-horizon 조작은 memory가 필요하지만, 그것이 반드시 action policy 내부에 있어야 하는가?
- VLA 내부 memory(MemoryVLA, SlotSSM, Mem-0)는 실패 궤적을 latent에 누적해 retry를 오염시킬 수 있음.
- Agentic toolbox 계열(Hi Robot, RoboClaw, Goal2Skill, Harness VLA 등)은 depth·보정 geometry·planner 기반 motion 등 여러 경로를 섞어 성능 귀속(attribution)이 불분명. 언어만으로 VLA를 호출하면 인스턴스·목적지·동작이 누락되어 VLA 능력이 과소평가됨.
- 2AM은 반대 설계점을 택함: **센싱 제한(RGB-only), 도구 폭 축소, 인터페이스 대역폭 확대**.

## 2. 방법론 심층 분석

### 2.1 Agent / Action Model 분리
- Agent: 지시문, 현재 agent-view RGB, consolidated history H_k로부터 command c_k 생성(Eq. 1).
- Action Model: 현재 agent-view·wrist RGB, robot state, c_k만으로 L-step 절대 EE action chunk 예측(Eq. 2). H_k에 직접 접근하지 않고 Agent 호출 간 지속 상태 없음(episodically stateless).
- Depth, mask, object pose, 2D→3D backprojection, 객체별 analytic motion은 어느 모델에도 들어가지 않음.

### 2.2 Compositional steering contract (Table 1)
- c_k = (언어 ℓ_k, q_grasp, q_place, q_move), 좌표는 현재 agent-view 기준 [0,1000]² 정규화, 없는 필드는 생략.
- Grasp: 다음 대상 인스턴스 binding(접촉 전). Place: 운반 중 목적지 유지. Move: 수 step 뒤 유용한 gripper 위치(웨이포인트가 아닌 방향 cue).
- 세 필드는 서로 다른 시간 스케일을 담당하며, 하나가 누락/오류여도 전체 명령이 무효화되지 않음.

### 2.3 Teacher 없는 supervision 복원
- 데모의 stage 라벨, object box/mask, gripper contact, object motion, 투영된 EE 궤적으로 subtask·grasp·place·move 라벨을 결정적으로 생성(VLM narration 불사용). Privileged annotation은 학습 타깃 생성에만 사용.
- Grasp/place = box 중심, move = t+Δ(Δ~U{10..20}) 시점의 투영 EE 위치.

### 2.4 불완전한 steering에 대한 강건 학습
- Constrained hint dropout: 가용 필드 내에서 독립 마스킹하되 최소 하나는 유지 → 언어만/점만/혼합 명령 생성.
- 좌표에 isotropic Gaussian noise, move 힌트의 future-window 샘플링 → 누락·위치 오차·시간 불일치 세 가지 인터페이스 오류를 모사.

## 3. 데이터 전략

- LIBERO-Mem 데모에 구조화된 힌트 라벨을 추가. 서브태스크 경계를 넘는 window도 허용해 grasp·lift·release 근처 연속 동작을 보존(Agent를 기다리며 멈추는 습관 방지).

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Action Model | Qwen3-VL-4B backbone + flow-matching Action Expert |
| Action | 16-step chunk, 다음 프레임 절대 EE pose + gripper |
| 입력 | agent-view + wrist RGB, robot state, steering command |
| Agent | Qwen3.8-27B (학습하지 않음), action chunk마다 재관측·재조향 |

## 5. 실험 설계 및 평가 프로토콜

- LIBERO-Mem 10 태스크: lift-and-return 1/3/5/7회 반복, 두 bowl 교환, 세 bowl 회전, 이전 basket 점유 상태 조건 배치 등(유형 M/S/R/O).
- 지표: strict SR(순서·반복 수 정확, 초과 수행 시 실패), relaxed SR(목표 도달 후 overshoot 무시), completion(순서 subgoal 달성 비율).
- 비교: 벤치 논문의 π0 / SlotVLA / SlotSSM(모두 strict 0, completion만 보고), 동일 backbone의 **in-house π0 재현**, language-only ablation(동일 Agent·Action Model, 2D 힌트 제거).

## 6. 실험 결과 심층 분석 (PDF Table 2, 3 직접 인용)

| Method | Strict SR | Relaxed SR | Completion |
|---|---|---|---|
| Best published (SlotSSM) | 0.00 | – | 14.80 |
| π0 reproduced | 12.25 | 37.42 | 70.79 |
| Language-only subtask | 7.25 | 19.42 | 53.72 |
| **2AM (Ours)** | 11.83 | **63.00** | **76.29** |

- 정렬된 π0 재현 대비 completion +5.50 pp, relaxed SR +25.58 pp, strict SR은 −0.42 pp로 비슷(저자도 균일한 SR 향상은 아니라고 명시).
- 태스크별(Table 2): 2AM이 10개 모두에서 relaxed SR 우위, completion은 7개에서 우위. T1–T6(반복 계열)은 π0의 strict SR이 높고, T7–T10(관계·가림)은 2AM이 우세(예: T9 strict 47.50 vs 0.00).
- 2AM의 63.0% relaxed vs 11.8% strict 격차 → "의도 상태 도달"과 "정확히 멈추기(semantic termination)"는 별개 문제.

## 7. Ablation 분석

- Language-only 대비 2D 힌트 추가 효과: strict +4.58, relaxed +43.58, completion +22.57 pp → 인터페이스 대역폭이 기억된 의도를 행동으로 옮기는 데 핵심.
- 실패 사례(Fig. 3): grasp 위치 오차, 잘못된 subtask 선택, 잘못된 basket binding — dropout·noise 학습은 누락/부정확한 힌트엔 강하지만 잘못된 고수준 의도는 교정 불가.
- 개별 힌트(grasp/place/move)별 기여는 아직 분리하지 않음.

## 8. 관련 연구 비교

- MemER: 고수준 정책이 keyframe을 검색해 텍스트 지시로 π0.5 하위 정책을 호출 — 2AM은 텍스트 외 2D 힌트를 추가한 "누락된 인터페이스"를 연구.
- Steerable Policies: subtask·atomic motion·point·trace 명령으로 VLA를 학습 — 2AM은 독립적으로 가용한 compositional 필드와 온라인 누락/노이즈 내성에 초점.
- HAMSTER(2D path), RT-H(language motion): 중간 표현의 중요성을 공유.
- MemoryVLA/SlotSSM/Mem-0: memory를 정책 내부에 배치하는 대조군.

## 9. 한계 및 미해결 문제

- 단일 시뮬레이션 벤치(LIBERO-Mem)만 평가, 실로봇 미검증(저자 명시).
- 2D 인터페이스로는 가려진 geometry, 힘, 연속 motion history를 표현할 수 없음.
- Strict SR은 π0 재현과 비슷 — 정확한 종료 판단은 미해결.
- 더 강한 Agent backbone, steering 빈도에 따른 성능-지연 trade-off 미측정. 코드 공개 언급 없음.

## 10. 총평

"memory는 Agent에, 동작은 stateless Action Model에"라는 깔끔한 분업을 통제된 조건(RGB-only, 대체 motor 경로 제거)에서 검증해 성능 귀속을 명확히 한 점이 좋다. 특히 강한 π0 재현을 정직하게 제시하고 strict SR 개선이 없음을 인정한 점은 신뢰도를 높인다. 반면 결과는 한 벤치에 국한되어 일반적 long-horizon 능력의 증거로 보기는 어렵다.

## 11. 🔥 예상 날카로운 질문 모음

1. grasp/place/move 각 힌트의 개별 기여는? Move 힌트만으로도 대부분의 이득이 나오는가?
2. Agent가 반복 횟수를 알고 있는데 왜 strict SR(종료)이 개선되지 않는가? "DONE" 명령을 contract에 추가하면?
3. 27B Agent의 호출 지연이 실로봇 제어 주기와 양립 가능한가?
4. RMBench, RoboMME 등 다른 memory 벤치에서도 같은 인터페이스가 유효한가?
5. Action Model이 힌트 없이 학습된 π0 재현과 동일 데이터로 학습된 것인가(데이터 양 차이)?

## 12. 재현성 및 후속 연구 제안

- 코드 공개 미언급. 라벨 복원 규칙(Eq. 4–7)은 명확히 기술되어 재구현 가능성은 높음.
- 후속(저자 계획 포함): (1) LIBERO-Mem·RMBench·RoboMemArena·RoboMME 통합 구현, (2) RGB-only 실로봇 closed-loop 검증, (3) 종료 판단을 위한 contract 확장, (4) 3D/force 힌트로 인터페이스 확장.

<!-- VERIFIED: pdf -->
