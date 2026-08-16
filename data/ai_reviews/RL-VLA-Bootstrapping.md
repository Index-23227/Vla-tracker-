# RL-VLA-Bootstrapping: RL-Only Bootstrapping of OpenVLA-OFT for a Novel Cable-Driven Robot Embodiment

- **arXiv**: 2608.01013v1 (2026-08-02, cs.RO)
- **저자/소속**: Damir Nurtdinov, Alexei Kornaev, Alexander Maloletov — Innopolis University, Volgograd State Technical University
- **백본**: OpenVLA-OFT (public `openvla/openvla-7b` 체크포인트 초기화)
- **벤치마크**: 자체 MuJoCo CDPR 시뮬레이션 (표준 벤치마크 없음)

> **한 줄 요약**: 임베디먼트별 시연 데이터를 **한 건도 쓰지 않고**, 시뮬레이터 기하 정보 기반 dense reward만으로 PPO→GRPO 2단계 RL을 돌려 케이블 구동 병렬 로봇(CDPR)에 OpenVLA-OFT를 정렬시킨 연구. 4방향 지시 평균 성공률 34.25% → 53.50%.

---

## 1. 배경 및 동기

기존 VLA 적응 레시피는 **임베디먼트별 시연 데이터**를 전제한다. OpenVLA는 새 Franka 셋업마다 태스크당 10–150 시연으로 LoRA 파인튜닝(전체 파라미터의 1.4%)하고, OpenVLA-OFT는 필터링된 성공 시연을 요구한다. 그러나 형태학적으로 완전히 새로운 커스텀 로봇에는 그 데이터셋 자체가 존재하지 않는다.

저자들의 질문은 명확하다: **"태스크 데이터셋이 아직 없을 때 연구자는 무엇을 해야 하는가?"**

---

## 2. 문제 정의

- **플랫폼**: 케이블 구동 병렬 로봇(cable-driven parallel robot, CDPR) + 최소 그리퍼
- **제어 인터페이스**: 5차원 — Cartesian x, y, z + yaw 회전 + 그리퍼 작동
- **전제**: 임베디먼트별 데이터셋 **전무** (zero-demo)
- **목표**: 지도학습 파인튜닝이 아니라 **RL만으로** 언어–비전–행동 매핑을 최초로 성립시키기

이는 "태스크 전이"가 아니라 **임베디먼트 호환 매핑의 최초 확립** 문제에 가깝다.

---

## 3. 선행 VLA-RL 연구와의 차별점 (Sec. II)

| 연구 | 시작점 | 보상 |
|------|--------|------|
| iRe-VLA [8] | RL과 지도학습 교대 | — |
| RIPT-VLA [7] | 최소 1개 이상의 시연 필요 | sparse binary success |
| SimpleVLA-RL [5,6] | 지도 초기화 존재 | binary 0/1, 최소 보상 엔지니어링 |
| **본 연구** | **시연 0건** | **dense 기하 보상** |

핵심 차이: 새 임베디먼트에서는 초기 성공 궤적이 너무 희소해 sparse reward 학습이 성립하지 않는다. 따라서 shaped signal이 **선택이 아니라 필수**라는 논리.

---

## 4. 학습 스택 (Sec. III)

단일 스택에 통합된 구성 요소:
- CDPR에 대한 **MuJoCo 임베디먼트 명세** 및 컨트롤러 래퍼 (저수준 실행은 시뮬레이션 내 PID)
- **YCB [11] + LIBERO [12] 에셋** 기반 랜덤 장면 생성
- RL 경로와 다운스트림 정책 실행 경로를 아우르는 **공유 action codec**
- OpenVLA-OFT 기반 PPO / GRPO train·eval 스크립트

정책 입력은 overview 카메라 + wrist 카메라 **2개 RGB**로, OpenVLA-OFT 멀티모달 인터페이스와 일치.

---

## 5. Stage 1: Directional PPO (Sec. IV-A)

4개 원시 방향 지시로 시작: `move left`, `move right`, `move forward`, `move backward`.

목적은 **객체 조건부 행동을 도입하기 전에** 지시 의미론이 CDPR 행동 공간으로 어떻게 사상되는지를 정책에 가르치는 것.

---

## 6. Stage 2: GRPO Continuation (Sec. IV-B)

PPO 체크포인트에서 이어받아 GRPO로 계속 학습하며 지시 공간을 확장:

```
move to <object>
```

대상 객체 8종: **apple, baseball, bowl, cup, mug, peach, pear, plate**.

이 단계는 Stage 1의 이득을 보존하면서 방향 grounding을 넘어설 수 있는지를 검증한다.

---

## 7. Dense Reward 설계 (Sec. IV-C)

논문의 핵심 알고리즘적 선택. d_t는 엔드이펙터와 지시 의존 타깃(방향 영역 또는 객체) 사이 현재 거리, a_t는 정책 행동:

```
r_t = w_p (d_{t-1} - d_t) + b_s · I[succ_t] - w_a · P(a_t)      (Eq. 1)
```

- `w_p (d_{t-1} - d_t)`: 타깃으로의 **진전(progress)** 보상
- `b_s · I[succ_t]`: 성공 보너스
- `- w_a P(a_t)`: 그리퍼 외 행동의 **포화(saturation) 근접 패널티** → 불안정 제어 억제

> ❓ **예상 질문**: 가중치 w_p, b_s, w_a의 구체적 값은?
> **답변**: 논문 본문(4페이지 워크숍 포맷)에 수치가 명시되지 않음. 재현은 공개 저장소에 의존해야 한다.

---

## 8. 학습 프로토콜 (Sec. IV-D)

- 백본: OpenVLA-OFT, `openvla/openvla-7b` 공개 체크포인트에서 초기화
- 학습 대상: **어댑터 + action-head 파라미터** (2개 이미지 입력, 8-step action chunk)
- Stage 1 PPO: **약 175시간**, NVIDIA A40 × 2
- Stage 2 GRPO: **약 170시간**, 동일 하드웨어
- 총 RL 예산: **약 345시간**, 임베디먼트 시연 **0건**

비교 대상: OpenVLA 사전학습은 A100 64장 × 15일, OpenVLA-OFT 공식 레시피는 A100/H100 8장 × 50K–150K steps.

---

## 9. 평가 프로토콜 및 주요 결과 (Sec. V–VI, Table I)

held-out 랜덤 장면에서 저장소 validator로 평가. 방향 지시는 **각 100 롤아웃**, 객체 조건부는 **총 400 롤아웃**.

| Instruction | PPO (%) | PPO→GRPO (%) | Δ (pp) |
|-------------|---------|--------------|--------|
| Move left | 17.00 | 52.00 | +35.00 |
| Move right | 43.00 | 52.00 | +9.00 |
| Move forward | 62.00 | 62.00 | +0.00 |
| Move backward | 15.00 | 48.00 | +33.00 |
| **Mean (4 directions)** | **34.25** | **53.50** | **+19.25** |

가장 큰 이득은 `move left`(+35pp)와 `move backward`(+33pp). `move forward`는 62%로 **유지**되어, 계속 학습이 최고 성능 지시를 붕괴시키지 않았음을 보인다.

---

## 10. 객체 조건부 결과 (Sec. VI-B, Fig. 1)

`move to <object>` 8종 평가 결과: **39/400 = 9.75%** strict success.

수치 자체는 낮지만, 다수의 검증 에피소드에서 정책이 **올바른 객체로 접근한 뒤 롤아웃 후반의 불안정성**으로 실패한다. Fig. 1(bowl, plate, baseball, mug, cup, peach, pear, apple 대표 롤아웃)은 실패 모드가 "객체 grounding 부재"가 아니라 **late-stage instability**임을 시사한다.

SimpleVLA-RL [5]에서 보고된 것과 유사한 **"pushcut" 스타일 지름길 행동**도 관찰되며, 저자들은 이를 보상 주도 적응이 실제로 일어났다는 증거이자 목표 근방 보상 설계 개선의 동기로 해석한다.

---

## 11. 한계 및 비판적 검토

- **시뮬레이션 전용**: 실로봇 전이 결과 없음. CDPR의 케이블 동역학은 sim-to-real 격차가 특히 클 수 있는 영역이다.
- **표준 벤치마크 부재**: LIBERO/CALVIN/SimplerEnv 수치가 전혀 없어 리더보드 직접 비교 불가. 저자들도 OpenVLA/OFT 수치와 직접 비교 불가함을 명시한다.
- **강건한 조작 미달성**: 4방향 53.50%, 객체 조건부 9.75%는 실사용 수준과 거리가 있다. 저자들 스스로 "does not yet establish robust manipulation"이라고 못박는다.
- **태스크 난이도의 비대칭**: `move forward`가 PPO 단계부터 62%로 이미 높았고 GRPO로 전혀 개선되지 않은 점은, 특정 방향이 CDPR 기구학상 구조적으로 쉬웠을 가능성을 시사한다.
- **보상 하이퍼파라미터 미공개**: Eq. 1의 계수가 본문에 없어 dense reward의 민감도 분석이 불가능하다.
- **ablation 부재**: dense reward vs sparse reward를 같은 환경에서 직접 비교한 실험이 없어, "dense가 필수"라는 핵심 주장이 논증에 머문다.
- **코드 익명화 링크**: `anonymous.4open.science` 링크만 제공되어 영구 접근성이 불확실하다.

---

## 12. 총평 및 VLA-Tracker에서의 위치

**VLA-Tracker 등재 판정: ACCEPTED (자체 정책 산출물 있음)**

이 논문은 frozen 정책 래퍼나 순수 프레임워크가 아니다. OpenVLA-OFT의 **어댑터와 action-head 파라미터를 PPO·GRPO로 실제 갱신**하여 새 체크포인트를 산출하며, 그 체크포인트의 성능을 held-out 롤아웃으로 측정한다. 따라서 등재 기준(정책 가중치를 학습·갱신하는 자체 산출물)을 충족한다.

기여의 성격은 SOTA 경신이 아니라 **방법론적 존재 증명**이다. "시연 데이터가 0건일 때 dense simulator reward만으로 언어 조건부 컨트롤러의 첫 버전을 만들 수 있는가"에 대해 34.25% → 53.50%라는 비자명한 답을 제시한다. 저자들이 제안하는 단계적 방법론 — (1) RL로 최초의 임베디먼트 정렬 컨트롤러 확보, (2) 그것으로 다운스트림 데이터 수집 비용 절감 또는 모방학습 초기화, (3) 더 풍부한 지시·엄격한 평가·실로봇 전이로 확장 — 은 커스텀 로봇 연구자에게 실용적 가치가 있다.

다만 리더보드 관점에서는 표준 벤치마크 수치가 없어 다른 모델과의 정렬 비교가 불가능하며, 4페이지 워크숍 분량의 제한으로 ablation과 하이퍼파라미터가 부족하다. **"새 임베디먼트 zero-demo RL 부트스트랩"이라는 방법론 카테고리의 참조점**으로 추적하는 것이 적절하다.

---

<!-- VERIFIED: pdf -->
