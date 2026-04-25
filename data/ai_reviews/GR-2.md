# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

> **한 줄 요약**: GR-1의 후속작으로, **38M 인터넷 비디오·~50B 토큰**으로 video generative pre-training 후 약 4만 robot trajectory로 fine-tune하여 **105개 multi-task에서 평균 97.7% SR**, end-to-end bin-picking에서 평균 79.0%, CALVIN ABCD-D에서 avg-len 4.64로 SOTA를 달성한 GPT-style VLA.

---

## 1. 배경 및 동기

- 대규모 robot trajectory 수집의 한계 → 인터넷 비디오를 통한 pre-training이 manipulation generalization의 열쇠라는 가설(Section 1).
- 핵심 목표: π(l, o_{t-h:t}, s_{t-h:t}) → o_{t+1}, a_{t:t+k} (Eq. 1–2). 단일 step action이 아닌 **action chunk**를 cVAE로 생성해 trajectory smoothness/실시간성 확보 (Section 2.1).

---

## 2. 방법론

### 2.1 두 단계 학습 (Section 2.1, Fig. 1)
1. **Video pre-training**: text + 첫 프레임 → 다음 프레임의 VQGAN 토큰을 autoregressive로 예측. 데이터: Howto100M(36M), Ego4D(1.2M), EPIC-KITCHENS(46k), SSV2(46k), Kinetics-700(121k), RT-1(82k), Bridge(25k) 등 합 38M (Fig. 2).
2. **Robot fine-tuning**: 다중 카메라 view + 로봇 state 추가, 출력은 미래 프레임 + cVAE 기반 action trajectory. 텍스트 인코더는 frozen, VQGAN도 frozen.

### 2.2 모델 스케일 (§3.5, Fig. 11)
| 변형 | trainable | 참고 |
|------|-----------|------|
| GR-2-S | 30M | scaling 분석 |
| **GR-2-B** | **95M** | 기본 보고 |
| GR-2-L | 312M | |
| GR-2-XL | 719M | |
- 기본값 230M 중 95M trainable. Pre-training validation loss(Ego4D/RT-1/in-domain robot)가 모델 크기에 따라 단조 감소.

### 2.3 Real-Robot 시스템 (§2.2)
- 7-DoF Kinova Gen3 + Robotiq 2F-85, head + wrist 듀얼 카메라.
- Cartesian trajectory를 자체 Whole-Body Control(WBC)로 200 Hz joint command로 변환, 충돌·manipulability 제약 통합.

### 2.4 데이터 증강 (Multi-task §3.1)
- Diffusion model로 새 객체 합성 + SAM 기반 배경 분리 + Latte 비디오 생성으로 robot motion 보존하며 OOD 시나리오 확장 ("GR-2 w/ DA").

---

## 3. 실험 결과

### 3.1 Multi-task (105 tasks, 8 skills) — Fig. 6

| 설정 | GR-1 | GR-2 (400 traj/task) | GR-2 w/ DA | GR-2 (50 traj/task) |
|------|------|----------------------|------------|---------------------|
| Simple (in-domain) | — | **97.7%** | 비슷 | 73.9% |
| Distractor | — | 우위 | — | — |
| Unseen Backgrounds | ~36 | **71.4%** | 더 향상 | GR-1 우위 |
| Unseen Environments | ~36 | **71.7%** | **87.0%** | — |
| Unseen Manipulation | — | 55.8% | 평균 OOD 74.7% | — |

- 50 traj/task로 줄여도 73.9% (Simple)로 강한 데이터 효율성.

### 3.2 End-to-End Bin Picking (122 objects, §3.2, Fig. 9)

| 설정 | GR-1 | **GR-2** |
|------|------|----------|
| Seen | 좋음 | 가장 높음 |
| Unseen | 급락 | Seen 수준 유지 |
| Cluttered Seen / Unseen | 급락 | Seen 수준 유지 |
| 평균 SR | **33.3%** | **79.0%** |

- 투명·변형·반사 객체까지 처리 (Fig. 8). 단일 명령 "move any object from the right basket to the left basket"로 일반화.

### 3.3 CALVIN ABCD-D (§3.3, Fig. 10)
- 1-task SR: GR-1 94.9% → **GR-2 98.6%**
- 5-task SR: GR-1 73.1% → **GR-2 85.9%**
- Average length: GR-1 4.21 → **GR-2 4.64** (RT-1, MT-ACT, HULC, RoboFlamingo 모두 능가).

### 3.4 자기회귀 비디오 생성 (§3.4)
- multi-task / bin-picking / CALVIN에서 예측 비디오와 실제 rollout이 거의 일치 (Fig. 12–17). "predicted action ≈ replay of predicted video" 가설.

---

## 4. 한계 및 미해결 문제

1. **Unseen Manipulation 55.8%** — 새 카테고리 객체·새 instruction 조합에서 약점. 실패 유형: novel shape grip 실패, 잘못된 객체 선택 (§3.1 results).
2. **재현 어려움**: 38M curation pipeline(MediaPipe hand filtering, Open-Sora re-captioning), VQGAN 가중치, in-domain robot 데이터 모두 비공개.
3. **데이터 비대칭**: pre-training은 단일 카메라, fine-tune은 멀티 view → multi-view 정렬을 얼마만큼 학습했는지 직접 비교 부재.
4. **컴퓨트**: 230M 본체 외 Latte/SAM/Detection model 등 augment pipeline 비용이 상당.
5. **추론 시 video generation 역할**: action은 cVAE 출력이지만 video token도 함께 생성. 이를 끄면 성능이 어떻게 변하는지 ablation은 보고되지 않음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★☆☆ — GR-1 대비 데이터·아키텍처 확장(seamless transfer 강조) |
| Technical depth | ★★★★☆ — VQGAN+GPT+cVAE+WBC까지 통합 |
| Experimental rigor | ★★★★★ — 5가지 OOD axis × 105 task × 122 object + CALVIN |
| Practical impact | ★★★★☆ — bin-picking 79% (Seen 수준 유지)는 산업적 의미 큼 |
| Open source | ☆ — 비공개 |

**강점**: 38M 비디오 + 4-size scaling으로 video pre-training의 효과를 정량화, CALVIN SOTA. **약점**: closed dataset, multi-view ablation 부재, unseen manipulation 약점 존재.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|-----------|
| 1 | 38M video curation의 핵심은? | Howto100M 36M 위주, hand filtering(MediaPipe) + Open-Sora re-captioning. 비공개 |
| 2 | 데이터를 1/8(50 traj/task)로 줄이면? | Simple 73.9% (vs 97.7%) — 매우 강한 데이터 효율 (Fig. 6) |
| 3 | Scaling이 saturate? | 30M→719M까지 validation loss 단조 감소, real SR도 향상 (Fig. 11). 한계점 미관측 |
| 4 | OpenVLA·π0와 동일 조건 비교? | 직접 비교 없음. CALVIN ABCD-D는 RoboFlamingo·HULC만 |
| 5 | 비디오 생성을 끄면? | ablation 미보고. "action ≈ replay video"라는 정성적 관찰만 (§3.4) |
| 6 | YAML(`calvin_abc_d_avg_len: 3.8`)과 paper 4.64의 차이? | paper Fig. 10은 ABCD-D split의 4.64. YAML 값은 ABC-D split이거나 구버전 추정 — **YAML 검토 필요** |

<!-- VERIFIED: pdf -->
