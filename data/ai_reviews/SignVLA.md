# SignVLA: Real-Time Sign Language-Guided Robotic Manipulation via Attention LSTM and Vision-Language-Action Models

> **한 줄 요약**: UCL 팀이 제안한 SignVLA는 MediaPipe 손 랜드마크 + Attention-LSTM(33-sign ASL 어휘) + temporal stability buffer로 구성된 경량 sign-to-text 프런트엔드를 사전학습 GR00T N1 (Eagle-2 VLM + Diffusion Transformer) 정책 앞에 붙여, 음성·텍스트 의존 VLA를 청각장애 사용자가 수어로 직접 제어 가능하게 한 최초의 sign-conditioned VLA로서, LIBERO-Object 98.50% / Spatial 97.50% / Goal 97.50% / Long 94.50% 성공률을 달성했다.

---

## 1. 배경 및 동기

기존 VLA(OpenVLA, RT-2, GR00T N1 등)는 음성 또는 텍스트 명령에 의존해 deaf·hard-of-hearing·speech-impaired 사용자 접근성이 제한적이다. 대규모 sign translation 모델(Transformer 기반)은 정확하지만 latency와 하드웨어 비용이 커 실시간 로봇 제어에 부적합. 저자들은 (1) modality mismatch(연속적 spatio-temporal 제스처 vs. 이산 토큰 명령), (2) 프레임 단위 인식 오류로 인한 command flickering 문제를 해결하기 위해 VLA 백본을 수정하지 않는 경량 sign-to-text 인터페이스를 제안한다.

---

## 2. 방법론 심층 분석

### 2.1 시스템 아키텍처 (Fig. 1)

- **Sign Perception**: 웹캠 RGB → MediaPipe Hands → 프레임당 544-dim landmark feature (양손 절대좌표 + 손목 상대좌표 + finger direction + frame-to-frame velocity, 결손 시 zero-pad).
- **Attention-LSTM**: T=32 frame 윈도우, 33-sign ASL 어휘. additive attention으로 hidden state $h_t$에 시간 가중치 $\alpha_t = \mathrm{softmax}(w^\top \tanh(W h_t))$ 적용 → context vector $c=\sum_t \alpha_t h_t$ → linear classifier.
- **Temporal Stability Buffer**: 6-frame window에서 동일 class가 3 consecutive frame argmax이고 8-frame cooldown 경과 시에만 커밋 → command flickering 방지.
- **LLM Gloss → Instruction**: gloss 시퀀스를 predefined template으로 자연어 명령(예: "pick up the butter and place it in the basket")으로 변환.
- **VLA 백엔드**: GR00T N1 (Eagle-2 VLM + Diffusion Transformer, 2.2B), Franka Emika Panda 7-DOF (100Hz planner / 1kHz servo).

### 2.2 학습

- **Sign Classifier**: ASL Citizen 33-sign subset (train 921 / val 224 / test 288), weighted cross-entropy + label smoothing, Gaussian noise + temporal masking 증강.
- **GR00T 정책**: LIBERO demo로 SFT (20,000 steps, batch 640), object-to-basket/container/surface + scene interaction 태스크.

### 2.3 핵심 설계 선택

VLA 백본 수정 없이 프런트엔드만 추가 → 기존 VLA 생태계와 호환. 정확한 sign translation 대신 task-constrained 어휘(LIBERO-Object) 사용으로 latency와 데이터 부담 동시 해소.

---

## 3. 실험 결과

### 3.1 Sign Recognition Ablation (Table I)

| Model | Top-1 | Top-5 |
|---|---|---|
| LSTM (200-word) | 25.93 | 60.36 |
| LSTM (LIBERO) | 34.72 | 75.35 |
| **Attn-LSTM (200-word)** | 67.01 | 89.23 |
| **Attn-LSTM (LIBERO)** | **88.89** | **96.53** |

Additive attention만으로 top-1 +41pp(200-word) / +54pp(LIBERO). 태스크 제약 어휘가 추가로 +22pp.

### 3.2 Per-class Accuracy (Table II)

288 test 샘플 전체 top-1 86.1%. 13개 sign이 100% 정확(BALL, BOWL, BUTTER, CHEESE, MOVE, ON, PHONE, PICK, PLATE, TOMATO, TURN, UP, YES). 주 오분류: BOTTLE/BOX/BOWL (유사 handshape), GIVE/TAKE/STOP (유사 motion arc).

### 3.3 Sign-Conditioned LIBERO Manipulation (Table III)

| Task Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 195/200 (**97.50%**) |
| LIBERO-Goal | 195/200 (**97.50%**) |
| LIBERO-Object | 197/200 (**98.50%**) |
| LIBERO-10 (Long) | 189/200 (**94.50%**) |

각 suite 200 episode 평가, sign-derived template 명령 사용. LIBERO-Object 최고 — 객체 중심 어휘와 정합. Long-horizon 약간 하락.

### 3.4 정성적 결과

Fig. 6: butter-to-basket sign 명령으로 grounding → grasp → placement 성공. Fig. 7-8: 실제 Franka 셋업 준비 단계(teleop demo collection → SFT → closed-loop).

---

## 4. 한계

1. **VLA novelty 부재**: 정책 자체는 사전학습 GR00T N1을 그대로 사용 — sign 프런트엔드가 본질적 기여. VLA 모델로 분류하지만 manipulation 능력은 GR00T N1에 귀속.
2. **어휘 제한**: 33-sign LIBERO-Object 한정. 일반 ASL 어휘로 확장 시 정확도 급락(67.01% top-1).
3. **Template 의존**: gloss → 자연어 변환이 predefined template 기반 — open-vocabulary 명령 생성 미검증.
4. **Real-world 미평가**: 본 논문은 LIBERO simulation만 평가. Franka real-world deployment는 "preparation" 단계로만 기술.
5. **LIBERO 평가 출처 모호**: Table III의 GR00T 성공률(94.5–98.5%)이 sign 입력으로 인한 것인지 텍스트 입력과 동일한지 명시적 ablation 없음.
6. **Sign latency 미보고**: MediaPipe + LSTM + buffer의 end-to-end latency 수치 부재.
7. **단일 사용자**: 다양한 signer(연령·인종·구사 능력)에 대한 일반화 미검증.

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★☆☆ — 첫 sign-language VLA 통합, 다만 component는 표준적 |
| Technical depth | ★★★☆☆ — Attention-LSTM은 견고하나 단순 |
| Experimental rigor | ★★★☆☆ — LIBERO simulation 한정, real-world 부재 |
| Practical impact | ★★★★☆ — Accessibility 차원에서 의미 있는 첫걸음 |

**강점**: Accessibility라는 underexplored 축을 VLA에 도입, 경량 프런트엔드로 기존 VLA 백본 재사용 가능. **약점**: 정책 novelty 부재, sign vs. text 명령 controlled 비교 누락.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | Sign 입력이 text baseline 대비 정말 의미 있는가? | Paper에는 동일 GR00T를 text instruction으로 평가한 controlled comparison이 없음. Table III의 높은 성공률은 GR00T의 능력일 가능성이 큼. |
| 2 | Open-vocabulary 수어로 확장 가능한가? | 현재 33-sign template 기반 — open-vocab는 ASL gloss-to-text 모델(예: gloss-free Transformer) 통합이 필요. |
| 3 | Real-time latency는? | MediaPipe(~30Hz) + LSTM(T=32) + 6-frame buffer + 8-frame cooldown → 대략 0.5–1초 latency 추정, 정확한 수치는 paper에 부재. |
| 4 | LIBERO-10 성능이 떨어진 이유? | Long-horizon에서 중간 sign 명령 변경 처리 미흡; "confirmed instruction is maintained unless new sign is explicitly confirmed" 정책의 한계. |

---

## 7. 데이터셋 및 평가 프로토콜

- **ASL Citizen** [22]: Microsoft 공개 대규모 ASL community-sourced dataset에서 33-sign 필터링.
- **LIBERO**: object-to-basket/container/surface/scene 태스크. 각 suite 200 episode.
- **Franka Emika Panda**: 7-DOF, 100Hz planner / 1kHz servo (real-world deployment 준비 단계).

---

## 8. 비교 모델

| 모델 | 입력 모달리티 | 백엔드 | LIBERO Avg |
|---|---|---|---|
| OpenVLA | text | Llama-2 + autoregressive | ~76% |
| GR00T N1 (paper baseline) | text | Eagle-2 + DiT | (text 조건 미보고) |
| **SignVLA** | **sign language** | Eagle-2 + DiT (GR00T N1) | **97.0%** |

직접 비교 baseline이 부족 — text-conditioned GR00T 점수가 같은 환경에서 보고됐다면 attribution이 명확해질 것.

---

## 9. 시사점

- VLA의 accessibility 차원(시각·청각·운동 장애 사용자) 연구 공백을 지적하고 실증.
- "VLA 백본은 그대로, 입력 인터페이스 모듈화"라는 디자인 패턴은 다른 alt-modality(gaze, BCI, gesture) 확장 가능성 시사.
- 경량 (MediaPipe + LSTM) 추론으로 edge 배포 친화적.

---

## 10. 후속 연구 방향

1. **Continuous sign translation**: gloss-free Transformer로 open-vocab 명령 생성.
2. **Bi-directional feedback**: 로봇이 sign으로 응답하는 양방향 HRI.
3. **Real-world Franka 평가**: 본 paper의 "preparation" 단계 완성.
4. **Multi-signer robustness**: signer 다양성에 대한 도메인 적응.
5. **Latency 정량화 및 최적화**: stability buffer 파라미터 자동 튜닝.

---

## 11. 결론

SignVLA는 VLA 시스템에 sign-language 입력을 결합한 최초의 통합 시스템이다. 핵심 기여는 정책 자체의 진보가 아니라, MediaPipe + Attention-LSTM + temporal stability + LLM template으로 구성된 경량 sign-to-text 모듈을 사전학습 GR00T N1과 결합해 청각장애 사용자가 로봇을 직접 제어할 수 있게 한 점이다. LIBERO 4-suite에서 평균 97.0% 성공률을 보고했으나, text baseline과의 controlled comparison과 real-world 평가가 후속 연구의 핵심 과제다.

---

## 12. 참고문헌 요약

- **GR00T N1** [4]: NVIDIA, arXiv:2503.14734. Eagle-2 VLM + Diffusion Transformer dual-system humanoid VLA.
- **OpenVLA** [3]: Kim et al., CoRL 2024. 7B open-source VLA baseline.
- **RT-2** [2]: 인터넷-스케일 pretraining의 VLA generalization.
- **ASL Citizen** [22]: Microsoft 대규모 ASL benchmark.
- **MediaPipe Hands** [18]: 실시간 손 랜드마크 추출.
- **Camgoz et al.** [14]: Sign language Transformer (recognition + translation joint).

<!-- VERIFIED: pdf -->
