# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

> **한 줄 요약**: VILA 기반 VLA가 자연어 형태의 mid-level action(예: "turn right 30 degrees")을 생성하고, 단일 단계(single-stage) RL로 학습된 LiDAR 기반 visual locomotion policy가 이를 실행하는 2-level 계층 구조로, R2R-CE Val-Unseen에서 SR 54%, RxR-CE에서 SR 49.3%, 실세계 25 instructions에서 88% 성공률을 달성한 RSS 2025 논문.

---

## 1. 배경 및 동기

- VLN(Vision-Language Navigation)은 대부분 wheeled robot 또는 Habitat의 discrete teleport 기반 → legged robot의 좁은 통로·계단·outdoor terrain 등 실제 challenge를 다루지 못함.
- End-to-end VLA가 quantized low-level command를 직접 출력하면 LLM/VLM의 자연어 reasoning capability를 충분히 활용하지 못함.
- 핵심 질문: **"reasoning을 자연어 도메인에 유지하면서 다리 로봇 control과 결합할 수 있는가?"**

---

## 2. 방법론

### 2.1 High-level VLA (Sec. II-A, Fig. 3)

- **Backbone**: VILA(SigLIP vision encoder + LLaMA-3 8B + MLP projector). VILA-2단계(visual-language pretrain) 모델을 SFT data blend로 1 epoch fine-tune.
- **Navigation Prompt**: 현재 frame 1장 + 이전 t-1 frame에서 첫 frame 포함 균등 샘플링. 둘을 textual cue("video of historical observations:" / "current observation:")로 구분 — 별도 special token 없이 pure language prompt 유지.
- **Output 파싱**: regex parser(ref. [35])로 LLM 출력에서 "turn left θ°", "move forward d cm", "stop" 추출. 모든 실험에서 100% 매칭 성공.
- **SFT data blend (4 source)**: (1) YouTube 인간 touring 비디오 2K개 → MASt3R metric pose estimation으로 step-wise action 추출 → 20K trajectory(Fig. 4); (2) R2R-CE/RxR-CE simulation; (3) EnvDrop·trajectory summarization·ScanQA auxiliary; (4) general VQA.

### 2.2 Visual Locomotion Policy (Sec. II-B)

- **Action**: desired joint position $q^d \in \mathbb{R}^{12}$ (Go2 12 leg DOF), PPO로 학습.
- **Input**: proprioception + LiDAR 기반 2.5D height map. 거리 명령은 fixed velocity{0.5 m/s, ±π/6 rad/s}로 cast.
- **Single-stage RL**: 기존의 teacher-student two-stage 접근 대신 RL로 직접 학습, Isaac Lab ray-casting으로 RTX 4090에서 60K FPS throughput 확보.
- 두 timescale: VLA는 저주파(고비용), locomotion policy는 real-time.

> ❓ **예상 질문**: VLA가 LLM 출력 형식(자연어)을 어겨 parser가 실패하면?
> **답변**: 저자들은 모든 실험에서 regex 매칭이 성공했다고 명시(Sec. II-A 마지막 단락). LLM의 일관된 출력 분포를 SFT data blend가 강제했다고 추정.

---

## 3. 실험 결과

### 3.1 R2R-CE / RxR-CE Val-Unseen (Table I)
| Method | Obs. | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ |
|---|---|---|---|---|---|
| NaVid (single-view RGB) | RGB | 37.0 | 35.0 | — | — |
| HNR* (panoramic+depth+odo+waypoint) | Pano. | 61.0 | 51.0 | 56.3 | 46.7 |
| **NaVILA (single-view RGB only)** | **RGB** | **54.0** | **49.0** | **49.3** | **44.0** |

- 단일 view RGB만으로 panoramic+depth+odometry를 쓴 simulator-pretrained waypoint predictor 기반 baseline에 근접/추월. NaVid 대비 R2R SR +17%p.
- Cross-dataset(Table II): R2R로만 학습 후 RxR-CE 평가 — NaVILA SR 34.3 vs NaVid 23.8 (+10.5%p).

### 3.2 ScanQA Spatial Scene Understanding (Table III)
- NaVILA(64 frames) CIDEr 102.7 / Bleu-4 16.9 / Meteor 20.1 → NaviLLM(75.9 CIDEr) 대비 +20+ points. Depth/3D 입력 없이 LEO·Scene-LLM과 동급.

### 3.3 VLN-CE-Isaac (Table IV) & Locomotion (Table V)
- **VLN-CE-Isaac (Go2)**: NaVILA-Vision SR 50.2 vs NaVILA-Blind 36.2 (+14%p) — vision-based locomotion의 obstacle avoidance 효과.
- **H1 humanoid**: SR 45.3 vs blind 24.4 (+21%p).
- **Low-level (Table V)**: NaVILA collision rate 0.81 vs ROA 3.09 (3.8× 감소).

### 3.4 Real-world (Table VI)
- Unitree Go2 25 instructions × 3 trial: 평균 SR 88%, 복잡 지시 75%. GPT-4o 대비 모든 환경(Workspace/Home/Outdoor)에서 우월.
- Booster T1 humanoid: VLA retraining 없이 cross-embodiment 작동 (Outdoor Simple SR 0.89).
- Human video ablation: NaVILA† (no human video) → outdoor에서 SR 0% → human video 추가 시 100%, 일반화 향상의 핵심.

---

## 4. 한계 및 미해결 문제

1. **Manipulation 미포함**: Navigation only — mobile manipulation(예: 문 열기 + 통과)으로의 통합 미해결.
2. **Image-based VLM 비용**: 저자들이 Sec. V의 Limitations에서 명시. Long-context LLM이 해결 방향.
3. **Mid-level granularity 고정**: {0.5 m/s, ±π/6 rad/s, stop} 4가지 cast — 미세한 속도 제어가 필요한 시나리오에서 한계.
4. **Real-world failure cases**: Appx. Sec. E에 명시(논문에서 정성적으로만 언급) — 더 큰 simulation 학습이 향후 개선점.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Mid-level language action + single-stage vision RL 결합 |
| **Technical depth** | ★★★★☆ — VILA SFT blend, MASt3R 기반 human video pipeline |
| **Experimental rigor** | ★★★★★ — Sim(R2R/RxR/Isaac) + ScanQA + Real(Go2/H1/T1) |
| **Practical impact** | ★★★★★ — Cross-embodiment, 실제 outdoor 작동 |

**강점**: Action을 자연어로 유지하여 LLM의 reasoning을 보존, single-stage RL로 학습 시간 단축, YouTube 비디오로부터 continuous navigation 학습.
**약점**: VLA 추론 latency, manipulation·복잡 속도 제어 부재.

---

## 6. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Single-view RGB로 panoramic+depth 기반 SOTA를 넘는다는 게 왜 가능한가? | VLM의 large-scale image-text pretrain이 panoramic 정보 결손을 reasoning으로 보완. Table I에서 simulator-pretrained waypoint predictor를 쓰지 않은 RGB-only baseline 중 SOTA 확인. |
| 2 | Mid-level language action이 fine-grained control을 잃지 않나? | Velocity는 fixed cast이지만 height map 기반 RL policy가 실제 obstacle avoidance를 담당. Table V에서 collision rate 0.81로 ROA 대비 3.8× 우월. |
| 3 | Human video 학습이 실제로 outdoor 일반화에 기여하는가? | Table VI 마지막 행: NaVILA† vs NaVILA — Outdoor Simple SR 0% → 100%로 결정적 효과. |
| 4 | Cross-embodiment가 retraining 없이 정말 가능? | Booster T1 결과(Workspace Simple SR 93%)가 retrain 없이 동일 VLA로 작동함을 입증. 단, locomotion policy는 robot별 학습 필요. |

<!-- VERIFIED: pdf -->
