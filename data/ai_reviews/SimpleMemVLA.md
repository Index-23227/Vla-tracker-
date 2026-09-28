# SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models

> **한 줄 요약**: 별도 memory 모듈 없이 Qwen3.5-4B VLM의 native video context(타임스탬프 포함)에 최대 1분 분량의 history를 그대로 넣고, 생성된 sub-task 문장의 hidden state만으로 flow-matching action expert를 조건화하는 VLA. RMBench 94.0%, RoboMME 88.3%, MIKASA-Robo 74.0%, RoboMemArena TSR 63.6%, LIBERO 97.5%, LIBERO-Plus zero-shot 78.4%.

---

## 1. 배경 및 동기

- Long-horizon manipulation은 partially observable: 다음 행동을 고르는 데 필요한 정보(가려진 물체 위치, 반복 횟수, 앞서 보인 색상 cue)가 수십 초~수 분 전 관측에만 존재.
- 기존 VLA memory는 세 계열: **retrieval bank**, **learned compressor**(token compression), **recurrent state**. 저자들은 이들이 모두 "미래 결정이 무엇을 요구할지 알기 전에 무엇을 남길지 정해야 한다"는 **write-time commitment** 문제를 공유한다고 지적.
- 이 설계들은 "분 단위 history는 너무 커서 직접 처리 불가"라는 가정에서 출발했으나, 현대 VLM은 262k-token context를 가지며 60 s history(2 fps)는 약 5.6k token에 불과 → 가정이 더 이상 성립하지 않음.
- 핵심 질문: **history를 그대로 주면 policy가 실제로 활용할 수 있는가?**

## 2. 방법론 심층 분석

### 2.1 Native video history as context
- Head-camera 프레임을 최근 T_w 초 동안 f_v fps로 subsample(최대 K = T_w·f_v 프레임) → backbone의 **video channel**로 입력, 사전학습과 동일한 plaintext timestamp를 temporal patch 앞에 부착.
- 현재 wrist 카메라는 image channel(타임스탬프 없음). "multi-frame 카메라는 video, single-frame 카메라는 image"라는 단일 규칙.
- 프롬프트에 embodiment, 카메라 배치, timestamp 규약을 plain text로 명시.

### 2.2 Sub-task hidden state를 통한 action 생성
- Backbone이 현재 sub-task 한 문장 g_t를 일반 assistant 응답으로 생성(CoT 아님).
- 조건 집합 C_t = [e(g)⊕h(g) 각 토큰] + ψ(proprio) — 즉 **sub-task span의 token embedding + contextual hidden state**만이 history에서 action expert로 가는 유일한 통로.
- Action expert: DiT-style flow-matching, noise 쪽으로 편향된 τ 샘플링, 적은 Euler step으로 chunk 생성.
- 학습 손실 L = λ_sub·L_sub(sub-task span CE, teacher forcing) + λ_act·L_act(flow matching). Sub-task 라벨은 cloud VLM이 오프라인으로 생성.

### 2.3 Streaming prefix prefill
- 연속 결정은 history prefix 대부분을 공유 → 현재 chunk 실행 중 공유 prefix를 prefill하여 KV cache 저장, 다음 결정에서는 새 temporal patch와 instruction만 처리. 출력은 full recomputation과 동일.
- Appendix F: sliding-window attention(SWA) 변형으로 무한 스트림 대응 탐색.

## 3. 데이터 전략

- Suite별 1개 모델(해당 suite 전 태스크 공유). LIBERO-Plus는 LIBERO 모델을 추가 학습 없이 재사용.
- Suite별 설정(Table 8): RMBench(Aloha-AgileX 양팔, d_a=14, 60 s @2fps, 120 frames, H=30), RoboMME(Panda, 60 s @2fps), MIKASA-Robo(Panda, 3 s @20fps), RoboMemArena(Franka, 126 s @1fps), LIBERO(Franka, 30 s @2fps, 60 frames, H=16).
- 실로봇: dual-arm Piper, Cover Blocks 180 demos, Put Back Block 308 demos로 fine-tune.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Backbone | Qwen3.5-4B (native video interface) |
| Action head | DiT flow-matching expert |
| Optimizer | AdamW, LR 1e-5 (backbone) / 5e-5 (action head), cosine, grad clip |
| 학습 자원 | 128× H100, suite당 14–28 h |
| 추론 | 1× H100 bf16, 60 s history에서 결정 지연 0.68 s (full recompute 1.02 s, single-frame ≈0.65 s) |
| 실로봇 제어 | 25 Hz, chunk 30 중 16 실행 |

## 5. 실험 설계 및 평가 프로토콜

- Memory 벤치 4종: RMBench(9 task 평균, Place-Mat 제외, 100 seeds/task), RoboMME(16 task, 50 ep/task), MIKASA-Robo(5 task), RoboMemArena(26 task, TSR/CSR, 51 rollouts/task).
- 일반 벤치: LIBERO 4 suite(500 trials/suite), LIBERO-Plus(10,030 perturbed task, zero-shot).
- **통제 비교**: 동일 backbone·데이터·sub-task 감독·action head에서 memory interface만 retrieval / token compression(64 tokens) / recurrent state(16 tokens)로 교체.
- History intervention(마스킹·donor splice), 지연 측정, ablation(window 길이, frame 순서, timestamp, hidden vs embedding).

## 6. 실험 결과 심층 분석 (PDF Table 직접 인용)

### RMBench (Table 1)
| Method | M(1) Avg | M(n) Avg | Overall |
|---|---|---|---|
| MemoryWAM (specialist) | 84.2 | 81.5 | 83.0 |
| EventVLA (multi-task) | 79.0 | 54.0 | 67.8 |
| **SimpleMemVLA** | **91.6** | **97.0** | **94.0** |

- 단일 multi-task 모델이 task별 specialist를 능가. Obs&PU 65, Battery 90으로 가장 큰 격차.

### RoboMME (Table 2)
| Method | Count | Perm. | Ref. | Imit. | AVG |
|---|---|---|---|---|---|
| MemER | 48.8 | 53.2 | 38.0 | 29.5 | 42.4 |
| FrameSamp (Modul) | 65.2 | 25.1 | 36.3 | 51.4 | 44.5 |
| Ours-Retrieval | 46.5 | 42.0 | 13.5 | 24.0 | 31.5 |
| Ours-Token Compression | 46.0 | 13.5 | 12.5 | 18.5 | 22.6 |
| Ours-Recurrent | 31.0 | 24.0 | 17.5 | 10.0 | 20.6 |
| **SimpleMemVLA** | **91.5** | **96.0** | **82.5** | **83.0** | **88.3** |
| GroundSG (GT VLM oracle, 참고) | 83.9 | 93.3 | 95.2 | 64.0 | 84.1 |

### MIKASA-Robo (Table 3)
- SimpleMemVLA 74.0 (SGT 99, Intercept 83, RC3/5/9 71/58/59) vs MemoryVLA++ 44.4. 비-VLA 참조 GMP 67.8.

### RoboMemArena (Table 4)
- 평균 TSR 63.6 / CSR 72.1 vs FrameSamp+Modul 46.2 / 63.9. Transferring(35.8/37.4)에서는 FrameSamp+Modul(63.8/72.1)에 뒤짐.

### LIBERO / LIBERO-Plus (Table 5, 6)
- LIBERO 98.2 / 98.8 / 98.0 / 95.0, 평균 **97.5** (RIPT-VLA 97.5와 동률, OpenVLA-OFT 97.1).
- LIBERO-Plus zero-shot Total **78.4** (MemoryVLA++ 73.1, OpenVLA-OFT 69.6). Camera 76.9로 크게 앞서지만 Language 68.6은 MemoryVLA++ 88.7보다 낮음.

### 실로봇 (Table 7)
- Cover Blocks 35/60 (58.3%), Put Back Block 28/40 (70.0%). 실패 원인은 주로 grasp 실패 등 저수준 실행 오류.

## 7. Ablation 분석

- **Window 길이**(Fig. 8a): cover-blocks는 증거가 약 25 s 전에 있어 30 s window는 성공, 15 s window는 실패(cliff). press-button은 window 축소 시 점진적 하락(slope). 현재 프레임만 쓰면 둘 다 실패.
- **순서/타임스탬프**(Fig. 8b): frame shuffle → 두 task 모두 실패. Timestamp를 0으로 → cover는 무영향, press(카운팅)는 크게 저하.
- **Hidden vs embedding**(Fig. 8d): 다른 목표의 hidden state로 교체하면 행동이 교체된 목표로 이동, embedding 교체는 거의 무영향 → contextual hidden state가 memory→action 인터페이스.
- **Staleness**(Fig. 8c): 한 결정 전의 hidden state만 재사용해도 성공률 급락 → 매 결정마다 history를 다시 읽어야 함.
- **History intervention**(Fig. 6): PickXtimes에서 pick 이벤트 하나를 마스킹하면 16/16에서 카운트가 정확히 1 감소; donor 영상 splice 시 행동이 donor 내용으로 전환 → "visual in-context learning".
- **Streaming**(Fig. 7): 45분(≈245k tokens) 입력에서 결정 경로 지연 32.1 s → 1.18 s.

## 8. 관련 연구 비교

| 계열 | 대표 | SimpleMemVLA 대비 |
|---|---|---|
| Retrieval bank | MemER, keyframe 선택 | 순서·타임스탬프 손실, RoboMME 통제 비교 31.5 |
| Learned compression | MemoryVLA/++, ContextVLA | 세부 percept 손실(Permanence 13.5) |
| Recurrent state | RMT, TTT, CronusVLA | 반복 업데이트로 증거 덮어씀(20.6) |
| Symbolic scene graph | SimpleSG/GroundSG | GT perception oracle도 84.1로 native context(88.3) 이하 |

- 기여의 본질은 "새 모듈"이 아니라 **VLM 사전학습 video 인터페이스를 그대로 재사용** + sub-task hidden state 병목 + prefix prefill이라는 엔지니어링 조합.

## 9. 한계 및 미해결 문제

- Window 밖으로 나간 증거는 복구 불가(cover-blocks 15 s window 실패). 262k 한계 ≈ 48분.
- Suite마다 T_w, f_v를 증거가 남는 시간 범위에 맞춰 수작업 설정(Table 8) — 태스크 사전지식 의존.
- Sub-task 라벨을 cloud VLM으로 생성 → 라벨 품질·비용 의존.
- 128× H100 학습, 4B backbone → 소규모 연구실 재현 부담.
- LIBERO-Plus Language 축(68.6)에서 약함; RoboMemArena Transferring에서 열세.
- 실로봇은 2개 task, 단일 플랫폼만 검증.

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "memory 모듈이 필요 없다"는 반직관적 주장을 통제 실험으로 뒷받침 |
| **Technical depth** | ★★★★☆ — hidden-state 병목, prefix prefill, intervention 분석이 탄탄 |
| **Experimental rigor** | ★★★★★ — memory 벤치 4종 + 일반 벤치 2종 + 동일 스택 통제 비교 + 실로봇 |
| **Practical impact** | ★★★★☆ — 코드 공개, 단일 모델로 다중 태스크, 지연을 single-frame 수준으로 |
| **Writing quality** | ★★★★☆ |

**강점**: write-time commitment라는 명확한 문제 정의와 그에 대한 최소주의 해법; 동일 backbone에서 세 memory 계열 재구현 비교.
**약점**: suite별 window 하이퍼파라미터 의존, 대규모 계산 자원, 초장기(수십 분 이상) 시나리오는 SWA 변형의 탐색적 결과에 머묾.

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | 성능 향상이 backbone(Qwen3.5-4B) 덕분 아닌가? | Table 2 통제 비교: 같은 backbone·데이터·감독에서 retrieval 31.5 / compression 22.6 / recurrent 20.6 vs native 88.3. |
| 2 | Sub-task 문장만으로 정보가 충분히 전달되나? | 전달되는 것은 문장 자체가 아니라 contextual hidden state. Fig. 8d에서 hidden 교체 시 행동이 따라가고, target 단어를 지워도 순서/카운트를 유지. |
| 3 | 지연은 실시간 제어에 충분한가? | 60 s history에서 0.68 s, real-time budget(16 steps @16.7 Hz = 0.96 s) 이내. |
| 4 | Window를 어떻게 정하나? 증거가 window 밖이면? | Suite별 수작업(Table 8). 밖으로 나가면 실패(Fig. 8a cliff) — 명백한 한계. |
| 5 | Visual in-context learning 주장은 과한가? | 편집된 history는 학습 중 등장하지 않았고 파라미터 업데이트 없이 donor 내용을 따름(PatternLock 96% 변경, 89% donor). 다만 태스크 분포 내 조작에 한정. |
| 6 | 범용 성능 손실은? | LIBERO 97.5로 최상위권 동률, LIBERO-Plus zero-shot 78.4로 최고. |
| 7 | RMBench 비교가 공정한가? | 대부분 baseline은 task별 specialist, SimpleMemVLA는 단일 multi-task 모델 — 오히려 불리한 조건. |
| 8 | 왜 Language perturbation에 약한가? | 논문은 suite별 분석(Table 11)만 제시; sub-task 생성이 instruction 변형에 민감할 가능성. |

## 12. 재현성 및 후속 연구 제안

- 코드: https://github.com/OpenBMB/SimpleMemVLA (공개). Suite별 설정은 Table 8에 모두 명시.
- 재현 난점: 128× H100 학습, cloud VLM sub-task 라벨링 파이프라인.
- 후속 방향: (1) 증거 위치에 따른 adaptive window/frame rate, (2) SWA 변형의 장시간 연속 운용 검증, (3) 더 작은 backbone에서의 native-context memory 유효성, (4) 다양한 실로봇 플랫폼·장기 태스크 확장.

<!-- VERIFIED: pdf -->
