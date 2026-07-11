# HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control

> **한 줄 요약**: 장기(long-horizon) 조작에서 "강한 추론 모델은 실시간 제어에 너무 느리고, 빠른 모델은 추론이 부족하다"는 **frequency-competence paradox**를, 고빈도 **Executor**(π0.5 VLA, transient memory) — 경량 **Sentry**(Qwen3-VL-8B, working memory) — 저빈도 **Planner**(GPT-4o, episodic memory)의 3계층 위계로 분해하여 해결. 크로스모달 시맨틱 스키마(이미지+고밀도 텍스트) 기반 episodic memory에 명시적 **Add/Update/Delete** 능동 관리를 도입, WidowX-250s 실로봇 3개 장기 과제에서 평균 task progress **90%** (Flat/FIFO memory 65%, Human oracle 94%), Planner API 호출 **~3배 절감** (5.4→1.8/subtask).

---

## 1. 배경 및 동기

### 마르코프 가정의 한계
- 대부분의 VLA는 p(a_t | o_t, l) 형태로 **현재 관측만** 조건화 — 과거 사건의 사슬이나 현재 보이지 않는 잠재 정보에 의존하는 **non-Markovian** 장기 과제에서 실패.
- 예: "Alice의 장난감을 모두 찾아라" — 이전에 열어본 상자의 내용물, 사용자 선호("Alice의 장난감은 오리")를 기억해야 함.

### Frequency-Competence Paradox
- 학습 기반 메모리(auxiliary loss, past-token prediction 등)는 제한된 context window와 수백 스텝 장거리 인과 최적화의 어려움에 부딪힘.
- LVLM을 메모리 컨테이너로 쓰는 접근(MemER 등)은 실시간 제어의 지연 상한 때문에 배포 가능한 VLM 규모가 제한 → 세계 지식/일반화가 약해짐.
- **핵심 통찰**: 대부분의 메모리 연산은 고수준 추론을 본질적으로 요구하지 않음 → 시간 해상도별로 역할을 분리하면 됨.

### 기존 메모리 시스템의 두 가지 결핍
1. **멀티모달 풍부성 부족**: "빨간 컵은 Alice의 컵" 같은 소유권·선호는 vision-only 궤적으로 포착 불가.
2. **수동적 축적의 정체**: passive accumulation만 지원하면 환경이 바뀔 때 낡은/모순된 지식으로 cognitive dissonance 발생.

---

## 2. 방법론 심층 분석: 3계층 메모리 위계

인간 인지의 multi-store 구조를 모사한 구조화 상태 공간:

| 계층 | 모듈 | 메모리 | 내용 | 빈도 |
|------|------|--------|------|------|
| Transient Memory (T_t) | **Executor** π_e (π0.5 VLA) | 즉시 관측 o_t | 반응적 감각-운동 협응의 최소 충분 상태 | 2 Hz (제어 10 Hz) |
| Working Memory (W_t) | **Sentry** π_s (Qwen3-VL-8B) | 최근 관측 슬라이딩 윈도우 {o_{t-h_s},...,o_t}, h_s=8 | 서브골 완료 검증, 급작스런 실패 감지 | n_m=5 간격 |
| Episodic Memory (E_t) | **Planner** π_p (GPT-4o) | Contextual(멀티모달 KV 스토어) + Procedural(서브골 리스트) | 객체 상태 변화, 공간 제약, 사용자 선호; done/active/pending 상태 플랜 | Sentry 트리거 시 |

### 정책 분해
- **Executor**: a_t ~ π_e(a_t | o_t, τ_t) — active subgoal τ_t가 복잡한 episodic history를 추상화한 조건 변수. local Markov 가정으로 stateless·고효율.
- **Sentry**: 게이팅 함수 u_t = I[π_s(W_t, τ_t) > δ] (t ≡ 0 mod n_m). u_t=1이면 "handover" — 서브골 완료/무효화 신호로 Planner 호출. u_t=0이면 E_{t+1} ← E_t로 고수준 추론 우회.
- **Planner**: 2단계 갱신 — (1) Context Retrieval: q_t = f_enc(l, W_t)로 TopK 검색, (2) Memory Update & Re-planning: (E^c_{t+1}, E^p_{t+1}) ← π_p(l, W_t, M_ret, E^p_t).

📌 [Figure 2 삽입] — Sentry가 진행 모니터링·버퍼링, 완료 감지 시 Planner가 궤적 리뷰 → 메모리 검색/통합 → 갱신된 지시를 Sentry에 반환하는 폐루프

---

## 3. 방법론 심층 분석: 크로스모달 스키마와 능동 메모리 관리

### What to memorize: Cross-modal Semantic Schemata
- Episodic memory를 **object-centric**으로 조직 — 시각 경험을 고밀도 텍스트 설명에 앵커링.
- 픽셀 공간에 보이지 않는 소유권·절차 규칙·인간 선호를 텍스트로, 공간·인식 정보를 이미지로 보존.
- 검색: OpenAI text-embedding-3로 캡션 임베딩, 코사인 유사도 기반 vector DB. 레코드 필드: id, tags(2-5개), data(type/value), image_path. 태그 기반 쿼리 우선.

### How to memorize: Active Management (CRUD)
- **Add**: 새 사실/선호 삽입. **Update**: 낡은 항목 수정 (시제 의미 유지). **Delete**: 중복/오류만 절제적으로 제거.
- 상태 변화 시 새 레코드 CREATE + 이전 레코드를 과거로 UPDATE — 검증 가능한 사실만 저장 ("Do NOT CREATE FAKE memory", 불확실하면 inspection 서브태스크 계획).
- 2턴 프로토콜: Turn 1 = QUERY만, Turn 2 = CRUD + 최종 plan_list (XML 구조 출력).

> ❓ **예상 질문**: 왜 학습이 아닌 프롬프트 기반 CRUD인가?
> **답변**: Sentry·Planner는 zero/few-shot으로 동작 — 사전학습된 일반화를 활용해 데이터 수집 없이 장기 논리를 처리. 도메인 특화 학습은 경량 Executor에만 국한하는 것이 설계 철학.

---

## 4. 구현 및 학습 세부

| 항목 | 값 |
|------|----|
| Executor | π0.5 (DROID 사전학습 공개 체크포인트) → 실로봇 시연 60개(정규 50 + corner case 10)로 파인튜닝, OpenPI 표준 레시피 |
| Executor 학습 | AdamW, peak LR 5e-5, batch 256, 30k steps, cosine decay, warmup 10k, EMA 0.999, action horizon 10 |
| Sentry | Qwen3-VL-8B, vLLM 로컬 서빙, 1x H100, temp 0.6, max len 8192 |
| Planner | GPT-4o API (본실험) / Qwen3-VL-30B 로컬 2x H100 (보충, 재현성) |
| 제어 | Executor 2 Hz, 10-action chunk (10 Hz) 예측, 5개 open-loop 실행; Sentry는 10 실행 스텝마다 질의 |
| 로봇 | WidowX-250s + 평행 그리퍼, 3인칭 RealSense D435 + 손목 카메라, ROS, leader-follower 텔레옵, 25 Hz→10 Hz 서브샘플, 224×224, LeRobot 포맷 |

---

## 5. 실험 설정

### 3개 실로봇 장기 과제 (각 20 trials/method)
| 과제 | 서브태스크 | 총 action steps | 핵심 메모리 도전 |
|------|-----------|-----------------|------------------|
| **Object Search** | 6 | ~1450 | 능동 탐사(불투명 상자 inspect)로 메모리 형성 + 환경 변화 시 갱신 + 선호 기반 검색 |
| **Counting** | 7 | ~1200 | 레시피 판독 후 지속적 시맨틱 메모리·누적 진행 추적 (David의 녹두 선호 등) |
| **Rearrangement** | 6 | ~1035 | "clear & restore" — 원래 배치의 장기 공간 메모리 + 사전 선호와 통합 복원 |

### 평가 지표
- **Task Progress**: 완료된 서브태스크 비율. **API Calls**: 서브태스크당 Planner 호출 수 (효율 프록시). **Memory Hit Rate**: Planner 호출 시점에 필요한 정보가 메모리에 존재할 확률.

### 비교군 (π_e·Planner 백본 고정, 메모리 컨텍스트·계획 빈도만 변화)
Transient Memory (Hi-robot 스타일 주기 호출) / Transient+Sentry / Flat Memory (MemER 스타일: 최근 8 관측 + FIFO 키프레임) / HiMe w/o Sentry / **HiMe** / Human High-level (oracle 상한).

---

## 6. 주요 결과

### Main (Fig. 4, 평균 Task Progress)
| 방법 | Object Search | Counting | Rearrangement | 평균 |
|------|--------------|----------|---------------|------|
| Transient Memory | 0 | 18 | 23 | 14 |
| Transient + Sentry | 0 | 23 | 56 | 26 |
| Flat Memory (MemER류) | 64 | 58 | 73 | 65 |
| HiMe w/o Sentry | 66 | 71 | 66 | 68 |
| **HiMe (Ours)** | **92** | **92** | **87** | **90** |
| Human High-level | 94 | 96 | 92 | 94 |

### 효율 (Table 2)
- API Calls/subtask: HiMe **1.8 / 2.6 / 1.4** vs Flat **5.4 / 4.8 / 6.2** — **~3배 절감**. VLM 추론이 지연을 지배하므로 직접적 속도 향상.
- Memory Hit: HiMe **94/98/92%** vs Flat **68/61/76%** — FIFO의 조기 관측 망각이 불필요한 재탐사를 유발.

### Human oracle과의 격차
- 평균 90 vs 94 — 고수준 계획 격차가 거의 해소됨. 잔여 오차는 주로 π_e의 물리적 실행 능력 한계.

---

## 7. Ablation 및 메커니즘 분석

### Q1: 메모리 표현 양식 (Fig. 5)
- **Only image 86 vs Only text 74** (Object Search) — 텍스트는 손실 압축; 초기 지각이 놓친 객체는 원시 시각 없이는 재접지(re-grounding) 불가.
- **Only text 91 vs Only image 78** (Counting) — 시맨틱·논리 의존성·선호는 픽셀에서 즉석 추출이 어려움.
- **크로스모달 평균 90** > text 84, image 80 — 이미지의 공간/인식 충실도 + 텍스트의 시맨틱 구조 결합.

### Q2: 능동 관리의 필요성 (Fig. 6)
- FIFO(8개 상한) 68% < No Management(append-only) 86% < **Ours 90%**.
- 망각의 비용: 조기 컨텍스트(예: 초기 관측된 객체 위치)가 후기 추론에 결정적. 일관성의 가치: Update/Delete 없이는 낡은 상태가 노이즈로 잔존.
- **통제 실험 (Table 9, Rearrangement)**: 같은 8-entry 예산에서도 Limited Active Management 80% > FIFO 64% — 이득이 메모리 용량만으로 설명되지 않음 (Unbounded 87%).

### Q4: Sentry 종료 판정 (Fig. 7)
- 프레임 1→8: Precision ~76→82%, Recall ~22→35%. 단일 프레임으로는 일시 정지와 진짜 완료 구분 곤란.
- **보수적 편향**: 높은 Precision·낮은 Recall — 조기 중단은 최소화하지만 "missing signal"로 무한 루프 위험 → **고정 간격 Planner fallback**으로 보완.

### 오픈소스 Planner 치환 (Fig. 9, Qwen3-VL-30B)
- Rearrangement에서 HiMe 83.3 > Flat 48.3 > Transient 25.0 — 경향 일관, 구조적 이득이 Planner 선택에 강건.
- 지연 (Table 11): GPT-4o API P50 38.6s vs Qwen3-VL-30B 로컬 **6.8s** — 로컬 서빙으로 추론 지연 대폭 감소, Sentry 게이팅 덕에 고빈도 루프와 분리됨.

---

## 8. 정성적 분석 및 실패 모드

- **자기 수정(self-correction)**: 모순 지시("장난감을 씻었으니 제자리에") 시 내부 지식 베이스를 Update — 기존 retrieval 기반 방법에 없는 능력.
- **메모리 확장성 (Table 10)**: 다중 라운드 Object Search에서 라운드 1→3에 메모리 5.5→14.0 엔트리, progress 92.5→66.7% — 장기 배포에서는 저장량이 아니라 **정확·최신 메모리 유지**가 병목.
- Transient Memory의 Object Search 0%는 inspect한 상자 내용을 전혀 유지 못하기 때문 — 메모리 없는 위계 VLA의 구조적 실패.
- Sentry 없는 고빈도 재계획은 일시적 시각 노이즈에 과민 → 서브태스크 간 불규칙 전환(erratic switching) 유발 (14→26% 개선이 이를 입증).

---

## 9. 관련 연구와의 비교

| 축 | MemER [5] | MemoryVLA / MAP-VLA | Hi-robot [8] | **HiMe** |
|----|-----------|--------------------|--------------|----------|
| 메모리 구조 | FIFO 큐 + 키프레임 | 정책 수준 메모리 뱅크 | 메모리 없음 (주기적 고수준) | 3계층 (transient/working/episodic) |
| 관리 | 수동 축적 | retrieval 중심 | — | 능동 Add/Update/Delete |
| 양식 | 시각 중심 | 시각/토큰 | — | 크로스모달 (이미지+텍스트 앵커) |
| 트리거 | 고정 빈도 | 매 스텝 | 고정 간격 | Sentry 이벤트 기반 게이팅 |

- 학습 기반(past-token prediction, SAM2Act)과 달리 **학습은 Executor에만** — 메모리·추론 계층은 프롬프트로 해결하여 데이터 효율적.

---

## 10. 강점

1. **원리적 문제 정의**: frequency-competence paradox를 시간 해상도 분리로 해소하는 명쾌한 아키텍처 — 각 계층이 인지과학의 multi-store 모델에 대응.
2. **메모리를 1급 객체로**: 수동 버퍼가 아닌 CRUD 가능한 동적 지식 시스템 — 자기 수정·선호 갱신 등 질적으로 새로운 능력.
3. **효율과 성능 동시 달성**: API 호출 3배 절감 + progress 65→90% — Sentry 게이팅의 이중 이득(일관성·관측 품질)을 분리 분석.
4. **꼼꼼한 ablation**: 양식(text/image/cross-modal), 관리(FIFO/No-mgmt/Active), 용량 통제(Table 9), Sentry 윈도우 크기, 오픈소스 Planner 치환까지 체계적.
5. **재현성 배려**: 코드 공개, Qwen3-VL-30B 대체 실험, 전체 프롬프트 공개(Appendix D), 학습 하이퍼파라미터 명시.

---

## 11. 한계 및 논의

1. **표준 벤치마크 부재**: LIBERO/CALVIN 등 시뮬레이션 벤치마크 비교 없음 (저자 인정) — 통제된 비교·재현성이 제한. 실로봇 20 trials × 3 tasks는 통계적으로 얇은 편.
2. **단순 프리미티브**: pick-and-place 중심 — dexterous 조작으로의 확장 미검증.
3. **확장성 열화**: 3라운드에서 이미 66.7%로 하락 (Table 10) — 매우 긴 horizon·대규모 환경에서의 특성 미규명.
4. **Sentry의 낮은 Recall (~35%)**: fallback으로 완화하지만, 완료 감지 실패 시 overshoot 위험은 잔존.
5. **외부 API 의존**: 본실험 Planner가 GPT-4o — 로컬 대체가 있지만 P50 38.6s의 API 지연은 실용 배포에 부담.
6. **비교 공정성**: Flat Memory가 MemER의 "유사(similar)" 재구현 — 원 구현과의 직접 비교는 아님.

---

## 12. 종합 평가

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 개별 요소(위계 정책, VLM 메모리, CRUD)는 기존하나, 3계층 시간 해상도 분해 + 크로스모달 능동 관리의 결합과 Sentry 게이팅 분석이 신선. |
| **Rigor** | ★★★★☆ — 풍부한 ablation과 효율 지표. 단 시뮬 벤치마크 부재·소규모 trial이 감점. |
| **Impact** | ★★★★☆ — 장기 non-Markovian 조작의 실용적 레시피; π0.5 등 어떤 VLA에도 model-agnostic하게 적용 가능. |
| **Reproducibility** | ★★★★☆ — 코드·프롬프트·오픈소스 Planner 경로 공개. 실로봇 셋업 의존은 불가피. |
| **Writing/Clarity** | ★★★★☆ — paradox 프레이밍이 명확, 그림·표가 주장을 잘 뒷받침. |

**강점**: 메모리를 "무엇을(크로스모달 스키마)·어떻게(능동 CRUD)" 기억할지의 두 축으로 체계화하고, Sentry 게이팅으로 성능·효율을 동시에 잡은 설계. **약점**: 표준 벤치마크 부재와 장기 확장성 열화. 평균 90%로 human high-level oracle(94%)에 근접 — 병목이 고수준 계획에서 저수준 실행으로 이동했음을 시사하는 의미 있는 결과.

<!-- VERIFIED: pdf -->
