# Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation

**arXiv**: [2607.05377](https://arxiv.org/abs/2607.05377) · **기관**: Tsinghua University, Shanghai AI Laboratory (+ PKU, USTC) · **공개일**: 2026-07-06

---

## 1. 한 줄 요약

Cortex는 고수준 VLM 플래너(System-2)와 저수준 VLA 실행기(System-1)를 "실행 가능성(executability)과 추적 가능성(tractability)"이 보장된 서브태스크 인터페이스로 양방향 정렬한 듀얼 시스템 에이전트 프레임워크로, LIBERO-Long 95.5%(+3.1), RoboTwin 2.0 86.8%(+4.1)로 단일(monolithic) VLA 대비 장기 과업 성능을 크게 끌어올렸다.

## 2. 문제 설정 및 동기

단일 VLA는 마르코프적 근시안(Markovian short-sightedness) 때문에 현재 관측만으로 행동을 생성하며, 장기 과업에서 진행 상태를 놓치고 같은 동작을 반복한다. 기존 계층형 듀얼 시스템도 (1) VLM의 계획이 로봇의 운동학적 제약을 무시하는 semantic-kinematic gap, (2) 시각적으로 모호한 서브태스크 전환 시점의 temporal ambiguity라는 두 가지 정렬 실패를 겪는다. Cortex는 "VLM은 VLA의 능력을 고려해 계획하고, VLA는 VLM 출력에 강건해야 한다"는 양방향 정렬을 프레임워크 수준에서 강제한다.

## 3. 핵심 기여

1. **서브태스크 인터페이스 표준화**: 조작 서브태스크를 32개 정규 스킬 프리미티브(Pick, Place, Pour, Unscrew, Handover 등)와 엄격한 언어 템플릿으로 표준화.
2. **대규모 메타데이터 파이프라인**: 4,000시간 이상의 오픈소스 비디오(AgibotWorld, Galaxea, BEHAVIOR-1K, RoboCerebra)를 자동 재주석하고, RoboTwin/RMBench에서 30시간의 절차적 시뮬레이션 데이터를 생성. 객체 속성·공간 식별자·도달성(reachability)까지 주석에 주입.
3. **Event-balanced sampling**: 서브태스크 경계 주변에 비대칭 시간 마진(ε1 pre / ε2 post, ε2 > ε1, 전환 구간 약 1초)을 두고 경계 전환 프레임을 밀도 있게 샘플링하여 전환 모호성 해소를 학습.
4. **Harness engineering**: 추론 시 스킬 제약 프롬프트, 시퀀스 매칭 기반 표준 명령 매핑, 비정상 전환 필터링, 타임아웃 기반 kinematic reset을 포함한 경량 중재 계층.

## 4. 방법론

- **구조**: System-2(Qwen3-VL-8B-Instruct 파인튜닝)가 지시문 + 관측 + 텍스트 메모리 M(t)를 입력받아 현재 스킬/서브태스크/갱신된 메모리를 JSON으로 출력(~2Hz). System-1(π0.5 파인튜닝, 3.62B)이 서브태스크 조건부로 액션 청크를 생성(~10Hz). 두 시스템은 비동기로 동작.
- **메모리 형식화**: M(t) = M(0) ⊕ Σ Φ(s_i) — 완료된 서브태스크의 의미적 요약을 누적하는 텍스트 메모리. 시각 버퍼가 아닌 행동 지향적 상태 추상화.
- **자동 경계 추론**: 상태-행동 특징 φs와 시각 특징 φv를 융합한 x_t에 대해 프레임-서브태스크 호환 비용(마할라노비스 거리), 지속시간 사전, 저모션 경계 페널티를 동적 계획법으로 최소화하여 단조 경계 시퀀스를 추론(Appendix A.1).
- **상태 이력 텍스트화**: 최근 30스텝 로봇 상태를 tanh 정규화 후 256-bin 정수 토큰으로 양자화해 프롬프트에 주입 — 버튼 누름 횟수 같은 고주파 진행 신호를 언어 백본이 직접 파싱(RMBench press_button에 핵심).

## 5. 실험 설정

- **Open-loop VLM 평가**: LLM-as-a-Judge(Qwen-3.5-9B)로 Spatial / Long-horizon / Counting 3개 축, step-level(teacher-forced) 및 episode-level(self-forced) 평가. 버킷당 5개 태스크, 총 15개 rollout 태스크, step-level 버킷당 약 1,000 샘플.
- **Closed-loop 시뮬레이션**: LIBERO-Long(모든 agentic 베이스라인이 π0.5를 공유 실행기로 사용, zero-shot 원시 지시문 조건), RoboTwin 2.0(50 clean + 500 randomized 데모의 data-scaling 설정), RMBench(7개 메모리 의존 태스크, 태스크당 100 rollout).
- **실기 로봇**: ARX ACONE 듀얼암, MEM 스타일 π_mem^sub 실행기(약 10시간 자동 분할 데이터로 파인튜닝), 화학 실험·세척 등 14-스텝 장기 과업, 태스크당 20회 시행.

## 6. 주요 결과

- **LIBERO-Long (Table 2)**: Cortex 95.5% — π0.5 92.4%, MemoryVLA 93.4%, OpenVLA-OFT 94.5%, Gemini-3.1-Pro(agentic) 91.0%, GPT-5.4 72.0%를 모두 상회. 동일 실행기 조건이므로 이득은 순수하게 고수준 계획에서 온다.
- **RoboTwin 2.0 (Table 8)**: short 86.0 / long 88.0 / overall 86.8% — π0.5(82.74%) 대비 +4.1, X-VLA 72.8%, DP3 55.2%. 특히 장기 분할에서 88.0%로, 단일 VLA들이 장기에서 무너지는 것과 대조적.
- **RMBench (Table 9)**: rearrange_blocks/put_back_block 100%, swap_blocks 99%, press_button 20%(타 모델 전부 0%) — 메모리 의존 태스크에서 압도적.
- **Open-loop VLM (Table 1)**: full harness 구성이 step-level 8.318, episode-level 7.810으로 GPT-5, Gemini, Qwen3-VL-8B 능가.
- **실기 zero-shot (Table 3)**: 14-스텝 화학 실험 SR 65%(progress 11.0/14), 세척 55% — end-to-end π0.5/π_mem은 모두 SR 0%. 인간+VLA(75%/70%)에 근접.

## 7. Ablation 분석

- **Harness 단계별 (Table 1)**: baseline(7.051) → w/o harness(7.213) → harness on skills(7.392) → full harness(8.318, step-level 평균) — 각 구성요소가 일관된 이득.
- **Event-balanced sampling (Table 5, Galaxea leave-episode-0-out)**: intra:boundary 비율을 3.77:1 → 2.23:1로 조정하면 총 샘플이 3.10M → 2.72M으로 줄어도 Avg. Total 7.58 → 8.18로 상승. "경계 근처 고충실도 감독이 정체 상태 관측 누적보다 중요하다"는 핵심 가설을 데이터 효율성 측면에서 입증.
- **세부 사례**: 속성/공간 grounding으로 place_object_basket 80%→85%, reachability 인지 handover 삽입으로 dump_bin_bigbin 92%→98%.

## 8. 이전 연구와의 비교

SayCan·Inner Monologue류 초기 플래너는 신체화 제약 없는 "관찰자"로서 운동학적으로 근거 없는 지시를 내렸고, 잠재공간 결합 듀얼 시스템(LCB, HiRT)은 해석 가능성과 진행 추적을 희생했다. MemoryVLA·MemER 등 메모리 증강 VLA는 계획 정렬 측면이 부족했다. Cortex는 (1) 스킬 어휘로 VLM 출력 공간을 제약하고 (2) 도달성 등 물리 원칙을 데이터 생성에 주입하며 (3) 텍스트 메모리로 명시적 진행 추적을 유지한다는 점에서 세 흐름을 통합한다. LIBERO-Long에서 MemoryVLA(93.4%)를 2.1%p 상회.

## 9. 강점

- 동일 실행기(π0.5) 통제 하의 비교로 고수준 계획의 기여를 깔끔하게 분리한 실험 설계.
- 주석 없는(annotation-free) 경계 추론 + 절차적 시뮬레이션 생성으로 4k+ 시간 규모의 메타데이터를 확보한 데이터 엔진의 확장성.
- End-to-end가 SR 0%인 14-스텝 실세계 화학 실험을 zero-shot 65%로 완수 — 재시도(stopper 3회 파지), 인간 개입 섭동(미리 열린 병뚜껑) 대응 등 폐루프 오류 복구의 질적 증거가 풍부.
- 상태 이력 텍스트화라는 모델 불가지론적(model-agnostic) 고주파 신호 주입 기법.

## 10. 약점 및 한계

- 텍스트 메모리는 공간 좌표와 시각적 뉘앙스를 버리므로 대규모 이동 조작에서 객체 인스턴스 대응이 끊길 수 있음(저자 인정).
- 표준 비전 인코더 기반이라 고주파 미시 상태 변화에 둔감 — 상태 텍스트화는 우회책이며 동적 환경에서는 한계.
- RoboTwin 평가에서 평가자 측 로컬 스케줄러(매칭 신뢰도·dwell-time 제약)가 System-2 예측을 필터링하는데, 이 장치 자체의 기여가 별도로 정량화되지 않음.
- LIBERO는 Long 스위트만 평가(Spatial/Object/Goal 미보고), agentic 베이스라인의 프롬프트 튜닝 공정성은 검증 어려움. 코드 미공개(프로젝트 페이지만 존재).

## 11. 향후 연구 방향

저자들은 시각 메모리 검색과 픽셀 수준 grounding을 통합한 듀얼 모드 메모리, 그리고 고유수용성(proprioceptive) 상태 토큰화의 시각 융합을 제시한다. 그 외에도 32개 스킬 어휘의 자동 확장, 평가자 측 스케줄러의 학습 기반 대체, 모바일 조작으로의 확장이 자연스러운 다음 단계다.

## 12. 세미나 토론 질문

1. 32개 스킬 프리미티브라는 고정 어휘는 표현력과 안정성의 트레이드오프인데, 도구 사용·변형체 조작처럼 어휘 밖 행동이 필요한 과업에서 이 인터페이스는 어떻게 확장되어야 하는가?
2. Event-balanced sampling의 이득(Table 5)이 데이터 볼륨 감소에도 유지되었는데, 경계 샘플 비율을 24%보다 더 높이면 어느 지점에서 "semantic patience" 학습이 무너지는가?
3. RoboTwin 폐루프 평가의 로컬 스케줄러(신뢰도·dwell-time 필터)를 제거하면 성능이 얼마나 떨어질까 — 이것은 방법의 일부인가, 평가 하네스의 일부인가?
4. 텍스트 메모리 vs 시각 버퍼(MemoryVLA류): 실패 복구 사례(stopper 재파지)에서 텍스트 메모리가 유리했던 이유를 시각 메모리로 재현할 수 있는가?

---

*이 리뷰는 arXiv 2607.05377 PDF 전문(본문 + Appendix A.1–A.5)을 직접 읽고 작성되었으며, 모든 수치는 논문 Table 1–9 및 본문에서 검증되었다.*

<!-- VERIFIED: pdf -->
