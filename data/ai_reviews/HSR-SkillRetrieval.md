# Hierarchical Skill Retrieval for Data-Efficient Adaptation of Vision-Language-Action Models (HSR)

> **한 줄 요약**: 목표 과제를 LLM으로 스킬 시퀀스로 분해하고, 사전 데이터셋의 스킬 신뢰도(held-out BC loss 기반)로 계획을 골라 스킬 단위 언어 검색 + VAE 행동 특징 재순위화로 시연을 가져온 뒤, SmolVLA의 action expert를 2단계(검색 데이터 사전학습 → 목표 데이터 미세조정)로 적응시키는 프레임워크. LIBERO-10 5-shot 평균 43.5%(최강 기준선 BR 33.2%), 실물 xArm 48/80(IWR 31/80).

- **arXiv**: 2608.24042 (2026-08-25, cs.RO)
- **소속**: Carnegie Mellon University, Robotics Institute — Haoran Hao, Shahram Najam Syed, Jeff Schneider, Jeffrey Ichnowski
- **프로젝트/코드**: hoar012.github.io/HSR-Project
- **백본**: SmolVLA-450M (action expert ~100M만 학습)

---

## 1. 배경 및 동기

대규모 로봇 데이터로 사전학습한 VLA도 목표 과제 시연이 적으면 적응 성능이 떨어진다. 사전 데이터셋(Open X-Embodiment, DROID, LIBERO-90 등)에서 관련 시연을 검색해 학습 데이터를 늘리는 retrieval 기반 적응이 대안이지만, 기존 방법(Behavior Retrieval, IWR, Flow Retrieval, STRAP)은 시각·운동 유사도나 과제 전체 지시문 매칭에 의존한다. "차 한 잔 만들기" 같은 장기 과제는 전체가 일치하는 시연은 드물지만 "컵 집기", "기계 켜기" 같은 하위 스킬은 풍부하다 — 이 계층 구조를 검색에 활용하자는 것이 출발점이다.

## 2. 문제 정의

- 사전학습된 VLA π_θ(o_t, T) → a_t, 목표 과제당 소량의 전문가 시연 D_t, 대규모 사전 데이터 D_prior.
- 목표: D_prior의 부분집합 D_r을 골라 D_t ∪ D_r로 적응했을 때 목표 과제 성능을 최대화.

## 3. 방법

1. **스킬 클러스터링**: D_prior의 지시문 텍스트 임베딩을 K-means(K=10)로 묶어 Pick, Open, Pour 등 스킬 집합 S 구성.
2. **스킬 신뢰도(Skill Confidence Score)**: D_prior를 train/val로 나눠 VLA를 미세조정한 뒤 스킬별 val BC loss L_val과 평균 궤적 길이 H를 min-max 정규화해 C(s) = exp(−α·L̃_val − β·H̃), α=0.6, β=0.8. 롤아웃 없이 쓰는 신뢰도 대리 지표(DataMIL 방식).
3. **계획 선택**: LLM(Qwen3-VL-4B)이 스킬 집합을 참고해 10개 후보 분해를 생성하고 각 계획의 의미 점수 S_sem을 부여. Score(p) = S_sem(p) · Π C(s(u_j))로 최고 계획 선택.
4. **Hybrid 검색**: 서브태스크별로 언어 유사도 상위 M% 에피소드를 가져온 뒤, D_prior로 학습한 VAE 행동 특징에서 목표 프레임과의 최근접 거리로 프레임을 재순위화해 상위 F%만 유지.
5. **2단계 적응**: (i) 언어 검색 데이터 D_lang으로 사전학습해 재사용 스킬 습득, (ii) D_t ∪ D_rerank로 미세조정. 기존 단일 단계 co-training과 대비.

## 4. 학습 절차

- SmolVLA-450M의 action expert(약 100M)만 업데이트, 단일 RTX 5090.
- LIBERO: 1단계 상위 10% 에피소드 검색, 2단계 재순위화 후 30% 유지(사전 데이터 프레임의 약 3%).
- 실물: 1단계에서 재순위화된 상위 3% DROID 프레임으로 스킬 사전학습, 2단계는 cross-embodiment 격차 때문에 목표 데이터만으로 미세조정.

## 5. 데이터

- **LIBERO**: 목표 = LIBERO-10(Long) 10개 과제 × 과제당 5 demo, 사전 = LIBERO-90(90과제 × 50 demo).
- **실물**: 7-DoF xArm, 과제당 20 demo, 사전 = DROID에서 무작위 10k 궤적. 과제는 Drawer(1 서브태스크), Trashcan(2), Cup-Drawer(3), Cup-Tea(4).

## 6. 실험 설정

- 기준선: BC(목표만), Random(사전 10% 무작위), All Data, Language 검색, Flow Retrieval, STRAP, Behavior Retrieval(BR), Importance Weighted Retrieval(IWR).
- LIBERO는 3 시드 × 50 에피소드, 실물은 과제당 20 trial(모델 간 동일 초기 상태 집합).

## 7. 주요 결과

**Table I – LIBERO-10, 5-shot (%)**

| Method | Book | Moka-M | Mug-P | Bowl-C | Mug-M | Avg |
|---|---|---|---|---|---|---|
| BC | 20.7 | 0.0 | 12.7 | 77.3 | 24.7 | 22.7 |
| STRAP | 68.0 | 6.7 | 16.7 | 58.0 | 40.0 | 31.9 |
| BR | **95.3** | 3.3 | 8.7 | 50.7 | 29.3 | 33.2 |
| IWR | 88.7 | 2.7 | 11.3 | 54.7 | 27.3 | 32.5 |
| **HSR** | 90.7 | **15.3** | **26.0** | **81.3** | **67.3** | **43.5** |

단일 pick-and-place 과제에서는 BR과 비슷하지만 복합 과제에서 격차가 크다(Mug-M에서 STRAP 대비 +27.3). Bowl-C는 HSR만 BC(77.3)를 넘는다.

**Table II – 실물 xArm (성공/20)**

| Task (#subtask) | BC | BR | IWR | HSR |
|---|---|---|---|---|
| Drawer (1) | 12 | 15 | 16 | **18** |
| Trashcan (2) | 7 | 3 | 4 | **8** |
| Cup-Drawer (3) | 5 | 4 | 7 | **13** |
| Cup-Tea (4) | 3 | 5 | 4 | **9** |
| Overall | 27/80 | 27/80 | 31/80 | **48/80** |

**Table III – 분해 전략 절제 (LIBERO-10 평균)**: LLM Only 37.1, LLM+Skill Set 37.7, HSR 43.5, Human 계획 44.9 → 계획 평가가 사람 수준에 근접.

**Fig. 5 절제**: 과제 분해 제거, 특징 재순위화 제거, 단일 단계 co-training 모두 성능 저하; BR + 2단계 적응도 HSR보다 낮다(수치는 그림에만 제시).

## 8. Related Work 상의 위치

- 상태-행동 잠재 유사도 검색(Behavior Retrieval, IWR), 광학 흐름(Flow Retrieval), 서브궤적 DTW(STRAP) 등 시각/운동 기반 검색과 달리 **언어·스킬 구조 기반** 검색.
- SayCan의 "LLM 계획을 로봇 능력으로 grounding" 아이디어를 **학습 시점 데이터 검색 인터페이스**로 옮김(테스트 시 계획 실행이 아님).
- DataMIL의 BC loss 대리 지표를 스킬 신뢰도 추정에 차용.

## 9. 강점

1. **장기 과제에서 효과가 뚜렷**: 서브태스크 수가 많을수록 향상폭이 커진다(Cup-Drawer 7→13, Cup-Tea 4~5→9).
2. 롤아웃 없는 스킬 신뢰도 추정으로 계획 선택 비용이 낮다.
3. 기준선 범위가 넓고(검색 방식 5종 + naive mixing), 3 시드 표준편차 보고.
4. 2단계 적응이 cross-embodiment 사전 데이터(DROID → xArm)에서도 작동.
5. 소형 모델(SmolVLA 450M, action expert만 학습)과 단일 GPU로 재현 부담이 작다.

## 10. 약점 및 한계

1. **표준 LIBERO 프로토콜이 아님**: 5-shot, LIBERO-10만 평가하므로 일반 LIBERO 리더보드 수치와 비교 불가. 절대 성능(43.5%)도 낮다.
2. **단일 백본**: SmolVLA 하나에서만 검증(저자도 한계로 명시).
3. **표준편차가 큼**: Mug-M 67.3±18.1, Bowl-C 81.3±11.0 등 분산이 커서 개별 과제 차이의 신뢰도가 제한적.
4. 주요 절제(Fig. 5, 6)가 그림으로만 제시되어 정량 비교가 어렵다.
5. 실물 평가는 과제당 20 trial, 기준선 수가 적다(BC/BR/IWR).
6. 검색 파이프라인 구성요소(LLM, BERT, VAE, 신뢰도 추정용 추가 미세조정)가 많아 전처리 비용이 존재.

## 11. 재현 및 확장 아이디어

- π0/π0.5, OpenVLA-OFT 등 다른 백본과 더 큰 목표 데이터(10/20/50 demo)에서 향상이 유지되는지 확인.
- 스킬 클러스터 K와 α, β 민감도 분석.
- 검색된 스킬 데이터를 테스트 시 온라인 재계획·실패 복구와 연계.
- 인간 비디오나 이종 embodiment 데이터를 스킬 단위로 검색하는 확장.

## 12. 총평

HSR은 "검색 단위를 과제에서 스킬로 내린다"는 단순하지만 설득력 있는 아이디어로, 소량 시연 VLA 적응에서 장기 복합 과제의 성능을 크게 올린다. 모델 구조 자체는 SmolVLA 그대로이고 기여는 데이터 선택·적응 절차에 있으며, 평가는 비표준 few-shot LIBERO-10과 소규모 실물 실험에 한정된다.

**한 문장 요약**: LLM 과제 분해 + 스킬 신뢰도 기반 계획 선택 + 하이브리드 검색 + 2단계 적응으로 SmolVLA를 5-shot LIBERO-10 43.5%(BR 33.2%), 실물 48/80(IWR 31/80)까지 끌어올린 데이터 효율 적응 기법.

<!-- VERIFIED: pdf -->
