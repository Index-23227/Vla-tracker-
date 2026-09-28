# ActSafeGuard — Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies

**arXiv**: 2609.11697 · **기관**: Shanghai Jiao Tong University, Shanghai Innovation Institute · **날짜**: 2026-09-10

## 1. 한 줄 요약
Flow-matching 액션 헤드의 각 이산 흐름 스텝에 파라미터 없는 레이 스케일링(ray-scaling) 연산자를 넣어 볼록 다면체 제약을 매 스텝 보장하고, 이 연산자를 통과하는 그래디언트로 π0.5/Fast-WAM을 end-to-end 미세조정해 100% 스텝 안전율과 동시에 과제 성공률(π0.5 기준 80.25/81.50)을 유지·향상한 방법.

## 2. 문제 설정
- VLA/WAM이 생성하는 행동 청크는 관절 한계·속도 한계·작업 공간 제약을 위반할 수 있고, 한 번의 위반도 하드웨어 손상으로 이어질 수 있다.
- 학습 측 방법(SafeVLA, SafeDojo)은 기대 비용·소프트 페널티라 결정론적 보장이 없다.
- 추론 측 방법(CBF-QP 투영, 마스킹, 절단)은 하드 제약을 보장하지만 학습에서 빠져 있어 학습–추론 불일치로 분포가 왜곡된다.
- 실제로 실행 가능한 데이터만으로 학습해도 무제약 π0.5의 평균 스텝 안전율(SSR)은 PosCons 51.42%, Fast-WAM은 22.99%에 불과(Table 2).

## 3. 핵심 아이디어
- 흐름 생성을 N-step 이산 과정으로 재정식화하고, 각 스텝 Δ = s·d, d = v/N, s = min(1, α).
- α = min_{i: a_i·d > 0} (b_i − a_i·x_k)/(a_i·d): 현재 상태에서 진행 방향으로 가장 가까운 제약 면까지의 레이 교차 거리. 넘어가면 정확히 경계까지만 이동.
- 초기 노이즈를 실행 가능 영역에서 샘플하면 전체 궤적이 영역 안에 머문다(Proposition 1).
- 경계에서의 국소 야코비안이 사선 투영 Q = I − d a^T/(a^T d)로 작동 → 바깥을 향하던 그래디언트가 경계를 따라 미끄러지는 방향으로 변환(boundary-sliding gradient).

## 4. 아키텍처
- π0.5: 공개 π0.5 백본의 VLM과 액션 전문가 전체를 미세조정, ActSafeGuard 연산자는 학습 파라미터 없음. H=50, N=10.
- Fast-WAM: Wan2.2-TI2V-5B 비디오 브랜치 + 30층 action-DiT(약 6.02B), 추론 시 action-DiT만 사용, N=20.
- 제약은 ALOHA-AgileX 14차원 행동 중 12개 팔 관절에만 적용, 그리퍼 2차원은 통과.

## 5. 학습 목표
- 이산 흐름 매칭 손실 L_DFM = ||Δ_φ − (x_1 − x_0)/N||²를 ActSafeGuard 적용 후의 Δ에 대해 계산(제약 차원만), 나머지 차원은 원 손실 유지.
- PosCons: 시연 통계 기반 관절 위치 박스. VelCons: 현재 고유수용 상태에 의존하는 스텝 간 변위 한계(상태 의존·시변 다면체).

## 6. 학습 절차
- RoboTwin 2.0 과제당 성공 궤적 300개 수집, 평가는 분리된 시드로 과제당 100 에피소드.
- π0.5: 50k step, 배치 32, 최대 lr 2.5e-5(워밍업 코사인), 4×H100. Fast-WAM: 200k step, lr 1e-4, 8×H200.
- 모든 기준선(Projection, Truncation, GaugeFlow)은 동일 백본 가중치·데이터 사용.

## 7. 주요 결과 (Table 1, RoboTwin 2.0 4과제, SR %)
| 백본 | 방법 | PosCons 평균 | PosCons+VelCons 평균 |
|---|---|---|---|
| π0.5 | 무제약 기준선 | 75.25 | 75.25 |
| π0.5 | Projection | 74.75 | 76.25 |
| π0.5 | Truncation | 77.25 | 76.75 |
| π0.5 | GaugeFlow | 75.00 | N/A |
| π0.5 | **ActSafeGuard** | **80.25** | **81.50** |
| Fast-WAM | 무제약 기준선 | 83.25 | 83.25 |
| Fast-WAM | Truncation | 79.50 | 81.75 |
| Fast-WAM | **ActSafeGuard** | **82.00** | **83.00** |
- π0.5 ActSafeGuard 과제별: PosCons lift pot 100, place shoe 90, hanging mug 34, place empty cup 97 / 동적 제약 100, 98, 31, 97.
- 모든 제약 방법은 SSR 100%(무제약은 51.42%/48.09%).
- MMD·LDLJ(부록 Table 8, 9)에서도 사후 보정보다 전문가 분포 정렬·부드러움을 더 잘 보존.

## 8. 어블레이션 — 미분 가능성 (Table 3)
- place shoe(동적 제약): ActSafeGuard 98.0 vs stop-gradient 3.0 — 경계 인지 그래디언트가 없으면 동적 제약 하에서 학습이 붕괴.
- place empty cup(정적): 97.0 vs 95.0, LDLJ −16.197 vs −16.991, MMD 0.00960 vs 0.01171.

## 9. 실제 로봇 (AgileX ALOHA, 과제당 5회)
- pick green cube(관절 한계 박스): 기준선 5/5, ActSafeGuard 5/5.
- guide the ball(비볼록 L자 EEF 통로): 기준선 4/5(통로 이탈), ActSafeGuard 5/5, 모든 스텝에서 제약 준수.
- 레이–경계 교차만 계산할 수 있으면 비볼록 영역에도 적용 가능함을 시연.

## 10. GaugeFlow 등과의 비교 포인트
- GaugeFlow는 정적 star-shaped 영역만 지원해 상태 의존 VelCons에 적용 불가(N/A).
- CBF-QP는 고차원 청크·다수 결합 부등식에서 반복 최적화가 필요하지만, ActSafeGuard는 닫힌형 레이 교차만 계산.
- ActSafeGuard는 원래 흐름 업데이트를 경계를 넘을 때만 수정하므로 사전학습 벡터장 사전지식을 덜 훼손.

## 11. 강점과 한계
**강점**
- 결정론적 하드 제약 보장과 학습 정렬을 동시에 달성, 추가 파라미터 0.
- 두 이질적 백본(VLA, WAM)과 정적/동적 제약 모두에서 검증.

**한계**
- 제약 집합을 사람이 명시해야 하며(시연 통계·하드웨어 사양), 인지 기반 자동 제약 발견은 미해결.
- RoboTwin 4과제의 비표준 제약 프로토콜이라 다른 VLA와의 직접 비교가 불가하고, hanging mug처럼 기저 성공률이 낮은 과제가 포함.
- 실로봇 평가가 과제당 5회로 통계적 힘이 약하다.

## 12. VLA-Tracker 관점 평가
"안전 레이어"라는 제목이지만 동결 정책 위의 게이트가 아니라, 연산자를 통과하는 손실로 π0.5(VLM+액션 전문가 전체)를 직접 미세조정해 자체 정책을 학습하므로 규칙상 ACCEPTED. 액션 생성은 이산화된 flow matching이므로 `action_head_category: flow_matching`. 평가는 표준 RoboTwin 2.0이 아니라 제약이 추가된 4과제 부분집합이므로 `robotwin_v2`에 넣지 않고 `robotwin_v2_constrained_poscons` / `_posvel`(π0.5 헤드라인) 및 Fast-WAM 변형·기준선·어블레이션·실로봇 블록으로 분리했다.

<!-- VERIFIED: pdf -->
