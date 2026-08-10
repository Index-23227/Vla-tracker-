# CoTinyVLA: Chain-of-Thought Distillation for a Sub-Billion-Parameter Vision-Language-Action Model

> **한 줄 요약**: 파라미터를 키우는 대신 **감독 신호를 구조화**해서 robustness를 산다. Qwen3.5-0.8B 위에 (a) 3인칭·손목 각 8프레임 = 16프레임 dual-view temporal input, (b) 35B teacher로부터 증류한 **episode-level `Plan` + chunk-level `Think`** 2단 CoT, (c) 40개 지시문 → 800개 paraphrase 증강을 얹어, **0.9B 모델이 LIBERO-Plus 4개 suite 전부에서 7B 최강 baseline(OpenVLA-OFT+)을 4.7~15.9점 앞선다.** 표준 LIBERO에서도 평균 97.5%로 RIPT-VLA(7B)와 동률, 추론 피크 메모리는 2.25 GiB.

- arXiv: 2607.25487 (v1, 2026-07-28, cs.AI)
- 저자: Minhyeok Lee(교신), Chiyoung Kim, Chanhoe Gu, Seongrok Kim, Sanghyuk Roy Choi, Donghwan Hwang, Donghun Ryu, Seokhyun Kim — 중앙대학교
- 코드: https://github.com/BrainJellyPie/CoTinyVLA

---

## 1. 배경 및 동기

- LIBERO-Plus(Fei et al., 2025) robustness 리더보드의 상위권은 전부 3~7B backbone이다: OpenVLA, OpenVLA-OFT, π0, π0-Fast, NORA, UniVLA, WorldVLA, RIPT-VLA, OpenVLA-OFT+.
- 그런데 7B급 VLA를 bfloat16으로 올리면 통상 20 GB 이상의 GPU 메모리가 필요하다. 모바일 매니퓰레이터·휴머노이드·보조 플랫폼이 쓰는 임베디드 가속기 예산으로는 감당이 안 된다.
- 이 격차가 논문의 출발점이다. 큰 모델의 강점이 "지각/기구학/언어 섭동에 대한 robustness"라면, **그 robustness는 파라미터에서 오는가, 아니면 학습 신호에서 오는가?**
- 저자들은 명시적으로 선을 긋는다: "컴팩트 VLA가 경쟁력 있다는 것은 이 논문의 *발견*이 아니라 *전제*다"(TinyVLA, SmolVLA, FLOWER, NORA가 이미 보였다). 진짜 질문은 **스케일 대신 무엇을 공급해야 하는가**이다.

---

## 2. 핵심 질문

> "표적화된 학습 신호가 컴팩트한 모델에게 대형 모델과 같은 robustness를 줄 수 있는가, 그리고 robustness 문제의 **어느 축**을 각 감독 형태가 담당하는가?"

두 번째 절반이 이 논문의 차별점이다. 단순히 "작아도 잘한다"가 아니라, LIBERO-Plus의 7개 섭동 축(Camera / Robot Initial States / Language / Light / Background / Noise / Layout)에 대해 **어떤 컴포넌트가 어떤 축을 담당하는지 분해**하겠다는 것이다.

---

## 3. 아키텍처 (§3.1, Fig. 1)

단일 vision-language-action 트랜스포머 + 입력단 proprioception projector + 출력단 action head. 4단 파이프라인.

**입력 (매 control step):**
- 3인칭 프레임 8장 `{I³ᵖ_{t-7..t}}`
- 손목 프레임 8장 `{I^wr_{t-7..t}}`
- 8차원 proprioception (end-effector pose + 그리퍼 관절 2개)
- 자연어 지시문 (학습 시 확률 0.8로 paraphrase 치환)

**토큰화:** 모든 이미지는 224×224로 인코딩되고, 각 이미지 앞에 `[Third frame i]` / `[Wrist frame i]` (i ∈ 1..8) 텍스트 마커가 붙는다. 시간·카메라 인덱스를 **토큰 스트림에 명시적으로** 넣는 inductive bias — 별도의 temporal positional embedding을 추가하지 않는다.

**Backbone:** Qwen3.5-0.8B. proprioception은 ~1M 파라미터 2-layer MLP로 hidden size에 투영되어 soft token 1개로 language stream 앞에 붙는다. 컨트롤 토큰 임베딩 + projector + head를 합쳐 총 ≈0.9B.

**Action head:** reasoning span 다음, 고정 query 위치의 최종 hidden state를 읽어 8-step 액션 청크 `a_{t:t+7} ∈ R^{8×7}`(EE delta 6 + gripper 1)를 **한 번의 forward로 회귀**한다. Action tokenization도 autoregressive 액션 디코딩도 없다 → action head 분류는 **regression**.

**왜 8프레임인가:** LIBERO의 20 Hz 제어에서 8프레임 ≈ 0.4초 = grasp onset 하나를 온전히 커버. 1프레임은 기구학 추론에 불충분, 더 길면 토큰이 폭증. 이미지 1장당 활성 메모리 ~32 MiB / forward 6.5 ms 추가에 그쳐, 16장 입력이 forward 피크 메모리를 1.76 → 2.20 GiB로 올릴 뿐이다.

---

## 4. Hierarchical Chain-of-Thought: `Plan` / `Think` (§3.2)

vocabulary에 `<plan>`, `</plan>`, `<think>`, `</think>` 4개 토큰을 추가하고, 액션 청크 **이전에** 두 층위의 reasoning을 autoregressive 생성한다.

| | 생성 빈도 | 내용 | 성격 |
|---|---|---|---|
| `Plan` | 에피소드당 **1회** (지시문 소비 직후) | 지시문을 순서 있는 subgoal 목록으로 분해 | **intent** — 에피소드 내내 불변 |
| `Think` | 액션 청크마다 | 고정 3-slot 스키마 | **state** — 매 청크 갱신 |

`Think` 스키마:
```
Phase:   <step k: short description>
Gripper: <OPEN | CLOSED | PARTIALLY_CLOSED>
Next:    <short action description>
```

- `Phase`의 어휘는 미리 고정되지 않고 **해당 에피소드의 `Plan`에 의해 한정**된다. `Gripper`는 닫힌 어휘라서 라벨 생성 시 proprioception으로 자동 검증 가능.
- free-form reasoning을 쓰지 않은 이유: 고정 슬롯이 span 길이를 유한하게 묶고, 감독을 **액션 정확도와 직결된 3개 변수**에 집중시킨다.
- **두 timescale 분리가 추론 비용에서 값을 한다.** `Plan`은 한 번 생성해 캐싱 가능 → steady-state 비용 반감. flat per-chunk trace에는 이런 절감이 없다.

---

## 5. Teacher와 증류 (§3.3)

- Teacher: **Qwen3.5-35B-A3B** (35B MoE VLM), 라벨 생성 시 4-bit 양자화.
- 데모 1개당 teacher 쿼리 **11회**: 첫 프레임+지시문으로 `Plan` 1개, 균등 간격 청크에서 `Think` 10개. 중간 청크는 가장 최근 라벨을 재사용해 모든 청크가 라벨을 갖는다.
- 라벨링의 두 가지 설계 포인트:
  1. `Think` 프롬프트가 해당 청크의 proprioception을 읽어 **그리퍼 힌트를 파생 제공**(손가락 간격 기반). 정지 프레임에서 열림/닫힘이 시각적으로 구분 안 되는 모호성을 회피 — `Gripper`를 이미지 추론이 아니라 **물리 상태**에 묶는다.
  2. **액션 청크 라벨은 teacher가 아니라 데모에서 직접** 가져온다. teacher는 reasoning trace만 공급.
- 즉 student는 **reasoning은 대형 VLM에서, action은 데모에서** 배운다.

**목적함수:** `L = α·L_act(â, a) + β·L_lm(ŷ_cot, y_cot)`, α=1.0, β=0.1. `L_act`는 8-step 7-DoF 청크에 대한 L1. 액션 목적이 지배적이도록 β를 낮게 뒀다.

---

## 6. Paraphrase 기반 지시문 증강 (§3.4)

- LIBERO-Plus 4개 학습 suite가 쓰는 base 지시문은 **단 40개**. 표면형이 좁아 소형 student가 과적합하기 쉽다.
- 각 지시문을 20개로 확장 → **800 variants**. 유형: 동사 치환(pick up → grab/lift/fetch/retrieve), 객체 동의어(black bowl → dark bowl / dark-coloured bowl; stove → stovetop/cooktop), 공손함 변형(direct / "could you" / "would you mind" / "I'd like").
- 학습 시 확률 **0.8**로 paraphrase 치환, 0.2로 원문 유지(평가 시 쓰는 reference wording에 대한 노출 보존). 평가에는 절대 사용하지 않는다.
- 효과는 **외과적**이다: 제거 시 Language Instructions 축이 12.3점 하락하고, 나머지 6개 비언어 축은 변화 없음.

---

## 7. 실험 설정 (§3.5, §4.1)

| 항목 | 값 |
|---|---|
| 학습 | LIBERO-Plus 4개 suite union, ≈2.4M action-chunk 샘플, **2 epoch** |
| Optimizer | AdamW, backbone lr 2e-5 / head lr 1e-4, 3% linear warmup |
| 기타 | gradient checkpointing, FlashAttention-2, train seed 42 / eval seed 7 |
| Compute | 8× H100 단일 노드, 2-epoch run당 ≈40시간 |
| Rollout cap | Spatial 220 / Object 280 / Goal 300 / Long 520 (OpenVLA-OFT 계열 관행) |
| LIBERO-Plus | 10,030 perturbed task, 7축 × 5난이도, task당 **1 rollout**(표준 프로토콜) |
| 표준 LIBERO | suite당 10 task × 50 trial = 500 trial |
| 추론 | `Plan` 1회/에피소드 + `Think`+청크 매 forward, 청크는 8 env step open-loop, greedy, 120-token cap |

Baseline은 LIBERO-Plus 논문에 보고된 11개 전부(전원 3B~7B).

---

## 8. 주요 결과 — LIBERO-Plus (Table 1, 2, 4, 5, 10, 11)

| Suite | CoTinyVLA (0.9B) | 최강 baseline (OpenVLA-OFT+, 7B) | 마진 (95% CI) |
|---|---|---|---|
| Spatial | **90.8** | 86.1 | +4.70 [+2.90, +6.50] |
| Object | **87.3** | 84.5 | +2.79 [+0.87, +4.71] |
| Goal | **86.6** | 70.7 | +15.87 [+13.68, +18.06] |
| Long | **80.7** | 77.7 | +3.01 [+0.77, +5.25] |

**네 구간 모두 0을 배제한다.** 그리고 이득이 몰리는 곳이 진단적이다.

- **Robot Initial States**(시작 자세에 기구학적 오프셋 주입)가 벤치마크에서 가장 어려운 축이다. 11개 baseline 중 어느 것도 어느 suite에서든 **53.2%를 넘지 못한다**(시각 축에서는 80~100%인데). CoTinyVLA는 Spatial 58.3%, Goal **73.6%** (최강 baseline 39.9% 대비 +33.7).
- Goal suite에서는 **7개 축 전부**에서 선두. Language Instructions도 83.7% vs 55.1%(+28.6).
- 시각 축은 4개 suite 모두 90% 이상, 3개 suite에서 96% 이상.
- 난이도 스케일링도 한 축이 독점한다. Spatial에서 Light/Background는 Lv1→Lv5 내내 93% 이상, Layout은 100→84%인데 **Robot Initial States는 94.4 → 7.1%**로 붕괴(Table 6).

---

## 9. 주요 결과 — 표준 LIBERO (Table 3)

| Model | Params | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| **CoTinyVLA** | **0.9B** | **99.4** | **100.0** | **98.6** | 92.0 | **97.5** |
| Evo-1 (최강 sub-1B) | 0.77B | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 |
| VLA-0-Smol | 0.5B | 92.2 | 97.2 | 95.6 | 91.2 | 94.1 |
| RIPT-VLA | 7B | 98.6 | 98.6 | 99.0 | 93.8 | 97.5 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0.5 | 3B | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |

- 평균 97.5%로 **7B 최강 baseline과 동률**, 최강 sub-1B(Evo-1 94.8%) 대비 +2.7점. Object에서 100.0%.
- 이 표의 역할은 "LIBERO-Plus 강건성을 in-distribution 성능 희생으로 산 게 아니다"의 증명이다.
- 유일한 약점: Long 92.0%로 OpenVLA-OFT(94.5)·RIPT-VLA(93.8)·Evo-1(92.3)에 뒤진다.

---

## 10. Ablation과 test-time intervention (Table 12, 15, 16)

Ablation은 예산 때문에 **1 epoch**로 학습(reference 87.6% vs 본 모델 90.8%) — 행 간 비교는 matched.

| Variant | Total | Δ | 어느 축이 무너지나 |
|---|---|---|---|
| Full (1-epoch ref) | 87.6 | – | – |
| **손목 뷰만** | 78.1 | **−9.5** | 전 축 |
| **단일 프레임, dual-view** | 82.9 | −4.7 | Robot 67.1→51.1 |
| 3인칭 뷰만 | 84.1 | −3.5 | Camera, Robot |
| **CoT 증류 전체 제거** | 84.2 | −3.4 | Robot 67.1→56.3, Layout |
| `Think`만 (`Plan` 제거) | 86.3 | −1.3 | Robot |
| `Plan`만 (`Think` 제거) | 85.4 | −2.2 | Robot |
| **Paraphrase 제거** | 85.6 | −2.0 | **Lang 86.9→74.6만** |
| 4프레임/뷰 | 86.7 | −0.9 | 미미 |

**핵심은 분리 가능성(separability)이다.** paraphrase는 언어 축만, CoT는 물리-상태 축(Robot/Layout)만, temporal history는 기구학 축만 건드린다. 그리고 고정 이미지 예산에서 "카메라 간·시간 간 배분" 자체가 8.6점(78.1 → 86.7 범위)을 좌우한다 — 입력 구조는 아키텍처가 지불하는 고정비가 아니라 **설계 변수**라는 것.

**Test-time intervention** (가중치 고정, paired, exact test):
- `Plan`을 빈 span으로 교체 → pooled 성공률 100% → **60%**; 다른 task의 `Plan`으로 교체 → **55%** (둘 다 p < 0.001). Goal(40%/35%)이 Spatial(80%/75%)보다 훨씬 크게 무너진다 — Goal은 subgoal 순서가 곧 과제이고 Spatial은 지시문만으로 intent가 복원되므로 정확히 예측대로다.
- 액션 자체도 움직인다: median L2 0.73~0.85, cosine 0.96~0.98. → `Plan`은 붙여놓은 주석이 아니라 **action head가 소비하는 표현에 실제로 들어간다.**
- reasoning span을 통째로 제거 → 성공률 **0%** (latency는 1.37s → 0.29s). 액션 예측이 reasoning-conditioned readout에 의존한다.
- history를 최근 프레임 반복으로 대체(토큰 수 불변) → 표준 LIBERO 100% → **60%**, Spatial regression rate 0.8% → 15.2%.

---

## 11. 추론 비용 (Table 13, 14, 15)

- Closed-loop(L40S, 3 rollout이 GPU 공유): steady-state **1.37 s/청크**, 초기 청크 2.76 s. 8 env step을 덮으므로 step당 reasoning 생성은 per-step CoT 대비 1/8.
- 시간의 **76%가 autoregressive 생성**(초기 청크는 88%). vision forward 250 ms, projector·head는 각 0.5 ms. → 병목은 아키텍처가 아니라 **디코딩**이고, 가중치를 건드리지 않는 표준 기법이 그대로 적용된다.
- `Plan` 캐싱: 2.65 s → 1.37 s (−48.5%), 생성 토큰 70.3 → 25.9 (−63.2%), 성공률 100% 불변.
- 토큰 예산 sweep: 48 → 28.0% truncation·SR 90%, 64 → 6.0%·95%, **96과 120은 구분 불가**. 정책이 닫는 태그에서 스스로 멈추므로 latency는 네 예산 모두 1.37 s로 평평 — 예산을 줄여봐야 얻는 게 없고 truncation 위험만 산다. 96이 최소 안전 예산.
- 메모리: allocated peak **2.25 GiB** (max 2.27), reserved ≤2.30 GiB. 생성 KV 캐시는 가중치 대비 0.05 GiB 증가에 불과.
- 표준 LIBERO 20 에피소드×4 suite closed-loop: 79/80 성공(98.75%), malformed·truncated 청크 **0%**.

---

## 12. 총평

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★☆☆ — 구성 요소는 대부분 기존 것이다. CoT rationale 증류(Hsieh 2023, Ho 2023), 로봇 CoT(ECoT), multi-frame 컴팩트 VLA(CronusVLA), 텍스트 증강(EDA, ParaBank). 새로운 것은 **2-timescale 스키마**(`Plan`=intent/캐시 가능, `Think`=state/고정 3슬롯)와, 각 컴포넌트를 섭동 축에 사상하는 **문제 분해 방식**이다. 저자도 "컴팩트 VLA의 경쟁력은 전제이지 발견이 아니다"라고 스스로 못 박는다. |
| **Rigor** | ★★★★★ — 이 논문의 진짜 강점. Wilson CI와 difference interval을 전 suite·전 baseline에 대해 보고하고, 재구성된 count의 반올림 모호성까지 명시한다. 9-run ablation을 1 epoch로 내린 이유와 그 대가(87.6 vs 90.8)를 숨기지 않는다. 무엇보다 **retraining ablation과 test-time intervention을 분리**해서, 후자는 동일 가중치·동일 에피소드의 paired exact test로 만들었다. Long의 Language 축 68.9% vs 94.8%라는 자기 최악 수치도 Discussion에 직접 적는다. |
| **Practical impact** | ★★★★☆ — 2.25 GiB / 0.9B는 임베디드 배포 가능 영역이다. 다만 1.37 s/청크는 실시간 제어로는 느리고, 저자도 이를 "디코딩 문제"로 규정하며 미해결로 남긴다. `Plan` 캐싱이 절반을 이미 회수했다는 점, 그리고 생성/실행 오버랩이 미평가라는 점이 남은 여지. |

**핵심 메시지**: "LIBERO-Plus에서 3~7B backbone에 귀속되던 robustness는 backbone을 요구하지 않는다 — 벤치마크가 섭동하는 축을 따라 감독을 구조화하면 0.9B로 충분하다."

가장 설득력 있는 증거는 순위표가 아니라 **Table 12의 축별 분리**다. paraphrase를 빼면 언어 축만 12.3점 떨어지고 나머지 6축은 그대로다. 이건 "여러 트릭을 쌓았더니 좋아졌다"가 아니라 **각 감독 형태가 어떤 실패 모드를 고치는지 인과적으로 지목**한 것이다. 커뮤니티에 던지는 질문은 "0.9B가 7B를 이겼는가"가 아니라, **"LIBERO-Plus가 측정하는 robustness 중 실제로 스케일을 요구하는 부분은 얼마인가"**이다.

---

### 한계 (저자 명시, §6)

1. 시뮬레이션 단일 embodiment(LIBERO Franka Panda)만. sim-to-real, cross-embodiment 미검증. 뷰당 8프레임을 넘는 history도 미검증.
2. 추론은 단일 L40S 동기 구현 기준. 생성/실행 오버랩과 실물 로봇 end-to-end latency 미평가.
3. 증류가 **강한 teacher에 의존**한다. 저렴한 teacher는 특히 구조화된 `Think` 속성의 라벨 품질을 떨어뜨릴 수 있다.
4. paraphrase는 어휘 통계로만 특성화했고, base 지시문과의 **의미 동등성은 별도 평가 필요**.
5. Long suite가 절대값 최저(80.7%)이며, 그 Language Instructions 축(68.9% vs 최강 baseline 94.8%)이 벤치마크 전체에서 가장 큰 축별 열위다. 장기 지평 실행과 지시문 변형이 겹치면 컴팩트 backbone + 8프레임 history는 여전히 불리하다.
6. 학습 seed 간 분산은 미측정(평가 분산은 특성화됨). 다른 multimodal backbone 계열도 future work.

### 예상 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | reasoning 토큰이 정말 액션에 쓰이나, 아니면 그냥 auxiliary loss인가? | 쓰인다. span 제거 시 성공률 0%, `Plan` 오염 시 100→55%, 그리고 액션 청크가 median L2 0.73~0.85만큼 이동한다(Table 16). action head는 span **다음** 위치의 hidden state를 읽으므로 구조적으로도 경로 위에 있다. |
| 2 | LIBERO-Plus로 학습하고 LIBERO-Plus로 평가하면 자기 벤치마크 과적합 아닌가? | 모든 baseline도 동일 프로토콜이고, Table 3의 표준 LIBERO(97.5%)가 in-distribution 능력 유지를 보인다. 다만 섭동 유형 자체가 학습·평가에 공유된다는 점은 남는 우려이며, 미지 섭동 유형에 대한 일반화는 측정되지 않았다. |
| 3 | 왜 free-form CoT가 아니라 3-slot 고정 스키마인가? | span 길이를 유한하게 묶어 토큰 예산(96)을 확보하고, 감독을 액션 정확도와 직결된 3변수에 집중시키기 위해. 부수 효과로 `Gripper`가 proprioception으로 자동 검증 가능해지고, closed-loop 1,552 청크에서 malformed 0%가 나온다. |
| 4 | 0.9B가 7B를 이겼다면 스케일이 무의미한가? | 아니다. 저자의 주장은 "이 벤치마크가 측정하는 robustness에 한해"다. 게다가 라벨을 만든 teacher가 35B이므로 **스케일은 학습 파이프라인에 여전히 존재**하고, 추론 시점에서만 제거됐다. |
| 5 | Robot Initial States는 왜 그렇게 어렵고 왜 여기서 가장 크게 이겼나? | 기구학 오프셋 하에서 그리퍼와 대상의 상대 위치를 파악하고 궤적을 재계획해야 한다. 정적 프레임 1장으로는 접근 속도·방향을 알 수 없다 — dual-view 16프레임이 정확히 이 정보를 준다. 단일 프레임 대체 시 이 축이 16.0점 빠지는 것이 직접 증거. |
| 6 | 1.37 s/청크는 실배포에 쓸 수 있나? | 8 env step을 덮으므로 실효 제어율은 그보다 낫지만, 76%가 디코딩이라 여전히 느리다. 저자는 이를 아키텍처가 아닌 디코딩 문제로 규정했고 `Plan` 캐싱으로 이미 절반을 회수했다. speculative decoding·생성/실행 오버랩이 남은 카드. |

<!-- VERIFIED: pdf -->
