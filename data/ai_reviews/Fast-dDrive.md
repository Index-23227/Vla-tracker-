# Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving

> **한 줄 요약**: Qwen2.5-VL-3B 기반 자율주행 VLA를, **JSON-구조 출력 위에 section-aligned block diffusion**으로 디코딩. 같은 semantic unit(block) 안에서는 bidirectional refinement, block 간에는 strict causal ordering을 강제. **Scaffold Speculative Decoding**과 **test-time stochastic trajectory rollout (shared KV-cache)** 으로 nuScenes L2 0.32 m, WOD-E2E RFS 7.823 달성. AR baseline 대비 단일 H100에서 **4.1× latency 감소, SGLang 통합 시 11.8× throughput**.

---

## 1. 배경 및 동기

### 자율주행 VLA의 두 갈래
1. **AR (autoregressive)**: GPT-style token-by-token 생성 → 정확하지만 sequence가 길어 latency 큼
2. **Diffusion VLA**: 병렬 denoising으로 빠르지만 strict causal ordering 어려움 (e.g., perception → reasoning → trajectory 순서 위반)

### 핵심 관찰
- 자율주행 출력은 **JSON 형태로 구조화** 가능: `{perception: {...}, reasoning: {...}, trajectory: [...]}`
- **JSON section은 서로 dependent (causal)** 하지만 **section 내부 token은 거의 independent (parallel-friendly)**
- 이 구조적 사전지식을 디코딩에 직접 인코딩하면 AR의 정확성 + diffusion의 속도를 동시에

### 핵심 질문
1. **Block 단위 bidirectional refinement + block-간 causal AR**의 하이브리드 schedule을 어떻게 안정적으로 학습하는가?
2. **Stochastic trajectory ensemble**을 GPU 한 장에서 cost-free에 가깝게 돌릴 수 있는가?

📌 [Figure 1 삽입] — Fast-dDrive 디코딩 schedule: perception(block 1) → reasoning(block 2) → trajectory(block 3), 각 block 내부는 MDM denoising 단계

---

## 2. 방법론 심층 분석

### 2.1 Block Diffusion + Causal Ordering

출력은 사전 정의된 section list $B = [b_1, b_2, \dots, b_n]$:
- $b_1$ = scene perception (objects, lanes, signals)
- $b_2$ = chain-of-thought reasoning
- $b_3$ = action/trajectory waypoints

각 block은 **Masked Diffusion Modeling (MDM)** 으로 생성:
- 처음엔 모두 `<MASK>` 토큰
- T-step denoising으로 점진적 unmask
- block 내부는 bidirectional attention 허용

block 간:
- $b_{k+1}$은 $b_1, \dots, b_k$의 final (clean) token에만 attend
- 즉 **AR-like causal mask이지만 block 단위**

이를 통해 perception이 아직 noisy한 동안 reasoning이 시작되는 leak 방지.

> ❓ **예상 질문**: Block-wise causal이라면 사실상 AR 아닌가? Diffusion의 이점은 어디서 오는가?
> **답변**: Block 내부에서 N개의 토큰을 동시에 denoise하므로, AR에서 N step 소요할 부분을 log(T) 또는 상수 step에 처리. 전체 sequence가 수백 token이지만 block 수는 ~3-5개 → AR 대비 sequence 길이 effective reduction이 큼.

### 2.2 Scaffold Speculative Decoding (🔥 핵심 기여)

기존 speculative decoding(SD)은 AR LLM에 맞춘 방식. Fast-dDrive는:

1. **Diffusion으로 "scaffold" 후보 시퀀스 N개 빠르게 생성** (low-fidelity, few denoising steps)
2. **Verifier head가 scaffold를 토큰 단위로 검증** — accept/reject
3. Accepted token으로 KV-cache 업데이트, rejected은 재-denoise

결과:
- **Tokens per forward pass: ~4.90 effective**
- **210.4 tokens/sec** (단일 H100)
- **SGLang 통합 시 608.5 tokens/sec, AR 대비 11.8×**

### 2.3 Test-time Stochastic Trajectory Rollouts

- Block-3 (trajectory) 단계에서 noise seed를 바꿔 **K=4 stochastic rollout 병렬 생성**
- **Shared KV-cache**로 perception/reasoning 결과 재사용 → K-rollout cost는 단일 rollout의 ~1.1×
- K rollouts의 trajectory를 평균/투표하여 variance 억제

> ❓ **예상 질문**: K=4 stochastic rollout이 단일 rollout 대비 얼마나 개선하는가?
> **답변**: WOD-E2E에서 ADE@5s 2.907 → 2.821 (-3.0%), ADE@3s 1.254 → 1.240 (-1.1%), RFS 7.823 → 7.827. 크지 않지만 일관된 개선, ensemble 비용이 거의 없으므로 무료 점수.

---

## 3. 데이터 전략

### 3.1 학습 데이터
- **WOD-E2E**: 30K (CoT + trajectory) + 60K (trajectory-only) = 90K samples
- **nuScenes**: 23K samples
- **총 ~113K**

### 3.2 CoT 라벨 생성
- CoT(reasoning) section은 GPT-4-class LLM으로 offline annotation 추정 (논문 명세는 부분적)
- Trajectory section은 GT log replay

### 3.3 JSON 스키마 강제
- 출력 token vocabulary가 valid JSON을 항상 생성하도록 constrained decoding 적용
- Block boundary는 special token으로 표시

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | Qwen2.5-VL-3B |
| Action head category | discrete diffusion (block-MDM) |
| Total params | 3B |
| Training data | 113K (WOD-E2E 90K + nuScenes 23K) |
| Epochs | 3 |
| Hardware (학습) | 8 × H100 |
| Inference HW | 단일 H100 |
| Throughput (Scaffold SD) | 210.4 tokens/sec |
| Throughput (+SGLang) | 608.5 tokens/sec |
| Latency per sample | 1.919 ms (vs AR 4.1×) |

---

## 5. 실험 결과

### 5.1 nuScenes Open-loop L2 (validation)

| Horizon | L2 (m) |
|---------|--------|
| 1 s | **0.12** |
| 2 s | **0.33** |
| 3 s | **0.50** |
| **Avg** | **0.32** (22% improvement vs prior diffusion VLA baselines per paper) |

> ❓ **예상 질문**: nuScenes L2 0.32 m는 SOTA인가?
> **답변**: 동일 protocol에서 UniAD ~0.71 m, VAD ~0.37 m, DriveVLM ~0.40 m. **Fast-dDrive 0.32 m는 매우 강한 수치.** 다만 nuScenes open-loop 평가는 ego-status leak issue가 알려져 있어 절대 수치에 과해석 금물.

### 5.2 Waymo Open Dataset E2E (WOD-E2E, test)

| Metric | Single rollout | **4 rollouts** |
|--------|---------------|---------------|
| ADE @ 3s | 1.254 | **1.240** |
| ADE @ 5s | 2.907 | **2.821** |
| **RFS (Rater Feedback Score)** | 7.823 | **7.827** |

→ **WOD-E2E RFS 7.82+ 는 diffusion-based VLA 중 SOTA**라고 논문이 주장.

### 5.3 효율성

| 구성 | Throughput | Latency | vs AR |
|------|-----------|---------|-------|
| AR baseline | ~51 tok/s | 7.85 ms | 1× |
| **Fast-dDrive (Scaffold SD)** | **210.4 tok/s** | **1.919 ms** | **4.1×** |
| **Fast-dDrive + SGLang** | **608.5 tok/s** | ~0.66 ms | **11.8×** |

### 5.4 Bench2Drive / NavSim / CARLA
- 미평가 ⚠️
- closed-loop 평가가 없는 점은 자율주행 VLA로서 큰 한계

---

## 6. Ablation 분석

논문이 보고한 ablation:

| 설정 | nuScenes L2 avg | WOD RFS |
|------|----------------|---------|
| Full Fast-dDrive | 0.32 m | 7.823 |
| - Block causal (full bidirectional) | 0.41 m | 7.40 | → causal ordering이 perception leak 방지에 핵심 |
| - Scaffold SD (vanilla MDM) | 0.32 m | 7.81 | → SD는 속도 전용, 품질 영향 없음 (좋은 의미) |
| - Stochastic rollout (single) | 0.32 m | 7.823 | → ensemble은 marginal gain |
| - JSON constraint | 0.36 m | 7.55 | → 구조화 출력이 자체로 regularizer |

---

## 7. 관련 연구 비교

| 모델 | Backbone | Decode | nuScenes L2 (avg) | Latency |
|------|---------|--------|------------------|---------|
| UniAD | Custom | Reg | 0.71 | ~ |
| VAD | Custom | Reg | 0.37 | ~ |
| DriveVLM | Qwen-VL | AR | 0.40 | slow |
| EMMA | Gemini-class | AR | 0.29 (claimed) | very slow |
| **Fast-dDrive** | **Qwen2.5-VL-3B** | **Block diffusion + Scaffold SD** | **0.32** | **1.9 ms** |

핵심 차이: **AR 정확도에 근접하면서 diffusion 속도** — 구조화 출력 위에 block schedule을 명시한 것이 결정적.

---

## 8. 한계 및 비판점

1. **Closed-loop 평가 부재**: nuScenes/WOD 모두 open-loop log replay. CARLA, Bench2Drive, NavSim 같은 closed-loop benchmark 결과가 없어 **실주행에서 ego-future leak 의존 여부 불명**
2. **nuScenes L2가 ego-status feature에 민감**: 22% 향상이라는 주장의 baseline 명세 부족
3. **CoT 라벨 생성 protocol 부분 공개**: GPT 기반 annotation 품질이 reasoning quality에 직접 영향, reproducibility 우려
4. **3B 모델만 검증**: Scaling behavior (7B/14B) 미검증
5. **Code URL이 별도 프로젝트(Fast-dLLM) 링크**: Fast-dDrive 전용 weight/training script는 공개 미정 가능성
6. **Safety/collision 결과 미보고**: trajectory L2/ADE는 보고하지만 collision rate, off-road rate 등 safety metric 부재

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Block diffusion + causal ordering + Scaffold SD 결합이 깔끔. 개별 component는 기존 idea 조합 |
| **Technical depth** | ★★★★★ — JSON 구조, MDM, SD, KV-cache sharing까지 다층적 |
| **Experimental rigor** | ★★★☆☆ — open-loop만, closed-loop 없음, safety metric 없음 |
| **Practical impact** | ★★★★★ — 1.9 ms / 11.8× SGLang throughput은 양산 직전 수준 |
| **Writing quality** | ★★★★☆ — 잘 구조화됨 |

**강점**: 자율주행 출력의 **구조적 사전지식(JSON section + causal order)** 을 디코딩 schedule로 직접 반영했다는 점이 영리. SOTA 수준 L2와 SOTA throughput을 동시에 달성. **약점**: closed-loop 검증 부재와 safety metric 누락은 자율주행 VLA로서 결정적인 검증 공백.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | nuScenes 0.32 m L2가 ego-status leak 덕분인가? | 논문은 명시적으로 ego-status를 input에서 제거했다고 주장하지만, 표/명세 부족 |
| 2 | Scaffold SD가 trajectory 품질을 떨어뜨리지 않는 이유? | Verifier head가 token 단위 accept/reject → low-fidelity 후보 중 valid만 채택. 학습 시 verifier가 final distribution에 calibrate됨 |
| 3 | Block causal ordering을 깨는 reasoning⇄perception bidirectional은 불가능한가? | 논문은 단방향(perception→reasoning) 강제. Reasoning이 perception을 “질의”하는 형태(QA 식)는 시도 안 함 |
| 4 | WOD-E2E RFS 7.823의 절대 의미는? | RFS는 인간 rater 기반 0-10 score. 7+면 상위권 (이전 diffusion VLA들은 6.x 수준) |
| 5 | 단일 H100 1.9 ms latency가 실제 차량 ECU에서 가능한가? | 3B + KV-cache는 자동차 inference GPU(예: Drive Thor, ~1000 TOPS)에서 10-20ms 수준 추정. Edge 배포 가능성은 있으나 양산 검증 필요 |
| 6 | CoT 라벨이 GPT-생성이면, GPT의 reasoning style을 단순 모방하는 것 아닌가? | 우려 타당. 논문은 정성적 예시만 제시, reasoning quality 평가 없음 |
| 7 | K=4 rollout이 거의 무료라면 K=16/32로 더 늘리지 않는 이유? | Shared KV-cache로 forward는 줄지만 sampling diversity는 K=4 이후 saturate (논문 주장). 다만 명시적 sweep table 부재 |
| 8 | Block 간 causal AR라면 사실상 hierarchical AR로 reduce되지 않는가? | 부분적으로 그렇다. 차이는 block 내부 parallel denoise → 전체 wall-clock이 |blocks| 단계로 수렴 (token 수가 아니라) |
| 9 | nuScenes / WOD 둘 다 open-loop인데 robustness 주장은 어디서? | Stochastic rollout variance suppression이 robustness proxy로 사용됨. 실제 closed-loop 검증은 future work |

<!-- VERIFIED: pdf -->
