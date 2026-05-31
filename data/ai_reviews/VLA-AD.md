# VLA-AD: Offline Semantic Guidance for Efficient Vision-Language-Action Policy Distillation

> **한 줄 요약**: OpenVLA-7B / π₀.₅-4B 같은 거대 VLA teacher를, **VLM이 오프라인으로 생성한 task-phase anchor와 multi-frame operating-direction description**을 보조 신호로 사용해 **158M 파라미터 student (8.6M LoRA trainable)**로 증류. LIBERO에서 teacher와 평균 0.27% 격차로 **44× 경량화 + 3.28× 추론 가속(12.5 Hz, RTX 4090)**을 달성.

---

## 1. 배경 및 동기

### 기존 VLA의 배포 병목
- OpenVLA-7B, π₀.₅-4B 등 SOTA VLA는 단일 RTX 4090에서 ~3.8 Hz로 동작 → real-time closed-loop control에 부적합
- 기존 VLA 증류 방법은 대부분 **action imitation 손실만 사용** → semantic 정보 손실로 student가 teacher의 multi-step planning 능력을 잃음
- 단순 KD는 다양성 부족 → student가 teacher 분포의 mode collapse에 빠짐

### 핵심 질문
1. **VLM의 풍부한 언어적 의미 정보를 distillation 과정에 어떻게 자연스럽게 주입할 수 있는가?**
2. **이 추가 신호가 training-only로 들어가도(즉 inference 시 cost-free) 실제 성능 향상을 줄 수 있는가?**

📌 [Figure 1 삽입] — VLA-AD 파이프라인: Teacher rollout → VLM이 phase/direction 라벨 생성 → Student multi-task 학습

---

## 2. 방법론 심층 분석

### 2.1 전체 구조

VLA-AD는 **teacher VLA의 rollout**을 모은 뒤, 각 frame에 **Qwen2.5-VL이 두 가지 의미적 라벨**을 자동 부여:

| 라벨 종류 | 정의 | 역할 |
|----------|------|------|
| **Task phase anchor** | 에피소드를 "reach / grasp / transport / place" 등 phase로 segmentation한 라벨 | 시간적 의미 구조 제공 |
| **Operating-direction description** | 직전 K-frame 비교로 "moving left / approaching object / closing gripper" 같은 짧은 자연어 동작 묘사 | 단기 모션 의도 캡처 |

Student는 세 가지 head를 동시에 학습:
1. **Action head**: K=5 chunk의 7-DoF action 예측 (MSE)
2. **Phase head**: 현재 phase 분류 (CE)
3. **Direction head**: 다음 동작의 자연어 description embedding 회귀 (Long-CLIP embedding 공간으로 cosine)

> ❓ **예상 질문**: Direction description을 그대로 텍스트 생성하지 않고 왜 CLIP embedding 회귀로 처리하는가?
> **답변**: Text generation head를 student에 추가하면 파라미터/지연이 폭증. Long-CLIP embedding 회귀는 **inference 시 phase/direction head를 비활성화 가능 → cost-free auxiliary**라는 핵심 설계 의도를 보존.

### 2.2 Student 아키텍처

- **Vision encoder**: Long-CLIP (긴 컨텍스트 텍스트 정렬 가능)
- **Adapter**: rank-8 LoRA → **trainable 파라미터 8.6M만**
- **Action head**: tri-stream MLP (각 stream은 chunk 내 다른 시점 담당)
- **총 파라미터**: 158M (OpenVLA-7B 대비 44×, π₀.₅-4B 대비 25× 압축)

### 2.3 학습 손실

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{phase}} + \lambda_2 \mathcal{L}_{\text{direction}}$$

- $\lambda_1, \lambda_2$ 값은 논문에서 sweep
- **Inference 시 phase/direction head를 끄고 action head만 사용** → semantic 라벨이 학습 시에만 regularizer 역할

> ❓ **예상 질문**: 이런 multi-task는 "label smoothing의 더 정교한 형태" 아닌가?
> **답변**: 부분적으로 그렇다. 다만 phase anchor는 단순 noise가 아니라 **task의 시간적 구조**를 제공하는 강한 inductive bias. Ablation에서 두 head 모두 제거 시 평균 -3.1% 하락 보고.

---

## 3. 데이터 전략

### 3.1 Teacher Rollout 수집
- **성공 에피소드만 수집** (LIBERO 4 suite에서 teacher가 성공한 trajectory)
- VLA-AD-OpenVLA: ~12K demos
- VLA-AD-π₀.₅: ~10K demos

### 3.2 VLM 자동 라벨링
- Qwen2.5-VL API 호출
- 총 **~81,000 frame** annotation
- 비용 **~$7 USD** (API 호출 기준)

> ❓ **예상 질문**: Qwen2.5-VL의 라벨 노이즈는 어떻게 처리했는가?
> **답변**: 명시적 noise filtering은 없음. Phase는 8-class softmax이므로 noise tolerance가 있으며, direction은 embedding 회귀라 cosine 유사도 기반으로 부드럽게 학습. 다만 **실패 episode가 학습에 없어서 OOD recovery 능력은 의문**.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Epochs | 30 |
| Batch size | 32 |
| LoRA rank | 8 (학습 파라미터 8.6M) |
| Total compute | ~22 GPU-hours per student |
| Annotation cost | ~$7 USD (Qwen2.5-VL) |
| Inference HW | RTX 4090 단일 GPU |
| Inference rate | **12.5 Hz** (vs OpenVLA 3.8 Hz, 3.28×) |

---

## 5. 실험 결과

### 5.1 LIBERO — OpenVLA-7B Teacher 변종

| Suite | Teacher (OpenVLA-7B) | **VLA-AD (158M)** | Gap |
|-------|---------------------|------------------|-----|
| libero_object | 62.0% | **62.5%** | +0.81% |
| libero_spatial | 79.0% | **79.5%** | +0.63% |
| libero_goal | 81.0% | **79.5%** | -1.85% |
| **Average** | **74.0%** | **73.8%** | **-0.27%** |

### 5.2 LIBERO — π₀.₅-4B Teacher 변종 (🔥 핵심 결과)

| Suite | Teacher (π₀.₅-4B) | **VLA-AD (158M)** | Gap |
|-------|------------------|------------------|-----|
| libero_object | 90.0% | **96.5%** | **+7.22%** |
| libero_spatial | 87.0% | **93.0%** | **+6.90%** |
| libero_goal | 94.5% | **94.0%** | -0.53% |
| **Average** | **90.5%** | **94.5%** | **+4.42%** |

> ❓ **예상 질문**: Student가 teacher를 +4.42% 능가한다는 게 가능한가?
> **답변**: 매우 의심스러운 지점. 가능한 설명:
> 1. **성공 episode만 학습** → student는 distribution이 깨끗해 noise 적음
> 2. **Phase/direction head의 regularization 효과**가 단순 BC보다 강함
> 3. **Teacher의 evaluation seed**와 student evaluation seed가 다를 가능성 — 정확한 protocol 공개 필요

### 5.3 효율성

| 모델 | 파라미터 | 추론 (Hz, RTX 4090) | 압축비 |
|------|---------|--------------------|--------|
| OpenVLA-7B (teacher) | 7B | 3.8 | 1× |
| π₀.₅-4B (teacher) | 4B | ~5.0 | 1× |
| **VLA-AD** | **158M** | **12.5** | **25-44×** |

---

## 6. Ablation 분석

논문에서 보고한 주요 ablation:

| 설정 | LIBERO avg | 변화 |
|------|-----------|------|
| Full VLA-AD | 94.5% | base |
| − Phase head | 92.7% | -1.8% |
| − Direction head | 93.0% | -1.5% |
| − Both (pure BC distill) | 91.4% | -3.1% |
| − LoRA (full FT) | 93.8% | -0.7% |

→ 두 보조 head 모두 의미 있는 기여, **direction head가 phase보다 약간 더 중요**.

---

## 7. 관련 연구 비교

| 방법 | Compression | LIBERO avg | Teacher 의존 | 추가 신호 |
|------|------------|-----------|------------|----------|
| TinyVLA | 5× | ~85% | ✗ (from scratch) | 없음 |
| DeeR-VLA | 2× (dynamic) | ~87% | ✗ | 없음 |
| OpenVLA pure BC distill | 44× | 91.4% | ✓ | 없음 |
| **VLA-AD** | **44×** | **94.5%** | ✓ | **Phase + Direction (VLM 생성)** |

핵심 차이: **VLM-as-supervisor**라는 패러다임 (action 자체가 아니라 의미를 distill).

---

## 8. 한계 및 비판점

1. **Student > Teacher의 +4.42% 결과는 method 효과인지 evaluation 차이인지 불분명**
   - 동일 seed/protocol 명시가 부족
   - 만약 evaluation noise라면 핵심 claim 약화
2. **OpenVLA 변종에서는 거의 동률(-0.27%)** — π₀.₅ 변종의 큰 향상은 **teacher capacity가 student에게 “과잉”이라 student가 dataset에 더 잘 fit**할 가능성
3. **LIBERO만 평가** — Long suite 결과 미보고, real-world / CALVIN / SimplerEnv 평가 없음
4. **VLM annotation 의존**: Qwen2.5-VL 라벨 품질이 task에 따라 가변적일 수 있음 (특히 contact-rich phase boundary)
5. **Open-source 안 됨** — 코드/체크포인트 공개 미정, 재현성 우려

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLM as offline supervisor 아이디어는 신선, 다만 phase/direction 자체는 새로운 개념 아님 |
| **Technical depth** | ★★★☆☆ — LoRA + multi-task head는 표준 구조 |
| **Experimental rigor** | ★★★☆☆ — LIBERO long suite 누락, evaluation protocol 명세 부족 |
| **Practical impact** | ★★★★★ — 158M / 12.5 Hz는 edge deployment에서 매우 매력적 |
| **Writing quality** | ★★★★☆ — 명료, ablation 충실 |

**강점**: "VLM이 만든 의미 신호를 training-only로 쓴다"는 깔끔한 설계. 8.6M 학습 파라미터로 SOTA-급 성능 달성. **약점**: Student > teacher 라는 결과의 통계적 안정성과 LIBERO 외 평가 부재.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Student가 teacher를 +4.42% 능가한다는 게 method 덕인가, eval seed 덕인가? | 동일 protocol을 강하게 주장하지만 raw seed-by-seed 표가 부재. Bootstrap CI 없는 평균 비교는 위험 |
| 2 | Phase / direction 라벨이 VLM 노이즈를 포함하면 student가 noise를 학습하지 않는가? | Soft loss(CE + cosine)이므로 부분 mitigates. 다만 contact-rich phase boundary에서 VLM이 자주 틀린다는 정성적 근거 있음 |
| 3 | libero_long 결과는 왜 없는가? | 논문이 보고하지 않음 — long-horizon에서 chunk size K=5의 myopia가 노출될 가능성 |
| 4 | 12.5 Hz가 closed-loop control에 충분한가? | 7-DoF arm은 보통 20-30 Hz 권장. 12.5 Hz는 quasi-static manipulation은 OK, dynamic task는 borderline |
| 5 | 같은 보조 신호를 teacher 자체 학습에도 쓰면 더 좋아지지 않나? | 매우 좋은 질문. 논문은 student-only 적용. Teacher에 적용하지 않은 이유 설명 없음 |
| 6 | 81K frame 라벨을 단 $7로 했다는데, rate-limit / annotation 품질은 검증되었나? | 정량적 verification 없음. Random sample manual check가 abstract level에서만 언급 |
| 7 | LoRA만 학습하면 backbone (Long-CLIP)이 manipulation에 specific하지 않은데 OK한가? | Long-CLIP의 generic representation + LoRA adapter가 충분히 task-specific해진다고 주장. 다만 OOD scene에서 약점 가능 |
| 8 | OpenVLA teacher variant는 거의 동률, π₀.₅ variant는 +4.42% — teacher 종류에 왜 이렇게 차이가 큰가? | π₀.₅이 flow-matching 기반이라 distill 시 mode coverage 차이가 큼. 이건 distill 자체의 한계가 아니라 teacher action distribution 특성. |

<!-- VERIFIED: pdf -->
