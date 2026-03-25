# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions

> **한 줄 요약**: 비디오에서 task-centric latent action을 학습하여 action annotation 없는 데이터로도 cross-embodiment VLA를 구축, OpenVLA 대비 우수 성능을 적은 compute로 달성.

---

## 1. 배경 및 동기

- Action-annotated data는 비싸고 제한적 → video data 활용 필요
- **Latent action model**: 연속 프레임 간 "무엇이 바뀌었는가"를 latent action으로 추출

---

## 2. 방법론

### Task-centric Latent Action Learning

1. Video에서 latent action 추출: $\mathbf{z} = E(\mathbf{o}_t, \mathbf{o}_{t+1})$
2. Task-relevant filtering: 무관한 변화(조명, 배경) 제거
3. Latent → real action decoder: 소량 labeled data로 학습

---

## 3. 실험 결과

| 모델 | SR (%) | Training Cost |
|------|--------|-------------|
| OpenVLA | 76.5 | 10K GPU-hrs |
| **UniVLA** | **~80** | **~2K GPU-hrs** |

---

## 4. 한계

1. Latent action의 해석가능성 부족
2. Decoder 학습에 여전히 labeled data 필요
3. Video quality 의존

---

## 5. 총평

| **Novelty** | ★★★★☆ | **Practical impact** | ★★★★☆ |

**강점**: Video data 활용으로 data bottleneck 완화. **약점**: Latent space의 한계.

---

## 6. 🔥 질문 모음

| 질문 | 답변 |
|------|------|
| AVDC와의 차이? | AVDC는 dense correspondence, UniVLA는 latent action. UniVLA가 더 compact representation |

<!-- VERIFIED: abstract-only -->
