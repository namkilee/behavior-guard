# BehaviorGuard Charter v2.1 (Research Mode — Model-Centric, Client-Partitioned)

Status: Active  
Phase: Research Mode  
Last Updated: 2026-02-23 (Asia/Seoul)  
Effective Until: User explicitly declares change of objective

---

## 0. One-Line Summary

BehaviorGuard Research Phase의 목표는 **로그 기반 세션 데이터를 입력으로 하는 이상행동 탐지 모델들을 실험적으로 비교·개선**하여, **Precision@K를 중심으로 가장 적절한 접근(통계 / Shallow Sequence / Deep Sequence + Fine-tuning/Compression)을 찾는 것**이다.

---

## 1. Current Objective

본 프로젝트의 1차 목적은 운영 적용이 아니다.

본 Research Phase의 목표는 다음과 같다:

> 동일한 세션 로그 데이터셋을 기반으로  
> (A) 통계 기반 접근, (B) Shallow Sequence 접근, (C) Deep Model Suite 접근을 비교하고,  
> 특히 Track C에서 **기존/사전학습 모델(LogLLM/LLM 포함)** 및 **fine-tuning, pruning/quantization 등 기법**을 적용하여  
> **이상행동 탐지 정확도(Precision@K)를 최대화**하는 전략을 탐색한다.

Risk Score는 오프라인 실험을 위한 분석 지표로만 사용한다.

---

## 2. Scope

### Included

- 세션 단위 행동 분석(Session-level)
- 동일 데이터셋 기준 모델 비교 실험
- Feature engineering 실험(Track A)
- Shallow sequence 실험(Track B: n-gram/Markov/entropy)
- Deep model 실험(Track C: multiple models & training strategies)
- Foundation/LogLLM/LLM 적용 실험(텍스트 직렬화 포함)
- Fine-tuning 전략 비교 (Full / LoRA / QLoRA / weak-label supervised)
- Model compression 실험 (Quantization / Pruning / (Optional) Distillation)
- Top-K 기반 human validation
- Precision@K, overlap, stability 기반 평가

### Excluded (Research Phase에서는 수행하지 않음)

- 실시간 Enforcement Layer 설계
- Production 배포 구조 설계
- Policy DSL 최종 확정
- Multi-tenant 운영 정책 확정
- LiteLLM 실시간 연동 구조 확정(Research 결과 이후로 유예)

---

## 3. Data Contract (Common Canonical Input)

### Canonical Input Artifact

- `out/common/sessions_packed_{day}.parquet`

### Canonical Join Key

- `(client_name, day, user_id, session_key)`

### Packed Array Columns (per session row)

- `event_time[]`
- `observation_id[]`
- `route_group[]`
- `outcome_class[]`
- `token[]`

### Invariants

- row = 1 session
- row 내부 배열 길이 정합성 유지 (동일 index는 동일 event를 의미)
- Common input 레이어는 “안정화된 전제”로 취급한다.

---

## 4. Core Principle: Common Input for All Models

Research loop의 속도와 공정 비교를 위해, **모든 모델의 입력은 동일한 common canonical input에서 파생**된다.

- 모델마다 “별도 맞춤 데이터”를 만들지 않는다.
- 단, 공통 파생물(예: tokens[])은 **한 번만 생성**하고 모든 모델이 공유한다.

### Optional Shared Derived Artifact (Track C 공통)

- `out/common/sessions_tokens_{day}.parquet`
  - `tokens[]` (문자열 토큰 시퀀스)
  - `seq_len`
  - (optional) `text` = `" ".join(tokens)` (Foundation/LLM 입력용)

> `sessions_tokens_{day}.parquet`는 `sessions_packed`에서 결정론적으로 생성되어야 하며, 모든 Track C 모델은 이를 공유 입력으로 사용한다.

---

## 5. Partition Policy (Client-Partitioned Training)

빠른 결과와 클라이언트별 행동 분포 차이를 고려하여, Research Phase에서는 다음을 기본 가정으로 고정한다:

> **모든 Track C 모델은 client별로 분리 학습한다.**

- 기본 파티션: `(client_name, day)` 또는 `(client_name)` (실험 설정에 따라)
- 최소 가정: “client별 학습” (client 수 4~5개 수준)
- 목적: client 간 분포 혼합으로 인한 모델 성능 저하를 방지하고 비교 가능성을 높임

---

## 6. Experiment Tracks

### Track A — Statistical Baseline (Feature-based)

목적: Deep/Sequence 접근의 성능 향상을 검증하기 위한 기준선

- Session-level feature table
- IsolationForest / LOF / Robust Z-score
- Feature interpretability 및 안정성 비교

산출물:
- 세션별 score + Top-K 리스트
- (optional) feature importance / SHAP-like 해석(Research 범위 내)

---

### Track B — Shallow Sequence

목적: 경량 sequence 접근의 성능/한계 확인 및 Track C 비교 축 제공

- Markov transition likelihood (NLL)
- N-gram rarity
- Sequence entropy 기반 anomaly score

산출물:
- 세션별 score + Top-K 리스트
- Track A와의 overlap 분석

---

### Track C — Deep Model Suite (4~5 Models, Model-Centric)

Track C는 단일 모델이 아니라 “Deep 접근법 + 학습/개선 기법” 실험군이다.

#### Track C Model Set (Frozen Target = 4~5)

Research Phase에서 시간 제한을 고려해 **4~5개 모델**만 운영한다. (추가 모델은 Phase 종료 전까지 원칙적으로 금지)

**권장 기본 세트 (5 models):**
- C1: LSTM Causal LM (Self-supervised next-token)
- C2: Small Transformer Causal LM (Self-supervised next-token)
- C3: Sequence Autoencoder (RNN 기반) (Reconstruction-based)
- C4: Foundation/LLM + LoRA (Text serialization, continued pretrain or LM)
- C5: Foundation/LLM + Weak-label Supervised (LoRA + classification head)

> 모든 C 모델은 동일한 canonical input에서 파생된 동일 입력(`tokens[]` 또는 `text`)을 사용한다.  
> Track A/B의 score는 Track C 입력으로 사용하지 않는다. (실험 오염 방지)

#### Techniques Under Test (Track C only)

- Fine-tuning: Full / LoRA / QLoRA
- Weak-label training: A/B Top-K 기반 pseudo-label + negative sampling
- Compression: Quantization / Pruning (optional, controlled)

산출물:
- 모델별 per-session anomaly score
- Top-K
- (optional) score stability / cost metrics

---

## 7. Evaluation Framework

모든 비교는 동일 데이터셋/동일 평가 기준에서 수행한다.

### Core Metrics

- Precision@K (Human validation 기반)
- Top-K overlap ratio (A vs B vs C, 그리고 C-model 간)
- Stability across time windows (day별/윈도우별)
- Risk score distribution consistency

### Model-Centric Metrics (Track C)

- Fine-tuning gain over baseline (ΔPrecision@K)
- Compression impact (ΔPrecision@K vs 비용 절감)
- Training stability (재현성, seed 민감도)
- Inference cost (상대 비교)

---

## 8. Decision Principles

- 설계 완성보다 실험 루프를 우선한다.
- Track C는 “정확도 향상 실험”이 핵심이며, 개선 기법은 Track C에서만 다룬다.
- 모델 수를 제한하여 비교의 통제 가능성을 유지한다. (Frozen 4~5 models)
- 입력 데이터는 통일한다. (Common input → optional shared derived artifact)
- 운영/정책 설계는 Research Phase 완료 전까지 고정하지 않는다.

---

## 9. Success Criteria

Research Phase는 다음 조건을 만족하면 완료로 간주한다:

1. 최소 3개 접근 축 비교 완료
   - Statistical baseline (Track A)
   - Shallow sequence (Track B)
   - Deep model suite (Track C, 4~5 models)

2. 동일 데이터셋 기준 Top-K 비교 및 overlap 분석 완료

3. Human validation 기반 Precision@K 비교 완료
   - A vs B vs 각 C 모델
   - Track C 내 fine-tuning / compression 효과 포함

4. 각 접근법의 장단점 문서화 완료
   - 성능(Precision@K), 안정성, 해석 가능성, 비용/복잡도

---

## 10. Change Control

이 Charter는 사용자가 다음과 같이 선언하기 전까지 유지된다:

"Research Phase 종료. 목표를 변경한다."

그 전까지 본 프로젝트는 Research Mode로 고정된다.

---