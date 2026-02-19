# BehaviorGuard Charter v1.0

Status: Active  
Phase: Research Mode  
Effective Until: User explicitly declares change of objective  

---

# 1. Current Objective

BehaviorGuard의 현재 목표는 운영 적용이 아니다.

본 프로젝트의 1차 목적은 다음과 같다:

> 로그 기반 행동 데이터를 활용하여  
> 통계 기반 접근, Shallow Sequence 접근, Deep Sequence 접근을 실험적으로 비교하고  
> 가장 적절한 이상행동 탐지 방법을 탐색하는 것이다.

이 단계에서는 다음을 수행하지 않는다:

- 실시간 차단 설계
- Production Enforcement Layer 설계
- Policy DSL 확정
- LiteLLM 실시간 연동 구조 확정

Risk Score는 오프라인 실험을 위한 분석 지표로만 사용한다.

---

# 2. Scope

## Included

- 세션 단위 행동 분석
- 유저/윈도우 단위 행동 분석
- Feature engineering 실험
- 통계 기반 모델 실험
- Shallow sequence (n-gram / Markov 등) 실험
- Deep sequence (Transformer/LSTM 등 최소 1개) 실험
- Top-K 기반 human validation
- 모델 안정성 및 설명 가능성 비교

## Excluded

- 실시간 Enforcement Layer 설계
- Production 배포 구조 설계
- Policy DSL 최종 확정
- Multi-tenant 운영 정책 설계

---

# 3. Success Criteria

Research Phase는 다음 조건을 만족하면 완료로 간주한다:

1. 최소 3개 접근법 비교 완료
   - 통계 기반 모델
   - Shallow sequence 모델
   - Deep sequence 모델 (최소 1개)

2. 동일 데이터셋 기준 Top-K 비교 수행

3. Human validation 기반 Precision@K 비교 수행

4. 각 접근법의 장단점 문서화 완료

---

# 4. Decision Principles

- 설계 완성보다 실험 루프를 우선한다.
- 모델 정확도는 human validation 기반으로 평가한다.
- 해석 가능성은 중요한 비교 지표다.
- 실험 결과가 충분히 축적되기 전까지 운영 설계는 고정하지 않는다.
- 모든 실험은 동일 데이터셋과 동일 평가 기준에서 비교한다.

---

# 5. Evaluation Framework

모든 모델은 다음 기준으로 평가한다:

## Core Metrics

- Precision@K (Human validation 기반)
- Top-K overlap ratio
- Stability across time windows
- Risk score distribution consistency

## Secondary Metrics

- Feature importance interpretability
- Model training stability
- Computational cost
- Scalability

---

# 6. Experiment Tracks

## Track A — Statistical Baseline

- Session-level feature table
- Isolation Forest / LOF / Robust Z-score
- Feature importance 분석

## Track B — Shallow Sequence

- Bigram / Trigram frequency
- Markov transition likelihood
- Sequence entropy 기반 anomaly score

## Track C — Deep Sequence

- Transformer / LSTM 기반 encoder
- Self-supervised pretraining 또는 weak-label training
- Sequence-level anomaly scoring

---

# 7. Change Control

이 Charter는 사용자가 명시적으로 다음과 같이 선언하기 전까지 유지된다:

"Research Phase 종료. 목표를 변경한다."

그 전까지 본 프로젝트는 Research Mode로 고정된다.

---

# 8. Version

Version: 1.0  
Phase: Research Mode  
Last Updated: 2026-02-19
