STOCK AI DATABASE - FINAL BASELINE
=================================

앞으로 phase3_migration.sql / phase4_migration.sql / phase5_migration.sql 식으로 DB를 쪼개지 않습니다.
DB 변경 기준 파일은 오직:

    sql/DB_FINAL_SCHEMA.sql

하나입니다.

이 파일에는 다음 범위를 미리 포함합니다.
- 종목/현재가/일봉/분봉
- DART 원본 재무제표
- 재무 지표/점수
- 데이터 동기화 이력
- ML feature store(JSONB)
- 모델 학습 실행/검증/백테스트 메타데이터
- 최신 예측 + 예측 이력
- 뉴스 원문/종목 매핑/NLP 분석
- DART 공시
- 랭킹 스냅샷/랭킹 구성요소
- 추천 성과/실제 수익률/정확도 추적
- 앱 설정 JSON 저장소

현재 DB에 한 번 적용:

docker exec -i stock-ai-postgres psql -U stock -d stock_ai < sql/DB_FINAL_SCHEMA.sql

재실행해도 IF NOT EXISTS 기반이라 안전하게 설계했습니다.
향후 기능은 가능한 한 JSONB/기존 컬럼을 사용해서 스키마 수정 없이 추가합니다.
