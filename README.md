# LB-IGD (Laplace–Beltrami Inverse Game Design)

**체스-유사 격자 게임에서, 맵/장애물 + 팩션별 말(개수/행마/사거리/공격 패턴/스탯)을 설계변수로 두고 self-play로 자동 밸런싱하는 IGD 프로젝트**

## 1. 무엇을 하는 프로젝트인가
이 레포의 목표는 아래를 자동으로 탐색하는 것입니다.

- 팩션별 말 설계: **개수 / 행마 패턴 / 이동 / 공격 패턴 / 사거리 / 공격력 / 체력**
- 맵 설계: **맵 크기 / 장애물 밀도·패턴**

평가는 self-play 결과(승률/무승부/교전 거리 등)로 수행하고, Outer loop(ES)가 설계변수를 업데이트합니다.  
또한 설계공간의 승률장 $P(x)$가 지나치게 뾰족해지는 방향을 억제하기 위해 **LBO(라플라시안) 기반 정규화**를 사용합니다.

## 2. 실행 (Quick Start)

### 1) 설치
Python(3.12 이상)이 설치되어 있어야 합니다. 패키지 관리자 `uv`를 사용합니다.

```bash
# uv 설치 (이미 있다면 패스)
pip install uv

# 필요한 라이브러리 한방에 설치
uv sync
```

### 2) 실행
설치가 끝났다면 바로 실행해 보세요.

```bash
# Windows
.venv/Scripts/python main.py

# Mac / Linux
.venv/bin/python main.py
```

### 3) 결과
`main.py`는 기본으로 “평가(evaluate)”를 수행합니다. 최적화는 `--opt`로 실행합니다.

```bash
# 평가(기본)
.venv/Scripts/python main.py

# 최적화
.venv/Scripts/python main.py --opt

# 더 안정적인 평가/최적화(느리지만 흔들림 감소)
.venv/Scripts/python main.py --slow
```

## 3. 스펙 문서
- `plan.md`: 팩션 체스 IGD 목표/설계변수/평가/최적화 루프
- `docs/lbo.md`: Bellman → HJB → diffusion → Laplace–Beltrami 정리 및 설계공간 LBO 정규화
- `docs/guide.md`: 비개발자용 상세 사용 설명서
