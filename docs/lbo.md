## 제2장 라플라스–벨트라미(LBO): 벨만 → HJB → 이토(Itô) → 라플라시안 → (다양체) LBO → (설계공간) 2차 차분 정규화

> 본 문서는 LB-IGD/Faction Chess IGD에서 **“왜 라플라시안/라플라스–벨트라미(LBO) 관점이 필요한가?”**를
> 강화학습의 기반인 **벨만 방정식**에서 출발해, 필요한 만큼만(최소한의 증명 스케치) 연결해서 설명합니다.
>
> - 관련 문서: 제1장 `docs/bellman.md`(계층적 정식화), 제3장 `docs/blackbox.md`(ES), 제4장 `docs/evaluation.md`(평가 프로토콜)
> - 코드 대응: `src/core/designer.py`(설계공간 LBO), `src/core/lbo.py`(그래프 라플라시안), `tests/test.py`

---

### 0. 표기(Notation)
이 문서에서 자주 쓰는 기호만 최소로 고정합니다.

- 설계 변수: \(x \in \mathcal{X}\) (맵/장애물/유닛 수/패턴/스탯 등)
- 설계가 만드는 게임: \(\mathcal{M}_x\) (플레이 레벨의 MDP)
- (플레이 레벨) 가치함수: \(V(s)\) 또는 \(V(x)\)
- (설계 레벨) 승률 지형: \(P(x)\in[0,1]\)
  - 2팩션이면 보통 \(P(x)=\mathrm{WinRate}_{p0}(x)\)
  - 다팩션이면 \(P(x)\)를 쌍대결 승률행렬 \(W_{f,g}(x)\)로 본다(`evaluation.md`)

---

### 1. 출발점: 이산 시간 벨만 방정식
강화학습의 표준 형태(이산 시간, 마르코프성 가정)에서, 최적 가치함수는

$$
V^*(s)
=
\max_{a\in\mathcal{A}}\Big(
r(s,a)+\gamma\,\mathbb{E}[V^*(s')\mid s,a]
\Big)
$$

를 만족합니다. 핵심은 “현재 가치 = 즉시 보상 + 미래 가치의 할인된 기대값”이라는 **재귀 구조**입니다.

---

### 2. 연속 시간으로 보내기: HJB(해밀턴–야코비–벨만)
연속 시간 버전은 “아주 짧은 시간 \(\Delta t\)”에 대해 동적계획 원리를 다시 쓰는 것으로 얻습니다.
직관적으로는 다음 형태를 가정합니다.

1) 상태는 연속 시간에서 확률미분방정식(SDE)을 따른다:

$$
dX_t = b(X_t,a_t)\,dt + \sigma(X_t,a_t)\,dW_t
$$

2) 할인율을 \(\rho>0\)로 두고, 짧은 시간 구간에서의 Bellman 원리를 쓴다:

$$
V(x)=
\max_a \mathbb{E}\Big[
\int_0^{\Delta t} e^{-\rho t} r(X_t,a)\,dt

+
e^{-\rho\Delta t} V(X_{\Delta t})
\Big].
$$

\(\Delta t\to 0\)에서의 1차 근사를 적용하면

- \(e^{-\rho \Delta t}\approx 1-\rho\Delta t\)
- \(\mathbb{E}[V(X_{\Delta t})]\approx V(x) + \Delta t\,\mathcal{L}^a V(x)\) (여기서 \(\mathcal{L}^a\)는 생성자(generator))

가 되고, 정리하면 HJB가 됩니다.

$$
0
=
\max_a\Big(
r(x,a)+\mathcal{L}^a V(x)-\rho V(x)
\Big).
$$

이제 핵심은 “\(\mathcal{L}^a\) 안에 무엇이 들어있는가?” 입니다.

---

### 3. 이토(Itô) 공식이 만들어내는 2차 미분 항(라플라시안의 출현)
SDE

$$
dX_t = b\,dt + \sigma\,dW_t
$$

에 대해, 충분히 매끄러운 함수 \(V(X_t)\)에 이토 공식을 적용하면

$$
dV
=
\nabla V^\top b\,dt
+
\frac{1}{2}\mathrm{Tr}\!\Big(\sigma\sigma^\top \nabla^2 V\Big)\,dt
+
\nabla V^\top \sigma\,dW_t.
$$

여기서 **2차 미분 항**이 바로

$$
\frac{1}{2}\mathrm{Tr}\!\Big(\sigma\sigma^\top \nabla^2 V\Big)
$$

입니다. 만약 확산이 등방성이라서 \(\sigma\sigma^\top = 2\nu I\) (스칼라 \(\nu>0\))라면

$$
\frac{1}{2}\mathrm{Tr}(2\nu I \nabla^2 V)
=
\nu\,\mathrm{Tr}(\nabla^2 V)
=
\nu\,\Delta V.
$$

즉, **확률적 확산을 포함하면 라플라시안 \(\Delta V\)** 가 자연스럽게 등장합니다.
이게 “확산이 곧 2차 미분(곡률)”이라는 핵심 연결고리입니다.

---

### 4. 왜 “라플라스–벨트라미(LBO)”인가? (좌표 불변성)
위의 \(\Delta\)는 유클리드 공간(\(\mathbb{R}^n\))의 라플라시안입니다.
하지만 상태/설계공간이 “굽은 공간(다양체)”로 모델링되어야 한다면, 좌표를 바꿔도 의미가 유지되는 연산자가 필요합니다.
그때 쓰는 것이 라플라스–벨트라미 연산자 \(\Delta_g\)입니다.

#### 4.1 정의(표준 형태)
리만 다양체 \((\mathcal{M},g)\)에서 스칼라 함수 \(f\)에 대한 LBO는

$$
\Delta_g f := \mathrm{div}_g(\nabla_g f)
$$

로 정의됩니다. 좌표계 \((x^1,\ldots,x^n)\)에서의 전개식은 다음입니다.

$$
\Delta_g f
=
\frac{1}{\sqrt{|g|}}\partial_i\Big(\sqrt{|g|}\,g^{ij}\partial_j f\Big).
$$

- \(g^{ij}\)는 메트릭 행렬 \(g\)의 역행렬 성분
- \(|g|\)는 \(\det(g)\)
- 반복 인덱스는 합(sum)으로 약속

유클리드 공간에서는 \(g=I\), \(|g|=1\)이므로 \(\Delta_g=\sum_i \partial_{ii}=\Delta\)로 돌아갑니다.

---

### 5. 설계공간에서의 LBO: “진짜 \(\Delta_g\)” 대신 “차분 기반 곡률 측정”
이 프로젝트의 설계공간 \(x\)는 이산(패턴 ID, 유닛 수)과 연속(스탯) 변수가 섞여 있고, 코드에서 클램프/정수화를 거칩니다.
따라서 엄밀한 의미의 매끄러운 다양체 \((\mathcal{X},g)\)를 직접 두고 \(\Delta_g\)를 계산하기보다, 다음 전략을 씁니다.

> **승률 지형 \(P(x)\)** 가 “너무 뾰족한 방향(곡률 큰 방향)”을 탐지해, 그 방향 업데이트를 약화한다.

이를 위해 2차 중앙 차분을 사용합니다.

#### 5.1 1차원 중앙 차분 증명 스케치
테일러 전개로

$$
f(x\pm h)=f(x)\pm h f'(x)+\frac{h^2}{2}f''(x)+O(h^3).
$$

따라서

$$
f(x+h)+f(x-h)-2f(x)=h^2 f''(x)+O(h^4)
$$

이고,

$$
f''(x)\approx \frac{f(x+h)+f(x-h)-2f(x)}{h^2}.
$$

즉 “양쪽을 더하면 1차항이 상쇄되고 2차항이 남는다”는 구조가 핵심입니다.

#### 5.2 다차원에서의 방향 2차 미분(헤시안) 연결
\(f:\mathbb{R}^d\to\mathbb{R}\)가 충분히 매끄럽고, 방향 벡터 \(e\in\mathbb{R}^d\)에 대해 테일러 전개를 쓰면

$$
f(x+\sigma e)
=
f(x)+\sigma\nabla f(x)^\top e+\frac{\sigma^2}{2}e^\top H(x)e+O(\sigma^3)
$$

이며 \(H(x)=\nabla^2 f(x)\)는 헤시안입니다. 같은 방식으로 \(f(x-\sigma e)\)를 더하면

$$
f(x+\sigma e)+f(x-\sigma e)-2f(x)
=
\sigma^2 e^\top H(x)e + O(\sigma^4).
$$

따라서

$$
\frac{f(x+\sigma e)+f(x-\sigma e)-2f(x)}{\sigma^2\|e\|^2}
\approx
\frac{e^\top H(x)e}{\|e\|^2}
$$

는 “그 방향의 곡률”을 근사합니다.

#### 5.3 “라플라시안”과의 연결(등방성 방향 평균)
만약 \(e\)가 등방성(isotropic) 분포에서 뽑히면(예: 표준정규), 위의 방향 곡률의 기대값은 \(\mathrm{Tr}(H)\)와 연결됩니다.
정확한 상수는 분포/정규화에 따라 달라지지만, 요지는

$$
\mathbb{E}\Big[\frac{e^\top H e}{\|e\|^2}\Big]
\propto
\mathrm{Tr}(H)
=
\Delta f
$$

로 “방향 곡률을 평균내면 라플라시안”이 된다는 점입니다.
프로젝트에서는 이 평균을 엄밀히 쓰기보다, 샘플별 \(|\Delta P|\)를 **안정성 지표**로 사용합니다.

---

### 6. 이 프로젝트에서의 사용: ES 업데이트를 “곡률로 가중”하기
이 프로젝트의 핵심은 다음입니다.

- 외부 목적은 승률/퇴화 방지/메타(거리 분포 등)로 구성된 \(J(x)\)이고(`docs/blackbox.md`)
- ES는 \(x\pm\sigma e\)를 평가해 업데이트 방향을 얻는데
- **승률 지형 \(P(x)\)** 의 곡률이 큰 샘플(불안정한 방향)은 업데이트에 덜 반영합니다.

#### 6.1 설계공간 라플라시안(2차 차분) 근사
2팩션 승률 \(P(x)\)에 대해 코드에서는 다음을 사용합니다.

$$
\Delta P(x)
\approx
\frac{P(x+\sigma e)+P(x-\sigma e)-2P(x)}{\sigma^2\|e\|^2}.
$$

다팩션의 경우 \(P\)가 행렬 \(W_{f,g}\)이므로, 각 쌍에 대해 위를 계산해 평균(또는 평균 절대값)을 사용합니다.

#### 6.2 공통 난수(CRN)가 중요한 이유
승률 추정 \(P(x;\omega)\)는 self-play/RL 노이즈가 큽니다.
따라서 같은 방향 \(e\)에 대해 center/pos/neg를 **동일 시드(=CRN)** 로 평가하여,
\(P(x+\sigma e)-P(x-\sigma e)\) 같은 “차이”의 분산을 낮춥니다.

#### 6.3 가중치로 쓰는 방식(현재 구현)
코드(`src/core/designer.py`)는

$$
w(e)=\frac{1}{1+\lambda\,|\Delta P(x)|}
$$

형태의 단순 가중치를 두고(\(\lambda\)는 현재 1.0), ES 그라디언트 누적에 곱합니다.
직관적으로 \(|\Delta P|\)가 크면 “그 방향은 설계가 예민하다”는 뜻이므로, 업데이트를 약하게 만듭니다.

---

### 7. (부록) 그래프 라플라시안과 디리클레 에너지(토이 실험 모듈)
`src/core/lbo.py`에는 그래프 라플라시안 \(L=D-W\)와 디리클레 에너지가 구현되어 있습니다.
여기서 \(W\)는 대칭 가중치 행렬, \(D\)는 차수 행렬입니다(\(D_{ii}=\sum_j W_{ij}\)).

#### 7.1 디리클레 에너지의 비음수성(증명 스케치)

$$
u^\top (D-W)u
=
\sum_i d_i u_i^2 - \sum_{i,j} w_{ij}u_i u_j.
$$

대칭 \(w_{ij}=w_{ji}\)를 가정하면,

$$
\frac{1}{2}\sum_{i,j} w_{ij}(u_i-u_j)^2
=
\frac{1}{2}\sum_{i,j} w_{ij}(u_i^2+u_j^2-2u_i u_j)
=
\sum_i d_i u_i^2 - \sum_{i,j} w_{ij}u_i u_j
$$

이므로

$$
\boxed{u^\top L u = \frac{1}{2}\sum_{i,j} w_{ij}(u_i-u_j)^2 \ge 0}
$$

가 됩니다(가중치 \(w_{ij}\ge 0\)). 이 성질은 테스트(`tests/test.py`)에서 간단히 확인합니다.

---

### 8. 실무 튜닝 포인트(설계/구현 관점)
- \(\sigma\)가 너무 작으면: 이산/클램프 때문에 \(x+\sigma e\)와 \(x-\sigma e\)가 같은 설계로 투영되어 차분이 0이 되기 쉽습니다.
- \(\sigma\)가 너무 크면: “국소 곡률”이 아니라 전역 비선형 변화까지 섞여 곡률 해석이 어려워집니다.
- CRN은 필수에 가깝습니다: self-play 노이즈가 크기 때문에, 같은 seed로 center/pos/neg를 묶는 것이 분산 감소에 큰 도움이 됩니다.
- 다팩션에서는 \(\Delta P\)를 행렬로 두고 요약해야 합니다(현재 구현은 쌍대결별 \(\Delta W_{i,j}\) 절대값 평균).
