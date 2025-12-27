## 제5장 라플라스–벨트라미(LBO): 벨만 방정식의 연속·기하 일반화

> 본 문서는 LB-IGD/Faction Chess IGD에서 “왜 라플라스–벨트라미(LBO)가 자연스러운가”와  
> “벨만 방정식의 일반화(연속 극한 + 기하 일반화)가 왜 LBO로 이어지는가”를 논문 스타일 유도로 정리한다.
>
> 주의: 이 레포의 LBO는 “플레이 상태공간 PDE를 직접 푸는 것”이 아니라, **설계공간에서의 승률장 $P(x)$ 곡률(라플라시안) 억제**로 구현된다(4절).

---

### 초록(요지)
- Bellman(이산 DP)은 “한 스텝 앞”으로의 재귀 관계(고정점)이다.
- 이를 연속 시간/연속 상태의 동적계획 원리(DPP)로 옮기면 HJB 편미분방정식(PDE)이 된다.
- 상태가 확률적으로 변화(확산 포함)하면 HJB에는 **2차 미분(확산) 항**이 필연적으로 나타난다.
- 유클리드 공간에서는 그 2차 항이 라플라시안 $\Delta$이고, 다양체(곡면)에서는 좌표 불변성을 위해 라플라스–벨트라미 $\Delta_g$로 일반화된다.

즉, “Bellman → LBO”는 다음 사슬로 이해하는 것이 정확하다.

$$
\text{Bellman / DPP}
\;\Rightarrow\;
\text{HJB (continuous-time DP)}
\;\Rightarrow\;
\text{diffusion term (2nd order)}
\;\Rightarrow\;
\Delta
\;\Rightarrow\;
\Delta_g
$$

---

## 1. 왜 라플라스–벨트라미인가(핵심 직관)
### 1.1 “확산”은 2차 연산자다
상태 값(가치, 신념, 승률 등)을 국소적으로 퍼뜨리는 동역학은 전형적으로

$$
\partial_t u
=
\text{(drift)}
\;+\;
\text{(diffusion)}
\;+\;
\text{(reaction/source)}
$$

꼴이다. 여기서 diffusion은 2차 미분 연산자(거칠기/곡률/스무딩을 반영)로 나타난다.

### 1.2 “좌표 불변”으로 쓰려면 $\Delta_g$가 필요하다
상태공간이 단순한 $\mathbb{R}^n$이 아니라, 곡면/다양체처럼 좌표 선택이 임의적인 공간이면:
- 같은 물리/의미를 좌표만 바꿔도 결과가 달라지면 안 된다(좌표 불변).

리만 다양체 $(\mathcal{M}, g)$에서 스칼라장 $f$에 대한 라플라스–벨트라미는 다음처럼 정의된다.

$$
\Delta_g f
=
\operatorname{div}_g(\nabla_g f)
=
\frac{1}{\sqrt{|g|}}
\partial_i\!\left(\sqrt{|g|}\,g^{ij}\partial_j f\right)
$$

여기서 $g^{ij}$는 메트릭 $g$의 역행렬 성분, $|g|$는 행렬식이다.

---

## 2. Bellman/DPP → HJB (논문 스타일 유도)
이 절은 “강화학습을 PDE로 비유”하는 게 아니라, **연속시간 확률 제어의 동적계획 원리(DPP)**를 통해 HJB를 얻는 표준 유도를 적는다.

### 2.1 문제 설정: 연속시간 확률 제어(제어 확산)
상태 $X_t \in \mathbb{R}^n$, 제어(행동) $a_t \in \mathcal{A}$에 대해 다음 SDE를 가정한다.

$$
dX_t = b(X_t, a_t)\,dt + \sigma(X_t, a_t)\,dW_t
$$

- $b$: drift(결정적 변화)
- $\sigma$: diffusion(확산 스케일)
- $W_t$: 표준 브라운 운동
- $A(x,a) := \sigma(x,a)\sigma(x,a)^\top$ (확산 행렬)

목적은 할인율 $\rho>0$ 하에서 보상 $r$의 할인 적분을 최대화하는 것이다.

$$
V(x)
:=
\sup_{a_\cdot}
\mathbb{E}_x\!\left[
\int_{0}^{\infty} e^{-\rho t}\,r(X_t, a_t)\,dt
\right]
$$

### 2.2 DPP(동적계획 원리): “작은 시간 $\Delta t$”에서의 1-step 분해
DPP는 다음을 말한다(표준 가정: $V$가 충분히 매끄럽고, $r,b,\sigma$가 적절한 정칙성을 가진다).

$$
V(x)
=
\sup_{a_\cdot}
\mathbb{E}_x\!\left[
\int_{0}^{\Delta t} e^{-\rho t}\,r(X_t, a_t)\,dt
+
e^{-\rho \Delta t}\,V(X_{\Delta t})
\right]
$$

### 2.3 생성자(Generator)와 Itô 전개: 2차 항이 “자동으로” 나온다
제어 $a$가 고정되어 있을 때의 미분 연산자(생성자)를 정의한다.

$$
(\mathcal{L}^a f)(x)
:=
\sum_{i} b_i(x,a)\,\partial_i f(x)
\;+\;
\frac{1}{2}\sum_{i,j} A_{ij}(x,a)\,\partial_i\partial_j f(x)
$$

Itô 공식과 표준 계산으로, 작은 $\Delta t$에서

$$
\mathbb{E}_x\!\left[V(X_{\Delta t})\right]
=
V(x) + \Delta t\,(\mathcal{L}^a V)(x) + o(\Delta t)
$$

또한 보상 적분과 할인은

$$
\int_{0}^{\Delta t} e^{-\rho t}\,r(X_t, a_t)\,dt
=
r(x,a)\,\Delta t + o(\Delta t),
\qquad
e^{-\rho \Delta t} = 1 - \rho \Delta t + o(\Delta t)
$$

로 근사된다(정확히는 $r$의 정칙성과 $X_t$의 연속성 하에서 성립하는 표준 극한).

### 2.4 HJB 도출(정지 문제)
2.2식에 2.3식을 대입하고, $V(x)$를 소거한 뒤 $\Delta t$로 나누고 $\Delta t\to 0$을 취하면:

$$
0
=
\sup_{a\in\mathcal{A}}
\Big(
r(x,a)
(\mathcal{L}^a V)(x)
-\rho V(x)
\Big)
$$

이 식이 (무한 시간 지평/정지) HJB이다.  
핵심은 $\mathcal{L}^a$에 2차 미분 항이 있으므로, **확산이 있으면 HJB는 2차 PDE가 된다는 점**이다.

---

## 3. $\Delta$에서 $\Delta_g$로: “기하 일반화”가 왜 LBO인가
### 3.1 유클리드에서 라플라시안 $\Delta$
확산이 등방성이라 $A(x,a) = 2D(x,a)\,I$이면,

$$
\frac{1}{2}\mathrm{Tr}\!\left(A\nabla^2 V\right)
=
\frac{1}{2}\mathrm{Tr}\!\left(2D I\nabla^2 V\right)
=
D\,\Delta V
$$

즉, 2차 항이 라플라시안으로 정리된다.

### 3.2 다양체에서 라플라스–벨트라미 $\Delta_g$
상태공간이 리만 다양체 $(\mathcal{M}, g)$이면, “등방성 확산”은 메트릭 $g$에 대해 정의된다.  
이때 유클리드의 $\Delta$는 좌표 불변 연산자로의 일반화가 필요하고, 그 표준이 $\Delta_g$이다.

특히 “등방성 확산”이 $A(x,a)$가 메트릭에 비례하는 형태(직관적으로 $A \propto g^{-1}$)로 주어지면,
HJB의 확산항은 $D\,\Delta_g V$로 나타난다.

요약하면:
- Bellman/DPP를 연속화하면 HJB가 된다.
- 확산이 있으면 HJB에 2차항이 생긴다.
- 유클리드에서는 그 2차항이 $\Delta$, 다양체에서는 $\Delta_g$가 된다.

따라서 “벨만의 일반화가 LBO”라는 문장은,
**연속화 + 확산 포함 + 좌표 불변 기하 일반화**라는 의미를 포함하는 경우에만 정확하다.

---

## 4. 이 레포에서의 LBO: 설계공간 승률장 $P(x)$의 곡률 억제
이 레포의 바깥 레벨(설계 최적화)은 설계변수 $x$로부터 “게임/팩션/맵”이 결정되고,  
그 결과로 관측되는 승률(또는 승률행렬)을 black-box로 평가한다.

### 4.1 왜 설계공간에서 라플라시안을 보나
$P(x)$가 설계공간에서 뾰족하면:
- 작은 설계 변화가 승률을 급변시킨다(불안정/비강건).
- 정수화/클램프(투영) 노이즈에 취약하다.
- 평가 노이즈(학습/매칭 시드)에 취약하다.

따라서 $P(x)$의 곡률(라플라시안 크기)을 억제하는 정규화는 “튼튼한 밸런스”를 선호하게 만든다.

### 4.2 구현: 무작위 방향 2차차분(중심 차분) + CRN
설계공간에서 임의 방향 $e$에 대해:

$$
\Delta P(x; e)
\approx
\frac{
P(\Pi(x+\sigma e))
+
P(\Pi(x-\sigma e))
-
2P(\Pi(x))
}{
\sigma^2\|e\|^2
}
$$

- $\Pi(\cdot)$: 이산/혼합 설계변수의 정수화·클램프 투영(코드에서 고정 규칙)
- CRN(공통 난수): $\Pi(x),\Pi(x\pm\sigma e)$를 **같은 seed 규칙**로 평가해 분산을 줄임

### 4.3 ES 업데이트에서의 사용(가중치로 곱함)
레포 구현은 $\lvert \Delta P\rvert$가 큰 방향의 업데이트를 약하게 만든다.

$$
w
=
\frac{1}{1+\lambda\lvert \Delta P\rvert},
\qquad
\hat g \leftarrow w\cdot \hat g
$$

즉, LBO는 “문구”가 아니라 **업데이트 식에 직접 반영**된다.

---

## 5. 체크리스트(문서/코드 일치 조건)
아래가 성립하면 “LBO를 제대로 쓴다”고 말할 수 있다.
- $\Delta P$ 계산에 $P(\Pi(x))$(center)가 실제로 들어간다.
- $\Delta P$가 update에 직접 반영된다(상쇄되지 않는다).
- CRN으로 후보 비교 분산을 낮춘다.
- 투영 규칙 $\Pi$가 실험 내에서 고정되어 있다.

---

## 참고(최소)
- Fleming & Soner, *Controlled Markov Processes and Viscosity Solutions*
- Øksendal, *Stochastic Differential Equations*
- Jost, *Riemannian Geometry and Geometric Analysis* (Laplacian/Laplace–Beltrami)

