import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, List
from src.core.simulation import run_simulation, run_simulation_all_pairs

def wasserstein_distance_1d(u_values, v_values):
    """
    1차원 경험분포 간 2-Wasserstein 거리의 제곱(W2^2) 근사.

    1D에서 동일 가중치 표본(경험분포)은 정렬 후 평균 제곱차로 W2^2를 계산할 수 있습니다.
    (관련 설명: docs/blackbox.md, docs/evaluation.md)
    """
    u_sorted = np.sort(np.asarray(u_values, dtype=np.float64))
    v_sorted = np.sort(np.asarray(v_values, dtype=np.float64))
    
    if u_sorted.size == 0 or v_sorted.size == 0:
        return 0.0

    # 샘플 수가 다르면 보간법 사용해야 하나, 여기서는 편의상 샘플링으로 맞춤
    min_len = int(min(int(u_sorted.size), int(v_sorted.size)))
    if min_len <= 0:
        return 0.0
    # 균등 간격으로 샘플링하여 길이 맞춤
    u_indices = np.linspace(0, int(u_sorted.size) - 1, min_len).astype(np.int64)
    v_indices = np.linspace(0, int(v_sorted.size) - 1, min_len).astype(np.int64)
    
    u_resampled = u_sorted[u_indices]
    v_resampled = v_sorted[v_indices]
    
    diff = u_resampled - v_resampled
    return float(np.mean(diff * diff))


def _normal_quantiles(*, mean: float, std: float, n: int, clamp_min: float | None = None) -> np.ndarray:
    """
    재현 가능한 목표 분포 표본 생성:
    - 난수 샘플링 대신, (0,1) 균등 격자 quantile을 정규분포 inverse-CDF로 변환합니다.
    - SciPy 없이 torch.erfinv를 사용합니다.
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if std <= 0:
        raise ValueError("std must be > 0")

    with torch.no_grad():
        # (0,1)에서 0/1을 피하는 균등 격자: (i+0.5)/n
        p = (torch.arange(int(n), dtype=torch.float64) + 0.5) / float(n)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)  # Φ^{-1}(p)
        q = float(mean) + float(std) * z
        if clamp_min is not None:
            q = torch.clamp(q, min=float(clamp_min))
        return q.cpu().numpy().astype(np.float64, copy=False)

from tqdm import tqdm
import concurrent.futures
import os

class DesignOptimizer:
    """
    blackbox.md 기반의 ES 최적화기
    목표: 승률 50:50 + 목표 교전 거리 분포
    """
    def __init__(
        self,
        initial_design: Dict[str, float],
        target_dist_mean: float,
        sigma: float = 0.1,
        lr: float = 0.1,
        n_samples: int = 4,
        train_episodes: int = 80,
        eval_episodes: int = 6,
        use_parallel: bool = True,
        max_workers: int = 0,
        base_seed: int = 42,
        verbose: bool = True,
    ):
        self.mean_design = initial_design.copy()
        self.target_dist_mean = target_dist_mean
        self.sigma = sigma # 탐색 노이즈 표준편차
        self.lr = lr       # 학습률
        self.n_samples = n_samples # ES 샘플 수 (짝수 권장)
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.use_parallel = use_parallel
        self.base_seed = base_seed
        self.verbose = verbose

        if max_workers and max_workers > 0:
            self.max_workers = max_workers
        else:
            # 너무 과한 프로세스 생성 방지
            cpu = os.cpu_count() or 1
            self.max_workers = min(4, max(1, cpu // 2))
        
        # 최적화할 키 목록 (맵/유닛수/패턴/이동/사거리/스탯)
        self.optimizable_keys = [
            "width",
            "height",
            # 장애물(맵 기하)
            "obstacle_density",
            "obstacle_pattern",
            # 유닛 수 (킹은 고정 1개라 제외)
            "p0_unit0_units",
            "p0_unit1_units",
            "p0_unit2_units",
            "p0_unit3_units",
            "p0_unit4_units",
            "p1_unit0_units",
            "p1_unit1_units",
            "p1_unit2_units",
            "p1_unit3_units",
            "p1_unit4_units",
            # 이동 패턴 ID (0~11, ES가 탐색)
            "p0_unit0_pattern",
            "p0_unit1_pattern",
            "p0_unit2_pattern",
            "p0_unit3_pattern",
            "p0_unit4_pattern",
            "p1_unit0_pattern",
            "p1_unit1_pattern",
            "p1_unit2_pattern",
            "p1_unit3_pattern",
            "p1_unit4_pattern",
            # 공격 패턴 ID (0~12, ES가 탐색)
            "p0_unit0_attack_pattern",
            "p0_unit1_attack_pattern",
            "p0_unit2_attack_pattern",
            "p0_unit3_attack_pattern",
            "p0_unit4_attack_pattern",
            "p1_unit0_attack_pattern",
            "p1_unit1_attack_pattern",
            "p1_unit2_attack_pattern",
            "p1_unit3_attack_pattern",
            "p1_unit4_attack_pattern",
            # 이동거리
            "p0_unit0_move",
            "p0_unit1_move",
            "p0_unit2_move",
            "p1_unit0_move",
            "p1_unit1_move",
            "p1_unit2_move",
            "p1_unit4_move",
            # 사거리
            "p0_unit0_range",
            "p0_unit1_range",
            "p0_unit2_range",
            "p0_unit3_range",
            "p0_unit4_range",
            "p1_unit0_range",
            "p1_unit1_range",
            "p1_unit2_range",
            "p1_unit3_range",
            "p1_unit4_range",
            # 스탯
            "p0_unit0_hp",
            "p0_unit1_hp",
            "p0_unit0_damage",
            "p1_unit0_hp",
            "p1_unit1_hp",
            "p1_unit0_damage",
            "p1_unit1_damage",
            "p1_unit2_damage",
            "p1_unit3_hp",
            "p1_unit4_damage",
        ]

    def get_loss(self, stats: Dict) -> float:
        # 다팩션 지원: 모든 팩션 쌍의 승률이 0.5에 가깝도록
        n_factions = int(stats.get("n_factions", 2))
        
        if n_factions > 2 and "win_matrix" in stats:
            # 다팩션: 모든 쌍의 승률 균형
            win_matrix = stats["win_matrix"]
            win_diff_sum = 0.0
            blowout_count = 0
            n_pairs = 0
            
            for (i, j), win_rate in win_matrix.items():
                if i < j:  # 중복 방지
                    win_diff_sum += (win_rate - 0.5) ** 2
                    if win_rate <= 0.05 or win_rate >= 0.95:
                        blowout_count += 1
                    n_pairs += 1
            
            win_diff = win_diff_sum / max(1, n_pairs)
            blowout_penalty = 2.0 * blowout_count
        else:
            # 2팩션: 기존 방식 (p0_win_rate가 없으면 0.5로 처리)
            p0 = float(stats.get("p0_win_rate", 0.5))
            win_diff = (p0 - 0.5) ** 2
            blowout_penalty = 0.0
            if p0 <= 0.05 or p0 >= 0.95:
                blowout_penalty = 2.0
        
        # 2. 분포 정합 손실: Wasserstein 거리
        dist_samples = stats.get("distance_samples", [])
        if dist_samples:
            target_samples = _normal_quantiles(mean=float(self.target_dist_mean), std=1.0, n=len(dist_samples), clamp_min=0.0)
            w2_dist = wasserstein_distance_1d(dist_samples, target_samples)
        else:
            w2_dist = 0.0

        # 2-b. 교전이 아예 없으면 강한 페널티
        no_engagement_penalty = 0.0
        if len(dist_samples) == 0:
            no_engagement_penalty = 10.0
        
        # 3. 무승부 페널티
        draw_penalty = float(stats["draw_rate"]) * 120.0
        
        # 총 손실
        total_loss = win_diff * 40.0 + blowout_penalty + w2_dist * 1.0 + draw_penalty + no_engagement_penalty
        return total_loss

    def get_harmonic_loss(self, stats_pos: Dict, stats_neg: Dict) -> float:
        """
        라플라스-벨트라미(Laplace-Beltrami) 정규화:
        승률 지형의 곡률(Curvature)을 페널티로 부과합니다.
        P(x+e) + P(x-e) ~ 1.0 (Harmonic condition at P=0.5)
        """
        p0_pos = float(stats_pos.get("p0_win_rate", 0.5))
        p0_neg = float(stats_neg.get("p0_win_rate", 0.5))
        
        # 1.0에서 벗어난 정도가 곧 Laplacian의 크기 (2차 미분 근사)
        curvature = abs((p0_pos + p0_neg) - 1.0)
        return curvature * 10.0

    def get_lbo_curvature(self, stats_center: Dict, stats_pos: Dict, stats_neg: Dict, eps_norm2: float) -> float:
        """
        설계공간에서의 LBO(라플라시안) 근사:
          ΔP(x) ≈ (P(x+σe) + P(x-σe) - 2P(x)) / (σ^2 ||e||^2)

        - stats_*: 동일 seed(=CRN)로 평가된 통계
        - eps_norm2: ||e||^2 (epsilon 벡터의 제곱노름)
        """
        denom = float(self.sigma * self.sigma) * float(max(1e-6, eps_norm2))

        n_factions = int(stats_center.get("n_factions", 2))
        if n_factions > 2 and "win_matrix" in stats_center:
            win_c = stats_center["win_matrix"]
            win_p = stats_pos["win_matrix"]
            win_n = stats_neg["win_matrix"]
            laps = []
            for (i, j), p_c in win_c.items():
                if i < j:
                    p_p = float(win_p.get((i, j), p_c))
                    p_n = float(win_n.get((i, j), p_c))
                    lap = (p_p + p_n - 2.0 * float(p_c)) / denom
                    laps.append(lap)
            if not laps:
                return 0.0
            return float(np.mean(np.abs(np.array(laps, dtype=np.float64))))

        p_c = float(stats_center.get("p0_win_rate", 0.5))
        p_p = float(stats_pos.get("p0_win_rate", p_c))
        p_n = float(stats_neg.get("p0_win_rate", p_c))
        lap = (p_p + p_n - 2.0 * p_c) / denom
        return float(abs(lap))

    def _clamp_design(self, d: Dict[str, float]) -> Dict[str, float]:
        """
        ES는 연속값을 뱉으므로, 격자/유닛수/스탯에 대해 최소한의 정수화/클램프를 적용합니다.
        """
        out = dict(d)

        # 맵 크기 (체스보다 크게 유지)
        w = int(round(float(out.get("width", 12))))
        h = int(round(float(out.get("height", 12))))
        w = max(10, min(24, w))
        h = max(10, min(24, h))
        # 대칭 배치/장애물 미러링을 단순하게 유지하기 위해 짝수 보드 우선
        if w % 2 == 1:
            w = w + 1 if w < 24 else w - 1
        if h % 2 == 1:
            h = h + 1 if h < 24 else h - 1
        out["width"] = int(w)
        out["height"] = int(h)

        # 장애물(밀도/패턴)
        out["obstacle_density"] = float(max(0.0, min(0.35, float(out.get("obstacle_density", 0.0)))))
        out["obstacle_pattern"] = int(max(0, min(3, int(round(float(out.get("obstacle_pattern", 0)))))))

        # 유닛 수(타입별): unit0~unit4, 0도 허용 (총합 변동 가능)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_units"
                val = int(round(float(out.get(key, 0))))
                if prefix == "p0":
                    out[key] = max(0, min(8, val))
                else:
                    out[key] = max(0, min(5, val))

        # 킹만 남는 구성을 막기 위해(퇴화 방지), 최소 1개는 유지
        for prefix in ("p0", "p1"):
            keys = [f"{prefix}_unit{i}_units" for i in range(5)]
            total = int(sum(int(out.get(k, 0)) for k in keys))
            if total <= 0:
                out[keys[0]] = 1
        
        # 이동 패턴 ID (0~11 정수)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_pattern"
                val = int(round(float(out.get(key, 0))))
                out[key] = max(0, min(11, val))

        # 공격 패턴 ID (0~12 정수). 기본은 이동 패턴을 따르도록 한다.
        for prefix in ("p0", "p1"):
            for i in range(5):
                move_key = f"{prefix}_unit{i}_pattern"
                atk_key = f"{prefix}_unit{i}_attack_pattern"
                if atk_key not in out:
                    out[atk_key] = int(out.get(move_key, 0))
                val = int(round(float(out.get(atk_key, 0))))
                out[atk_key] = max(0, min(12, val))

        # 이동 거리 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_move"
                if key in out:
                    out[key] = float(max(1.0, min(5.0, float(out.get(key, 1.0)))))
        
        # 사거리 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_range"
                if key in out:
                    out[key] = float(max(1.0, min(8.0, float(out.get(key, 1.0)))))
        
        # HP (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_hp"
                if key in out:
                    out[key] = float(max(1.0, min(10.0, float(out.get(key, 2.0)))))
        
        # 데미지 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_damage"
                if key in out:
                    out[key] = float(max(0.5, min(3.0, float(out.get(key, 1.0)))))

        # 에피소드 길이(너무 길면 속도 폭발)
        out["max_steps"] = int(max(60, min(200, int(round(float(out.get("max_steps", 120)))))))

        return out

    def _eval_pair(self, design_pos: Dict[str, float], design_neg: Dict[str, float], seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 다팩션이면 all_pairs, 아니면 기존 방식
        n_factions = int(design_pos.get("n_factions", 2))
        if n_factions > 2:
            stats_pos = run_simulation_all_pairs(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_neg = run_simulation_all_pairs(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        else:
            stats_pos = run_simulation(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_neg = run_simulation(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        return stats_pos, stats_neg

    def _eval_triplet(
        self,
        design_center: Dict[str, float],
        design_pos: Dict[str, float],
        design_neg: Dict[str, float],
        seed: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        동일 seed(CRN)에서 center/pos/neg를 함께 평가하여, 설계공간 LBO(2차차분)를 계산할 수 있게 한다.
        """
        n_factions = int(design_center.get("n_factions", 2))
        if n_factions > 2:
            stats_c = run_simulation_all_pairs(design_center, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_p = run_simulation_all_pairs(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_n = run_simulation_all_pairs(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        else:
            stats_c = run_simulation(design_center, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_p = run_simulation(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_n = run_simulation(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        return stats_c, stats_p, stats_n

    def _sample_antithetic_pairs(self, step_index: int) -> Tuple[List[Tuple[int, Dict[str, float], Dict[str, float], int]], List[Dict[str, float]]]:
        """
        Antithetic sampling(+/-)을 구성합니다.
        반환:
          - pairs: (sample_index, design_pos, design_neg, seed)
          - epsilons: epsilon 벡터(dict) 목록 (index로 접근)
        """
        pairs: List[Tuple[int, Dict[str, float], Dict[str, float], int]] = []
        epsilons: List[Dict[str, float]] = []

        half = int(self.n_samples // 2)
        for i in range(half):
            epsilon = {k: float(np.random.randn()) for k in self.optimizable_keys}
            epsilons.append(epsilon)

            design_pos = self.mean_design.copy()
            design_neg = self.mean_design.copy()
            for k in self.optimizable_keys:
                base = float(self.mean_design.get(k, 0.0))
                design_pos[k] = base + float(self.sigma) * float(epsilon[k])
                design_neg[k] = base - float(self.sigma) * float(epsilon[k])

            design_pos = self._clamp_design(design_pos)
            design_neg = self._clamp_design(design_neg)

            seed = int(self.base_seed + step_index * 1000 + i)
            pairs.append((i, design_pos, design_neg, seed))

        return pairs, epsilons

    def _evaluate_triplets(self, pairs: List[Tuple[int, Dict[str, float], Dict[str, float], int]]):
        """
        center/pos/neg를 같은 seed(CRN)로 평가합니다.
        반환: (sample_index, stats_center, stats_pos, stats_neg) 목록(정렬됨)
        """
        if not pairs:
            return []

        # tqdm: pair 단위로 진행률 표시
        pbar = tqdm(total=len(pairs), desc="ES Eval", leave=False)

        def _serial_eval():
            out = []
            for i, dpos, dneg, seed in pairs:
                stats_c, stats_pos, stats_neg = self._eval_triplet(self.mean_design, dpos, dneg, seed)
                out.append((i, stats_c, stats_pos, stats_neg))
                pbar.update(1)
            return out

        n_factions = int(self.mean_design.get("n_factions", 2))
        sim_func = run_simulation_all_pairs if n_factions > 2 else run_simulation

        eval_results = []
        if self.use_parallel and self.max_workers > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = {
                        ex.submit(sim_func, self.mean_design, self.train_episodes, self.eval_episodes, seed): (i, "center")
                        for i, _, _, seed in pairs
                    }
                    futures.update(
                        {
                            ex.submit(sim_func, dpos, self.train_episodes, self.eval_episodes, seed): (i, "pos")
                            for i, dpos, _, seed in pairs
                        }
                    )
                    futures.update(
                        {
                            ex.submit(sim_func, dneg, self.train_episodes, self.eval_episodes, seed): (i, "neg")
                            for i, _, dneg, seed in pairs
                        }
                    )

                    tmp: Dict[int, Dict[str, Any]] = {}
                    done_pair = {i: 0 for i, _, _, _ in pairs}
                    for fut in concurrent.futures.as_completed(futures):
                        i, sign = futures[fut]
                        stats = fut.result()
                        if i not in tmp:
                            tmp[i] = {}
                        tmp[i][sign] = stats
                        done_pair[i] += 1
                        if done_pair[i] == 3:
                            eval_results.append((i, tmp[i]["center"], tmp[i]["pos"], tmp[i]["neg"]))
                            pbar.update(1)
            except Exception:
                eval_results = _serial_eval()
        else:
            eval_results = _serial_eval()

        pbar.close()
        eval_results.sort(key=lambda x: x[0])
        return eval_results

    def _accumulate_es_gradients(
        self,
        eval_results: list,
        epsilons: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], list]:
        """
        평가 결과를 ES 그라디언트 누적과 로깅용 결과로 변환합니다.
        반환:
          - gradients: optimizable_keys별 누적 그라디언트(sum)
          - results: (loss_pos, loss_neg, lbo) 목록
        """
        gradients = {k: 0.0 for k in self.optimizable_keys}
        results = []

        inv_denom = 1.0 / float(2.0 * float(self.sigma))
        for i, stats_center, stats_pos, stats_neg in eval_results:
            epsilon = epsilons[int(i)]
            eps_norm2 = float(sum(float(v) * float(v) for v in epsilon.values()))

            loss_pos = self.get_loss(stats_pos)
            loss_neg = self.get_loss(stats_neg)

            lbo = self.get_lbo_curvature(stats_center, stats_pos, stats_neg, eps_norm2=eps_norm2)
            lbo_w = 1.0 / (1.0 + 1.0 * float(lbo))

            if self.verbose:
                print(
                    f"    Sample {i+1} (+): Loss={loss_pos:.4f} LBO={lbo:.4f} w={lbo_w:.3f} | P0={stats_pos['p0_win_rate']:.2f} "
                    f"P1={stats_pos['p1_win_rate']:.2f} Draw={stats_pos['draw_rate']:.2f} Dist={stats_pos['avg_distance']:.2f}"
                )
                print(
                    f"    Sample {i+1} (-): Loss={loss_neg:.4f} LBO={lbo:.4f} w={lbo_w:.3f} | P0={stats_neg['p0_win_rate']:.2f} "
                    f"P1={stats_neg['p1_win_rate']:.2f} Draw={stats_neg['draw_rate']:.2f} Dist={stats_neg['avg_distance']:.2f}"
                )

            diff = float(loss_pos) - float(loss_neg)
            for k in self.optimizable_keys:
                gradients[k] += float(lbo_w) * diff * float(epsilon[k]) * inv_denom

            results.append((loss_pos, loss_neg, lbo))

        return gradients, results

    def step(self, step_index: int = 0):
        # ES (Score Function Estimator)
        # J_sigma(x) ~ E[J(x + sigma*epsilon)]
        pairs, epsilons = self._sample_antithetic_pairs(step_index=step_index)
        eval_results = self._evaluate_triplets(pairs)
        gradients, results = self._accumulate_es_gradients(eval_results, epsilons)

        # 평균 그라디언트로 업데이트
        denom = float(max(1, int(self.n_samples // 2)))
        avg_grad = {k: float(v) / denom for k, v in gradients.items()}

        for k in self.optimizable_keys:
            base = float(self.mean_design.get(k, 0.0))
            self.mean_design[k] = base - float(self.lr) * float(avg_grad[k])

        # mean 자체도 클램프 (폭주/0으로 붕괴 방지)
        self.mean_design = self._clamp_design(self.mean_design)
            
        # 첫 원소(loss_pos)의 평균으로 로깅용 스칼라를 만든다
        avg_loss = float(np.mean([r[0] for r in results])) if results else 0.0
        return avg_loss, self.mean_design

