import sys

from src.core.designer import DesignOptimizer
from src.core.simulation import run_simulation

def main():
    # Windows 터미널에서 한글 출력 깨짐 방지
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("LB-IGD Started (Laplace–Beltrami Inverse Game Design)")
    print("Usage: main.py [--opt] [--slow]")

    mode = "opt" if "--opt" in sys.argv else "eval"
    fast = "--slow" not in sys.argv

    # 체스형 프리셋(말 개수 + 행마 중심)
    # 매핑: unit0=pawn, unit1=rook, unit2=knight, unit3=bishop, unit4=queen, king=king
    chess_design = {
        "n_factions": 2,
        "width": 12,
        "height": 12,
        "obstacle_density": 0.0,
        "obstacle_pattern": 0,

        # 말 개수(체스 표준)
        "p0_unit0_units": 8,
        "p0_unit1_units": 2,
        "p0_unit2_units": 2,
        "p0_unit3_units": 2,
        "p0_unit4_units": 1,

        "p1_unit0_units": 8,
        "p1_unit1_units": 2,
        "p1_unit2_units": 2,
        "p1_unit3_units": 2,
        "p1_unit4_units": 1,

        # 행마(0~11)
        "p0_unit0_pattern": 11,  # pawn(전진)
        "p0_unit1_pattern": 0,   # rook(직선 슬라이드)
        "p0_unit2_pattern": 4,   # knight
        "p0_unit3_pattern": 1,   # bishop(대각 슬라이드)
        "p0_unit4_pattern": 2,   # queen(전방향 슬라이드)

        "p1_unit0_pattern": 11,
        "p1_unit1_pattern": 0,
        "p1_unit2_pattern": 4,
        "p1_unit3_pattern": 1,
        "p1_unit4_pattern": 2,

        # 이동 거리(슬라이드는 길게, 점프는 1)
        "p0_unit0_move": 1.0,
        "p0_unit1_move": 5.0,
        "p0_unit2_move": 1.0,
        "p0_unit3_move": 5.0,
        "p0_unit4_move": 5.0,

        "p1_unit0_move": 1.0,
        "p1_unit1_move": 5.0,
        "p1_unit2_move": 1.0,
        "p1_unit3_move": 5.0,
        "p1_unit4_move": 5.0,

        # 공격 패턴(0~12): pawn_diag(12), rook(0), knight(4), bishop(1), queen(2)
        "p0_unit0_attack_pattern": 12,
        "p0_unit1_attack_pattern": 0,
        "p0_unit2_attack_pattern": 4,
        "p0_unit3_attack_pattern": 1,
        "p0_unit4_attack_pattern": 2,

        "p1_unit0_attack_pattern": 12,
        "p1_unit1_attack_pattern": 0,
        "p1_unit2_attack_pattern": 4,
        "p1_unit3_attack_pattern": 1,
        "p1_unit4_attack_pattern": 2,

        # (체스형) 공격 사거리: 슬라이딩 말은 길게, 점프/폰은 1
        "p0_unit0_range": 1.0,
        "p0_unit1_range": 8.0,
        "p0_unit2_range": 1.0,
        "p0_unit3_range": 8.0,
        "p0_unit4_range": 8.0,
        "p0_king_range": 1.0,

        "p1_unit0_range": 1.0,
        "p1_unit1_range": 8.0,
        "p1_unit2_range": 1.0,
        "p1_unit3_range": 8.0,
        "p1_unit4_range": 8.0,
        "p1_king_range": 1.0,

        # (체스형) 한 번 맞으면 제거
        "p0_unit0_hp": 1.0,
        "p0_unit1_hp": 1.0,
        "p0_unit2_hp": 1.0,
        "p0_unit3_hp": 1.0,
        "p0_unit4_hp": 1.0,
        "p0_king_hp": 1.0,

        "p1_unit0_hp": 1.0,
        "p1_unit1_hp": 1.0,
        "p1_unit2_hp": 1.0,
        "p1_unit3_hp": 1.0,
        "p1_unit4_hp": 1.0,
        "p1_king_hp": 1.0,

        "p0_unit0_damage": 1.0,
        "p0_unit1_damage": 1.0,
        "p0_unit2_damage": 1.0,
        "p0_unit3_damage": 1.0,
        "p0_unit4_damage": 1.0,
        "p0_king_damage": 1.0,

        "p1_unit0_damage": 1.0,
        "p1_unit1_damage": 1.0,
        "p1_unit2_damage": 1.0,
        "p1_unit3_damage": 1.0,
        "p1_unit4_damage": 1.0,
        "p1_king_damage": 1.0,

        "max_steps": 120,
        "no_attack_limit": 80,
        "shaping_scale": 0.05,
    }

    if mode == "eval":
        train_episodes = 30 if fast else 200
        eval_episodes = 30 if fast else 120
        seeds = [0, 1, 2] if fast else [0, 1, 2, 3, 4]

        win_rates = []
        draw_rates = []
        avg_dists = []

        for seed in seeds:
            stats = run_simulation(chess_design, train_episodes=train_episodes, eval_episodes=eval_episodes, seed=seed)
            win_rates.append(float(stats["p0_win_rate"]))
            draw_rates.append(float(stats["draw_rate"]))
            avg_dists.append(float(stats["avg_distance"]))

        w = float(sum(win_rates) / max(1, len(win_rates)))
        var = float(sum((x - w) ** 2 for x in win_rates) / max(1, len(win_rates)))
        d = float(sum(draw_rates) / max(1, len(draw_rates)))
        dist = float(sum(avg_dists) / max(1, len(avg_dists)))

        print("-" * 50)
        print("Chess-like design evaluation")
        print(f"  seeds         : {seeds}")
        print(f"  train_episodes: {train_episodes}")
        print(f"  eval_episodes : {eval_episodes}")
        print(f"  p0_win_mean   : {w:.3f}")
        print(f"  p0_win_var    : {var:.4f}")
        print(f"  draw_mean     : {d:.3f}")
        print(f"  avg_dist_mean : {dist:.3f}")
        return

    # 1) 초기 설계 (팩션 체스: 말 개수/행마/맵/장애물까지 모두 설계변수)
    initial_design = {
        "n_factions": 2,
        "width": 12,
        "height": 12,
        "obstacle_density": 0.12,
        "obstacle_pattern": 1,

        # P0 (예: 물량형)
        "p0_unit0_units": 5,
        "p0_unit1_units": 3,
        "p0_unit2_units": 1,
        "p0_unit3_units": 2,
        "p0_unit4_units": 1,

        # P1 (예: 기동/화력형)
        "p1_unit0_units": 3,
        "p1_unit1_units": 2,
        "p1_unit2_units": 2,
        "p1_unit3_units": 1,
        "p1_unit4_units": 1,

        # 이동 패턴(0~11)
        "p0_unit0_pattern": 0,
        "p0_unit1_pattern": 2,
        "p0_unit2_pattern": 4,
        "p0_unit3_pattern": 0,
        "p0_unit4_pattern": 0,

        "p1_unit0_pattern": 0,
        "p1_unit1_pattern": 2,
        "p1_unit2_pattern": 5,
        "p1_unit3_pattern": 7,
        "p1_unit4_pattern": 9,

        # 이동/사거리/스탯
        "p0_unit0_move": 2.0,
        "p0_unit1_move": 1.0,
        "p0_unit2_move": 1.0,

        "p1_unit0_move": 3.0,
        "p1_unit1_move": 3.0,
        "p1_unit2_move": 2.0,
        "p1_unit4_move": 3.0,

        "p0_unit1_range": 3.0,
        "p0_unit4_range": 4.0,
        "p1_unit1_range": 5.0,
        "p1_unit4_range": 6.0,

        "p0_unit0_hp": 3.0,
        "p0_unit1_hp": 2.0,
        "p0_unit0_damage": 1.0,

        "p1_unit0_hp": 4.0,
        "p1_unit1_hp": 3.0,
        "p1_unit0_damage": 1.5,
        "p1_unit1_damage": 1.5,
        "p1_unit2_damage": 1.5,
        "p1_unit3_hp": 7.0,
        "p1_unit4_damage": 2.2,

        "max_steps": 90,
        "no_attack_limit": 60,
        "shaping_scale": 0.05,
    }

    target_dist_mean = 3.0
    train_episodes = 12 if fast else 250
    eval_episodes = 8 if fast else 20
    n_samples = 4 if fast else 8

    optimizer = DesignOptimizer(
        initial_design,
        target_dist_mean=target_dist_mean,
        sigma=0.2,
        lr=0.1,
        n_samples=n_samples,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        use_parallel=True,
        max_workers=0,
        base_seed=42,
        verbose=True,
    )

    outer_steps = 6 if fast else 20
    for step in range(1, outer_steps + 1):
        print(f"\n[Step {step}] Starting Optimization...")
        loss, current_design = optimizer.step(step_index=step)
        design_str = ", ".join(
            [f"{k}: {v:.2f}" for k, v in current_design.items() if k in optimizer.optimizable_keys]
        )
        print(f"Step {step:2d} | Loss: {loss:.4f} | Design: {design_str}")

    print("-" * 50)
    print("Optimization Finished.")
    print("Final Design:", current_design)

if __name__ == "__main__":
    main()
