# src/curriculum.py
import math
from types import SimpleNamespace

def _to_namespace(obj):
    """递归将 dict 转成 SimpleNamespace，支持点号访问"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


class Curriculum:
    def __init__(self, args):
        # ✅ 新增：支持 dict 传入
        if isinstance(args, dict):
            args = _to_namespace(args)

        # args.dims and args.points each contain start, end, inc, interval attributes
        self.n_dims_truncated = args.dims.start  # 初始化起始维度
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims # 
        self.n_points_schedule = args.points 
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):  # 每隔 interval 增加 inc
        if self.step_count % schedule.interval == 0:
            var += schedule.inc
        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor(total_steps / n_steps) * inc
    return min(final_var, lim)
