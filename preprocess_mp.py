# preprocess_mp.py
"""多进程预计算子进程入口（避免 __main__ 在 spawn 下无法 pickle）。"""


def preprocess_fold_worker(task_kwargs):
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import preprocess_fold

    preprocess_fold(**task_kwargs)
