"""证明 LinearEnv.copy() 会推进全局 RNG（因为 __init__ 末尾调 reset() 抽了一整局新牌库）。"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dgisim.env.linear_env import LinearEnv  # noqa: E402


def pos():
    """Mersenne Twister 状态数组里的当前位置索引，用来观察消耗量。"""
    return random.getstate()[1][-1]


random.seed(7)
env = LinearEnv()
p0 = pos()
env.copy()
p1 = pos()
print(f"seed;env -> 位置 {p0}；env.copy() -> 位置 {p1}；copy 消耗随机数 = {p1 - p0}")

# 等价性：copy() 与"再初始化一局"消耗量是否一致
random.seed(7)
env = LinearEnv()
env.copy()
pa = pos()

random.seed(7)
env = LinearEnv()
LinearEnv()          # 一局完整初始化
pb = pos()

print(f"copy() 之后位置 {pa}；'再开一局'之后位置 {pb}；等价 = {pa == pb}")

# 后续随机流是否被污染（这才是致命的地方）
random.seed(7)
env = LinearEnv()
base = [round(random.random(), 9) for _ in range(3)]

random.seed(7)
env = LinearEnv()
env.copy()
after = [round(random.random(), 9) for _ in range(3)]

print(f"不做 copy 的后续流 : {base}")
print(f"做了 copy 的后续流 : {after}")
print(f"后续流被改变       : {base != after}")
