# pairwise-atb-turn-queue

## 领域

游戏、图形与多媒体

## 主题

回合行动队列与动画锁

## 说明

同速稳定排序、代际校验与逻辑帧超时。

初始状态下 `gameplay/atb_turn_queue.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
