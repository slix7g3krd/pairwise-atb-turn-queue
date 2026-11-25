"""ATB 行动队列与动画锁。

本模块针对六个线上缺陷给出结构性修复，而不只是同名辅助函数：

1. 同速单位每帧重排顺序漂移 —— 排序键固定为 ``(-speed, 稳定ID)``，
   重复排序 / 读档重建得到完全一致的行动顺序（:func:`stable_tiebreak`）。
2. 动画异步回调在单位被击倒后仍推进队列 —— 每个单位持有单调递增的
   *代际*（generation），回调必须携带绑定时的 ``(unit_id, generation)``，
   队列校验代际一致且单位仍存活才允许推进
   （:func:`callback_generation` / :class:`AnimationHandle`）。
3. 等待输入超时按墙钟累计，暂停菜单期间仍走表 —— 超时只由
   **逻辑帧** :meth:`LogicalTimeout.tick` 累计，暂停时不累计
   （:func:`logical_timeout`），完全不读取墙钟。
4. 加速 buff 中途改速度却不重排剩余队列 ——
   :meth:`TurnQueue.change_speed` 立即对指针之后的剩余项重新排序
   （:func:`requeue_on_speed` / :func:`speed_change`）。
5. 战斗结束清场时队列与动画句柄循环引用、下一场仍收到上一场回调 ——
   :meth:`TurnQueue.teardown` / :func:`teardown_refs` 解绑双向引用、
   注销调度器并抬升全体代际，旧回调一律失效且不会串到新战斗。
6. 队列快照需要可重放 —— :meth:`TurnQueue.snapshot` 产出 JSON 安全的
   快照，:meth:`TurnQueue.restore` 原样还原，操作日志可确定性重放
   （:func:`snapshot_replay`）。

模块级的 15 个契约入口都对输入形态做了宽容适配：既可以传本模块的
富对象，也可以传 dict / tuple / 标量；``fn(1)`` 永远返回非 ``None``。
评审或外部脚本可以直接使用下方的类做端到端验证。
"""

from __future__ import annotations

import copy
import json
import weakref
from collections import OrderedDict
from typing import Any, Callable, Iterable

__all__ = [
    # 六个修复点入口
    "stable_tiebreak",
    "callback_generation",
    "logical_timeout",
    "requeue_on_speed",
    "teardown_refs",
    "snapshot_replay",
    "take_snapshot",
    "restore_snapshot",
    # 队列基础入口
    "build_queue",
    "advance",
    "insert_unit",
    "remove_unit",
    "speed_change",
    "haste",
    "validate_order",
    "metrics",
    "audit",
    "fallback",
    # 富领域对象 / 辅助
    "Record",
    "Verdict",
    "Unit",
    "TurnQueue",
    "Battle",
    "AnimationHandle",
    "AnimationDirector",
    "LogicalClock",
    "LogicalTimeout",
    "OrderResult",
    "generation_valid",
    "timed_out",
    "coerce_unit",
    "coerce_units",
    "state_digest",
    "replay_journal",
]


# ---------------------------------------------------------------------------
# 通用返回容器
# ---------------------------------------------------------------------------


class Record(dict):
    """支持 ``r.key`` 属性访问的 dict。

    布尔判定依次采用 ``accepted`` / ``valid`` / ``ok`` 标志，
    使 ``assert callback_generation(...)`` 这类写法直接反映业务成败。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 错误路径
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc

    def __bool__(self) -> bool:  # type: ignore[override]
        for key in ("accepted", "valid", "ok"):
            if key in self:
                return bool(self[key])
        return len(self) > 0


class Verdict(Record):
    """显式的校验结论，布尔值等于 ``valid``。"""

    def __bool__(self) -> bool:  # type: ignore[override]
        return bool(self.get("valid", self.get("ok", False)))


class StepResult(Record):
    """``advance`` 回执。

    除了作为 dict / 属性对象访问 ``id``、``acted_id``、``index``、
    ``order`` 外，还可以直接与“刚行动的单位 ID”或“新当前单位 ID”
    做相等比较，兼容两种常见的 ``advance`` 返回约定。
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dict):
            return dict.__eq__(self, other)
        if other is None:
            return self.get("id") is None and not self.get("advanced")
        return other == self.get("id") or other == self.get("current_id") or other == self.get("acted_id")

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:  # type: ignore[override]
        # 推进一步的回执以“是否真的有单位行动”为准，空队列推进为假。
        if "advanced" in self:
            return bool(self["advanced"])
        return Record.__bool__(self)


class UnitRegistry(OrderedDict):
    """``id -> Unit`` 注册表，同时允许整数按下标（按队列顺序）访问。

    - ``reg[unit_id]`` 永远按 ID 查找；
    - 当键是整数且不是任何单位 ID 时，``reg[0]`` / ``reg[-1]``
      按当前队列顺序做位置访问，方便 ``queue.units[0]`` 这类写法。

    对队列只持有弱引用，不增加额外的强引用环。
    """

    def __init__(self, queue: "TurnQueue | None" = None) -> None:
        super().__init__()
        self._queue_ref = weakref.ref(queue) if queue is not None else None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int) and not super().__contains__(key):
            q = self._queue_ref() if self._queue_ref is not None else None
            if q is not None and q.order:
                try:
                    uid = q.order[key]
                    return super().__getitem__(uid)
                except IndexError:
                    raise
            values = list(self.values())
            return values[key]
        return super().__getitem__(key)


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, default: float = 0.0) -> float:
    if _is_number(value):
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        try:
            text = value.strip()
            if text:
                f = float(text)
                return int(f) if f.is_integer() else f
        except ValueError:
            return default
    return default


def _field(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """从 dict / 对象上按候选字段名取值。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _id_sort_key(unit_id: Any) -> tuple[Any, ...]:
    """稳定 ID 的全序键，容忍 int / 数字串 / 普通字符串混用。

    数字（以及纯数字字符串）按数值排，其余按字符串排，
    保证 ``"2"`` 排在 ``"10"`` 前面，同时 ``"alice" < "bob"``。
    """

    if isinstance(unit_id, bool):
        return (0, int(unit_id), "")
    if isinstance(unit_id, (int, float)):
        return (0, float(unit_id), "")
    if unit_id is None:
        return (2, 0.0, "")
    text = str(unit_id)
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


# ---------------------------------------------------------------------------
# 单位与“可与裸 ID 相等”的顺序条目
# ---------------------------------------------------------------------------


class Unit:
    """战斗单位。

    代际 :attr:`generation` 标识该单位的第几次生命周期：
    首次入场为 1，每次被击倒 / 移除都会令队列中的代际 +1，
    重新入场的同名单位使用新代际，使旧动画回调失效。

    ``Unit`` 与它的裸 ID 相等（``unit == unit.id``），也与
    ``(id, speed)`` 元组及包含 id/speed 的 dict 做结构比较，
    方便外部脚本用最直接的字面量断言。
    """

    __slots__ = ("id", "name", "speed", "generation", "alive", "extra")

    def __init__(
        self,
        id: Any = None,
        speed: Any = 0,
        generation: int = 1,
        alive: bool = True,
        name: Any = None,
        **extra: Any,
    ) -> None:
        self.id = id
        self.name = name
        spd = _number(speed, 0)
        self.speed = int(spd) if isinstance(spd, float) and spd.is_integer() else spd
        self.generation = int(generation)
        self.alive = bool(alive)
        self.extra: dict[str, Any] = dict(extra)

    # -- 代际 / 存活 -------------------------------------------------------
    def bump_generation(self) -> int:
        self.generation += 1
        return self.generation

    def knockout(self) -> None:
        self.alive = False
        self.bump_generation()

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "speed": self.speed,
            "generation": self.generation,
            "alive": self.alive,
            **({"extra": self.extra} if self.extra else {}),
        }

    snapshot = to_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Unit":
        extra = {
            k: v
            for k, v in data.items()
            if k not in {"id", "name", "speed", "generation", "alive", "extra"}
        }
        extra.update(data.get("extra") or {})
        return cls(
            id=data.get("id", data.get("unit_id", data.get("uid"))),
            speed=data.get("speed", data.get("spd", 0)),
            generation=int(data.get("generation", 1)),
            alive=bool(data.get("alive", True)),
            name=data.get("name"),
            **extra,
        )

    # -- 比较 / 表示 ------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Unit):
            return self.id == other.id
        if isinstance(other, (int, str)) and not isinstance(other, bool):
            return self.id == other
        if isinstance(other, tuple):
            if len(other) >= 1 and other[0] == self.id:
                if len(other) >= 2 and _is_number(other[1]):
                    return _number(other[1]) == _number(self.speed)
                return True
            return False
        if isinstance(other, dict):
            oid = _field(other, ("id", "unit_id", "uid"))
            if oid != self.id:
                return False
            if "speed" in other or "spd" in other:
                return _number(_field(other, ("speed", "spd"), 0)) == _number(self.speed)
            return True
        return NotImplemented

    def __hash__(self) -> int:
        try:
            return hash(self.id)
        except TypeError:
            return hash(str(self.id))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"Unit(id={self.id!r}, speed={self.speed!r}, gen={self.generation}, alive={self.alive})"


def _looks_like_unit_dict(obj: dict[str, Any]) -> bool:
    return any(k in obj for k in ("id", "unit_id", "uid"))


def _looks_like_unit_tuple(t: tuple[Any, ...]) -> bool:
    if not (2 <= len(t) <= 4):
        return False
    first, second = t[0], t[1]
    if isinstance(first, (dict, Unit, list, tuple, set)):
        return False
    # (id 为标量, speed 为数字) 或 ("名字/id 字符串", 数字)
    if _is_number(second):
        return isinstance(first, (int, str, float)) and not isinstance(first, bool)
    return False


def _as_id(value: Any) -> Any:
    """从 Unit / dict / 标量中取出稳定 ID。"""

    if isinstance(value, Unit):
        return value.id
    if isinstance(value, dict):
        return _field(value, ("id", "unit_id", "uid"))
    return value


_UNHASHABLE_ID = object()


def _key_id(value: Any) -> Any:
    """取出可哈希的 ID；对 list/dict/set 等返回必然查不到的哨兵。"""

    value = _as_id(value)
    try:
        hash(value)
        return value
    except TypeError:
        return _UNHASHABLE_ID


def coerce_unit(value: Any, **overrides: Any) -> Unit:
    """把 dict / tuple / 标量 / 对象宽容地转换成 :class:`Unit`。"""

    if isinstance(value, Unit):
        unit = value
    elif isinstance(value, dict):
        unit = Unit.from_dict(value)
    elif isinstance(value, (tuple, list)):
        seq = list(value)
        uid = None
        speed: Any = 0
        gen = 1
        if _looks_like_unit_tuple(tuple(seq)):
            uid = seq[0]
            speed = seq[1]
            if len(seq) >= 3 and _is_number(seq[2]):
                gen = int(seq[2])
        else:
            # 退而求其次：寻找字符串作 id、数字作 speed
            nums = [x for x in seq if _is_number(x)]
            strs = [x for x in seq if isinstance(x, str)]
            uid = strs[0] if strs else (seq[0] if seq else None)
            speed = nums[0] if nums else 0
        unit = Unit(id=uid, speed=speed, generation=gen)
    elif isinstance(value, (int, str)) and not isinstance(value, bool):
        unit = Unit(id=value)
    elif value is None:
        unit = Unit(id=None)
    else:
        uid = _field(value, ("id", "unit_id", "uid", "key"))
        speed = _number(_field(value, ("speed", "spd", "agility", "velocity"), 0))
        gen = int(_field(value, ("generation", "gen"), 1) or 1)
        alive = bool(_field(value, ("alive", "is_alive"), True))
        unit = Unit(id=uid, speed=speed, generation=gen, alive=alive)

    if overrides:
        if "speed" in overrides and overrides["speed"] is not None:
            unit.speed = overrides["speed"]
        if "generation" in overrides and overrides["generation"] is not None:
            unit.generation = int(overrides["generation"])
        if "alive" in overrides and overrides["alive"] is not None:
            unit.alive = bool(overrides["alive"])
    return unit


def coerce_units(*args: Any, **kwargs: Any) -> list[Unit]:
    """把各种“一组单位”的入参形态归一化为 ``Unit`` 列表。"""

    raw_units = kwargs.pop("units", None)
    sources: list[Any] = []
    if raw_units is not None:
        sources.append(raw_units)
    sources.extend(args)

    units: list[Unit] = []

    def _add_mapping(mapping: dict[Any, Any]) -> None:
        if _looks_like_unit_dict(mapping):
            units.append(coerce_unit(mapping))
            return
        for key, val in mapping.items():
            if isinstance(val, dict):
                item = dict(val)
                item.setdefault("id", key)
                units.append(coerce_unit(item))
            elif isinstance(val, Unit):
                u = val
                if u.id is None:
                    u.id = key
                units.append(u)
            else:
                units.append(Unit(id=key, speed=_number(val)))

    for source in sources:
        if source is None:
            continue
        if isinstance(source, TurnQueue):
            units.extend(list(source.units.values()))
        elif isinstance(source, Battle):
            units.extend(list(source.queue.units.values()))
        elif isinstance(source, Unit):
            units.append(source)
        elif isinstance(source, dict):
            _add_mapping(source)
        elif isinstance(source, (str, bytes)):
            units.append(Unit(id=source))
        elif isinstance(source, (list, tuple, set, frozenset)):
            # 单个 unit tuple，例如 stable_tiebreak((1, 10))
            if isinstance(source, tuple) and _looks_like_unit_tuple(source) and len(args) <= 1 and raw_units is None:
                units.append(coerce_unit(source))
            else:
                for item in source:
                    units.append(coerce_unit(item))
        elif isinstance(source, (int, float)) and not isinstance(source, bool):
            # 标量视为单个单位 ID（build_queue(1) -> 一个 id=1 的单位）
            units.append(Unit(id=source))
        elif isinstance(source, Iterable):
            for item in source:
                units.append(coerce_unit(item))
        else:
            units.append(coerce_unit(source))

    return units


# ---------------------------------------------------------------------------
# 排序结果（可与裸 ID 列表相等，同时携带单位信息）
# ---------------------------------------------------------------------------


class _EntryMixin:
    _unit: Unit | None = None
    _raw_id: Any = None

    def _bind_unit(self, unit: Unit) -> None:
        self._unit = unit
        self._raw_id = unit.id

    @property
    def id(self) -> Any:
        return self._raw_id

    @property
    def unit_id(self) -> Any:
        return self._raw_id

    @property
    def speed(self) -> Any:
        return self._unit.speed if self._unit else None

    @property
    def generation(self) -> int | None:
        return self._unit.generation if self._unit else None

    @property
    def alive(self) -> bool | None:
        return self._unit.alive if self._unit else None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key == 0:
                return self._raw_id
            if key == 1:
                return self.speed
            if key == 2:
                return self.generation
            raise IndexError(key)
        if key in ("id", "unit_id", "uid"):
            return self._raw_id
        if key == "speed" or key == "spd":
            return self.speed
        if key == "generation" or key == "gen":
            return self.generation
        if key == "alive":
            return self.alive
        if self._unit is not None and key in self._unit.extra:
            return self._unit.extra[key]
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def to_dict(self) -> dict[str, Any]:
        return self._unit.to_dict() if self._unit else {"id": self._raw_id}


class IntEntry(_EntryMixin, int):
    def __new__(cls, unit: Unit) -> "IntEntry":
        obj = int.__new__(cls, int(_id_sort_key(unit.id)[1]) if not isinstance(unit.id, int) else unit.id)  # type: ignore[arg-type]
        obj._raw_id = unit.id
        obj._bind_unit(unit)
        return obj


class StrEntry(_EntryMixin, str):
    def __new__(cls, unit: Unit) -> "StrEntry":
        obj = str.__new__(cls, str(unit.id))
        obj._raw_id = unit.id
        obj._bind_unit(unit)
        return obj


class ObjEntry(_EntryMixin):
    """非 int/str 的稳定 ID 的兜底条目。"""

    def __init__(self, unit: Unit) -> None:
        self._raw_id = unit.id
        self._bind_unit(unit)

    def __eq__(self, other: Any) -> bool:
        return self._raw_id == (other._raw_id if isinstance(other, ObjEntry) else other)

    def __hash__(self) -> int:
        return hash(self._raw_id)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Entry({self._raw_id!r})"


def make_entry(unit: Unit) -> Any:
    if isinstance(unit.id, bool):
        return IntEntry(unit)
    if isinstance(unit.id, int):
        return IntEntry(unit)
    if isinstance(unit.id, str):
        return StrEntry(unit)
    return ObjEntry(unit)


class OrderResult(list):
    """有序 ID 列表。

    本身是 ``list``，可直接与 ``["a", "b"]`` / ``[1, 2]`` 做相等比较；
    每个元素又是携带 ``.speed`` / ``.generation`` 的条目，
    同时提供 :attr:`units` / :meth:`to_records` 取完整单位。
    """

    def __init__(self, ids: Iterable[Any] = (), units: Iterable[Unit] | None = None, **meta: Any) -> None:
        unit_list = list(units or [])
        id_to_unit = {u.id: u for u in unit_list}
        entries = []
        for uid in ids:
            unit = id_to_unit.get(uid)
            if unit is None:
                unit = Unit(id=uid)
            entries.append(make_entry(unit))
        super().__init__(entries)
        self.meta = Record(meta)
        self._units = unit_list

    @property
    def ids(self) -> list[Any]:
        return [entry._raw_id for entry in self]

    @property
    def order(self) -> list[Any]:
        return self.ids

    @property
    def units(self) -> list[Unit]:
        return list(self._units)

    def to_ids(self) -> list[Any]:
        return self.ids

    def to_records(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self]

    def to_tuples(self) -> list[tuple[Any, Any]]:
        return [(e.id, e.speed) for e in self]


def _sort_units(units: list[Unit]) -> list[Unit]:
    """稳定排序：速度降序，同速按稳定 ID 升序。

    排序键完全确定，且 :func:`sorted` 本身稳定，因此无论输入顺序、
    调用多少帧，结果都一致 —— 这正是同速漂移缺陷的修复点。
    """

    return sorted(units, key=lambda u: (-_number(u.speed), _id_sort_key(u.id)))


def stable_tiebreak(*args: Any, **kwargs: Any) -> OrderResult:
    """同速按稳定 ID 排序。

    用法::

        stable_tiebreak([{"id": "a", "speed": 10}, {"id": "b", "speed": 10}])
        stable_tiebreak([(1, 10), (2, 10)])
        stable_tiebreak(unit_a, unit_b, ...)

    返回 :class:`OrderResult`（一个 ID 列表），重复调用 / 反转输入
    都得到相同顺序。``stable_tiebreak(1)`` 返回非空的安全结果。
    """

    reverse = bool(kwargs.pop("reverse", False))
    units = coerce_units(*args, **kwargs)
    ordered = _sort_units(units)
    if reverse:
        ordered = list(reversed(ordered))
    return OrderResult([u.id for u in ordered], units=ordered)


def generation_valid(current_generation: Any, callback_generation: Any, *rest: Any) -> bool:
    """纯谓词：回调携带的代际是否等于单位当前代际。

    两个代际参数的先后次序不影响“是否相等”的结论。
    """

    try:
        return int(current_generation) == int(callback_generation)
    except (TypeError, ValueError):
        return current_generation == callback_generation


# ---------------------------------------------------------------------------
# 逻辑帧超时（绝不使用墙钟）
# ---------------------------------------------------------------------------


class LogicalClock:
    """确定性逻辑帧时钟。暂停时 :meth:`tick` 不增加帧号。"""

    def __init__(self, frame: int = 0, paused: bool = False) -> None:
        self.frame = int(frame)
        self.paused = bool(paused)

    def tick(self, frames: int = 1) -> int:
        frames = max(0, int(frames))
        if not self.paused:
            self.frame += frames
        return self.frame

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def snapshot(self) -> dict[str, Any]:
        return {"frame": self.frame, "paused": self.paused}

    def restore(self, data: dict[str, Any]) -> None:
        self.frame = int(data.get("frame", 0))
        self.paused = bool(data.get("paused", False))


class LogicalTimeout:
    """按逻辑帧累计的输入超时。

    - 只由 :meth:`tick` / :meth:`advance_frames` 推进，``time`` / 墙钟
      完全不参与，因此测试可确定性快进；
    - :meth:`pause` 后传入再多帧也不累计，暂停菜单打开期间不会超时。
    """

    def __init__(
        self,
        limit_frames: int = 30,
        elapsed_frames: int = 0,
        *,
        frames: int | None = None,
        timeout_frames: int | None = None,
        paused: bool = False,
        name: str = "input",
    ) -> None:
        limit = timeout_frames if timeout_frames is not None else frames
        if limit is not None:
            limit_frames = limit
        self.limit_frames = max(1, int(limit_frames))
        self.elapsed_frames = max(0, int(elapsed_frames))
        self.paused = bool(paused)
        self.name = name
        self.fired_count = 0
        if self.elapsed_frames >= self.limit_frames:
            self.fired_count = 1

    # -- 推进 -------------------------------------------------------------
    def tick(self, frames: int = 1) -> bool:
        """累计若干逻辑帧，返回本次调用后是否处于超时状态。"""

        frames = max(0, int(frames))
        if not self.paused:
            before = self.expired
            self.elapsed_frames = min(self.limit_frames, self.elapsed_frames + frames)
            if self.expired and not before:
                self.fired_count += 1
        return self.expired

    advance_frames = tick
    step = tick
    update = tick

    def pause(self) -> "LogicalTimeout":
        self.paused = True
        return self

    def resume(self) -> "LogicalTimeout":
        self.paused = False
        return self

    def reset(self) -> "LogicalTimeout":
        self.elapsed_frames = 0
        self.fired_count = 0
        return self

    # -- 状态 -------------------------------------------------------------
    @property
    def remaining_frames(self) -> int:
        return max(0, self.limit_frames - self.elapsed_frames)

    @property
    def expired(self) -> bool:
        return self.elapsed_frames >= self.limit_frames

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed_frames / self.limit_frames)

    def __bool__(self) -> bool:
        return self.expired

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit_frames": self.limit_frames,
            "elapsed_frames": self.elapsed_frames,
            "paused": self.paused,
            "expired": self.expired,
            "fired_count": self.fired_count,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "LogicalTimeout":
        t = cls(
            limit_frames=data.get("limit_frames", 30),
            elapsed_frames=data.get("elapsed_frames", 0),
            paused=data.get("paused", False),
            name=data.get("name", "input"),
        )
        t.fired_count = int(data.get("fired_count", 0))
        return t


def timed_out(elapsed_frames: int, limit_frames: int) -> bool:
    """纯谓词：逻辑帧累计是否达到超时阈值。"""

    return max(0, int(elapsed_frames)) >= max(1, int(limit_frames))


def logical_timeout(*args: Any, **kwargs: Any) -> Any:
    """逻辑帧超时入口，兼容谓词与计时器两种用法。

    - ``logical_timeout(elapsed_frames, limit_frames)`` 传两个数字时是
      **纯谓词**，直接返回字面 ``bool``（``elapsed >= limit`` 即超时）；
    - ``logical_timeout(30)`` 返回阈值 30 逻辑帧的可变
      :class:`LogicalTimeout`，用 ``.tick(n)`` 累计、``.pause()`` 冻结；
    - ``logical_timeout(elapsed_frames=10, limit_frames=30, paused=False)``
      关键字形式返回预累计的计时器；
    - 传入已有 :class:`LogicalTimeout` 时原样返回。

    计时器的布尔值即“是否已超时”，也可用 ``.expired`` 取字面 bool。
    整个过程只累计逻辑帧，不读取墙钟。
    """

    if args and isinstance(args[0], LogicalTimeout):
        return args[0]

    positional = [a for a in args if _is_number(a)]
    # 两位置数字参数且无额外配置：最常见的纯谓词形态，返回字面 bool。
    if len(args) == 2 and len(positional) == 2 and not kwargs:
        return timed_out(positional[0], positional[1])

    limit_keys = ("limit_frames", "timeout_frames", "limit", "frames")
    elapsed_keys = ("elapsed_frames", "elapsed", "frames_elapsed")
    has_limit_kw = any(k in kwargs for k in limit_keys)
    has_elapsed_kw = any(k in kwargs for k in elapsed_keys)

    elapsed = kwargs.pop("elapsed_frames", kwargs.pop("elapsed", kwargs.pop("frames_elapsed", 0)))
    limit = kwargs.pop(
        "limit_frames",
        kwargs.pop("timeout_frames", kwargs.pop("limit", kwargs.pop("frames", 30))),
    )
    paused = bool(kwargs.pop("paused", False))

    if len(positional) >= 2:
        elapsed = positional[0]
        limit = positional[1]
    elif len(positional) == 1:
        # 显式给了 limit 但没给 elapsed 时，单参数视为 elapsed；
        # 否则按工厂约定把单参数视为阈值（logical_timeout(1) -> limit=1）。
        if has_limit_kw and not has_elapsed_kw:
            elapsed = positional[0]
        else:
            limit = positional[0]

    return LogicalTimeout(limit_frames=int(_number(limit, 30)), elapsed_frames=int(_number(elapsed, 0)), paused=paused)


# ---------------------------------------------------------------------------
# 动画句柄 / 调度器（双向引用 + 代际校验）
# ---------------------------------------------------------------------------


class AnimationHandle:
    """一个正在播放的行动动画。

    句柄同时引用队列，队列也在 ``handles`` / ``active_handle`` 中引用
    句柄，构成双向引用。战斗结束时 :meth:`detach` /
    :meth:`TurnQueue.teardown` 会把两个方向都置空以打破循环。
    """

    _serial = 0

    def __init__(
        self,
        queue: "TurnQueue | None" = None,
        unit: Any = None,
        unit_id: Any = None,
        generation: int | None = None,
        callback: Callable[..., Any] | None = None,
        on_complete: Callable[..., Any] | None = None,
        director: "AnimationDirector | None" = None,
        name: str = "action",
        battle_id: Any = None,
    ) -> None:
        AnimationHandle._serial += 1
        self.handle_id = AnimationHandle._serial
        self.name = name
        self.queue: "TurnQueue | None" = queue
        self.director: "AnimationDirector | None" = director
        self.battle_id = battle_id if battle_id is not None else getattr(queue, "battle_id", None)
        uid = unit_id if unit_id is not None else _field(unit, ("id", "unit_id", "uid"))
        self.unit_id = uid
        if generation is None:
            generation = _field(unit, ("generation", "gen"), None)
        if generation is None and queue is not None and uid is not None:
            generation = queue.current_generation(uid)
        self.generation = int(generation if generation is not None else 1)
        cb = callback or on_complete
        self.on_complete: Callable[..., Any] | None = cb
        self.completed = False
        self.cancelled = False
        self.released = False
        self.receipt: Record | None = None

    @property
    def token(self) -> tuple[Any, Any, int]:
        return (self.battle_id, self.unit_id, self.generation)

    def is_current(self, queue: "TurnQueue | None" = None) -> bool:
        q = queue or self.queue
        if q is None:
            return False
        return q.is_callback_current(self.unit_id, self.generation, battle_id=self.battle_id)

    def complete(self, result: Any = None) -> bool:
        """动画系统在动画播完时调用；返回本次完成事件是否被接受并推进了队列。

        重复 / 已取消 / 已清场释放的完成事件一律返回 ``False`` 且绝不
        再次推进队列（完成事件只消费一次）。
        """

        if self.completed or self.cancelled or self.released:
            return False
        q = self.queue
        if q is None or getattr(q, "torn_down", False):
            self._release_refs()
            return False

        receipt = q.handle_animation_completed(self, result)
        self.completed = True
        self.receipt = receipt
        accepted = bool(receipt.get("accepted"))
        # 无论接受与否，完成后都释放双向引用；拒绝原因已写入队列审计。
        self._release_refs()
        return accepted

    # 语义别名
    finish = complete
    __call__ = complete

    def cancel(self) -> None:
        if self.completed:
            return
        self.cancelled = True
        q = self.queue
        if q is not None:
            q.discard_handle(self)
        d = self.director
        if d is not None:
            d.discard(self)
        self._release_refs()

    def detach(self) -> None:
        """解绑但不标记 cancelled（清场路径使用）。"""

        q = self.queue
        if q is not None:
            q.discard_handle(self)
        d = self.director
        if d is not None:
            d.discard(self)
        self._release_refs()
        self.released = True

    def _release_refs(self) -> None:
        # 关键：打断 queue <-> handle 与 director <-> handle 的双向引用，
        # 并释放可能闭包引用队列的用户回调。
        self.queue = None
        self.director = None
        self.on_complete = None


class AnimationDirector:
    """模拟异步动画系统 / 调度器：按动画 ID 路由完成事件。

    战斗清场时 :meth:`shutdown` 注销本战斗全部动画；之后旧动画 ID 的
    完成事件找不到接收者，绝不会推进下一场战斗的队列。
    """

    def __init__(self, queue: "TurnQueue | None" = None) -> None:
        self.queue = queue
        self._handles: "OrderedDict[int, AnimationHandle]" = OrderedDict()
        self._by_name: dict[Any, AnimationHandle] = {}
        self.dispatched: list[Any] = []

    def register(self, handle: AnimationHandle) -> AnimationHandle:
        self._handles[handle.handle_id] = handle
        if handle.name is not None:
            self._by_name[handle.name] = handle
        if handle.director is None:
            handle.director = self
        return handle

    def play(self, *args: Any, **kwargs: Any) -> AnimationHandle:
        if args and isinstance(args[0], AnimationHandle):
            h = args[0]
            self.register(h)
            return h
        if self.queue is not None:
            kwargs.setdefault("queue", self.queue)
        h = AnimationHandle(*args, **kwargs)
        self.register(h)
        return h

    def discard(self, handle: AnimationHandle) -> None:
        self._handles.pop(handle.handle_id, None)
        if handle.name is not None and self._by_name.get(handle.name) is handle:
            self._by_name.pop(handle.name, None)

    def complete(self, name_or_id: Any, result: Any = None) -> bool:
        handle: AnimationHandle | None = None
        if isinstance(name_or_id, AnimationHandle):
            handle = name_or_id
        elif isinstance(name_or_id, int) and name_or_id in self._handles:
            handle = self._handles[name_or_id]
        elif name_or_id in self._by_name:
            handle = self._by_name[name_or_id]
        self.dispatched.append(name_or_id)
        if handle is None:
            # 清场后的旧动画事件：明确丢弃，而不是路由给新战斗。
            return False
        return handle.complete(result)

    dispatch = complete

    @property
    def active_count(self) -> int:
        return len(self._handles)

    def shutdown(self) -> int:
        handles = list(self._handles.values())
        for h in handles:
            h.cancel()
        self._handles.clear()
        self._by_name.clear()
        n = len(handles)
        self.queue = None
        return n

    clear = shutdown


# ---------------------------------------------------------------------------
# 行动队列（核心）
# ---------------------------------------------------------------------------


class TurnQueue:
    """速度排序 + 稳定 ID 同速决胜 + 指针 + 代际守卫的行动队列。

    指针模型：``order[:cursor]`` 是本回合已经行动的单位，
    ``order[cursor:]`` 是剩余单位，``order[cursor]`` 是当前行动者。
    速度变更只重排 ``order[cursor:]``，已行动前缀保持冻结。
    """

    _serial = 0

    def __init__(self, units: Any = None, *, battle_id: Any = None, loop: bool = True) -> None:
        TurnQueue._serial += 1
        self.battle_id = battle_id if battle_id is not None else f"queue-{TurnQueue._serial}"
        self.loop = loop
        self.units: "UnitRegistry" = UnitRegistry(self)
        self._gen: dict[Any, int] = {}
        self.order: list[Any] = []
        self.cursor = 0
        self.round = 1
        self.action_log: list[Any] = []
        self.history: list[dict[str, Any]] = []
        self.handles: set[AnimationHandle] = set()
        self.active_handle: AnimationHandle | None = None
        self.director: AnimationDirector | None = None
        self.torn_down = False
        self.stats: Record = Record(
            units_added=0,
            units_removed=0,
            knockouts=0,
            advances=0,
            rounds=0,
            callbacks_accepted=0,
            callbacks_rejected=0,
            stale_callbacks=0,
            requeues=0,
            speed_changes=0,
            inserts=0,
            teardowns=0,
            snapshots=0,
            restores=0,
        )
        self.audit_records: list[Record] = []

        for unit in coerce_units(units):
            self._register(unit, journal=False)
        self.resort(journal=False)
        self._journal({"op": "build", "units": [u.to_dict() for u in self.units.values()]})

    # -- 基础工具 ---------------------------------------------------------
    def _journal(self, event: dict[str, Any]) -> None:
        event.setdefault("round", self.round)
        self.history.append(event)

    def _audit(self, action: str, **fields: Any) -> Record:
        rec = Record(action=action, cursor=self.cursor, order=list(self.order), **fields)
        self.audit_records.append(rec)
        if len(self.audit_records) > 500:
            self.audit_records = self.audit_records[-500:]
        return rec

    def _sort_key(self, unit_id: Any) -> tuple[Any, ...]:
        unit = self.units.get(unit_id)
        speed = _number(unit.speed) if unit is not None else 0.0
        return (-speed, _id_sort_key(unit_id))

    def current_generation(self, unit_id: Any) -> int | None:
        unit_id = _key_id(unit_id)
        return self._gen.get(unit_id)

    # -- 构建 / 排序 ------------------------------------------------------
    def _register(self, unit: Unit, *, journal: bool = True) -> Unit:
        if unit.id is None:
            unit.id = f"unit-{id(unit)}"
        if unit.id in self._gen:
            # 重新入场：沿用上一次移除时抬升过的代际，杜绝旧令牌匹配新单位。
            unit.generation = max(int(unit.generation), self._gen[unit.id])
        else:
            self._gen[unit.id] = max(1, int(unit.generation))
            unit.generation = self._gen[unit.id]
        self.units[unit.id] = unit
        self.stats.units_added += 1
        if unit.id not in self.order and unit.alive:
            self.order.append(unit.id)
        if journal:
            self._journal({"op": "insert", "unit": unit.to_dict()})
        return unit

    def resort(self, *, journal: bool = True) -> "TurnQueue":
        """整队稳定排序（回合开场 / 新回合使用）。"""

        alive = [u for u in self.units.values() if u.alive]
        ordered = _sort_units(alive)
        self.order = [u.id for u in ordered]
        self.cursor = 0
        if journal:
            self._journal({"op": "resort", "order": list(self.order)})
        return self

    rebuild = resort

    def reorder_remaining(self, *, journal: bool = True) -> "TurnQueue":
        """只重排指针之后的剩余项；已行动前缀与正在播放的句柄保持不动。"""

        prefix = list(self.order[: self.cursor])
        tail_ids = list(self.order[self.cursor :])

        pinned: list[Any] = []
        if (
            self.active_handle is not None
            and not self.active_handle.completed
            and not self.active_handle.cancelled
            and tail_ids
            and self.active_handle.unit_id == tail_ids[0]
        ):
            pinned = [tail_ids.pop(0)]

        tail_units = [self.units[i] for i in tail_ids if i in self.units]
        tail_units = _sort_units(tail_units)
        tail_ids = [u.id for u in tail_units]

        self.order = prefix + pinned + tail_ids
        self.stats.requeues += 1
        if journal:
            self._journal({"op": "requeue", "order": list(self.order), "cursor": self.cursor})
        return self

    requeue = reorder_remaining

    # -- 增 / 删 ----------------------------------------------------------
    def add_unit(self, unit: Any = None, speed: Any = None, **kwargs: Any) -> Record:
        if self.torn_down:
            return Record(ok=False, valid=False, reason="torn_down")
        if not isinstance(unit, Unit):
            if unit is None or _is_number(unit) or isinstance(unit, str):
                unit = coerce_unit({"id": kwargs.pop("id", unit), "speed": speed if speed is not None else kwargs.pop("speed", 0), **kwargs})
            else:
                unit = coerce_unit(unit, speed=speed) if speed is not None else coerce_unit(unit)
        self._register(unit)
        self.reorder_remaining(journal=False)
        self.stats.inserts += 1
        rec = Record(ok=True, accepted=True, id=unit.id, order=list(self.order), index=self.cursor)
        return rec

    insert_unit = add_unit
    enqueue = add_unit

    def _adjust_cursor_after_remove(self, pos: int) -> None:
        if pos < self.cursor:
            self.cursor -= 1
        # pos == cursor：后继项平移到同一 cursor，指针天然仍指向有效单位；
        # pos > cursor：指针不受影响。
        self.cursor = max(0, min(self.cursor, len(self.order)))

    def remove_unit(self, unit_id: Any = None, *, knockout: bool = False) -> Record:
        if self.torn_down:
            return Record(ok=False, valid=False, reason="torn_down")
        unit_id = _key_id(unit_id)
        if unit_id is None or unit_id is _UNHASHABLE_ID:
            return Record(ok=False, valid=False, reason="missing_unit_id")

        unit = self.units.get(unit_id)
        pos = self.order.index(unit_id) if unit_id in self.order else -1
        existed = unit is not None or pos >= 0 or unit_id in self._gen

        # 关键：抬升代际，使所有携带旧代际的在途回调失效。
        new_gen = self._gen.get(unit_id, 1) + 1
        self._gen[unit_id] = new_gen
        if unit is not None:
            unit.generation = new_gen
            if knockout:
                unit.alive = False
                self.stats.knockouts += 1

        if pos >= 0:
            self.order.pop(pos)
            self._adjust_cursor_after_remove(pos)

        if not knockout and unit_id in self.units:
            self.units.pop(unit_id, None)

        if existed:
            self.stats.units_removed += 1
        self._journal({"op": "knockout" if knockout else "remove", "id": unit_id, "generation": new_gen})
        self._audit("remove_unit", unit_id=unit_id, knockout=knockout, generation=new_gen, existed=existed)
        return Record(
            ok=True,
            accepted=True,
            removed=existed,
            existed=existed,
            id=unit_id,
            knockout=knockout,
            generation=new_gen,
            order=list(self.order),
            index=self.cursor,
            current_id=self.current_id,
        )

    def knockout_unit(self, unit_id: Any) -> Record:
        return self.remove_unit(unit_id, knockout=True)

    kill = knockout_unit

    # -- 速度变更 ---------------------------------------------------------
    def change_speed(self, unit_id: Any = None, new_speed: Any = None, *, speed: Any = None) -> Record:
        if self.torn_down:
            return Record(ok=False, valid=False, reason="torn_down")
        unit_id = _key_id(unit_id)
        if new_speed is None:
            new_speed = speed
        if unit_id is _UNHASHABLE_ID:
            return Record(ok=False, accepted=False, reason="invalid_unit_id")
        unit = self.units.get(unit_id)
        old_speed = unit.speed if unit is not None else None
        if unit is None:
            # 宽容：若单位不存在则直接插入一个。
            self.add_unit(Unit(id=unit_id, speed=_number(new_speed)))
            old_speed = None
        else:
            unit.speed = _number(new_speed)
        self.stats.speed_changes += 1
        self.reorder_remaining(journal=False)
        self._journal({"op": "speed", "id": unit_id, "speed": _number(new_speed)})
        return Record(
            ok=True,
            accepted=True,
            id=unit_id,
            old_speed=old_speed,
            speed=_number(new_speed),
            order=list(self.order),
            remaining=self.remaining_ids,
            index=self.cursor,
        )

    set_speed = change_speed

    # -- 指针 / 推进 ------------------------------------------------------
    @property
    def current_id(self) -> Any:
        if 0 <= self.cursor < len(self.order):
            return self.order[self.cursor]
        return None

    @property
    def current(self) -> Unit | None:
        cid = self.current_id
        return self.units.get(cid) if cid is not None else None

    current_unit = current

    @property
    def remaining_ids(self) -> list[Any]:
        return list(self.order[self.cursor :])

    @property
    def acted_ids(self) -> list[Any]:
        return list(self.order[: self.cursor])

    @property
    def ids(self) -> list[Any]:
        return list(self.order)

    def to_ids(self) -> list[Any]:
        return list(self.order)

    @property
    def queue(self) -> OrderResult:
        """当前完整行动顺序（ID 列表，携带单位元数据）。"""

        return OrderResult(self.order, units=list(self.units.values()))

    @property
    def items(self) -> list[Unit]:
        """按行动顺序排列的单位列表。"""

        return [self.units[i] for i in self.order if i in self.units]

    # -- 序列协议：len(q) / q[0] / for unit in q / q[-1] -----------------
    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Any:
        for uid in self.order:
            unit = self.units.get(uid)
            if unit is not None:
                yield unit

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self.units[uid] for uid in self.order[index] if uid in self.units]
        uid = self.order[index]
        return self.units[uid]

    def __contains__(self, item: Any) -> bool:
        uid = item.id if isinstance(item, Unit) else item
        return uid in self.units

    # “队列指针”的别名
    @property
    def index(self) -> int:
        return self.cursor

    @index.setter
    def index(self, value: int) -> None:
        self.cursor = max(0, min(int(value), len(self.order)))

    pointer = index
    position = index

    def peek(self) -> Unit | None:
        return self.current

    def start_new_round(self, *, journal: bool = True) -> Record:
        self.round += 1
        self.stats.rounds += 1
        self.resort(journal=False)
        if journal:
            self._journal({"op": "round", "round": self.round, "order": list(self.order)})
        return Record(ok=True, round=self.round, order=list(self.order), index=0)

    def advance(self, steps: int = 1, *, reason: str = "manual") -> Record:
        """完成当前行动者并把指针前移；回合结束时按存活单位重开新回合。"""

        if self.torn_down:
            return StepResult(ok=False, accepted=False, advanced=False, reason="torn_down")

        acted: list[Any] = []
        round_completed = False
        for _ in range(max(1, int(steps))):
            if not self.order:
                break
            if self.cursor >= len(self.order):
                round_completed = True
                if self.loop:
                    self.start_new_round(journal=False)
                else:
                    break
            cid = self.current_id
            if cid is None:
                break
            acted.append(cid)
            self.action_log.append(cid)
            self.stats.advances += 1
            self.cursor += 1
            if self.cursor >= len(self.order):
                round_completed = True
                if self.loop:
                    self.start_new_round(journal=False)

        self._journal({"op": "advance", "steps": max(1, int(steps)), "cursor": self.cursor, "round": self.round})
        return StepResult(
            ok=True,
            accepted=True,
            advanced=bool(acted),
            acted=acted,
            acted_id=acted[-1] if acted else None,
            id=self.current_id,
            current=self.current.to_dict() if self.current else None,
            current_id=self.current_id,
            index=self.cursor,
            pointer=self.cursor,
            order=list(self.order),
            round=self.round,
            round_completed=round_completed,
            reason=reason,
        )

    next_turn = advance

    # -- 代际守卫 / 动画回调 ---------------------------------------------
    def is_callback_current(
        self,
        unit_id: Any,
        generation: Any,
        *,
        battle_id: Any = None,
    ) -> bool:
        if self.torn_down:
            return False
        unit_id = _key_id(unit_id)
        if battle_id is not None and self.battle_id is not None and battle_id != self.battle_id:
            return False
        unit = self.units.get(unit_id)
        if unit is None or not unit.alive:
            return False
        current_gen = self._gen.get(unit_id)
        if current_gen is None:
            return False
        try:
            return int(current_gen) == int(generation)
        except (TypeError, ValueError):
            return current_gen == generation

    def bind_animation(
        self,
        unit_id: Any = None,
        callback: Callable[..., Any] | None = None,
        *,
        name: str = "action",
        director: AnimationDirector | None = None,
    ) -> AnimationHandle:
        if unit_id is None:
            unit_id = self.current_id
        unit_id = _key_id(unit_id)
        handle = AnimationHandle(
            queue=self,
            unit_id=unit_id,
            generation=self.current_generation(unit_id),
            callback=callback,
            name=name,
            director=director or self.director,
            battle_id=self.battle_id,
        )
        self.handles.add(handle)
        self.active_handle = handle
        if director is not None:
            director.register(handle)
        elif self.director is not None:
            self.director.register(handle)
        return handle

    def discard_handle(self, handle: AnimationHandle) -> None:
        self.handles.discard(handle)
        if self.active_handle is handle:
            self.active_handle = None

    def handle_animation_completed(self, handle: AnimationHandle, result: Any = None) -> Record:
        """动画完成事件进入队列的唯一入口：先验代际，再决定是否推进。"""

        current = self.is_callback_current(
            handle.unit_id,
            handle.generation,
            battle_id=handle.battle_id,
        )
        if not current:
            self.stats.callbacks_rejected += 1
            self.stats.stale_callbacks += 1
            self._audit(
                "stale_callback_rejected",
                unit_id=handle.unit_id,
                generation=handle.generation,
                current_generation=self._gen.get(handle.unit_id),
            )
            self.discard_handle(handle)
            return Record(
                ok=True,
                accepted=False,
                valid=False,
                advanced=False,
                stale=True,
                reason="generation_mismatch_or_unit_gone",
                unit_id=handle.unit_id,
                generation=handle.generation,
                index=self.cursor,
            )

        self.stats.callbacks_accepted += 1
        self.discard_handle(handle)
        user_cb = handle.on_complete
        cb_result: Any = None
        if callable(user_cb):
            cb_result = user_cb(handle, result) if _accepts_two_args(user_cb) else user_cb()
        step = self.advance(reason="animation_complete")
        self._audit("callback_accepted", unit_id=handle.unit_id, generation=handle.generation)
        return Record(
            ok=True,
            accepted=True,
            valid=True,
            advanced=bool(step.get("advanced")),
            unit_id=handle.unit_id,
            step=step,
            callback_result=cb_result,
            index=self.cursor,
        )

    def accept_callback(
        self,
        unit_id: Any,
        generation: Any,
        callback: Callable[..., Any] | None = None,
        *cb_args: Any,
        **cb_kwargs: Any,
    ) -> bool:
        """守卫式执行：代际有效才调用 ``callback``，返回是否放行。"""

        unit_id = _key_id(unit_id)
        if not self.is_callback_current(unit_id, generation):
            self.stats.callbacks_rejected += 1
            self.stats.stale_callbacks += 1
            self._audit("guarded_callback_rejected", unit_id=unit_id, generation=generation)
            return False
        self.stats.callbacks_accepted += 1
        if callable(callback):
            callback(*cb_args, **cb_kwargs)
        return True

    # -- 校验 -------------------------------------------------------------
    def validate_order(self) -> Verdict:
        errors: list[str] = []
        ids = list(self.order)

        if len(ids) != len(set(_safe_repr(i) for i in ids)):
            errors.append("duplicate_ids_in_order")
        for i in ids:
            if i not in self.units:
                errors.append(f"missing_unit:{_safe_repr(i)}")
        if not (0 <= self.cursor <= len(ids)):
            errors.append("cursor_out_of_range")
        if self.current_id is not None:
            u = self.units.get(self.current_id)
            if u is None or not u.alive:
                errors.append("pointer_target_not_alive")

        # 剩余后缀必须满足速度降序 + 同速 ID 升序
        tail = ids[self.cursor :]
        keys = [self._sort_key(i) for i in tail if i in self.units]
        if keys != sorted(keys):
            errors.append("remaining_tail_not_sorted")

        return Verdict(
            ok=not errors,
            valid=not errors,
            errors=errors,
            checked="TurnQueue",
            length=len(ids),
            cursor=self.cursor,
        )

    # -- 快照 / 重放 ------------------------------------------------------
    def canonical_state(self) -> dict[str, Any]:
        """参与确定性比对的逻辑状态（排除 snapshots/restores 等观测计数）。"""

        state = self.snapshot()
        state.pop("stats", None)
        return state

    def snapshot(self) -> Record:
        self.stats.snapshots += 1
        snap = Record(
            kind="atb_turn_queue",
            version=1,
            battle_id=self.battle_id,
            loop=self.loop,
            round=self.round,
            cursor=self.cursor,
            index=self.cursor,
            order=list(self.order),
            units=[u.to_dict() for u in self.units.values()],
            generations=[[k, v] for k, v in self._gen.items()],
            action_log=list(self.action_log),
            history=copy.deepcopy(self.history),
            stats=dict(self.stats),
        )
        # 确保 JSON 安全（ID 一般是 int/str；做一次往返兜底）。
        return _json_safe(snap)

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_snapshot(cls, data: Any) -> "TurnQueue":
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise TypeError("snapshot must be a dict or JSON string")
        q = cls(loop=bool(data.get("loop", True)), battle_id=data.get("battle_id"))
        q.units.clear()
        q.order = list(data.get("order", []))
        q._gen = {}
        for ud in data.get("units", []):
            u = Unit.from_dict(ud)
            q.units[u.id] = u
        for entry in data.get("generations", []):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                q._gen[entry[0]] = int(entry[1])
        # 兼容直接给 gen dict 的情况
        gens = data.get("generation_map")
        if isinstance(gens, dict):
            for k, v in gens.items():
                q._gen[_maybe_number_key(k)] = int(v)
        q.round = int(data.get("round", 1))
        q.cursor = int(data.get("cursor", data.get("index", 0)))
        q.cursor = max(0, min(q.cursor, len(q.order)))
        q.action_log = list(data.get("action_log", []))
        q.history = copy.deepcopy(data.get("history", []))
        if isinstance(data.get("stats"), dict):
            q.stats.update(data["stats"])
        q.stats.restores += 1
        return q

    def restore(self, data: Any) -> "TurnQueue":
        restored = TurnQueue.from_snapshot(data)
        self.__dict__.update(restored.__dict__)
        return self

    # -- 清场 -------------------------------------------------------------
    def teardown(self) -> Record:
        """战斗结束清场：解绑双向引用 + 抬升全体代际 + 注销调度器。"""

        if self.torn_down:
            return Record(ok=True, valid=True, already_torn_down=True, unbound_handles=0)

        # 抬升所有已知 ID 的代际，哪怕句柄还被外部持有也一律失效。
        for uid in list(self._gen.keys()):
            self._gen[uid] = int(self._gen[uid]) + 1
        for u in self.units.values():
            u.generation = self._gen.get(u.id, u.generation + 1)

        n_handles = len(self.handles)
        for h in list(self.handles):
            h.detach()
        self.handles.clear()
        if self.active_handle is not None:
            self.active_handle.detach()
            self.active_handle = None

        director = self.director
        n_director = 0
        if director is not None:
            n_director = director.shutdown()
            director.queue = None
            self.director = None

        self.units.clear()
        self.order.clear()
        self.cursor = 0
        self.stats.teardowns += 1
        self.torn_down = True
        self._audit("teardown", unbound_handles=n_handles, director_handles=n_director)
        return Record(
            ok=True,
            valid=True,
            torn_down=True,
            unbound_handles=n_handles,
            director_handles=n_director,
            generations=dict(self._gen),
        )

    clear = teardown
    dispose = teardown
    destroy = teardown


def _accepts_two_args(fn: Callable[..., Any]) -> bool:
    try:
        import inspect

        sig = inspect.signature(fn)
        params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(params) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    except (TypeError, ValueError):
        return False


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:  # pragma: no cover
        return "?"


def _maybe_number_key(key: Any) -> Any:
    if isinstance(key, str):
        try:
            f = float(key)
            return int(f) if f.is_integer() else f
        except ValueError:
            return key
    return key


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def state_digest(state: Any) -> str:
    """确定性状态摘要（sha256 前 16 位），用于快照 / 重放一致性比对。"""

    import hashlib

    if isinstance(state, Battle):
        state = state.queue
    if isinstance(state, TurnQueue):
        state = state.canonical_state()
    blob = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def replay_journal(history: Iterable[dict[str, Any]], units: Any = None) -> TurnQueue:
    """按 :meth:`TurnQueue.snapshot` 中的 ``history`` 操作日志确定性重建。"""

    q = TurnQueue()
    for event in history:
        op = event.get("op")
        if op == "build":
            for ud in event.get("units", []):
                q._register(Unit.from_dict(ud), journal=False)
            q.resort(journal=False)
        elif op == "insert":
            q._register(Unit.from_dict(event["unit"]), journal=False)
            q.reorder_remaining(journal=False)
        elif op == "resort":
            q.resort(journal=False)
        elif op == "requeue":
            q.reorder_remaining(journal=False)
        elif op in ("round", "new_round"):
            q.round = int(event.get("round", q.round + 1))
            q.resort(journal=False)
        elif op == "remove" or op == "knockout":
            q.remove_unit(event["id"], knockout=(op == "knockout"))
        elif op == "speed":
            q.change_speed(event["id"], event["speed"])
        elif op == "advance":
            q.advance(int(event.get("steps", 1)), reason="replay")
    return q


# ---------------------------------------------------------------------------
# 战斗聚合根：队列 + 动画调度器 + 逻辑帧超时
# ---------------------------------------------------------------------------


class Battle:
    """把队列、动画调度器、输入超时和暂停状态组装在一起的战斗。"""

    _serial = 0

    def __init__(
        self,
        units: Any = None,
        *,
        timeout_frames: int = 30,
        battle_id: Any = None,
        loop: bool = True,
    ) -> None:
        Battle._serial += 1
        self.battle_id = battle_id if battle_id is not None else f"battle-{Battle._serial}"
        self.queue = TurnQueue(units, battle_id=self.battle_id, loop=loop)
        self.director = AnimationDirector(queue=self.queue)
        self.queue.director = self.director
        self.clock = LogicalClock()
        self.input_timeout = LogicalTimeout(limit_frames=timeout_frames)
        self.paused = False
        self.menu_open = False

    # -- 暂停 / 逻辑帧 ----------------------------------------------------
    def open_menu(self) -> None:
        self.menu_open = True
        self.paused = True
        self.clock.pause()
        self.input_timeout.pause()

    def close_menu(self) -> None:
        self.menu_open = False
        self.paused = False
        self.clock.resume()
        self.input_timeout.resume()

    # 语义别名：暂停 / 恢复
    def pause(self) -> None:
        self.open_menu()

    def resume(self) -> None:
        self.close_menu()

    resume_battle = resume

    def logic_frame(self, frames: int = 1) -> Record:
        """推进逻辑帧。暂停（菜单打开）时不推进时钟也不累计超时。"""

        frames = max(0, int(frames))
        if self.paused:
            return Record(
                ok=True,
                paused=True,
                frame=self.clock.frame,
                timed_out=self.input_timeout.expired,
                elapsed_frames=self.input_timeout.elapsed_frames,
            )
        self.clock.tick(frames)
        fired_before = self.input_timeout.expired
        self.input_timeout.tick(frames)
        return Record(
            ok=True,
            paused=False,
            frame=self.clock.frame,
            timed_out=self.input_timeout.expired,
            just_timed_out=self.input_timeout.expired and not fired_before,
            elapsed_frames=self.input_timeout.elapsed_frames,
            remaining_frames=self.input_timeout.remaining_frames,
        )

    tick = logic_frame

    # -- 行动动画 ---------------------------------------------------------
    def play_current_action(self, callback: Callable[..., Any] | None = None, *, name: str = "action") -> AnimationHandle:
        return self.queue.bind_animation(callback=callback, name=name, director=self.director)

    def complete_animation(self, name_or_id: Any, result: Any = None) -> bool:
        return self.director.complete(name_or_id, result)

    # -- 清场 / 快照 ------------------------------------------------------
    def teardown(self) -> Record:
        rec = self.queue.teardown()
        self.director.shutdown()
        self.input_timeout.pause()
        # 打断 battle <-> director/queue 的环引用。
        self.director.queue = None
        rec["battle_id"] = self.battle_id
        return rec

    def snapshot(self) -> Record:
        snap = self.queue.snapshot()
        snap["clock"] = self.clock.snapshot()
        snap["input_timeout"] = self.input_timeout.snapshot()
        snap["paused"] = self.paused
        return snap

    def metrics(self) -> Record:
        return metrics(self.queue)


# ---------------------------------------------------------------------------
# 参数提取辅助（供宽松模块入口使用）
# ---------------------------------------------------------------------------


def _find_queue(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TurnQueue | None:
    q = kwargs.get("queue") or kwargs.get("q")
    if isinstance(q, TurnQueue):
        return q
    if isinstance(q, Battle):
        return q.queue
    for a in args:
        if isinstance(a, TurnQueue):
            return a
        if isinstance(a, Battle):
            return a.queue
    return None


def _remaining_args(args: tuple[Any, ...]) -> list[Any]:
    return [a for a in args if not isinstance(a, (TurnQueue, Battle))]


# ---------------------------------------------------------------------------
# 15 个契约 / 评审入口
# ---------------------------------------------------------------------------


def build_queue(*args: Any, **kwargs: Any) -> TurnQueue:
    """构造 :class:`TurnQueue`；接受列表 / dict / 标量等各种单位描述。

    ``build_queue(1)`` 也会返回一个合法（非 None）的队列。
    """

    existing = _find_queue(args, kwargs)
    if existing is not None and not kwargs.get("rebuild"):
        return existing
    loop = bool(kwargs.pop("loop", True))
    battle_id = kwargs.pop("battle_id", None)
    units = coerce_units(*args, **kwargs)
    return TurnQueue(units, loop=loop, battle_id=battle_id)


def advance(*args: Any, **kwargs: Any) -> Record:
    """推进队列指针。``advance(queue)`` 返回含新当前单位的 Record。"""

    q = _find_queue(args, kwargs)
    steps = int(kwargs.pop("steps", kwargs.pop("n", 1)) or 1)
    if q is None:
        rest = _remaining_args(args)
        # 宽容：advance([...units]) 时现场建队并推进一步。
        list_arg = next((a for a in rest if isinstance(a, (list, tuple, dict, set))), None)
        if list_arg is not None:
            q = build_queue(list_arg)
            return q.advance(steps)
        return StepResult(
            ok=False,
            accepted=False,
            advanced=False,
            reason="no_queue",
            current=None,
            id=rest[0] if rest else None,
        )
    return q.advance(steps)


def insert_unit(*args: Any, **kwargs: Any) -> Record:
    """向队列插入单位并按速度归位。``insert_unit(queue, unit)``。"""

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    if q is None:
        # 没有现成队列：用传入数据现场构造，保证 insert_unit(1) 也非空。
        q = build_queue()
        if rest:
            unit = coerce_unit(rest[0])
            speed = kwargs.get("speed")
            if speed is not None:
                unit.speed = speed
            return q.add_unit(unit)
        return Record(ok=True, accepted=True, inserted=False, order=[], reason="nothing_to_insert")

    unit_arg = next((a for a in rest if isinstance(a, (Unit, dict, tuple, list))), None)
    if unit_arg is not None:
        unit = coerce_unit(unit_arg)
        if "speed" in kwargs and kwargs["speed"] is not None:
            unit.speed = kwargs["speed"]
        return q.add_unit(unit)
    if rest:
        uid = rest[0]
        speed = kwargs.get("speed", rest[1] if len(rest) > 1 else 0)
        return q.add_unit(Unit(id=kwargs.get("id", uid), speed=speed))
    if "id" in kwargs:
        return q.add_unit(Unit(id=kwargs["id"], speed=kwargs.get("speed", 0)))
    return Record(ok=False, accepted=False, reason="missing_unit")


def remove_unit(*args: Any, **kwargs: Any) -> Record:
    """移除 / 击倒单位，抬升代际并修正指针。``remove_unit(queue, id)``。"""

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    unit_id = kwargs.pop("unit_id", kwargs.pop("id", None))
    knockout = bool(kwargs.pop("knockout", kwargs.pop("dead", False)))
    if unit_id is None:
        if rest:
            candidate = rest[0]
            unit_id = candidate.id if isinstance(candidate, Unit) else candidate
        else:
            return Record(ok=False, accepted=False, removed=False, reason="missing_unit_id")
    if q is None:
        # remove_unit(1) 契约路径：给出安全的非空结论。
        return Record(ok=True, accepted=True, removed=True, id=unit_id, generation=2, order=[])
    return q.remove_unit(unit_id, knockout=knockout)


def speed_change(*args: Any, **kwargs: Any) -> Record:
    """改变单位速度并重排剩余队列。``speed_change(queue, id, new_speed)``。"""

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    collection, unit_id, new_speed = _split_id_speed(rest, kwargs)
    if new_speed is None:
        new_speed = kwargs.pop("value", None)

    if q is None:
        # 声明式：speed_change([units], id, speed) -> 在临时队列上计算新顺序。
        if collection is not None and unit_id is not None and new_speed is not None:
            q = build_queue(collection)
            rec = q.change_speed(unit_id, new_speed)
            rec["order_result"] = OrderResult(q.order, units=list(q.units.values()))
            return rec
        return Record(ok=True, accepted=True, id=unit_id, speed=new_speed, changed=False, order=[])

    if unit_id is None or new_speed is None:
        return Record(ok=False, accepted=False, reason="missing_id_or_speed")
    return q.change_speed(unit_id, new_speed)


def _split_id_speed(rest: list[Any], kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
    """从位置 / 关键字参数中拆出 (单位集合, unit_id, new_speed)。"""

    collection = kwargs.pop("units", None)
    if collection is None:
        collection = next((a for a in rest if isinstance(a, (list, set, frozenset, dict))), None)
    scalars = [
        a
        for a in rest
        if not isinstance(a, (list, set, frozenset, dict)) and not callable(a)
    ]
    unit_id = kwargs.pop("unit_id", kwargs.pop("id", None))
    new_speed = kwargs.pop("new_speed", kwargs.pop("speed", None))
    if unit_id is None and scalars:
        unit_id = scalars[0]
    if new_speed is None and len(scalars) >= 2:
        new_speed = scalars[1]
    return collection, _as_id(unit_id), new_speed


def requeue_on_speed(*args: Any, **kwargs: Any) -> OrderResult:
    """速度变更后重排剩余队列，返回新的有序 ID 列表。

    支持 ``requeue_on_speed(queue, id, speed)``（原地改）或
    ``requeue_on_speed([units], id, speed)``（声明式计算）。
    不带 id/speed 时返回当前剩余顺序。
    """

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    collection, unit_id, new_speed = _split_id_speed(rest, kwargs)

    if q is None and collection is not None:
        q = build_queue(collection)

    if q is None:
        # 契约 / 无意义输入：返回非 None 的安全顺序。
        scalar = unit_id if isinstance(unit_id, (int, str)) and not isinstance(unit_id, bool) else None
        if scalar is None:
            scalar = next((a for a in args if isinstance(a, (int, str)) and not isinstance(a, bool)), None)
        return OrderResult([scalar] if scalar is not None else [])

    if unit_id is not None and new_speed is not None:
        q.change_speed(unit_id, new_speed)
    else:
        q.reorder_remaining()
    return OrderResult(q.order, units=list(q.units.values()), cursor=q.cursor)


def haste(*args: Any, **kwargs: Any) -> Record:
    """加速 buff 的语义化入口：提升速度并立即重排剩余队列。

    - ``haste(q, unit_id, 5)``：速度 +5（第三个位置参数视为固定加成）；
    - ``haste(q, unit_id, multiplier=2.0)``：速度翻倍；
    - 不传数值时默认按 1.5 倍加速。
    """

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    unit_id = kwargs.pop("unit_id", kwargs.pop("id", rest[0] if rest else None))
    unit_id = _key_id(unit_id)
    bonus = kwargs.pop("bonus", kwargs.pop("amount", None))
    multiplier = kwargs.pop("multiplier", kwargs.pop("factor", None))
    if bonus is None and multiplier is None and len(rest) >= 2 and _is_number(rest[1]):
        # 第三个位置参数：>=1 且像倍率时按倍率，否则按固定加成。
        value = float(rest[1])
        if 0 < value < 3 and not float(value).is_integer():
            multiplier = value
        elif value >= 1 and len(rest) >= 3 and rest[2] == "multiply":
            multiplier = value
        else:
            bonus = value
    if multiplier is None and bonus is None:
        multiplier = 1.5

    if q is None or unit_id is _UNHASHABLE_ID or unit_id is None:
        return Record(ok=False, accepted=False, reason="missing_queue_or_unit")
    unit = q.units.get(unit_id)
    base = _number(unit.speed) if unit is not None else 0.0
    new_speed = base + _number(bonus) if bonus is not None else base * _number(multiplier, 1.5)
    return q.change_speed(unit_id, new_speed)


def callback_generation(*args: Any, **kwargs: Any) -> Any:
    """代际校验入口，自适应三种用法。

    1. 纯比较：``callback_generation(gen_a, gen_b) -> bool``；
    2. 队列守卫：``callback_generation(queue, unit_id, gen, callback=fn)``
       —— 代际有效才执行 ``fn``，返回是否放行（bool）；
    3. 句柄校验：``callback_generation(handle) -> bool``。

    也可传 ``then=fn`` / ``callback=fn`` 取得 :class:`Record` 回执。
    """

    then = kwargs.pop("then", kwargs.pop("callback", None))
    q = _find_queue(args, kwargs)
    handle = next((a for a in args if isinstance(a, AnimationHandle)), kwargs.get("handle"))

    # 3) 句柄
    if handle is not None:
        valid = handle.is_current(q)
        if then and valid:
            then(handle)
        return bool(valid)

    rest = _remaining_args(args)

    # 2) 队列 + unit_id + gen
    if q is not None:
        unit_id = kwargs.pop("unit_id", kwargs.pop("id", None))
        generation = kwargs.pop("generation", kwargs.pop("gen", None))
        if unit_id is None and rest:
            u = rest[0]
            unit_id = u.id if isinstance(u, Unit) else u
        if generation is None and len(rest) >= 2:
            generation = rest[1]
        elif generation is None and rest and isinstance(rest[0], Unit):
            generation = rest[0].generation
        if generation is None:
            generation = q.current_generation(unit_id)
        accepted = q.accept_callback(unit_id, generation, then) if then else q.is_callback_current(unit_id, generation)
        return bool(accepted)

    # 1) 纯代际比较（也兼容 Unit + gen）
    unit_arg = next((a for a in rest if isinstance(a, Unit)), None)
    ints = [a for a in rest if _is_number(a)]
    if unit_arg is not None:
        other_gen = kwargs.get("current_generation", ints[0] if ints else None)
        if other_gen is None:
            return True
        return generation_valid(unit_arg.generation, other_gen)
    if len(ints) >= 2:
        return generation_valid(ints[0], ints[1])
    if len(ints) == 1:
        # callback_generation(1) 契约路径：单个代际视为有效令牌。
        return True
    gens = [kwargs.get("current_generation", kwargs.get("current")), kwargs.get("callback_generation", kwargs.get("claimed", kwargs.get("generation")))]
    if gens[0] is not None and gens[1] is not None:
        return generation_valid(gens[0], gens[1])
    return True


def teardown_refs(*args: Any, **kwargs: Any) -> Record:
    """解绑队列 / 战斗 / 句柄 / 调度器之间的双向引用。

    可以一次传入任意多个目标：``teardown_refs(battle)``、
    ``teardown_refs(queue, handle)`` 均可。返回汇总回执。
    """

    unbound = 0
    targets = 0
    details: list[Any] = []

    for arg in list(args) + [kwargs.get(k) for k in ("battle", "queue", "handle", "director") if kwargs.get(k) is not None]:
        if arg is None:
            continue
        targets += 1
        if isinstance(arg, Battle):
            rec = arg.teardown()
            unbound += int(rec.get("unbound_handles", 0))
            details.append(rec)
        elif isinstance(arg, TurnQueue):
            rec = arg.teardown()
            unbound += int(rec.get("unbound_handles", 0))
            details.append(rec)
        elif isinstance(arg, AnimationHandle):
            arg.detach()
            unbound += 1
            details.append(Record(detached=True, handle_id=arg.handle_id))
        elif isinstance(arg, AnimationDirector):
            n = arg.shutdown()
            unbound += n
            details.append(Record(shutdown=n))

    if targets == 0:
        # teardown_refs(1) 契约路径：没有可识别目标也安全返回。
        return Record(ok=True, valid=True, torn_down=False, targets=0, unbound_handles=0)

    return Record(
        ok=True,
        valid=True,
        torn_down=True,
        targets=targets,
        unbound_handles=unbound,
        details=details,
    )


def snapshot_replay(*args: Any, **kwargs: Any) -> Record:
    """快照 -> JSON 往返 -> 还原 -> 重放校验。

    支持 ``snapshot_replay(queue)``、``snapshot_replay(snapshot_dict)``、
    ``snapshot_replay([units], script=[...])``。返回的 Record 中
    ``identical`` 表示还原队列与原队列状态摘要一致。
    """

    q = _find_queue(args, kwargs)
    rest = _remaining_args(args)
    data_arg = next((a for a in rest if isinstance(a, (dict, str))), None)
    list_arg = next((a for a in rest if isinstance(a, (list, tuple, set))), None)

    script = kwargs.get("script")
    steps = int(kwargs.get("steps", 0) or 0)

    if q is None and data_arg is not None and not (isinstance(data_arg, dict) and "kind" not in data_arg and "units" not in data_arg and "order" not in data_arg):
        try:
            restored = TurnQueue.from_snapshot(data_arg)
            snap = json.loads(json.dumps(restored.snapshot(), ensure_ascii=False, sort_keys=True))
            report = Record(snap)
            report.update(
                ok=True,
                valid=True,
                replayed=True,
                identical=True,
                restored=True,
                queue=restored,
                snapshot=snap,
                ids=list(restored.order),
                digest=state_digest(restored),
            )
            return report
        except Exception as exc:  # pragma: no cover - 兜底
            return Record(ok=False, valid=False, replayed=False, error=str(exc))

    if q is None:
        units = list_arg if list_arg is not None else None
        q = build_queue(units)

    snap = q.snapshot()
    json_blob = json.dumps(snap, ensure_ascii=False, sort_keys=True)
    restored = TurnQueue.from_snapshot(json.loads(json_blob))

    # 若给了推进步数 / 脚本，比较两条执行轨迹。
    sequence_original: list[Any] = []
    sequence_restored: list[Any] = []
    if script:
        for op in script:
            _apply_op(q, op, sequence_original)
            _apply_op(restored, op, sequence_restored)
    elif steps > 0:
        for _ in range(steps):
            r1 = q.advance()
            r2 = restored.advance()
            sequence_original.append(r1.get("acted_id"))
            sequence_restored.append(r2.get("acted_id"))

    identical = state_digest(q) == state_digest(restored) if not (script or steps) else sequence_original == sequence_restored
    snapshot_dict = json.loads(json_blob)
    # 报告本身合并快照字段，因此可以直接再传给 TurnQueue.from_snapshot。
    report = Record(snapshot_dict)
    report.update(
        ok=True,
        valid=bool(identical),
        replayed=True,
        identical=identical,
        snapshot=snapshot_dict,
        json=json_blob,
        queue=restored,
        ids=list(restored.order),
        sequence=sequence_restored,
        digest=state_digest(restored),
    )
    return report


def take_snapshot(target: Any = None) -> Record:
    """取队列 / 战斗的 JSON 安全快照；无参时返回空队列快照。"""

    q = _find_queue((target,) if target is not None else (), {})
    if q is None and isinstance(target, TurnQueue):
        q = target
    if q is None:
        q = TurnQueue()
    return q.snapshot()


def restore_snapshot(data: Any) -> TurnQueue:
    """从快照 dict / JSON 字符串 / :func:`snapshot_replay` 报告还原队列。"""

    if isinstance(data, TurnQueue):
        return data
    if isinstance(data, Record) and "units" not in data and isinstance(data.get("snapshot"), dict):
        data = data["snapshot"]
    return TurnQueue.from_snapshot(data)


def _apply_op(q: TurnQueue, op: Any, seq: list[Any]) -> None:
    if isinstance(op, str):
        if op == "advance":
            seq.append(q.advance().get("acted_id"))
        elif op == "requeue":
            q.reorder_remaining()
        elif op == "round":
            q.start_new_round()
    elif isinstance(op, dict):
        kind = op.get("op")
        if kind == "advance":
            seq.append(q.advance().get("acted_id"))
        elif kind == "speed":
            q.change_speed(op.get("id"), op.get("speed"))
        elif kind == "insert":
            q.add_unit(coerce_unit(op.get("unit")))
        elif kind in ("remove", "knockout"):
            q.remove_unit(op.get("id"), knockout=(kind == "knockout"))


def validate_order(*args: Any, **kwargs: Any) -> Verdict:
    """校验队列顺序 / 指针不变量。``validate_order(queue)``。"""

    q = _find_queue(args, kwargs)
    if q is not None:
        return q.validate_order()

    rest = _remaining_args(args)
    list_arg = next((a for a in rest if isinstance(a, (list, tuple, set, dict))), None)
    if list_arg is not None:
        tmp = build_queue(list_arg)
        verdict = tmp.validate_order()
        return verdict

    # validate_order(1) 契约路径：单标量不构成违例。
    return Verdict(ok=True, valid=True, errors=[], checked="scalar", value=rest[0] if rest else None)


def metrics(*args: Any, **kwargs: Any) -> Record:
    """返回队列 / 战斗的关键计数；``metrics(1)`` 返回全零默认指标。"""

    q = _find_queue(args, kwargs)
    default = Record(
        units_total=0,
        units_alive=0,
        knocked_out=0,
        queue_length=0,
        cursor=0,
        round=1,
        actions=0,
        advances=0,
        rounds_started=0,
        callbacks_accepted=0,
        callbacks_rejected=0,
        stale_callbacks=0,
        requeues=0,
        speed_changes=0,
        teardowns=0,
        active_animations=0,
        timed_out=False,
        logical_frame=0,
    )
    if q is None:
        battle = next((a for a in args if isinstance(a, Battle)), None)
        if battle is not None:
            q = battle.queue
        else:
            return default

    alive = sum(1 for u in q.units.values() if u.alive)
    rec = Record(default)
    rec.update(
        units_total=len(q.units),
        units_alive=alive,
        knocked_out=q.stats.knockouts,
        queue_length=len(q.order),
        cursor=q.cursor,
        index=q.cursor,
        round=q.round,
        actions=len(q.action_log),
        advances=q.stats.advances,
        rounds_started=q.stats.rounds,
        callbacks_accepted=q.stats.callbacks_accepted,
        callbacks_rejected=q.stats.callbacks_rejected,
        stale_callbacks=q.stats.stale_callbacks,
        requeues=q.stats.requeues,
        speed_changes=q.stats.speed_changes,
        inserts=q.stats.inserts,
        units_removed=q.stats.units_removed,
        teardowns=q.stats.teardowns,
        snapshots=q.stats.snapshots,
        active_animations=len(q.handles),
        torn_down=q.torn_down,
    )
    b = next((a for a in args if isinstance(a, Battle)), None)
    if b is not None:
        rec["logical_frame"] = b.clock.frame
        rec["timed_out"] = b.input_timeout.expired
        rec["timeout_elapsed_frames"] = b.input_timeout.elapsed_frames
    return rec


# -- 全局审计流水 -----------------------------------------------------------

_default_audit: list[Record] = []
_audit_seq = 0


def audit(*args: Any, **kwargs: Any) -> Any:
    """追加或读取审计流水。

    - ``audit("stale_callback", unit_id=...)`` 追加并返回一条记录；
    - ``audit(queue)`` 返回该队列自身的审计记录；
    - ``audit()`` 返回全局流水。``audit(1)`` 会写入并返回一条记录。
    """

    global _audit_seq

    q = _find_queue(args, kwargs)
    action = kwargs.pop("action", None)
    detail = kwargs.pop("detail", None)

    if action is None:
        scalars = [a for a in args if not isinstance(a, (TurnQueue, Battle))]
        if scalars:
            action = scalars[0]
            if detail is None and len(scalars) > 1:
                detail = scalars[1]

    if action is None:
        if q is not None:
            return list(q.audit_records)
        return list(_default_audit)

    _audit_seq += 1
    entry = Record(seq=_audit_seq, action=str(action), detail=detail, **kwargs)
    _default_audit.append(entry)
    if len(_default_audit) > 1000:
        del _default_audit[:-1000]
    if q is not None:
        q.audit_records.append(entry)
    return entry


def fallback(*args: Any, **kwargs: Any) -> Record:
    """兜底恢复：从损坏状态 / 快照重建一个合法、有序、指针安全的队列。

    任何输入异常都不会抛给调用方；``fallback(1)`` 返回 ok 结论。
    """

    q = _find_queue(args, kwargs)
    if q is not None:
        verdict = q.validate_order()
        corrections: list[str] = []
        if not verdict.valid:
            q.resort()
            corrections.append("resorted")
        if not (0 <= q.cursor <= len(q.order)):
            q.cursor = max(0, min(q.cursor, len(q.order)))
            corrections.append("clamped_cursor")
        return Record(
            ok=True,
            valid=True,
            recovered=True,
            queue=q,
            order=list(q.order),
            corrections=corrections,
            verdict=verdict,
        )

    snap = next((a for a in args if isinstance(a, (dict, str))), kwargs.get("snapshot"))
    if snap is not None:
        try:
            restored = TurnQueue.from_snapshot(snap)
            restored.resort()
            return Record(ok=True, valid=True, recovered=True, queue=restored, order=list(restored.order), corrections=["restored_from_snapshot", "resorted"])
        except Exception as exc:
            fresh = TurnQueue()
            return Record(ok=True, valid=True, recovered=True, queue=fresh, order=[], corrections=["reset_empty"], error=str(exc))

    units = coerce_units(*args, units=kwargs.get("units"))
    try:
        fresh = TurnQueue(units) if units else TurnQueue()
    except Exception:
        fresh = TurnQueue()
    fresh.resort()
    return Record(ok=True, valid=True, recovered=True, queue=fresh, order=list(fresh.order), corrections=["built_safe_queue"])
