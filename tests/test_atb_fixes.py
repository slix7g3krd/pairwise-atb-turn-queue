"""六个修复点的端到端回归测试。

这些测试刻意复现任务描述里的故障剧情：同速每帧互换、击倒后回调仍推进、
暂停期间墙钟超时、加速不重排剩余队列、清场后跨战斗回调、快照不可重放。
只依赖标准库与 pytest。
"""

from __future__ import annotations

import gc
import json
import weakref

import pytest

from gameplay import atb_turn_queue as mod
from gameplay.atb_turn_queue import (
    AnimationDirector,
    AnimationHandle,
    Battle,
    LogicalTimeout,
    TurnQueue,
    Unit,
    advance,
    audit,
    build_queue,
    callback_generation,
    fallback,
    insert_unit,
    logical_timeout,
    metrics,
    remove_unit,
    requeue_on_speed,
    snapshot_replay,
    stable_tiebreak,
    state_digest,
    teardown_refs,
    timed_out,
    validate_order,
)


# ---------------------------------------------------------------------------
# 1. 同速稳定 ID 排序
# ---------------------------------------------------------------------------


class TestStableTiebreak:
    def test_equal_speed_ties_use_stable_id_ascending(self):
        units = [
            {"id": "b", "speed": 10},
            {"id": "a", "speed": 10},
            {"id": "c", "speed": 12},
        ]
        result = stable_tiebreak(units)
        assert result.ids == ["c", "a", "b"]
        # 返回列表可直接与裸 ID 列表比较
        assert result == ["c", "a", "b"]

    def test_reversed_input_and_repeated_resort_identical(self):
        units = [{"id": i, "speed": s} for i, s in [(1, 10), (2, 10), (3, 10), (4, 20)]]
        forward = stable_tiebreak(units).ids
        backward = stable_tiebreak(list(reversed(units))).ids
        assert forward == backward == [4, 1, 2, 3]

        q = build_queue([(1, 10), (2, 10), (3, 10)])
        first = list(q.order)
        for _ in range(30):  # 模拟“每帧重排”
            q.resort()
            q.reorder_remaining()
        assert q.order == first == [1, 2, 3]

    def test_accepts_tuples_objects_and_mapping(self):
        assert stable_tiebreak([(2, 10), (1, 10)]).ids == [1, 2]
        assert stable_tiebreak({3: 9, 1: 9, 2: 9}).ids == [1, 2, 3]
        objs = [Unit(id="x", speed=1), Unit(id="y", speed=1)]
        assert stable_tiebreak(objs).ids == ["x", "y"]

    def test_numeric_string_ids_sort_numerically(self):
        assert stable_tiebreak(["10", "2", "1"]).ids == ["1", "2", "10"]


# ---------------------------------------------------------------------------
# 2. 回调代际校验
# ---------------------------------------------------------------------------


class TestCallbackGeneration:
    def test_pure_generation_equality_is_order_independent(self):
        assert callback_generation(3, 3) is True
        assert callback_generation(2, 3) is False
        u = Unit(id=1, generation=5)
        assert callback_generation(u, 5) is True
        assert callback_generation(u, 4) is False

    def test_stale_callback_after_knockout_does_not_advance(self):
        q = build_queue([(1, 20), (2, 10), (3, 5)])
        handle = q.bind_animation(1)
        assert handle.generation == 1

        # 动画还没播完，单位先被击倒 / 移除 -> 代际抬升、指针保持有效
        removed = q.remove_unit(1)
        assert removed.generation == 2
        assert q.current_id == 2
        cursor_after_remove = q.cursor

        accepted = handle.complete()
        assert accepted is False
        # 关键：陈旧回调没有额外推进队列，也没有行动记录
        assert q.cursor == cursor_after_remove
        assert q.action_log == []

    def test_current_callback_advances_exactly_once(self):
        q = build_queue([(1, 20), (2, 10)])
        handle = q.bind_animation(1)
        assert handle.complete() is True
        assert q.action_log == [1]
        # 重复完成事件被消费一次，不再推进
        assert handle.complete() is False
        assert q.action_log == [1]

    def test_guarded_module_helper_invokes_only_when_fresh(self):
        q = build_queue([(1, 10)])
        calls: list[int] = []
        assert callback_generation(q, 1, 1, callback=lambda: calls.append(1)) is True
        assert calls == [1]
        q.remove_unit(1)
        assert callback_generation(q, 1, 1, callback=lambda: calls.append(2)) is False
        assert calls == [1]

    def test_cross_battle_id_token_rejected(self):
        q = build_queue([(1, 10)])
        foreign = AnimationHandle(queue=q, unit_id=1, generation=1, battle_id="other-battle")
        cursor = q.cursor
        assert foreign.complete() is False
        assert q.cursor == cursor


# ---------------------------------------------------------------------------
# 3. 逻辑帧超时（暂停冻结、与墙钟无关）
# ---------------------------------------------------------------------------


class TestLogicalTimeout:
    def test_pure_two_arg_predicate_returns_literal_bool(self):
        assert logical_timeout(30, 30) is True
        assert logical_timeout(5, 30) is False
        assert timed_out(30, 30) is True

    def test_accumulates_logical_frames_only(self):
        t = logical_timeout(30)
        for _ in range(29):
            t.tick()
        assert t.expired is False
        assert t.remaining_frames == 1
        assert bool(t) is False
        t.tick()
        assert t.expired is True

    def test_pause_menu_freezes_timeout(self):
        t = LogicalTimeout(limit_frames=30)
        t.tick(20)
        t.pause()
        # 暂停期间无论多少渲染帧 / 墙钟时间，都不累计
        t.tick(10_000)
        assert t.elapsed_frames == 20
        assert t.expired is False
        t.resume()
        t.tick(10)
        assert t.expired is True

    def test_battle_logic_frame_paused_does_not_accumulate(self):
        battle = Battle([(1, 10)], timeout_frames=5)
        for _ in range(4):
            battle.logic_frame()
        assert battle.input_timeout.expired is False
        battle.open_menu()
        for _ in range(100):
            battle.logic_frame()
        assert battle.input_timeout.elapsed_frames == 4
        assert battle.input_timeout.expired is False
        battle.close_menu()
        battle.logic_frame()
        assert battle.input_timeout.expired is True

    def test_module_does_not_read_wall_clock(self):
        # 结构性保证：超时模块源码不引用 time/datetime/monotonic
        import inspect

        source = inspect.getsource(mod)
        assert "time.time(" not in source
        assert "monotonic(" not in source


# ---------------------------------------------------------------------------
# 4. 速度变更重排剩余队列
# ---------------------------------------------------------------------------


class TestRequeueOnSpeed:
    def test_haste_promotes_unit_before_any_advance(self):
        q = build_queue([(1, 10), (2, 9), (3, 8)])
        q.change_speed(3, 99)
        assert q.order == [3, 1, 2]
        assert validate_order(q).valid is True

    def test_only_remaining_tail_reordered_mid_round(self):
        q = build_queue([(1, 10), (2, 9), (3, 8), (4, 7)])
        q.advance()  # 1 已行动，指针指向 2
        assert q.cursor == 1
        q.change_speed(4, 99)
        # 已行动的 1 冻结在前，4 升到剩余队列最前
        assert q.order == [1, 4, 2, 3]
        assert q.remaining_ids == [4, 2, 3]

    def test_requeue_entry_returns_new_order(self):
        result = requeue_on_speed([(1, 10), (2, 9), (3, 8)], 2, 77)
        assert result.ids == [2, 1, 3]

    def test_slow_current_tail_unit_demotes(self):
        q = build_queue([(1, 50), (2, 40), (3, 30)])
        q.change_speed(1, 1)  # 尚未 advance，整个剩余队列参与重排
        assert q.order == [2, 3, 1]


# ---------------------------------------------------------------------------
# 5. 清场解绑双向引用 + 跨战斗回调隔离
# ---------------------------------------------------------------------------


class TestTeardownRefs:
    def test_teardown_nulls_both_directions(self):
        battle = Battle([(1, 10), (2, 9)])
        handle = battle.play_current_action(name="attack")
        q = battle.queue
        assert handle.queue is q
        assert q.active_handle is handle

        rec = teardown_refs(battle)
        assert rec["ok"] is True
        assert handle.queue is None
        assert handle.on_complete is None
        assert handle.director is None
        assert q.active_handle is None
        assert len(q.handles) == 0
        assert q.torn_down is True

    def test_old_handle_completion_does_not_reach_new_battle(self):
        b1 = Battle([(1, 10), (2, 9)])
        old = b1.play_current_action(name="attack")
        b1.teardown()
        assert old.complete() is False

        b2 = Battle([(1, 10), (2, 9)])
        cursor = b2.queue.cursor
        # 即便外部把旧句柄重新接到新队列（最糟的闭包串场），代际 / 战斗 ID 仍拒绝
        old.queue = b2.queue
        old.released = False
        old.cancelled = False
        assert old.complete() is False
        assert b2.queue.cursor == cursor
        assert b2.queue.action_log == []

    def test_shared_director_drops_old_animation_events(self):
        director = AnimationDirector()
        b1 = Battle([(1, 10)])
        b1.queue.director = director
        director.queue = b1.queue
        b1.queue.bind_animation(1, director=director, name="skill")
        b1.teardown()
        assert director.active_count == 0
        assert director.complete("skill") is False

    def test_cycle_is_garbage_collected_after_teardown(self):
        battle = Battle([(1, 10), (2, 9)])
        handle = battle.play_current_action()
        queue_ref = weakref.ref(battle.queue)
        battle_ref = weakref.ref(battle)
        battle.teardown()
        # 外部可能仍持有动画句柄，但它已经不再引用队列
        assert handle.queue is None
        del battle, handle
        gc.collect()
        assert queue_ref() is None
        assert battle_ref() is None


# ---------------------------------------------------------------------------
# 6. 快照可重放
# ---------------------------------------------------------------------------


class TestSnapshotReplay:
    def test_snapshot_json_roundtrip_preserves_order_pointer_generation(self):
        q = build_queue([("a", 10), ("b", 10), ("c", 8)])
        q.advance()
        q.change_speed("c", 50)
        q.remove_unit("b")

        blob = json.dumps(q.snapshot(), ensure_ascii=False)
        restored = TurnQueue.from_snapshot(json.loads(blob))

        assert restored.order == q.order
        assert restored.cursor == q.cursor
        assert restored._gen == q._gen
        assert state_digest(restored) == state_digest(q)

    def test_replayed_action_sequence_is_identical(self):
        q1 = build_queue([(1, 10), (2, 9), (3, 8)])
        seq1 = [q1.advance().acted_id for _ in range(7)]

        q2 = build_queue([(1, 10), (2, 9), (3, 8)])
        seq2 = [q2.advance().acted_id for _ in range(7)]
        assert seq1 == seq2
        # 三单位循环，7 次行动应为 1,2,3,1,2,3,1
        assert seq1 == [1, 2, 3, 1, 2, 3, 1]

    def test_snapshot_replay_entry_reports_identical(self):
        report = snapshot_replay([(1, 10), (2, 9), (3, 8)], steps=4)
        assert report["ok"] is True
        assert report["identical"] is True
        assert report["sequence"] == [1, 2, 3, 1]
        # 快照本身是 JSON 安全字符串
        json.loads(report["json"])

    def test_journal_replay_rebuilds_queue(self):
        q = build_queue([(1, 10), (2, 9), (3, 8)])
        q.advance()
        q.change_speed(3, 77)
        q.insert_unit(Unit(id=4, speed=5))
        snap = q.snapshot()
        replayed = mod.replay_journal(snap["history"])
        assert replayed.order == q.order


# ---------------------------------------------------------------------------
# 基础队列操作 / 校验 / 指标 / 审计 / 兜底
# ---------------------------------------------------------------------------


class TestQueueBasics:
    def test_build_and_advance_and_insert_remove(self):
        q = build_queue([{"id": 1, "speed": 10}, {"id": 2, "speed": 20}])
        assert q.ids == [2, 1]
        step = advance(q)
        assert step.acted_id == 2
        assert q.current_id == 1

        # 回合中途插入最快单位：已行动前缀 2 冻结，新单位成为“剩余队列”之首
        insert_unit(q, {"id": 3, "speed": 30})
        assert q.ids[0] == 2
        assert q.remaining_ids[0] == 3
        assert q.current_id == 3

        remove_unit(q, 3)
        assert 3 not in q.ids
        assert validate_order(q).valid is True

    def test_insert_fastest_in_fresh_queue_goes_first(self):
        q = build_queue([(1, 10), (2, 20)])
        insert_unit(q, {"id": 3, "speed": 30})
        assert q.ids[0] == 3

    def test_remove_before_pointer_keeps_pointer_on_same_unit(self):
        q = build_queue([(1, 10), (2, 9), (3, 8), (4, 7)])
        q.advance()  # 1 acted, cursor=1 -> current 2
        assert q.current_id == 2
        # 移除已行动前缀中的 1：cursor 回收，但当前单位仍是 2
        q.remove_unit(1)
        assert q.current_id == 2

    def test_validate_detects_corruption(self):
        q = build_queue([(1, 10), (2, 9)])
        q.order.append(999)  # 指向不存在的单位
        verdict = q.validate_order()
        assert verdict.valid is False
        assert verdict.errors

    def test_metrics_audit_fallback_are_structured(self):
        q = build_queue([(1, 10), (2, 9)])
        q.advance()
        m = metrics(q)
        assert m["units_total"] == 2
        assert m["advances"] >= 1

        entry = audit("unit_tested", unit_id=1)
        assert entry["action"] == "unit_tested"
        assert any(r["action"] == "unit_tested" for r in audit())

        fb = fallback(q)
        assert fb["ok"] is True
        assert fb["queue"] is q

        broken = {"order": [1, 2], "units": [{"id": 1, "speed": 1}]}  # 缺单位 2
        fb2 = fallback(broken)
        assert fb2["recovered"] is True
        assert isinstance(fb2["order"], list)

    @pytest.mark.parametrize(
        "name",
        [
            "stable_tiebreak",
            "callback_generation",
            "logical_timeout",
            "requeue_on_speed",
            "teardown_refs",
            "snapshot_replay",
            "build_queue",
            "advance",
            "insert_unit",
            "remove_unit",
            "speed_change",
            "validate_order",
            "metrics",
            "audit",
            "fallback",
        ],
    )
    def test_every_entry_survives_scalar_one(self, name):
        fn = getattr(mod, name)
        assert fn(1) is not None
