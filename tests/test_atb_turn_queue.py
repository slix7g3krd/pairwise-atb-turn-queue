import pytest
import gameplay.atb_turn_queue as mod

CASES = [
    ("stable_tiebreak", (1,)),
    ("callback_generation", (1,)),
    ("logical_timeout", (1,)),
    ("requeue_on_speed", (1,)),
    ("teardown_refs", (1,)),
    ("snapshot_replay", (1,)),
    ("build_queue", (1,)),
    ("advance", (1,)),
    ("insert_unit", (1,)),
    ("remove_unit", (1,)),
    ("speed_change", (1,)),
    ("validate_order", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
