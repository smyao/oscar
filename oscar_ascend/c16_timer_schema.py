"""Archive #133/#145/#164: actor-local device clocks, never cross-core wall sums."""
SCHEMA_ID = "oscar_c16_device_cycles_v1"
SCHEMA_VERSION = 1
CLOCK_HZ = 50_000_000
CLOCK_SOURCE = "csrc/experiments/c16_timer_20261019/clock_contract.json: pinned native TIME_CYCLE=50 ticks/us provenance; A2 ascend910b4 build only"
MAGIC = 0x4F534331
HEADER_ROWS = 8
PHASE_NAMES = (
    "task_metadata", "query_load", "history_load", "precise_load", "unpack_k", "unpack_v",
    "publish_kv", "qk", "softmax_score_load", "softmax_math", "softmax_publish_p", "pv",
    "state_restore", "state_spill", "final_rv", "normalize", "output_publish",
    "cross_core_wait", "b1_load", "bf16_cast", "task_other", "cluster_other", "init", "process_other")
PHASE_CATEGORIES = {name: ("wait" if name == "cross_core_wait" else
    "compute" if name in {"unpack_k", "unpack_v", "qk", "softmax_math", "pv", "final_rv", "normalize", "bf16_cast"} else
    "data_movement" if name in {"history_load", "publish_kv", "softmax_score_load", "softmax_publish_p",
                               "state_restore", "state_spill", "output_publish", "b1_load"} else "mixed")
    for name in PHASE_NAMES}
COUNTER_ROWS = HEADER_ROWS + len(PHASE_NAMES)


def counter_shape(cores):
    if type(cores) is not int or not 1 <= cores <= 32:
        raise ValueError("C16 timer requires 1..32 Cube cores")
    return (3 * cores, COUNTER_ROWS, 2)


def decode_ticks(rows, cores):
    expected = counter_shape(cores)
    issues, actors = [], []
    if not isinstance(rows, list) or len(rows) != expected[0]:
        return {"valid": False, "issues": ["actor_count_mismatch"], "actors": []}
    for index, values in enumerate(rows):
        if (not isinstance(values, list) or len(values) != COUNTER_ROWS or
                any(not isinstance(row, list) or len(row) != 2 or any(type(v) is not int for v in row) for row in values)):
            issues.append(f"actor{index}:invalid_row_shape")
            continue
        core, engine_index = divmod(index, 3)
        engine = ("AIC", "AIV0", "AIV1")[engine_index]
        total, recorded_sum = values[3]
        flags, ready = values[4]
        phase_sum = sum(row[0] for row in values[HEADER_ROWS:])
        expected_lane = -1 if engine_index == 0 else engine_index - 1
        valid_header = (values[0] == [MAGIC, SCHEMA_VERSION] and values[1] == [index, engine_index]
                        and values[2] == [core, expected_lane] and values[7] == [len(PHASE_NAMES), COUNTER_ROWS * 2])
        if not valid_header:
            issues.append(f"actor{index}:invalid_header")
        if ready != 1 or flags != 0 or values[5][1] != 0:
            issues.append(f"actor{index}:unfinished_or_overflow:{flags}:{ready}")
        if total < 0 or any(v < 0 for row in values[HEADER_ROWS:] for v in row):
            issues.append(f"actor{index}:negative_counter")
        if phase_sum != recorded_sum or phase_sum > total:
            issues.append(f"actor{index}:exclusive_accounting_mismatch")
        if values[6][1] - values[6][0] != total:
            issues.append(f"actor{index}:clock_interval_mismatch")
        actors.append({"core": core, "engine": engine, "actor_index": index,
                       "total_cycles": total, "other_cycles": total - phase_sum,
                       "flags": flags, "ready": ready, "max_stack_depth": values[5][0],
                       "phases": {name: {"cycles": values[HEADER_ROWS + p][0],
                                         "count": values[HEADER_ROWS + p][1]}
                                  for p, name in enumerate(PHASE_NAMES)}})
    return {"valid": not issues, "issues": issues, "actors": actors}
