"""Archive #126/#145/#148/#154/#157 and startup D.4: independent flag review.

Enumerate every ordering of the Cube and two AIV host protocol actors for one
16-member round. This models mode-2 broadcast/ack tokens only; DMA progress,
pipeline ordering, arithmetic, device execution and performance are not modeled.
The reviewed kernel is read and fingerprinted, never edited or executed.
"""

from collections import deque
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "csrc/experiments/c16_l1_reuse_20261015/c16_candidate.cpp"


def enumerate_protocol():
    cube = []
    for member in range(16):
        if member >= 4:
            cube.append(("wait_p", member % 4))
        cube.append(("set_s", member % 4))
    cube.extend(("wait_p", slot) for slot in range(4))
    vector = [action for member in range(16)
              for action in (("wait_s", member % 4), ("set_p", member % 4))]
    # Each mode-2 score broadcasts one token to each AIV. Each AIV sends
    # its own P token; Cube can consume the pair only after both arrive.
    start = (0, 0, 0, 0, 0)
    pending, seen, findings = deque([start]), {start}, []
    terminal = 0
    while pending:
        state = pending.popleft()
        pcs, score, pready = list(state[:3]), state[3], state[4]
        if pcs == [len(cube), len(vector), len(vector)]:
            terminal += 1
            if score or pready:
                findings.append({"kind": "undrained", "state": state})
            continue
        moves = 0
        for actor, program in enumerate((cube, vector, vector)):
            if pcs[actor] >= len(program):
                continue
            kind, slot = program[pcs[actor]]
            s, p = score, pready
            mask = ((1 << slot) | (1 << (slot + 4)) if actor == 0
                    else 1 << (slot + (actor - 1) * 4))
            if kind == "wait_p":
                if p & mask != mask:
                    continue
                p &= ~mask
            elif kind == "set_s":
                if s & mask:
                    findings.append({"kind": "score_flag_overwrite", "state": state})
                    continue
                s |= mask
            elif kind == "wait_s":
                if not s & mask:
                    continue
                s &= ~mask
            else:
                if p & mask:
                    findings.append({"kind": "p_flag_overwrite", "state": state})
                    continue
                p |= mask
            after = pcs.copy()
            after[actor] += 1
            successor = (*after, s, p)
            moves += 1
            if successor not in seen:
                seen.add(successor)
                pending.append(successor)
        if moves == 0:
            findings.append({"kind": "deadlock", "state": state})
    return {"states": len(seen), "terminal_states": terminal, "findings": findings}


def main():
    result = {"status": "host_protocol_enumerated", "source": SOURCE,
              "source_sha256": hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(),
              "protocol": "16 members, 4 score/ack pairs, 1 Cube, 2 independent AIV actors",
              **enumerate_protocol(),
              "limitations": ["host token model only", "no DMA or Matmul simulation",
                              "no NPU execution", "no numerical or performance acceptance"]}
    output = Path(__file__).parent / "independent_review.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["findings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
