"""Archive #126/#129/#148/#155: host-only proof of a fresh current suffix.

A native CPU seq-length UPPER bound equal to this request's query length
proves its historical context is zero, provided device metadata is checked
before execution. Positive/unknown bounds never justify dropping history.
This module plans only; no serving route is currently enabled.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CurrentOnlySuffix:
    prefix_requests: int
    prefix_tokens: int
    fresh_requests: int
    fresh_tokens: int
    padded_tokens: int
    cumulative: tuple[int, ...]


def plan_current_suffix(query_starts, sequence_upper_bounds, *, actual_tokens,
                        padded_tokens):
    starts=tuple(query_starts);upper=tuple(sequence_upper_bounds)
    if (not starts or starts[0]!=0 or len(starts)<2 or
            any(type(v) is not int for v in (*starts,*upper)) or
            any(b<a for a,b in zip(starts,starts[1:])) or
            type(actual_tokens) is not int or type(padded_tokens) is not int or
            not 0<actual_tokens<=padded_tokens or actual_tokens not in starts):
        raise ValueError('invalid native CPU query/length capacity')
    end=starts.index(actual_tokens)
    if end>len(upper) or any(starts[i+1]==starts[i] for i in range(end)):
        raise ValueError('real query intervals must be nonempty and have length bounds')
    for i in range(end):
        if upper[i]<starts[i+1]-starts[i]:
            raise ValueError('native sequence upper bound is below query length')
    begin=end
    while begin>0 and upper[begin-1]==starts[begin]-starts[begin-1]:
        begin-=1
    if begin==end:
        return None
    prefix=starts[begin]
    return CurrentOnlySuffix(begin,prefix,end-begin,actual_tokens-prefix,
                             padded_tokens-actual_tokens,
                             tuple(value-prefix for value in starts[begin+1:end+1]))
