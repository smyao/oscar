"""Explicit P -> MemCache -> D handoff for OSCAR's complete hybrid state.

Archive #37-49/#127/#128: retain native source pages until every TP save.
Archive #157/#163/#171: committed GDN cursor, never speculative or hashed-only
state. #183: bounded missing-manifest polling, never a silent cache fallback.
Native evidence: MooncakeConnectorScheduler._truncate_request_for_prefill,
Scheduler._free_request/_update_from_kv_xfer_finished and WAITING_FOR_REMOTE_KVS.
P computes h(N-1); D executes the original final prompt token. P's sample is
discarded. Payloads are immutable cache objects, reclaimed by MemCache eviction,
like shared prefixes; they are not durable or leased active swap objects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
import time
import uuid

from .plans import PayloadPlan, expected_keys
from .scheduler import OffloadLifecycleError, RequestSnapshot, freeze_block_ids, token_digest
from .transport import TransferError

SCHEMA = "oscar-pd-v1"
TICKET_FIELDS = {"schema", "namespace", "nonce", "prompt_digest", "prompt_tokens", "committed_tokens"}


def prompt_digest(request):
    tokens = request.prompt_token_ids
    if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or getattr(request, "prompt_embeds", None) is not None
            or getattr(request, "mm_features", None) or getattr(request, "lora_request", None)
            or getattr(request, "resumable", False)):
        raise OffloadLifecycleError("OSCAR PD currently requires non-streaming text input without embeddings/MM/LoRA")
    return hashlib.sha256(json.dumps([tokens, getattr(request, "cache_salt", None)],
                                     separators=(",", ":")).encode()).hexdigest()


def validate_ticket(ticket, namespace):
    if (not isinstance(ticket, dict) or set(ticket) != TICKET_FIELDS
            or ticket["schema"] != SCHEMA or ticket["namespace"] != namespace
            or not isinstance(ticket["nonce"], str) or re.fullmatch(r"[a-f0-9]{32}", ticket["nonce"]) is None
            or not isinstance(ticket["prompt_digest"], str)
            or re.fullmatch(r"[a-f0-9]{64}", ticket["prompt_digest"]) is None
            or type(ticket["prompt_tokens"]) is not int or ticket["prompt_tokens"] < 1
            or type(ticket["committed_tokens"]) is not int
            or ticket["committed_tokens"] != ticket["prompt_tokens"] - 1):
        raise OffloadLifecycleError("incompatible OSCAR PD ticket/model/PT/layout/TP identity")
    return ticket


def index_key(ticket):
    return f"oscar-pd-index-v1:{ticket['namespace']}:{ticket['nonce']}"


def base_key(ticket):
    return f"prefix/data/pd/{ticket['nonce']}"


@dataclass(frozen=True)
class Handoff:
    snapshot: RequestSnapshot
    ticket: dict


@dataclass(frozen=True)
class HandoffReceipt:
    handoff: Handoff
    rank: int


class PDState:
    def __init__(self, connector, role, timeout):
        self.connector, self.role, self.timeout = connector, role, timeout
        self.requests = {}
        self.pending = []
        self.publishing = {}
        self.receipts = []
        self.sent = set()
        self.waiting = {}
        self.matches = {}

    def prepare(self, request):
        """Before native add_request/cache lookup, not after local hash matching."""
        params = getattr(request, "kv_transfer_params", None)
        if params is None:
            return  # Ordinary probe/health requests retain the existing path.
        if not isinstance(params, dict):
            raise OffloadLifecycleError("PD transfer parameters must be an object")
        producing, consuming = params.get("do_remote_decode", False), params.get("do_remote_prefill", False)
        if type(producing) is not bool or type(consuming) is not bool or producing == consuming:
            raise OffloadLifecycleError("PD request must select exactly one transfer direction")
        if self.role != ("prefill" if producing else "decode"):
            raise OffloadLifecycleError("PD request was routed to the wrong role")
        existing = self.requests.get(request.request_id)
        if existing is not None:
            if existing[0] is request:
                return
            raise OffloadLifecycleError("PD request ID is still owned by an earlier generation")
        digest = prompt_digest(request)
        if producing:
            if request.max_tokens != 1 or getattr(request.sampling_params, "prompt_logprobs", None) is not None:
                raise OffloadLifecycleError("P requests require max_tokens=1 and no prompt logprobs")
            ticket = dict(schema=SCHEMA, namespace=self.connector.namespace, nonce=uuid.uuid4().hex,
                          prompt_digest=digest, prompt_tokens=request.num_prompt_tokens,
                          committed_tokens=request.num_prompt_tokens - 1)
            if ticket["committed_tokens"]:
                # The identical boundary used by native Mooncake for Mamba.
                # Trim hashes too, before any local prefix lookup can see N.
                request.prompt_token_ids = request.prompt_token_ids[:-1]
                request._all_token_ids.pop()
                request.num_prompt_tokens -= 1
                del request.block_hashes[request.num_prompt_tokens // self.connector._hash_block_size():]
        else:
            ticket = dict(validate_ticket(params.get("oscar_pd"), self.connector.namespace))
            if digest != ticket["prompt_digest"] or request.num_prompt_tokens != ticket["prompt_tokens"]:
                raise OffloadLifecycleError("D prompt/tokenization/cache salt differs from P")
            if getattr(request.sampling_params, "prompt_logprobs", None) is not None:
                raise OffloadLifecycleError("PD cannot reconstruct full prompt logprobs from a KV checkpoint")
        self.requests[request.request_id] = (request, ticket)

    def lookup(self, request, local_tokens):
        state = self.requests.get(request.request_id)
        if state is None or self.role != "decode" or getattr(request, "_oscar_pd_loaded", False):
            return None
        ticket = state[1]
        count = ticket["committed_tokens"]
        if local_tokens >= count:
            request._oscar_pd_loaded = True
            return 0, False
        now = time.monotonic()
        start, next_query = self.waiting.setdefault(request.request_id, (now, now))
        if now - start >= self.timeout:
            raise TransferError("OSCAR PD handoff publication timed out; D will not recompute the prompt")
        if now < next_query:
            return None, False
        self.waiting[request.request_id] = (start, now + 0.05)
        c = self.connector
        manifest = c.metadata_client.get(index_key(ticket))
        if manifest is None:
            return None, False
        generation = c.scheduler_state.observe_request(request)
        plan = PayloadPlan(request.request_id, generation, 0, c.scheduler_state.epoch,
                           count, (), base_key(ticket), local_tokens=local_tokens)
        expected = self.manifest(Handoff(self.snapshot(request, (), count), ticket), plan)
        if manifest != expected:
            raise TransferError("OSCAR PD manifest has a different ticket or incomplete/mixed TP payload")
        if not all(c.transport.exists(expected["keys"])):
            raise TransferError("OSCAR PD committed payload was evicted or is incomplete")
        self.matches[request.request_id] = plan
        return count - local_tokens, True

    def allocate(self, request, blocks):
        plan = self.matches.pop(request.request_id, None)
        if plan is None:
            return None
        request._oscar_pd_loaded = True
        self.waiting.pop(request.request_id, None)
        return replace(plan, block_ids=freeze_block_ids(blocks))

    def snapshot(self, request, blocks, count):
        c = self.connector
        generation = c.scheduler_state.observe_request(request)
        return RequestSnapshot(request.request_id, generation, 0, c.scheduler_state.epoch,
                               freeze_block_ids(blocks), count, request.num_prompt_tokens,
                               0, token_digest(()), c.scheduler_state.scheduler.cache_config.mamba_cache_mode)

    def finish(self, request, blocks):
        state = self.requests.pop(request.request_id, None)
        self.waiting.pop(request.request_id, None)
        self.matches.pop(request.request_id, None)
        if state is None or self.role != "prefill":
            return False, None
        # Aborted/failed P requests never advertise a usable ticket.
        if getattr(request.status, "name", str(request.status)) != "FINISHED_LENGTH_CAPPED":
            return False, None
        ticket = state[1]
        count = ticket["committed_tokens"]
        if count:
            handoff = Handoff(self.snapshot(request, blocks, count), ticket)
            self.pending.append(handoff)
            self.publishing[ticket["nonce"]] = (handoff, {})
        return bool(count), dict(do_remote_prefill=True, do_remote_decode=False, oscar_pd=ticket)

    def drain(self):
        result = tuple(self.pending)
        self.pending.clear()
        return result

    def save(self, handoffs):
        c = self.connector
        for handoff in handoffs:
            validate_ticket(handoff.ticket, c.namespace)
            snapshot = handoff.snapshot
            resolved = c.tracker.resolve(c.runner, snapshot, clone=False)
            plan = PayloadPlan(snapshot.request_id, snapshot.generation, 0, snapshot.epoch,
                               handoff.ticket["committed_tokens"], resolved.block_ids, base_key(handoff.ticket))
            c._save_resolved(plan, resolved)
            self.receipts.append(HandoffReceipt(handoff, c.rank))
            self.sent.add(snapshot.request_id)

    def manifest(self, handoff, plan):
        c = self.connector
        return dict(namespace=c.namespace, index_key=index_key(handoff.ticket),
                    committed_tokens=plan.committed_tokens, base_key=plan.base_key,
                    oscar_pd=handoff.ticket, keys=list(expected_keys(c.namespace, c.world_size, plan, c.geometry)))

    def acknowledge(self, receipt):
        entry = self.publishing.get(receipt.handoff.ticket["nonce"])
        if entry is None or entry[0] != receipt.handoff or not 0 <= receipt.rank < self.connector.world_size:
            raise OffloadLifecycleError("stale or conflicting PD save completion")
        handoff, ranks = entry
        if receipt.rank in ranks:
            raise OffloadLifecycleError("duplicate PD save completion")
        ranks[receipt.rank] = receipt
        if len(ranks) == self.connector.world_size:
            s, ticket = handoff.snapshot, handoff.ticket
            plan = PayloadPlan(s.request_id, s.generation, 0, s.epoch,
                               ticket["committed_tokens"], (), base_key(ticket))
            self.connector.metadata_client.put(index_key(ticket), self.manifest(handoff, plan))
            del self.publishing[ticket["nonce"]]
            print(f"[oscar] PD_HANDOFF_PUBLISHED request={s.request_id} tokens={plan.committed_tokens}", flush=True)
