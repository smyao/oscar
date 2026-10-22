"""Archive #26-49/#69/#127-129/#137/#140/#155: host storage contracts only.

Small CPU tensors and meta tensors validate addresses/identities. They do not
execute a CANN model, simulate performance, or prove NPU/graph/async safety.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from oscar_ascend.request_windows import WindowBatch, WindowLeaseError
from oscar_ascend.request_window_storage import (
    INT64_MAX, INVALID_TAG, FixedWindowMap, RequestWindowLayout,
)

ROOT = Path(__file__).resolve().parents[1]


def event(*, new=(), finished=(), preempted=(), resumed=()):
    return NS(scheduled_new_reqs=[NS(req_id=key) for key in new],
              scheduled_cached_reqs=NS(resumed_req_ids=set(resumed)),
              finished_req_ids=set(finished), preempted_req_ids=set(preempted))


class LayoutTests(unittest.TestCase):
    def target(self):
        config = json.loads((ROOT / "configs/target.json").read_text())
        # Keep MTP rollback coverage independent of the serving default.
        config["speculative_config"] = {"num_speculative_tokens": 3}
        return RequestWindowLayout.from_config(config,
                                               num_kv_heads=1, head_dim=256)

    def small(self):
        return RequestWindowLayout(3, 2, 5, 3, 1, 64, 256)

    def test_target_budget_meta_views_and_tag_limits(self):
        layout = self.target()
        self.assertEqual(layout.kv_shape, (128, 323, 1, 256))
        self.assertEqual(layout.tag_shape, (128, 323))
        self.assertEqual(layout.map_shape, (128, 2))
        self.assertEqual(layout.required_bytes, 128 * 323 * (1024 + 8))
        self.assertEqual(layout.map_bytes, 2048)
        self.assertEqual(layout.position_bits, 18)
        self.assertEqual(layout.max_epoch, (1 << 45) - 1)
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8, device="meta")
        key, value, tags = layout.views(raw, testing=True)
        self.assertEqual((key.shape, value.shape, tags.shape), (layout.kv_shape, layout.kv_shape, layout.tag_shape))
        self.assertEqual((key.dtype, value.dtype, tags.dtype), (torch.bfloat16, torch.bfloat16, torch.int64))
        self.assertEqual(layout.encode_tag(layout.max_epoch, layout.max_model_len - 1), INT64_MAX)

    def test_ring_address_is_logical_and_sidecar_views_match_byte_offsets(self):
        layout = self.small()
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8)
        with self.assertRaisesRegex(WindowLeaseError, "explicit boolean"):
            layout.views(raw, testing="false")
        key, value, tags = layout.views(raw, testing=True)
        tags.fill_(INVALID_TAG)
        for slot in range(layout.max_num_seqs):
            for position in (0, 1, 2, 7, 8, 31, 127, 255):
                addr = layout.address(slot, position)
                expected = position if position < 2 else 2 + position % 8
                self.assertEqual(addr.row, expected)
                self.assertEqual(key[slot, addr.row].data_ptr() - raw.data_ptr(), addr.key_offset)
                self.assertEqual(value[slot, addr.row].data_ptr() - raw.data_ptr(), addr.value_offset)
                self.assertEqual(tags[slot, addr.row].data_ptr() - raw.data_ptr(), addr.tag_offset)
                self.assertLess(addr.tag_offset + 7, layout.required_bytes)
        # A physical dual-segment boundary does not reset logical ring modulo.
        target = self.target()
        self.assertEqual(target.row_for_position(2816), 64 + 2816 % 259)
        self.assertEqual(target.row_for_position(5632), 64 + 5632 % 259)

    def test_mtp_reject_retention_and_no_ring_writer_collision(self):
        layout = self.target()
        for end in (0, 1, 64, 65, 256, 323, 5632, 65100, 65615):
            keep = [position for span in layout.retained_ranges(end) for position in span]
            self.assertLessEqual(len(keep), layout.rows)
            self.assertEqual(len({layout.row_for_position(pos) for pos in keep}), len(keep))
            for rejected in range(min(layout.speculative_tokens, end) + 1):
                committed = end - rejected
                needed = set(range(min(layout.sink_tokens, committed)))
                needed.update(range(max(layout.sink_tokens, committed - layout.recent_tokens), committed))
                self.assertTrue(needed <= set(keep), (end, rejected))

    def test_epoch_blocks_reused_window_and_same_physical_page_identity(self):
        layout = self.small()
        leases = layout.make_leases()
        old = leases.update(event(new=("a",)), ["a"])
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8)
        _, _, tags = layout.views(raw, testing=True)
        tags.fill_(INVALID_TAG)
        position = 7
        row = layout.row_for_position(position)
        tags[old.slots[0], row] = layout.encode_tag(old.epochs[0], position)
        # The packed physical page may be reused at the same address; it is
        # deliberately absent from sidecar identity/address calculations.
        fresh = leases.update(event(finished=("a",), new=("b",)), ["b"])
        self.assertEqual(old.slots, fresh.slots)
        self.assertNotEqual(tags[fresh.slots[0], row].item(), layout.encode_tag(fresh.epochs[0], position))
        for epoch, pos in ((1, 0), (5, 255), (layout.max_epoch, 255)):
            self.assertEqual(layout.decode_tag(layout.encode_tag(epoch, pos)), (epoch, pos))

    def test_integer_bounds_and_wrong_device_or_storage_fail(self):
        layout = self.small()
        for epoch, position in ((0, 1), (True, 1), (layout.max_epoch + 1, 0), (1, -1), (1, 256), (1, True)):
            with self.subTest(epoch=epoch, position=position), self.assertRaises(WindowLeaseError):
                layout.encode_tag(epoch, position)
        for slot in (-1, 3, True):
            with self.assertRaises(WindowLeaseError):
                layout.address(slot, 0)
        with self.assertRaises(WindowLeaseError):
            replace(layout, max_model_len=1 << 63)
        with self.assertRaises(WindowLeaseError):
            replace(layout, max_num_seqs=INT64_MAX)
        with self.assertRaises(WindowLeaseError):
            replace(layout, recent_tokens=0)
        non_power_two = replace(layout, max_model_len=255)
        with self.assertRaises(WindowLeaseError):
            non_power_two.decode_tag(1 * 256 + 255)
        raw = torch.empty(layout.required_bytes, dtype=torch.uint8)
        with self.assertRaisesRegex(WindowLeaseError, "NPU"):
            layout.views(raw)
        with self.assertRaises(WindowLeaseError):
            layout.views(raw[:-1], testing=True)
        backing = torch.empty(layout.required_bytes + 1, dtype=torch.uint8)
        with self.assertRaises(WindowLeaseError):
            layout.views(backing[1:], testing=True)


class MapTests(unittest.TestCase):
    def setUp(self):
        self.layout = RequestWindowLayout(3, 2, 5, 3, 1, 64, 256)
        self.host = torch.empty(self.layout.map_shape, dtype=torch.int64)
        self.device = torch.full(self.layout.map_shape, -99, dtype=torch.int64)
        self.binding = FixedWindowMap(self.layout, self.host, self.device, testing=True)
        self.leases = self.layout.make_leases()

    def test_reorder_and_same_address_map_updates_without_device_publication(self):
        ptrs = (self.host.data_ptr(), self.device.data_ptr())
        batch = self.leases.update(event(new=("a", "b")), ["a", "b"])
        self.binding.write_host(batch)
        self.assertEqual(self.host.tolist(), [[0, 1], [1, 2], [-1, 0]])
        self.assertEqual(self.device.tolist(), [[-99, -99]] * 3)
        self.binding.write_host(self.leases.update(event(), ["b", "a"]))
        self.assertEqual(self.host.tolist(), [[1, 2], [0, 1], [-1, 0]])
        self.assertEqual((self.host.data_ptr(), self.device.data_ptr()), ptrs)
        self.binding.write_host(self.leases.update(event(finished=("a", "b")), [], padded_rows=3))
        self.assertEqual(self.host.tolist(), [[-1, 0]] * 3)

    def test_map_rejects_aliases_reallocation_bad_epoch_duplicate_slot_and_bad_padding(self):
        with self.assertRaisesRegex(WindowLeaseError, "explicit boolean"):
            FixedWindowMap(self.layout, self.host, self.device, testing="false")
        with self.assertRaisesRegex(WindowLeaseError, "alias"):
            FixedWindowMap(self.layout, self.host, self.host, testing=True)
        self.host.fill_(77)
        for batch in (WindowBatch(("a",), (0,), (self.layout.max_epoch + 1,)),
                      WindowBatch(("a", "b"), (0, 0), (1, 2)),
                      WindowBatch(("a",), (0, -1), (1, 1))):
            with self.assertRaises(WindowLeaseError):
                self.binding.write_host(batch)
            self.assertEqual(self.host.tolist(), [[77, 77]] * 3)
        self.binding.device_map = self.device.clone()
        with self.assertRaisesRegex(WindowLeaseError, "address/shape changed"):
            self.binding.write_host(WindowBatch((), (), ()))

    def test_configured_capacity_has_no_silent_lease_or_map_overrun(self):
        batch = self.leases.update(event(new=("a", "b", "c")), ["c", "a", "b"])
        self.binding.write_host(batch)
        self.assertEqual(self.host.shape, (3, 2))
        with self.assertRaises(WindowLeaseError):
            self.leases.update(event(new=("d",)), ["d"])
        self.assertEqual(len(self.leases.snapshot()), 3)
        with self.assertRaises(WindowLeaseError):
            self.layout.map_rows(WindowBatch(("a",), (3,), (1,)))


if __name__ == "__main__":
    unittest.main()
