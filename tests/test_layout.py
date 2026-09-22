"""Archive #17–20/#34/#36/#37–49: independent byte/identity regressions.

These are host geometry tests. They establish no NPU numerical, graph,
throughput, or real cache-lifecycle acceptance.
"""

import unittest
from types import SimpleNamespace

from oscar_ascend.integration.metadata import MetadataCapacity, OscarMetadataError, from_common
from oscar_ascend.layout import HybridPageLayout, MemoryBudget, SlotLayout


class TensorShape:
    """Bomb on data access so metadata tests cannot accidentally permit sync."""

    def __init__(self, *shape):
        self.shape = shape

    def __getattr__(self, name):
        raise AssertionError(f"metadata attempted tensor data access: {name}")


class CommonMetadata:
    causal = True
    query_start_loc = TensorShape(130)
    seq_lens = TensorShape(129)
    slot_mapping = TensorShape(16896)
    block_table_tensor = TensorShape(129, 2052)
    num_reqs = 128
    num_actual_tokens = 512
    max_query_len = 4
    max_seq_len = 262144
    positions = TensorShape(16896)

    @property
    def seq_lens_cpu(self):
        raise AssertionError("deprecated CPU mirror read")

    @property
    def query_start_loc_cpu(self):
        raise AssertionError("query offsets were copied to the host")


class LayoutTests(unittest.TestCase):
    def test_uncached_request_span_is_not_full_page_alignment(self):
        layout = HybridPageLayout(SlotLayout(256), 30720, 393216, 262144,
                                  native_page_size_bytes=817152,
                                  native_mamba_cache_mode="none")
        self.assertEqual(layout.alignment_tokens, 128)
        self.assertEqual(layout.block_size, 2816)
        self.assertEqual(layout.virtual_blocks_per_page, 22)
        self.assertLessEqual(layout.payload_bytes, layout.ssm_bytes)
        self.assertGreater((layout.block_size + 128) * 136, layout.ssm_bytes)
        self.assertEqual(layout.scheduler_block_size, 2883584)
        from oscar_ascend.lifecycle import SnapshotLayout
        snapshot = SnapshotLayout(layout, 64, 256, 3)
        self.assertEqual(snapshot.required_bytes, 333336)
        for page in range(3):
            start, end = layout.packed_interval(3, page)
            self.assertGreaterEqual(start, 3 * 30720)
            self.assertLessEqual(end, 3 * 30720 + (page + 1) * 393216)
            self.assertLessEqual(snapshot.interval(3, page)[1], 3 * 817152)

    def test_request_span_alignment_remains_required_for_cached_modes(self):
        for mode in ("align", "all"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "aligned OSCAR block"):
                HybridPageLayout(SlotLayout(256), 30720, 393216, 262144,
                                 native_mamba_cache_mode=mode)

    def test_pr_byte_offsets(self):
        slot = SlotLayout(256)
        self.assertEqual((slot.k_scale_offset, slot.k_zero_offset, slot.v_codes_offset,
                          slot.v_scale_offset, slot.v_zero_offset, slot.slot_bytes),
                         (64, 66, 68, 132, 134, 136))
        self.assertEqual(SlotLayout(128).slot_bytes, 72)

    def test_non_multiple_dimensions_and_heads_are_charged_separately(self):
        slot = SlotLayout(129, 65, 3)
        self.assertEqual(slot.slot_bytes, 33 + 4 + 17 + 4)
        self.assertEqual(slot.token_bytes, 174)

    def test_lcm_alignment_prevents_scheduler_lcm_explosion(self):
        layout = HybridPageLayout(SlotLayout(256), 15360, 786432, 768)
        self.assertEqual(layout.block_size, 5376)
        self.assertEqual(layout.scheduler_block_size, 5376)
        self.assertEqual(layout.virtual_blocks_per_page, 42)
        self.assertLessEqual(layout.payload_bytes, 786432)
        self.assertGreater((layout.block_size + 768) * 136, 786432)

    def test_full_regions_are_disjoint_from_all_conv_and_other_ssm_pages(self):
        layout = HybridPageLayout(SlotLayout(256), 15360, 786432, 768)
        nb = 19
        for full_id in range(nb):
            first, end = layout.packed_interval(nb, full_id)
            self.assertGreaterEqual(first, nb * 15360)
            self.assertLessEqual(end, (nb * 15360) + (full_id + 1) * 786432)
            for gdn_id in range(nb):
                conv_first, conv_end = layout.conv_interval(nb, gdn_id)
                self.assertTrue(end <= conv_first or conv_end <= first)
                if full_id != gdn_id:
                    ssm_first, ssm_end = layout.ssm_interval(nb, gdn_id)
                    self.assertTrue(end <= ssm_first or ssm_end <= first)

    def test_aos_formula_would_corrupt_a_different_gdn_page(self):
        # Independent counterexample: an AoS full page 1 starts inside GDN
        # page 0's SSM under the native SoA layout. This rules out b*P.
        nb, c, m = 19, 15360, 786432
        wrong = c + m
        self.assertTrue(nb * c <= wrong < nb * c + m)

    def test_native_virtual_slots_preserve_physical_identity(self):
        layout = HybridPageLayout(SlotLayout(128, 256, 2), 15360, 786432, 128)
        for physical in (0, 1, 7):
            for token in (0, 127, 128, layout.block_size - 1):
                virtual = physical * layout.virtual_blocks_per_page + token // 128
                slot = virtual * 128 + token % 128
                self.assertEqual(layout.virtual_to_physical(virtual, token % 128), (physical, token))
                self.assertEqual(layout.slot_to_physical(slot), (physical, token))
                self.assertEqual(layout.address_from_slot(8, slot, 1),
                                 8 * 15360 + physical * 786432 + token * 208 + 104)

    def test_padding_and_out_of_bounds_have_no_address(self):
        layout = HybridPageLayout(SlotLayout(256), 15360, 786432, 128)
        for operation in (lambda: layout.address_from_slot(8, -1),
                          lambda: layout.address(8, 8, 0),
                          lambda: layout.address(8, 0, layout.block_size),
                          lambda: layout.address(8, 0, 0, 1)):
            with self.assertRaises(ValueError):
                operation()

    def test_insufficient_ssm_capacity_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            HybridPageLayout(SlotLayout(256), 128, 1000, 128)

    def test_native_padded_page_is_charged_without_inventing_fp32_state(self):
        # Native patch_mamba_config.py aligns one K page to SSM and reserves
        # both K/V, so BF16 SSM=393216 can still have total P=801792.
        layout = HybridPageLayout(SlotLayout(256), 15360, 393216, 768,
                                  native_page_size_bytes=801792)
        self.assertEqual(layout.page_size_bytes, 801792)
        self.assertEqual(layout.native_padding_bytes, 393216)
        self.assertEqual(layout.block_size, 2304)
        self.assertEqual(layout.ssm_interval(19, 1), (19 * 15360 + 393216, 19 * 15360 + 2 * 393216))
        budget = MemoryBudget(layout, 100, 17, 17, 128, 64, 256, 3)
        self.assertEqual(budget.raw_pool_bytes, 100 * 17 * 801792)

    def test_physical_pools_not_layer_count_and_all_extras_are_charged(self):
        layout = HybridPageLayout(SlotLayout(256), 15360, 786432, 768)
        budget = MemoryBudget(layout, 100, 17, 17, 128, 64, 256, 3, 1000, 2000, 3000)
        raw = 100 * 17 * 801792
        window = 17 * 128 * 323 * 1024
        self.assertEqual(budget.total_bytes, raw + window + 6000)
        self.assertEqual(budget.blocks_within_budget(budget.total_bytes), 100)
        # One reserved null block + full group blocks + explicit GDN blocks.
        self.assertEqual(budget.blocks_for_contexts((5376, 5377), 2, 3), 1 + 1 + 2 + 12)


class MetadataTests(unittest.TestCase):
    def test_native_buffers_are_reused_without_data_reads(self):
        common = CommonMetadata()
        result = from_common(common)
        self.assertIs(result.seq_lens, common.seq_lens)
        self.assertIs(result.query_start_loc, common.query_start_loc)
        self.assertIs(result.slot_mapping, common.slot_mapping)
        self.assertIs(result.block_tables, common.block_table_tensor)
        self.assertEqual(result.max_query_len, 4)
        self.assertEqual(result.block_tables.shape[1], 2052)

    def test_archive_36_truncated_block_table_capacity_fails(self):
        with self.assertRaisesRegex(OscarMetadataError, "2052/2048"):
            from_common(CommonMetadata(), capacity=MetadataCapacity(129, 130, 16896, 2048))

    def test_mtp_capture_padding_is_counted(self):
        with self.assertRaisesRegex(OscarMetadataError, "129/128"):
            from_common(CommonMetadata(), capacity=MetadataCapacity(128, 130, 16896, 2052))

    def test_dummy_capture_only_marks_origin_and_does_not_mutate_slots(self):
        result = from_common(CommonMetadata(), capture_origin=True)
        self.assertTrue(result.capture_origin)
        self.assertIs(result.slot_mapping, CommonMetadata.slot_mapping)

    def test_noncausal_is_explicitly_rejected(self):
        with self.assertRaisesRegex(OscarMetadataError, "causal"):
            from_common(SimpleNamespace(causal=False))


if __name__ == "__main__":
    unittest.main()
