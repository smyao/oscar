"""Archive #129/#132-135/#145/#148: source identity and honest work accounting.

Synthetic tables below test the accounting helper only; they are never written
as runtime inputs or counted as real AscendC prepare/device evidence.
"""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import generate,fixture
from csrc.experiments.c4_phase_profile import markers,compare
from csrc.experiments.window_batch_decode import replay_camodel as decode_replay
import copy
import struct
from unittest.mock import patch


class ContractTests(unittest.TestCase):
    def test_both_clones_restore_every_original_byte(self):
        for profile in (False,True):
            text,evidence,fields=generate.generate(profile)
            self.assertTrue(evidence['inverse_source_bytes_identical'])
            self.assertEqual(evidence['new_completion_fences'],0)
            self.assertEqual(len(fields),32)
            self.assertEqual(fields[24:26],['state_spill','state_restore'])
            self.assertEqual(fields[31],'finite_reduce')
            self.assertIn('kHalfRows=32, kHalfKv=16',text)
            self.assertNotIn('attention_striped_unpack_padded.h',text)
            self.assertEqual(text,(generate.HERE/(evidence['arm']+'.cpp')).read_text())

    def table(self,n=fixture.N):
        rows=[]
        for token in range(n):
            count=21 if token%21==0 else 0
            for kind in range(3):
                begin=end=0
                if count:
                    if kind==0:begin,end=64,min(769,max(64,769+token+count-256))
                    elif kind==1:end=64+769-min(769,max(64,769+token+1-256))
                    else:begin=end=769
                rows.append([token,count,0,begin,end,0,kind,kind,769,0,0,0,0,0,0,0])
        return rows

    def test_all_frontier_window_and_empty_current_tasks_remain(self):
        value=fixture.task_ledger(self.table())
        self.assertEqual(value['task_slots'],4032)
        self.assertEqual(value['leaders'],192)
        self.assertEqual(value['followers'],3840)
        self.assertEqual(value['cluster_count'],13)
        self.assertEqual(value['cluster_members'],52)
        self.assertEqual(value['kv256_units'],{'history':72,'window':67,'current':0})
        self.assertEqual(len(value['owners']),20)
        self.assertEqual(sum(owner['c1_tasks'] for owner in value['owners']),140)
        self.assertEqual(value['mature_clusters'][0],[252,64,769] if isinstance(value['mature_clusters'][0],list) else (252,64,769))

    def test_filtered_or_unsuppressed_tables_fail(self):
        with self.assertRaises(ValueError):fixture.task_ledger(self.table()[3:])
        rows=self.table();rows[2][4]=790
        with self.assertRaisesRegex(ValueError,'not suppressed'):fixture.task_ledger(rows)

    def test_declared_q336_keeps_frontier_cluster_and_all_sources(self):
        value=fixture.task_ledger(self.table(336))
        self.assertEqual((value['task_slots'],value['leaders'],value['followers']),(1008,48,960))
        self.assertEqual((value['cluster_count'],value['cluster_members']),(1,4))
        self.assertEqual(value['kv256_units'],{'history':36,'window':19,'current':0})
        self.assertEqual(sum(owner['c1_tasks'] for owner in value['owners']),44)
        self.assertEqual(value['owners'][16]['window'],2)  # source1 uses 16 query tiles, not the old 64
        data=b''.join(struct.pack('<16q',*row) for row in self.table(336))
        stats=markers.expected_cluster_stats(data)
        self.assertEqual(sum(row[3] for row in stats),3)
        self.assertEqual(sum(row[5] for row in stats),33)
        p=fixture.plan(336)
        self.assertEqual(p['case'],'c4_q336_r1_c769_s1')
        self.assertEqual(p['predeclared_arm_deadline_seconds'],900)
        self.assertFalse(p['prior_q1344_failure']['ticks_usable'])
        self.assertIn('low-occupancy',p['measurement_scope'])
        for invalid in (84,335,337,2048,True):
            with self.assertRaises(ValueError):fixture.plan(invalid)

    def test_plan_cannot_be_reported_as_device_execution(self):
        p=fixture.plan()
        self.assertEqual(p['predeclared_arm_deadline_seconds'],900)
        self.assertEqual(p['validation']['normal_camodel'],'not_run')
        self.assertEqual(p['mature_history_kv256_units'],3)
        self.assertIn('native current FIA',p['not_measured'])

    def test_schema3_accepts_state_markers_without_changing_decode_parser(self):
        for field in (24,25,31):
            tag=0x5a77c01d00000000|(field<<1)
            text=f'[info] [100] MOV_XD_XN XD:X0=0x{tag:x}'
            self.assertEqual(markers.marker_events([text])[0]['field'],field)
            with self.assertRaises(ValueError):decode_replay.marker_events([text])
        with self.assertRaises(ValueError):markers.decoded_profile(b'\0'*(20*2304))
        parsed=markers.decoded_profile(b'\0'*(20*3072))
        self.assertFalse(parsed['timing_usable'])

    def test_finite_reduce_counts_real_heads_and_visibility(self):
        totals=[[0,0,0],[0,0,0]]
        for row in self.table():
            for lane in range(2):totals[lane][row[7]]+=markers.finite_reduce_count(row,lane)
        self.assertEqual(totals,[[12096,4288,0],[11718,4154,0]])
        row=self.table()[0].copy();row[1]=1;row[3]=64;row[4]=69;row[8]=10
        self.assertEqual(markers.finite_reduce_count(row,0),0)  # no visible history
        row[7]=1;row[0]=255;row[8]=769;row[3]=64;row[4]=65
        self.assertEqual(markers.finite_reduce_count(row,0),0)  # sink excluded, tail empty
        row[3]=0;row[4]=64
        self.assertEqual(markers.finite_reduce_count(row,0),6)
        self.assertEqual(markers.finite_reduce_count(row,1),0)  # padded lane

    def test_cluster_counts_keep_distinct_two_pass_ownership(self):
        data=b''.join(struct.pack('<16q',*row) for row in self.table())
        values=markers.expected_cluster_stats(data)
        self.assertEqual(sum(row[0] for row in values),13)
        self.assertEqual(sum(row[1] for row in values),52)
        self.assertEqual(sum(row[2] for row in values),12)
        self.assertEqual(sum(row[3] for row in values),39)
        self.assertEqual(sum(row[5] for row in values),33)
        self.assertEqual(sum(row[6] for row in values),52)
        self.assertTrue(any(row[1]!=row[6] for row in values))

    def test_paired_gate_rejects_changed_output_and_negative_large_perturbation(self):
        base={field:'same' for field in ('case','cores','attrs','scale','source_sha256','dependency_sha256',
            'library_sha256','input_sha256','oracle_sha256','acceptance_sha256','task_ledger',
            'simulator_configuration','workspace_bytes_per_core','oracle_scope','source2_suppressed')}
        base.update(task_ticks=100000,outputs={name:{'sha256':name} for name in ('partial','lse','status','cluster_stats')})
        other=copy.deepcopy(base);other['task_ticks']=104000
        with patch.object(compare,'checked_arm',side_effect=[(base,{}),(other,{})]):
            self.assertEqual(compare.compare(None,None,None,None)['perturbation_gate'],'passed')
        other['task_ticks']=94000
        with patch.object(compare,'checked_arm',side_effect=[(base,{}),(other,{})]):
            self.assertEqual(compare.compare(None,None,None,None)['perturbation_gate'],'failed')
        other['outputs']['cluster_stats']['sha256']='changed'
        with patch.object(compare,'checked_arm',side_effect=[(base,{}),(other,{})]),self.assertRaisesRegex(ValueError,'bitwise'):
            compare.compare(None,None,None,None)


if __name__=='__main__':unittest.main()
