"""Focused checks for the published cross-task mechanism summary."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/analyze_cross_task_mechanism.py'
spec=importlib.util.spec_from_file_location('cross_task_mechanism_summary',SCRIPT)
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

def test_marker_render_detects_stale_table():
 document='before\n<!-- BEGIN CROSS_TASK_MECHANISM -->\nold\n<!-- END CROSS_TASK_MECHANISM -->\nafter\n'
 rendered=summary.marked_document(document,'new table\n')
 assert rendered!=document
 assert summary.marked_document(rendered,'new table\n')==rendered
 assert 'before\n' in rendered and '\nafter\n' in rendered

def test_public_source_hash_failure(monkeypatch):
 original=summary.digest
 target=summary.RAW/'readout_results/action_strength/lewm/scratch/s3072.json'
 def corrupt(path):
  return '0'*64 if Path(path)==target else original(path)
 monkeypatch.setattr(summary,'digest',corrupt)
 with pytest.raises(ValueError,match='result SHA mismatch'):
  summary.build()

def test_main_table_uses_full_development_nre():
 outputs=summary.build()
 rows=summary.load(summary.OUT/'cross_task_mechanism_v1.json')['rows']
 row=next(x for x in rows if x['id']=='motion_damping/pldm/scratch/s3072')
 full=row['original_protocol']['full_development_response_nre']
 probe=row['gradient']['probe_16_query_response_nre']
 assert full>1.5 and probe<0.9
 table=outputs[summary.RAW/'table.md'].decode()
 assert f'{full:.3f}' in table
 assert '| 运动阻尼 | PLDM |' in table
