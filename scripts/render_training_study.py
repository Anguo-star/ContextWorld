#!/usr/bin/env python3
"""Generate research tables from the published aggregate; --check detects drift."""
import argparse,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
data=json.loads((ROOT/'docs/research/data/icl_training_study_v1.json').read_text());rows=data['results'];assert len(rows)==36
models=[('lewm','scratch'),('lewm','warmstart'),('pldm','scratch'),('pldm','warmstart'),('dinowm','scratch'),('dinowm','projected')]
tasks=[('action_strength','Strength'),('contact_friction','Friction'),('motion_damping','Damping'),('robot_arm_mass','Mass'),('cube_gripper_carry','Cube'),('portal_exit','Portal')]
index={(r['task'],r['model'],r['initialization']):r for r in rows};assert len(index)==36
header='| 任务 | LeWM默认 | LeWM预训练 | PLDM默认 | PLDM预训练 | DINO默认 | DINO转换 |\n|---|---:|---:|---:|---:|---:|---:|'
def table(a,b,precision):
 lines=[header]
 for t,label in tasks:
  vals=[]
  for model,init in models:
   r=index[t,model,init];assert r['pair_count']==256 and sorted(r['cem_seeds'])==list(range(42,48));assert abs(sum(r['cem_seed_success_percent'])/6-r['cem_success_rate_percent'])<1e-8
   vals.append(f"{r[a]:.{precision}f} / {r[b]:.{precision}f}")
  lines.append('| '+label+' | '+' | '.join(vals)+' |')
 return '\n'.join(lines)
ref=['| 任务 | 原权重 ICL / CEM | 较小数据scratch ICL / CEM |','|---|---:|---:|']
for r,(_,label) in zip(data['lewm_historical_reference'],tasks):ref.append(f"| {label} | {r['original_icl_percent']:.2f} / {r['original_cem_percent']:.2f} | {r['small_scratch_icl_percent']:.2f} / {r['small_scratch_cem_percent']:.2f} |")
path=ROOT/'docs/research/Training_Data_and_Initialization.md';old=path.read_text();new=old
for name,value in [('RESULTS',table('icl_accuracy_percent','cem_success_rate_percent',2)),('RESPONSE',table('response_gain','normalized_response_error',3)),('REFERENCE','\n'.join(ref))]:
 pattern=f'<!-- BEGIN GENERATED {name} -->.*?<!-- END GENERATED {name} -->';new,count=re.subn(pattern,f'<!-- BEGIN GENERATED {name} -->\n{value}\n<!-- END GENERATED {name} -->',new,flags=re.S);assert count==1
if args.check:
 if old!=new:raise SystemExit('Research tables differ from JSON; run scripts/render_training_study.py')
 print('36 result rows and generated tables verified.')
else:path.write_text(new)
