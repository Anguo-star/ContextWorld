"""Reproduce the reference's single-case loss/NRE panel with measured data."""
from pathlib import Path
import csv,json
from svg_style import SVG
from training_style import BLUE, DARK_BLUE, plot_frame, nre_frame, nre_points

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
rows=list(csv.DictReader((ROOT/'docs/research/data/native_training_loss_v1/summary.csv').open()))
run='contact_friction_pldm_num32k_v1_warmstart3073_s3072'
loss=sorted([r for r in rows if r['run']==run and r['metric']=='fit__pred_loss' and 5<=int(r['checkpoint_epoch'])<=10],key=lambda r:int(r['checkpoint_epoch']))
assert [int(r['checkpoint_epoch']) for r in loss]==list(range(5,11))
rows=list(csv.DictReader((ROOT/'docs/research/data/native_epoch_checkpoints_v1/summary.csv').open()))
nre=sorted([r for r in rows if r['task']=='contact_friction' and r['family']=='pldm' and r['regime']=='joint' and r['split']=='loader_fit'],key=lambda r:int(r['epoch']))
assert [int(r['epoch']) for r in nre]==[5,10]
ys=[float(r['mean']) for r in loss];nr=[float(r['NRE']) for r in nre]
reduction=100*(1-ys[-1]/ys[0])
s=SVG(660,324)
s.text(192,26,'Prediction loss',22,weight=700,anchor='middle')
s.text(192,53,f'−{reduction:.2f}%',21,DARK_BLUE,700,'middle')
s.text(512,26,'Training-query NRE',22,weight=700,anchor='middle')
s.text(512,53,'≈ 1',21,DARK_BLUE,700,'middle')


xf,yf=plot_frame(s,70,80,245,172,.019,.027,[(.020,'0.020'),(.022,'0.022'),(.024,'0.024'),(.026,'0.026')])
s.line(70,yf(ys[0]),315,yf(ys[0]),BLUE,1.5,'6 3')
pts=' '.join(f'{xf(t)},{yf(v)}' for t,v in zip(range(5,11),ys))
s.raw(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
for t,v in zip(range(5,11),ys):s.circle(xf(t),yf(v),5.0,BLUE,DARK_BLUE,.8)
s.arrow(306,yf(ys[0])+2,306,yf(ys[-1])-3,BLUE,2.4,9)
xf,yf=nre_frame(s,390,80,h=172)
nre_points(s,xf,yf,nr)
s.save(HERE/'training_two_panels.svg')
(HERE/'training_two_panels.json').write_text(json.dumps({'task':'contact_friction','family':'pldm','scheme':'Original → Mixed (Joint)','loss_run':run,'loss_epochs':list(range(5,11)),'prediction_loss':ys,'loss_reduction_percent':reduction,'NRE_epochs':[5,10],'training_NRE':nr,'training_accuracy_percent':[float(r['main_percent']) for r in nre]},indent=2)+'\n')
print('Single-case measured training panel saved')
