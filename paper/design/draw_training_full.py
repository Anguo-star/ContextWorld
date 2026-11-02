"""Supplementary task and training controls, using Figure 4's visual template."""
from pathlib import Path
import csv, json
from svg_style import SVG
from training_style import BLUE, ORANGE, plot_frame, marker

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'docs/research/data'
loss=list(csv.DictReader((DATA/'native_training_loss_v1/summary.csv').open()))
nre=list(csv.DictReader((DATA/'native_epoch_checkpoints_v1/summary.csv').open()))
s=SVG(1380,410)
records=[]
for offset,task,title in [(0,'contact_friction','Contact friction'),(700,'motion_damping','Motion damping')]:
    s.text(offset+345,28,title,25,weight=700,anchor='middle')
    s.text(offset+192,66,'Relative prediction loss',22,weight=700,anchor='middle')
    s.text(offset+512,66,'Response error (NRE)',22,weight=700,anchor='middle')
    lx,ly=plot_frame(s,offset+70,96,245,190,.75,1.05,[(.8,'0.8'),(.9,'0.9'),(1,'1.0')])
    nx,ny=plot_frame(s,offset+390,96,245,190,0,1.5,[(0,'0.0'),(.5,'0.5'),(1,'1.0'),(1.5,'1.5')])
    s.line(offset+390,ny(1),offset+635,ny(1),'#626a75',1.5,'6 3')
    for regime in ['joint','frozen']:
        suffix='num32k_v1_warmstart3073_s3072' if regime=='joint' else 'encoder_frozen_v1_s3072'
        run=f'{task}_pldm_{suffix}'
        rows=sorted([r for r in loss if r['run']==run and r['metric']=='fit__pred_loss' and 5<=int(r['checkpoint_epoch'])<=10],key=lambda r:int(r['checkpoint_epoch']))
        epochs=[int(r['checkpoint_epoch']) for r in rows]
        assert epochs==list(range(5,11))
        base=float(rows[0]['mean'])
        means=[float(r['mean'])/base for r in rows]
        low=[float(r['q25'])/base for r in rows];high=[float(r['q75'])/base for r in rows]
        band=list(zip(epochs,low))+list(zip(epochs,high))[::-1]
        points=' '.join(f'{lx(t)},{ly(v)}' for t,v in band)
        s.raw(f'<polygon points="{points}" fill="{BLUE}" fill-opacity="0.08"/>')
        points=' '.join(f'{lx(t)},{ly(v)}' for t,v in zip(epochs,means))
        dash=' stroke-dasharray="7 4"' if regime=='frozen' else ''
        s.raw(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="2.2"{dash}/>')
        for t,v in zip(epochs,means):marker(s,lx(t),ly(v),regime,BLUE,radius=4.5)
        record=dict(task=task,regime=regime,run=run,loss_epochs=epochs,loss_mean=[float(r['mean']) for r in rows],relative_loss=means,relative_q25=low,relative_q75=high,nre={})
        for split in ['loader_fit','development']:
            selected=sorted([r for r in nre if r['task']==task and r['family']=='pldm' and r['regime']==regime and r['split']==split],key=lambda r:int(r['epoch']))
            assert [int(r['epoch']) for r in selected]==[5,10]
            values=[float(r['NRE']) for r in selected]
            record['nre'][split]=dict(epochs=[5,10],values=values)
            for t,v in zip([5,10],values):
                marker(s,nx(t),ny(v),regime,ORANGE,filled=split=='loader_fit',radius=5 if split=='loader_fit' else 8)
        records.append(record)
# Shape/line style denotes training scheme; fill denotes the NRE data split.
y=384;gray='#505963'
s.line(240,y,290,y,gray,2.2);marker(s,265,y,'joint',gray)
s.text(302,y+7,'Joint',20)
s.line(397,y,447,y,gray,2.2,'7 4');marker(s,422,y,'frozen',gray)
s.text(459,y+7,'Frozen Encoder',20)
s.text(710,y+7,'NRE:',20)
marker(s,782,y,'joint',gray);marker(s,799,y,'frozen',gray)
s.text(815,y+7,'Training',20)
marker(s,953,y,'joint',gray,filled=False);marker(s,970,y,'frozen',gray,filled=False)
s.text(986,y+7,'Development',20)
s.save(HERE/'training_full.svg')
(HERE/'training_full.json').write_text(json.dumps({'loss_normalization':'Each scheme divided by its own epoch-5 mean prediction loss','rows':records},indent=2)+'\n')
print('Supplementary training controls saved with shared axes and measured points.')
