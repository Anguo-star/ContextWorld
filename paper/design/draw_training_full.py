"""Supplementary task and training controls, using Figure 4's visual template."""
from pathlib import Path
import csv, json
from svg_style import SVG
from training_style import BLUE, ORANGE, DARK_BLUE, DARK_ORANGE, plot_frame, marker, nre_frame, nre_points

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'docs/research/data'
loss=list(csv.DictReader((DATA/'native_training_loss_v1/summary.csv').open()))
nre=list(csv.DictReader((DATA/'native_epoch_checkpoints_v1/summary.csv').open()))
test=json.loads((DATA/'figure5_checkpoint_test_v1/summary.json').read_text())
assert all(r['split']=='test' for r in test['rows'])
s=SVG(1380,410)
records=[]
for offset,task,title in [(0,'contact_friction','Contact friction'),(700,'motion_damping','Motion damping')]:
    s.text(offset+345,28,title,25,weight=700,anchor='middle')
    s.text(offset+192,66,'Relative prediction loss',22,weight=700,anchor='middle')
    s.text(offset+512,66,'Response error (NRE)',22,weight=700,anchor='middle')
    lx,ly=plot_frame(s,offset+70,96,245,190,.75,1.05,[(.8,'0.8'),(.9,'0.9'),(1,'1.0')])
    nx,ny=nre_frame(s,offset+390,96)
    for regime in ['joint','frozen']:
        color=BLUE
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
        s.raw(f'<polygon points="{points}" fill="{color}" fill-opacity="0.08"/>')
        points=' '.join(f'{lx(t)},{ly(v)}' for t,v in zip(epochs,means))
        dash=' stroke-dasharray="7 4"' if regime=='frozen' else ''
        s.raw(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        for t,v in zip(epochs,means):marker(s,lx(t),ly(v),regime,color,radius=4.5)
        record=dict(task=task,regime=regime,run=run,loss_epochs=epochs,loss_mean=[float(r['mean']) for r in rows],relative_loss=means,relative_q25=low,relative_q75=high,nre={})
        for split in ['training','test']:
            if split=='training':
                selected=sorted([r for r in nre if r['task']==task and r['family']=='pldm' and r['regime']==regime and r['split']=='loader_fit'],key=lambda r:int(r['epoch']))
            else:
                selected=sorted([r for r in test['rows'] if r['task']==task and r['regime']==regime],key=lambda r:int(r['epoch']))
            assert [int(r['epoch']) for r in selected]==[5,10]
            values=[float(r['NRE']) for r in selected]
            record['nre'][split]=dict(epochs=[5,10],values=values,
                                     checkpoint_sha256=[r['checkpoint_sha256'] for r in selected])
            nre_points(s,nx,ny,values,regime,split,offset=-.20 if split=='training' else .20)
        if task=='contact_friction' and regime=='joint':
            reduction=100*(1-means[-1])
            s.text(offset+304,ly(means[-1])+28,f'Joint: −{reduction:.2f}%',18,DARK_BLUE,700,'end')
            s.text(offset+512,ny(1)+32,'NRE ≈ 1 (Joint)',20,DARK_ORANGE,700,'middle')
        records.append(record)
# Each NRE legend entry identifies both scheme and split directly.
y=384
s.text(55,y+7,'NRE:',20)
for x,regime,split,label in [
        (130,'joint','training','Joint · Training'),
        (390,'joint','test','Joint · Test'),
        (630,'frozen','training','Frozen Encoder · Training'),
        (1000,'frozen','test','Frozen Encoder · Test')]:
    color=ORANGE
    marker(s,x+17.5,y,regime,color,filled=split=='training')
    s.text(x+47,y+7,label,20)
s.save(HERE/'training_full.svg')
(HERE/'training_full.json').write_text(json.dumps({'loss_normalization':'Each scheme divided by its own epoch-5 mean prediction loss','test_source':'docs/research/data/figure5_checkpoint_test_v1/summary.json','NRE_horizontal_offset_epochs':{'training':-.20,'test':.20},'rows':records},indent=2)+'\n')
print('Supplementary training controls saved with shared axes and measured points.')
