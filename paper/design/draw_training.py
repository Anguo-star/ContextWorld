"""Reproduce the reference's single-case loss/NRE panel with measured data."""
from pathlib import Path
import csv,json
from svg_style import SVG
BLUE, ORANGE, DARK_BLUE = "#0078ff", "#ff780a", "#064789"

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
s=SVG(660,340)
s.rect(1,1,658,338,'white','#69717d',1.1,12)
s.text(192,29,'Prediction loss',20,weight=700,anchor='middle')
s.text(512,29,'Response error (NRE)',20,weight=700,anchor='middle')

def plot_frame(x,y,w,h,lo,hi,ticks,label):
    xf=lambda t:x+(t-4.7)/5.6*w
    yf=lambda v:y+h-(v-lo)/(hi-lo)*h
    for t in range(5,11):
        xx=xf(t);s.line(xx,y,xx,y+h,'#e4e5e8',1.1);s.line(xx,y+h,xx,y+h+6,'#171a1d',1.5);s.text(xx,y+h+24,str(t),16,anchor='middle')
    for val,lab in ticks:
        yy=yf(val);s.line(x,yy,x+w,yy,'#e4e5e8',1.1);s.line(x-5,yy,x,yy,'#171a1d',1.5);s.text(x-10,yy+5,lab,15,anchor='end')
    s.line(x,y,x,y+h,'#171a1d',1.5);s.line(x,y+h,x+w,y+h,'#171a1d',1.5)
    s.text(x+w/2,y+h+49,'Epoch',18,anchor='middle')
    s.text(x-51,y+h/2,label,18,anchor='middle',transform=f'rotate(-90 {x-51} {y+h/2})')
    return xf,yf

xf,yf=plot_frame(70,56,245,170,.019,.027,[(.020,'0.020'),(.022,'0.022'),(.024,'0.024'),(.026,'0.026')],'Loss')
s.line(70,yf(ys[0]),315,yf(ys[0]),BLUE,1.5,'6 3')
pts=' '.join(f'{xf(t)},{yf(v)}' for t,v in zip(range(5,11),ys))
s.raw(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
for t,v in zip(range(5,11),ys):s.circle(xf(t),yf(v),5.0,BLUE,DARK_BLUE,.8)
s.arrow(306,yf(ys[0])+2,306,yf(ys[-1])-3,BLUE,2.4,9)
s.text(309,yf(ys[0])-11,f'−{reduction:.2f}%',19,DARK_BLUE,700,'end')
xf,yf=plot_frame(390,56,245,170,0,1.5,[(0,'0.0'),(.5,'0.5'),(1,'1.0'),(1.5,'1.5')],'NRE')
s.line(390,yf(1),635,yf(1),'#626a75',1.5,'6 3')
s.text(512,yf(1)-14,'zero-response reference = 1',14,'#525b69',anchor='middle')
for t,v in zip([5,10],nr):s.circle(xf(t),yf(v),5.8,ORANGE,'white',.6)
s.rect(13,287,634,42,'#e9ebef','none',r=11)
s.text(330,314,'Lower prediction loss does not ensure an accurate conditional response.',15.7,'#414b59',700,'middle')
s.save(HERE/'training_two_panels.svg')
(HERE/'training_two_panels.json').write_text(json.dumps({'task':'contact_friction','family':'pldm','scheme':'Original → Mixed (Joint)','loss_run':run,'loss_epochs':list(range(5,11)),'prediction_loss':ys,'loss_reduction_percent':reduction,'NRE_epochs':[5,10],'training_NRE':nr},indent=2)+'\n')
print('Single-case measured training panel saved')
