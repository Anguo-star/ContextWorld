from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
OUT=Path(__file__).resolve().parent;ROOT=OUT/'assets/pusher_pair'
p=json.loads((ROOT/'provenance.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.1,4.1));fig.subplots_adjust(0,0,1,1)
ax.set_xlim(0,1400);ax.set_ylim(810,0);ax.axis('off')
blue='#1778c8';orange='#ed7922';ink='#22262b';gray='#656e79';edge='#d1d7df'
def box(x,y,w,h,fill='white',stroke=edge,lw=.7,r=8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',linewidth=lw,edgecolor=stroke,facecolor=fill))
def txt(x,y,s,size=7.5,color=ink,ha='left',weight='normal',va='center',**kw):
 ax.text(x,y,s,fontsize=size,color=color,ha=ha,weight=weight,va=va,**kw)
def arrow(x,y,xx,yy,color=gray,lw=.9,head=7):
 ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',color=color,lw=lw,mutation_scale=head))
def section(letter,title,y):
 box(18,y,35,35,ink,ink,r=6);txt(35.5,y+17.5,letter,10,'white',ha='center',weight='bold')
 txt(70,y+17.5,title,10.2,weight='bold')
section('A','Same observation and actions, different futures',15)
txt(1380,34,'PushT · Pusher displacement',6.5,gray,ha='right')
# Common crop for every original frame; preserve recorded image appearance.
imgs={};xs=[];ys=[]
for mode in ['low_gain','high_gain']:
 for step in [0,5,10,15]:
  im=np.array(Image.open(ROOT/p['frames'][mode][str(step)]['file']).convert('RGB'))
  imgs[mode,step]=im
  mask=(im.min(2)<190);mask[:6,:]=False;mask[-6:,:]=False;mask[:,:6]=False;mask[:,-6:]=False
  yy,xx=np.where(mask);xs.extend(xx);ys.extend(yy)
cx=(min(xs)+max(xs))/2;cy=(min(ys)+max(ys))/2
side=max(max(xs)-min(xs),max(ys)-min(ys))+28
x0=max(0,min(224-side,cx-side/2));y0=max(0,min(224-side,cy-side/2))
# Interaction history occupies two frames; arrows denote recorded actions.
cols=[180,415,665,920,1165];w=170
for x,label in [(382,'Interaction history'),(750,'Same current\nobservation'),(1005,'Same future\nactions'),(1250,'Different true\nfutures')]:txt(x,81,label,7.0,ha='center',weight='semibold')
for mode,cyy,c,label,gain in [('low_gain',115,blue,'A',60),('high_gain',330,orange,'B',140)]:
 box(20,cyy+44,142,79,'#edf5fc' if mode=='low_gain' else '#fff2e7',stroke='none',r=7)
 txt(35,cyy+66,f'Condition {label}',7.2,c,weight='semibold')
 txt(35,cyy+91,f'Action gain {gain}',6.5)
 for j,step in [(0,0),(1,5),(2,10),(4,15)]:
  x=cols[j];im=imgs[mode,step]
  ax.imshow(im,extent=(x-x0*w/side,x+(224-x0)*w/side,cyy+(224-y0)*w/side,cyy-y0*w/side),zorder=1,clip_path=None)
  # Explicit crop, without recoloring or altering recorded scene elements.
  clip=Rectangle((x,cyy),w,w,transform=ax.transData)
  ax.images[-1].set_clip_path(clip)
  ax.add_patch(Rectangle((x,cyy),w,w,fill=False,lw=.8,edgecolor=c if j==4 else edge,zorder=3))
  if j==4:
   # Mark the shared starting pusher position, then point to its recorded future.
   start=np.array(p['frames'][mode]['10']['state'][:2])*224/512
   end=np.array(p['frames'][mode]['15']['state'][:2])*224/512
   a=(start-np.array([x0,y0]))*w/side+np.array([x,cyy])
   b=(end-np.array([x0,y0]))*w/side+np.array([x,cyy])
   ax.plot(a[0],a[1],marker='o',ms=3,mec='#777f89',mfc='none',mew=.6,zorder=4)
   arrow(*a,*b,color=c,lw=1.05)
 # Actual five query commands, identical in both rows.
 x=cols[3];box(x,cyy+40,w,90,'#f7f8fa',edge,r=6)
 acts=np.array(p['action_blocks']['query_10_to_15'])
 for j,a in enumerate(acts):
  cc=np.array([x+21+j*32,cyy+83]);d=a/np.max(np.linalg.norm(acts,axis=1))*32
  arrow(*(cc-d/2),*(cc+d/2),color=ink,lw=.9,head=3.5)
 txt(x+w/2,cyy+117,'Action sequence',6.8,gray,ha='center')
 for j in range(4):
  left=cols[j]+w+7;right=cols[j+1]-9
  arrow(left,cyy+w/2,right,cyy+w/2)
  if j<2:txt((left+right)/2,cyy+w/2+25,'actions',5.8,gray,ha='center')
# No repeated per-frame headers; the continuous arrows show time order.
ax.plot([18,1382],[311,311],color=edge,lw=.55,ls=(0,(4,4)))
ax.plot([18,1382],[517,517],color=edge,lw=.7)
section('B','How evaluation works',534)
# Two separate calls to the same frozen model.
for y,c,name in [(620,blue,'A'),(680,orange,'B')]:
 box(29,y,145,39,'#edf5fc' if name=='A' else '#fff2e7',stroke=c,lw=.7,r=5)
 txt(101.5,y+19.5,f'History {name}',7.5,c,ha='center',weight='semibold')
 arrow(178,y+19.5,238,y+19.5,color=c)
box(237,574,225,25,'#f7f8fa',stroke='none',r=3)
txt(349.5,586.5,'Same observation + actions',5.8,gray,ha='center')
arrow(349.5,600,349.5,610)
box(242,611,215,126,'#f3f5f7','#aeb7c2',r=9)
txt(349.5,638,'Latent\nworld model',7.8,ha='center',weight='semibold')
txt(349.5,677,'Fixed weights',7.5,ha='center')
txt(349.5,716,'Two separate calls',6.6,gray,ha='center')
for y,c,name in [(639.5,blue,'A'),(699.5,orange,'B')]:
 arrow(461,y,515,y,color=c)
 txt(549,y,rf'$\hat{{z}}_{name}$',12,c,ha='center')
 arrow(580,y,635,y,color=c)
# Error matrix: row = prediction; column = actual future.
mx=644;my=610;cw=118;ch=60
for j,(lab,c) in enumerate([('A',blue),('B',orange)]):txt(mx+(j+.5)*cw,my-21,f'True\nfuture {lab}',6.8,c,ha='center',weight='semibold')
for i in range(2):
 for j in range(2):
  fill=('#edf5fc' if i==0 else '#fff2e7') if i==j else '#f7f8fa'
  ax.add_patch(Rectangle((mx+j*cw,my+i*ch),cw,ch,facecolor=fill,edgecolor='white',lw=2))
  txt(mx+(j+.5)*cw,my+(i+.5)*ch,rf'$d_{{{"AB"[i]}{"AB"[j]}}}$',12,ha='center')
txt(mx+cw,754,'Latent MSE · lower is better',6.1,gray,ha='center')
# Comparison directions are explicit and correct.
arrow(922,637,983,637,ink,1.1)
txt(1002,630,'Correct future?',8,weight='semibold')
txt(1002,652,r'Across a row: $d_{AA}<d_{AB}$',6.7,gray)
arrow(953,678,953,720,ink,1.1)
txt(1002,691,'Helpful history?',8,weight='semibold')
txt(1002,713,r'Down a column: $d_{AA}<d_{BA}$',6.7,gray)
# Coverage is secondary, not another panel of small images.
ax.plot([18,1382],[780,780],color=edge,lw=.7)
txt(22,798,'ContextWorld',8.5,weight='bold')
txt(225,798,'9 tasks · 4 environments',7.5)
txt(1378,798,'TwoRoom  ·  PushT  ·  Reacher  ·  Cube',7.2,gray,ha='right')
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'benchmark_overview.{ext}',dpi=220,bbox_inches='tight',pad_inches=.025)
(OUT/'overview_crop.json').write_text(json.dumps({'crop_xywh':[float(v) for v in [x0,y0,side,side]],'pair_id':p['pair_id']},indent=2))
print('Saved real-data overview design')
