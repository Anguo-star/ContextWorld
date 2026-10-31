from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
loss=list(csv.DictReader((ROOT/'docs/research/data/native_training_loss_v1/summary.csv').open()))
nre=list(csv.DictReader((ROOT/'docs/research/data/native_epoch_checkpoints_v1/summary.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(3.45,2.15),gridspec_kw={'wspace':.40})
fig.subplots_adjust(left=.12,right=.98,bottom=.34,top=.84)
colors={'contact_friction':'#1778c8','motion_damping':'#ed7922'}
handles=[];records=[]
for task,title in [('contact_friction','Friction'),('motion_damping','Damping')]:
 for regime,label,style,marker in [('joint','Joint','-','o'),('frozen','Frozen encoder','--','s')]:
  suffix='num32k_v1_warmstart3073_s3072' if regime=='joint' else 'encoder_frozen_v1_s3072'
  run=f'{task}_pldm_{suffix}'
  rows=sorted([r for r in loss if r['run']==run and r['metric']=='fit__pred_loss' and 5<=int(r['checkpoint_epoch'])<=10],key=lambda r:int(r['checkpoint_epoch']))
  assert len(rows)==6
  base=float(rows[0]['mean']); xs=[int(r['checkpoint_epoch']) for r in rows]; ys=[float(r['mean'])/base for r in rows]
  c=colors[task];face=c if regime=='joint' else 'white'
  axs[0].plot(xs,ys,color=c,ls=style,marker=marker,markersize=3.5,markerfacecolor=face,lw=1.5)
  axs[0].fill_between(xs,[float(r['q25'])/base for r in rows],[float(r['q75'])/base for r in rows],color=c,alpha=.07,lw=0)
  nr=sorted([r for r in nre if r['task']==task and r['family']=='pldm' and r['regime']==regime and r['split']=='loader_fit'],key=lambda r:int(r['epoch']))
  assert [int(r['epoch']) for r in nr]==[5,10]
  ny=[float(r['NRE']) for r in nr]
  axs[1].plot([5,10],ny,color=c,ls='none',marker=marker,markerfacecolor=face,markersize=4.7,markeredgewidth=1)
  handles.append(Line2D([],[],color=c,ls=style,marker=marker,markerfacecolor=face,markersize=3.8,lw=1.3,label=f'{title} · {label}'))
  records.append({'task':task,'regime':regime,'loss_epochs':xs,'relative_loss':ys,'NRE_epochs':[5,10],'training_NRE':ny})
for ax in axs:
 ax.set_xlim(4.7,10.3);ax.set_xticks([5,6,7,8,9,10]);ax.set_xlabel('Epoch',labelpad=3)
 ax.grid(color='#e7e9ec',lw=.6);ax.set_axisbelow(True);ax.tick_params(length=2.7,pad=2)
 for sp in ax.spines.values():sp.set_linewidth(.7)
axs[0].set_title('Prediction loss',fontweight='semibold',pad=7)
axs[0].set_ylim(.78,1.04);axs[0].set_yticks([.8,.9,1]);axs[0].set_ylabel('Relative loss',labelpad=4)
axs[1].set_title('Response error (NRE)',fontweight='semibold',pad=7)
axs[1].set_ylim(0,1.18);axs[1].set_yticks([0,.5,1]);axs[1].set_ylabel('NRE',labelpad=3)
axs[1].axhline(1,color='#737982',ls='--',lw=.85,zorder=1)
axs[1].text(7.5,1.075,'Zero-response reference',fontsize=5.9,ha='center',color='#555b65')
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.02),ncol=2,frameon=False,fontsize=6.2,handlelength=1.8,columnspacing=1.6,labelspacing=.65)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'training_two_panels.{ext}',dpi=230,bbox_inches='tight',pad_inches=.035)
(OUT/'training_two_panels.json').write_text(json.dumps(records,indent=2))
print('Saved two-panel scientific plot draft')
