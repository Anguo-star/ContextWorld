"""Regenerate manuscript figures from the checked-in experimental records."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})


def training_evidence():
    with (ROOT / 'docs/research/data/native_training_loss_v1/summary.csv').open() as f:
        loss = list(csv.DictReader(f))
    with (ROOT / 'docs/research/data/native_epoch_checkpoints_v1/summary.csv').open() as f:
        nre = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.05), sharex='col',
                              gridspec_kw={'hspace': .38, 'wspace': .30})
    fig.subplots_adjust(left=.16,right=.985,bottom=.25,top=.83)
    colors = {'joint': '#2765a8', 'frozen': '#c46124'}
    for col, (task, title) in enumerate([('contact_friction', 'Contact friction'),
                                        ('motion_damping', 'Motion damping')]):
        loss_ax, nre_ax = axes[:, col]
        for regime in ['joint', 'frozen']:
            suffix = 'num32k_v1_warmstart3073_s3072' if regime == 'joint' else 'encoder_frozen_v1_s3072'
            run = f'{task}_pldm_{suffix}'
            values = sorted([r for r in loss if r['run'] == run and
                             r['metric'] == 'fit__pred_loss' and
                             5 <= int(r['checkpoint_epoch']) <= 10],
                            key=lambda r: int(r['checkpoint_epoch']))
            assert [int(r['checkpoint_epoch']) for r in values] == list(range(5, 11))
            base = float(values[0]['mean'])
            xs = list(range(5, 11))
            loss_ax.plot(xs, [float(r['mean'])/base for r in values],
                         color=colors[regime], marker='o', markersize=2.3, linewidth=1.2)
            loss_ax.fill_between(xs, [float(r['q25'])/base for r in values],
                                 [float(r['q75'])/base for r in values],
                                 color=colors[regime], alpha=.12, linewidth=0)
            for split, marker in [('loader_fit', 'o'), ('development', 's')]:
                rows = sorted([r for r in nre if r['task'] == task and r['family'] == 'pldm'
                               and r['regime'] == regime and r['split'] == split],
                              key=lambda r: int(r['epoch']))
                assert [int(r['epoch']) for r in rows] == [5, 10]
                nre_ax.plot([5, 10], [float(r['NRE']) for r in rows],
                            color=colors[regime], linestyle='none', marker=marker,
                            markersize=3.3 if split=='loader_fit' else 4.8,
                            markeredgewidth=.8,
                            markerfacecolor=colors[regime] if split=='loader_fit' else 'none')
            if task=='contact_friction' and regime=='joint':
                reduction=100*(1-float(values[-1]['mean'])/base)
                loss_ax.text(5.2,.78,f'Joint −{reduction:.2f}%',fontsize=6.3,color=colors[regime])
                train=[r for r in nre if r['task']==task and r['family']=='pldm'
                       and r['regime']==regime and r['split']=='loader_fit']
                assert all(round(float(r['NRE']),2)==1.01 for r in train)
                nre_ax.text(7.5,.84,'Train ≈ 1.01',color=colors[regime],fontsize=6.4,ha='center')
        loss_ax.set_title(title,fontsize=8,pad=6)
        loss_ax.set_ylim(.75,1.05);loss_ax.set_yticks([.8,.9,1.0])
        nre_ax.set_ylim(0,1.13);nre_ax.set_yticks([0,.5,1.0])
        nre_ax.axhline(1,color='#707780',linewidth=.8,linestyle=':',zorder=0)
        nre_ax.set_xticks([5,10])
        for ax in [loss_ax,nre_ax]:
            ax.set_xlim(4.65,10.35)
            ax.grid(axis='y',color='#e8ebef',linewidth=.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7,pad=2,length=2.5)
            for spine in ax.spines.values():spine.set_linewidth(.6)
    axes[0,0].set_ylabel('Relative loss',fontsize=7.5,labelpad=3)
    axes[1,0].set_ylabel('NRE',fontsize=7.5,labelpad=3)
    fig.text(.56,.145,'Training epoch',ha='center',fontsize=7.5)
    methods=[Line2D([],[],color=colors['joint'],label='Joint'),
             Line2D([],[],color=colors['frozen'],label='Frozen Encoder')]
    splits=[Line2D([],[],color='.3',marker='o',linestyle='none',markersize=3.3,label='Training'),
            Line2D([],[],color='.3',marker='s',linestyle='none',markerfacecolor='none',markersize=4.8,label='Development')]
    fig.legend(handles=methods,loc='upper center',bbox_to_anchor=(.56,1.01),
               frameon=False,ncol=2,fontsize=7,handlelength=1.6,columnspacing=1.2)
    fig.legend(handles=splits,loc='lower center',bbox_to_anchor=(.56,-.03),
               title='NRE data split',title_fontsize=7,frameon=False,ncol=2,fontsize=7,
               handlelength=1,columnspacing=1.4)
    fig.savefig(OUT/'training_evidence.pdf',pad_inches=.035)
    plt.close(fig)


def protocol_overview():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.1, 1.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    def box(x, y, w, h, text, color='#f5f6f8', size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',
                                    linewidth=.8,edgecolor='#687384',facecolor=color))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(a,b,color='#596676',style='-'):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color=color,linestyle=style))
    ax.text(1.6,2.82,'Paired interaction histories',ha='center',weight='bold')
    box(.15,1.7,2.6,.55,'History A: dynamics A', '#eaf2fb')
    box(.15,.75,2.6,.55,'History B: dynamics B', '#fff0e4')
    box(3.3,2.05,2.65,.55,'Same current observation\nand future actions',size=7.5)
    box(3.55,.85,2.15,.7,'Same world model\nSeparate calls; fixed weights')
    arrow((4.62,2.04),(4.62,1.62))
    arrow((2.82,1.98),(3.48,1.38),'#2765a8')
    arrow((2.82,1.02),(3.48,1.08),'#c46124')
    box(6.35,1.65,3.35,.55,'Compare prediction A with true future A','#eaf2fb',7.5)
    box(6.35,.7,3.35,.55,'Compare prediction B with true future B','#fff0e4',7.5)
    arrow((5.77,1.38),(6.28,1.92),'#2765a8')
    arrow((5.77,1.08),(6.28,.98),'#c46124')
    ax.text(8.02,2.82,'Score predictions and history benefit',ha='center',weight='bold')
    ax.text(4.98,.13,'History control: swap A and B at the input; keep each scoring target fixed.',
            ha='center',fontsize=8)
    fig.tight_layout(pad=.2)
    fig.savefig(OUT/'protocol_overview.pdf')
    plt.close(fig)



def multistep_prediction():
    """Per-horizon ratios; average normalized runs, never average scene ratios."""
    import gzip
    import json
    import numpy as np
    data = ROOT / 'docs/research/data'
    summary = json.loads((data / 'multistep_prediction_coverage_v2.json').read_text())
    with gzip.open(data / 'multistep_prediction_coverage_v2_queries.json.gz', 'rt') as f:
        queries = json.load(f)
    horizons = np.array(summary['protocol']['horizons_raw_steps'])
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True)
    records = []
    colors = {'original': '#777777', 'frozen': '#c46124'}
    for ax, family, title in zip(axes, ['lewm', 'pldm'], ['LeWM', 'PLDM']):
        ax.axhspan(1,3.35,color='#f3f4f6',zorder=0)
        for regime in ['original', 'frozen']:
            row, = [r for r in summary['rows'] if r['task'] == 'action_strength'
                    and r['family'] == family and r['regime'] == regime]
            runs = [queries[k] for k in row['run_ids']]
            ids = [q['query_id'] for q in runs[0]]
            assert all([q['query_id'] for q in r] == ids for r in runs)
            clusters = [q.get('bootstrap_cluster', q['query_id']) for q in runs[0]]
            assert all([q.get('bootstrap_cluster', q['query_id']) for q in r] == clusters for r in runs)
            groups = [[i for i,c in enumerate(clusters) if c == key] for key in sorted(set(clusters))]
            rng = np.random.default_rng(summary['protocol']['bootstrap']['seed'])
            ix = rng.integers(0, len(groups), (4000, len(groups)))
            values, draws, overall = [], [], []
            for run in runs:
                n = np.array([q['horizon_numerators'] for q in run])
                b = np.array([q['horizon_denominators'] for q in run])
                assert n.shape == b.shape == (len(ids), 5)
                assert np.all(b.sum(0) > 0)
                values.append(n.sum(0) / b.sum(0))
                overall.append(n.sum() / b.sum())
                nc = np.stack([n[g].sum(0) for g in groups])
                bc = np.stack([b[g].sum(0) for g in groups])
                den = bc[ix].sum(1)
                assert np.all(den > 0)
                draws.append(nc[ix].sum(1) / den)
            mean = np.mean(values, 0)
            lo, hi = np.quantile(np.mean(draws, 0), [.025, .975], axis=0)
            assert np.isclose(np.mean(overall), 1-row['score']/100, rtol=1e-9)
            ax.plot(horizons, mean, color=colors[regime], marker='o' if regime=='frozen' else 's',
                    markersize=3, linewidth=1.2)
            ax.fill_between(horizons, lo, hi, color=colors[regime], alpha=.14, linewidth=0)
            records.append(dict(family=family, regime=regime, run_ids=row['run_ids'],
                                horizons=horizons.tolist(), per_horizon_ratio=mean.tolist(),
                                ci95=np.stack([lo,hi],axis=1).tolist(), aggregate_ratio=float(np.mean(overall))))
        ax.axhline(1, color='#424952', linestyle='--', linewidth=1.15)
        ax.text(.03,.96,'Higher error than\nmean-future reference',transform=ax.transAxes,
                va='top',fontsize=6.2,color='#4c535c')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(horizons)
        ax.set_ylim(0, 3.35)
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Simulator steps', fontsize=7.5)
    axes[0].set_ylabel(r'Prediction error $E_t/B_t$ $\downarrow$', fontsize=7.5)
    handles = [Line2D([],[],color=colors['original'],marker='s',markersize=3,label='Original'),
               Line2D([],[],color=colors['frozen'],marker='o',markersize=3,label='Original → Mixed (Frozen Encoder)'),
               Line2D([],[],color='#424952',linestyle='--',linewidth=1.15,label='Mean-future reference')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=7, ncol=1,
               bbox_to_anchor=(.5,-.015), handlelength=1.5, labelspacing=.25)
    fig.tight_layout(rect=(0,.22,1,1),pad=.3,w_pad=.5)
    fig.savefig(OUT/'multistep_prediction.pdf')
    plt.close(fig)
    return dict(split='development', task='action_strength',
                metric='sum of prediction energies / sum of mean-future energies at each horizon; equal mean over training runs',
                interval='95% source-cluster bootstrap; 4000 resamples; conditional on checkpoints', rows=records)


def scaling_comparisons():
    """Paired endpoint scores; both panels use the same accuracy scale."""
    import json
    data = ROOT/'docs/research/data'
    study = json.loads((data/'icl_training_study_v2.json').read_text())
    capacity = json.loads((data/'model_capacity_scaling_v1/final_comparison.json').read_text())
    data_records = []
    for regime in ['scratch', 'joint']:
        rows = [r for r in study['rows'] if r['task']=='action_strength' and r['model']=='lewm'
                and r['regime']==regime and r.get('training_pair_count')
                and (not r['comparison_variant'] or r['comparison_variant'].startswith('scale_'))]
        rows.sort(key=lambda r:r['training_pair_count'])
        assert len(rows)==len({r['training_pair_count'] for r in rows})
        for r in rows:
            st=r['statistics']['scores']['main']
            data_records.append(dict(regime=regime,pairs=r['training_pair_count'],
                                     accuracy=st['mean'],n=st['n'],sd=st.get('std')))
    capacity_records=[]
    for r in capacity['rows']:
        small=r['small']['main_percent'];large=r['scaling10x']['main_percent']
        capacity_records.append(dict(task=r['task'],family=r['model'],small=small,
                                     large=large,delta=large-small))
    fig = plt.figure(figsize=(3.45, 3.25))
    axes = [fig.add_axes([.15, .59, .81, .27]),
            fig.add_axes([.15, .13, .81, .27])]
    before_data, after_data = [], []
    for regime in ['scratch', 'joint']:
        small, = [r for r in data_records if r['regime']==regime and r['pairs']==2048]
        large, = [r for r in data_records if r['regime']==regime and r['pairs']==32768]
        before_data.append(small['accuracy']); after_data.append(large['accuracy'])
    def bars(ax, before, after, labels):
        import numpy as np
        x = np.arange(len(before))
        ax.bar(x-.16, before, .27, color='#b9c1c9', label='Smaller scale', zorder=3)
        ax.bar(x+.16, after, .27, color='#426d99', label='Larger scale', zorder=3)
        ax.set_ylim(0,110); ax.set_yticks([0,50,100])
        ax.axhline(50,color='#525c68',linestyle='--',linewidth=.85,zorder=4)
        ax.set_ylabel('Accuracy (%)',fontsize=7.5,labelpad=3)
        ax.set_xticks(x, labels, fontsize=7)
        ax.tick_params(length=2.5,labelsize=7)
        ax.grid(axis='y',color='#e8ebef',linewidth=.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        for i,(small,large) in enumerate(zip(before,after)):
            ax.text(i,max(small,large)+4,f'Δ {large-small:+.2f}'.replace('-', '−'),
                    ha='center',fontsize=6.6)
    bars(axes[0],before_data,after_data,['Mixed','Original → Mixed\n(Joint)'])
    labels=[('LeWM' if r['family']=='lewm' else 'PLDM')+'\n'+
            ('friction' if r['task']=='contact_friction' else 'damping') for r in capacity_records]
    bars(axes[1],[r['small'] for r in capacity_records],
         [r['large'] for r in capacity_records],labels)
    fig.text(.15,.99,'(a) Data: 2,048 → 32,768 pairs',fontsize=8,va='top')
    fig.text(.15,.48,'(b) Models: 18M → 183M',fontsize=8,va='top')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',
               bbox_to_anchor=(.13,.94),ncol=2,frameon=False,fontsize=7,
               handlelength=1.1,columnspacing=1.5,borderaxespad=0)
    fig.savefig(OUT/'scaling_comparisons.pdf',pad_inches=.035)
    plt.close(fig)
    return dict(split='development',data=data_records,capacity=capacity_records)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    training_evidence()
    protocol_overview()
    import json
    records = dict(multistep=multistep_prediction(), scaling=scaling_comparisons())
    (OUT/'experiment_plot_data.json').write_text(json.dumps(records, indent=2)+'\n')
