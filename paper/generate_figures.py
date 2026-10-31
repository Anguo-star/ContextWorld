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
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.6), sharex='col')
    colors = {'joint': '#2765a8', 'frozen': '#c46124'}
    for col, (task, title) in enumerate([('contact_friction', 'Contact friction'),
                                        ('motion_damping', 'Motion damping')]):
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
            axes[0, col].plot(xs, [float(r['mean'])/base for r in values],
                              color=colors[regime], marker='o', markersize=3)
            axes[0, col].fill_between(xs, [float(r['q25'])/base for r in values],
                                      [float(r['q75'])/base for r in values],
                                      color=colors[regime], alpha=.12, linewidth=0)
            for split, linestyle, marker in [('loader_fit', '-', 'o'), ('development', '--', 's')]:
                rows = sorted([r for r in nre if r['task'] == task and r['family'] == 'pldm'
                               and r['regime'] == regime and r['split'] == split],
                              key=lambda r: int(r['epoch']))
                assert [int(r['epoch']) for r in rows] == [5, 10]
                axes[1, col].plot([5, 10], [float(r['NRE']) for r in rows],
                                  color=colors[regime], linestyle=linestyle, marker=marker,
                                  markersize=4)
        axes[0, col].set_title(title)
        axes[0, col].set_ylim(.75, 1.08)
        axes[1, col].set_ylim(0, 1.13)
        axes[1, col].axhline(1, color='.5', linewidth=.7, linestyle=':')
        axes[1, col].set_xlabel('Training epoch')
        axes[1, col].set_xticks(range(5, 11))
        for ax in axes[:, col]:
            ax.grid(axis='y', alpha=.15)
    axes[0, 0].set_ylabel('Prediction loss\n/ epoch 5')
    axes[1, 0].set_ylabel('Response error\n(NRE)')
    handles = [Line2D([], [], color=colors['joint'], label='Joint'),
               Line2D([], [], color=colors['frozen'], label='Frozen Encoder'),
               Line2D([], [], color='.3', marker='o', label='Training queries'),
               Line2D([], [], color='.3', marker='s', linestyle='--', label='Development')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(OUT / 'training_evidence.pdf')
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
        ax.axhline(1, color='#333333', linestyle=':', linewidth=.9)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(horizons)
        ax.set_ylim(0, 3.2)
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Simulator steps', fontsize=7.5)
    axes[0].set_ylabel(r'Prediction error $E_t/B_t$ $\downarrow$', fontsize=7.5)
    handles = [Line2D([],[],color=colors['original'],marker='s',markersize=3,label='Original'),
               Line2D([],[],color=colors['frozen'],marker='o',markersize=3,label='Original → Mixed (Frozen Encoder)'),
               Line2D([],[],color='#333333',linestyle=':',label='Mean-future reference')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=7, ncol=1,
               bbox_to_anchor=(.5,-.015), handlelength=1.5, labelspacing=.25)
    fig.tight_layout(rect=(0,.22,1,1),pad=.3,w_pad=.5)
    fig.savefig(OUT/'multistep_prediction.pdf')
    plt.close(fig)
    return dict(split='development', task='action_strength',
                metric='sum of prediction energies / sum of mean-future energies at each horizon; equal mean over training runs',
                interval='95% source-cluster bootstrap; 4000 resamples; conditional on checkpoints', rows=records)


def scaling_comparisons():
    import json
    import numpy as np
    data = ROOT/'docs/research/data'
    study = json.loads((data/'icl_training_study_v2.json').read_text())
    capacity = json.loads((data/'model_capacity_scaling_v1/final_comparison.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.2),gridspec_kw={'width_ratios':[1,1.15]})
    colors = {'scratch':'#777777','joint':'#2765a8'}
    data_records = []
    ax = axes[0]
    for regime,label in [('scratch','Mixed'),('joint','Original → Mixed (Joint)')]:
        rows = [r for r in study['rows'] if r['task']=='action_strength' and r['model']=='lewm'
                and r['regime']==regime and r.get('training_pair_count')
                and (not r['comparison_variant'] or r['comparison_variant'].startswith('scale_'))]
        rows.sort(key=lambda r:r['training_pair_count'])
        xs = [r['training_pair_count'] for r in rows]
        ys = [r['statistics']['scores']['main']['mean'] for r in rows]
        assert len(xs)==len(set(xs))
        ax.plot(xs,ys,color=colors[regime],marker='s' if regime=='scratch' else 'o',
                markersize=4,label=label,linewidth=1.4)
        for x,y,r in zip(xs,ys,rows):
            st=r['statistics']['scores']['main']
            if st['n']>1 and st.get('std') is not None:
                ax.errorbar(x,y,yerr=st['std'],color=colors[regime],capsize=2,linewidth=.8)
            ax.annotate(f'{y:.1f}',(x,y),xytext=(0,5 if regime=='joint' else -11),
                        textcoords='offset points',ha='center',fontsize=7,color=colors[regime])
            data_records.append(dict(regime=regime,pairs=x,accuracy=y,n=st['n'],sd=st.get('std')))
    ax.axhline(50,color='.65',ls=':',lw=.7)
    ax.set_xscale('log',base=2)
    ax.set_xticks([2048,8192,32768],['2,048','8,192','32,768'])
    ax.set_xlim(1500,45000); ax.set_ylim(35,102)
    ax.set_yticks([50,75,100]); ax.set_xlabel('Training pairs')
    ax.set_ylabel('Development accuracy (%) ↑')
    ax.set_title('(a) Data: LeWM, pusher displacement',fontsize=8)
    ax.grid(axis='y',alpha=.15)
    ax=axes[1]
    labels=[]; capacity_records=[]
    for y,r in enumerate(capacity['rows']):
        small=r['small']['main_percent'];large=r['scaling10x']['main_percent']
        labels.append(('Friction' if r['task']=='contact_friction' else 'Damping')+' · '+('LeWM' if r['model']=='lewm' else 'PLDM'))
        ax.plot([small,large],[y,y],color='.5',lw=1)
        ax.scatter([small],[y],color='#777777',s=28,marker='o',zorder=3)
        ax.scatter([large],[y],color='#2765a8',s=30,marker='x',zorder=4)
        ax.text(63,y,f'{large-small:+.2f}',va='center',ha='right',fontsize=7)
        capacity_records.append(dict(task=r['task'],family=r['model'],small=small,large=large,delta=large-small))
    ax.set_yticks(range(len(labels)),labels,fontsize=7)
    ax.set_ylim(3.5,-.9); ax.set_xlim(46,64)
    ax.set_xticks([50,55,60]);ax.axvline(50,color='.65',ls=':',lw=.7)
    ax.text(63,-.65,'Δ (pp)',fontsize=7,ha='right')
    ax.set_xlabel('Development accuracy (%) ↑')
    ax.set_title('(b) Capacity: Mixed training',fontsize=8)
    ax.grid(axis='x',alpha=.15)
    handles = [Line2D([],[],color=colors['scratch'],marker='s',label='Mixed'),
               Line2D([],[],color=colors['joint'],marker='o',label='Original → Mixed (Joint)'),
               Line2D([],[],color='#777777',marker='o',lw=0,label='18M'),
               Line2D([],[],color='#2765a8',marker='x',lw=0,label='183M')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=7,frameon=False,
               bbox_to_anchor=(.5,-.01),handlelength=1.4,columnspacing=1.5)
    fig.tight_layout(rect=(0,.10,1,1),pad=.5,w_pad=2)
    fig.savefig(OUT/'scaling_comparisons.pdf')
    plt.close(fig)
    return dict(split='development',data=data_records,capacity=capacity_records)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    training_evidence()
    protocol_overview()
    import json
    records = dict(multistep=multistep_prediction(), scaling=scaling_comparisons())
    (OUT/'experiment_plot_data.json').write_text(json.dumps(records, indent=2)+'\n')
