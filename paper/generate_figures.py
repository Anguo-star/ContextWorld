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


def reference_style_figures():
    """Build the overview and training plots, then export vector PDFs."""
    import subprocess
    import sys
    import shutil
    design = ROOT / 'paper' / 'design'
    for script in ['draw_overview.py', 'draw_training.py', 'draw_training_full.py']:
        subprocess.run([sys.executable, str(design / script)], check=True)
    subprocess.run(['node', str(design / 'render_figures.cjs')], check=True)
    for source, target in [('benchmark_overview', 'protocol_overview'),
                           ('training_two_panels', 'training_evidence'),
                           ('training_full', 'training_evidence_full')]:
        shutil.copyfile(design / (source + '.pdf'), OUT / (target + '.pdf'))


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
        ax.set_title(title, fontsize=8)
        ax.set_xticks(horizons, range(1, len(horizons)+1))
        ax.set_ylim(0, 3.35)
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Rollout step', fontsize=7.5)
    axes[0].set_ylabel('Prediction error /\nbaseline error', fontsize=7.5)
    handles = [Line2D([],[],color=colors['original'],marker='s',markersize=3,label='Original'),
               Line2D([],[],color=colors['frozen'],marker='o',markersize=3,label='Original → Mixed (Frozen Encoder)')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=7, ncol=1,
               bbox_to_anchor=(.5,-.015), handlelength=1.5, labelspacing=.25)
    fig.tight_layout(rect=(0,.18,1,1),pad=.3,w_pad=.5)
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
    axes = [fig.add_axes([.15, .60, .81, .27]),
            fig.add_axes([.15, .10, .81, .27])]
    before_data, after_data = [], []
    for regime in ['scratch', 'joint']:
        small, = [r for r in data_records if r['regime']==regime and r['pairs']==2048]
        large, = [r for r in data_records if r['regime']==regime and r['pairs']==32768]
        before_data.append(small['accuracy']); after_data.append(large['accuracy'])
    def bars(ax, before, after, labels, scales):
        import numpy as np
        x = np.arange(len(before))
        ax.bar(x-.16, before, .27, color='#b9c1c9', label=scales[0], zorder=3)
        ax.bar(x+.16, after, .27, color='#426d99', label=scales[1], zorder=3)
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
            for offset,value in [(-.16,small),(.16,large)]:
                ax.text(i+offset,value+3,f'{value:.1f}',
                        ha='center',va='bottom',fontsize=6.2)
    bars(axes[0],before_data,after_data,['Mixed','Original → Mixed\n(Joint)'],
         ['2k pairs','32k pairs'])
    labels=[('LeWM' if r['family']=='lewm' else 'PLDM')+'\n'+
            ('friction' if r['task']=='contact_friction' else 'damping') for r in capacity_records]
    bars(axes[1],[r['small'] for r in capacity_records],
         [r['large'] for r in capacity_records],labels,['18M','183M'])
    fig.text(.15,.99,'(a) ContextWorld data scaling',fontsize=8,va='top')
    fig.text(.15,.49,'(b) Model scaling',fontsize=8,va='top')
    for ax,y in zip(axes,[.94,.44]):
        fig.legend(*ax.get_legend_handles_labels(),loc='upper left',
                   bbox_to_anchor=(.13,y),ncol=2,frameon=False,fontsize=7,
                   handlelength=1.1,columnspacing=1.5,borderaxespad=0)
    fig.savefig(OUT/'scaling_comparisons.pdf',pad_inches=.035)
    plt.close(fig)
    return dict(split='development',data=data_records,capacity=capacity_records)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    reference_style_figures()
    import json
    records = dict(multistep=multistep_prediction(), scaling=scaling_comparisons())
    (OUT/'experiment_plot_data.json').write_text(json.dumps(records, indent=2)+'\n')
