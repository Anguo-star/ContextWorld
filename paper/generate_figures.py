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
    subprocess.run([sys.executable, str(design / 'draw_overview.py'),
                    '--example', 'pusher', '--response-fidelity'], check=True)
    for script in ['draw_training.py', 'draw_training_full.py']:
        subprocess.run([sys.executable, str(design / script)], check=True)
    subprocess.run(['node', str(design / 'render_figures.cjs'),
                    'benchmark_overview_pusher_response', 'paired_evaluation_design',
                    'training_two_panels', 'training_full'], check=True)
    for source, target in [('benchmark_overview_pusher_response', 'protocol_overview'),
                           ('training_two_panels', 'training_evidence'),
                           ('training_full', 'training_evidence_full')]:
        shutil.copyfile(design / (source + '.pdf'), OUT / (target + '.pdf'))


def multistep_prediction():
    """Per-horizon ratios; average normalized runs, never average scene ratios."""
    import gzip
    import json
    import numpy as np
    data = ROOT / 'docs/research/data'
    summary = json.loads((data / 'multistep_prediction_test_v1/summary.json').read_text())
    with gzip.open(data / 'multistep_prediction_test_v1/summary_queries.json.gz', 'rt') as f:
        queries = json.load(f)
    assert summary['evaluation_split'] == 'test'
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
            records.append(dict(family=family, regime=regime, run_ids=row['run_ids'],
                                horizons=horizons.tolist(), per_horizon_ratio=mean.tolist(),
                                ci95=np.stack([lo,hi],axis=1).tolist(), aggregate_ratio=float(np.mean(overall))))
        ax.set_title(title, fontsize=8)
        ax.set_xticks(horizons, range(1, len(horizons)+1))
        ax.set_ylim(0, 3.35)
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Rollout step', fontsize=7.5)
    axes[0].set_ylabel('Normalized\nprediction error', fontsize=7.5)
    handles = [Line2D([],[],color=colors['original'],marker='s',markersize=3,label='Original'),
               Line2D([],[],color=colors['frozen'],marker='o',markersize=3,label='Original → Mixed (Frozen Encoder)')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=7, ncol=1,
               bbox_to_anchor=(.5,-.015), handlelength=1.5, labelspacing=.25)
    fig.tight_layout(rect=(0,.18,1,1),pad=.3,w_pad=.5)
    fig.savefig(OUT/'multistep_prediction.pdf')
    plt.close(fig)
    return dict(split='test', task='action_strength',
                metric='sum of prediction energies / sum of mean-future energies at each horizon; equal mean over training runs',
                interval='95% source-cluster bootstrap; 4000 resamples; conditional on checkpoints', rows=records)


def scaling_comparisons():
    """Test accuracy for data scaling and task-dependent encoder freezing."""
    import json
    data = ROOT / 'docs/research/data'
    scale_source = 'docs/research/data/figure3_data_scaling_test_v1/metrics.json'
    test_source = 'docs/research/data/current_reference_test_v1/metrics.json'
    scaling = json.loads((ROOT / scale_source).read_text())
    test = json.loads((ROOT / test_source).read_text())
    assert scaling['evaluation_split'] == test['evaluation_split'] == 'test'
    data_records = []
    for regime in ['scratch', 'joint']:
        for pairs in [2048, 32768]:
            row, = [r for r in scaling['cells'] if r['task'] == 'action_strength'
                    and r['family'] == 'lewm' and r['regime'] == regime
                    and r['training_pair_count'] == pairs]
            st = row['scores']['accuracy']
            data_records.append(dict(regime=regime, pairs=pairs,
                                     accuracy=st['mean'], n=st['n'], sd=st['std'],
                                     runs=row['runs']))
    encoder_tasks = [
        ('lewm', 'contact_friction', 'Contact\nfriction'),
        ('lewm', 'robot_arm_mass', 'Arm mass'),
        ('pldm', 'contact_friction', 'Contact\nfriction'),
        ('pldm', 'motion_damping', 'Motion\ndamping'),
    ]
    encoder_records = []
    for family, task, _ in encoder_tasks:
        for regime in ['joint', 'frozen']:
            row, = [r for r in test['cells'] if r['task'] == task
                    and r['family'] == family and r['regime'] == regime]
            st = row['scores']['accuracy']
            encoder_records.append(dict(task=task, family=family, regime=regime,
                                        accuracy=st['mean'], n=st['n'], sd=st['std'],
                                        runs=row['runs']))
    fig = plt.figure(figsize=(3.45, 3.25))
    axes = [fig.add_axes([.15, .60, .81, .27]),
            fig.add_axes([.15, .065, .81, .27])]

    def bars(ax, before, after, labels, legend, positions=None):
        import numpy as np
        x = np.arange(len(before)) if positions is None else np.asarray(positions)
        ax.bar(x-.16, before, .27, color='#b9c1c9', label=legend[0], zorder=3)
        ax.bar(x+.16, after, .27, color='#426d99', label=legend[1], zorder=3)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('Accuracy (%)', fontsize=7.5, labelpad=3)
        ax.set_xticks(x, labels, fontsize=7)
        ax.tick_params(length=2.5, labelsize=7)
        ax.grid(axis='y', color='#e8ebef', linewidth=.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        for i, values in zip(x, zip(before, after)):
            for offset, value in zip([-.16, .16], values):
                ax.text(i+offset, value+3, f'{value:.1f}',
                        ha='center', va='bottom', fontsize=6.5)

    bars(axes[0], [r['accuracy'] for r in data_records if r['pairs'] == 2048],
         [r['accuracy'] for r in data_records if r['pairs'] == 32768],
         ['Mixed', 'Original → Mixed\n(Joint)'], ['2k pairs', '32k pairs'])
    bars(axes[1], [r['accuracy'] for r in encoder_records if r['regime'] == 'joint'],
         [r['accuracy'] for r in encoder_records if r['regime'] == 'frozen'],
         [label for _, _, label in encoder_tasks], ['Joint', 'Frozen Encoder'],
         positions=[0, 1, 2.35, 3.35])
    for center, label in [(0.5, 'LeWM'), (2.85, 'PLDM')]:
        axes[1].text(center, 1.04, label, transform=axes[1].get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=7, fontweight='medium')
    fig.text(.15, .99, '(a) Data scaling', fontsize=8, va='top')
    fig.text(.15, .49, '(b) Encoder freezing', fontsize=8, va='top')
    for ax, y in zip(axes, [.94, .44]):
        fig.legend(*ax.get_legend_handles_labels(), loc='upper left',
                   bbox_to_anchor=(.13, y), ncol=2, frameon=False, fontsize=7,
                   handlelength=1.1, columnspacing=1.5, borderaxespad=0)
    fig.savefig(OUT / 'scaling_comparisons.pdf', pad_inches=.035)
    plt.close(fig)
    return dict(split='test', data_source=scale_source, encoder_source=test_source,
                data=data_records, encoder=encoder_records)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    reference_style_figures()
    import json
    records = dict(multistep=multistep_prediction(), scaling=scaling_comparisons())
    (OUT/'experiment_plot_data.json').write_text(json.dumps(records, indent=2)+'\n')
