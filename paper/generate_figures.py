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
    axes[0, 0].set_ylabel('Prediction loss / epoch 5')
    axes[1, 0].set_ylabel('Response error (NRE)')
    handles = [Line2D([], [], color=colors['joint'], label='Joint'),
               Line2D([], [], color=colors['frozen'], label='Frozen Encoder'),
               Line2D([], [], color='.3', marker='o', label='Training queries'),
               Line2D([], [], color='.3', marker='s', linestyle='--', label='Development')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(OUT / 'training_evidence.pdf')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    training_evidence()
