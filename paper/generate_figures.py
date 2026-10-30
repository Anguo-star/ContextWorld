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


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    training_evidence()
    protocol_overview()
