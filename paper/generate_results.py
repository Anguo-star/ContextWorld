"""Populate supplemental result tables from frozen repository summaries."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/research/data'
read = lambda name: json.loads((DATA / name).read_text())
STUDY = read('icl_training_study_v2.json')
MULTI = read('multistep_prediction_coverage_v2.json')
TASKS = [('speed','Speed'), ('action_strength','Pusher displacement'),
         ('robot_arm_mass','Arm mass'), ('action_delay','Action delay'),
         ('contact_friction','Contact friction'), ('motion_damping','Motion damping'),
         ('cube_gripper_carry','Gripper carry rule'), ('door','Door traversability'),
         ('portal_exit','Portal exit')]
FAMILIES = [('lewm','LeWM'),('pldm','PLDM'),('dinowm','DINO-WM')]
LABELS = {'original':'Original', 'scratch':'Mixed',
          'joint':r'Original $\rightarrow$ Mixed (Joint)',
          'frozen':r'Original $\rightarrow$ Mixed (Frozen Encoder)'}
rows = {(r['task'],r['model'],r['regime']):r for r in STUDY['rows'] if not r['comparison_variant']}
mult = {(r['task'],r['family'],r['regime']):r for r in MULTI['rows']}
assert len(mult)==86

def mean(r,key):
    stat = r['statistics']['scores'].get(key)
    return stat['mean'] if stat else None

def fmt(v,dec=2):
    return '---' if v is None else f'{v:.{dec}f}'

text = [r'''\begingroup
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}p{1.8cm}lp{3.65cm}rrrp{2.5cm}r@{}}
\caption{Development prediction diagnostics and original-environment CEM. Entries average the available training repetitions; $E/B$ intervals are 95\% source-bootstrap intervals. History and CEM are percentages. Scheme names match the main text.}\\
\toprule
Task & Model & Scheme & History $\uparrow$ & NRE $\downarrow$ & $E/B$ $\downarrow$ & $E/B$ interval & CEM $\uparrow$\\
\midrule
\endfirsthead
\toprule
Task & Model & Scheme & History $\uparrow$ & NRE $\downarrow$ & $E/B$ $\downarrow$ & $E/B$ interval & CEM $\uparrow$\\
\midrule
\endhead
\bottomrule
\endfoot''']
for task,name in TASKS:
    for family,model in FAMILIES:
        for regime,label in LABELS.items():
            key=(task,family,regime)
            if key not in mult:continue
            m=mult[key];r=rows[key]
            assert r['evaluation_split']=='development'
            eb=1-m['score']/100
            lo,hi=1-m['ci95'][1]/100,1-m['ci95'][0]/100
            text.append(' & '.join([name,model,label,fmt(mean(r,'history')),fmt(mean(r,'nre'),3),fmt(eb,3),f'[{lo:.3f}, {hi:.3f}]',fmt(mean(r,'cem'))])+r' \\')
    text.append(r'\midrule')
text[-1]=r'\end{longtable}\endgroup'
text.append(r'''\subsection{Data Scale}
These comparisons use Development and hold training epochs fixed.
The sample counts describe pairs, not necessarily independent source episodes.
Pusher displacement uses 2,048, 8,192, and 32,768 pairs; friction and damping compare 8,192 with 32,768; mass and portal exit compare 2,048 with 32,768.
Gripper carry expands from 2,048 to 10,000 source episodes.
Portal expansion also broadens initial distances and action magnitudes, so its comparison changes coverage as well as size.
\begingroup\small
\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}p{2cm}lp{4cm}rrrr@{}}
\caption{Data-scale comparisons on Development. $n$ counts training repetitions contributing to task accuracy and NRE. Planning repetitions can differ and are not counted as additional training runs.}\\
\toprule
Task & Model & Scheme & Pairs & $n$ & Accuracy (\%) & NRE\\
\midrule\endfirsthead
\toprule
Task & Model & Scheme & Pairs & $n$ & Accuracy (\%) & NRE\\
\midrule\endhead
\bottomrule\endfoot''')
for task,name in TASKS:
    if task in ['speed','action_delay','door']:continue
    for family,model in FAMILIES:
        for regime,label in LABELS.items():
            selected=[r for r in STUDY['rows'] if r['task']==task and r['model']==family and r['regime']==regime and (r['comparison_variant'] or not r['comparison_variant']) and r.get('training_pair_count')]
            # Only show schemes with an actual size comparison.
            selected=[r for r in selected if not r['comparison_variant'] or r['comparison_variant'].startswith('scale_')]
            if len({r['training_pair_count'] for r in selected})<2:continue
            for r in sorted(selected,key=lambda x:x['training_pair_count']):
                st=r['statistics']['scores']['main']
                text.append(' & '.join([name,model,label,f"{r['training_pair_count']:,}",str(st['n']),fmt(st['mean']),fmt(mean(r,'nre'),3)])+r' \\')
text.append(r'\end{longtable}\endgroup')
cap=read('model_capacity_scaling_v1/final_comparison.json')
text.append(r'''\subsection{Model Capacity}
Both model sizes use the same 32,768-pair datasets, training seed, Mixed initialization, and 10-epoch recipe.
The larger model changes the visual backbone from ViT-Tiny to ViT-Base, the representation width from 192 to 768, and predictor depth from six to nine layers.
This is a joint expansion of the encoder and predictor, not a predictor-only intervention.
\begin{center}\small
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
& & \multicolumn{2}{c}{Accuracy (\%)} & \multicolumn{2}{c}{NRE} & \multicolumn{2}{c}{CEM (\%)}\\
Task & Model & 18M & 183M & 18M & 183M & 18M & 183M\\
\midrule''')
for r in cap['rows']:
    a,b=r['small'],r['scaling10x']
    text.append(' & '.join([dict(TASKS)[r['task']],dict(FAMILIES)[r['model']],fmt(a['main_percent']),fmt(b['main_percent']),fmt(a['nre'],3),fmt(b['nre'],3),fmt(a['cem_percent']),fmt(b['cem_percent'])])+r' \\')
text.append(r'''\bottomrule\end{tabular}\end{center}
The four endpoint comparisons do not show a consistent improvement from increasing capacity under this recipe.
Only measured checkpoints are used in the training-curve analysis: epochs 1--10 for the large LeWM models and epochs 5 and 10 for the other models.
There is no shared trajectory of accurate training responses followed by worsening Development response error across these runs.
This does not exclude earlier overfitting where checkpoints are unavailable, or gains from retuning the larger models.
''')
p=ROOT/'paper/main_en.tex';s=p.read_text();start='% BEGIN GENERATED PAPER RESULTS';end='% END GENERATED PAPER RESULTS'
assert s.count(start)==s.count(end)==1
s=s[:s.index(start)+len(start)]+'\n'+'\n'.join(text)+'\n'+s[s.index(end):]
p.write_text(s)
print('Generated 86 diagnostic rows and scale tables from frozen source files.')
