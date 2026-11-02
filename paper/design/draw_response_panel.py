"""Original paired-error matrix and schematic NRE explanation for overview previews."""
import math


def draw_original_error_grid(s):
    BLUE, ORANGE = '#0094ff', '#ff780a'
    DARK_BLUE, DARK_ORANGE, OUTLINE = '#0044b5', '#c52e00', '#8795aa'
    for j,(letter,dark) in enumerate([('A',DARK_BLUE),('B',DARK_ORANGE)]):
        xx=861+j*170
        s.rect(xx,593,169,44,'#e0f0fe' if j==0 else '#fff0e3',OUTLINE,1,5)
        s.text(xx+84.5,624,'True future '+letter,22,dark,700,'middle')
        s.rect(773,636+j*68,88,66,'#e1f0ff' if j==0 else '#fff0e2',OUTLINE,1,5)
        s.lines(817,664+j*68,['Pred','from '+letter],21,gap=23,fill=dark,weight=700,anchor='middle')
    for i in range(2):
     for j in range(2):
        xx=861+j*170;yy=637+i*68
        fill='url(#blue)' if i==j==0 else 'url(#orange)' if i==j==1 else 'url(#neutral)'
        stroke=BLUE if i==j==0 else ORANGE if i==j==1 else '#c3cbd7'
        s.rect(xx,yy,169,66,fill,stroke,1.2,4)
        s.raw(f'<text x="{xx+84.5}" y="{yy+45}" text-anchor="middle" style="font-family:Georgia,serif;font-style:italic" font-size="36">d<tspan font-size="24" baseline-shift="sub">{"AB"[i]}{"AB"[j]}</tspan></text>')


def draw_original_error_matrix(s, formula, check):
    BLUE, ORANGE = '#0094ff', '#ff780a'
    DARK_BLUE, DARK_ORANGE, OUTLINE = '#0044b5', '#c52e00', '#8795aa'
    s.rect(779,485,525,93,'#e3f2fe',r=11)
    s.text(1041.5,514,'Future discrimination (row-wise)',23,DARK_BLUE,700,'middle')
    for xx,first,second in [(839,'AA','AB'),(1073,'BB','BA')]:
        s.rect(xx,526,188,44,'white','#b3bfd0',1.1,11);formula(xx+27,555,first,second,25);check(xx+160,548)
    draw_original_error_grid(s)
    s.text(1031,797,'Prediction error (lower is better)',18,'#667488',anchor='middle')
    s.rect(1219,590,200,212,'#fff0e4',r=12)
    s.text(1319,621,'History benefit',23,DARK_ORANGE,700,'middle')
    s.text(1319,649,'(column-wise)',22,DARK_ORANGE,anchor='middle')
    for yy,first,second in [(669,'AA','BA'),(731,'BB','AB')]:
        s.rect(1227,yy,185,48,'white','#c6c4c3',1.1,11);formula(1246,yy+32,first,second,25);check(1385,yy+24)


def draw_arrow_error_matrix(s):
    # Preserve the approved cell design; replace the surrounding inequalities.
    s.raw('<g id="arrow-error-grid" transform="translate(609,595) scale(0.9) translate(-773,-593)">')
    draw_original_error_grid(s)
    s.raw('</g>')
    row_color, column_color = '#0044b5', '#c52e00'
    s.text(840, 553, 'Future discrimination', 20, row_color, 700, 'middle')
    # Row A: AB -> AA. Row B: BA -> BB.
    s.arrow(917.25, 577, 764.25, 577, row_color, 2.3, 8)
    s.arrow(764.25, 775, 917.25, 775, row_color, 2.3, 8)
    # Column A: BA -> AA. Column B: AB -> BB.
    s.arrow(592, 725.5, 592, 664.3, column_color, 2.3, 8)
    s.arrow(1012, 664.3, 1012, 725.5, column_color, 2.3, 8)
    s.text(1045, 695, 'History benefit', 18, column_color, 700, 'middle', transform='rotate(90 1045 695)')
    s.text(834, 802, 'Arrowheads: lower error', 17, '#667488', anchor='middle')


def draw_response_panel(s, scene, panel, formula, check, prediction_images=False):
    blue, orange = '#0094ff', '#ff780a'
    dark_blue, dark_orange = '#0044b5', '#c52e00'
    ink, muted, outline = '#263547', '#667488', '#8795aa'
    panel(20, 471, 1408, 346, 'B', 'Paired evaluation protocol', 32)
    s.raw('<g id="history-prediction-flow" transform="translate(37,554) scale(0.825) translate(-37,-537)">')
    for mode, y, col, dark, letter in [
        ('cannot_hold', 537, blue, dark_blue, 'A'),
        ('can_hold', 678, orange, dark_orange, 'B'),
    ]:
        s.rect(37, y, 200, 125, '#f1f9ff' if letter == 'A' else '#fff7f0', col, 1.2, 10)
        s.rect(39, y+2, 196, 33, 'url(#blue)' if letter == 'A' else 'url(#orange)', r=8)
        s.text(137, y+25, 'History '+letter, 24, dark, 700, 'middle')
        for t, x in [(0, 49), (1, 101), (2, 177)]:
            scene(mode, t, x, y+42, 46, 52)
        s.text(162, y+72, '…', 21, anchor='middle')
        s.text(137, y+115, 'observations + actions', 18, '#536172', anchor='middle')
        s.arrow(240, y+62.5, 277, y+62.5, col, 2, 8)
    s.rect(280, 550, 170, 240, 'url(#model)', outline, 1.3, 11)
    s.lines(365, 630.5, ['Frozen', 'latent world', 'model'], 26, gap=30, weight=700, anchor='middle')
    for angle in [0, math.pi/3, 2*math.pi/3]:
        ux, uy = math.cos(angle), math.sin(angle)
        s.line(365-18*ux, 721.5-18*uy, 365+18*ux, 721.5+18*uy, '#348fbc', 2.2)
        for sign in [-1, 1]:
            x, y = 365+sign*18*ux, 721.5+sign*18*uy
            for side in [-1, 1]:
                s.line(x, y, x-sign*7*ux+side*4*uy, y-sign*7*uy-side*4*ux, '#348fbc', 2)
    for y, col, dark, letter in [(549, blue, dark_blue, 'A'), (690, orange, dark_orange, 'B')]:
        s.arrow(453, y+50.5, 491, y+50.5, col, 2, 8)
        if not prediction_images:
            s.rect(495, y, 165, 101, '#f4faff' if letter == 'A' else '#fff9f3', col, 1.2, 10)
            s.rect(497, y+2, 161, 32, '#e4f2ff' if letter == 'A' else '#ffeedf', r=8)
            s.text(577.5, y+25, 'Prediction from '+letter, 20, dark, 700, 'middle')
            s.raw(f'<text x="555" y="{y+82}" font-size="43" fill="{dark}" style="font-family:Georgia,serif;font-style:italic">z<tspan font-size="25" baseline-shift="sub">{letter}</tspan></text>')
            s.raw(f'<path d="M 557 {y+49} l 9 -7 l 9 7" fill="none" stroke="{dark}" stroke-width="2"/>')
    s.raw('</g>')
    if prediction_images:
        # Real future images are visual references, not decoded predictions.
        for mode, y, col, dark, letter in [
            ('cannot_hold', 553.0625, blue, dark_blue, 'A'),
            ('can_hold', 669.3875, orange, dark_orange, 'B'),
        ]:
            s.rect(415, y, 137, 105, '#f4faff' if letter == 'A' else '#fff9f3', col, 1.0, 8)
            s.rect(417, y+2, 133, 26, '#e4f2ff' if letter == 'A' else '#ffeedf', r=6)
            s.text(483.5, y+20, 'Prediction '+letter, 16.5, dark, 700, 'middle')
            scene(mode, 3, 423, y+32, 65, 52)
            s.text(455.5, y+99, 'Reference', 14, '#667488', anchor='middle')
            s.raw(f'<text x="505" y="{y+73}" font-size="32" fill="{dark}" style="font-family:Georgia,serif;font-style:italic">z<tspan font-size="19" baseline-shift="sub">{letter}</tspan></text>')
            s.raw(f'<path d="M 507 {y+47} l 7 -5 l 7 5" fill="none" stroke="{dark}" stroke-width="1.6"/>')
            s.text(522, y+99, 'latent', 14, dark, anchor='middle')
    draw_arrow_error_matrix(s)
    s.line(1103, 542, 1103, 790, '#dbe2eb', 1)

    # Blue/orange and neutral fills follow the supplied response-fidelity design.
    # These vectors illustrate differences between conditions, not data projections.
    true_color, predicted_color = '#3187C6', '#E47E47'
    s.rect(1119, 537, 298, 254, '#FCFDFE', '#E1E8EF', 1, 10)
    s.text(1268, 560, 'Response fidelity', 21, ink, 700, 'middle')
    s.text(1268, 583, 'Difference between conditions', 15, muted, anchor='middle')
    origin, true, predicted = (1144, 706), (1311, 611), (1275, 657)
    s.arrow(*origin, *true, true_color, 3, 10)
    s.arrow(*origin, *predicted, predicted_color, 3, 10)
    s.circle(*origin, 3, '#63788D')
    s.line(*true, *predicted, '#7F8B9B', 1.3, '4 4')
    s.text(1320, 610, 'True', 17, '#277EBF', 700)
    s.text(1294, 675, 'Predicted', 17, '#D86632', 700)
    s.raw('<text x="1338" y="636" font-size="23" fill="#277EBF" style="font-family:Georgia,serif;font-style:italic">Δz</text>')
    s.raw('<text x="1334" y="704" font-size="23" fill="#D86632" style="font-family:Georgia,serif;font-style:italic">Δẑ</text>')
    # Dataset-level ratio of sums, consistent with the manuscript's NRE definition.
    s.rect(1126, 724, 284, 60, '#F3F6F9', '#E1E8EF', .8, 8)
    s.text(1135, 761, 'NRE =', 19, '#33485E', 700)
    s.line(1205, 754, 1398, 754, '#63788D', 1)
    def term(x, y, numerator):
        inside = 'Δẑ<tspan baseline-shift="sub" font-size="12">i</tspan> − Δz<tspan baseline-shift="sub" font-size="12">i</tspan>' if numerator else 'Δz<tspan baseline-shift="sub" font-size="12">i</tspan>'
        s.raw(f'<text x="{x}" y="{y}" text-anchor="middle" fill="#33485E" font-size="19" style="font-family:Georgia,serif">∑<tspan baseline-shift="sub" font-size="12">i</tspan> ‖{inside}‖<tspan baseline-shift="super" font-size="12">2</tspan></text>')
    term(1301.5, 743, True)
    term(1301.5, 776, False)
