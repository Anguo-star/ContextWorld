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


def draw_controlled_error_comparisons(s, scene):
    """Illustrate error comparisons with latent markers and real target images."""
    ink, muted, line = '#263547', '#667488', '#8795aa'
    blue, orange = '#0044b5', '#c52e00'
    s.raw('<g id="controlled-error-comparisons">')
    s.rect(574, 488, 430, 316, '#FCFDFE', '#E1E8EF', 1, 10)
    s.text(789, 512, 'Error comparisons', 20, ink, 700, 'middle')
    s.text(789, 534, 'Latent-space schematic · condition A', 16, muted, anchor='middle')

    def prediction(x, y, condition, label_x, label_y):
        color = blue if condition == 'A' else orange
        s.raw(f'<path d="M {x} {y-6} L {x+6} {y} L {x} {y+6} L {x-6} {y} Z" fill="white" stroke="{color}" stroke-width="2"/>')
        s.text(label_x, label_y, 'Prediction '+condition, 16, color, 700, 'middle')

    def target(x, y, condition, image_x, image_y):
        color = blue if condition == 'A' else orange
        mode = 'cannot_hold' if condition == 'A' else 'can_hold'
        # The circle is a latent target; its attached image is the true observation.
        s.line(x+5, y, image_x-3, image_y+25, color, 1, '2 3')
        s.circle(x, y, 5, color)
        scene(mode, 3, image_x, image_y, 70, 50)
        s.text(image_x+35, image_y-5, 'True future '+condition, 15.5, color, 700, 'middle')

    def error_segment(x1, y1, x2, y2, label_x, label_y):
        s.line(x1, y1, x2, y2, line, 1.6)
        s.text(label_x, label_y, 'error', 15, muted, anchor='middle')

    s.text(592, 557, '1. Same prediction', 17, ink, 700)
    error_segment(648, 625, 760, 600, 723, 598)
    error_segment(648, 625, 888, 648, 775, 655)
    prediction(648, 625, 'A', 650, 603)
    target(760, 600, 'A', 778, 575)
    target(888, 648, 'B', 914, 625)

    s.line(592, 683, 986, 683, '#E1E8EF', 1)
    s.text(592, 698, '2. Same true future', 17, ink, 700)
    error_segment(758, 737, 888, 748, 834, 735)
    error_segment(640, 760, 888, 748, 775, 776)
    prediction(758, 737, 'A', 760, 723)
    prediction(640, 760, 'B', 650, 744)
    target(888, 748, 'A', 910, 723)
    s.text(789, 795, 'Closer states indicate lower error', 15.5, ink, anchor='middle')
    s.raw('</g>')


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
    draw_controlled_error_comparisons(s, scene)

    # Blue/orange and neutral fills follow the supplied response-fidelity design.
    # These vectors illustrate differences between conditions, not data projections.
    true_color, predicted_color = '#3187C6', '#E47E47'
    s.rect(1024, 488, 375, 316, '#FCFDFE', '#E1E8EF', 1, 10)
    s.text(1211.5, 512, 'Response fidelity', 20, ink, 700, 'middle')
    s.text(1211.5, 534, 'Difference between conditions', 16.5, muted, anchor='middle')
    # Enlarge the schematic vectors uniformly, preserving their directions.
    origin = (1074, 700)
    true = (origin[0]+1.2*167, origin[1]-1.2*95)
    predicted = (origin[0]+1.2*131, origin[1]-1.2*49)
    s.arrow(*origin, *true, true_color, 3, 10)
    s.arrow(*origin, *predicted, predicted_color, 3, 10)
    s.circle(*origin, 3, '#63788D')
    s.line(*true, *predicted, '#7F8B9B', 1.3, '4 4')
    s.text(1287, 588, 'True', 18, '#277EBF', 700)
    s.text(1255, 665, 'Predicted', 18, '#D86632', 700)
    s.raw('<text x="1305" y="616" font-size="24" fill="#277EBF" style="font-family:Georgia,serif;font-style:italic">Δz</text>')
    s.raw('<text x="1295" y="692" font-size="24" fill="#D86632" style="font-family:Georgia,serif;font-style:italic">Δẑ</text>')
    # Dataset-level ratio of sums, consistent with the manuscript's NRE definition.
    s.rect(1034, 731, 355, 63, '#F3F6F9', '#E1E8EF', .8, 8)
    s.text(1048, 770, 'NRE =', 20, '#33485E', 700)
    s.line(1128, 762, 1373, 762, '#63788D', 1)
    def term(x, y, numerator):
        inside = 'Δẑ<tspan baseline-shift="sub" font-size="13">i</tspan> − Δz<tspan baseline-shift="sub" font-size="13">i</tspan>' if numerator else 'Δz<tspan baseline-shift="sub" font-size="13">i</tspan>'
        s.raw(f'<text x="{x}" y="{y}" text-anchor="middle" fill="#33485E" font-size="21" style="font-family:Georgia,serif">∑<tspan baseline-shift="sub" font-size="13">i</tspan> ‖{inside}‖<tspan baseline-shift="super" font-size="13">2</tspan></text>')
    term(1250.5, 749, True)
    term(1250.5, 784, False)
