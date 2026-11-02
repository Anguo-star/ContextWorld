"""Compact paired-error matrix and schematic NRE explanation for overview previews."""
import math


def draw_response_panel(s, scene, panel):
    blue, orange = '#0094ff', '#ff780a'
    dark_blue, dark_orange = '#0044b5', '#c52e00'
    ink, muted, outline, purple = '#263547', '#667488', '#8795aa', '#7657aa'
    panel(20, 471, 1408, 346, 'B', 'Paired evaluation protocol', 32)
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
        s.arrow(240, y+65, 277, y+65, col, 2, 8)
    s.text(365, 543, 'Shared future actions', 18, '#536172', anchor='middle')
    s.arrow(365, 548, 365, 565, '#536172', 1.5, 6)
    s.rect(280, 568, 170, 181, 'url(#model)', outline, 1.3, 11)
    s.lines(365, 619, ['Frozen', 'latent world', 'model'], 26, gap=30, weight=700, anchor='middle')
    for angle in [0, math.pi/3, 2*math.pi/3]:
        ux, uy = math.cos(angle), math.sin(angle)
        s.line(365-18*ux, 710-18*uy, 365+18*ux, 710+18*uy, '#348fbc', 2.2)
        for sign in [-1, 1]:
            x, y = 365+sign*18*ux, 710+sign*18*uy
            for side in [-1, 1]:
                s.line(x, y, x-sign*7*ux+side*4*uy, y-sign*7*uy-side*4*ux, '#348fbc', 2)
    for y, col, dark, letter in [(545, blue, dark_blue, 'A'), (681, orange, dark_orange, 'B')]:
        s.arrow(453, y+53, 491, y+53, col, 2, 8)
        s.rect(495, y, 165, 101, '#f4faff' if letter == 'A' else '#fff9f3', col, 1.2, 10)
        s.rect(497, y+2, 161, 32, '#e4f2ff' if letter == 'A' else '#ffeedf', r=8)
        s.text(577.5, y+25, 'Prediction from '+letter, 20, dark, 700, 'middle')
        s.raw(f'<text x="555" y="{y+82}" font-size="43" fill="{dark}" style="font-family:Georgia,serif;font-style:italic">z<tspan font-size="25" baseline-shift="sub">{letter}</tspan></text>')
        s.raw(f'<path d="M 557 {y+49} l 9 -7 l 9 7" fill="none" stroke="{dark}" stroke-width="2"/>')
    s.line(681, 534, 681, 799, '#dbe2eb', 1)
    s.text(880, 542, 'Prediction error', 23, ink, 700, 'middle')
    s.text(910, 568, 'True future', 20, muted, anchor='middle')
    for j, (letter, dark) in enumerate([('A', dark_blue), ('B', dark_orange)]):
        x = 764+j*147
        s.rect(x, 579, 145, 31, '#e0f0fe' if j == 0 else '#fff0e3', outline, 1, 5)
        s.text(x+72.5, 603, letter, 23, dark, 700, 'middle')
        y = 612+j*67
        s.rect(697, y, 65, 64, '#e1f0ff' if j == 0 else '#fff0e2', outline, 1, 5)
        s.lines(729.5, y+24, ['Pred', 'from '+letter], 19, gap=23, fill=dark, weight=700, anchor='middle')
    for i in range(2):
        for j in range(2):
            x, y = 764+j*147, 612+i*67
            fill = 'url(#blue)' if i == j == 0 else 'url(#orange)' if i == j == 1 else 'url(#neutral)'
            stroke = blue if i == j == 0 else orange if i == j == 1 else '#c3cbd7'
            s.rect(x, y, 145, 64, fill, stroke, 1.2, 4)
            s.raw(f'<text x="{x+72.5}" y="{y+44}" text-anchor="middle" style="font-family:Georgia,serif;font-style:italic" font-size="34">d<tspan font-size="23" baseline-shift="sub">{"AB"[i]}{"AB"[j]}</tspan></text>')
    s.text(704, 774, 'Rows → Future discrimination', 20, dark_blue)
    s.text(704, 801, 'Columns ↓ History benefit', 20, dark_orange)
    s.line(1080, 534, 1080, 799, '#dbe2eb', 1)

    # This is a conceptual high-dimensional difference comparison, not a data projection.
    s.rect(1097, 529, 320, 277, '#faf8fe', '#ddd4ec', 1, 10)
    s.text(1257, 552, 'Response fidelity', 24, purple, 700, 'middle')
    s.text(1257, 577, 'Difference between conditions', 17, muted, anchor='middle')
    origin, true, predicted = (1144, 714), (1322, 613), (1280, 664)
    s.arrow(*origin, *true, '#405268', 3, 10)
    s.arrow(*origin, *predicted, purple, 3, 10)
    s.circle(*origin, 3.5, '#64748b')
    s.line(*true, *predicted, '#9b8eae', 1.5, '4 4')
    s.text(1331, 611, 'True', 19, '#405268', 700)
    s.text(1296, 682, 'Predicted', 19, purple, 700)
    s.raw('<text x="1351" y="640" font-size="26" fill="#405268" style="font-family:Georgia,serif;font-style:italic">Δz</text>')
    s.raw(f'<text x="1340" y="713" font-size="26" fill="{purple}" style="font-family:Georgia,serif;font-style:italic">Δẑ</text>')
    # Dataset-level ratio of sums, consistent with Eq. (response-error) in the paper.
    s.text(1112, 774, 'NRE =', 21, purple, 700)
    s.line(1190, 766, 1403, 766, '#5c5870', 1.2)
    def term(x, y, numerator):
        inside = 'Δẑ<tspan baseline-shift="sub" font-size="13">i</tspan> − Δz<tspan baseline-shift="sub" font-size="13">i</tspan>' if numerator else 'Δz<tspan baseline-shift="sub" font-size="13">i</tspan>'
        s.raw(f'<text x="{x}" y="{y}" text-anchor="middle" fill="#403d50" font-size="21" style="font-family:Georgia,serif">∑<tspan baseline-shift="sub" font-size="13">i</tspan> ‖{inside}‖<tspan baseline-shift="super" font-size="13">2</tspan></text>')
    term(1296.5, 754, True)
    term(1296.5, 790, False)
