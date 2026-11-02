"""Paired-evaluation layouts: original matrix and three aligned geometric views."""
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



# All three schematics share these four conceptual states. Their locations are
# illustrative, not measured embeddings; the NRE arrows use the same differences.
SCHEMATIC_STATES = {
    'pred_A': (92, 151), 'true_A': (221, 151),
    'pred_B': (56, 280), 'true_B': (211, 310),
}
RESPONSE_PANEL_EXTRA_HEIGHT = 228


def draw_response_panel(s, scene, panel, formula, check, prediction_images=False):
    """Reproduce the supplied three-card design, with real history observations."""
    blue, orange, purple = '#0874db', '#e27817', '#6456ba'
    dark_blue, dark_orange = '#064b9a', '#a94b0a'
    ink, muted = '#122840', '#64758a'
    s.raw('<g id="paired-evaluation-panel">')
    s.raw('<defs>'
          '<linearGradient id="protocol-model" x2="1" y2="1"><stop stop-color="#f0f6ff"/><stop offset="1" stop-color="#e5e6f4"/></linearGradient>'
          '<linearGradient id="metric-blue" x2="0" y2="1"><stop stop-color="#eaf5ff"/><stop offset="1" stop-color="#dcecff"/></linearGradient>'
          '<linearGradient id="metric-orange" x2="0" y2="1"><stop stop-color="#fff4e9"/><stop offset="1" stop-color="#ffeddf"/></linearGradient>'
          '<linearGradient id="metric-purple" x2="0" y2="1"><stop stop-color="#f3f0ff"/><stop offset="1" stop-color="#e9e5fc"/></linearGradient>'
          '</defs>')
    panel(20, 471, 1408, 574, 'B', 'Paired evaluation protocol', 32)

    def latent(x, y, letter, predicted=False, size=29, color=None, anchor='middle'):
        color = color or (dark_blue if letter == 'A' else dark_orange)
        s.raw(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}" style="font-family:Georgia,serif;font-style:italic">z<tspan baseline-shift="sub" font-size="{size*.64}">{letter}</tspan></text>')
        if predicted:
            # Anchor-aware hat over z, with the subscript outside its extent.
            xx = x-size*.20 if anchor == 'middle' else x+size*.18
            s.raw(f'<path d="M {xx-size*.18} {y-size*.78} l {size*.18} {-size*.13} l {size*.18} {size*.13}" fill="none" stroke="{color}" stroke-width="1.6"/>')

    def route(x, y, xx, yy, color):
        mid=(x+xx)/2
        s.raw(f'<path d="M {x} {y} H {mid-5} Q {mid} {y} {mid} {y+5 if yy>y else y-5} V {yy-5 if yy>y else yy+5} Q {mid} {yy} {mid+5} {yy} H {xx}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        s.arrow(xx-6,yy,xx,yy,color,1.8,6)

    # The left-hand flow uses source images, never generated scene illustrations.
    for mode, yy, letter, col, dark in [
        ('cannot_hold', 623, 'A', blue, dark_blue),
        ('can_hold', 798, 'B', orange, dark_orange),
    ]:
        s.rect(36, yy, 145, 105, '#fbfdff' if letter=='A' else '#fffdfb', '#a7cff4' if letter=='A' else '#edc6a5', 1.1, 11)
        s.rect(38, yy+2, 141, 28, 'url(#metric-blue)' if letter=='A' else 'url(#metric-orange)', r=9)
        s.text(108.5, yy+23, 'History '+letter, 21, dark, 700, 'middle')
        for t, xx in [(0,45),(1,88),(2,138)]:
            scene(mode,t,xx,yy+38,34,40)
        s.text(130,yy+64,'…',16,muted,anchor='middle')
        s.text(108.5,yy+94,'observations + actions',13,muted,anchor='middle')
        route(182,yy+52.5,207,697 if letter=='A' else 829,col)
    s.rect(208, 665, 110, 196, 'url(#protocol-model)', '#a6b7d1', 1.2, 12)
    for angle in [0,math.pi/3,2*math.pi/3]:
        ux,uy=math.cos(angle),math.sin(angle)
        s.line(263-15*ux,699-15*uy,263+15*ux,699+15*uy,'#589bc4',2)
        for sign in [-1,1]:
            x,y=263+sign*15*ux,699+sign*15*uy
            for side in [-1,1]:s.line(x,y,x-sign*5*ux+side*3*uy,y-sign*5*uy-side*3*ux,'#589bc4',1.5)
    s.lines(263,747,['Frozen','latent world','model'],19,gap=26,fill=ink,weight=700,anchor='middle')
    s.text(263,839,'weights fixed',14.5,muted,anchor='middle')
    for mode,yy,letter,col,dark in [('cannot_hold',631,'A',blue,dark_blue),('can_hold',806,'B',orange,dark_orange)]:
        route(319,697 if letter=='A' else 829,345,yy+44.5,col)
        s.rect(346,yy,91,89,'#fbfdff' if letter=='A' else '#fffdfb','#a7cff4' if letter=='A' else '#edc6a5',1.1,10)
        s.rect(348,yy+2,87,26,'url(#metric-blue)' if letter=='A' else 'url(#metric-orange)',r=8)
        s.text(391.5,yy+22,'Prediction '+letter,16,dark,700,'middle')
        if prediction_images:
            scene(mode,3,352,yy+35,43,34)
            s.text(373.5,yy+82,'Reference',11.5,muted,anchor='middle')
            latent(416,yy+63,letter,True,28,dark)
        else:
            latent(391.5,yy+72,letter,True,39,dark)
    s.line(450,550,450,944,'#e0e7ef',1)

    # Three equal cards, matching the supplied mockup's information hierarchy.
    card_y, card_w, card_h = 545, 308, 408
    for idx,(xx,title,col,tint,border) in enumerate([
        (466,'Future discrimination',blue,'blue','#c7dff6'),
        (786,'History benefit',orange,'orange','#e6d9cc'),
        (1106,'Response fidelity',purple,'purple','#d8d3ec'),
    ],1):
        s.rect(xx,card_y,card_w,card_h,'#fcfdff',border,1.1,14)
        s.rect(xx+8,card_y+8,card_w-16,38,'url(#metric-'+tint+')',r=10)
        s.circle(xx+30,card_y+27,13,col)
        s.text(xx+30,card_y+34,str(idx),19,'white',700,'middle')
        s.text(xx+52,card_y+34,title,22,ink,700)

    def dist_label(x,y,first,second,color=muted):
        s.raw(f'<text x="{x}" y="{y}" fill="{color}" font-size="19" text-anchor="middle" style="font-family:Georgia,serif;font-style:italic;paint-order:stroke;stroke:#fcfdff;stroke-width:4;stroke-linejoin:round">d<tspan baseline-shift="sub" font-size="12">{first}{second}</tspan></text>')

    for xx,kind in [(466,'prediction'),(786,'future')]:
        s.raw(f'<g id="comparison-fix-{kind}">')
        caption = ['Hold the prediction fixed;', 'compare the two true futures.'] if kind=='prediction' else ['Hold the true future fixed;', 'compare predictions from A and B.']
        s.lines(xx+154,card_y+73,caption,16.5,gap=22,fill=ink,anchor='middle')
        points={k:(xx+v[0],card_y+v[1]) for k,v in SCHEMATIC_STATES.items()}
        focus=points['pred_A' if kind=='prediction' else 'true_A']
        s.circle(*focus,26,'#e8f3ff')
        # Both views contain the same four edges; emphasis changes the fixed node.
        active={'AA','AB'} if kind=='prediction' else {'AA','BA'}
        for ij in ['AB','BA','BB','AA']:
            a,b=points['pred_'+ij[0]],points['true_'+ij[1]]
            matched=ij[0]==ij[1]
            col=(blue if ij=='AA' else orange) if matched else '#8796a8'
            if ij not in active:col=('#b7d0e9' if ij=='AA' else '#e8ccb3') if matched else '#d4dce5'
            s.line(*a,*b,col,2.7 if ij in active else 1.6,None if matched else '5 5')
            if ij=='AA':lp=((a[0]+b[0])/2,(a[1]+b[1])/2-10)
            elif ij=='BB':lp=((a[0]+b[0])/2,(a[1]+b[1])/2+20)
            elif ij=='AB':lp=(a[0]+.64*(b[0]-a[0])+16,a[1]+.64*(b[1]-a[1])-4)
            else:lp=(a[0]+.35*(b[0]-a[0])-16,a[1]+.35*(b[1]-a[1])-3)
            dist_label(*lp,*ij,ink if ij in active else '#8996a5')
        for key,(x,y) in points.items():
            letter=key[-1];pred=key.startswith('pred');col=blue if letter=='A' else orange
            s.circle(x,y,8.5,col if pred else 'white',col,2.3)
            label_y=y-18 if letter=='A' else y+31
            if key == 'true_B':
                latent(x+20,y-8,letter,pred,25,anchor='start')
            else:
                latent(x,label_y,letter,pred,25)
        s.rect(xx+12,card_y+340,card_w-24,56,'url(#metric-'+('blue' if kind=='prediction' else 'orange')+')',r=8)
        a,b=('AB','BA') if kind=='prediction' else ('BA','AB')
        formula(xx+25,card_y+377,'AA',a,21)
        s.text(xx+155,card_y+377,';',19,muted,anchor='middle')
        formula(xx+176,card_y+377,'BB',b,21)
        s.raw('</g>')

    # Rotate both difference vectors identically; their magnitudes and relative
    # directions therefore agree with the four-point schematics.
    s.lines(1260,card_y+73,['The predicted response should', 'match the true response.'],16.5,gap=22,fill=ink,anchor='middle')
    angle=-3*math.pi/4
    def response(predicted):
        prefix='pred_' if predicted else 'true_'
        a,b=SCHEMATIC_STATES[prefix+'A'],SCHEMATIC_STATES[prefix+'B']
        dx,dy=b[0]-a[0],b[1]-a[1]
        return ((dx*math.cos(angle)-dy*math.sin(angle))*1.5,(dx*math.sin(angle)+dy*math.cos(angle))*1.5)
    origin=(1142,card_y+297)
    true=tuple(a+b for a,b in zip(origin,response(False)))
    pred=tuple(a+b for a,b in zip(origin,response(True)))
    s.line(*true,*pred,'#a0aabc',1.8,'5 4')
    s.arrow(*origin,*true,blue,3,9)
    s.arrow(*origin,*pred,orange,3,9)
    s.circle(*origin,3.5,'#4c6078')
    def delta_label(x,y,predicted):
        color=dark_orange if predicted else dark_blue
        s.raw(f'<text x="{x}" y="{y}" fill="{color}" font-size="28" style="font-family:Georgia,serif;font-style:italic">Δ{"ẑ" if predicted else "z"}</text>')
        s.text(x,y+22,'predicted' if predicted else 'true',16,color,700)
    delta_label(true[0]+13,true[1]-2,False)
    delta_label(pred[0]+13,pred[1]+7,True)
    s.rect(1118,card_y+340,284,56,'url(#metric-purple)',r=8)
    # Compact ratio of sums, retaining dataset-level NRE rather than one-pair NRE.
    s.text(1131,card_y+377,'NRE =',18,ink,700)
    s.line(1192,card_y+368,1386,card_y+368,'#34415b',1)
    def term(y,numerator):
        inside='Δẑ<tspan baseline-shift="sub" font-size="10">i</tspan> − Δz<tspan baseline-shift="sub" font-size="10">i</tspan>' if numerator else 'Δz<tspan baseline-shift="sub" font-size="10">i</tspan>'
        s.raw(f'<text x="1289" y="{y}" text-anchor="middle" fill="{ink}" font-size="20" style="font-family:Georgia,serif">∑<tspan baseline-shift="sub" font-size="10">i</tspan> ‖{inside}‖<tspan baseline-shift="super" font-size="10">2</tspan></text>')
    term(card_y+360,True);term(card_y+391,False)

    # Short, shared key: color denotes condition, fill distinguishes prediction/target.
    s.rect(34,970,1380,59,'#fcfdff','#dce4ed',1,12)
    s.text(51,1007,'Key',18,ink,700)
    s.circle(113,1001,7,blue,blue,2);s.circle(134,1001,7,orange,orange,2)
    s.text(153,1007,'Prediction',17,ink)
    s.circle(272,1001,7,'white',blue,2);s.circle(293,1001,7,'white',orange,2)
    s.text(312,1007,'True future',17,ink)
    s.line(425,983,425,1016,'#dbe2ec',1)
    s.line(445,1001,476,1001,blue,2.5);s.text(489,1007,'Matched condition',17,ink)
    s.line(663,1001,698,1001,'#8796a8',2,'5 4');s.text(711,1007,'Other condition',17,ink)
    s.line(871,983,871,1016,'#dbe2ec',1)
    s.circle(902,1001,13,'#e8f3ff');s.circle(902,1001,6,blue)
    s.text(925,1007,'Fixed for comparison',17,ink)
    s.text(1396,1007,'Shorter distance → lower error',17,muted,anchor='end')
    s.raw('</g>')
