"""Recreate the supplied reference's cards, framing and visual hierarchy."""
from pathlib import Path
import json,math
from svg_style import SVG, BLUE, DARK_BLUE, ORANGE, DARK_ORANGE

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets/cube_pair'
p=json.loads((ASSETS/'provenance.json').read_text())
assert (ASSETS/p['frames']['cannot_hold']['2']['file']).read_bytes() == (ASSETS/p['frames']['can_hold']['2']['file']).read_bytes()
for step in range(4):
    assert p['frames']['cannot_hold'][str(step)]['action_block'] == p['frames']['can_hold'][str(step)]['action_block']

s=SVG(1491,1063)
s.panel(11,10,1470,555,'A','Same observation and action, different futures')
s.text(1450,42,'Red traces: five-step vertical commands',16,'#6b4141',anchor='end')
# Same reference visual style, now using a real Cube history pair.
xs=[213,476,734,979,1244]
ws=[218,217,203,222,221]
headers=['History t − 2','History t − 1','Current observation','Same future action','Ground-truth future']
viewbox='57 115 126 71'

def frame(mode,index,x,y,w,h):
    s.rect(x,y,w,h,'white','#3f454b',1.5,4)
    step=[0,1,2,2,3][index]
    s.image(ASSETS/p['frames'][mode][str(step)]['file'],x+1,y+1,w-2,h-2,viewbox,r=3)

def action_strip(mode,step,x,y,w,label):
    # The five recorded vertical commands, rather than an invented robot path.
    vals=p['frames'][mode][str(step)]['action_block'][2::5]
    ww=104;xx=x+(w-ww)/2;ym=y+27
    s.text(x+w/2,y+11,label,15,'#7d2424',700,'middle')
    s.line(xx,ym,xx+ww,ym,'#bec3cc',1)
    pts=[]
    for j,v in enumerate(vals):
        left=xx+j*ww/5;right=xx+(j+1)*ww/5;yy=ym-v/.34*10
        pts.extend([(left,yy),(right,yy)])
    s.raw('<polyline points="'+' '.join(f'{a},{b}' for a,b in pts)+'" fill="none" stroke="#d73332" stroke-width="1.8"/>')

for mode,head_y,img_y,col,dark,letter,kind in [
    ('cannot_hold',69,103,BLUE,DARK_BLUE,'A','Cannot'),
    ('can_hold',345,379,ORANGE,DARK_ORANGE,'B','Can')]:
    s.rect(26,img_y+9,173,105,'#e4f1ff' if letter=='A' else '#fff0e2',r=8)
    s.text(41,img_y+44,f'Condition {letter}:',24,dark,700)
    s.lines(41,img_y+73,[f'{kind} carry','the cube'],22,gap=26,weight=700)
    for j,(x,w,label) in enumerate(zip(xs,ws,headers)):
        fill='url(#'+('blue' if letter=='A' else 'orange')+')' if j in [2,3] else 'url(#neutral)'
        s.rect(x,head_y,w,37,fill,r=6)
        s.text(x+w/2,head_y+25,label,18.5,dark if j in [2,3] else '#101216',700,'middle')
        frame(mode,j,x,img_y,w,122)
        if j in [0,1,3]:
            st={0:0,1:1,3:2}[j]
            strip_x,strip_w = (xs[j]+ws[j]-57,150) if j in [0,1] else (x,w)
            action_strip(mode,st,strip_x,img_y+127,strip_w,'Lift command' if j in [0,3] else 'Reverse command')
        if j in [1,4]:
            s.text(x+w/2,img_y+119,'Cube stays' if letter=='A' else 'Cube lifts',16,'white',700,'middle',paint_order='stroke',stroke='#1e2733',stroke_width=3)

    for j in range(4):
        s.arrow(xs[j]+ws[j]+9,img_y+64,xs[j+1]-9,img_y+64,'#303940',2.4,8)
s.line(26,329,1465,329,'#bdc5d0',1.3,'7 6')
for x,w,label in [(734,203,'Same current observation'),(979,222,'Same future action')]:
    s.brace(x,274,w,1);s.brace(x,340,w,-1)
    s.text(x+w/2,313,label,20,weight=700,anchor='middle')

# Same reference B styling, widened after omitting C.
s.raw('<g transform="translate(0,78)">')
s.panel(11,501,1470,364,'B','How evaluation works')
for mode,y,col,dark,letter in [('cannot_hold',580,BLUE,DARK_BLUE,'A'),('can_hold',723,ORANGE,DARK_ORANGE,'B')]:
    s.rect(27,y,190,119,'#f8fbff' if letter=='A' else '#fffaf6',col,1.2,8)
    s.rect(28,y+1,188,30,'url(#blue)' if letter=='A' else 'url(#orange)',r=7)
    s.text(122,y+23,f'History {letter}',20,dark,700,'middle')
    for j,st in enumerate([0,1,2]):
        xx=39+j*47;yy=y+44+(2-j)*3
        s.rect(xx-2,yy-2,62,66,'#cbd0d6','#717986',.8,3)
        s.image(ASSETS/p['frames'][mode][str(st)]['file'],xx,yy,58,62,viewbox)
    s.text(122,y+136,'Observations + actions',16,'#515966',anchor='middle')
    sy=y+60
    s.line(219,sy,273,sy,col,2.2)
    ty=650 if letter=='A' else 772
    s.line(273,sy,273,ty,col,2.2);s.arrow(273,ty,333,ty,col,2.2,8)

s.text(420,575,'Same observation',18,'#3d4652',anchor='middle')
s.text(420,597,'and future actions',18,'#3d4652',anchor='middle')
s.arrow(420,605,420,617,'#6b7380',1.8,7)
s.rect(339,620,162,185,'url(#model)','#8d95a2',1.8,12)
s.lines(420,666,['Frozen','latent world','model'],23,gap=26,weight=700,anchor='middle')
# Snowflake, matching the reference's visual frozen-weight cue.
for a in [0,math.pi/3,2*math.pi/3]:
    dx,dy=17*math.cos(a),17*math.sin(a)
    s.line(420-dx,765-dy,420+dx,765+dy,'#438fb9',2)
    for sign in [-1,1]:
        xx,yy=420+sign*dx,765+sign*dy
        ux,uy=sign*math.cos(a),sign*math.sin(a)
        for side in [-1,1]:s.line(xx,yy,xx-7*ux+side*4*uy,yy-7*uy-side*4*ux,'#438fb9',1.7)
s.text(420,836,'Separate calls · shared weights',16,'#515966',anchor='middle')

for y,col,dark,letter in [(589,BLUE,DARK_BLUE,'A'),(732,ORANGE,DARK_ORANGE,'B')]:
    s.arrow(504,y+61,615,y+61,col,2.3,9)
    s.rect(620,y,175,111,'#f8fbff' if letter=='A' else '#fffaf6',col,1.2,8)
    s.rect(621,y+1,173,32,'url(#blue)' if letter=='A' else 'url(#orange)',r=7)
    s.text(707.5,y+23,f'Prediction from {letter}',18,dark,700,'middle')
    # Symbolic latent output, never a decoded image or invented feature values.
    s.raw(f'<text x="696" y="{y+88}" style="font-family:Georgia,serif;font-style:italic" font-size="43" fill="{dark}">z<tspan font-size="22" baseline-shift="sub">{letter}</tspan></text>')
    s.raw(f'<path d="M 696 {y+53} L 706 {y+46} L 716 {y+53}" fill="none" stroke="{dark}" stroke-width="1.7"/>')

# Row and column comparisons are corrected relative to the supplied sketch.
mx,my,cw,ch=1050,632,150,96
s.brace(mx,579,2*cw,-1)
s.text(mx+cw,548,'Future discrimination',19,weight=700,anchor='middle')
for j,(letter,col,dark) in enumerate([('A',BLUE,DARK_BLUE),('B',ORANGE,DARK_ORANGE)]):
    s.rect(mx+j*cw,585,cw-4,42,'url(#blue)' if j==0 else 'url(#orange)',r=8)
    s.text(mx+(j+.5)*cw-2,612,f'True future {letter}',20,dark,700,'middle')
    s.rect(mx-94,my+j*ch,88,ch-4,'#e7f2ff' if j==0 else '#fff0e2',r=7)
    s.lines(mx-50,my+j*ch+40,['Pred','from '+letter],19,gap=24,fill=dark,weight=700,anchor='middle')
for i in range(2):
    for j in range(2):
        s.rect(mx+j*cw,my+i*ch,cw-4,ch-4,'url(#blue)' if i==j==0 else 'url(#orange)' if i==j==1 else '#ebecef',BLUE if i==j==0 else ORANGE if i==j==1 else '#d5d8de',1.5)
        s.raw(f'<text x="{mx+(j+.5)*cw-2}" y="{my+i*ch+60}" text-anchor="middle" style="font-family:Georgia,serif;font-style:italic" font-size="34" fill="#20242a">d<tspan font-size="23" baseline-shift="sub">{"AB"[i]}{"AB"[j]}</tspan></text>')
s.raw('<path d="M 946 632 Q 928 632 928 650 L 928 708 Q 928 723 914 728 Q 928 733 928 748 L 928 810 Q 928 828 946 828" fill="none" stroke="#171a1f" stroke-width="2"/>')
s.text(902,730,'History benefit',19,weight=700,anchor='middle',transform='rotate(-90 902 730)')
s.text(1200,851,'Latent-space MSE · lower is better',16,'#515966',anchor='middle')

# Preserve the reference footer layout; the environment thumbnails are real observations.
s.rect(12,878,1468,96,'white','#69717d',1.2,11)
s.text(61,933,'ContextWorld:',27,weight=700)
s.text(254,933,'9 tasks across 4 environments',22)
s.line(555,900,555,956,'#7c8490',1.5)
thumbs=[('TwoRoom',HERE/'assets/environments/tworoom.jpg',615),('PushT',HERE/'assets/pusher_pair/pusher_low_gain_step10.jpg',832),('Reacher',HERE/'assets/environments/reacher.jpg',1045),('Cube',HERE/'assets/environments/cube.jpg',1260)]
for title,path,x in thumbs:
    s.rect(x,889,76,72,'#e8e9eb','#747c87',1,5)
    s.image(path,x+2,891,72,68)
    s.text(x+91,932,title,18,weight=700)
s.raw('</g>')
s.save(HERE/'benchmark_overview.svg')
(HERE/'overview_crop.json').write_text(json.dumps({'source':'assets/cube_pair/provenance.json','source_type':'Real Cube paired observations','source_viewbox':[57,115,126,71],'pair_id':p['pair_id'],'shared_current_and_action_images':True,'reference_dimensions':[1491,1055],'output_dimensions':[1491,1063],'layout':'Reference A and B styling, with recorded history-action strips; C omitted.'},indent=2)+'\n')
print('Reference-style benchmark overview saved')
