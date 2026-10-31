"""Reference-style overview with native high-resolution Cube renderings."""
from pathlib import Path
import json, math
from svg_style import SVG, BLUE, DARK_BLUE, ORANGE, DARK_ORANGE

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets/cube_pair_hd'
p=json.loads((ASSETS/'provenance.json').read_text())
raw=json.loads((HERE/'assets/cube_pair/provenance.json').read_text())
assert (ASSETS/p['frames']['cannot_hold']['2']['file']).read_bytes()==(ASSETS/p['frames']['can_hold']['2']['file']).read_bytes()
for t in range(4):
    assert raw['frames']['cannot_hold'][str(t)]['action_block']==raw['frames']['can_hold'][str(t)]['action_block']
s=SVG(1491,954)
s.panel(11,10,1470,477,'A','Same observation and action, different futures')
s.text(1450,43,'Cube carry-rule example',20,'#525b69',anchor='end')
xs=[213,558,902,1247];w=218;h=122
headers=['History t − 2','History t − 1','Current observation','Ground-truth future']
# Query actions are between state frames, never represented as an observation.
s.raw('<rect x="1132" y="103" width="105" height="352" rx="8" fill="#f5f7fa" stroke="#b6bfcb" stroke-width="1.2" stroke-dasharray="5 4"/>')
s.lines(1184.5,259,['Same future','action'],18,gap=23,weight=700,anchor='middle')

def frame(mode,t,x,y):
    s.rect(x,y,w,h,'white','#4f5359',1.3,4)
    s.image(ASSETS/p['frames'][mode][str(t)]['file'],x+1,y+1,w-2,h-2,r=3)
    if t in [1,3]:
        # Reference height and arrow derive from the rendered cube's actual bounds.
        lo=p['frames']['cannot_hold'][str(t)]['cube_bbox']
        hi=p['frames'][mode][str(t)]['cube_bbox']
        scale=min((w-2)/1600,(h-2)/900)
        ox=x+(w-1600*scale)/2;oy=y+(h-900*scale)/2
        bottom=oy+lo[3]*scale;current=oy+hi[3]*scale
        left=ox+lo[0]*scale;right=ox+max(lo[2],hi[2])*scale
        s.line(left-10,bottom+2,right+12,bottom+2,'#5d6670',1.5,'3 2')
        if mode=='can_hold':
            s.arrow(left-7,bottom+1,left-7,current+1,ORANGE,2.0,3.2)
        label='Cube stays' if mode=='cannot_hold' else 'Cube lifts'
        dark=DARK_BLUE if mode=='cannot_hold' else DARK_ORANGE
        # Outcome labels sit below the image, leaving the visible cube unobscured.
        s.text(x+w/2,y+h+23,label,21,dark,700,'middle')

def action_between(x1,x2,y,reverse=False):
    center=(x1+x2)/2
    label='Reverse' if reverse else 'Lift'
    s.text(center-11,y+40,label,20,'#ba3030',700,'middle')
    ax=center+46 if reverse else center+36
    s.arrow(ax,y+22 if reverse else y+43,ax,y+43 if reverse else y+22,'#d23835',2.3,6)
    s.arrow(x1+12,y+64,x2-12,y+64,'#3e454d',2.4,8)

for mode,header_y,y,letter,dark,description in [
    ('cannot_hold',69,103,'A',DARK_BLUE,['Cannot carry','the cube']),
    ('can_hold',301,333,'B',DARK_ORANGE,['Can carry','the cube'])]:
    s.rect(26,y+9,173,105,'#e4f1ff' if letter=='A' else '#fff0e2',r=8)
    s.text(41,y+44,f'Condition {letter}:',24,dark,700)
    s.lines(41,y+73,description,22,gap=26,weight=700)
    for t,(x,label) in enumerate(zip(xs,headers)):
        fill='url(#blue)' if t==2 and letter=='A' else 'url(#orange)' if t==2 else 'url(#neutral)'
        s.rect(x,header_y,w,37,fill,r=6)
        s.text(x+w/2,header_y+25,label,18.5,dark if t==2 else '#101216',700,'middle')
        frame(mode,t,x,y)
    for j in range(3):action_between(xs[j]+w,xs[j+1],y,reverse=j==1)
s.line(26,285,1125,285,'#bdc5d0',1.3,'7 6')
s.line(1242,285,1465,285,'#bdc5d0',1.3,'7 6')
s.brace(xs[2],230,w,1);s.brace(xs[2],296,w,-1)
s.text(xs[2]+w/2,269,'Same current observation',20,weight=700,anchor='middle')

s.panel(11,501,1470,334,'B','How evaluation works')
# Compact, connected flow: histories -> shared model -> predictions -> errors.
for mode,y,col,dark,letter in [('cannot_hold',575,BLUE,DARK_BLUE,'A'),('can_hold',701,ORANGE,DARK_ORANGE,'B')]:
    s.rect(40,y,210,108,'#f8fbff' if letter=='A' else '#fffaf6',col,1.2,8)
    s.rect(41,y+1,208,29,'url(#blue)' if letter=='A' else 'url(#orange)',r=7)
    s.text(145,y+23,f'History {letter}',20,dark,700,'middle')
    for j,t in enumerate([0,1,2]):
        xx=51+j*55;yy=y+39+(2-j)*3
        s.rect(xx-2,yy-2,82,57,'#d4d8de','#747d88',.8,3)
        s.image(ASSETS/p['frames'][mode][str(t)]['file'],xx,yy,78,53)
    s.text(145,y+123,'Observations + actions',16,'#515966',anchor='middle')
    s.arrow(253,y+63,300,y+63,col,2.2,8)
s.text(400,566,'Same observation and future actions',17,'#46505d',anchor='middle')
s.arrow(400,575,400,601,'#6b7380',1.8,7)
s.rect(305,608,190,174,'url(#model)','#8d95a2',1.8,12)
s.lines(400,650,['Frozen','latent world','model'],24,gap=27,weight=700,anchor='middle')
for a in [0,math.pi/3,2*math.pi/3]:
    dx,dy=16*math.cos(a),16*math.sin(a)
    s.line(400-dx,747-dy,400+dx,747+dy,'#438fb9',2)
    for sign in [-1,1]:
        xx,yy=400+sign*dx,747+sign*dy;ux,uy=sign*math.cos(a),sign*math.sin(a)
        for side in [-1,1]:s.line(xx,yy,xx-7*ux+side*4*uy,yy-7*uy-side*4*ux,'#438fb9',1.7)
s.text(400,813,'Separate calls · shared weights',16,'#515966',anchor='middle')
for y,col,dark,letter in [(583,BLUE,DARK_BLUE,'A'),(709,ORANGE,DARK_ORANGE,'B')]:
    s.arrow(499,y+55,555,y+55,col,2.3,9)
    s.rect(560,y,190,100,'#f8fbff' if letter=='A' else '#fffaf6',col,1.2,8)
    s.rect(561,y+1,188,30,'url(#blue)' if letter=='A' else 'url(#orange)',r=7)
    s.text(655,y+23,f'Prediction from {letter}',18,dark,700,'middle')
    s.raw(f'<text x="642" y="{y+80}" style="font-family:Georgia,serif;font-style:italic" font-size="40" fill="{dark}">z<tspan font-size="21" baseline-shift="sub">{letter}</tspan></text>')
    s.raw(f'<path d="M 642 {y+48} L 651 {y+42} L 660 {y+48}" fill="none" stroke="{dark}" stroke-width="1.7"/>')
s.arrow(776,704,858,704,'#66707b',1.8,8)
s.text(817,691,'MSE',18,'#515966',anchor='middle')
mx,my,cw,ch=1030,618,180,84
s.brace(mx,575,2*cw,-1)
s.text(mx+cw,544,'Future discrimination',20,weight=700,anchor='middle')
for j,(letter,dark) in enumerate([('A',DARK_BLUE),('B',DARK_ORANGE)]):
    s.rect(mx+j*cw,579,cw-4,35,'url(#blue)' if j==0 else 'url(#orange)',r=7)
    s.text(mx+(j+.5)*cw-2,603,f'True future {letter}',20,dark,700,'middle')
    s.rect(mx-96,my+j*ch,90,ch-4,'#e7f2ff' if j==0 else '#fff0e2',r=7)
    s.lines(mx-51,my+j*ch+34,['Pred','from '+letter],19,gap=24,fill=dark,weight=700,anchor='middle')
for i in range(2):
    for j in range(2):
        s.rect(mx+j*cw,my+i*ch,cw-4,ch-4,'url(#blue)' if i==j==0 else 'url(#orange)' if i==j==1 else '#ebecef',BLUE if i==j==0 else ORANGE if i==j==1 else '#d5d8de',1.5)
        s.raw(f'<text x="{mx+(j+.5)*cw-2}" y="{my+i*ch+52}" text-anchor="middle" style="font-family:Georgia,serif;font-style:italic" font-size="33" fill="#20242a">d<tspan font-size="22" baseline-shift="sub">{"AB"[i]}{"AB"[j]}</tspan></text>')
s.raw('<path d="M 924 618 Q 907 618 907 634 L 907 682 Q 907 695 893 702 Q 907 709 907 722 L 907 770 Q 907 786 924 786" fill="none" stroke="#171a1f" stroke-width="2"/>')
s.text(881,702,'History benefit',19,weight=700,anchor='middle',transform='rotate(-90 881 702)')
s.text(mx+cw,815,'Latent-space MSE · lower is better',16,'#515966',anchor='middle')

s.rect(12,848,1468,96,'white','#69717d',1.2,11)
s.text(61,903,'ContextWorld:',27,weight=700)
s.text(254,903,'9 tasks across 4 environments',22)
s.line(555,870,555,926,'#7c8490',1.5)
thumbs=[('TwoRoom',HERE/'assets/environments/tworoom.jpg',615),('PushT',HERE/'assets/pusher_pair/pusher_low_gain_step10.jpg',832),('Reacher',HERE/'assets/environments/reacher.jpg',1045),('Cube',ASSETS/'can_hold_t0.png',1260)]
for title,path,x in thumbs:
    s.rect(x,859,76,72,'#e8e9eb','#747c87',1,5)
    s.image(path,x+2,861,72,68)
    s.text(x+91,902,title,18,weight=700)
s.save(HERE/'benchmark_overview.svg')
(HERE/'overview_crop.json').write_text(json.dumps({'source':'assets/cube_pair_hd/provenance.json','source_type':'Native high-resolution rendering of the existing Cube pair for visualization; original model inputs unchanged','pair_id':p['pair_id'],'output_dimensions':[1491,954],'layout':'Four state frames, actions only between states; compact B; C omitted.'},indent=2)+'\n')
print('Reference-style overview with high-resolution Cube views saved')
