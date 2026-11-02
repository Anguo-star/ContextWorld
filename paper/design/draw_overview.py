"""Editable replica of the supplied main-figure schematic. Scene art is illustrative."""
from pathlib import Path
import json, math
from svg_style import SVG, data_uri
import base64
HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'assets/reference/main_figure_design.png'
BLUE='#0094ff'; ORANGE='#ff780a'; DARK_BLUE='#0044b5'; DARK_ORANGE='#c52e00'; INK='#111620'; OUTLINE='#8795aa'
s=SVG(1448,1086)
s.raw(f'<defs><image id="reference-art" href="{data_uri(REFERENCE)}" width="1448" height="1086"/></defs>')
regular=base64.b64encode((HERE/'assets/fonts/RobotoCondensed-Regular.ttf').read_bytes()).decode()
bold=base64.b64encode((HERE/'assets/fonts/RobotoCondensed-Bold.ttf').read_bytes()).decode()
s.raw(f'<style>@font-face{{font-family:ReferenceNarrow;src:url(data:font/ttf;base64,{regular});font-weight:400}}@font-face{{font-family:ReferenceNarrow;src:url(data:font/ttf;base64,{bold});font-weight:700}}</style>')
# Layout coordinates follow the supplied 1448 x 1086 reference, not an inferred layout.
s.raw('<defs><linearGradient id="current-bg" x2="0" y2="1"><stop stop-color="#eef8ff"/><stop offset="1" stop-color="#def2fe"/></linearGradient><linearGradient id="card-bg" x2="1" y2="1"><stop stop-color="#f3f6fb"/><stop offset="1" stop-color="#eaf0f8"/></linearGradient></defs>')
def panel(x,y,w,h,letter,title,size=32):
    s.rect(x,y,w,h,'white',OUTLINE,1.15,14)
    s.rect(x,y,54,53,'url(#badge)',r=11)
    s.text(x+27,y+38,letter,34,'white',700,'middle')
    s.text(x+69,y+37,title,size,weight=700)
def brace(x,y,h):
    s.raw(f'<path d="M {x} {y} Q {x+10} {y} {x+10} {y+20} L {x+10} {y+h/2-18} Q {x+10} {y+h/2-3} {x+20} {y+h/2} Q {x+10} {y+h/2+3} {x+10} {y+h/2+18} L {x+10} {y+h-20} Q {x+10} {y+h} {x} {y+h}" fill="none" stroke="#14191e" stroke-width="1.8"/>')
def formula(x,y,first,second,size=28):
    s.raw(f'<text x="{x}" y="{y}" font-size="{size}" style="font-family:Georgia,serif;font-style:italic"><tspan fill="{DARK_BLUE if first=="AA" else DARK_ORANGE}">d<tspan baseline-shift="sub" font-size="{size*.64}">{first}</tspan></tspan><tspan> &lt; </tspan><tspan>d<tspan baseline-shift="sub" font-size="{size*.64}">{second}</tspan></tspan></text>')
def check(x,y):
    s.raw(f'<path d="M {x-7} {y} l 6 8 l 12 -20" fill="none" stroke="#21842c" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/>')
# Reuse the supplied illustration assets, not generated experimental observations.
# Both rows deliberately reuse the same current-observation crop.
SCENES={
    'cannot_hold':[(223,111,179,109),(511,111,177,109),(801,113,175,109),(1123,112,180,109)],
    'can_hold':[(223,111,179,109),(511,307,177,108),(801,113,175,109),(1123,307,180,108)],
}
def reference_crop(box,x,y,w,h,r=3):
    ident=f'illustration{len(s.parts)}'
    s.raw(f'<defs><clipPath id="{ident}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath></defs>')
    viewbox=' '.join(map(str,box))
    s.raw(f'<g clip-path="url(#{ident})"><svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{viewbox}" preserveAspectRatio="xMidYMid slice" overflow="hidden"><use href="#reference-art"/></svg></g>')
def scene(mode,t,x,y,w=181,h=112):
    s.rect(x,y,w,h,'#eef1f6','#8f99a8',1.2,4)
    box=SCENES[mode][t]
    if w<100:
        box=([53,115,207][t],721 if mode=='can_hold' and t==1 else 580,57,49)
    reference_crop(box,x+1,y+1,w-2,h-2)

panel(20,15,1408,442,'A','Same current observation and future actions, different futures',32)
s.text(1416,47,'Cube carry-rule schematic',22,'#606b7b',anchor='end')
s.rect(787,73,203,362,'url(#current-bg)','#b3dcfc',.8,10)
s.line(29,258,1416,258,'#ccd5df',1.3,'4 4')
brace(996,113,290);s.rect(1016,202,94,116,'url(#card-bg)',r=10)
s.lines(1063,245,['Identical','current','observation'],21,gap=23,weight=700,anchor='middle',style='font-family:ReferenceNarrow')
brace(1317,111,293);s.rect(1333,110,86,294,'url(#card-bg)',r=10)
s.lines(1376,245,['Different','future','outcomes'],21,gap=24,weight=700,anchor='middle',style='font-family:ReferenceNarrow')
for mode,y,letter,dark,desc in [('cannot_hold',110,'A',DARK_BLUE,['Cannot carry','the cube']),('can_hold',305,'B',DARK_ORANGE,['Can carry','the cube'])]:
    s.rect(36,y-1,175,111,'#e2f2ff' if letter=='A' else '#fff0e4',r=11)
    s.text(48,y+36,f'Condition {letter}:',26,dark,700)
    s.lines(48,y+64,desc,24,gap=28,weight=700)
    for t,x,title in [(0,222,'History t − 2'),(1,510,'History t − 1'),(2,800,'Current observation'),(3,1122,'Ground-truth future')]:
        hy=y-36
        s.rect(x-3,hy,187,37,'url(#neutral)' if t!=2 else '#e1f3ff',r=6)
        s.text(x+90.5,hy+25,title,20,DARK_BLUE if t==2 else INK,700,'middle')
        scene(mode,t,x,y)
        if t in [1,3]:s.text(x+90.5,y+132,'Not lifted' if letter=='A' else 'Lifted',23,dark,700,'middle')
    for x1,x2,label in [(418,499,'Lift'),(702,782,'Reverse'),(1017,1109,'Lift')]:
        s.text((x1+x2)/2,y+32,label,22,'#cf0000',700,'middle')
        s.arrow(x1,y+48 if letter=='A' else y+49,x2,y+48 if letter=='A' else y+49,'#344054',2,9)

panel(20,471,1408,346,'B','Paired evaluation protocol',32)
for mode,y,col,dark,letter in [('cannot_hold',537,BLUE,DARK_BLUE,'A'),('can_hold',678,ORANGE,DARK_ORANGE,'B')]:
    s.rect(37,y,247,125,'#f1f9ff' if letter=='A' else '#fff7f0',col,1.2,10)
    s.rect(39,y+2,243,33,'url(#blue)' if letter=='A' else 'url(#orange)',r=8)
    s.text(160.5,y+25,'History '+letter,24,dark,700,'middle')
    for t,xx in [(0,52),(1,114),(2,206)]:scene(mode,t,xx,y+42,59,52)
    s.text(190,y+72,'…',23,anchor='middle')
    s.text(160.5,y+115,'observations + actions',19,'#536172',anchor='middle')
    s.arrow(287,y+65,331,y+65,col,2,9)
s.text(427,543,'Shared future actions',18,'#536172',anchor='middle')
s.arrow(427,548,427,565,'#536172',1.5,6)
s.rect(334,568,186,181,'url(#model)',OUTLINE,1.3,11)
s.lines(427,619,['Frozen','latent world','model'],26,gap=30,weight=700,anchor='middle')
# Snowflake: vector icon rather than a font glyph.
for angle in [0,math.pi/3,2*math.pi/3]:
    ux,uy=math.cos(angle),math.sin(angle)
    s.line(427-18*ux,710-18*uy,427+18*ux,710+18*uy,'#348fbc',2.2)
    for sign in [-1,1]:
        xx,yy=427+sign*18*ux,710+sign*18*uy
        for side in [-1,1]:s.line(xx,yy,xx-sign*7*ux+side*4*uy,yy-sign*7*uy-side*4*ux,'#348fbc',2)
s.text(427,778,'Separate calls · shared weights',17,'#667488',anchor='middle')
for y,col,dark,letter in [(545,BLUE,DARK_BLUE,'A'),(681,ORANGE,DARK_ORANGE,'B')]:
    s.arrow(523,y+53,567,y+53,col,2,9)
    s.rect(571,y,176,101,'#f4faff' if letter=='A' else '#fff9f3',col,1.2,10)
    s.rect(573,y+2,172,32,'#e4f2ff' if letter=='A' else '#ffeedf',r=8)
    s.text(659,y+25,'Prediction from '+letter,20,dark,700,'middle')
    s.raw(f'<text x="636" y="{y+82}" font-size="43" fill="{dark}" style="font-family:Georgia,serif;font-style:italic">z<tspan font-size="25" baseline-shift="sub">{letter}</tspan></text>')
    s.raw(f'<path d="M 638 {y+49} l 9 -7 l 9 7" fill="none" stroke="{dark}" stroke-width="2"/>')

s.rect(779,485,525,93,'#e3f2fe',r=11)
s.text(1041.5,514,'Future discrimination (row-wise)',23,DARK_BLUE,700,'middle')
for xx,first,second in [(839,'AA','AB'),(1073,'BB','BA')]:
    s.rect(xx,526,188,44,'white','#b3bfd0',1.1,11);formula(xx+27,555,first,second,25);check(xx+160,548)
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
s.text(1031,797,'Prediction error (lower is better)',18,'#667488',anchor='middle')
s.rect(1219,590,200,212,'#fff0e4',r=12)
s.text(1319,621,'History benefit',23,DARK_ORANGE,700,'middle')
s.text(1319,649,'(column-wise)',22,DARK_ORANGE,anchor='middle')
for yy,first,second in [(669,'AA','BA'),(731,'BB','AB')]:
    s.rect(1227,yy,185,48,'white','#c6c4c3',1.1,11);formula(1246,yy+32,first,second,25);check(1385,yy+24)

panel(20,831,1408,236,'C','ContextWorld: 9 tasks across 4 environments',31)
# These are illustrative environment icons from the same supplied design.
cards=[(34,374,'TwoRoom','4 tasks',(48,899,139,137),['Speed','Action delay','Door traversability','Portal exit'],142),(421,353,'PushT','3 tasks',(435,897,122,133),['Pusher displacement','Contact friction','Motion damping'],125),(787,301,'Reacher','1 task',(802,897,100,134),['Arm mass'],104),(1100,320,'Cube','1 task',(1115,897,121,134),['Gripper carry rule'],125)]
for x,w,title,count,box,tasks,iw in cards:
    s.rect(x,881,w,172,'url(#card-bg)',r=15)
    ix=x+13;iy=896;ih=137
    s.rect(ix,iy,iw,ih,'#e7ebf1','#c6ced8',1,7)
    reference_crop(box,ix+1,iy+1,iw-2,ih-2,6)
    tx=ix+iw+14;tw=x+w-16-tx
    s.text(tx,915,title,24,weight=700)
    title_width={'TwoRoom':112,'PushT':75,'Reacher':96,'Cube':61}[title]
    s.text(tx+title_width+6,915,'('+count+')',19,'#596779')
    for k,t in enumerate(tasks):
        yy=929+k*29
        s.rect(tx,yy,tw,25,'white',r=7)
        s.text(tx+14,yy+19,t,18,'#41536c')

s.save(HERE/'benchmark_overview.svg')
(HERE/'overview_crop.json').write_text(json.dumps({
    'design_reference':'https://chatgpt.com/share/6ae80941-7830-83ea-a2af-476c8f212ea8',
    'source':'assets/reference/main_figure_design.png',
    'output_dimensions':[1448,1086],
    'scene_source':'Illustrations supplied in the design reference; not recorded observations or model predictions.',
    'scene_crops':SCENES,
    'shared_current_crop':SCENES['cannot_hold'][2],
    'history_thumbnails_share_initial_and_current':True,
    'layout':'A controlled Cube schematic; B paired row/column error comparisons; C nine tasks in four environments.',
    'semantic_adjustments':['Shared future-action input explicitly shown.','Lowercase latent z follows manuscript notation.','Current-observation crop is identical across conditions.'],
},indent=2)+'\n')
print('Reference-matched schematic saved.')
