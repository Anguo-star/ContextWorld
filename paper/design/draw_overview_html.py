"""Reproduce Figure 1 with HTML/CSS, native text and the recorded images.

No SVG or canvas is used. Coordinates, crops, actions and schematic geometry
match the earlier design; the browser lays out each label as one text run.
"""
from pathlib import Path
from html import escape
import base64
import hashlib
import json
import math

from draw_response_panel import SCHEMATIC_STATES

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1448, 1168
ASSETS = HERE / 'assets/pusher_pair'
PROVENANCE = json.loads((ASSETS / 'provenance.json').read_text())
INK, OUTLINE = '#111620', '#8795aa'
BLUE, ORANGE = '#0874db', '#e27817'
DARK_BLUE, DARK_ORANGE = '#064b9a', '#a94b0a'
GRADIENTS = {
    'badge': 'linear-gradient(135deg,#383c41,#15171a)',
    'neutral': 'linear-gradient(#f0f0f2,#e4e5e8)',
    'current': 'linear-gradient(#eef8ff,#def2fe)',
    'card': 'linear-gradient(135deg,#f3f6fb,#eaf0f8)',
    'model': 'linear-gradient(135deg,#f0f6ff,#e5e6f4)',
    'blue': 'linear-gradient(#eaf5ff,#dcecff)',
    'orange': 'linear-gradient(#fff4e9,#ffeddf)',
    'purple': 'linear-gradient(#f3f0ff,#e9e5fc)',
}


def data_uri(path):
    mime = 'font/ttf' if path.suffix == '.ttf' else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


class HTML:
    def __init__(self):
        self.parts = []
        self.offset_y = 0

    def element(self, style, content='', cls='shape'):
        self.parts.append(f'<div class="{cls}" style="{style}">{content}</div>')

    def rect(self, x, y, w, h, fill='white', stroke=None, sw=1, r=0):
        border = f'border:{sw}px solid {stroke};' if stroke else ''
        # SVG borders straddled their coordinates; preserve that extent in CSS.
        inset = sw / 2 if stroke else 0
        self.element(f'left:{x-inset}px;top:{y+self.offset_y-inset}px;'
                     f'width:{w+2*inset}px;height:{h+2*inset}px;'
                     f'background:{GRADIENTS.get(fill,fill)};'
                     f'border-radius:{r}px;{border}')

    def text(self, x, y, value, size=20, color=INK, weight=400,
             anchor='start', math_text=False, narrow=False, html=False,
             halo=False):
        shift = {'start': '0', 'middle': '-50%', 'end': '-100%'}[anchor]
        family = 'Math' if math_text else 'Narrow' if narrow else 'FigureText'
        # With line-height:1, Roboto's hhea ascent/descent put its baseline here.
        baseline = .842 if not math_text else .846
        style = (f'left:{x}px;top:{y+self.offset_y-size*baseline}px;'
                 f'transform:translateX({shift});font-family:{family};'
                 f'font-size:{size}px;font-weight:{weight};color:{color};')
        if math_text:
            style += 'font-style:italic;font-synthesis:style;'
        if halo:
            style += 'text-shadow:0 0 2px #fcfdff,0 0 2px #fcfdff,0 0 2px #fcfdff;'
        self.element(style, value if html else escape(value), 'label')

    def lines(self, x, y, values, size=20, gap=24, **kwargs):
        for i, value in enumerate(values):
            self.text(x, y+i*gap, value, size, **kwargs)

    def line(self, x, y, xx, yy, color='#242a30', width=2, dash=None):
        length = math.hypot(xx-x, yy-y)
        angle = math.degrees(math.atan2(yy-y, xx-x))
        if dash:
            on, off = dash
            paint = (f'background:repeating-linear-gradient(to right,{color} 0,'
                     f'{color} {on}px,transparent {on}px,transparent {on+off}px);')
        else:
            paint = f'background:{color};'
        self.element(f'left:{x}px;top:{y+self.offset_y-width/2}px;'
                     f'width:{length}px;height:{width}px;transform-origin:0 50%;'
                     f'transform:rotate({angle}deg);{paint}')

    def polygon(self, points, fill):
        left, top = min(p[0] for p in points), min(p[1] for p in points)
        w, h = max(p[0] for p in points)-left, max(p[1] for p in points)-top
        coords = ','.join(f'{x-left}px {y-top}px' for x,y in points)
        self.element(f'left:{left}px;top:{top+self.offset_y}px;width:{w}px;'
                     f'height:{h}px;background:{fill};clip-path:polygon({coords});')

    def arrow(self, x, y, xx, yy, color='#303940', width=2.4, head=8):
        self.line(x,y,xx,yy,color,width)
        length = math.hypot(xx-x, yy-y)
        ux, uy = (xx-x)/length, (yy-y)/length
        self.polygon([(xx,yy),(xx-head*ux+head*.45*uy,yy-head*uy-head*.45*ux),
                      (xx-head*ux-head*.45*uy,yy-head*uy+head*.45*ux)], color)

    def circle(self, x, y, r, fill, stroke=None, sw=1):
        self.rect(x-r,y-r,2*r,2*r,fill,stroke,sw,999)

    def crop(self, path, box, x, y, w, h, r=3, fit='meet'):
        bx, by, bw, bh = box
        scale = min(w/bw,h/bh) if fit == 'meet' else max(w/bw,h/bh)
        ox, oy = (w-bw*scale)/2-bx*scale, (h-bh*scale)/2-by*scale
        # All original screenshots are 224 square. Preserve the actual bytes.
        content = (f'<img alt="Recorded simulation observation" src="{data_uri(path)}" '
                   f'style="position:absolute;left:{ox}px;top:{oy}px;'
                   f'width:{224*scale}px;height:{224*scale}px;max-width:none">')
        self.element(f'left:{x}px;top:{y+self.offset_y}px;width:{w}px;'
                     f'height:{h}px;overflow:hidden;border-radius:{r}px;', content)

    def panel(self, x, y, w, h, letter, title, size=32):
        self.rect(x,y,w,h,'white',OUTLINE,1.15,14)
        self.rect(x,y,54,53,'badge',r=11)
        self.text(x+27,y+38,letter,34,'white',700,'middle')
        self.text(x+69,y+37,title,size,weight=700)

    def save(self, path, view_y=0, view_h=HEIGHT):
        fonts = []
        for family, name in [('FigureText','Roboto'),('Narrow','RobotoCondensed')]:
            for weight, suffix in [(400,'Regular'),(700,'Bold')]:
                url = data_uri(HERE/f'assets/fonts/{name}-{suffix}.ttf')
                fonts.append(f'@font-face{{font-family:{family};src:url({url});font-weight:{weight}}}')
        css = ''.join(fonts) + f'''
@page {{ size:{WIDTH}px {view_h}px; margin:0; }}
* {{ box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
html,body {{ margin:0; padding:0; width:{WIDTH}px; height:{view_h}px; background:white; }}
.viewport {{ position:relative; width:{WIDTH}px; height:{view_h}px; overflow:hidden; }}
.figure {{ position:absolute; top:{-view_y}px; left:0; width:{WIDTH}px; height:{HEIGHT}px; }}
.shape,.label {{ position:absolute; }}
.label {{ white-space:pre; line-height:1; letter-spacing:normal; font-kerning:normal;
  font-variant-ligatures:normal; font-synthesis:none; text-rendering:geometricPrecision; }}
/* Native inline sub/superscripts preserve editable mathematical text. */
sub,sup {{ font-size:64%; line-height:0; position:relative; vertical-align:baseline; }}
sub {{ bottom:-.27em; }} sup {{ top:-.6em; }}
.hat {{ position:relative; display:inline-block; }}
.hat::before {{ content:''; position:absolute; left:50%; top:.04em;
  width:.24em; height:.24em; border-left:.055em solid; border-top:.055em solid;
  transform:translateX(-50%) rotate(45deg); }}
'''
        # Use the same installed serif family as the old figure's Georgia fallback.
        content = '\n'.join(self.parts).replace('font-family:Math;', "font-family:'DejaVu Serif',serif;")
        path.write_text(f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                        f'<title>ContextWorld — Figure 1</title><style>{css}</style></head>'
                        f'<body><div class="viewport"><div class="figure">{content}'
                        '</div></div></body></html>\n')


def frame(s, mode, t, x, y, w, h, annotate=False):
    record = PROVENANCE['frames'][mode][str(t*5)]
    s.rect(x,y,w,h,'white','#8f99a8',1.2,4)
    s.crop(ASSETS/record['file'],(27,20,146,146),x+1,y+1,w-2,h-2)
    size = min(w-2,h-2)
    pixel_scale = size/146
    ox, oy = x+(w-size)/2-27*pixel_scale, y+(h-size)/2-20*pixel_scale
    if annotate and t == 3 and mode == 'high_gain':
        outline = json.loads((ASSETS/'current_t_outline.json').read_text())
        assert hashlib.sha256((ASSETS/outline['source']).read_bytes()).hexdigest() == outline['sha256']
        points = [(ox+px*pixel_scale,oy+py*pixel_scale) for px,py in outline['vertices']]
        s.polygon(points,'rgba(134,152,167,.16)')
        for a,b in zip(points,points[1:]+points[:1]):
            s.line(*a,*b,'rgba(110,130,148,.9)',1.2,(3.5,3))
    if annotate and t in [1,2,3]:
        scale = pixel_scale*224/512
        start = PROVENANCE['frames'][mode][str((t-1)*5)]['state'][:2]
        end = record['state'][:2]
        ax,ay = ox+start[0]*scale,oy+start[1]*scale
        bx,by = ox+end[0]*scale,oy+end[1]*scale
        length = math.hypot(bx-ax,by-ay)
        ux,uy = (bx-ax)/length,(by-ay)/length
        s.circle(ax,ay,4,'white','#46556b',1.3)
        s.line(ax+5*ux,ay+5*uy,bx-8*ux,by-8*uy,'#46556b',1.6,(3,2))
        s.arrow(bx-11*ux,by-11*uy,bx-7.5*ux,by-7.5*uy,'#46556b',1.3,4.5)


def latent(s,x,y,letter,pred=False,size=29,color=None,anchor='middle'):
    z = '<span class="hat">z</span>' if pred else 'z'
    s.text(x,y,f'{z}<sub>{letter}</sub>',size,
           color or (DARK_BLUE if letter=='A' else DARK_ORANGE),
           anchor=anchor,math_text=True,html=True)


def comparison(s,x,y,first,second,size=20):
    color = '#0044b5' if first=='AA' else '#c52e00'
    s.text(x,y,f'<span style="color:{color}">d<sub>{first}</sub></span>'
           f' &lt; d<sub>{second}</sub>',size,math_text=True,html=True)


def route(s,x,y,xx,yy,color):
    mid = (x+xx)/2
    s.line(x,y,mid,y,color,1.8)
    s.line(mid,y,mid,yy,color,1.8)
    s.arrow(mid,yy,xx,yy,color,1.8,6)


def draw():
    for mode in ['low_gain','high_gain']:
        for record in PROVENANCE['frames'][mode].values():
            assert hashlib.sha256((ASSETS/record['file']).read_bytes()).hexdigest() == record['sha256']
    assert PROVENANCE['frames']['low_gain']['10']['sha256'] == PROVENANCE['frames']['high_gain']['10']['sha256']
    s = HTML()
    s.panel(20,15,1408,454,'A','Different histories, different futures')
    s.rect(787,73,203,386,'current','#b3dcfc',.8,10)
    s.line(220,272,1416,272,'#ccd5df',1.3,(4,4))
    # A CSS brace: two joined half-braces, not an image or an SVG glyph.
    for y,corner in [(111,'0 12px 12px 0'),(272.5,'0 12px 12px 0')]:
        s.element(f'left:1317px;top:{y}px;width:11px;height:161.5px;'
                  f'border-right:1.8px solid #14191e;border-radius:{corner};')
    s.rect(1333,110,86,324,'card',r=10)
    s.lines(1376,258,['Different','future','outcomes'],21,24,weight=700,anchor='middle',narrow=True)
    s.circle(55,253,6,'#527fff','#3563b3',1)
    s.text(73,260,'Pusher',19,'#41536c')
    s.line(42,278,64,278,'#46556b',1.6,(3,2))
    s.arrow(60,278,67,278,'#46556b',1.3,5)
    s.text(73,284,'Movement',17,'#41536c')
    columns = [(0,222,'History t − 2'),(1,510,'History t − 1'),
               (2,800,'Current observation'),(3,1122,'Ground-truth future')]
    for t,x,title in columns:
        s.rect(x-3,74,187,37,'neutral' if t!=2 else '#e1f3ff',r=6)
        s.text(x+90.5,99,title,20,'#0044b5' if t==2 else INK,700,'middle')
    for mode,y,letter,dark,desc in [
        ('low_gain',110,'A','#0044b5',['Smaller','action response']),
        ('high_gain',298,'B','#c52e00',['Larger','action response'])]:
        s.rect(36,y+16,175,111,'#e2f2ff' if letter=='A' else '#fff0e4',r=11)
        s.text(48,y+52,f'Condition {letter}:',24,dark,700)
        s.lines(48,y+81,desc,20,25,weight=700)
        for t,x,title in columns:
            frame(s,mode,t,x+5,y,171,136,annotate=True)
            if t==3:
                s.text(x+90.5,y+154,'Pusher moves less' if letter=='A' else 'Pusher moves more',20,dark,700,'middle')
        blocks = ['context_0_to_5','context_5_to_10','query_10_to_15']
        for j,(x1,x2) in enumerate([(418,499),(702,782),(1017,1109)]):
            center = (x1+x2)/2
            s.text(center,y+38,'Move',20,'#cf0000',700,'middle')
            commands = PROVENANCE['action_blocks'][blocks[j]]
            dx,dy = [sum(a[k] for a in commands) for k in range(2)]
            norm = math.hypot(dx,dy)
            ux,uy = dx/norm,dy/norm
            s.arrow(center-15*ux,y+68-15*uy,center+15*ux,y+68+15*uy,'#cf0000',2.5,7)
            s.arrow(x1,y+105,x2,y+105,'#344054',2,9)

    s.offset_y = 12
    s.panel(20,471,1408,416,'B','Evaluation')
    s.offset_y -= 36
    muted, ink = '#64758a','#122840'
    for mode,y,letter,col,dark in [('low_gain',589,'A',BLUE,DARK_BLUE),('high_gain',751,'B',ORANGE,DARK_ORANGE)]:
        s.rect(36,y,176,112,'#fbfdff' if letter=='A' else '#fffdfb','#a7cff4' if letter=='A' else '#edc6a5',1.1,11)
        s.rect(38,y+2,172,29,'blue' if letter=='A' else 'orange',r=9)
        s.text(124,y+24,'History '+letter,22,dark,700,'middle')
        for t,x in [(0,44),(1,99),(2,163)]:
            frame(s,mode,t,x,y+39,42,43)
        s.text(151,y+66,'…',16,muted,anchor='middle')
        s.text(124,y+101,'observations + actions',14.5,muted,anchor='middle')
        route(s,213,y+56,248,665 if letter=='A' else 787,col)
    s.rect(249,623,138,206,'model','#a6b7d1',1.2,12)
    for angle in [0,math.pi/3,2*math.pi/3]:
        ux,uy = math.cos(angle),math.sin(angle)
        s.line(318-15*ux,656-15*uy,318+15*ux,656+15*uy,'#589bc4',2)
        for sign in [-1,1]:
            x,y = 318+sign*15*ux,656+sign*15*uy
            for side in [-1,1]:
                s.line(x,y,x-sign*5*ux+side*3*uy,y-sign*5*uy-side*3*ux,'#589bc4',1.5)
    s.lines(318,702,['Frozen','latent world','model'],21.5,28,color=ink,weight=700,anchor='middle')
    s.text(318,808,'weights fixed',15,muted,anchor='middle')
    for y,letter,col,dark in [(594,'A',BLUE,DARK_BLUE),(756,'B',ORANGE,DARK_ORANGE)]:
        route(s,388,665 if letter=='A' else 787,425,y+51,col)
        s.rect(426,y,106,102,'#fbfdff' if letter=='A' else '#fffdfb','#a7cff4' if letter=='A' else '#edc6a5',1.1,10)
        s.rect(428,y+2,102,28,'blue' if letter=='A' else 'orange',r=8)
        s.text(479,y+23,'Prediction '+letter,17.3,dark,700,'middle')
        latent(s,479,y+81,letter,True,39,dark)
    s.line(546,541,546,901,'#e0e7ef',1)
    card_y = 533
    for i,(x,title,tint,border) in enumerate([
        (560,'Adaptation accuracy','blue','#c7dff6'),
        (829,'History sensitivity','orange','#e6d9cc'),
        (1098,'Response fidelity','purple','#d8d3ec')]):
        w = 256 if i<2 else 316
        if i!=1:
            s.rect(x,card_y,525 if i==0 else 316,376,'#fcfdff',border,1.1,14)
        s.rect(x+8,card_y+8,w-16,38,tint,r=10)
        s.text(x+w/2,card_y+34,title,20.5,ink,700,'middle')
    s.lines(688,card_y+65,['Same prediction,','compare true futures'],16,22,color=ink,anchor='middle')
    s.lines(957,card_y+65,['Same true future,','compare histories'],16,22,color=ink,anchor='middle')
    points = {k:(580+.9*v[0]+(219 if k.startswith('true_') else 0),card_y+8.1+.9*v[1])
              for k,v in SCHEMATIC_STATES.items()}
    for ij in ['AB','BA','BB','AA']:
        a,b = points['pred_'+ij[0]],points['true_'+ij[1]]
        matched = ij[0]==ij[1]
        col = (BLUE if ij=='AA' else ORANGE) if matched else '#8796a8'
        s.line(*a,*b,col,2.7,None if matched else (5,5))
        if ij=='AA': pos=((a[0]+b[0])/2,(a[1]+b[1])/2-10)
        elif ij=='BB': pos=((a[0]+b[0])/2,(a[1]+b[1])/2+20)
        elif ij=='AB': pos=(a[0]+.68*(b[0]-a[0]),a[1]+.68*(b[1]-a[1])+23)
        else: pos=(a[0]+.3*(b[0]-a[0]),a[1]+.3*(b[1]-a[1])-12)
        s.text(*pos,f'd<sub>{ij}</sub>',19,(DARK_BLUE if ij=='AA' else DARK_ORANGE) if matched else ink,
               anchor='middle',math_text=True,html=True,halo=True)
    for key,(x,y) in points.items():
        letter, pred = key[-1],key.startswith('pred')
        col = BLUE if letter=='A' else ORANGE
        s.circle(x,y,8.5,col if pred else 'white',col,2.3)
        if key=='true_B': latent(s,x+18,y-8,letter,pred,23.5,anchor='start')
        else: latent(s,x-15 if key=='pred_B' else x,y-27 if letter=='A' else y+29,letter,pred,23.5)
    for x,kind in [(560,'prediction'),(829,'future')]:
        s.rect(x+12,card_y+310,232,52,'blue' if kind=='prediction' else 'orange',r=8)
        a,b = ('AB','BA') if kind=='prediction' else ('BA','AB')
        comparison(s,x+17,card_y+345,'AA',a)
        comparison(s,x+141,card_y+345,'BB',b)
    s.lines(1256,card_y+65,['Does the predicted response','match the true response?'],16,22,color=ink,anchor='middle')
    origin = (1128,card_y+250)
    def response(predicted):
        prefix = 'pred_' if predicted else 'true_'
        a,b = SCHEMATIC_STATES[prefix+'A'],SCHEMATIC_STATES[prefix+'B']
        dx,dy,angle = b[0]-a[0],b[1]-a[1],math.radians(-115)
        return (origin[0]+(dx*math.cos(angle)-dy*math.sin(angle))*.9*1.65,
                origin[1]+(dx*math.sin(angle)+dy*math.cos(angle))*.9*1.65)
    true,pred = response(False),response(True)
    s.line(*true,*pred,'#a0aabc',1.8,(5,4))
    s.arrow(*origin,*true,BLUE,3,9)
    s.arrow(*origin,*pred,ORANGE,3,9)
    s.circle(*origin,3.5,'#4c6078')
    for x,y,predicted in [(true[0]+13,true[1]-2,False),(pred[0]-85,pred[1]-20,True)]:
        color = DARK_ORANGE if predicted else DARK_BLUE
        s.text(x,y,'Δ<span class="hat">z</span>' if predicted else 'Δz',26,color,math_text=True,html=True)
        s.text(x,y+22,'predicted' if predicted else 'true',16,color,700)
    s.rect(1110,card_y+310,292,52,'purple',r=8)
    s.text(1142,card_y+345,'NRE =',17,ink,700)
    s.line(1198,card_y+336,1369,card_y+336,'#34415b',1)
    for y,num in [(card_y+329,True),(card_y+357,False)]:
        inside = 'Δ<span class="hat">z</span><sub>i</sub> − Δz<sub>i</sub>' if num else 'Δz<sub>i</sub>'
        s.text(1283.5,y,f'∑<sub>i</sub> ‖{inside}‖<sup>2</sup>',18,ink,
               anchor='middle',math_text=True,html=True)

    s.offset_y = 82
    s.panel(20,831,1408,236,'C','ContextWorld: 9 tasks across 4 environments',31)
    s.offset_y += 6
    cards = [
        (34,374,'TwoRoom','4 tasks',HERE/'assets/environments/tworoom.jpg',['Speed','Action delay','Door traversability','Portal exit'],142),
        (421,353,'PushT','3 tasks',ASSETS/'pusher_low_gain_step10.jpg',['Pusher displacement','Contact friction','Motion damping'],125),
        (787,301,'Reacher','1 task',HERE/'assets/environments/reacher.jpg',['Arm mass'],104),
        (1100,320,'Cube','1 task',HERE/'assets/environments/cube.jpg',['Gripper carry rule'],125)]
    for x,w,title,count,path,tasks,iw in cards:
        s.rect(x,881,w,172,'card',r=15)
        ix,iy,ih = x+13,896,137
        s.rect(ix,iy,iw,ih,'#e7ebf1','#c6ced8',1,7)
        s.crop(path,(0,0,224,224),ix+1,iy+1,iw-2,ih-2,6)
        tx = ix+iw+14
        tw = x+w-16-tx
        s.text(tx,915,title,24,weight=700)
        title_width = {'TwoRoom':112,'PushT':75,'Reacher':96,'Cube':61}[title]
        s.text(tx+title_width+6,915,'('+count+')',19,'#596779')
        for k,task in enumerate(tasks):
            yy = 929+k*29
            s.rect(tx,yy,tw,25,'white',r=7)
            s.text(tx+14,yy+19,task,18,'#41536c')
    s.save(HERE/'benchmark_overview_pusher_response.html')
    s.save(HERE/'paired_evaluation_design.html',475,432)
    print('HTML overview generated; recorded image hashes and shared observation verified.')


if __name__ == '__main__':
    draw()
