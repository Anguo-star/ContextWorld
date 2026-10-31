"""Reference-matched vector drawing primitives; SVG text remains editable."""
from pathlib import Path
from html import escape
import base64

HERE = Path(__file__).resolve().parent
BLUE = '#087bd7'
DARK_BLUE = '#064789'
ORANGE = '#ff7b12'
DARK_ORANGE = '#b63d00'
INK = '#111216'


def data_uri(path):
    path = Path(path)
    mime = 'image/jpeg' if path.suffix == '.jpg' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


class SVG:
    def __init__(self, width, height):
        regular = base64.b64encode((HERE/'assets/fonts/Roboto-Regular.ttf').read_bytes()).decode()
        bold = base64.b64encode((HERE/'assets/fonts/Roboto-Bold.ttf').read_bytes()).decode()
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<defs><style>',
            f'@font-face{{font-family:Reference;src:url(data:font/ttf;base64,{regular}) format("truetype");font-weight:400}}',
            f'@font-face{{font-family:Reference;src:url(data:font/ttf;base64,{bold}) format("truetype");font-weight:700}}',
            'text{font-family:Reference,sans-serif}', '</style>',
            '<linearGradient id="neutral" x2="0" y2="1"><stop stop-color="#f0f0f2"/><stop offset="1" stop-color="#e4e5e8"/></linearGradient>',
            '<linearGradient id="blue" x2="0" y2="1"><stop stop-color="#e5f2ff"/><stop offset="1" stop-color="#b9ddff"/></linearGradient>',
            '<linearGradient id="orange" x2="0" y2="1"><stop stop-color="#fff0e2"/><stop offset="1" stop-color="#ffdab7"/></linearGradient>',
            '<linearGradient id="badge" x2="1" y2="1"><stop stop-color="#383c41"/><stop offset="1" stop-color="#15171a"/></linearGradient>',
            '<linearGradient id="model" x2="1" y2="1"><stop stop-color="#f0f0f3"/><stop offset="1" stop-color="#d7d9df"/></linearGradient>',
            '</defs><rect width="100%" height="100%" fill="white"/>']

    def raw(self, text):
        self.parts.append(text)

    def rect(self, x, y, w, h, fill='white', stroke='none', sw=1, r=0):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, text, size=20, fill=INK, weight=400, anchor='start', **attrs):
        more = ' '.join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k,v in attrs.items())
        self.raw(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {more}>{escape(text)}</text>')

    def lines(self, x, y, texts, size=20, gap=24, **kwargs):
        for j,t in enumerate(texts): self.text(x,y+j*gap,t,size=size,**kwargs)

    def line(self, x, y, xx, yy, color='#242a30', width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<line x1="{x}" y1="{y}" x2="{xx}" y2="{yy}" stroke="{color}" stroke-width="{width}"{d}/>')

    def arrow(self, x, y, xx, yy, color='#303940', width=2.4, head=8):
        import math
        dx,dy=xx-x,yy-y;norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
        self.line(x,y,xx,yy,color,width)
        a=(xx-head*ux+head*.45*uy,yy-head*uy-head*.45*ux)
        b=(xx-head*ux-head*.45*uy,yy-head*uy+head*.45*ux)
        self.raw(f'<polygon points="{xx},{yy} {a[0]},{a[1]} {b[0]},{b[1]}" fill="{color}"/>')

    def circle(self, x,y,r,fill,stroke='none',sw=1):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def panel(self, x,y,w,h,letter,title):
        self.rect(x,y,w,h,'white','#69717d',1.25,12)
        self.rect(x+2,y+2,44,42,'url(#badge)','#3a3d41',1,6)
        self.text(x+24,y+31,letter,29,'white',700,'middle')
        self.text(x+63,y+35,title,32,weight=700)

    def brace(self,x,y,w,sign=1,label=None):
        d=sign
        path=f'M {x} {y} Q {x} {y+14*d} {x+20} {y+14*d} L {x+w/2-20} {y+14*d} Q {x+w/2-6} {y+14*d} {x+w/2} {y+25*d} Q {x+w/2+6} {y+14*d} {x+w/2+20} {y+14*d} L {x+w-20} {y+14*d} Q {x+w} {y+14*d} {x+w} {y}'
        self.raw(f'<path d="{path}" fill="none" stroke="#16191d" stroke-width="2"/>')

    def image(self,path,x,y,w,h,viewbox=None,r=3):
        ident=f'clip{len(self.parts)}'
        self.raw(f'<defs><clipPath id="{ident}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath></defs>')
        self.raw(f'<g clip-path="url(#{ident})">')
        if viewbox:
            self.raw(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{viewbox}" preserveAspectRatio="xMidYMid meet"><image href="{data_uri(path)}" width="224" height="224"/></svg>')
        else:self.raw(f'<image href="{data_uri(path)}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet"/>')
        self.raw('</g>')

    def save(self,path):
        Path(path).write_text('\n'.join(self.parts+['</svg>']))
