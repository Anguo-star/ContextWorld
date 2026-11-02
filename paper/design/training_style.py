"""Shared axes and colors for the main and supplementary training figures."""
BLUE, ORANGE, DARK_BLUE = '#0078ff', '#ff780a', '#064789'


def plot_frame(s, x, y, w, h, lo, hi, ticks):
    xf = lambda t: x + (t-4.7)/5.6*w
    yf = lambda v: y+h-(v-lo)/(hi-lo)*h
    for t in range(5, 11):
        xx = xf(t)
        s.line(xx,y,xx,y+h,'#e4e5e8',1.1)
        s.line(xx,y+h,xx,y+h+6,'#171a1d',1.5)
        s.text(xx,y+h+28,str(t),19,anchor='middle')
    for val, lab in ticks:
        yy = yf(val)
        s.line(x,yy,x+w,yy,'#e4e5e8',1.1)
        s.line(x-5,yy,x,yy,'#171a1d',1.5)
        s.text(x-10,yy+7,lab,19,anchor='end')
    s.line(x,y,x,y+h,'#171a1d',1.5)
    s.line(x,y+h,x+w,y+h,'#171a1d',1.5)
    s.text(x+w/2,y+h+58,'Epoch',20,anchor='middle')
    return xf, yf


def marker(s, x, y, regime, color, *, filled=True, radius=5):
    fill = color if filled else 'none'
    if regime == 'joint':
        s.circle(x,y,radius,fill,color,1.6)
    else:
        s.rect(x-radius,y-radius,2*radius,2*radius,fill,color,1.6)
