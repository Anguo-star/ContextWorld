"""Shared axes and colors for the main and supplementary training figures."""
BLUE, ORANGE, DARK_BLUE, DARK_ORANGE = '#0078ff', '#ff780a', '#064789', '#b63d00'


def plot_frame(s, x, y, w, h, lo, hi, ticks, *, epochs=range(5, 11), xlim=(4.7, 10.3)):
    xf = lambda t: x + (t-xlim[0])/(xlim[1]-xlim[0])*w
    yf = lambda v: y+h-(v-lo)/(hi-lo)*h
    for t in epochs:
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
    fill = color if filled else 'white'
    if regime == 'joint':
        s.circle(x,y,radius,fill,color,1.6)
    else:
        s.rect(x-radius,y-radius,2*radius,2*radius,fill,color,1.6)


def nre_frame(s, x, y, w=245, h=190):
    xf, yf = plot_frame(s, x, y, w, h, 0, 1.5,
                        [(0, '0.0'), (.5, '0.5'), (1, '1.0'), (1.5, '1.5')],
                        epochs=(5, 10), xlim=(4.5, 10.5))
    s.line(x, yf(1), x+w, yf(1), '#626a75', 1.5, '6 3')
    s.text(x+w/2, yf(1)-12, 'Zero-response reference', 18,
           fill='#505963', anchor='middle')
    return xf, yf


def nre_points(s, xf, yf, values, regime='joint', split='training', offset=0):
    for epoch, value in zip((5, 10), values):
        marker(s, xf(epoch+offset), yf(value), regime, ORANGE,
               filled=split == 'training', radius=5.8)
