# svg diagram maker

import copy
import numpy as np

# defaults
size_base = (200, 200)
rect_base = (0, 0, 1, 1)

def demangle(k):
    return k.replace('_', '-')

def dict_repr(d):
    return ' '.join([f'{demangle(k)}="{v}"' for k, v in d.items()])

def map_coords(rect0, rect1):
    xa0, ya0, xb0, yb0 = rect0
    xa1, ya1, xb1, yb1 = rect1
    w0, h0 = xb0 - xa0, yb0 - ya0
    w1, h1 = xb1 - xa1, yb1 - ya1
    xa2, ya2 = xa0 + xa1*w0, ya0 + ya1*h0
    xb2, yb2 = xa0 + xb1*w0, ya0 + yb1*h0
    return xa2, ya2, xb2, yb2

def merge(d1, **d2):
    return {**d1, **d2}

def display(x, **kwargs):
    if type(x) is not SVG:
        x = SVG([x], **kwargs)
    return x.svg()

class Context:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

    def coords(self, rect):
        rect1 = map_coords(self.rect, rect)
        ctx = self.copy()
        ctx.rect = rect1
        return ctx

    def clone(self, **kwargs):
        return Context(**{**self.__dict__, **kwargs})

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

class Element:
    def __init__(self, tag, **attr):
        self.tag = tag
        self.attr = attr

    def __repr__(self):
        attr = dict_repr(self.attr)
        return f'{self.tag}: {attr}'

    def _repr_svg_(self):
        return SVG((self, (0.01, 0.01, 0.99, 0.99)), size=size_base).svg()

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx=None):
        if ctx is None:
            ctx = Context(rect=size_base)

        props = dict_repr(self.props(ctx))
        inner = self.inner(ctx)

        pre = ' ' if len(props) > 0 else ''
        pad = '\n' if len(inner) > 0 else ''

        return f'<{self.tag}{pre}{props}>{inner}</{self.tag}>'

class Container(Element):
    def __init__(self, children=None, tag='g', **attr):
        super().__init__(tag=tag, **attr)
        if children is None:
            self.children = []
        else:
            self.children = [
                (c if type(c) is tuple else (c, rect_base)) for c in children
            ]

    def inner(self, ctx):
        return '\n' + '\n'.join([c.svg(ctx.coords(r)) for c, r in self.children]) + '\n'

class SVG(Container):
    def __init__(self, children=None, size=size_base, **attr):
        if children is not None and type(children) is not list:
            children = [children]
        super().__init__(children=children, tag='svg', **attr)
        self.size = size

    def _repr_svg_(self):
        return self.svg()

    def props(self, ctx):
        w, h = self.size
        base = dict(width=w, height=h)
        return {**base, **self.attr}

    def svg(self):
        rect0 = (0, 0) + self.size
        ctx = Context(rect=rect0)
        return Element.svg(self, ctx)

    def save(self, path):
        s = self.svg()
        with open(path, 'w+') as fid:
            fid.write(s)

class Line(Element):
    def __init__(self, x1=0, y1=0, x2=1, y2=1, **attr):
        super().__init__(tag='line', **attr)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        x1p = x1 + w*self.x1
        y1p = y1 + h*self.y1
        x2p = x1 + w*self.x2
        y2p = y1 + h*self.y2

        base = dict(x1=x1p, y1=y1p, x2=x2p, y2=y2p, stroke='black')
        return {**base, **self.attr}

class Rect(Element):
    def __init__(self, x=0, y=0, w=1, h=1, **attr):
        super().__init__(tag='rect', **attr)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w0, h0 = x2 - x1, y2 - y1

        x = x1 + w0*self.x
        y = y1 + h0*self.y
        w = w0*self.w
        h = h0*self.h

        base = dict(x=x, y=y, width=w, height=h, fill='none', stroke='black')
        return {**base, **self.attr}

class Ellipse(Element):
    def __init__(self, cx=0.5, cy=0.5, rx=0.5, ry=0.5, **attr):
        super().__init__(tag='ellipse', **attr)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        cx = x1 + w*self.cx
        cy = y1 + h*self.cy
        rx = w*self.rx
        ry = h*self.ry

        base = dict(cx=cx, cy=cy, rx=rx, ry=ry, fill='none', stroke='black')
        return {**base, **self.attr}

class Circle(Ellipse):
    def __init__(self, cx=0.5, cy=0.5, r=0.5, **attr):
        super().__init__(cx, cy, r, r, **attr)

class Text(Element):
    def __init__(self, x=0, y=1, text='', font_size=0.05, **attr):
        super().__init__(tag='text', **attr)
        self.x = x
        self.y = y
        self.text = text
        self.font_size = font_size

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        x = x1 + w*self.x
        y = y1 + h*self.y
        fs = h*self.font_size

        base = dict(x=x, y=y, font_size=f'{fs}px', stroke='black')
        return {**base, **self.attr}

    def inner(self, ctx):
        return self.text

class SymPath(Element):
    def __init__(self, fy=None, fx=None, xlim=None, ylim=None, tlim=None, N=100, **attr):
        super().__init__(tag='polyline', **attr)

        if fx is not None and fy is not None:
            tvals = np.linspace(*tlim, N)
            if type(fx) is str:
                xvals = eval(fx, {'t': tvals})
            else:
                xvals = fx(tvals)
            if type(fy) is str:
                yvals = eval(fy, {'t': tvals})
            else:
                yvals = fy(tvals)
            xvals *= np.ones_like(tvals)
            yvals *= np.ones_like(tvals)
        elif fy is not None:
            xvals = np.linspace(*xlim, N)
            if type(fy) is str:
                yvals = eval(formula, {'x': xvals})
            else:
                yvals = fy(xvals)
            yvals *= np.ones_like(xvals)
        elif fx is not None:
            yvals = np.linspace(*ylim, N)
            if type(fx) is str:
                xvals = eval(formula, {'y': yvals})
            else:
                xvals = fx(yvals)
            xvals *= np.ones_like(yvals)
        else:
            raise Exception('Must specify either fx or fy')

        if xlim is None:
            self.xlim = np.min(xvals), np.max(xvals)
        else:
            self.xlim = xlim
        if ylim is None:
            self.ylim = np.min(yvals), np.max(yvals)
        else:
            self.ylim = ylim

        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        xrange = xmax - xmin
        yrange = ymax - ymin

        self.xnorm = (xvals-xmin)/xrange if xrange != 0 else 0.5*np.ones_like(xvals)
        self.ynorm = (ymax-yvals)/yrange if yrange != 0 else 0.5*np.ones_like(yvals)

    def props(self, ctx):
        cx1, cy1, cx2, cy2 = ctx.rect
        cw, ch = cx2 - cx1, cy2 - cy1

        xcoord = cx1 + self.xnorm*cw
        ycoord = cy1 + self.ynorm*ch

        points = ' '.join([f'{x},{y}' for x, y in zip(xcoord, ycoord)])
        base = dict(points=points, fill='none', stroke='black')
        return {**base, **self.attr}

class Axis(Element):
    def __init__(self, orient, pos=0, **attr):
        super().__init__(tag='line', **attr)
        self.orient = orient
        self.pos = pos

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        if self.orient in ('h', 'x'):
            fy = (ctx.ymax-self.pos)/(ctx.ymax-ctx.ymin)
            y = y1 + fy*h
            y1p,  y2p = y, y
            x1p, x2p = x1, x2
        elif self.orient in ('v', 'y'):
            fx = (self.pos-ctx.xmin)/(ctx.xmax-ctx.xmin)
            x = x1 + fx*w
            x1p, x2p = x, x
            y1p, y2p = y1, y2

        base = dict(x1=x1p, y1=y1p, x2=x2p, y2=y2p, stroke='black')
        return {**base, **self.attr}

class Plot(Element):
    def __init__(self, lines=None, xaxis=0, yaxis=0, padding=0.05, **attr):
        super().__init__(tag='g', **attr)

        if xaxis is not None and not isinstance(xaxis, Element):
            xaxis = Axis('x', xaxis)
        if yaxis is not None and not isinstance(yaxis, Element):
            yaxis = Axis('y', yaxis)

        self.lines = lines if lines is not None else []
        self.axes = [xaxis, yaxis]
        self.padding = padding

    def inner(self, ctx):
        xmins, xmaxs = zip(*[c.xlim for c in self.lines])
        ymins, ymaxs = zip(*[c.ylim for c in self.lines])
        xmin, xmax = min(xmins), max(xmaxs)
        ymin, ymax = min(ymins), max(ymaxs)
        xrange = xmax - xmin
        yrange = ymax - ymin

        x1s = [(x-xmin)/xrange if xrange != 0 else 0.5 for x in xmins]
        y1s = [(ymax-y)/yrange if yrange != 0 else 0.5 for y in ymaxs]
        x2s = [(x-xmin)/xrange if xrange != 0 else 0.5 for x in xmaxs]
        y2s = [(ymax-y)/yrange if yrange != 0 else 0.5 for y in ymins]
        rects = [r for r in zip(x1s, y1s, x2s, y2s)]

        rpad = (self.padding, self.padding, 1 - self.padding, 1 - self.padding)
        ctx1 = ctx.coords(rpad)

        xpad, ypad = self.padding*xrange, self.padding*yrange
        ctx2 = ctx.clone(xmin=xmin-xpad, xmax=xmax+xpad, ymin=ymin-ypad, ymax=ymax+ypad)

        lines = '\n'.join([c.svg(ctx1.coords(r)) for c, r in zip(self.lines, rects)])
        axes = '\n'.join([a.svg(ctx2.coords(rect_base)) for a in self.axes])

        return lines + '\n' + axes

class Node(Container):
    def __init__(self, cx=0.5, cy=0.5, text=None, rx=None, ry=None, font_size=0.05, font_aspect=2, pad=None, **attr):
        text_width = (font_size/font_aspect)*len(text)
        text_height = font_size

        if pad is None:
            pad = 0.5*text_height
        if ry is None:
            ry = 0.5*text_height + pad
        if rx is None:
            rx = 0.5*text_width + pad

        tx = cx - 0.5*text_width
        ty = cy + 0.5*text_height

        bubble = Ellipse(cx=cx, cy=cy, rx=rx, ry=ry)
        label = Text(x=tx, y=ty, text=text, font_size=font_size)

        super().__init__(children=[bubble, label], **attr)
