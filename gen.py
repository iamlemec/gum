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
    if type(x) is not Diagram:
        x = Diagram([x], **kwargs)
    return x.svg()

class Context:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def coords(self, rect):
        rect1 = map_coords(self.rect, rect)
        ctx = self.copy()
        ctx.rect = rect1
        return ctx

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

        return f'<{self.tag}{pre}{props}>{pad}{inner}{pad}</{self.tag}>'

class Container(Element):
    def __init__(self, children=None, tag='g', **attr):
        super().__init__(tag=tag, **attr)
        if children is None:
            self.children = []
        else:
            self.children = [
                (c if type(c) is tuple else (c, rect_base)) for c in children
            ]

    def add(self, child, rect=rect_base):
        self.children.append((child, rect))
        return self

    def inner(self, ctx):
        return '\n'.join([c.svg(ctx.coords(r)) for c, r in self.children])

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
    def __init__(self, *args, **kwargs):
        super().__init__(tag='line', *args, **kwargs)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        base = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke='black')
        return {**base, **self.attr}

class Rect(Element):
    def __init__(self, *args, **kwargs):
        super().__init__(tag='rect', *args, **kwargs)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1
        base = dict(x=x1, y=y1, width=w, height=h, fill='none', stroke='black')
        return {**base, **self.attr}

class SymPath(Element):
    def __init__(self, formula, xlim, ylim=None, N=100, *args, **kwargs):
        super().__init__(tag='path', *args, **kwargs)

        xvals = np.linspace(*xlim, N)
        if type(formula) is str:
            yvals = eval(formula, {'x': xvals})
        else:
            yvals = formula(xvals)
        yvals *= np.ones_like(xvals)

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

        path = (
            f'M {xcoord[0]},{ycoord[0]}'
            + ' '.join([f'L {x},{y}' for x, y in zip(xcoord[1:], ycoord[1:])])
        )

        base = dict(d=path, fill='none', stroke='black')
        return {**base, **self.attr}

class Plot(Element):
    def __init__(self, lines=None, **attr):
        super().__init__(tag='g', **attr)
        self.lines = lines if lines is not None else []

    def add(self, line):
        self.lines.append(line)

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

        return '\n'.join([c.svg(ctx.coords(r)) for c, r in zip(self.lines, rects)])
