################################
## gen — an svg diagram maker ##
################################

import os
import copy
import numpy as np

import fonts

# defaults
size_base = (200, 200)
rect_base = (0, 0) + size_base
frac_base = (0, 0, 1, 1)
ns_svg = 'http://www.w3.org/2000/svg'

##
## basic tools
##

def demangle(k):
    return k.replace('_', '-')

def dict_repr(d):
    return ' '.join([f'{demangle(k)}="{v}"' for k, v in d.items()])

def merge(d1, **d2):
    return {**d1, **d2}

def display(x, **kwargs):
    if type(x) is not SVG:
        x = SVG([x], **kwargs)
    return x.svg()

##
## context
##

# rect0 — pixel rect
# rect1 — fraction rect
def map_coords(prect, frect=frac_base, aspect=None):
    pxa, pya, pxb, pyb = prect
    fxa, fya, fxb, fyb = frect

    pw, ph = pxb - pxa, pyb - pya
    fw, fh = fxb - fxa, fyb - fya

    pxa1, pya1 = pxa + fxa*pw, pya + fya*ph
    pxb1, pyb1 = pxa + fxb*pw, pya + fyb*ph

    if aspect is not None:
        pw1, ph1 = fw*pw, fh*ph
        asp1 = pw1/ph1

        if asp1 == aspect: # just right
            pass
        elif asp1 > aspect: # too wide
            pw2 = aspect*ph1
            dpw = pw1 - pw2
            pxa1 += 0.5*dpw
            pxb1 -= 0.5*dpw
        elif asp1 < aspect: # too tall
            ph2 = pw1/aspect
            dph = ph2 - ph1
            pya1 -= 0.5*dph
            pyb1 += 0.5*dph

    return pxa1, pya1, pxb1, pyb1

class Context:
    def __init__(self, rect=rect_base, **kwargs):
        self.rect = rect
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

    def __call__(self, rect, aspect=None):
        rect1 = map_coords(self.rect, rect, aspect=aspect)
        ctx = self.copy()
        ctx.rect = rect1
        return ctx

    def clone(self, **kwargs):
        return Context(**{**self.__dict__, **kwargs})

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

##
## core types
##

class Element:
    def __init__(self, tag, unary=False, aspect=None, **attr):
        self.tag = tag
        self.unary = unary
        self.aspect = aspect
        self.attr = attr

    def __repr__(self):
        attr = dict_repr(self.attr)
        return f'{self.tag}: {attr}'

    def __add__(self, other):
        return Container([self, other])

    def _repr_svg_(self):
        return SVG(self).svg()

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx=None):
        if ctx is None:
            ctx = Context()

        props = dict_repr(self.props(ctx))
        pre = ' ' if len(props) > 0 else ''

        if self.unary:
            return f'<{self.tag}{pre}{props} />'
        else:
            inner = self.inner(ctx)
            return f'<{self.tag}{pre}{props}>{inner}</{self.tag}>'

    def save(self, fname, **kwargs):
        SVG(self, **kwargs).save(fname)

def rectify(r):
    if r is None:
        return frac_base
    elif type(r) is not tuple:
        return (r, r, 1-r, 1-r)
    elif len(r) == 2:
        rx, ry = r
        return (rx, ry, 1-rx, 1-ry)
    else:
        return r

class Container(Element):
    def __init__(self, children=None, tag='g', **attr):
        super().__init__(tag=tag, **attr)
        if children is None:
            children = []
        if isinstance(children, dict):
            children = [(c, r) for c, r in children.items()]
        if not isinstance(children, list):
            children = [children]
        children = [
            (c if type(c) is tuple else (c, None)) for c in children
        ]
        children = [(c, rectify(r)) for c, r in children]
        self.children = children

    def inner(self, ctx):
        inside = '\n'.join([c.svg(ctx(r, c.aspect)) for c, r in self.children])
        return f'\n{inside}\n'

class SVG(Container):
    def __init__(self, children=None, size=size_base, **attr):
        if children is not None and not isinstance(children, (list, dict)):
            children = [children]
        super().__init__(children=children, tag='svg', **attr)
        self.size = size

    def _repr_svg_(self):
        return self.svg()

    def props(self, ctx):
        w, h = self.size
        base = dict(width=w, height=h, xmlns=ns_svg)
        return {**base, **self.attr}

    def svg(self):
        rect0 = (0, 0) + self.size
        ctx = Context(rect=rect0)
        return Element.svg(self, ctx)

    def save(self, path):
        s = self.svg()
        with open(path, 'w+') as fid:
            fid.write(s)

##
## basic elements
##

# should this be atomic? or just an angle?
class Line(Element):
    def __init__(self, x1=0, y1=0, x2=1, y2=1, **attr):
        super().__init__(tag='line', unary=True, **attr)
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
    def __init__(self, **attr):
        base = dict(fill='none', stroke='black')
        attr1 = {**base, **attr}
        super().__init__(tag='rect', unary=True, **attr1)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w0, h0 = x2 - x1, y2 - y1

        x, y = x1, y1
        w, h = w0, h0

        base = dict(x=x, y=y, width=w, height=h)
        return {**base, **self.attr}

class Square(Rect):
    def __init__(self, **attr):
        super().__init__(aspect=1, **attr)

class Ellipse(Element):
    def __init__(self, cx=0.5, cy=0.5, rx=0.5, ry=0.5, **attr):
        base = dict(fill='none', stroke='black')
        attr1 = {**base, **attr}
        super().__init__(tag='ellipse', unary=True, **attr1)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        cx = x1 + 0.5*w
        cy = y1 + 0.5*h
        rx = 0.5*w
        ry = 0.5*h

        base = dict(cx=cx, cy=cy, rx=rx, ry=ry)
        return {**base, **self.attr}

class Circle(Ellipse):
    def __init__(self, **attr):
        super().__init__(aspect=1, **attr)

# this is an atomic element
class Text(Element):
    def __init__(self, text='', font_family='monospace', **attr):
        self.text_width, self.text_height = fonts.get_text_size(text, font=font_family)
        self.text = text

        base_aspect = self.text_width/self.text_height
        super().__init__(tag='text', aspect=base_aspect, font_family=font_family, **attr)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        fs = h/self.text_height

        base = dict(x=x1, y=y2, font_size=f'{fs}px', stroke='black')
        return {**base, **self.attr}

    def inner(self, ctx):
        return self.text

class TextDebug(Container):
    def __init__(self, text='', font_family='monospace', **attr):
        label = Text(text=text, font_family=font_family, **attr)
        boxes = Rect(stroke='red')
        outer = Rect(stroke='blue', stroke_dasharray='5 5')

        # mimic regular Text
        self.text_width = label.text_width
        self.text_height = label.text_height

        # get full font shaping info
        cluster, shapes, deltas, offsets = fonts.get_text_shape(text, font=font_family)
        shapes = [(w, h) for w, h in shapes]
        deltas = [(w, -h) for w, h in deltas]

        # compute character boxes
        if len(deltas) == 0:
            crects = []
        else:
            tw, th = self.text_width, self.text_height
            cumdel = [(0, 0)] +  [tuple(x) for x in np.cumsum(deltas[:-1], axis=0)]
            dshapes = [(dx, sy) for (dx, _), (_, sy) in zip(deltas, shapes)]
            rects = [(cx, cy, cx+dx, cy+dy) for (cx, cy), (dx, dy) in zip(cumdel, dshapes)]
            crects = [(x1/tw, 1-y2/th, x2/tw, 1-y1/th) for x1, y1, x2, y2 in rects]

        # render proportionally
        children = [label]
        children += [(boxes, tuple(frac)) for frac in crects]
        children += [outer]

        super().__init__(tag='g', children=children, aspect=label.aspect, **attr)

shape_classes = {
    'ellipse': Ellipse,
    'rect': Rect,
}

class Node(Container):
    def __init__(self, text=None, pad=0.15, shape='ellipse', text_args={}, shape_args={}, **attr):
        ShapeClass = shape_classes.get(shape, shape)

        label = Text(text=text, **text_args)
        outer = ShapeClass(**shape_args)

        aspect0 = label.aspect
        aspect1 = aspect0*(1+2*pad/aspect0)/(1+2*pad)

        super().__init__(children={label: pad, outer: None}, aspect=aspect1, **attr)

##
## plotting
##

class SymPath(Element):
    def __init__(self, fy=None, fx=None, xlim=None, ylim=None, tlim=None, N=100, **attr):
        super().__init__(tag='polyline', unary=True, **attr)

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

class Tick(Container):
    def __init__(self, text=None, loc='l', thick=3, pad=0.5, text_args={}, **attr):
        if text is None:
            line = Line(pad, 0.5, 1-pad, 0.5, stroke_width=thick)
            aspect = 1
            children = [line]
        else:
            text = Text(text, **text_args)
            line = Line(0, 0.5, 1, 0.5, stroke_width=thick)
            rect = Rect(stroke='red')

            taspect = text.aspect
            aspect = taspect + pad + 1
            ftext, fpad = taspect/aspect, (taspect+pad)/aspect

            children = {
                text: (0, 0, ftext, 1),
                line: (fpad, 0, 1, 1),
                rect: None
            }

        super().__init__(children=children, aspect=aspect, **attr)

class Scale(Container):
    def __init__(self, ticks, height=0.05, tick_args={}, **attr):
        if type(ticks) is dict:
            ticks = [(k, v) for k, v in ticks.items()]
        children = {
            Tick(s, **tick_args): (0, 1-(x-height/2), 1, 1-(x+height/2))
            for x, s in ticks
        }
        aspect = height*max([c.aspect for c in children.keys()])
        super().__init__(children=children, aspect=aspect, **attr)

class Axis(Element):
    def __init__(self, orient, pos=0, **attr):
        super().__init__(tag='line', unary=True, **attr)
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
        ctx1 = ctx(rpad)

        xpad, ypad = self.padding*xrange, self.padding*yrange
        ctx2 = ctx.clone(xmin=xmin-xpad, xmax=xmax+xpad, ymin=ymin-ypad, ymax=ymax+ypad)

        lines = '\n'.join([c.svg(ctx1(r)) for c, r in zip(self.lines, rects)])
        axes = '\n'.join([a.svg(ctx2(frac_base)) for a in self.axes])

        return lines + '\n' + axes
