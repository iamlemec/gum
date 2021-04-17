################################
## gen — an svg diagram maker ##
################################

import os
import copy
import numpy as np
from collections import defaultdict
from math import sqrt

import fonts

##
## defaults
##

# namespace
ns_svg = 'http://www.w3.org/2000/svg'

# sizing
size_base = 200
rect_base = (0, 0, 100, 100)
frac_base = (0, 0, 1, 1)

# specific elements
default_tick_size = 0.05
default_font_family = 'Montserrat'

##
## basic tools
##

def demangle(k):
    return k.replace('_', '-')

def rounder(x, prec=13):
    if type(x) is float:
        xr = round(x, ndigits=prec)
        if (xr % 1) == 0:
            return int(xr)
        else:
            return x
    else:
        return x

def props_repr(d):
    return ' '.join([f'{demangle(k)}="{rounder(v)}"' for k, v in d.items()])

def value_repr(x):
    if type(x) is str:
        return f'"{x}"'
    else:
        return x

def rule_repr(d, tab=4*' '):
    return '\n'.join([f'{tab}{demangle(k)}: {value_repr(v)};' for k, v in d.items()])

def style_repr(d):
    return '\n\n'.join([
        tag + ' {\n' + rule_repr(rules) + '\n}' for tag, rules in d.items()
    ])

def dispatch(d, keys):
    rest = {}
    subs = defaultdict(dict)
    for k, v in d.items():
        for s in keys:
            if k.startswith(f'{s}_'):
                k1 = k[len(s)+1:]
                subs[s][k1] = v
            else:
                rest[k] = v
    return rest, *[subs[s] for s in keys]

def prefix(d, pre):
    return {f'{pre}_{k}': v for k, v in d.items()}

def display(x, **kwargs):
    if type(x) is not SVG:
        x = SVG([x], **kwargs)
    return x.svg()

def dedict(d, default=None):
    if type(d) is dict:
        d = [(k, v) for k, v in d.items()]
    return [
        (x if type(x) is tuple else (x, default)) for x in d
    ]

##
## math tools
##

def cumsum(a):
    tot = 0
    for x in a:
        tot += x
        yield tot

##
## rect tools
##

def pos_rect(r):
    if r is None:
        return frac_base
    elif type(r) is not tuple:
        return (0, 0, r, r)
    elif len(r) == 2:
        rx, ry = r
        return (0, 0, rx, ry)
    else:
        return r

def pad_rect(p, base=frac_base):
    xa, ya, xb, yb = base
    if p is None:
        return base
    elif type(p) is not tuple:
        return (xa+p, ya+p, xb-p, yb-p)
    elif len(p) == 2:
        px, py = p
        return (xa+px, ya+py, xb-px, yb-py)
    else:
        pxa, pya, pxb, pyb = p
        return (xa+pxa, ya+pya, xb-pxb, yb-pyb)

def merge_rects(rects):
    xa, ya, xb, yb = zip(*rects)
    return min(xa), min(ya), max(xb), max(yb)

def rect_dims(rect):
    xa, ya, xb, yb = rect
    w, h = xb - xa, yb - ya
    return w, h

def rect_aspect(rect):
    w, h = rect_dims(rect)
    return w/h

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
        attr = props_repr(self.attr)
        return f'{self.tag}: {attr}'

    def __add__(self, other):
        return Container([self, other])

    def __or__(self, other):
        return HStack([self, other])

    def __and__(self, other):
        return VStack([self, other])

    def _repr_svg_(self):
        frame = Frame(self, padding=0.01)
        return SVG(frame).svg()

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx=None):
        if ctx is None:
            ctx = Context()

        props = props_repr(self.props(ctx))
        pre = ' ' if len(props) > 0 else ''

        if self.unary:
            return f'<{self.tag}{pre}{props} />'
        else:
            inner = self.inner(ctx)
            return f'<{self.tag}{pre}{props}>{inner}</{self.tag}>'

    def save(self, fname, **kwargs):
        SVG(self, **kwargs).save(fname)

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
        children = [(c, pos_rect(r)) for c, r in children]
        self.children = children

    def child(self, i):
        return self.children[i][0]

    def inner(self, ctx):
        inside = '\n'.join([c.svg(ctx(r, c.aspect)) for c, r in self.children])
        return f'\n{inside}\n'

class SVG(Container):
    def __init__(self, children=None, size=size_base, clip=True, **attr):
        if children is not None and not isinstance(children, (list, dict)):
            children = [children]
        super().__init__(children=children, tag='svg', **attr)

        if clip:
            ctx = Context(rect=(0, 0, 1, 1))
            rects = [ctx(r, aspect=c.aspect).rect for c, r in self.children]
            total = merge_rects(rects)
            aspect = rect_aspect(total)
        else:
            aspect = 1

        if type(size) is not tuple:
            size = size*sqrt(aspect), size/sqrt(aspect)

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
        return Element.svg(self, ctx=ctx)

    def save(self, path):
        s = self.svg()
        with open(path, 'w+') as fid:
            fid.write(s)

##
## layouts
##

class Box(Container):
    def __init__(self, children, **attr):
        super().__init__(children=children, tag='g', **attr)

class Frame(Container):
    def __init__(self, child, padding=0, margin=0, border=None, aspect=None, **attr):
        mrect = pad_rect(margin)
        prect = pad_rect(padding)
        trect = pad_rect(padding, base=mrect)

        children = [(child, trect)]
        if border is not None:
            attr, rect_args = dispatch(attr, ['rect'])
            if child.aspect is not None:
                pw, ph = rect_dims(prect)
                raspect = child.aspect*(ph/pw)
            else:
                raspect = None
            rect = Rect(stroke_width=border, aspect=raspect, **rect_args)
            children += [(rect, mrect)]

        if aspect is None and child.aspect is not None:
            tw, th = rect_dims(trect)
            aspect = child.aspect*(th/tw)

        super().__init__(children=children, aspect=aspect, **attr)

class Point(Container):
    def __init__(self, child, x=0.5, y=0.5, r=0.5, aspect=None, **attr):
        if type(r) is not tuple:
            rx, ry = r, r
        else:
            rx, ry = r
        aspect = child.aspect if aspect is None else aspect
        children = [(child, (x-rx, y-ry, x+rx, y+ry))]
        super().__init__(children=children, aspect=aspect, **attr)

# TODO
class Scatter(Container):
    pass

class VStack(Container):
    def __init__(self, children, expand=True, aspect=None, **attr):
        n = len(children)
        children, heights = zip(*dedict(children, 1/n))
        aspects = np.array([(c.aspect if c.aspect is not None else n) for c in children])

        heights = np.array(heights)
        if expand:
            heights /= aspects
        heights /= np.sum(heights)

        cheights = np.r_[0, np.cumsum(heights)]
        children = [
            (c, (0, fh0, 1, fh1)) for c, fh0, fh1 in zip(children, cheights[:-1], cheights[1:])
        ]

        aspect0 = np.max(heights*aspects)
        aspect = aspect0 if aspect is None else aspect

        super().__init__(children=children, aspect=aspect, **attr)

class HStack(Container):
    def __init__(self, children, expand=True, aspect=None, **attr):
        n = len(children)
        children, widths = zip(*dedict(children, 1/n))
        aspects = np.array([(c.aspect if c.aspect is not None else n) for c in children])

        widths = np.array(widths)
        if expand:
            widths *= aspects
        widths /= np.sum(widths)

        cwidths = np.r_[0, np.cumsum(widths)]
        children = [
            (c, (fw0, 0, fw1, 1)) for c, fw0, fw1 in zip(children, cwidths[:-1], cwidths[1:])
        ]

        aspect0 = 1/np.max(widths/aspects)
        aspect = aspect0 if aspect is None else aspect

        super().__init__(children=children, aspect=aspect, **attr)

# TODO
class Grid:
    pass

##
## geometric
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
    def __init__(self, **attr):
        base = dict(fill='none', stroke='black')
        attr1 = base | attr
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

class Bullet(Circle):
    def __init__(self, **attr):
        base = dict(fill='black')
        attr1 = base | attr
        super().__init__(**attr1)

##
## text
##

class Text(Element):
    def __init__(self, text='', font_family=default_font_family, **attr):
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
    def __init__(self, text='', font_family=default_font_family, **attr):
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

class Node(Container):
    def __init__(self, text=None, pad=0.15, shape=Rect, text_args={}, shape_args={}, **attr):
        attr, text_args, shape_args = dispatch(attr, ['text', 'shape'])

        label = Text(text=text, **text_args)
        outer = shape(**shape_args)

        aspect0 = label.aspect
        aspect1 = aspect0*(1+2*pad/aspect0)/(1+2*pad)

        children = {
            label: pad_rect(pad),
            outer: None
        }

        super().__init__(children=children, aspect=aspect1, **attr)

##
## curves
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

##
## axes
##

class HTick(Container):
    def __init__(self, text, thick=1, pad=0.5, text_scale=1, debug=False, **attr):
        attr, text_args = dispatch(attr, ['text'])

        if 'font_weight' not in text_args:
            text_args['font_weight'] = 200

        line = Line(0, 0.5, 1, 0.5, stroke_width=thick)

        if text is None or (type(text) is str and len(text) == 0):
            aspect = 1
            anchor = 0.5
            pad = 0
            children = [line]
        else:
            TextClass = TextDebug if debug else Text
            text = TextClass(text, **text_args) if type(text) is str else text
            tsize = text.aspect*text_scale

            width = tsize + pad + 1
            height = text_scale
            aspect = width/height
            anchor = 1 - 0.5/width

            children = {
                text: (0, 0, tsize/width, 1),
                line: ((tsize+pad)/width, 0, 1, 1)
            }

        super().__init__(children=children, aspect=aspect, **attr)
        self.anchor = anchor
        self.width = width
        self.height = height

class VTick(Container):
    def __init__(self, text, thick=1, pad=0.5, text_scale=1, debug=False, **attr):
        attr, text_args = dispatch(attr, ['text'])

        if 'font_weight' not in text_args:
            text_args['font_weight'] = 200

        line = Line(0.5, 0, 0.5, 1, stroke_width=thick)

        if text is None or (type(text) is str and len(text) == 0):
            aspect = 1
            anchor = 0.5
            pad = 0
            children = [line]
        else:
            TextClass = TextDebug if debug else Text
            text = TextClass(text, **text_args) if type(text) is str else text
            taspect = text.aspect*text_scale

            width = taspect
            height = 1 + pad + text_scale
            aspect = width/height
            anchor = 1 - 0.5/height

            children = {
                line: (0, 0, 1, 1/height),
                text: (0, (1+pad)/height, 1, 1)
            }

        super().__init__(children=children, aspect=aspect, **attr)
        self.anchor = anchor
        self.width = width
        self.height = height

class VScale(Container):
    def __init__(self, ticks, tick_size=default_tick_size, tick_args={}, **attr):
        if type(ticks) is dict:
            ticks = [(k, v) for k, v in ticks.items()]

        elems = [HTick(s, **tick_args) for _, s in ticks]
        locs = [x for x, _ in ticks]

        # aspect per tick and overall
        width0 = max([e.width for e in elems]) if len(elems) > 0 else 1
        aspect0 = max([e.aspect for e in elems]) if len(elems) > 0 else 1
        aspect = width0/(1/tick_size)

        # middle of the tick (fractional)
        anchor = (width0-0.5)/width0

        children = {
            e: (1-e.width/width0, 1-(x-tick_size/2), 1, 1-(x+tick_size/2))
            for e, x in zip(elems, locs)
        }

        super().__init__(children=children, aspect=aspect, **attr)
        self.anchor = anchor

class HScale(Container):
    def __init__(self, ticks, tick_size=default_tick_size, tick_args={}, **attr):
        if type(ticks) is dict:
            ticks = [(k, v) for k, v in ticks.items()]

        elems = [VTick(s, **tick_args) for _, s in ticks]
        locs = [x for x, _ in ticks]

        # aspect per tick and overall
        height0 = max([e.height for e in elems]) if len(elems) > 0 else 1
        aspect0 = max([e.aspect for e in elems]) if len(elems) > 0 else 1
        aspect = (1/tick_size)/height0

        # middle of the tick (fractional)
        anchor = 0.5/height0

        children = {
            e: (x-e.aspect/aspect/2, 1-e.height/height0, x+e.aspect/aspect/2, 1)
            for e, x in zip(elems, locs)
        }

        super().__init__(children=children, aspect=aspect, **attr)
        self.anchor = anchor

class VAxis(Container):
    def __init__(self, ticks, tick_size=default_tick_size, **attr):
        attr, tick_args = dispatch(attr, ['tick'])
        scale = VScale(ticks, tick_size=tick_size, tick_args=tick_args)
        line = Line(scale.anchor, 0, scale.anchor, 1)
        super().__init__(children=[scale, line], aspect=scale.aspect, **attr)
        self.anchor = scale.anchor

class HAxis(Container):
    def __init__(self, ticks, tick_size=default_tick_size, **attr):
        attr, tick_args = dispatch(attr, ['tick'])
        scale = HScale(ticks, tick_size=tick_size, tick_args=tick_args)
        line = Line(0, scale.anchor, 1, scale.anchor)
        super().__init__(children=[scale, line], aspect=scale.aspect, **attr)
        self.anchor = scale.anchor

class Axes(Container):
    def __init__(self, xticks=[], yticks=[], aspect=None, **attr):
        attr, xaxis_args, yaxis_args = dispatch(attr, ['xaxis', 'yaxis'])

        # adjust tick_size for aspect
        if aspect is not None:
            xtick_size = xaxis_args.get('tick_size', default_tick_size)
            ytick_size = yaxis_args.get('tick_size', default_tick_size)
            xaxis_args['tick_size'] = xtick_size/sqrt(aspect)
            yaxis_args['tick_size'] = ytick_size*sqrt(aspect)

        if xticks is not None:
            xaxis = HAxis(xticks, **xaxis_args)
            fx = 1 - xaxis.anchor
            ax0 = xaxis.aspect
        else:
            xaxis = None
            fx = 1
            ax0 = 1

        if yticks is not None:
            yaxis = VAxis(yticks, **yaxis_args)
            fy = yaxis.anchor
            ay0 = yaxis.aspect
        else:
            yaxis = None
            fy = 1
            ay0 = 0

        # sans anchor aspects
        ax1 = ax0/fx
        ay1 = fy*ay0

        # square aspect?
        aspect0 = ax1*(1+ay1)/(1+ax1)
        aspect = aspect0 if aspect is None else aspect

        # constraints
        # 1: a0 = fy*wy + wx
        # 2: 1 = hy + fx*hx
        # 3: ax = wx/hx
        # 4: ay = wy/hy

        # get axes sizes
        hy = (ax1-aspect)/(ax1-ay1)
        wy = ay0*hy/aspect
        hx = (1-hy)/fx
        wx = ax0*hx/aspect

        # compute anchor point
        cx = fy*wy
        cy = 1 - fx*hx

        children = {}
        if xaxis is not None:
            children[xaxis] = (1-wx, 1-hx, 1, 1)
        if yaxis is not None:
            children[yaxis] = (0, 0, wy, hy)

        super().__init__(children=children, aspect=aspect, **attr)
        self.anchor = cx, cy

# TODO: refurbish this
class Plot(Container):
    def __init__(self, lines=None, xlim=None, ylim=None, **attr):
        self.axes = Axes()
        self.lines = lines if lines is not None else []

        super().__init__(**attr)
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
