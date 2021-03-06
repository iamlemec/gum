################################
## gum — an svg diagram maker ##
################################

import os
import copy
import numpy as np
from collections import defaultdict
from math import sqrt, tan, pi, inf, isinf

from .fonts import get_text_size, get_text_shape

##
## defaults
##

# namespace
ns_svg = 'http://www.w3.org/2000/svg'

# sizing
size_base = 250
rect_base = (0, 0, 100, 100)
frac_base = (0, 0, 1, 1)
prec_base = 13

# specific elements
default_tick_size = 0.05
default_nticks = 5
default_font_family = 'Montserrat'
default_emoji_font = 'NotoColorEmoji'
default_font_weight = 'Regular'

##
## basic tools
##

def demangle(k):
    return k.replace('_', '-')

def rounder(x, prec=prec_base):
    if type(x) is str and x.endswith('px'):
        x1 = x[:-2]
        if x1.replace('.', '', 1).isnumeric():
            suf = 'px'
            x = float(x1)
    else:
        suf = ''
    if isinstance(x, (float, np.floating)):
        xr = round(x, ndigits=prec)
        if (xr % 1) == 0:
            ret = int(xr)
        else:
            ret = xr
    else:
        ret = x
    return str(ret) + suf

def props_repr(d, prec=prec_base):
    return ' '.join([
        f'{demangle(k)}="{rounder(v, prec=prec)}"' for k, v in d.items()
    ])

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

def dedict(d, default=None):
    if type(d) is dict:
        d = [(k, v) for k, v in d.items()]
    return [
        (x if type(x) is tuple else (x, default)) for x in d
    ]

def cumsum(a, zero=True):
    s = 0
    c = [0] if zero else []
    for x in a:
        s += x
        c.append(s)
    return c

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

def rad_rect(p, default=None):
    if len(p) == 1:
        r, = p
        x, y = 0.5, 0.5
        rx, ry = r, r
    elif len(p) == 2:
        x, y = p
        rx, ry = default, default
    elif len(p) == 3:
        x, y, r = p
        rx, ry = r, r
    elif len(p) == 4:
        x, y, rx, ry = p
    return (x-rx, y-ry, x+rx, y+ry)

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

# prect — outer rect (absolute)
# frect — inner rect (fraction)
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
    def __init__(self, rect=rect_base, prec=prec_base, **kwargs):
        self.rect = rect
        self.prec = prec
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
        kwargs1 = self.__dict__ | kwargs
        return Context(**kwargs1)

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
        return HStack([self, other], expand=True)

    def __xor__(self, other):
        return HStack([self, other], expand=False)

    def __and__(self, other):
        return VStack([self, other], expand=True)

    def __mul__(self, other):
        return VStack([self, other], expand=False)

    def _repr_svg_(self):
        frame = Frame(self, padding=0.01)
        return SVG(frame).svg()

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx=None, prec=prec_base):
        if ctx is None:
            ctx = Context(prec=prec)

        props = props_repr(self.props(ctx), prec=ctx.prec)
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

    def inner(self, ctx):
        inside = '\n'.join([c.svg(ctx(r, c.aspect)) for c, r in self.children])
        return f'\n{inside}\n'

# this can have an aspect, which is utilized by layouts
class Spacer(Element):
    def __init__(self, aspect=None, **attr):
        super().__init__(tag=None, aspect=aspect, **attr)

    def svg(self, ctx=None):
        return ''

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
            if aspect >= 1:
                size = size, size/aspect
            else:
                size = size*aspect, size

        self.size = size

    def _repr_svg_(self):
        return self.svg()

    def props(self, ctx):
        w, h = self.size
        base = dict(width=w, height=h, xmlns=ns_svg)
        return base | self.attr

    def svg(self, prec=prec_base):
        rect0 = (0, 0) + self.size
        ctx = Context(rect=rect0, prec=prec)
        return Element.svg(self, ctx=ctx)

    def save(self, path):
        s = self.svg()
        with open(path, 'w+') as fid:
            fid.write(s)

##
## layouts
##

class Box(Container):
    def __init__(self, children, aspect=None, **attr):
        super().__init__(children=children, aspect=aspect, **attr)

class Frame(Container):
    def __init__(self, child, padding=0, margin=0, border=None, aspect=None, **attr):
        mrect = pad_rect(margin)
        prect = pad_rect(padding)
        trect = pad_rect(padding, base=mrect)

        children = []

        if border is not None:
            attr, rect_args = dispatch(attr, ['rect'])
            if aspect is None and child.aspect is not None:
                pw, ph = rect_dims(prect)
                raspect = child.aspect*(ph/pw)
            else:
                raspect = aspect
            rect = Rect(stroke_width=border, aspect=raspect, **rect_args)
            children += [(rect, mrect)]

        children += [(child, trect)]

        if aspect is None and child.aspect is not None:
            tw, th = rect_dims(trect)
            aspect = child.aspect*(th/tw)

        super().__init__(children=children, aspect=aspect, **attr)

class Point(Container):
    def __init__(self, child, x=0.5, y=0.5, r=0.5, aspect=None, **attr):
        if type(r) is not tuple:
            r = r,
        pos = x, y, *r
        aspect = child.aspect if aspect is None else aspect
        children = [(child, rad_rect(pos))]
        super().__init__(children=children, aspect=aspect, **attr)

class Scatter(Container):
    def __init__(self, locs, r=None, **attr):
        locs = dedict(locs)
        children = [
            (child, rad_rect(pos, default=r)) for child, pos in locs
        ]
        super().__init__(children=children, **attr)

class VStack(Container):
    def __init__(self, children, expand=True, aspect=None, **attr):
        n = len(children)
        children, heights = zip(*dedict(children, 1/n))
        aspects = [c.aspect for c in children]

        if expand:
            heights = [h/(a or 1) for h, a in zip(heights, aspects)]
            total = sum(heights)
            heights = [h/total for h in heights]

        cheights = cumsum(heights)
        children = [
            (c, (0, fh0, 1, fh1)) for c, fh0, fh1 in zip(children, cheights[:-1], cheights[1:])
        ]

        aspects0 = [h*a for h, a in zip(heights, aspects) if a is not None]
        aspect0 = max(aspects0) if len(aspects0) > 0 else None
        aspect = aspect0 if aspect is None else aspect

        super().__init__(children=children, aspect=aspect, **attr)

class HStack(Container):
    def __init__(self, children, expand=True, aspect=None, **attr):
        n = len(children)
        children, widths = zip(*dedict(children, 1/n))
        aspects = [c.aspect for c in children]

        if expand:
            widths = [w*(a or 1) for w, a in zip(widths, aspects)]
            total = sum(widths)
            widths = [w/total for w in widths]

        cwidths = cumsum(widths)
        children = [
            (c, (fw0, 0, fw1, 1)) for c, fw0, fw1 in zip(children, cwidths[:-1], cwidths[1:])
        ]

        aspects0 = [a/w for w, a in zip(widths, aspects) if a is not None]
        aspect0 = min(aspects0) if len(aspects0) > 0 else None
        aspect = aspect0 if aspect is None else aspect

        super().__init__(children=children, aspect=aspect, **attr)

# TODO
class Grid:
    pass

##
## geometric
##

class Ray(Element):
    def __init__(self, theta=-45, **attr):
        if theta == -90:
            theta = 90
        elif theta < -90 or theta > 90:
            theta = ((theta + 90) % 180) - 90
        direc0 = tan(theta*(pi/180))

        if theta == 90:
            direc = inf
            aspect = None
        elif theta == 0:
            direc = 0
            aspect = None
        else:
            direc = direc0
            aspect = 1/abs(direc0)

        attr1 = dict(stroke='black') | attr
        super().__init__(tag='line', aspect=aspect, unary=True, **attr1)
        self.direc = direc

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1
        if isinf(self.direc):
            x1 = x2 = x1 + 0.5*w
        elif self.direc == 0:
            y1 = y2 = y1 + 0.5*h
        elif self.direc > 0:
            y1, y2 = y2, y1
        base = dict(x1=x1, y1=y1, x2=x2, y2=y2)
        return base | self.attr

class VLine(Element):
    def __init__(self, pos=0.5, aspect=None, **attr):
        attr1 = dict(stroke='black') | attr
        super().__init__(tag='line', unary=True, aspect=aspect, **attr1)
        self.pos = pos

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        x1 = x2 = x1 + self.pos*w

        base = dict(x1=x1, y1=y1, x2=x2, y2=y2)
        return base | self.attr

class HLine(Element):
    def __init__(self, pos=0.5, aspect=None, **attr):
        attr1 = dict(stroke='black') | attr
        super().__init__(tag='line', unary=True, aspect=aspect, **attr1)
        self.pos = pos

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        y1 = y2 = y1 + self.pos*h

        base = dict(x1=x1, y1=y1, x2=x2, y2=y2)
        return base | self.attr

class Rect(Element):
    def __init__(self, **attr):
        attr1 = dict(fill='none', stroke='black') | attr
        super().__init__(tag='rect', unary=True, **attr1)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w0, h0 = x2 - x1, y2 - y1

        x, y = x1, y1
        w, h = w0, h0

        base = dict(x=x, y=y, width=w, height=h)
        return base | self.attr

class Square(Rect):
    def __init__(self, **attr):
        super().__init__(aspect=1, **attr)

class Ellipse(Element):
    def __init__(self, **attr):
        attr1 = dict(fill='none', stroke='black') | attr
        super().__init__(tag='ellipse', unary=True, **attr1)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        cx = x1 + 0.5*w
        cy = y1 + 0.5*h
        rx = 0.5*w
        ry = 0.5*h

        base = dict(cx=cx, cy=cy, rx=rx, ry=ry)
        return base | self.attr

class Circle(Ellipse):
    def __init__(self, **attr):
        super().__init__(aspect=1, **attr)

class Bullet(Circle):
    def __init__(self, **attr):
        attr1 = dict(fill='black') | attr
        super().__init__(**attr1)

##
## text
##

def weight_map(x):
    if type(x) is str:
        if x == 'light':
            return 'Light', 200
        elif x == 'Bold':
            return 'Bold', 700
        else:
            return 'Regular', 400
    elif type(x) is int:
        if x <= 300:
            return 'Light', x
        elif x <= 550:
            return 'Regular', x
        else:
            return 'Bold', x
    else:
        return 'Regular', 400

class Text(Element):
    def __init__(
        self, text='', font_family=default_font_family,
        font_weight=default_font_weight, **attr
    ):
        str_weight, num_weight = weight_map(font_weight)

        self.text_width, self.text_height = get_text_size(
            text, font=font_family, weight=str_weight
        )
        self.text = text

        base_aspect = self.text_width/self.text_height
        super().__init__(
            tag='text', aspect=base_aspect, font_family=font_family,
            font_weight=num_weight, **attr
        )

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        fs = h/self.text_height

        base = dict(x=x1, y=y2, font_size=f'{fs}px', stroke='black')
        return base | self.attr

    def inner(self, ctx):
        return self.text

class Emoji(Element):
    def __init__(self, text='', font_family=default_emoji_font, **attr):
        self.text_width, self.text_height = get_text_size(text, font=font_family)
        self.text = text

        base_aspect = self.text_width/self.text_height
        super().__init__(tag='text', aspect=base_aspect, font_family=font_family, **attr)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1
        fs = h/self.text_height

        # magic offsets
        x0, y0 = x1, y2 - 0.125*h
        fs0 = 1.25*fs

        base = dict(x=x0, y=y0, font_size=f'{fs0}px', stroke='black')
        return base | self.attr

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
        cluster, shapes, deltas, offsets = get_text_shape(text, font=font_family)
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

        super().__init__(children=children, aspect=label.aspect, **attr)

class Node(Container):
    def __init__(self, text, padding=0.2, shape=Rect, debug=False, **attr):
        attr, text_args, shape_args = dispatch(attr, ['text', 'shape'])

        # generate core elements
        if type(text) is str:
            TextClass = TextDebug if debug else Text
            text = TextClass(text=text, **text_args)
        outer = shape(**shape_args)

        # auto-scale single number padding
        aspect0 = text.aspect
        if type(padding) is not tuple:
            padding = padding/aspect0, padding
        aspect1 = (aspect0+2*padding[0])/(1+2*padding[1])

        children = {
            text: pad_rect(padding),
            outer: None
        }

        super().__init__(children=children, aspect=aspect1, **attr)

##
## curves
##

class Polygon(Element):
    def __init__(self, points, **attr):
        self.points = points
        super().__init__(tag='polygon', unary=True, **attr)

    def props(self, ctx):
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        pc = [(x1 + fx*w, y1 + fy*h) for fx, fy in self.points]
        points = ' '.join([f'{x},{y}' for x, y in pc])

        base = dict(points=points, fill='none', stroke='black')
        return base | self.attr

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
        x1, y1, x2, y2 = ctx.rect
        w, h = x2 - x1, y2 - y1

        xc = x1 + self.xnorm*w
        yc = y1 + self.ynorm*h
        points = ' '.join([f'{x},{y}' for x, y in zip(xc, yc)])

        base = dict(points=points, fill='none', stroke='black')
        return base | self.attr

##
## axes
##

class HTick(Container):
    def __init__(self, text, thick=1, pad=0.5, text_scale=1, debug=False, **attr):
        attr, text_args = dispatch(attr, ['text'])

        if 'font_weight' not in text_args:
            text_args['font_weight'] = 'light'

        line = HLine(stroke_width=thick)

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
            text_args['font_weight'] = 'light'

        line = VLine(stroke_width=thick)

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
        line = VLine(scale.anchor)
        super().__init__(children=[scale, line], aspect=scale.aspect, **attr)
        self.anchor = scale.anchor

class HAxis(Container):
    def __init__(self, ticks, tick_size=default_tick_size, **attr):
        attr, tick_args = dispatch(attr, ['tick'])
        scale = HScale(ticks, tick_size=tick_size, tick_args=tick_args)
        line = HLine(scale.anchor)
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

class Plot(Container):
    def __init__(self, lines, xlim=None, ylim=None, xticks=None, yticks=None, aspect=None, **attr):
        attr, xaxis_args, yaxis_args = dispatch(attr, ['xaxis', 'yaxis'])

        # allow singleton lines
        if type(lines) is not list:
            lines = [lines]

        # collect line ranges
        xmins, xmaxs = zip(*[c.xlim for c in lines])
        ymins, ymaxs = zip(*[c.ylim for c in lines])

        # determine coordinate limits
        if xlim is None:
            xmin, xmax = min(xmins), max(xmaxs)
        else:
            xmin, xmax = xlim
        if ylim is None:
            ymin, ymax = min(ymins), max(ymaxs)
        else:
            ymin, ymax = ylim

        # x/y coordinate functions
        xmap = lambda x: (x-xmin)/(xmax-xmin)
        ymap = lambda y: (y-ymin)/(ymax-ymin)

        # map lines into coordinates
        coords = [
            (xmap(x1), 1-ymap(y2), xmap(x2), 1-ymap(y1))
            for x1, x2, y1, y2 in zip(xmins, xmaxs, ymins, ymaxs)
        ]

        # construct/map ticks if needed
        if xticks is None or type(xticks) is int:
            xtick_num = xticks if xticks is not None else default_nticks
            xtick_vals = np.linspace(xmin, xmax, xtick_num)
            xticks = {xmap(x): str(f'{x:.2f}') for x in xtick_vals}
        else:
            if type(xticks) is list:
                xticks = {x: str(x) for x in xticks}
            xticks = {xmap(x): t for x, t in xticks.items()}
        if yticks is None or type(yticks) is int:
            ytick_num = yticks if yticks is not None else default_nticks
            ytick_vals = np.linspace(ymin, ymax, ytick_num)
            yticks = {ymap(y): str(f'{y:.2f}') for y in ytick_vals}
        else:
            if type(yticks) is list:
                yticks = {y: str(y) for y in yticks}
            yticks = {ymap(y): t for y, t in yticks.items()}

        # create axes
        axis_args = prefix(xaxis_args, 'xaxis') | prefix(yaxis_args, 'yaxis')
        axes = Axes(xticks=xticks, yticks=yticks, aspect=aspect, **axis_args)

        # map lines into plot area
        lbox = (axes.anchor[0], 0, 1, axes.anchor[1])
        children = {
            ln: map_coords(lbox, cr) for ln, cr in zip(lines, coords)
        }
        children[axes] = None

        super().__init__(children=children, aspect=axes.aspect, **attr)
