##
## font shaping interface
##

import gi
gi.require_version('HarfBuzz', '0.0')
from gi.repository import HarfBuzz as hb
from gi.repository import GLib

import os
import fontconfig as fc

unit = 64 # this seems to get us to 12 point font

def get_font_path(name=''):
    conf = fc.Config.get_current()
    pat = fc.Pattern.name_parse(name)
    font, stat = conf.font_match(pat)
    path, code = font.get(fc.PROP.FILE, 0)
    return path

def get_text_shape(text, size=12, font='', path=None):
    if path is None:
        path = get_font_path(font)
    base, ext = os.path.splitext(path)

    with open(path, 'rb') as fid:
        fontdata = fid.read()

    bdata = GLib.Bytes.new(fontdata)
    blob = hb.glib_blob_create(bdata)
    face = hb.face_create(blob, 0)

    font = hb.font_create(face)
    upem = hb.face_get_upem(face)
    hb.font_set_scale(font, upem, upem)
    # hb.font_set_ptem(font, size)

    if ext == '.ttf':
        hb.ft_font_set_funcs(font)
    elif ext == '.otf':
        hb.ot_font_set_funcs(font)

    buf = hb.buffer_create()
    hb.buffer_add_utf8(buf, text.encode('utf-8'), 0, -1)

    hb.buffer_guess_segment_properties(buf)
    hb.shape(font, buf, [])
    infos = hb.buffer_get_glyph_infos(buf)
    positions = hb.buffer_get_glyph_positions(buf)

    norm = upem/size
    wh_extract = lambda ext: (ext.extents.width / norm, -ext.extents.height / norm)
    shapes = [wh_extract(hb.font_get_glyph_extents(font, i.codepoint)) for i in infos]
    deltas = [(p.x_advance / norm, p.y_advance / norm) for p in positions]
    offsets = [(p.x_offset / norm, p.y_offset / norm) for p in positions]

    return shapes, deltas, offsets
