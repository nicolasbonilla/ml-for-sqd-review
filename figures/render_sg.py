# run with: /usr/bin/python3 render_sg.py  (system python has pymol)
import pymol
pymol.finish_launching(['pymol', '-qc'])
from pymol import cmd, util

def setup():
    cmd.bg_color("white"); cmd.set("orthoscopic", 1); cmd.set("ray_shadows", 1)
    cmd.set("ray_shadow_decay_factor", 0.1); cmd.set("ambient", 0.42); cmd.set("direct", 0.55)
    cmd.set("specular", 0.25); cmd.set("shininess", 55); cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_color", "black")
    cmd.set("ray_opaque_background", 0); cmd.set("surface_quality", 2)
    cmd.set("transparency", 0.22); cmd.set("two_sided_lighting", 1)
    cmd.set("sphere_scale", 0.26); cmd.set("stick_radius", 0.13); cmd.set("stick_color", "grey30")

def render(xyz, cube, out, iso=0.028, size=(1500, 1250)):
    cmd.delete("all"); cmd.load(xyz, "mol"); cmd.hide("everything")
    cmd.show("sticks", "mol"); cmd.show("spheres", "mol"); util.cbaw("mol")
    cmd.load(cube, "omap"); cmd.isosurface("sp", "omap", iso); cmd.isosurface("sn", "omap", -iso)
    cmd.set("surface_color", "orange", "sp"); cmd.set("surface_color", "marine", "sn")
    cmd.show("surface", "sp"); cmd.show("surface", "sn")
    cmd.orient("mol"); cmd.turn("x", -12); cmd.turn("y", 8); cmd.zoom("mol", 2.6)
    cmd.ray(size[0], size[1]); cmd.png(out, dpi=300); print("wrote", out, flush=True)

setup()
render("/w/n2.xyz", "/w/n2_sg_true.cube", "/w/o_n2_sg_true.png")
print("sg render done")
