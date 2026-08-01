# run with:  /usr/bin/python3 render_orb.py   (system python has the pymol module)
import pymol
pymol.finish_launching(['pymol', '-qc'])       # headless, quiet, no GUI
from pymol import cmd
from pymol import util

def setup():
    cmd.bg_color("white")
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadows", 1)
    cmd.set("ray_shadow_decay_factor", 0.1)
    cmd.set("ambient", 0.42)
    cmd.set("direct", 0.55)
    cmd.set("specular", 0.25)
    cmd.set("shininess", 55)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)          # crisp black outlines (journal style)
    cmd.set("ray_trace_color", "black")
    cmd.set("ray_opaque_background", 0)
    cmd.set("surface_quality", 2)
    cmd.set("transparency", 0.22)
    cmd.set("two_sided_lighting", 1)
    cmd.set("sphere_scale", 0.26)
    cmd.set("stick_radius", 0.13)
    cmd.set("stick_color", "grey30")

def render(xyz, cube, out, iso=0.028, surf=True, size=(1500, 1250)):
    cmd.delete("all")
    cmd.load(xyz, "mol")                  # atoms (guaranteed selectable)
    cmd.hide("everything")
    cmd.show("sticks", "mol")
    cmd.show("spheres", "mol")
    util.cbaw("mol")                      # colour by element, carbon light-grey
    if surf:
        cmd.load(cube, "omap")            # volumetric map for the isosurface
        cmd.isosurface("sp", "omap", iso)
        cmd.isosurface("sn", "omap", -iso)
        cmd.set("surface_color", "orange", "sp")
        cmd.set("surface_color", "marine", "sn")
        cmd.show("surface", "sp"); cmd.show("surface", "sn")
    cmd.orient("mol")
    cmd.turn("x", -12); cmd.turn("y", 8)
    cmd.zoom("mol", 2.6 if surf else 1.4)
    cmd.ray(size[0], size[1])
    cmd.png(out, dpi=300)
    print("wrote", out, flush=True)

setup()
# N2
render("/w/n2.xyz", None,                "/w/o_n2_struct.png", surf=False)
render("/w/n2.xyz", "/w/n2_piu_homo1.cube","/w/o_n2_pi.png")
render("/w/n2.xyz", "/w/n2_sg_homo.cube",  "/w/o_n2_sg.png")
render("/w/n2.xyz", "/w/n2_pig_lumo.cube", "/w/o_n2_pistar.png")
render("/w/n2.xyz", "/w/n2_su_lumo3.cube", "/w/o_n2_sustar.png")
# H2O
render("/w/h2o.xyz", None,                 "/w/o_h2o_struct.png", surf=False)
render("/w/h2o.xyz", "/w/h2o_b1_homo.cube", "/w/o_h2o_homo.png")
render("/w/h2o.xyz", "/w/h2o_a1_lumo.cube", "/w/o_h2o_lumo.png")
print("all renders done")
