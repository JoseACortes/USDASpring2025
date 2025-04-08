import bpy
import math
import mathutils

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set scene to use centimeters
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.01

# Geometry creation functions
def create_box(name, origin, dx, dy, dz):
    # origin = mathutils.Vector(origin) - mathutils.Vector((dx, dy, dz))
    bpy.ops.mesh.primitive_cube_add(size=1, location=origin, scale=(dx, dy, dz))
    obj = bpy.context.active_object
    obj.name = name

def create_sphere(name, center, radius):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center)
    bpy.context.active_object.name = name

def create_cylinder(name, base, height_vec, radius):
    vec = mathutils.Vector(height_vec)
    center = mathutils.Vector(base) + vec / 2
    length = vec.length
    direction = vec.normalized()
    up = mathutils.Vector((0, 0, 1))
    axis = up.cross(direction)
    angle = up.angle(direction)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=center)
    obj = bpy.context.active_object
    obj.name = name
    if axis.length > 0:
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = (angle, *axis)

# Create geometry
create_sphere("sph_100000", (0, 0, 0), 800)
create_box("box_5000", (37.5, 0.0, 5.0), 5, 60.1, 20)
create_box("box_6000", (32.5, 0.0, 5.0), 5, 55, 20)
create_box("box_7000", (25.0, 0.0, 5.0), 10, 55, 20)
create_box("box_8000", (17.5, 0.0, 5.0), 5, 55, 20)
create_box("box_9000", (-8.0, 16.25, 5.0), 46, 2.5, 20)
create_box("box_10000", (-8.0, 22.5, 5.0), 46, 10, 20)
create_box("box_11000", (-8.0, -16.25, 5.0), 46, 2.5, 20)
create_box("box_12000", (-8.0, -22.5, 5.0), 46, 10, 20)
create_box("box_2000", (55.0, 14.7, 3.0), 12.7, 12.7, 15.24)
create_box("box_3000", (55.0, 0.35, 3.0), 12.7, 12.7, 15.24)
create_box("box_4000", (55.0, -14.7, 3.0), 12.7, 12.7, 15.24)
create_box("box_1000", (0.0, 0.0, -50.0), 200, 200, 50) # Ground
create_cylinder("rcc_13000", (68, -75.7, 4), (0, 25, 0), 29)
create_cylinder("rcc_14000", (68, -75.7, 4), (0, 25, 0), 15.24)
create_cylinder("rcc_15000", (68, -74.4, 4), (0, 22.4, 0), 27.7)
create_cylinder("rcc_16000", (68, 75.7, 4), (0, -25, 0), 29)
create_cylinder("rcc_17000", (68, 75.7, 4), (0, -25, 0), 15.24)
create_cylinder("rcc_18000", (68, 74.4, 4), (0, -22.4, 0), 27.7)
create_sphere("source", (0, 0, 0), 1)

# Add camera
cam_data = bpy.data.cameras.new(name='Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0, -200, -25)
target = mathutils.Vector((0, 0, 0))
direction = target - cam_obj.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()
bpy.context.scene.camera = cam_obj

# Add light behind the camera
light_data = bpy.data.lights.new(name="Light", type='SUN')
light_data.energy = 10
light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.location = (0, 0, 200)

# Set render engine and settings
scene.render.image_settings.file_format = 'JPEG'
scene.render.filepath = "//render.png"
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Simple blue sky material
sky_mat = bpy.data.materials.new(name="SkyMaterial")
sky_mat.use_nodes = True
bsdf = sky_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0, 0, 1, 1)
bsdf.inputs['Alpha'].default_value = 0.1

# Other materials
wheel_mat = bpy.data.materials.new(name="WheelMaterial")
wheel_mat.use_nodes = True
bsdf = wheel_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0, 0, 0, .2)

shield_mat = bpy.data.materials.new(name="ShieldMaterial")
shield_mat.use_nodes = True
bsdf = shield_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (.5, .5, .5, .5)

detector_mat = bpy.data.materials.new(name="DetectorMaterial")
detector_mat.use_nodes = True
emission = detector_mat.node_tree.nodes.new(type='ShaderNodeEmission')
emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1)
emission.inputs['Strength'].default_value = 1.0
diffuse = detector_mat.node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
diffuse.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1)
mix_shader = detector_mat.node_tree.nodes.new(type='ShaderNodeMixShader')
mix_shader.inputs[0].default_value = 0.5
output = detector_mat.node_tree.nodes.get('Material Output')
links = detector_mat.node_tree.links
links.new(diffuse.outputs['BSDF'], mix_shader.inputs[1])
links.new(emission.outputs['Emission'], mix_shader.inputs[2])
links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

ground_mat = bpy.data.materials.new(name="GroundMaterial")
ground_mat.use_nodes = True
bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0, .05, 0, 1)  # A natural earthy brown

# Assign materials
sky_obj = bpy.data.objects.get("sph_100000")
if sky_obj:
    sky_obj.data.materials.clear()
    sky_obj.data.materials.append(sky_mat)

for name in ["rcc_13000", "rcc_14000", "rcc_15000", "rcc_16000", "rcc_17000", "rcc_18000"]:
    obj = bpy.data.objects.get(name)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(wheel_mat)

for name in [f"box_{i}" for i in [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]]:
    obj = bpy.data.objects.get(name)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(shield_mat)

for name in ["box_2000", "box_3000", "box_4000"]:
    obj = bpy.data.objects.get(name)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(detector_mat)

for name in ["box_1000"]:
    obj = bpy.data.objects.get(name)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(ground_mat)
# Set the world background color
world = bpy.data.worlds['World']
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.5, 0.7, 1, 1)  # Light blue color


# Render the scene
bpy.ops.render.render(write_still=True)