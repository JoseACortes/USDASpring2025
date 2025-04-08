#!/bin/bash
BLENDER_EXEC="blender"  # Change this to your actual Blender executable
SCRIPT="mcnp_geometry.py"

"$BLENDER_EXEC" --background --python "$SCRIPT"