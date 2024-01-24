#!/bin/sh
rm -f src/correlation/*.spv
for ENTRYPOINT in src/correlation/*.comp.glsl; do
    ENTRYPOINT_NAME=$(basename $ENTRYPOINT | cut -f 1 -d ".")
    echo "Compiling $ENTRYPOINT_NAME..."
    glslangValidator -V -g0 $ENTRYPOINT -e $ENTRYPOINT_NAME \
	-o src/correlation/${ENTRYPOINT_NAME}.spv
done
