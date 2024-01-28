#!/bin/sh
rm -f *.spv
for ENTRYPOINT in *.comp.glsl; do
    ENTRYPOINT_NAME=$(basename $ENTRYPOINT | cut -f 1 -d ".")
    echo "Compiling $ENTRYPOINT_NAME..."
    glslangValidator -V -g0 $ENTRYPOINT -e $ENTRYPOINT_NAME \
	-o ${ENTRYPOINT_NAME}.spv
done
