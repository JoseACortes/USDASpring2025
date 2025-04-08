#!/bin/bash

# Update tlmgr
tlmgr update --self

# Install required LaTeX packages
tlmgr install \
    amsmath \
    amsfonts \
    amssymb \
    amsthm \
    graphicx \
    xcolor \
    comment \
    mathrsfs \
    multirow \
    array \
    hyperref \
    multicol \
    ragged2e \
    caption \
    babel-english \
    rotating \
    enumerate \
    tikz \
    bm \
    csquotes \
    biblatex

echo "All required LaTeX packages have been installed."