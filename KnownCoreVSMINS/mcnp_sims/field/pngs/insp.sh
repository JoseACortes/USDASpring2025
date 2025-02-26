
i="0_0"
rm plot.ps comout outp
mcnp6 ip inp=../compute/input/$i.txt notek com=com plotm = plot
ps2pdf plot.ps
convert -density 300 plot.pdf -quality 90 $i.png
