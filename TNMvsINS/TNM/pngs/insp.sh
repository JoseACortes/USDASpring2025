
i="tnm"
rm plot.ps comout outp
mcnp6 ip inp=../input/$i.inp notek com=com plotm = plot
ps2pdf plot.ps
convert -density 300 plot.pdf -quality 90 $i.png
