import sys

import matplotlib.pyplot as plt
from pandas import read_csv


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 26,
    # "font.size": 16,
})
cmap = plt.cm.Set1
colors = cmap.colors
titles_2_colors = dict(cold_start='black', 
                       nearest_neighbor=colors[6], 
                       silver=colors[2],
                       nesterov=colors[4],
                       l2ws=colors[3],
                       prev_sol=colors[4],
                       reg_k0=colors[3],
                       reg_k5=colors[0],
                       lasco=colors[1],
                       reg_k30=colors[5],
                       reg_k60=colors[2],
                    #    reg_k120=colors[0],
                       obj_k0=colors[3],
                       obj_k5=colors[0],
                       obj_k15=colors[1],
                       obj_k30=colors[5],
                       obj_k60=colors[2])
                    #    obj_k120='gray')
titles_2_styles = dict(cold_start='-.', 
                       nearest_neighbor='-.',
                       nesterov='-.',
                       silver='-.', 
                       l2ws='-.',
                       prev_sol='-.',
                       reg_k0='-',
                       reg_k5='-',
                       lasco='-',
                       reg_k30='-',
                       reg_k60='-',
                       reg_k120='-',
                       obj_k0='-',
                       obj_k5='-',
                       obj_k15='-',
                       obj_k30='-',
                       obj_k60='-')
                    #    obj_k120='-')
titles_2_markers = dict(cold_start='v', 
                       nearest_neighbor='<', 
                       nesterov='^',
                       silver='D',
                       l2ws='>',
                       prev_sol='^',
                       reg_k0='>',
                       reg_k5='o',
                       lasco='s',
                       reg_k30='x',
                       reg_k60='D',
                    #    reg_k120='-',
                       obj_k0='>',
                       obj_k5='o',
                       obj_k15='s',
                       obj_k30='x',
                       obj_k60='D')
titles_2_marker_starts = dict(cold_start=0, 
                       nearest_neighbor=16, 
                       silver=20,
                       nesterov=23,
                       l2ws=8,
                       prev_sol=23,
                       reg_k0=8,
                       reg_k5=4,
                       lasco=12,
                       reg_k30=0,
                       reg_k60=20,
                    #    reg_k120='-',
                       obj_k0=8,
                       obj_k5=4,
                       obj_k15=12,
                       obj_k30=0,
                       obj_k60=20)

