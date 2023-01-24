import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount=0.5):  
    # --------------------- SOURCE: @IanHincks ---------------------
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# Define colors

leiden = '#001158'
leiden_light = '#8088AC'

science = '#f46e32'
science_light = '#FAB799'

galapagos = '#004C45'
galapagos_light = '#80A6A2'

purple = '#993366'
purple_light = '#CC99B3'

my_colors = [leiden, science, galapagos, purple]
my_light_colors = [leiden_light, science_light, galapagos_light, purple_light ]



