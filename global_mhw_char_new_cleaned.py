# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script visualizes four global characteristics of marine heatwave (MHW) events:
# mean SST, frequency, duration, and intensity. It uses gridded data and plots a 
# 2×2 panel map with appropriate projections and color bars for each metric.
#
# Outputs:
# - A 2x2 panel global map saved as 'MHW_Characteristics_2x2_ed.pdf'
#   showing:
#     • Mean SST of MHWs
#     • Mean frequency of MHWs
#     • Mean duration of MHWs
#     • Mean relative intensity (i_max_rel) of MHWs
#
# For a detailed description of the methodologies and further insights, please refer to:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2024). 
# *Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide*.
# Science Advances.
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

# Load your grouped data
grouped_data = pd.read_csv('.../grouped_mhw_data_new.csv')

# Create a 2x2 grid of subplots with less spacing using GridSpec
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.1)

# List of plot parameters
plot_params = [
    {
        'data': 'mean_sst',
        'title': 'Mean SST of MHW Events',
        'cmap': 'coolwarm',
        'levels': [-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30],
        'cbar_label': '°C',
        'cbar_ticks': [-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30],
        'cbar_ticklabels': ['-2.5', '0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20', '22.5', '25', '27.5', '>30']
    },
    {
        'data': 'mean_events_per_year',
        'title': 'Mean Frequency of MHW Events',
        'cmap': 'OrRd',
        'levels': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'cbar_label': 'events per year',
        'cbar_ticks': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'cbar_ticklabels': ['0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '>2.0']
    },
    {
        'data': 'mean_duration_per_year',
        'title': 'Mean Duration of MHW Events',
        'cmap': 'PuOr_r',
        'levels': [5, 10, 15, 20, 25, 30],
        'cbar_label': 'days per year',
        'cbar_ticks': [5, 10, 15, 20, 25, 30],
        'cbar_ticklabels': ['5', '10', '15', '20', '25', '>30']
    },
    {
        'data': 'mean_intensity',
        'title': 'Mean Intensity (i_max_rel) of MHW Events',
        'cmap': 'RdYlBu_r',
        'levels': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        'cbar_label': '°C above threshold',
        'cbar_ticks': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        'cbar_ticklabels': ['0.0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '>2.5']
    }
]

# Generate the subplots using GridSpec
for i, params in enumerate(plot_params):
    ax = fig.add_subplot(gs[i])
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax)
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()

    parallels = np.arange(-90., 91., 30.)
    meridians = np.arange(-180., 181., 60.)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=15, color='gray', dashes=[1,3])
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=15, color='gray', dashes=[1,3])

    x, y = m(grouped_data['lon'].values, grouped_data['lat'].values)
    xi = np.linspace(grouped_data['lon'].min(), grouped_data['lon'].max(), 500)
    yi = np.linspace(grouped_data['lat'].min(), grouped_data['lat'].max(), 500)
    xi, yi = np.meshgrid(xi, yi)
    x_grid, y_grid = m(xi, yi)
    zi = griddata((x, y), grouped_data[params['data']].values, (x_grid, y_grid), method='cubic')

    if params['title'] == 'Mean SST of MHW Events':
        # Add contours first
        contour = m.contour(x_grid, y_grid, zi, levels=[26.5], colors='black', linewidths=1.5, linestyles='solid')
        m.fillcontinents(color='lightgray', zorder=2)
        m.drawcountries(linewidth=0.5, color='black', zorder=3)
        m.drawcoastlines(linewidth=0.5, color='black', zorder=4)

    # Plot the contourf
    contourf = m.contourf(x_grid, y_grid, zi, levels=params['levels'], cmap=params['cmap'], extend='max', latlon=False)
    m.fillcontinents(color='lightgray', zorder=1)

    # Add colorbar to the existing axis
    cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', pad=0.075, aspect=50, ticks=params['cbar_ticks'])
    cbar.set_label(params['cbar_label'], fontsize=20)
    cbar.ax.set_xticklabels(params['cbar_ticklabels'], fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.set_title(params['title'], fontsize=22)

plt.subplots_adjust(left=0.03, right=0.95, top=0.93, bottom=0.05, wspace=0.1, hspace=0.01)
plt.savefig('.../MHW_Characteristics_2x2_ed.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
