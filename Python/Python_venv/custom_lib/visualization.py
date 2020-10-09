#   Authors: Jessica Leoni (jessica.leoni@polimi.it)
#            Francesco Zinnari (francesco.zinnari@polimi.it)
#            Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
#   Date: 2019/05/03.
#
#   If you are going to use fierClass in your research project, please cite its reference article
#   S. Gelmini, S. Formentin, et al. "fierClass: A multi-signal, cepstrum-based, time series classifier,"
#   Engineering Applications of Artificial Intelligence, Volume 87, 2020, https://doi.org/10.1016/j.engappai.2019.103262.
#
#   Copyright and license: © Jessica Leoni, Francesco Zinnari, Simone Gelmini, Politecnico di Milano
#   Licensed under the [MIT License](LICENSE).
#
#  In case of need, feel free to ask the author.

################################################################
#                           Libraries                          #
################################################################
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Grid, LinearAxis
from bokeh.models import Plot, Step, Legend, LegendItem
from bokeh.models import Title
from bokeh.plotting import figure
from bokeh.palettes import Accent
from bokeh.io import curdoc, output_file, save



################################################################
#                          Cepstra Plot                        #
################################################################
def train_cepstra_plot(order, train_cepstra, label):
	# Color palette definition (one for each classes)
    colors = Accent.get(train_cepstra.shape[0])

    # x-axis definition
    x = list(range(order))

    # Cepstra plots. Each plot contains the cepstrum of
    # the considered signal for each available class.
    plots = [figure() for i in range(train_cepstra.shape[1])]
    glyphs = [plot.line(x, train_cepstra[class_idx,0,:order],
                color=colors[class_idx], line_width=2,
                legend_label=label[class_idx]) 
              for plot in plots 
              for i in range(2)
              for class_idx in range(train_cepstra.shape[0])]

    # Legend settings
    for plot in plots:
        plot.legend.location = "top_right"
        plot.legend.click_policy= "hide"

    output_file("trainData Cepstra Plot.html") 
    save(gridplot(children = plots, ncols = 3, 
    	merge_tools = False, 
        plot_width=300, plot_height=300))


################################################################
#               Predictions vs Actual Labels Plot              #
################################################################
def results_plot(y_test,y_pred,title,featurefusion):
	# x-axis definition
	x = list(range(len(y_test)))

	# Classes vs subclasses evaluation plot
	if(featurefusion!='conv'):
	    source = ColumnDataSource(dict(x=x, 
	    	y1=y_test, y2=y_pred[0]))
	else:
	    source = ColumnDataSource(dict(x=x, 
	    	y1=y_test, y2=y_pred))

	plot = Plot(plot_width=900, plot_height=700,
	            title=Title(text=title, align="center"))

	# Actual labels
	glyph1 = Step(x="x", y="y1", line_width=2, 
		line_color="#f46d43")
	plot.add_glyph(source, glyph1)

	# Predictedd labels
	glyph2 = Step(x="x", y="y2", line_width=2, 
		line_dash="dashed", line_color="#1d91d0")
	plot.add_glyph(source, glyph2)

	xaxis = LinearAxis()
	plot.add_layout(xaxis, 'below')

	yaxis = LinearAxis()
	plot.add_layout(yaxis, 'left')

	# Legend
	li1 = LegendItem(label='True', 
		renderers=[plot.renderers[0]])
	li2 = LegendItem(label='Predicted', 
		renderers=[plot.renderers[1]])
	legend1 = Legend(items=[li1, li2], 
		location='top_right')

	plot.add_layout(Grid(dimension=0, 
		ticker=xaxis.ticker))
	plot.add_layout(Grid(dimension=1, 
		ticker=yaxis.ticker))
	plot.add_layout(legend1)
	plot.legend.click_policy= "hide"

	curdoc().add_root(plot)
	output_file(str(title)+".html") 
	save(plot)