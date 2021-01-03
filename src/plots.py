import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import Image


def inter_dropdown_plot(x,
                        y,
                        data,
                        options,
                        group_column,
                        default_value,
                        dropdown_label,
                        title,
                        xtitle,
                        ytitle,
                        legend_titles=None,
                        output_image=False,
                        blend_legend=False,
                        width=800,
                        height=400,
                        scale=2,
                        traces_visibility=None,
                        modes=None,
                        template='simple_white'):
    """
    Creates interactive plotly plot with a dropdown
    or prints the plot as png.

    Parameters
    ----------

    x : str
        The name of the column of the DataFrame whose data will go
        on the x-axis.

    y : list or array
        The name(s) of the column(s) of the DataFrame whose data
        will go on the y-axis.

    data : pandas DataFrame
        The DataFrame containing the data.

    options : list or array
        The options shown in the dropdown menu.

    group_column : str
        The column of the DataFrame used to filter data.

    default_value : str
        The default value of the dropdown menu.

    dropdown_label : str
        The label shown next to the dropdown menu.

    title : str
        The title of the plot.

    xtitle : str
        Label of the x-axis

    ytitle : str
        Label of the y-axis

    legend_titles : list or array (default=None)
        The labels shown in the legend.

    output_image : bool (default=False)
        Indicates whether to produce the interactive plot
        or the png image.

    blend_legend : bool (default=False)
        Indicates whether the legend will be in the top right
        corner, outside or inside the plot.

    width : int (default=800)
        Width of the image

    height : int (default=400)
        Height of the image

    scale : int (default=2)
        Scale of the image

    traces_visibility : list or array (default=None)
        A list or array that signals if the
        corresponding line in the plot (by position in `data`)
        is visible or not.

    modes : list or array (default=None)
        A list or array containing the type of plot
        of each trace.

    template : str (default='simple_white')
        The template to use.

    Returns
    -------
    Image or ipywidget
    """

    origin = widgets.Dropdown(
        options=list(options),
        value=default_value,
        description=dropdown_label + ':'
    )

    # set possibility of passing y as dictionary or zip or tuple?
    if legend_titles is None:
        legend_titles = y

    if traces_visibility is None:
        traces_visibility = [True] * len(y)

    if modes is None:
        modes = ['lines'] * len(y)

    traces = []
    for i, trace in enumerate(y):
        traces.append(
            go.Scatter(
                x=data[data[group_column] == default_value][x],
                y=data[data[group_column] == default_value][trace],
                name=legend_titles[i],
                mode=modes[i],
                visible=traces_visibility[i]
            )
        )

    g = go.FigureWidget(
        data=traces,
        layout=go.Layout(
            title=dict(text=title + default_value),
            barmode='overlay',
            xaxis_title=xtitle,
            yaxis_title=ytitle)
        )

    g.update_layout(template=template, title_x=0.5)

    if blend_legend:
        g.update_layout(legend=dict(yanchor="top",
                                    y=0.99,
                                    xanchor="right",
                                    x=0.99))

    def response(change):
        if origin.value in options:
            temp_df = data[data[group_column] == origin.value]
            with g.batch_update():
                for i, trace in enumerate(y):
                    g.data[i].y = temp_df[trace]

                g.layout.barmode = 'overlay'
                g.layout.xaxis.title = xtitle
                g.layout.yaxis.title = ytitle
                g.layout.title = title + origin.value

    origin.observe(response, names="value")
    widget = widgets.VBox([origin, g])

    if output_image:
        return Image(widget.children[1].to_image(format="png",
                                                 width=width,
                                                 height=height,
                                                 scale=scale))
    else:
        return widget


def general_plot(t,
                 data,
                 title=None,
                 names=None,
                 traces_visibility=None,
                 modes=None,
                 blend_legend=False,
                 xanchor='right',
                 output_image=False,
                 width=800,
                 height=400,
                 scale=2,
                 xtitle='Time (days)',
                 ytitle='Number of individuals',
                 template='simple_white',
                 output_figure=False):
    """
    Plots a SIRD model output

    Parameters
    ----------

    t : list or array
        A list of timestamps (days).

    data : list or array
        A list or array containing data to plot.

    title : str (default=None)
        Name of the province or region

    names : list or array (default=None)
        Legend name of each curve of the plot

    traces_visibility : list or array (default=None)
        A list or array that signals if the
        corresponding line in the plot (by position in `data`)
        is visible or not.

    modes : list or array (default=None)
        A list or array containing the type of plot
        of each trace.

    blend_legend : bool (default=False)
        Indicates whether the legend will be in the top right
        corner, outside or inside the plot.

    xanchor : str (default='right')
        Position of the blended legend inside the plot.
        Works only if `blend_legend` is `True`.

    output_image : bool (default=False)
        Indicates whether to produce the interactive plot
        or the png image.

    width : int (default=800)
        Width of the image

    height : int (default=400)
        Height of the image

    scale : int (default=2)
        Scale of the image

    xtitle : str
        Label of the x-axis

    ytitle : str
        Label of the y-axis

    template : str (default='simple_white')
        The template to use.
    """

    if names is None:
        names = ['Susceptible', 'Infected', 'Recovered', 'Dead']

    if traces_visibility is None:
        traces_visibility = [True] * len(data)

    if modes is None:
        modes = ['lines'] * len(data)

    fig = go.Figure()
    for i, comp in enumerate(data):
        fig.add_trace(go.Scatter(
            x=t,
            y=comp,
            mode=modes[i],
            name=names[i],
            visible=traces_visibility[i])
        )

    fig.update_layout(title=title,
                      xaxis_title=xtitle,
                      yaxis_title=ytitle,
                      template=template,
                      barmode='overlay',
                      title_x=0.5)

    if blend_legend:
        xpos = 0.99 if xanchor is 'right' else 0.08

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor=xanchor,
            x=xpos)
        )

    if output_image:
        return Image(fig.to_image(format="png",
                                  width=width,
                                  height=height,
                                  scale=scale))
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def custom_plot(df,
                ydata,
                title,
                xtitle,
                ytitle,
                group_column='denominazione_provincia',
                area_name='Firenze',
                xdata='data',
                modes=None,
                traces_visibility=None,
                legend_titles=None,
                blend_legend=False,
                xanchor='right',
                template='simple_white',
                show_title=True,
                horiz_legend=False):

    if legend_titles is None:
        legend_titles = ydata

    if traces_visibility is None:
        traces_visibility = [True] * len(ydata)

    if modes is None:
        modes = ['lines'] * len(ydata)

    fig = go.Figure()
    for i, trace in enumerate(ydata):
        fig.add_trace(go.Scatter(
            x=df[df[group_column] == area_name][xdata],
            y=df[df[group_column] == area_name][trace],
            mode=modes[i],
            name=legend_titles[i],
            visible=traces_visibility[i])
        )

    fig.update_layout(xaxis_title=xtitle,
                      yaxis_title=ytitle,
                      template=template,
                      barmode='overlay')

    if show_title:
        fig.update_layout(
            title=title + area_name,
            title_x=0.5)

    if blend_legend:
        xpos = 0.99 if xanchor is 'right' else 0.08

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor=xanchor,
            x=xpos)
        )

    if horiz_legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                xanchor="center",
                x=.5,
                y=-.2
            )
        )

    return fig


def data_for_plot(compart,
                  df,
                  column,
                  comp_array,
                  province,
                  window=7,
                  names=None,
                  modes=None,
                  query='20200604',
                  is_regional=False):
    """
    Utility function that returns data useful for plots.
    """

    if names is None:
        names = ['Real',
                 'Real (rolling ' + str(window) + ' days)',
                 'Predicted']

    if modes is None:
        modes = ['lines'] * 3

    title = 'SIRD ' + compart + ' of ' + province

    if is_regional:
        group_column = 'denominazione_regione'
        query = query + ' > data'
    else:
        group_column = 'Province'
        query = query + ' > Date'

    d1_real = df[df[group_column] == province].query(query)[column]
    d2_rolling = d1_real.rolling(window=window).mean().fillna(0)
    data = [d1_real.values, d2_rolling.values, comp_array]

    return names, title, data, modes
