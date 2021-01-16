import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import Image


def inter_dropdown_plot(
    x,
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
    template="simple_white",
):
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
        options=list(options), value=default_value, description=dropdown_label + ":"
    )

    # set possibility of passing y as dictionary or zip or tuple?
    if legend_titles is None:
        legend_titles = y

    if traces_visibility is None:
        traces_visibility = [True] * len(y)

    if modes is None:
        modes = ["lines"] * len(y)

    traces = []
    for i, trace in enumerate(y):
        traces.append(
            go.Scatter(
                x=data[data[group_column] == default_value][x],
                y=data[data[group_column] == default_value][trace],
                name=legend_titles[i],
                mode=modes[i],
                visible=traces_visibility[i],
            )
        )

    g = go.FigureWidget(
        data=traces,
        layout=go.Layout(
            title=dict(text=title + default_value),
            barmode="overlay",
            xaxis_title=xtitle,
            yaxis_title=ytitle,
        ),
    )

    g.update_layout(template=template, title_x=0.5)

    if blend_legend:
        g.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

    def response(change):
        if origin.value in options:
            temp_df = data[data[group_column] == origin.value]
            with g.batch_update():
                for i, trace in enumerate(y):
                    g.data[i].y = temp_df[trace]

                g.layout.barmode = "overlay"
                g.layout.xaxis.title = xtitle
                g.layout.yaxis.title = ytitle
                g.layout.title = title + origin.value

    origin.observe(response, names="value")
    widget = widgets.VBox([origin, g])

    if output_image:
        return Image(
            widget.children[1].to_image(
                format="png", width=width, height=height, scale=scale
            )
        )
    else:
        return widget


def general_plot(
    t,
    data,
    title=None,
    names=None,
    traces_visibility=None,
    modes=None,
    blend_legend=False,
    xanchor="right",
    output_image=False,
    width=800,
    height=400,
    scale=2,
    xtitle="Time (days)",
    ytitle="Number of individuals",
    template="simple_white",
    output_figure=False,
    horiz_legend=False,
):
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
        names = ["Susceptible", "Infected", "Recovered", "Dead"]

    if traces_visibility is None:
        traces_visibility = [True] * len(data)

    if modes is None:
        modes = ["lines"] * len(data)

    fig = go.Figure()
    for i, comp in enumerate(data):
        fig.add_trace(
            go.Scatter(
                x=t, y=comp, mode=modes[i], name=names[i], visible=traces_visibility[i]
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template=template,
        barmode="overlay",
        title_x=0.5,
    )

    if blend_legend:
        xpos = 0.99 if xanchor == "right" else 0.08

        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor=xanchor, x=xpos))

    if horiz_legend:
        ypos = -0.2 if xtitle == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def custom_plot(
    df,
    ydata,
    title,
    xtitle,
    ytitle,
    group_column="denominazione_provincia",
    area_name="Firenze",
    xdata="data",
    modes=None,
    traces_visibility=None,
    legend_titles=None,
    blend_legend=False,
    xanchor="right",
    template="simple_white",
    show_title=True,
    horiz_legend=False,
    icu=None,
):

    if legend_titles is None:
        legend_titles = ydata

    if traces_visibility is None:
        traces_visibility = [True] * len(ydata)

    if modes is None:
        modes = ["lines"] * len(ydata)

    fig = go.Figure()
    for i, trace in enumerate(ydata):
        fig.add_trace(
            go.Scatter(
                x=df[df[group_column] == area_name][xdata],
                y=df[df[group_column] == area_name][trace],
                mode=modes[i],
                name=legend_titles[i],
                visible=traces_visibility[i],
            )
        )

    if icu is not None:
        fig.add_hline(
            y=icu,
            line_dash="dash",
            line_color="grey",
            annotation_text="ICU capacity: " + str(icu),
        )

    fig.update_layout(
        xaxis_title=xtitle, yaxis_title=ytitle, template=template, barmode="overlay"
    )

    if title is not None:
        if show_title:
            title = (title + area_name,)

        fig.update_layout(title=title, title_x=0.5)

    if blend_legend:
        xpos = 0.99 if xanchor == "right" else 0.08

        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor=xanchor, x=xpos))

    if horiz_legend:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=-0.2)
        )

    return fig


def daily_main_indic_plot(
    area,
    df,
    y_cols,
    y_labels,
    x_col="data",
    template="plotly_white",
    output_image=False,
    width=800,
    height=800,
    scale=2,
    output_figure=False,
    title=None,
    horiz_legend=True,
):

    n_data = len(y_cols)
    n_rows = n_data // 2 if n_data % 2 == 0 else (n_data + 1) // 2

    fig = make_subplots(rows=n_rows, cols=2)

    i = 0
    for r in range(1, n_rows + 1):
        for c in range(1, 3):
            fig.add_trace(
                go.Bar(name=y_labels[i], x=df[x_col], y=df[y_cols[i]]), row=r, col=c
            )

            i += 1

    fig.add_shape(
        type="line",
        y0=0,
        y1=0,
        x0=df.iloc[0][x_col],
        x1=df.iloc[-1][x_col],
        line=dict(width=1),
        row="all",
        col="all",
        opacity=0.5,
    )

    if title is None:
        title_desc = "Daily changes in the main indicators"
        title = title_desc + " - " + area if not output_figure else title_desc

    fig.update_layout(height=height, title_text=title, template=template, title_x=0.5)

    if horiz_legend:
        ypos = -0.2 if title == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def autocorr_indicators_plot(
    df,
    x_col,
    y_cols,
    y_labels,
    template="plotly_white",
    title="Auto-correlations",
    output_image=False,
    width=950,
    height=500,
    scale=2,
    output_figure=False,
    horiz_legend=True,
):

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        fig.add_trace(
            go.Scatter(x=df[x_col], y=df[col], name=y_labels[i], line_shape="spline")
        )

    tot_weeks = int(df[x_col].values[-1] / 7)

    for w in range(1, tot_weeks + 1):
        units = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
        }

        x = w * 7
        n_weeks_str = units[w]
        weeks_str = "<br>week" if w == 1 else "<br>weeks"
        ann_text = n_weeks_str + weeks_str

        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color="green",
            annotation_text=ann_text,
            annotation_position="bottom right",
        )

    fig.update_yaxes(title_text="Auto-correlation")
    fig.update_xaxes(title_text="Days")
    fig.update_layout(template=template, title=title, title_x=0.5)

    if horiz_legend:
        ypos = -0.2 if title == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def cross_corr_cases_plot(
    df,
    x_col="giorni",
    y_col="crosscor_decessi_nuovi_positivi",
    template="plotly_white",
    title="Cross-correlation",
    output_image=False,
    width=800,
    height=400,
    scale=2,
    output_figure=False,
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], line_shape="spline"))
    fig.add_vline(
        x=df[y_col].idxmax(),
        line_dash="dash",
        line_color="green",
        annotation_text="max<br>corr.",
        annotation_position="bottom right",
    )

    fig.update_yaxes(title_text="Cross-correlation")
    fig.update_xaxes(title_text="Days")

    fig.update_layout(template=template, title=title, title_x=0.5)

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def trend_corr_plot(
    df,
    y_cols,
    y_labels,
    days_max_corr,
    data_column="data",
    template="plotly_white",
    title=None,
    output_image=False,
    width=800,
    height=400,
    scale=2,
    output_figure=False,
    horiz_legend=True,
):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df[data_column], y=df[y_cols[0]], name=y_labels[0], line_shape="spline"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[data_column],
            y=df[y_cols[1]].shift(-days_max_corr),
            name=y_labels[1] + " of " + str(days_max_corr) + "d after",
            line_shape="spline",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text=y_labels[0], secondary_y=False)
    fig.update_yaxes(title_text=y_labels[1], secondary_y=True)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        title_x=0.5,
        template=template,
    )

    if horiz_legend:
        ypos = -0.2 if title == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if title is None:
        fig.update_layout(
            title_text="Trend "
            + y_labels[0]
            + " - "
            + y_labels[1]
            + " (shift of "
            + str(days_max_corr)
            + "d)"
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def plot_ts_decomp(
    x_dates,
    ts_true,
    decomp_res,
    output_image=False,
    width=800,
    height=1000,
    scale=2,
    output_figure=False,
    title=None,
    template="plotly_white",
):

    fig = make_subplots(
        rows=4, cols=1, subplot_titles=("Original", "Trend", "Seasonal", "Residuals")
    )
    fig.add_trace(go.Scatter(name="Original", x=x_dates, y=ts_true), row=1, col=1)
    fig.add_trace(go.Scatter(name="Trend", x=x_dates, y=decomp_res.trend), row=2, col=1)
    fig.add_trace(
        go.Scatter(name="Seasonal", x=x_dates, y=decomp_res.seasonal), row=3, col=1
    )
    fig.add_trace(
        go.Scatter(name="Residuals", x=x_dates, y=decomp_res.resid, mode="markers"),
        row=4,
        col=1,
    )

    if title is not None:
        fig.update_layout(title=title)

    fig.update_layout(height=height, template=template, title_x=0.5, showlegend=False)

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def plot_tstat_models(
    df,
    train,
    test,
    fitted_vals,
    yhat,
    column,
    template="plotly_white",
    title=None,
    output_image=False,
    width=800,
    height=400,
    scale=2,
    output_figure=False,
    horiz_legend=True,
    ylabel="Number of individuals",
):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df[column].values, name="Real", mode="markers")
    )

    if isinstance(yhat, pd.DataFrame):
        fig.add_trace(
            go.Scatter(
                x=yhat.index,
                y=yhat["mean_ci_lower"] - yhat["mean"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=yhat.index,
                y=yhat["mean_ci_upper"] - yhat["mean"],
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(80, 148, 197, 0.25)",
                fill="tonexty",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fitted_pred = np.concatenate((fitted_vals, yhat["mean"]))
    else:
        fitted_pred = np.concatenate((fitted_vals, yhat))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=fitted_pred,
            name="Fitted",
            mode="lines",
            line=dict(color="#ef553b"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[test.first_valid_index()],
            y=[train[column].max() - 1],
            text="prediction<br>start",
            mode="text",
            name="",
            showlegend=False,
        )
    )
    fig.add_shape(
        type="line",
        y0=train[column].min(),
        y1=train[column].max(),
        x0=test.first_valid_index(),
        x1=test.first_valid_index(),
        line=dict(width=1.5, dash="dash"),
        opacity=0.4,
    )

    if title is not None:
        fig.update_layout(title=title, title_x=0.5)

    fig.update_layout(template=template, yaxis_title=ylabel)

    if horiz_legend:
        ypos = -0.2 if title == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def plot_fbp_comp(
    df,
    template="plotly_white",
    title=None,
    output_image=False,
    width=800,
    height=800,
    scale=2,
    output_figure=False,
):

    fig = make_subplots(rows=3, cols=1, subplot_titles=("Trend", "Holidays", "Weekly"))
    fig.add_trace(
        go.Scatter(x=df.ds, y=df.trend.values, name="Trend", mode="lines"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            name="Upper Bound",
            x=df.ds,
            y=df.trend_upper,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Lower Bound",
            x=df.ds,
            y=df.trend_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.ds, y=df.holidays.values, name="Holidays", mode="lines"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Upper Bound",
            x=df.ds,
            y=df.holidays_upper,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Lower Bound",
            x=df.ds,
            y=df.holidays_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.ds, y=df.weekly.values, name="Weekly", mode="lines"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Upper Bound",
            x=df.ds,
            y=df.weekly_upper,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Lower Bound",
            x=df.ds,
            y=df.weekly_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    if title is not None:
        fig.update_layout(title=title, title_x=0.5)

    fig.update_layout(
        template=template,
        height=height,
        showlegend=False,
    )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def anomalies_plot(
    df,
    compart,
    window,
    template="plotly_white",
    title=None,
    output_image=False,
    width=800,
    height=400,
    scale=1.96,
    output_figure=False,
    horiz_legend=True,
    ylabel="Number of individuals",
):

    rolling_mean = df.loc[:, [compart]][compart].rolling(window=window).mean().dropna()
    df_filt = df.loc[:, [compart]].iloc[window - 1 :]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            name="Rolling mean trend",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index, y=df_filt[compart], name="Real values", mode="lines"
        )
    )

    mae = mean_absolute_error(df_filt[compart], rolling_mean)

    deviation = np.std(df_filt[compart] - rolling_mean)

    lower_bound = rolling_mean - (mae + scale * deviation)
    upper_bound = rolling_mean + (mae + scale * deviation)

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=upper_bound,
            name="Upper/lower bound",
            mode="lines",
            line=dict(dash="dash", width=1.5, color="green"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=lower_bound,
            name="Upper/lower bound",
            mode="lines",
            line=dict(dash="dash", width=1.5, color="green"),
            showlegend=False,
        )
    )

    anomalies = pd.DataFrame(index=rolling_mean.index, columns=df_filt.columns)

    anomalies[df_filt[compart] < lower_bound] = df_filt[df_filt[compart] < lower_bound]

    anomalies[df_filt[compart] > upper_bound] = df_filt[df_filt[compart] > upper_bound]

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=anomalies[compart],
            mode="markers",
            name="Anomaly",
            showlegend=False,
            marker=dict(color="red", size=10),
        )
    )

    fig.add_shape(
        type="line",
        y0=0,
        y1=0,
        x0=df_filt.index[0],
        x1=df_filt.index[-1],
        line=dict(width=1.5, dash="dash"),
        opacity=0.5,
    )

    if title is not None:
        fig.update_layout(title=title, title_x=0.5)

    fig.update_layout(template=template, yaxis_title=ylabel)

    if horiz_legend:
        ypos = -0.2 if title == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def ac_plot(
    data,
    ci,
    template="plotly_white",
    title=None,
    output_image=False,
    width=800,
    height=400,
    scale=2,
    output_figure=False,
):

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(data.shape[0])),
            y=data,
            hoverinfo="skip",
            showlegend=False,
            width=0.2,
            marker_color="black",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(data.shape[0])),
            y=data,
            mode="markers",
            name="Autocorrelation",
            showlegend=False,
            marker=dict(color="rgb(31, 119, 180)", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(data.shape[0])),
            y=ci[1:, 0] - data[1:],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(data.shape[0])),
            y=ci[1:, 1] - data[1:],
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(80, 148, 197, 0.25)",
            fill="tonexty",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if title is not None:
        fig.update_layout(title=title, title_x=0.5)

    fig.update_layout(template=template, xaxis_title="Lags")

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def discsid_param_plot(
    t,
    data,
    title=None,
    names=None,
    traces_visibility=None,
    modes=None,
    blend_legend=False,
    xanchor="right",
    output_image=False,
    width=800,
    height=600,
    scale=2,
    xtitle="Time (days)",
    ytitle=None,
    template="simple_white",
    output_figure=False,
    horiz_legend=False,
):

    if traces_visibility is None:
        traces_visibility = [True] * len(data)

    if modes is None:
        modes = ["lines"] * len(data)

    n_data = len(data)
    n_rows = n_data // 2 if n_data % 2 == 0 else (n_data + 1) // 2

    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes="all")

    i = 0
    for r in range(1, n_rows + 1):
        for c in range(1, 3):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=data[i],
                    mode=modes[i],
                    name=names[i],
                    visible=traces_visibility[i],
                ),
                row=r,
                col=c,
            )

            i += 1

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template=template,
        barmode="overlay",
        title_x=0.5,
        height=height,
        xaxis_showticklabels=True,
        xaxis2_showticklabels=True,
    )

    if blend_legend:
        xpos = 0.99 if xanchor == "right" else 0.08

        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor=xanchor, x=xpos))

    if horiz_legend:
        ypos = -0.2 if xtitle == "" else -0.3

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos)
        )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )
    else:
        if output_figure:
            return fig
        else:
            return fig.show()


def data_for_plot(
    compart,
    df,
    column,
    comp_array,
    province,
    window=7,
    names=None,
    modes=None,
    query="20200604",
    is_regional=False,
):
    """
    Utility function that returns data useful for plots.
    """

    if names is None:
        names = ["Real", "Real (rolling " + str(window) + " days)", "Predicted"]

    if modes is None:
        modes = ["lines"] * 3

    title = "SIRD " + compart + " of " + province

    if is_regional:
        group_column = "denominazione_regione"
        query = query + " > data"
    else:
        group_column = "Province"
        query = query + " > Date"

    d1_real = df.loc[(df[group_column] == province), :].query(query).loc[:, column]
    d2_rolling = d1_real.rolling(window=window).mean().fillna(0)
    data = [d1_real.values, d2_rolling.values, comp_array]

    return names, title, data, modes
