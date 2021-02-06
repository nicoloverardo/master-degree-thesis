"""
Some classes are taken from the example:
https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from IPython.display import Image


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=None,
    ):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))

            # And cache it for next time
            self._example = result
        return result

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot_plotly(
        self,
        model,
        plot_col="New_cases",
        template="plotly_white",
        output_image=False,
        width=800,
        height=400,
        scale=2,
        output_figure=False,
        horiz_legend=True,
    ):

        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]

        plot_col_str = plot_col.replace("_", " ").capitalize()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.input_indices,
                y=inputs[0, :, plot_col_index],
                mode="lines+markers",
                name="Inputs",
                marker_size=6,
                hovertemplate=(
                    "<b>Inputs</b><br>Day: "
                    "%{x}<br>%{text}: %{y}<extra></extra>"),
                text=[plot_col_str] * len(self.input_indices),
            )
        )
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.label_indices,
                    y=labels[0, :, label_col_index],
                    mode="markers",
                    marker=dict(
                        color="#2ca02c", size=12, line=dict(color="black", width=1)
                    ),
                    name="Labels",
                    hovertemplate=(
                        "<b>Labels</b><br>Day: "
                        "%{x}<br>%{text}: %{y}<extra></extra>"),
                    text=[plot_col_str] * len(self.label_indices),
                )
            )

            if model is not None:
                predictions = model(inputs)
                fig.add_trace(
                    go.Scatter(
                        x=self.label_indices,
                        y=predictions[0, :, label_col_index],
                        mode="markers",
                        marker_symbol="x",
                        marker_line_color="black",
                        marker_color="#ff7f0e",
                        marker_line_width=1,
                        marker_size=12,
                        name="Predictions",
                        hovertemplate=(
                            "<b>Predictions</b><br>Day: "
                            "%{x}<br>%{text}: %{y}<extra></extra>"),
                        text=[plot_col_str] * len(self.label_indices),
                    )
                )

        fig.update_layout(
            xaxis_title="Days",
            yaxis_title=f"{plot_col_str}",
            template=template,
            barmode="overlay",
        )

        if horiz_legend:
            ypos = -0.3

            fig.update_layout(
                legend=dict(
                    orientation="h", yanchor="top", xanchor="center", x=0.5, y=ypos
                )
            )

        if output_image:
            return Image(
                fig.to_image(format="png", width=width, height=height, scale=scale)
            )

        if output_figure:
            return fig

        return fig.show()

    def plot(
        self,
        model=None,
        plot_col="New_cases",
        max_subplots=1,
        figsize=(8, 3),
        output_figure=False,
    ):

        fig, ax = plt.subplots(figsize=figsize)

        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col}")
        ax.plot(
            self.input_indices,
            inputs[0, :, plot_col_index],
            label="Inputs",
            marker=".",
            zorder=-10,
        )

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is not None:
            ax.scatter(
                self.label_indices,
                labels[0, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                predictions = model(inputs)
                ax.scatter(
                    self.label_indices,
                    predictions[0, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

        plt.legend()

        plt.xlabel("Days")

        if output_figure:
            return fig

        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()

        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs

        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, out_steps):
        super().__init__()

        self.out_steps = out_steps

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state


def compile_and_fit(
    model, window, patience=5, epochs=100, monitor="val_loss", verbose=1
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()],
    )

    history = model.fit(
        window.train,
        epochs=epochs,
        validation_data=window.val,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return history


def plot_metrics(history):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    metrics = list(history.history.keys())

    # Compute Rows required
    n_tot = len(metrics)
    n_cols = 2
    n_rows = n_tot // n_cols
    n_rows += n_tot % n_cols

    plt.figure(figsize=(12, 10))

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(n_rows, n_cols, n + 1)

        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history[metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )

        plt.xlabel("Epoch")
        plt.ylabel(name)

        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

    plt.show()


def plot_comparison_results(
    metrics_names, val_performance, performance, figsize=(10, 5), output_figure=False
):
    metrics = [
        metric
        for metric in metrics_names
        if "loss" not in metric and "val_" not in metric
    ]

    n_tot = len(metrics)
    n_cols = 2
    n_rows = n_tot // n_cols
    n_rows += n_tot % n_cols

    x = np.arange(len(performance))
    xs = range(len(performance))
    width = 0.3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, ax in enumerate(axs.flat):
        metric_index = metrics_names.index(metrics[i])
        val_mae = [v[metric_index] for v in val_performance.values()]
        test_mae = [v[metric_index] for v in performance.values()]

        ax.set_ylabel(metrics[i].replace("_", " ").capitalize())
        ax.bar(x - 0.17, val_mae, width, label="Validation")
        ax.bar(x + 0.17, test_mae, width, label="Test")
        ax.set_xticks(xs)
        ax.set_xticklabels(list(performance.keys()))

    plt.legend()

    if output_figure:
        return fig

    plt.show()
