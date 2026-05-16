from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_surface_3d(strike_axis: np.ndarray, expiry_axis: np.ndarray, iv_grid: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Surface(
                x=strike_axis,
                y=expiry_axis,
                z=iv_grid,
                colorscale="Viridis",
                colorbar={"title": "IV"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Moneyness/Strike",
            "yaxis_title": "Expiry (years)",
            "zaxis_title": "Implied Volatility",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
    )
    return fig


def plot_surface_heatmap(strike_axis: np.ndarray, expiry_axis: np.ndarray, iv_grid: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            x=strike_axis,
            y=expiry_axis,
            z=iv_grid,
            colorscale="Turbo",
            colorbar={"title": "IV"},
        )
    )
    fig.update_layout(title=title, xaxis_title="Moneyness/Strike", yaxis_title="Expiry (years)")
    return fig


def plot_surface_sequence_animation(
    strike_axis: np.ndarray,
    expiry_axis: np.ndarray,
    surfaces: np.ndarray,
    title: str = "Surface Evolution",
) -> go.Figure:
    frames = [
        go.Frame(
            data=[go.Surface(x=strike_axis, y=expiry_axis, z=surfaces[i], colorscale="Viridis")],
            name=str(i),
        )
        for i in range(len(surfaces))
    ]
    fig = go.Figure(
        data=[go.Surface(x=strike_axis, y=expiry_axis, z=surfaces[0], colorscale="Viridis")],
        frames=frames,
    )
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "Moneyness/Strike", "yaxis_title": "Expiry", "zaxis_title": "IV"},
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}}]},
                ],
            }
        ],
    )
    return fig


def plot_term_structure_slice(expiry_axis: np.ndarray, actual: np.ndarray, predicted: np.ndarray, strike_idx: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=expiry_axis, y=actual[:, strike_idx], mode="lines+markers", name="Actual"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=expiry_axis, y=predicted[:, strike_idx], mode="lines+markers", name="Predicted"),
        row=1,
        col=1,
    )
    fig.update_layout(
        title=f"Term Structure Slice @ strike index {strike_idx}",
        xaxis_title="Expiry (years)",
        yaxis_title="Implied Volatility",
    )
    return fig
