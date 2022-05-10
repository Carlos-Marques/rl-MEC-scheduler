import plotly.graph_objects as go

def add_plot(fig, x, y, y_lower, y_upper, label, rgb_str):
    x_rev = x[::-1]
    y_lower = y_lower[::-1]

    fig.add_trace(go.Scatter(
    x=x+x_rev,
    y=y_upper+y_lower,
    fill='toself',
    fillcolor=f'rgba({rgb_str},0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name=label,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        line_color=f'rgb({rgb_str})',
        name=label,
    ))

    return fig
