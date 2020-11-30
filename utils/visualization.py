import plotly.graph_objs as go  # visualization
import plotly.offline as py  # visualization
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt  # visualization


def pie_display(telecom):
    trace = go.Pie(labels=telecom["Churn"].value_counts().keys().tolist(),
                   values=telecom["Churn"].value_counts().values.tolist(),
                   marker=dict(colors=['blue', 'lime'],
                               line=dict(color="white", width=1.1)
                               ),
                   rotation=90,
                   hoverinfo="label+value+text",
                   hole=.5
                   )

    layout = go.Layout(dict(title="Customer churn in training data",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            )
                       )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)


def variable_distribution(telecom):
    # Separating columns to be visualized
    out_cols = list(set(telecom.nunique()[telecom.nunique() < 6].keys().tolist()
                        + telecom.select_dtypes(include='object').columns.tolist()))
    viz_cols = [x for x in telecom.columns if x not in out_cols] + ['Churn']

    sns.pairplot(telecom[viz_cols], diag_kind="kde")
    plt.show()


def variable_summary(df_telecom_og):
    summary = (df_telecom_og[[i for i in df_telecom_og.columns]].
               describe().transpose().reset_index())

    summary = summary.rename(columns={"index": "feature"})
    summary = np.around(summary, 3)

    val_lst = [summary['feature'], summary['count'],
               summary['mean'], summary['std'],
               summary['min'], summary['25%'],
               summary['50%'], summary['75%'], summary['max']]

    trace = go.Table(header=dict(values=summary.columns.tolist(),
                                 line=dict(color=['#506784']),
                                 fill=dict(color=['#119DFF']),
                                 ),
                     cells=dict(values=val_lst,
                                line=dict(color=['#506784']),
                                fill=dict(color=["lightgrey", '#F5F8FF'])
                                ),
                     columnwidth=[200, 60, 100, 100, 60, 60, 80, 80, 80])
    layout = go.Layout(dict(title="Training variable Summary"))
    figure = go.Figure(data=[trace], layout=layout)
    py.iplot(figure)