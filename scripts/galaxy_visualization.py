import plotly.graph_objects as go
import numpy as np

def generate_galaxy(data):

    brands = data["brand"].unique()

    sentiment_map = {
        "Positive":1,
        "Negative":-1
    }

    data["score"] = data["sentiment"].map(sentiment_map)

    brand_scores = data.groupby("brand")["score"].sum()

    planets_x=[]
    planets_y=[]
    planets_z=[]
    planet_size=[]
    planet_color=[]
    labels=[]

    for brand in brands:

        score = brand_scores[brand]

        x = np.random.uniform(-10,10)
        y = np.random.uniform(-10,10)
        z = np.random.uniform(-10,10)

        planets_x.append(x)
        planets_y.append(y)
        planets_z.append(z)

        planet_size.append(abs(score)*10 + 20)

        if score>=0:
            planet_color.append("cyan")
        else:
            planet_color.append("red")

        labels.append(brand)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(

        x=planets_x,
        y=planets_y,
        z=planets_z,

        mode='markers+text',

        marker=dict(
            size=planet_size,
            color=planet_color,
            opacity=0.9
        ),

        text=labels,
        textposition="top center"

    ))

    fig.update_layout(

        title="🌌 AI Brand Reputation Galaxy",

        scene=dict(

            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)

        ),

        margin=dict(l=0,r=0,b=0,t=40)

    )

    return fig