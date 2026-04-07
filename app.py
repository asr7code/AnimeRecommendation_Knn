from flask import Flask, render_template, request
import gzip
import pickle

app = Flask(__name__)

# Load compressed model and pivot
with gzip.open("pivot.pkl.gz", "rb") as f:
    pivot = pickle.load(f)

with gzip.open("knn_model.pkl.gz", "rb") as f:
    model = pickle.load(f)


def recommend(anime_name):
    if not anime_name or len(anime_name.strip()) < 2:
        return ["Anime not found"]

    anime_name = anime_name.lower().strip()

    matches = [anime for anime in pivot.index if anime_name in anime.lower()]

    if not matches:
        return ["Anime not found"]

    anime_name = matches[0]

    index = pivot.index.get_loc(anime_name)

    distances, indices = model.kneighbors(
        pivot.iloc[index, :].values.reshape(1, -1),
        n_neighbors=6
    )

    recommendations = [pivot.index[i] for i in indices[0][1:]]

    return recommendations


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    anime_name = request.form["anime"]
    results = recommend(anime_name)

    return render_template("result.html", anime_name=anime_name, results=results)


if __name__ == "__main__":
    app.run(debug=True)
