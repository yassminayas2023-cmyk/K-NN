import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Classification de fleurs avec KNN", page_icon="🌸")
st.title("🌸 Classification d'espèces de fleurs avec KNN")
st.write(
    "Cette application utilise l'algorithme **KNN** (K plus proches voisins) "
    "pour deviner l'espèce d'une fleur à partir de 4 mesures."
)

# ------------------------------------------------------------------
# 1. Données
# ------------------------------------------------------------------
COLONNES = ["longueur_sepale", "largeur_sepale", "longueur_petale", "largeur_petale"]
ESPECES = ["setosa", "versicolor", "virginica"]


@st.cache_data
def charger_donnees():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=COLONNES)
    y = pd.Series(iris.target, name="espece")
    return X, y


X, y = charger_donnees()

# ------------------------------------------------------------------
# 2. Réglages du modèle (barre latérale)
# ------------------------------------------------------------------
st.sidebar.header("Réglages de KNN")
k = st.sidebar.slider("k (nombre de voisins)", 1, 20, 5)
distance = st.sidebar.selectbox("Distance", ["euclidean", "manhattan"])
normaliser = st.sidebar.checkbox("Normaliser les données", value=True)
taille_test = st.sidebar.slider("Part des données pour le test", 0.1, 0.5, 0.2, step=0.05)

# ------------------------------------------------------------------
# 3. Entraînement
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=taille_test, random_state=42, stratify=y
)

modele = make_pipeline(
    StandardScaler() if normaliser else "passthrough",
    KNeighborsClassifier(n_neighbors=k, metric=distance),
)
modele.fit(X_train, y_train)

precision = accuracy_score(y_test, modele.predict(X_test))
st.sidebar.metric("Précision sur le test", f"{precision:.1%}")


def predire(df):
    """Retourne un tableau avec l'espèce prédite et la confiance."""
    classes = modele.predict(df[COLONNES])
    proba = modele.predict_proba(df[COLONNES]).max(axis=1)
    resultat = df.copy()
    resultat["espece_predite"] = [ESPECES[c] for c in classes]
    resultat["confiance"] = (proba * 100).round(0).astype(int).astype(str) + " %"
    return resultat


# ------------------------------------------------------------------
# 4. Interface
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Une fleur", "Plusieurs fleurs", "Évaluation du modèle"])

# ---------- Onglet 1 : une seule fleur ----------
with tab1:
    st.subheader("Prédire l'espèce d'une fleur")
    c1, c2 = st.columns(2)
    valeurs = {}
    for i, col in enumerate(COLONNES):
        zone = c1 if i % 2 == 0 else c2
        valeurs[col] = zone.slider(
            col.replace("_", " "),
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean()),
            step=0.1,
        )
    fleur = pd.DataFrame([valeurs])

    if st.button("Prédire", key="une"):
        res = predire(fleur)
        st.success(f"Espèce prédite : **{res['espece_predite'][0]}** (confiance {res['confiance'][0]})")

        # Les k voisins utilisés pour décider
        transfo = modele[:-1]
        knn = modele[-1]
        dist, idx = knn.kneighbors(transfo.transform(fleur[COLONNES]))
        voisins = X_train.iloc[idx[0]].copy()
        voisins["espece"] = [ESPECES[c] for c in y_train.iloc[idx[0]]]
        voisins["distance"] = dist[0].round(3)
        st.write(f"Les **{k} voisins les plus proches** qui ont voté :")
        st.dataframe(voisins, use_container_width=True)

        # Graphique pétales
        fig, ax = plt.subplots()
        for num, nom in enumerate(ESPECES):
            m = y == num
            ax.scatter(X.loc[m, "longueur_petale"], X.loc[m, "largeur_petale"], label=nom, alpha=0.6)
        ax.scatter(
            fleur["longueur_petale"], fleur["largeur_petale"],
            color="black", marker="*", s=300, label="Ta fleur",
        )
        ax.set_xlabel("longueur pétale")
        ax.set_ylabel("largeur pétale")
        ax.legend()
        st.pyplot(fig)

# ---------- Onglet 2 : plusieurs fleurs ----------
with tab2:
    st.subheader("Prédire l'espèce de plusieurs fleurs")
    st.write(
        "Remplis le tableau (tu peux ajouter des lignes) **ou** importe un fichier CSV "
        f"avec les colonnes : `{', '.join(COLONNES)}`."
    )

    fichier = st.file_uploader("Importer un CSV", type="csv")

    if fichier is not None:
        donnees = pd.read_csv(fichier)
        manquantes = [c for c in COLONNES if c not in donnees.columns]
        if manquantes:
            st.error(f"Colonnes manquantes dans le CSV : {manquantes}")
            donnees = None
    else:
        exemple = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.9, 3.1, 5.8, 2.1]],
            columns=COLONNES,
        )
        donnees = st.data_editor(exemple, num_rows="dynamic", use_container_width=True)

    if donnees is not None and st.button("Prédire toutes les fleurs", key="plusieurs"):
        donnees = donnees.dropna(subset=COLONNES)
        res = predire(donnees)
        st.dataframe(res, use_container_width=True)
        st.write("Nombre de fleurs par espèce prédite :")
        st.bar_chart(res["espece_predite"].value_counts())
        st.download_button(
            "Télécharger les résultats (CSV)",
            res.to_csv(index=False).encode("utf-8"),
            file_name="resultats_knn.csv",
            mime="text/csv",
        )

# ---------- Onglet 3 : évaluation ----------
with tab3:
    st.subheader("Le modèle est-il bon ?")
    st.write(f"Précision sur {len(X_test)} fleurs jamais vues : **{precision:.1%}**")

    st.write("**Matrice de confusion** (lignes = vraie espèce, colonnes = espèce prédite)")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, modele.predict(X_test)), display_labels=ESPECES
    ).plot(ax=ax, colorbar=False)
    st.pyplot(fig)

    st.write("**Quel k choisir ?** (validation croisée sur les données d'entraînement)")
    valeurs_k = list(range(1, 21))
    scores = [
        cross_val_score(
            make_pipeline(
                StandardScaler() if normaliser else "passthrough",
                KNeighborsClassifier(n_neighbors=kk, metric=distance),
            ),
            X_train, y_train, cv=5,
        ).mean()
        for kk in valeurs_k
    ]
    fig, ax = plt.subplots()
    ax.plot(valeurs_k, scores, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Précision (validation croisée)")
    st.pyplot(fig)
    st.info(f"Meilleur k trouvé : **{valeurs_k[int(np.argmax(scores))]}**")
