import numpy as np
import pandas as pd
import joblib
import faiss



model_rank = joblib.load("models/lightgbm_ranker.pkl")
encoders = joblib.load("models/encoders.pkl")

user_index_map = joblib.load("models/user_index_map.pkl")
item_id_map = joblib.load("models/item_id_map.pkl")
item_index_to_id = joblib.load("models/item_index_to_id.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")

faiss_index = faiss.read_index("models/faiss_index.index")

user_embeddings = np.load("models/user_embeddings.npy")
item_embeddings = np.load("models/item_embeddings.npy")




events_df = pd.read_csv("data/events.csv")
users_df = pd.read_csv("data/users.csv")
ranking_dataset = pd.read_csv("data/ranking_dataset.csv", low_memory=False)

ranking_train_clean = ranking_dataset


def recommend(user_id, top_k=10):

    import numpy as np
    import pandas as pd

    
    if user_id not in user_index_map:

        cold = events_df.copy()

        cold["days_until_event"] = (
            pd.to_datetime(cold["event_date"], errors="coerce") - pd.Timestamp.now()
        ).dt.days

        cold = cold[cold["days_until_event"] >= 0]

        pop = cold["item_id"].map(
            ranking_dataset.groupby("item_id").size()
        ).fillna(0)

        cold["popularity"] = np.log1p(pop)

        return (
            cold.sort_values(["popularity","days_until_event"], ascending=[False,True])
            .head(top_k)[["item_id","popularity"]]
        )



    
    user_topics = ranking_train_clean[
        ranking_train_clean["user_id"] == user_id
    ]["topic"].value_counts().index.tolist()



    
    user_vec = user_embeddings[user_index_map[user_id]].reshape(1,-1).astype("float32")

    scores, items = faiss_index.search(user_vec, 1500)

    faiss_candidates = [
        item_index_to_id[i] for i in items[0] if i in item_index_to_id
    ]


    
    popular_items = (
        ranking_dataset
        .groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .head(300)
        .index
        .tolist()
    )


    
    recent_items = (
        events_df
        .sort_values("event_date", ascending=False)
        .head(200)["item_id"]
        .tolist()
    )


   
    candidate_items = list(set(faiss_candidates) | set(popular_items) | set(recent_items))



   
    candidates = pd.DataFrame({
        "user_id": user_id,
        "item_id": candidate_items
    })


    
    candidates = candidates.merge(events_df, on="item_id", how="left")
    candidates = candidates.merge(users_df, on="user_id", how="left")


    
    candidates["days_until_event"] = (
        pd.to_datetime(candidates["event_date"], errors="coerce") - pd.Timestamp.now()
    ).dt.days

    candidates = candidates[candidates["days_until_event"] >= -1]


    
    if len(user_topics) > 0:

        topic_candidates = candidates[
            candidates["topic"].isin(user_topics)
        ]

        if len(topic_candidates) >= top_k:
            candidates = topic_candidates



    def haversine(lat1, lon1, lat2, lon2):

        R = 6371

        lat1, lon1, lat2, lon2 = map(
            np.radians, [lat1, lon1, lat2, lon2]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat/2)**2 +
            np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        )

        c = 2*np.arcsin(np.sqrt(a))

        return R*c


    candidates["geo_distance"] = haversine(
        candidates["user_latitude"],
        candidates["user_longitude"],
        candidates["event_latitude"],
        candidates["event_longitude"]
    )


    
    candidates["time_score"] = np.exp(
        -0.05 * candidates["days_until_event"].clip(lower=0)
    )


    
    pop = candidates["item_id"].map(
        ranking_dataset.groupby("item_id").size()
    ).fillna(0)

    candidates["popularity"] = np.log1p(pop)



   
    def emb_sim(row):

        u = row["user_id"]
        i = row["item_id"]

        if u in user_index_map and i in item_id_map:

            ui = user_index_map[u]
            ii = item_id_map[i]

            if ui < len(user_embeddings) and ii < len(item_embeddings):

                return float(
                    np.dot(
                        user_embeddings[ui],
                        item_embeddings[ii]
                    )
                )

        return 0


    candidates["embedding_similarity"] = candidates.apply(emb_sim, axis=1)



    
    X_test = candidates[feature_columns].copy().fillna(0)

    for col in encoders:

        le = encoders[col]

        X_test[col] = X_test[col].astype(str)

        known = set(le.classes_)

        X_test[col] = X_test[col].apply(
            lambda x: x if x in known else le.classes_[0]
        )

        X_test[col] = le.transform(X_test[col])



    
    pred = model_rank.predict(X_test)

    if len(pred.shape) == 2:
        candidates["model_score"] = pred[:,1]
    else:
        candidates["model_score"] = pred



    
    candidates["topic_match"] = candidates["topic"].apply(
        lambda x: 1 if x in user_topics else 0
    )


    
    max_dist = candidates["geo_distance"].max() + 1e-6

    candidates["distance_score"] = 1 - (
        candidates["geo_distance"] / max_dist
    )


    
    candidates["final_score"] = (
          0.45 * candidates["model_score"]
        + 0.20 * candidates["topic_match"]
        + 0.15 * candidates["distance_score"]
        + 0.10 * candidates["time_score"]
        + 0.10 * candidates["popularity"]
    )



   
    exploration_weight = 0.05

    candidates["final_score"] = (
        (1 - exploration_weight) * candidates["final_score"]
        + exploration_weight * np.random.rand(len(candidates))
    )



    
    selected = []
    remaining = candidates.copy()
    lambda_div = 0.7


    def embedding_similarity(item1, item2):

        if item1 in item_id_map and item2 in item_id_map:

            v1 = item_embeddings[item_id_map[item1]]
            v2 = item_embeddings[item_id_map[item2]]

            return np.dot(v1, v2)

        return 0


    while len(selected) < min(top_k, len(remaining)):

        if len(selected) == 0:

            idx = remaining["final_score"].idxmax()

        else:

            remaining["div_penalty"] = remaining["item_id"].apply(
                lambda item: max(
                    embedding_similarity(item, s["item_id"])
                    for s in selected
                )
            )

            remaining["mmr"] = (
                lambda_div * remaining["final_score"]
                - (1 - lambda_div) * remaining["div_penalty"]
            )

            idx = remaining["mmr"].idxmax()

        selected.append(remaining.loc[idx])
        remaining = remaining.drop(idx)


    result = pd.DataFrame(selected)    
    
    

   
    result = result[["item_id","final_score"]]
    
    
    result = result.merge(
        events_df[
            [
                "item_id",
                "topic",
                "type",
                "event_location",
                "event_college",
                "event_date"
            ]
        ],
        on="item_id",
        how="left"
    )
    
    return (
        result.sort_values("final_score", ascending=False)[
            [
                "item_id",
                "topic",
                "type",
                "event_location",
                "event_college",
                "event_date",
                "final_score"
            ]
        ]
        .reset_index(drop=True)
    )
