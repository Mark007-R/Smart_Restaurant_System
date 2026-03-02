import os

import pandas as pd
from flask import render_template, session


def _placeholder_image(restaurant_name):
    hash_value = sum(ord(char) * (index + 1) for index, char in enumerate(restaurant_name))
    return f"https://source.unsplash.com/800x600/?restaurant,food&sig={hash_value % 10000}"


def _extract_restaurants(dataset_folder):
    dataset_configs = [
        ("mumbaires.csv", "Restaurant Name", "Rating", "Address"),
        ("Resreviews.csv", "Restaurant", "Rating", None),
        ("reviews.csv", "business_name", "rating", None),
        ("zomato.csv", "name", "rate", "address"),
        ("zomato2.csv", "Restaurant_Name", "Avg_Rating_Restaurant", None),
    ]

    restaurants = {}

    for filename, name_col, rating_col, address_col in dataset_configs:
        path = os.path.join(dataset_folder, filename)
        if not os.path.exists(path):
            continue

        try:
            dataframe = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            dataframe.columns = dataframe.columns.str.strip()

            if name_col not in dataframe.columns:
                continue

            for _, row in dataframe.iterrows():
                name = str(row.get(name_col, "")).strip()
                if not name or name.lower() == "nan":
                    continue

                if name not in restaurants:
                    restaurants[name] = {
                        "name": name,
                        "rating": row.get(rating_col, "N/A") if rating_col in dataframe.columns else "N/A",
                        "address": row.get(address_col, "Address not available") if address_col and address_col in dataframe.columns else "Address not available",
                        "photo": _placeholder_image(name),
                    }
        except Exception:
            continue

    return list(restaurants.values())


def register_user_routes(app, db, User, Review, login_required):
    @app.route("/user/dashboard", methods=["GET"])
    @login_required
    def user_dashboard():
        restaurants = _extract_restaurants(app.config.get("DATASET_FOLDER", "datasets"))
        return render_template(
            "user_dashboard.html",
            restaurants=restaurants[:20],
            total_reviews=Review.query.count(),
            total_restaurants=min(20, len(restaurants)),
            username=session.get("username", "User"),
        )

    @app.route("/user/health", methods=["GET"])
    @login_required
    def user_health():
        return {
            "status": "ok",
            "role": session.get("role"),
            "username": session.get("username"),
        }
