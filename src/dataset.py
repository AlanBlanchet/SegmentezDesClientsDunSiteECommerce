import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime


def to_datetime(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")


# Conservons le nombre de minutes passées avant une réponse
def minutes_f(x):
    return x.total_seconds() / 60


def calculate_diffs(o):
    o = o.order_purchase_timestamp.values
    if len(o) == 1:
        return 0
    time_diffs = pd.Series([])
    for i in range(len(o) - 1):
        o1 = to_datetime(o[i + 1])
        o2 = to_datetime(o[i])
        time_diffs[len(time_diffs)] = minutes_f(o1 - o2)

    return time_diffs.mean()


# Fonction récupérée sur le premier kaggle
def segment(x):
    if x == "123":
        return "Essentiel"
    elif x in ["311", "312", "313"]:
        return "Parti"
    elif x in ["111", "112", "113"]:
        return "Nouveau"
    elif x in ["323", "213", "223"]:
        return "Rentable"
    elif x in ["221", "222", "321", "322"]:
        return "Loyal"
    else:
        return "Régulier"


def get_data_until(month=0, days=0):
    data = Path("data")
    raw = data / "raw"
    processed = data / "processed"

    csvs = [*data.rglob("*.csv")]

    if not raw.exists() or not processed.exists():
        raw.mkdir(exist_ok=True)
        processed.mkdir(exist_ok=True)
        for csv in csvs:
            shutil.move(csv, raw)

    customers = pd.read_csv(raw / "olist_customers_dataset.csv")
    geolocations = pd.read_csv(raw / "olist_geolocation_dataset.csv")
    order_items = pd.read_csv(raw / "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(raw / "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(raw / "olist_order_reviews_dataset.csv")
    orders = pd.read_csv(raw / "olist_orders_dataset.csv")
    products = pd.read_csv(raw / "olist_products_dataset.csv")

    # Erreur de nommage des colonnes. C'est "length" et non "lenght"
    products.columns = products.columns.map(lambda x: x.replace("lenght", "length"))

    # Il y a des valeurs avec " " comme valeur
    order_reviews.loc[
        order_reviews.review_comment_title == " ", "review_comment_title"
    ] = None

    order_reviews["review_level"] = order_reviews.review_comment_title.isna().replace(
        {True: 0, False: 1}
    ) + order_reviews.review_comment_message.isna().replace({True: 0, False: 1})
    order_reviews["review_level"].value_counts()

    order_reviews = order_reviews.drop(
        columns=["review_comment_title", "review_comment_message"]
    )

    creation = order_reviews.review_creation_date.map(to_datetime)
    answer = order_reviews.review_answer_timestamp.map(to_datetime)
    delay = answer - creation

    order_reviews["review_answer_delay"] = delay.map(minutes_f)
    # On peut supprimer les autres dates car elle ne sont plus utiles pour nous. On aurait pu les utiliser pour remarquer une potentielle amélioration du traitement des retours mais ce n'est pas l'objectif ici
    order_reviews = order_reviews.drop(
        columns=["review_creation_date", "review_answer_timestamp"]
    )

    # Je profite de drop la colonne approved pour également drop celle correspondant à l'arrivée chez l'expéditeur car elle n'apportera pas de plus value selon moi
    orders = orders.drop(columns=["order_approved_at", "order_delivered_carrier_date"])

    never_delivered = orders.order_delivered_customer_date.isna()

    is_delivered = orders.order_status.isin(["delivered"])

    orders.loc[
        never_delivered & is_delivered, "order_delivered_customer_date"
    ] = orders.loc[never_delivered & is_delivered, "order_estimated_delivery_date"]

    orders[never_delivered].head()

    orders_with_dates = orders.loc[is_delivered]
    purchase = orders_with_dates.order_purchase_timestamp.map(to_datetime)
    estimated = orders_with_dates.order_estimated_delivery_date.map(to_datetime)
    delivered = orders_with_dates.order_delivered_customer_date.map(to_datetime)

    orders.loc[is_delivered, "estimation_error"] = (delivered - estimated).map(
        minutes_f
    )
    orders.loc[is_delivered, "delivery_delay"] = (delivered - purchase).map(minutes_f)

    orders.loc[:, "delay_respected"] = orders.estimation_error <= 0
    orders.delay_respected.value_counts()

    products = products.drop(
        columns=[
            "product_name_length",
            "product_description_length",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
            "product_photos_qty",
        ]
    )

    customer_orders = pd.merge(customers, orders, on="customer_id")

    category_names = products.product_category_name.value_counts(normalize=True)
    others = category_names < 0.02
    sum_of_others = category_names[others].sum()
    category_names = category_names[~others]
    category_names["other"] = sum_of_others

    order_dates = orders.order_purchase_timestamp.apply(to_datetime)
    max_order_date = max(order_dates)
    order_date_diff = (max_order_date - order_dates).map(minutes_f)
    orders["order_date_lately"] = order_date_diff

    F = (
        customer_orders.sort_values(by="order_purchase_timestamp")
        .groupby(by="customer_unique_id")
        .apply(calculate_diffs)
    )

    big_group = pd.merge(customers, orders, on="customer_id")

    # Remove months
    # --------------------------------
    from dateutil.relativedelta import relativedelta

    compare_date = max_order_date - relativedelta(months=month, days=days)

    order_datetimes = big_group.order_purchase_timestamp.apply(to_datetime)

    big_group = big_group[order_datetimes < compare_date].reset_index(drop=True)
    # --------------------------------

    big_group = big_group.loc[
        big_group.order_delivered_customer_date.notna()
        & big_group.delivery_delay.notna()
    ]

    big_group = pd.merge(big_group, order_payments, on="order_id")

    geolocations = geolocations.groupby(by="geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]
    ].mean()
    mean = geolocations[["geolocation_lat", "geolocation_lng"]].mean()
    big_group[["lat", "lng"]] = big_group.customer_zip_code_prefix.apply(
        lambda x: geolocations.loc[x] if (geolocations.index == x).any() else mean
    )

    grouped_customer = big_group.groupby("customer_unique_id")
    unique_customer = grouped_customer[
        ["order_date_lately", "delivery_delay", "estimation_error"]
    ].mean()
    unique_customer = unique_customer.rename(columns={"order_date_lately": "recency"})
    unique_customer["number_of_orders"] = grouped_customer.customer_unique_id.count()
    unique_customer["respected_ratio"] = (
        grouped_customer.delay_respected.sum() / unique_customer.number_of_orders
    )

    unique_customer["amount"] = grouped_customer.payment_value.sum()

    unique_customer[["lat", "lng"]] = grouped_customer[["lat", "lng"]].agg(
        lambda x: pd.Series.mode(x)[0]
    )

    unique_customer["frequency"] = unique_customer.index.map(lambda x: F[x])

    big_group = pd.merge(big_group, order_reviews, on="order_id")
    big_group = pd.merge(big_group, order_items, on="order_id")
    big_group = pd.merge(big_group, products, on="product_id")
    big_group = big_group[big_group.product_category_name.notna()]

    unique_customer = unique_customer.loc[big_group.customer_unique_id.unique()]
    unique_customer = unique_customer.sort_index()
    grouped_customer = big_group.groupby("customer_unique_id")

    mean_features = [
        "freight_value",
        "price",
        "review_answer_delay",
        "review_score",
        "review_level",
    ]
    unique_customer[mean_features] = grouped_customer[mean_features].mean()

    unique_customer["wealthy"] = grouped_customer["price"].sum() > 500

    unique_customer["frequent_cat"] = grouped_customer.product_category_name.agg(
        lambda x: pd.Series.mode(x)[0]
    )

    unique_customer["R"] = pd.qcut(
        unique_customer["recency"], 3, labels=[1, 2, 3]
    ).astype(str)
    # Les fréquences sont trop inéquilibrées
    unique_customer["F"] = unique_customer["number_of_orders"].apply(
        lambda x: "1" if x == 1 else "2"
    )
    unique_customer["M"] = pd.qcut(
        unique_customer["amount"], 3, labels=[1, 2, 3]
    ).astype(str)

    unique_customer["RFM"] = unique_customer[["R", "F", "M"]].apply("".join, axis=1)

    unique_customer["segment"] = unique_customer["RFM"].apply(segment)

    return unique_customer.reset_index()
