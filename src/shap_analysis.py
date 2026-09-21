"""SHAP interpretation of the Gradient Boosting SPI_sensor surrogate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from run_analysis import FEATURES, RANDOM_STATE, RESULTS, calculate_components, calculate_rankings, load_data


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = calculate_rankings(calculate_components(load_data()))
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(data["SPI_sensor"], errors="coerce")

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_i = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train_i, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_i)

    importance = pd.DataFrame(
        {
            "Descriptor": FEATURES,
            "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("Mean_abs_SHAP", ascending=False)
    importance.to_csv(RESULTS / "shap_feature_importance.csv", index=False)

    shap.summary_plot(shap_values, X_test_i, max_display=14, show=False)
    plt.xlabel(r"SHAP value (impact on predicted SPI$_{sensor}$)")
    plt.tight_layout()
    plt.savefig(RESULTS / "SHAP_SPI_sensor.png", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

