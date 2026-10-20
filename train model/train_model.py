import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path


_MODELS_DIR = Path(__file__).parent.parent / "models"
_PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
MODEL_FILE = _MODELS_DIR / "housing_model.pkl"
PIPELINE_FILE = _MODELS_DIR / "housing_pipeline.pkl"


def build_pipeline(numerical_attributes, categorical_attributes):
	numerical_pipeline = Pipeline(
		[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	categorical_pipeline = Pipeline(
		[
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	return ColumnTransformer(
		[
			("num", numerical_pipeline, numerical_attributes),
			("cat", categorical_pipeline, categorical_attributes),
		]
	)


def train_model():
	housing_df = pd.read_csv(Path(__file__).parent / ".." / "dataset" / "housing.csv")
	housing_df["income_category"] = pd.cut(
		housing_df["median_income"],
		bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
		labels=[1, 2, 3, 4, 5],
	)

	split_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

	for train_index, test_index in split_strategy.split(
		housing_df, housing_df["income_category"]
	):
		_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
		housing_df.loc[test_index].drop("income_category", axis=1).to_csv(
			_PREDICTIONS_DIR / "input_data.csv", index=False
		)
		housing = housing_df.loc[train_index].drop("income_category", axis=1)

	housing_labels = housing["median_house_value"].copy()
	housing_features = housing.drop("median_house_value", axis=1)

	numerical_attributes = (
		housing_features.drop("ocean_proximity", axis=1).columns.tolist()
	)
	categorical_attributes = ["ocean_proximity"]

	pipeline = build_pipeline(numerical_attributes, categorical_attributes)
	housing_prepared = pipeline.fit_transform(housing_features)

	model = RandomForestRegressor(random_state=42)
	model.fit(housing_prepared, housing_labels)

	_MODELS_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, MODEL_FILE)
	joblib.dump(pipeline, PIPELINE_FILE)

	print("Model and pipeline saved successfully and trained on the data")


if __name__ == "__main__":
	train_model()
