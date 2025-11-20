import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib

DATA_PATH = 'data/nigeria_crop_yield.csv'
MODEL_PATH = 'models/nigeria_yield_model.pkl'


def train():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)

    # Target: Yield
    # Features: Item, Year, Area, Rain, Temp
    # DROP Production (Target Leakage) and the Target itself
    X = df.drop(columns=['yield_kg_ha', 'production_tonnes'])
    y = df['yield_kg_ha']

    categorical_features = ['Item']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5
        ))
    ])

    print("Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    score = r2_score(y_test, predictions)
    print(f"Model Trained! R² Score: {score:.4f}")

    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()