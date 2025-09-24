# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 2. 데이터 불러오기
df = pd.read_csv('./data/data1.csv')

# passorfail을 int로 변환 (float일 경우 대비)
df['passorfail'] = df['passorfail'].astype(int)

# 3. 데이터 분할
X = df.drop(columns='passorfail')
y = df['passorfail']

train_x, test_x, train_y, test_y = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=0
)

# 4. 범주형 변수 지정
cat_cols = ['working', 'mold_code']

# 5. 전처리기 정의
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# 6. 파이프라인 정의 (전처리 + 랜덤포레스트)
rf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', RandomForestClassifier(random_state=42))
])

# 7. 하이퍼파라미터 후보 지정
param_grid = {
    'model__n_estimators': [100, 200],        # 트리 개수
    'model__max_depth': [5, 10, None],        # 트리 깊이
    'model__min_samples_split': [2, 5],       # 분할 최소 샘플 수
    'model__min_samples_leaf': [1, 2],        # 리프 최소 샘플 수
    'model__criterion': ['gini', 'entropy']   # 지니 or 엔트로피
}

# 8. K-Fold 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 9. GridSearchCV 정의 (recall 최적화)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='recall',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# 10. 학습
grid_search.fit(train_x, train_y)

# 11. 최적 파라미터 출력
print("Best Parameters:", grid_search.best_params_)
print("Best Recall Score (CV):", grid_search.best_score_)

# 12. 최적 모델 평가
best_model = grid_search.best_estimator_

train_pred = best_model.predict(train_x)
test_pred = best_model.predict(test_x)

print("\nTrain Accuracy:", accuracy_score(train_y, train_pred))
print("Test Accuracy :", accuracy_score(test_y, test_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(test_y, test_pred))

print("\nClassification Report (Test):")
print(classification_report(test_y, test_pred, digits=4))

# 13. ✅ 모델 저장
joblib.dump(best_model, './after_valid_randomforest.pkl')
print("✅ 최적 랜덤포레스트 모델 저장 완료!")

# 14. 저장한 모델 불러오기
loaded_model = joblib.load('./after_valid_randomforest.pkl')

# 불러온 모델로 예측
y_pred = loaded_model.predict(test_x)
print("Reloaded Model Test Accuracy :", accuracy_score(test_y, y_pred))
