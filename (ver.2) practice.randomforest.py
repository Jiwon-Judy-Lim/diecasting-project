# 모델조합: RandomForest + SMOTE + GridSearchCV + threshold변경

# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # imblearn의 Pipeline
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. 데이터 불러오기
df = pd.read_csv('./data/data1.csv')
df['passorfail'] = df['passorfail'].astype(int)

# 3. 데이터 분할
X = df.drop(columns='passorfail')
y = df['passorfail']

# 30(test):70(train)으로 분할
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

# 6. 파이프라인 정의 (전처리 + SMOTE + 랜덤포레스트)
rf_pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# 7. 하이퍼파라미터 후보 지정
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__criterion': ['gini', 'entropy']
}

# 8. K-Fold 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 9. GridSearchCV 정의 (recall 기준 최적화)
grid_search = GridSearchCV(
    estimator=rf_pipeline,
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
test_pred = best_model.predict(test_x)  #threshold 0.5로 설정됨(.predict는 기본 임계값 사용)

print("\nTrain Accuracy:", accuracy_score(train_y, train_pred))
print("Test Accuracy :", accuracy_score(test_y, test_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(test_y, test_pred))

print("\nClassification Report (Test):")
print(classification_report(test_y, test_pred, digits=4))

#threshold 조정
# (1) 불량 클래스(=1)의 예측 확률 추출
y_prob = best_model.predict_proba(test_x)[:, 1]

# (2) threshold 값 설정 (기본 0.5 → 0.4로 낮춰보기)
threshold = 0.4
y_pred_threshold = (y_prob >= threshold).astype(int)

# (3) 평가 지표 출력
print(f"\n=== Threshold = {threshold} ===")
print("Accuracy :", accuracy_score(test_y, y_pred_threshold))
print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred_threshold))
print("\nClassification Report:\n", classification_report(test_y, y_pred_threshold, digits=4))


# 13. 모델 저장
joblib.dump(best_model, './ver.2_randomforest.pkl')
print("✅ 최적 랜덤포레스트(SMOTE 포함) 모델 저장 완료!")

# 14. 모델 불러오기
loaded_model = joblib.load('./ver.2_randomforest.pkl')
y_pred = loaded_model.predict(test_x)
print("Reloaded Model Test Accuracy :", accuracy_score(test_y, y_pred))
