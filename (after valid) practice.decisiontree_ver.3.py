# 1. DecisionTreeClassifier + K-Fold(5번) + GridSearch CV


# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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

# 5. 전처리기 정의 (범주형은 원핫인코딩, 나머지는 그대로)
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# 6. 파이프라인 정의 (전처리 + 모델)
clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', DecisionTreeClassifier(random_state=42))
])

# 7. 하이퍼파라미터 후보 지정
param_grid = {
    'model__max_depth': [3, 5, 7, 10, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__criterion': ['gini', 'entropy']
}

# 8. K-Fold 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 9. GridSearchCV 정의 (불량 recall 최적화)
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='recall',   # recall 기준으로 최적화
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

#과적합 의심됨(accuracy 1.0/다만 이전모델보다 recall값 개선)


# ###############################################
# joblib.dump(best_model, './after_valid_decisiontree.pkl')  # 저장
# # joblib.dump(저장하고 싶은 모델 변수명(모델이름), '저장하고싶은 경로랑 파일명.pkl')
# # print("✅ 모델 저장 완료!")

# loaded_model = joblib.load('./after_valid_decisiontree.pkl')  # 저장한 모델 로드

# # 불러온 모델로 예측
# y_pred = loaded_model.predict(test_x)
# print("Test Accuracy :", accuracy_score(test_y, y_pred))

##############################################
#2-1. 과적합 여부 확인(학습곡선)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label="Train score")
plt.plot(train_sizes, test_scores_mean, label="CV score")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.show()


#2-2. 과적합 여부 확인(검증곡선)
from sklearn.model_selection import validation_curve

param_range = [3, 5, 7, 10, 15, None]
train_scores, test_scores = validation_curve(
    clf, X, y,
    param_name="model__max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

plt.plot(param_range, train_scores.mean(axis=1), label="Train score")
plt.plot(param_range, test_scores.mean(axis=1), label="CV score")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Curve")
plt.show()
