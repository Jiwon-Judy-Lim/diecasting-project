# 1. DecisionTreeClassifier + K-Fold(5번) + GridSearch CV + 하이퍼파라미터 조정
# 사유: 왜 다시 조정해야 하냐면?
# GridSearch는 recall 기준으로 최적값을 찾았어요 → 그 결과가 max_depth=None이었죠.
# 근데 Validation Curve를 보니까 max_depth가 깊어질수록 train↑, CV↓ 패턴이 뚜렷해요 → 전형적인 과적합 시그널.
# 따라서 recall만 보지 말고, **generalization(일반화)**도 고려해서 파라미터를 다시 탐색하는 게 좋아요.

# 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 데이터 불러오기
df = pd.read_csv('./data/data1.csv')

# passorfail을 int로 변환 (float일 경우 대비)
df['passorfail'] = df['passorfail'].astype(int)

# 데이터 분할
X = df.drop(columns='passorfail')
y = df['passorfail']

train_x, test_x, train_y, test_y = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=0
)

# 범주형 변수 지정
cat_cols = ['working', 'mold_code']

# 전처리기 정의 (범주형은 원핫인코딩, 나머지는 그대로)
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# 파이프라인 정의 (전처리 + 모델)
clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', DecisionTreeClassifier(random_state=42))
])

# 하이퍼파라미터 후보 지정
param_grid = {
    'model__max_depth': [3, 5, 7, 9], #none없이 max depth 지정
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


#Conclusion: max_depth 제한-> 모델이 단순해져서 일반화 성능 좋아짐. recall 손해
# 즉, "안정성 vs 불량 검출력"의 트레이드오프가 생긴 거예요.
# 제한 없음(None) → recall ↑, 하지만 과적합 위험
# 제한 있음(예: 5~9) → recall ↓, 대신 과적합 ↓
# 