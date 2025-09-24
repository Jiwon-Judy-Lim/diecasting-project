
# 1. 의사결정나무(decision tree) / DecisionTreeRegressor
# (1)데이터분할 및 전처리
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




df = pd.read_csv('./data/data1.csv')

X = df.drop(columns= 'passorfail')
y = df['passorfail']

train , test= train_test_split(
                    df,
                    test_size = 0.3, 
                    stratify = df['passorfail'], 
                    random_state = 0)

###
train_x = train.drop(columns = 'passorfail')
train_y = train['passorfail']

test_x = test.drop(columns = 'passorfail')
test_y = test['passorfail']


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 범주형 변수 지정
cat_cols = ['working', 'mold_code']

# 2. 전처리기 정의 (범주형은 원핫인코딩, 수치형은 그대로 통과)
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'  # 숫자형 컬럼은 그대로 사용
)

# 3. 파이프라인 정의 (전처리 + 모델)
clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', DecisionTreeClassifier(random_state=42, max_depth=5))
])

# 4. 모델 학습
clf.fit(train_x, train_y)

# 5. 예측
train_pred = clf.predict(train_x)
test_pred = clf.predict(test_x)

# 6. 평가
print("Train Accuracy:", accuracy_score(train_y, train_pred))
print("Test Accuracy :", accuracy_score(test_y, test_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(test_y, test_pred))

print("\nClassification Report (Test):")
print(classification_report(test_y, test_pred))

joblib.dump(clf, './before_valid_decisiontree.pkl')  # 저장
# joblib.dump(저장하고 싶은 모델 변수명, '저장하고싶은 경로랑 파일명.pkl')
# print("✅ 모델 저장 완료!")


import joblib
loaded_model = joblib.load('./before_valid_decisiontree.pkl')  # 저장한 모델 로드

# 불러온 모델로 예측
y_pred = loaded_model.predict(test_x)
print("Test Accuracy :", accuracy_score(test_y, y_pred))