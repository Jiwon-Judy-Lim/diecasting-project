import pandas as pd
import numpy as np

# 1. 의사결정나무(decision tree) / DecisionTreeRegressor
# (1)데이터분할
train=pd.read_csv("./data/train.csv") #train은 훈련시킬 데이터
test=pd.read_csv('./data/test.csv') #test는 검증용 데이터(모델의 성능을 검증하는데 쓰는 데이터)

train.info()
test.info()
train.columns
train.dtypes

drop_cols = [
    'line','name','mold_name','time','date',
    'working','emergency_stop','registration_time',
    'tryshot_signal','heating furnance'
]

# (2) 입력데이터와 정답데이터 분류
train_X = train.drop(['passorfail','upper_mold_temp3','lower_mold_temp3'] + drop_cols, axis=1) #독립변수()
train_y = train['passorfail'] #y는 종속변수(내가 예측하고 싶은 값)

test_X = test.drop(['upper_mold_temp3','lower_mold_temp3'] + drop_cols, axis=1) # test데이터에서 drop할 변수는 train과 동일해야함.


# # (3) 데이터 전처리 진행
# from sklearn.impute import SimpleImputer #결측치 처리(방법: 평균, 중앙값, 최빈값, 고정값으로 채움)
# from sklearn.preprocessing import OneHotEncoder, StandardScaler #Onehot: 범주형변수 -> 수치형 변수 변환 #StandardScaler수치형 변수의 스케일을 표준화
# from sklearn.compose import ColumnTransformer, make_column_transformer #컬럼별로 다른 전처리 적용
# from sklearn.pipeline import Pipeline, make_pipeline #여러단계(전처리->모델)를 하나로 묶는 통합 프로세스

# #데이터전처리
# train['passorfail'].value_counts() #y변수(종속변수) 범주형 -> 숫자형으로 인코딩
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# train_y = labelencoder.fit_transform(train_y) #라벨 매핑학습(fit) + 변환(transform) 동시 실행
# test_y = labelencoder.transform(test_y) #위에 매핑해둔 라벨로 변환만해야함


# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline, make_pipeline
# num_columns = train_X.select_dtypes('number').columns.tolist()
# num_preprocess = make_pipeline(
#     StandardScaler(),
#     PCA(n_components=0.8, svd_solver='full'))
# preprocess = ColumnTransformer(
#     [("num", num_preprocess, num_columns)]
# )

# #모델생성
# from sklearn.tree import DecisionTreeClassifier
# full_pipe = Pipeline(
#     [
#         ("preprocess", preprocess),
#         ("classifier", DecisionTreeClassifier())
#     ]
# )

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 모델 생성 (랜덤시드 고정하면 재현 가능)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)

# 2. 모델 학습 (훈련)
dt_model.fit(train_X, train_y)

# 3. train 데이터에 대한 예측 (훈련 데이터 정확도 확인용)
train_pred = dt_model.predict(train_X)
print("Train Accuracy:", accuracy_score(train_y, train_pred))

# 4. test 데이터에 대한 예측
test_pred = dt_model.predict(test_X)

# 5. 예측 결과 DataFrame으로 저장 (제출용 or 확인용)
output = pd.DataFrame({
    "id": test["id"],            # test 데이터에 id 컬럼이 있다고 가정
    "passorfail_pred": test_pred
})

print(output.head())

