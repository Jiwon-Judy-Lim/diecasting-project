# 주요변수들이 제품 품질(양품/불량)에 미치는 영향
import pandas as pd
import numpy as np

df = pd.read_csv('./data/train.csv')
df.info()
df.head()
df.columns

df.head()

df["passorfail"].value_counts()
df["name"].value_counts()
df["line"].value_counts()
df["mold_name"].value_counts()
df["working"].value_counts()
df["emergency_stop"].value_counts() #emergency stop은 on이 전부, off는 없음
df["molten_temp"].value_counts()
df["facility_operation_cycleTime"].value_counts() #히스토그램
df["facility_operation_cycleTime"].dtype #히스토그램 그려봐야할듯
df["low_section_speed"].value_counts() #히스토그램 그려봐야할듯
df["low_section_speed"].max()
df["high_section_speed"].value_counts() 
df["high_section_speed"].max()
df["high_section_speed"].min() 
df["molten_volume"].value_counts()
df["cast_pressure"].value_counts()  
df["biscuit_thickness"].value_counts()  
df["EMS_operation_time"].value_counts()  


pd.set_option('display.max_columns', None)
df.head(5)

#line, name, mold_name column없애도될듯, time(수집시간), date(수집일자), count(일자별생산번호)도 빼도될듯
#working(가동여부) 는 필요할듯. 왜냐, 설비 기계 가동여부가 품질불량에 영향을 주는지 알아야 하므로
#emergency stop은 모두 on? 다 비상정지했다는건가
#molten_temp, facility_operation_CycleTime, production_CycleTime(오전, 오후, 새벽 으로 묶어서 봐야할듯)
#facility_operation_cycleTime 하고 production_cycleTime은 왜 시간이 아니고 숫자지?

#low_section_speed 분포 확인(히스토그램)
import matplotlib.pyplot as plt

# 정상 구간 (0 ~ 300 mm/s)
import matplotlib.pyplot as plt

subset_normal = df[df["low_section_speed"] < 300]

plt.figure(figsize=(10,6))
plt.hist(subset_normal["low_section_speed"].dropna(),
         bins=30, color='steelblue', edgecolor='black')
plt.title("Histogram of Low Section Speed (0~300 mm/s)")
plt.xlabel("Low Section Speed (mm/s)")
plt.ylabel("Frequency")
plt.show()
# 주 사용구간(100~150 근처) 패턴이 뚜렷


# df["high_section_speed"] 히스토그램
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.hist(df["high_section_speed"].dropna(),
         bins=40, color='steelblue', edgecolor='black')
plt.title("Histogram of High Section Speed (0~400 mm/s)")
plt.xlabel("High Section Speed (mm/s)")
plt.ylabel("Frequency")
plt.show()


#상금형온도, 하금형온도 분포도
import matplotlib.pyplot as plt

upper_cols = ["upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3"]
lower_cols = ["lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3"]

plt.figure(figsize=(15,8))

# 상금형 온도
for i, col in enumerate(upper_cols, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")

# 하금형 온도
for i, col in enumerate(lower_cols, 1):
    plt.subplot(2, 3, i+3)
    plt.hist(df[col].dropna(), bins=30, color="tomato", edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

df.shape
df.dtypes
df.info()

#결측치 값 구하기
df["upper_mold_temp1"].isnull().sum()
df["upper_mold_temp2"].isnull().sum()
df["upper_mold_temp3"].isnull().sum()

df["lower_mold_temp1"].isnull().sum()
df["lower_mold_temp2"].isnull().sum()
df["lower_mold_temp3"].isnull().sum()

#시간대별 그래프 그리기
import matplotlib.pyplot as plt

# 요일별 평균 온도 계산
weekday_mean = df.groupby("weekday")["upper_mold_temp1"].mean()

plt.figure(figsize=(8,5))
plt.plot(weekday_mean.index, weekday_mean.values, marker='o', linestyle='-', color='steelblue')
plt.title("Average Upper Mold Temp1 by Weekday")
plt.xlabel("Weekday (0=Mon, 6=Sun)")
plt.ylabel("Upper Mold Temp1 (°C)")
plt.grid(True)
plt.show()

# upper, lower mold 합쳐서
weekday_mean = df.groupby("weekday")[[
    "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
    "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3"
]].mean()

plt.figure(figsize=(10,6))

# Upper
plt.plot(weekday_mean.index, weekday_mean["upper_mold_temp1"], marker='o', label='Upper Mold Temp1')
plt.plot(weekday_mean.index, weekday_mean["upper_mold_temp2"], marker='s', label='Upper Mold Temp2')
plt.plot(weekday_mean.index, weekday_mean["upper_mold_temp3"], marker='^', label='Upper Mold Temp3')

# Lower
plt.plot(weekday_mean.index, weekday_mean["lower_mold_temp1"], marker='o', linestyle='--', label='Lower Mold Temp1')
plt.plot(weekday_mean.index, weekday_mean["lower_mold_temp2"], marker='s', linestyle='--', label='Lower Mold Temp2')
plt.plot(weekday_mean.index, weekday_mean["lower_mold_temp3"], marker='^', linestyle='--', label='Lower Mold Temp3')

plt.title("Average Upper & Lower Mold Temps by Weekday")
plt.xlabel("Weekday (0=Mon, 6=Sun)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.show()

#요일별 Boxplot
plt.figure(figsize=(8,5))
df.boxplot(column="upper_mold_temp1", by="weekday", grid=False)
plt.title("Upper Mold Temp1 Distribution by Weekday")
plt.suptitle("")  # 기본 제목 제거
plt.xlabel("Weekday (0=Mon, 6=Sun)")
plt.ylabel("Upper Mold Temp1 (°C)")
plt.show()

#upper_mold_temp1
import matplotlib.pyplot as plt

# date 컬럼을 날짜형으로 변환 (만약 아직 object라면)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

plt.figure(figsize=(12,6))

# mold_code별로 선 그래프 그리기
for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["upper_mold_temp1"], 
             label=f"Mold {code}", alpha=0.7)

plt.title("Upper Mold Temp1 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Upper Mold Temp1 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

#upper_mold_temp1(ver.2)
# date 컬럼을 날짜형으로 변환 (만약 아직 object라면)
df_drop1 = df[df["id"] != 35449].copy()
df_drop1["date"] = pd.to_datetime(df_drop1["date"], errors="coerce")
plt.figure(figsize=(12,6))

# mold_code별로 선 그래프 그리기
# Lower Mold Temp1: 1/29(Mold 8722)
for code, group in df_drop1.groupby("mold_code"):
    plt.plot(group["date"], group["upper_mold_temp1"], 
             label=f"Mold {code}", alpha=0.7)

plt.title("Upper Mold Temp1 Over Time (ver.2)")
plt.xlabel("Date")
plt.ylabel("Upper Mold Temp1 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

#upper_mold_temp2
# date 컬럼을 날짜형으로 변환 (만약 아직 object라면)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
plt.figure(figsize=(12,6))

# mold_code별로 선 그래프 그리기
for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["upper_mold_temp2"], 
             label=f"Mold {code}", alpha=0.7)

plt.title("Upper Mold Temp2 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Upper Mold Temp2 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

#upper_mold_temp2(ver.2)
# date 컬럼을 날짜형으로 변환 (만약 아직 object라면)
df_drop = df[df["id"] != 42632].copy()
df_drop["date"] = pd.to_datetime(df_drop["date"], errors="coerce")
plt.figure(figsize=(12,6))

# mold_code별로 선 그래프 그리기
# Lower Mold Temp1: 1/29(Mold 8722)
for code, group in df_drop.groupby("mold_code"):
    plt.plot(group["date"], group["upper_mold_temp2"], 
             label=f"Mold {code}", alpha=0.7)

plt.title("Upper Mold Temp2 Over Time (ver.2)")
plt.xlabel("Date")
plt.ylabel("Upper Mold Temp2 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

#upper_mold_temp3 -> 센서고장 및 이상으로 판단(이상치 많아서 column drop)
# date 컬럼을 날짜형으로 변환 (만약 아직 object라면)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

plt.figure(figsize=(12,6))

# mold_code별로 선 그래프 그리기
for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["upper_mold_temp3"], 
             label=f"Mold {code}", alpha=0.7)

plt.title("Upper Mold Temp3 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Upper Mold Temp3 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()


# 양품(passorfail=1)과 불량(passorfail=0) 그룹별로 온도 평균, 분산 등을 비교
df.groupby("passorfail")[["upper_mold_temp1","upper_mold_temp2","upper_mold_temp3",
                          "lower_mold_temp1","lower_mold_temp2","lower_mold_temp3"]].mean()

# 분포 비교 (시각화)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.boxplot(x="passorfail", y="upper_mold_temp1", data=df)
plt.title("Upper Mold Temp1 vs Pass/Fail")
plt.show()

#lower_mold_1 
df["date"] = pd.to_datetime(df["date"], errors="coerce")

plt.figure(figsize=(12,6))

for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["lower_mold_temp1"], label=f"Mold {code}", alpha=0.7)

plt.title("Lower Mold Temp1 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Lower Mold Temp1 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()



#1/29일자 Mold 8722
import matplotlib.pyplot as plt
import pandas as pd

# date/time 합쳐서 datetime 생성
df["date"] = pd.to_datetime(df["date"])
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])

# Mold 8722, 1/29 데이터만 필터링
df_sel = df[(df["mold_code"] == 8722) & (df["date"] == pd.Timestamp("2023-01-29"))]


# 시계열 그래프 (원본 데이터)
cols = ["lower_mold_temp1","lower_mold_temp2","lower_mold_temp3",
        "upper_mold_temp1","upper_mold_temp2","upper_mold_temp3"]

plt.figure(figsize=(14,7))
for col in cols:
    plt.plot(df_sel["datetime"], df_sel[col], linestyle="-", label=col, alpha=0.7)

plt.title("Mold 8722 Temperature Trend (2023-01-29, raw data)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

print(df["date"].dtype)

#lower_mold_2 
df["date"] = pd.to_datetime(df["date"], errors="coerce")

plt.figure(figsize=(12,6))

for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["lower_mold_temp2"], label=f"Mold {code}", alpha=0.7)

plt.title("Lower Mold Temp2 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Lower Mold Temp2 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

#lower_mold_3
df["date"] = pd.to_datetime(df["date"], errors="coerce")

plt.figure(figsize=(12,6))

for code, group in df.groupby("mold_code"):
    plt.plot(group["date"], group["lower_mold_temp3"], label=f"Mold {code}", alpha=0.7)

plt.title("Lower Mold Temp3 Over Time (by Mold Code)")
plt.xlabel("Date")
plt.ylabel("Lower Mold Temp3 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show() 


#lower_mold_3(ver.2)
df_drop2= df[df["id"] != 69410].copy()
df_drop2["date"] = pd.to_datetime(df_drop2["date"], errors="coerce")

plt.figure(figsize=(12,6))

for code, group in df_drop2.groupby("mold_code"):
    plt.plot(group["date"], group["lower_mold_temp3"], label=f"Mold {code}", alpha=0.7)

plt.title("Lower Mold Temp3 Over Time (by Mold Code)_ver.2")
plt.xlabel("Date")
plt.ylabel("Lower Mold Temp3 (°C)")
plt.legend(title="Mold Code")
plt.grid(True)
plt.show() 


# 3번: 양품일때, 불량품일때 분포 확인
import seaborn as sns
import matplotlib.pyplot as plt

cols = ["upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
        "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3"]

fig, axes = plt.subplots(2, 3, figsize=(18,10))  # 2행 3열 서브플랏

for i, col in enumerate(cols):
    row, col_idx = divmod(i, 3)
    sns.violinplot(y="passorfail", x=col, data=df, ax=axes[row, col_idx], split=True, orient="h")
    axes[row, col_idx].set_title(f"{col} by PassOrFail")
    axes[row, col_idx].set_ylabel("PassOrFail (0=Fail, 1=Pass)")
    axes[row, col_idx].set_xlabel("Temperature (°C)")

plt.tight_layout()
plt.show()


#1번: (daytime) temp별 움직임 (1/29)
##Upper Mold Temp1
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 카피본 생성
df_copy = df.copy()

# date + time 합쳐서 datetime 생성 (date는 날짜, time은 시:분:초만 사용)
df_copy["datetime"] = pd.to_datetime(
    df_copy["date"].dt.strftime("%Y-%m-%d") + " " + df_copy["time"].dt.strftime("%H:%M:%S")
)

# 특정 날짜 필터링
target_date = pd.to_datetime("2019-01-29").date()
day_data = df_copy[df_copy["date"].dt.date == target_date].copy()
day_data = day_data.sort_values("datetime")

# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="upper_mold_temp1",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Upper Mold Temp1 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


##Upper Mold temp2
# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="upper_mold_temp2",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Upper Mold Temp2 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


##Upper Mold temp3
# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="upper_mold_temp3",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Upper Mold Temp3 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

##Lower Mold temp1
# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="lower_mold_temp1",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Lower Mold Temp1 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


##Lower Mold temp2
# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="lower_mold_temp2",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Lower Mold Temp2 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

##Lower Mold temp3
# 시계열 플롯
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="lower_mold_temp3",
    hue="mold_code",
    palette="tab10"
)

# x축 포맷: HH:MM:SS
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# x축 범위: 하루 고정
plt.xlim(pd.to_datetime("2019-01-29 00:00:00"),
         pd.to_datetime("2019-01-29 23:59:59"))

plt.title("Lower Mold Temp3 by Mold Code (2019-01-29)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.legend(title="Mold Code", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

#(daytime)몰드코드별로 
## 2019-01-29
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

# 카피본 생성
df_copy = df.copy()

# date + time 합쳐서 datetime 생성 (date는 날짜, time은 시:분:초만 사용)
df_copy["datetime"] = pd.to_datetime(
    df_copy["date"].dt.strftime("%Y-%m-%d") + " " + df_copy["time"].dt.strftime("%H:%M:%S")
)

# 특정 날짜 필터링
target_date = pd.to_datetime("2019-01-29").date()
day_data = df_copy[df_copy["date"].dt.date == target_date].copy()
day_data = day_data.sort_values("datetime")

# 센서 데이터를 long-form으로 변환 (melt)
sensor_cols = ["upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
               "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3"]

day_melt = day_data.melt(
    id_vars=["datetime", "mold_code"],
    value_vars=sensor_cols,
    var_name="sensor",
    value_name="temperature"
)

# 몰드코드별 subplot
unique_codes = day_melt["mold_code"].unique()
n_codes = len(unique_codes)

fig, axes = plt.subplots(n_codes, 1, figsize=(14, 4*n_codes), sharex=True)

if n_codes == 1:
    axes = [axes]  # subplot이 하나일 경우 리스트로 변환

for ax, code in zip(axes, unique_codes):
    subset = day_melt[day_melt["mold_code"] == code]
    sns.lineplot(
        data=subset,
        x="datetime",
        y="temperature",
        hue="sensor",
        ax=ax,
        palette="tab10"
    )
    ax.set_title(f"Mold Code {code} - Temperatures (2019-01-29)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(title="Sensor", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

# ✅ 모든 subplot의 x축 범위를 하루(00:00~23:59)로 고정
for ax in axes:
    ax.set_xlim(pd.to_datetime("2019-01-29 00:00:00"),
                pd.to_datetime("2019-01-29 23:59:59"))

# x축 HH:MM 표시
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

plt.xlabel("Time (HH:MM:SS)")
plt.tight_layout()
plt.show()

#(daytime)몰드코드별로 
## 2019-01-04
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

# 카피본 생성
df_copy = df.copy()

# date + time 합쳐서 datetime 생성 (date는 날짜, time은 시:분:초만 사용)
df_copy["datetime"] = pd.to_datetime(
    df_copy["date"].dt.strftime("%Y-%m-%d") + " " + df_copy["time"].dt.strftime("%H:%M:%S")
)

# 특정 날짜 필터링
target_date = pd.to_datetime("2019-01-04").date()
day_data = df_copy[df_copy["date"].dt.date == target_date].copy()
day_data = day_data.sort_values("datetime")

# 센서 데이터를 long-form으로 변환 (melt)
sensor_cols = ["upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
               "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3"]

day_melt = day_data.melt(
    id_vars=["datetime", "mold_code"],
    value_vars=sensor_cols,
    var_name="sensor",
    value_name="temperature"
)

# 몰드코드별 subplot
unique_codes = day_melt["mold_code"].unique()
n_codes = len(unique_codes)

fig, axes = plt.subplots(n_codes, 1, figsize=(14, 4*n_codes), sharex=True)

if n_codes == 1:
    axes = [axes]  # subplot이 하나일 경우 리스트로 변환

for ax, code in zip(axes, unique_codes):
    subset = day_melt[day_melt["mold_code"] == code]
    sns.lineplot(
        data=subset,
        x="datetime",
        y="temperature",
        hue="sensor",
        ax=ax,
        palette="tab10"
    )
    ax.set_title(f"Mold Code {code} - Temperatures (2019-01-04)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(title="Sensor", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

# ✅ 모든 subplot의 x축 범위를 하루(00:00~23:59)로 고정
for ax in axes:
    ax.set_xlim(pd.to_datetime("2019-01-04 00:00:00"),
                pd.to_datetime("2019-01-04 23:59:59"))

# x축 HH:MM 표시
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

plt.xlabel("Time (HH:MM:SS)")
plt.tight_layout()
plt.show()


# 날짜만 추출해서 그룹별로 몇 개의 mold_code가 있었는지 확인
mold_per_day = df.groupby(df["date"].dt.date)["mold_code"].nunique()

# 총 몰드 개수
total_molds = df["mold_code"].nunique()

# 모든 몰드(예: 5개)가 다 가동된 날짜만 필터링
full_days = mold_per_day[mold_per_day == total_molds]

print("날짜별 가동된 몰드 수:")
print(mold_per_day)
print("\n모든 몰드가 다 가동된 날짜:")
print(full_days.index)


# lower temp2 mold 8722 이상치여부 확인
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

# 카피본 생성
df_copy = df.copy()

# datetime 생성 (date + time 합치기)
df_copy["datetime"] = pd.to_datetime(
    df_copy["date"].dt.strftime("%Y-%m-%d") + " " + df_copy["time"].dt.strftime("%H:%M:%S")
)

# 2019-01-02 + Mold Code 8722 필터링
target_date = pd.to_datetime("2019-01-02").date()
day_data = df_copy[
    (df_copy["date"].dt.date == target_date) & 
    (df_copy["mold_code"] == 8722)
].copy()

# 정렬
day_data = day_data.sort_values("datetime")

# 시계열 플롯 (Lower Mold Temp2)
plt.figure(figsize=(14,6))
sns.lineplot(
    data=day_data,
    x="datetime",
    y="lower_mold_temp2",
    marker="o",
    color="red"
)

# x축 HH:MM 포맷
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 단위 눈금
plt.xticks(rotation=45)

# 타이틀 및 라벨
plt.title("Lower Mold Temp2 - Mold 8722 (2019-01-02)")
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()
