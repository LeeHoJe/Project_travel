import pandas as pd
import numpy as np

from DBManager import DBManager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#pip install seaborn
mydb = DBManager()
sql="""
select a.*
from (
        select 
               방문지정보.VISIT_AREA_NM as VISIT_NM
               ,방문지정보.VISIT_AREA_TYPE_CD as VISIT_TYPE
               ,여행.mvmn_nm as MOVING
               ,gender          
               ,age_grp                 as AGE
               ,marr_stts               as MARR
               ,travel_styl_1           as 자연_도시
               ,travel_styl_2           as 숙박_당일
               ,travel_styl_4           as 비싼숙소_저렴한숙소
               ,travel_styl_5           as 휴양OR휴식_체험
               ,travel_styl_6           as 숨은여행지_유명여행지
               ,travel_styl_8           as 비촬영여행지_사진촬영여행지
               ,TRAVEL_STATUS_ACCOMPANY as 동반자수
               ,TRAVEL_MOTIVE_1         
               ,TRAVEL_COMPANIONS_NUM
        from 여행객, 방문지정보, 여행
        where 여행객.traveler_id=여행.traveler_id
        and 여행.travel_id=방문지정보.travel_id
        and 방문지정보.VISIT_AREA_NM not LIKE '%국제공항%'
        and 방문지정보.VISIT_AREA_NM not LIKE '%휴게소%'
        and 방문지정보.VISIT_AREA_NM not LIKE '%집%'
        and 방문지정보.VISIT_AREA_NM not LIKE '%사무실%'
        and 방문지정보.VISIT_AREA_NM not LIKE '%숙소%'
        
        order by 방문지정보.VISIT_AREA_NM
) a
where VISIT_NM in (
                    select VISIT_NM
                    from (
                    select 
                           방문지정보.VISIT_AREA_NM as VISIT_NM
                           ,방문지정보.VISIT_AREA_TYPE_CD as VISIT_TYPE
                           ,여행.mvmn_nm as MOVING
                           ,gender          
                           ,age_grp                 as AGE
                           ,marr_stts               as MARR
                           ,travel_styl_1           as 자연_도시
                           ,travel_styl_2           as 숙박_당일
                           ,travel_styl_4           as 비싼숙소_저렴한숙소
                           ,travel_styl_5           as 휴양OR휴식_체험
                           ,travel_styl_6           as 숨은여행지_유명여행지
                           ,travel_styl_8           as 비촬영여행지_사진촬영여행지
                           ,TRAVEL_STATUS_ACCOMPANY as 동반자수
                           ,TRAVEL_MOTIVE_1         
                           ,TRAVEL_COMPANIONS_NUM
                    from 여행객, 방문지정보, 여행
                    where 여행객.traveler_id=여행.traveler_id
                    and 여행.travel_id=방문지정보.travel_id
                    and 방문지정보.VISIT_AREA_NM not LIKE '%국제공항%'
                    and 방문지정보.VISIT_AREA_NM not LIKE '%휴게소%'
                    and 방문지정보.VISIT_AREA_NM not LIKE '%집%'
                    and 방문지정보.VISIT_AREA_NM not LIKE '%사무실%'
                    and 방문지정보.VISIT_AREA_NM not LIKE '%숙소%'
                    order by 방문지정보.VISIT_AREA_NM
                    )
                    group by VISIT_NM
                    having count(*) >=80
                )

"""
df=pd.read_sql(con = mydb.conn,sql=sql)
print(df.head())


# for cnt in df:
#     print(df[cnt].value_counts())
#     print("%.2f%%" % (105.0 / 300.0 * 100.0))
#     print('=' * 50)
######################카운트출력#############
# for col in df:
#     counts = df[col].value_counts()
#     total_count = len(df)
#     percentages = (counts / total_count) * 100
#     print(pd.concat([counts, percentages], axis=1, keys=['Counts', 'Percentages']))
#     #print("%.2f%%"%(pd.concat([counts, percentages], axis=1, keys=['Counts', 'Percentages'])))
#     print('=' * 50)
x=df[['VISIT_TYPE','MOVING','GENDER','AGE','MARR','자연_도시','숙박_당일','비싼숙소_저렴한숙소','휴양OR휴식_체험','숨은여행지_유명여행지','비촬영여행지_사진촬영여행지','동반자수', 'TRAVEL_MOTIVE_1', 'TRAVEL_COMPANIONS_NUM']]
x_encoded = pd.get_dummies(x, columns=['VISIT_TYPE', 'MOVING', 'GENDER', 'AGE', 'MARR', '자연_도시', '숙박_당일', '비싼숙소_저렴한숙소', '휴양OR휴식_체험', '숨은여행지_유명여행지', '비촬영여행지_사진촬영여행지', '동반자수', 'TRAVEL_MOTIVE_1', 'TRAVEL_COMPANIONS_NUM'])
y = df[['VISIT_NM']]

model=tree.DecisionTreeClassifier()

print(df['MOVING'])
enc_class={}

def encoding_label(x):
    le=LabelEncoder()
    le.fit(x)
    label=le.transform(x)
    enc_class[x.name]=le.classes_
    return label
x_data = x[x.columns].apply(encoding_label)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y, train_size=0.8, test_size=0.2)



model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

pred=model.predict(x_test)
prob = model.predict_proba(x_test)
print(pred, prob)
test_accuracy = accuracy_score(pred, y_test)
print('acc test', test_accuracy)
train_pred=model.predict(x_train)
prob = model.predict_proba(x_train)
train_accuracy = accuracy_score(train_pred, y_train)
print('acc train', train_accuracy)


#model.save('traver_01.h5')
import joblib

joblib.dump(model, 'decision_tree_model.pkl')

loaded_model = joblib.load('decision_tree_model.pkl')

# You can now use 'loaded_model' to make predictions or perform other tasks.

#########입력

test=pd.DataFrame({'VISIT_TYPE':3,'MOVING':1,'GENDER':2,'AGE':1,'MARR':1,'자연_도시':1,'숙박_당일':1,'비싼숙소_저렴한숙소':1,'휴양OR휴식_체험':1
                      ,'숨은여행지_유명여행지':1,'비촬영여행지_사진촬영여행지':1,'동반자수':1, 'TRAVEL_MOTIVE_1':1
                      , 'TRAVEL_COMPANIONS_NUM':0},index=[0])
pred1=model.predict(test)
prob1 = model.predict_proba(test)
print(pred1, prob1)
# test_accuracy1 = accuracy_score(pred1, prob1)
# print('acc test', test_accuracy1)



# train_input=x_train.reshape(-1,1)
# print(model.score(x_train,y.reshape(-1,1)))

#
# cat_col = df.columns[df.dtypes == object]
#
# # for col in cat_col:
# #     print(df[col].value_counts())
# #     print('=' * 50)
# # print(df['GENDER'])

# cat_cols = df.select_dtypes(include=['object']).columns
# for col in cat_cols:
#     plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
#     sns.countplot(x=col, data=df, order=df[col].value_counts().index)
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
#     plt.title(f'Count of {col}')
#     plt.rcParams['font.family'] = 'Malgun Gothic'
#     plt.show()
# print(df.nunique())
# print(df.shape)

#df2 = sns.df('VISIT_AREA_NM')
#sns.barplot(x='gender', y='VISIT_AREA_NM', data=df)
