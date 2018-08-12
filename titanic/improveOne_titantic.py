import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# seaborn 绘图神器
import seaborn as sns
import re
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 将两个文件放在同目录下，通过pandas读取数据
# train为训练集（给模型训练用），test为测试集（给训练好的模型测试用）
data_train = pd.read_csv('dataSet/train.csv')
data_test = pd.read_csv('dataset/test.csv')
# 合并训练集和测试集
df = data_train.append(data_test)

## 1. 查看train和test合并后的数据情况
print('-------df.info--------')
print(df.info())

print('df.shape==',df.shape)
print('合并后一共{0}条数据'.format(str(df.shape[0])))

## 查看各项特征的缺失值数量
print(pd.isnull(df).sum())

# 对数值型特征进行简单的描述性统计，包括均值，中位数，众数，方差，标准差，最大值，最小值等
df.describe()

'''
描述性统计主要 用于粗略判断 哪些特征存在,异常值初始观察（主要观察一下最大与最小值）：
Fare：船票价格平均值33.2，中位数14，平均值比中位数大很多，说明该特征分布是严重的右偏，又看到最大值512，所以512很可能是隐患的异常值。
Age：最小值为0.17，最大值为80，0.17是大概刚出生一个半月的意思，而80年龄有些过大。 SibSp与Parch：Sibsp最大为8，可能是异常，但又看到Parch最大值为9。这两个特征同时出现大的数值，说明了这个数值是有可能的，我们进步一观察。 结论： 通过以上观察和分析
，我们看到了一些可能的异常值，但是也不敢肯定。这需要我们进一步通过可视化来清楚的显示并结合对业务的理解来确定。
'''

# 防中文乱码设置
plt.style.use("bmh")
plt.rc('font', family='SimHei', size=13)


'''
   定类/定序特征分析

'''
# 查看各特征值的类型数量
print('----各特征值的类型数量汇总-----')
cat_list = ['Pclass','Name','Sex','SibSp','Embarked','Parch','Ticket','Cabin']
for n,i in enumerate(cat_list):
    Cabin_cat_num = df[i].value_counts().index.shape[0]
    print('{0}. {1}特征的类型数量是: {2}'.format(n+1,i,Cabin_cat_num))
'''
小结 1
类型较少的特征 Pclass,Sex,SibSp,Embarked,Parch等可进行可视化分析
类型较多的特征 name和Ticket和Cabin 不方便进行可视化
'''
# 依次对 sex,SibSp和parch 做对 Survived 的 分类柱状图
f,[ax1, ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
# sns.counplot 计数条形图

# 设置画布的主标题 和 尺寸
f.suptitle('数据类型特征分析',size=20,y=1.1)

sns.countplot(x = 'Sex', hue = 'Survived',data=data_train,ax=ax1)
sns.countplot(x='Pclass', hue='Survived', data=data_train, ax=ax2)
sns.countplot(x='Embarked', hue='Survived', data=data_train, ax=ax3)

ax1.set_title('Sex特征分析')
ax2.set_title('Pclass特征分析')
ax3.set_title('Embarked特征分析')

f, [ax1,ax2] = plt.subplots(1,2,figsize=(20,5))
sns.countplot(x='SibSp', hue='Survived', data=data_train, ax=ax1)
sns.countplot(x='Parch', hue='Survived', data=data_train, ax=ax2)
ax1.set_title('SibSp特征分析')
ax2.set_title('Parch特征分析')


plt.show()

'''
小结 2：
Sex： 女性获救比例远远高于男性
Pclass： 社会等级为3的总人数最多，但是获救率非常低
Embarked： 登陆港口S数量最多，但是获救率也是最低的，C港口获救率最高；
SibSp： 兄弟姐妹数量最低为0的人数最多，但是获救率最低，而为1的获救率相对较高，超过50%
Parch： 情况基本同上
就以上5个特征来看,Sex和Pclass两个特征是其中非常有影响的两个。
'''

# 在 Pclass 即不同的社会等级下，男性和女性在不同登陆港口下的数量对比

grid = sns.FacetGrid(df, col='Pclass', hue='Sex', palette='seismic', size=4)
grid.map(sns.countplot, 'Embarked', alpha=0.8)
grid.add_legend()

'''
# 小结 3
Pclass为1和2的时候，Q港口数量几乎为零，而Pclass3的Q港口人数甚至比C港口多。
Pclass为2的港口中，男性与女性在S和C港口的数量分布呈现相反趋势，与其他Pclass等级截然不同，这说明Pclass2可能是社会中某个共性群体，
这个群体多为女性，而男性很少。
'''

'''
定距、定比特征分析
'''
# kde分布
f,ax = plt.subplots(figsize=(10,5))
sns.kdeplot(data_train.loc[(data_train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age特征分布 - Surviver V.S. Not Survivors', fontsize = 15)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)


'''
小结 4
小于15岁以下的乘客获救率非常高，大于15岁的乘客没有明显差别。
'''

# 箱型图特征分析
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(20,6))

sns.boxplot(x="Pclass", y="Age", data=data_train, ax =ax1)
sns.swarmplot(x="Pclass", y="Age", data=data_train, ax =ax1)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 3),'Age'] , color='b',shade=True, label='Pcalss3',ax=ax2)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 1),'Age'] , color='g',shade=True, label='Pclass1',ax=ax2)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 2),'Age'] , color='r',shade=True, label='Pclass2',ax=ax2)
ax1.set_title('Age特征在Pclass下的箱型图', fontsize = 18)
ax2.set_title("Age特征在Pclass下的kde图", fontsize = 18)
fig.show()
'''
小结 5：
不同Pclass下的年龄分布也不同，三个分布的中位数大小按 Pclass1 > Pclass2 > Pclass3 排列。这也符合实际情况，Pclass1的乘客是社会上的拥有一定财富和地位的成功人士，年龄比较大，而Pclass3的人数最多， 因为大多数
人还都是普通人（有钱人毕竟少数），并且这些人多是年轻人，年龄在20-30之间。
'''

# Sex，Pclass分类条件下的 Age年龄对Survived的散点图
grid = sns.FacetGrid(data_train, row='Sex', col='Pclass', hue='Survived', palette='seismic', size=3.5)
grid.map(plt.scatter, 'PassengerId', 'Age', alpha=0.8)
grid.add_legend()

'''
小结 6：

从散点图来分析：:
Pclass1和Pclass2的女性几乎都是Survived的，Pclass3中女性Survived则不是很明显了。
Pclass1的男性生还率最高，Pclass2和Pclass3的生还率比较低，但是Pclass2中年龄小的乘客几乎全部生存。
印证了原则：妇女和孩子优先营救。
'''


'''
  三 、 数据预处理
#步骤#
1、 对 异常值和缺失数据 进行清洗
2、 进行特征转化,数值化处理，便于算法运算
3、 更具具体情况对一些定比特征进行“分箱”操作
4、 尝试做出一些认为非常有影响力的 “衍生变量”，并加入到数据中
5、 整理数据，建立一个模型（sklearn），输出预测结果
'''

# 1、Fare 缺失值处理
print('----Fare 缺失值----')
print(df[df['Fare'].isnull()])

# 定位该缺失值人的其他特点， 找同类人群进行填补缺失值
# df.loc 行查询
df.loc[(df['Pclass']==3)&(df['Age']>60)&(df['Sex']=='male')]

# 定位该缺失值人的其他特点， 找同类人群进行填补缺失值
print(df.loc[(df['Pclass']==3)&(df['Age']>60)&(df['Sex']=='male')])
# 填充该同样类别的平均值
df.loc[df.Name == 'Storey, Mr. Thomas','Fare']=df[(df['Age']>=60)&(df['Pclass']==3)&(df['Sex']=='male')]['Fare'].mean()

# 2、Embarked特征缺失值：
# 上面可视化分析过，pclass1且为女性的情况下，Q港口几乎为0，而C港口最多，其次S港口，因此用众数C港口进行填补。
df['Embarked'] = df['Embarked'].fillna('C')

# 3、Cain特征有70%的缺失值， 较为严重，如果进行大量的填补会引入更多噪声。因为缺失值也是一种值，这里将Cabin是否为缺失值作为一个新的特征来处理。
# 将Cabin 空值填充 0 并且 创建新的特征值 CabingCat，赋值给 CabinCat
df['CabinCat'] = df.Cabin.fillna('0')
#  取Cabin 中每个元素的第一个值  ,用 apply
df['CabinCat'] = df.Cabin.fillna('0').apply(lambda x: x[0])
# 对 CabinCat 进行 按类别向量化 并赋值给 df['CabinCat']
df['CabinCat'] = pd.Categorical(df['CabinCat']).codes

fig,ax = plt.subplots(figsize=(10,5))
sns.countplot(x='CabinCat',hue='Survived',data=df)

'''
  四 、特征工程(1)
特征工程是 抽取出原特征值中更多信息，并且生成新的特征值，简单来说，是一种升维的方法 现在对 Name，SibSp +Parch,Embarked,Sex进行特征工程
Title：从Name中提取Title信息，因为同为男性，Mr.和 Master.的生还率是不一样的；
TitleCat：映射并量化Title信息，虽然这个特征可能会与Sex有共线性，但是我们先衍生出- 来，后进行筛选；
FamilySize：可视化分析部分看到SibSp和Parch分布相似，固将SibSp和Parch特征进行组合；
NameLength：从Name特征衍生出Name的长度，因为有的国家名字越短代表越显贵；
CabinCat：按照Cabin类型进行数值化；

方法有：特征信息抽取法，组合相加法，分箱法，类别数值化，独立编码分组

'''

# 从Name中提取Title信息，因为同为男性，Mr.和 Master.的生还率是不一样的
df["Title"] = df["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

# 量化Title信息
df["TitleCat"] = df.loc[:,'Title'].map(title_mapping)

# SibSp和Parch特征进行组合
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 根据FamilySize分布进行分箱  分箱操作 cut
df["FamilySize"] = pd.cut(df["FamilySize"], bins=[0,1,4,20], labels=[0,1,2])

# 从Name特征衍生出Name的长度
df["NameLength"] = df["Name"].apply(lambda x: len(x))

# 量化Embarked特征
df["Embarked"] = pd.Categorical(df.Embarked).codes

# 对Sex特征进行  独热编码分组
df = pd.concat([df,pd.get_dummies(df['Sex'])],axis=1)

'''
四 、特征工程(2)
   高级衍生法
   1、 多个特征值中抽取特定的特征值
   2、 单个特征值中抽取有用的信息，形成新的特征值
   3、 衍生特征法：
      (1)将某单个特征值转成DataFrame,并对同值计算value_counts()
        eg: table_ticket = pd.DataFrame(df["Ticket"].value_counts())
      (2)给 该DataFrame 的columns 重新命名
        eg:table_ticket.rename(columns={'Ticket':'Ticket_Numbers'}, inplace=True)
      (3) 对table_ticket 按照 若干个条件 生成几个对目标分类相关度高的特征值
        eg:table_ticket['Ticket_dead_women'] = df.Ticket[(df.person == 'female_adult')
                                    & (df.Survived == 0.0)
                                    & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()
        table_ticket['Ticket_dead_women'] = table_ticket['Ticket_dead_women'].fillna(0)
        table_ticket['Ticket_dead_women'][table_ticket['Ticket_dead_women'] > 0] = 1.0
      (4)分箱法处理 数量不多但有区域的特征值
        eg:table_ticket["Ticket_Numbers"] = pd.cut(table_ticket["Ticket_Numbers"], bins=[0,1,4,20], labels=[0,1,2])
      (5) 将 table_ticket 合并到 df中
        eg:df = pd.merge(df, table_ticket, left_on="Ticket",right_index=True, how='left', sort=False)
'''

# 1、 妇女/儿童 男士标签
# 妇女/儿童 男士标签
child_age = 18
def get_person(passenger):
    age,sex = passenger
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'

# pd.concat([df, pd.DataFrame(df[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)
# 复制 一份df 的备份 df_new
df_new = df.copy()
# 对  df 的 Age，Sex 两列中的元素进行 按照 child,female_adult 和male_adult 进行分类，并赋值给 df_new['AgeSex']
df_new['person']=df[['Age','Sex']].apply(get_person,axis=1)
# 将 df 和 df_new['person'].to_frame 进行合并   axis=1表示 列column 合并
# 作用：df 多了一个特征值 person
df=pd.concat([df,df_new['person'].to_frame()],axis=1)

# 2、cabin 数量衍生特征
def get_Cabintp(cabine):
    cabine_search = re.findall('\d+', cabine)
    if cabine_search:
        return len(cabine_search)
    return '0'

# 给 Cabin 空值元素一个 空格字符串，防止正则搜索报错
df["Cabin_Sum"] = df["Cabin"].fillna(" ")
df["Cabin_Sum"]=df["Cabin_Sum"].apply(get_Cabintp)

# 3、ticket 的衍生特征
table_ticket = pd.DataFrame(df["Ticket"].value_counts())
# 给 DataFrame 某个特征值 重新命名
table_ticket.rename(columns={'Ticket':'Ticket_Numbers'}, inplace=True)
table_ticket.rename(columns={'Ticket':'Ticket_Numbers'}, inplace=True)

##  Ticket_dead_women
table_ticket['Ticket_dead_women'] = df.Ticket[(df.person == 'female_adult')
                                    & (df.Survived == 0.0)
                                    & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()

table_ticket['Ticket_dead_women'] = table_ticket['Ticket_dead_women'].fillna(0)
table_ticket['Ticket_dead_women'][table_ticket['Ticket_dead_women'] > 0] = 1.0

##  Ticket_dead_women
table_ticket['Ticket_surviving_men'] = df.Ticket[(df.person == 'male_adult')
                                    & (df.Survived == 1.0)
                                    & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()

table_ticket['Ticket_surviving_men'] = table_ticket['Ticket_surviving_men'].fillna(0)
table_ticket['Ticket_surviving_men'][table_ticket['Ticket_surviving_men'] > 0] = 1.0

#  Ticket数量分箱
table_ticket["Ticket_Numbers"] = pd.cut(table_ticket["Ticket_Numbers"], bins=[0,1,4,20], labels=[0,1,2])

# df 和 table_ticket 合并  左连接是 Ticket
df = pd.merge(df, table_ticket, left_on="Ticket",right_index=True, how='left', sort=False)

# 4、 surname衍生特征
df['surname'] = df["Name"].apply(lambda x: x.split(',')[0].lower())
# 构建 surname df
table_surname = pd.DataFrame(df["surname"].value_counts())
table_surname.rename(columns={'surname':'Surname_Numbers'}, inplace=True)

table_surname['Surname_dead_women'] = df.surname[(df.person == 'female_adult')
                                    & (df.Survived == 0.0)
                                    & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()

table_surname['Surname_dead_women'] = table_surname['Surname_dead_women'].fillna(0)
table_surname['Surname_dead_women'][table_surname['Surname_dead_women'] > 0] = 1.0

table_surname['Surname_surviving_men'] = df.surname[(df.person == 'male_adult')
                                    & (df.Survived == 1.0)
                                    & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()

table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)
table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0

# surname数量分箱
table_surname["Surname_Numbers"] = pd.cut(table_surname["Surname_Numbers"], bins=[0,1,4,20], labels=[0,1,2])

df = pd.merge(df, table_surname, left_on="surname",right_index=True,how='left', sort=False)

# 5、Age 缺失值处理

# Age 缺失值处理  用极限决策树处理缺失值
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

classers = ['Fare','Parch','Pclass','SibSp','TitleCat',
            'CabinCat','female','male', 'Embarked', 'FamilySize', 'NameLength']

etr = ExtraTreesRegressor(n_estimators=200,random_state=0)

X_train = df[classers][df['Age'].notnull()]

Y_train = df['Age'][df['Age'].notnull()]
X_test = df[classers][df['Age'].isnull()]

etr.fit(X_train.as_matrix(),np.ravel(Y_train))
age_preds = etr.predict(X_test.as_matrix())
df['Age'][df['Age'].isnull()] = age_preds

# 6、 其他特征值处理
# FamilySize  Surname_Numbers   person  特征值进行处理

# person 进行 hot 编码
df = pd.concat([df,pd.get_dummies(df['person'])],axis=1)
# 将FamilySize、Surname_Numbers分箱后的分类类型 改成 int64类型
df.FamilySize=df.FamilySize.astype('int64')
df.Surname_Numbers=df.Surname_Numbers.astype('int64')

'''
五、 特征值选择

'''


from sklearn.feature_selection import SelectKBest, f_classif,chi2
# 训练集 分类结果标签
target = data_train["Survived"].values


#输入模型的特征值
features= ['Age','Embarked', 'Fare','Parch',
       'Pclass', 'SibSp','CabinCat',
       'TitleCat', 'NameLength', 'female', 'male','Ticket_dead_women',
        'Ticket_surviving_men','Surname_dead_women', 'Surname_surviving_men',
         'child','female_adult','male_adult','Cabin_Sum','FamilySize'
          ]

# 训练集数
train = df[0:891].copy()

test = df[892:].copy()
#   返回k个最佳特征的训练器 Select features according to the k highest scores.
selector = SelectKBest(f_classif,k=len(features))
# 将训练数据 输入模型中
selector.fit(train[features],target)
# 评分
scores = -np.log10(selector.pvalues_)

# 对评分的排序
indices = np.argsort(scores)[::-1]

print("Features importance :")
for f in range(len(scores)):
    print("%0.2f %s" % (scores[indices[f]],features[indices[f]]))

'''
六、 建模预测
  这里采用随机森林建模预测
'''

# cross_validation 用于交叉验证
from sklearn import cross_validation
# cross_val_score
from sklearn.model_selection import cross_val_score
# 随机森林分类器   n_estimators 设置树的数量    min_samples_split最小叶节点停止条件  设置权重class_weight
rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745,1:0.255})

# 交叉验证，建模随机森林
# 提高模型的性能 ，减少过拟合

kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(rfc, train[features], target, cv=kf)

print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std()*100, 'RFC Cross Validation'))

rfc.fit(train[features], target)
score = rfc.score(train[features], target)

# 预测目标值
rfc.fit(train[features], target)

print('RandomForest acc is', np.mean(cross_val_score(rfc, train[features], train['Survived'], cv=10)))
