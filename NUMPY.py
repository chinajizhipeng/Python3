import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##数据加载、储存与文件格式 《利用PY进行数据分析167》
pd.read_csv()
pd.read_csv('',header=None) #不要文件的列名，PD默认分配
pd.read_csv('',names=['a'.'b']) #
pd.read_csv('',index_col=['key1','ker2']) #层次化索引
pd.read_csv('',skiprows=[0,2,3]) #跳过第一、三、四行
pd.read_csv('',na_values=["NULL"]) #表示缺失值的字符串
pd.read_csv('',nrows=5) #读取5行
pd.read_table()
pd.read_table('',sep=',')
pd.read_fwf()
pd.read_clipboard()
df.to_csv('') #将数据写到一个以逗号分隔的文件中
df.to_csv('',sep='|') #以|分隔
df.to_csv('',na_rep='NULL') #缺失值以NULL表示
df.to_csv('', index=False, header=False) #不显示索引和列标签
df.to_csv('', index=False, columns=['a', 'b', 'c']) #写出一部分列，并指定列标签
#Series读写
ts.to_csv() #输出csv
Series.from_csv()
#读取XLS文件
import xlrd,openpyxl
xlsfile = pd.ExcelFile('')
table = xlsfile.parse('sheet1') #读取表1
#固定类型数组
array.('i',ob) #i为整数型
np.array(ob,dtype = 'float32') #浮点数类型 Numpy数据类型《利用PY进行数据分析》86
np.zeros(10) #创建长度为10的数组 ，全为0
np.ones(10) #全为1
np.full((3,5),3.14) #3*5矩阵 全为3.14
np.arange(0,20,2) #0,2,4...
np.linspace(0,1,5) #5个数均匀的分布在0-1
np.random.random(size=(3,3)) #0-1均匀分布的3*3矩阵
np.random.normal(0, 1, (3,3)) #正太分布0-1
np.random.randint(0, 10, (3,3)) #随机0-10
np.eye() #单位矩阵
np.column_stack((a,b)) #列组合
b.ndim  #给出数组的维数
d.size  #给出数组元素的个数
d.itemsize #给出数组中的元素在内存中所占的字节数
d.nbytes #真哥数组所占的存储空间
x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
display()
print("x3 ndim: ", x3.ndim)  #数组的维度
print("x3 shape:", x3.shape) #每个维度的大小
print("x3 size: ", x3.size)  #数组的总大小
print("dtype:", x3.dtype) #数组的数据类型
print("itemsize:", x3.itemsize, "bytes") #每个数组元素的大小
print("nbytes:", x3.nbytes, "bytes")     #数组总字节的大小
ob[x,y,z] #多维数组总用逗号分隔的索引数组获取元素
b.ravel() #将多维度数组展平
b.flatten() #展平
b.shape = (3,2)
b.reshape(2,3)
b.resize(2,3) #直接改变
b.transpose() #转置
#切片 数组切片返回的是数组数据的试图，不是数值数据的副本
ob[x:y:z]
ob[::2] #每隔两个取一次
x2[:2, :3]  # two rows, three columns #多维
x2[:3, ::2]  # all rows, every other column
x2[::-1, ::-1]
x2_copy=x[:2,:2].copy() #创建副本 不修改值
grid = np.arange(1, 10).reshape((3, 3)) #数组变形，返回非副本视图
x[np.newaxis, :] #获取行向量
x[:, np.newaxis] #获取列向量
np.concatenate([x, y]) #一维 轴向合并
np.concatenate([grid, grid], axis=1) #二维，第一个轴0 第二个轴1
np.vstack(array) #垂直化
np.vstack([x, grid]) #垂直栈数组链接，变成一维
np.hstack([grid, y]) #水平栈数组链接，变成一维
np.dstack() #按照第三个维度拼接
ob.repeat(2)
ob.repeat([2,3,4])
ob.repeat(2,axis=0) #
np.msort(name) #np排序
np.where(logreturn>0) #返回大于零的索引
#数组的分裂
np.split(x, [3, 5]) #0-2行，3-4行，5： 行切分
left, right = np.hsplit(grid, [2]) #从第2列裂开
left, right = np.vsplit(grid, [2]) #从第2行裂开
np.hsplit(a,3) #沿着水平方向分割为3个相同大小的子数组
np.vsplit(a,3) #沿着垂直方向分割为3个相同大小的子数组

#通用函数
+	np.add	        Addition (e.g., 1 + 1 = 2)
-	np.subtract	Subtraction (e.g., 3 - 2 = 1)
-	np.negative	Unary negation (e.g., -2)
*	np.multiply	Multiplication (e.g., 2 * 3 = 6)
/	np.divide	Division (e.g., 3 / 2 = 1.5)
//	np.floor_divide	Floor division (e.g., 3 // 2 = 1)
**	np.power	Exponentiation (e.g., 2 ** 3 = 8)
%	np.mod	        Modulus/remainder (e.g., 9 % 4 = 1)
n1 + 1
np.add(nq,1)
a.absolute #绝对值
np.log1
np.exp1
np.power(2,x,out=y) #将结果储存在y中 #xy长度一样
np.power(2,x,out=y[::2]) #每隔2元素储存一次
#聚合
np.add.reduce(x)            #累计相加
np.add.reduce(q1,axis=1)    
np.multiply.reduce(x)       #累计相乘
np.add.accumulate(x)        #累计相加，储存每次计算的中间结果
np.multiply.accumulate(x)   #累计相乘，储存每次计算的中间结果
np.multiply.outer(x,y) #获得两个数组所有元素对的函数运算结果
#NP中可用的聚合函数，多维聚合中添加维度，ig:M.min(axis=0) #每列的最小值
Name	        NaN-safe Version	Description
np.sum	        np.nansum	        Compute sum of elements
np.prod	        np.nanprod	        Compute product of elements
np.mean	        np.nanmean	        Compute mean of elements
np.std	        np.nanstd	        Compute standard deviation
np.var	        np.nanvar	        Compute variance
np.min	        np.nanmin	        Find minimum value
np.max	        np.nanmax	        Find maximum value
np.argmin	np.nanargmin	        Find index of minimum value
np.argmax	np.nanargmax	        Find index of maximum value
np.median	np.nanmedian	        Compute median of elements
np.percentile	np.nanpercentile	Compute rank-based statistics of elements
np.any	        N/A	                Evaluate whether any elements are true
np.all	        N/A	                Evaluate whether all elements are true
np.percentile(heights, 25) #25百分位
#布尔掩码
#比较操作的通用函数
==	np.equal		!=	np.not_equal
<	np.less		        <=	np.less_equal
>	np.greater		>=	np.greater_equal
x < 3 #返回布尔数组
np.count_nonzero(x < 6) #返回个数
np.count_nonzero(q1 > 6,axis = 0) #列方向
np.sum(x<6,axis=1) #返回每一列小于6的个数，x为布尔数组
np.any()
np.all()
np.sum(x>1) & (x<0)
&	np.bitwise_and		|	np.bitwise_or
^	np.bitwise_xor		~	np.bitwise_not
x[x>5] #返回子数据集，即掩码操作
#排序
ob.sort() #直接对原来的对象进行操作，原来的数据进行操作
np.sort(ob) #不对原来的数据进行修改
np.sort(ob,axis=0) #二维按列排序 
np.sort(ob,axis=1) #二维按行排序
np.argsort() #返回的是原始数组排好序的索引值

#部分排序，分隔
np.partition(x,3) #最左侧返回第3小的值，剩下的随机  为负时想要最大的k个数
np.partition(x,2,axis=1) #二维最左侧返回前2小的值，剩下的随机
#np读取csv
import pandas as pd
data = pd.read_csv('')
np1 = np.array(data['asd']) #读取data中的asd变量
###Pandas
import numpy as np
import pandas as pd
#numpy

#Series
ob = pd.Series(data,index=index) #生成Series对象,data是列表或者NP数组
ob.values #返回的结果是Numpy数组
ob.index #读取索引标签
ob.index.is_unique #索引值是否唯一
ob=pd.Series([1,1,1],index=['a','b','c']) #可自定义索引值
pd.Series(dict) #将数组转化为Series对象
data['b']
data['e'] = 1.25
data['a':'c'] #slicing by explicit index，包含最后一个索引
data[0:2]     # slicing by implicit integer index 不包含最后一个索引
data[(data > 0.3) & (data < 0.8)] #掩码
data[['a', 'e']] #花式切片
data.loc[1:3] #显示索引 ，给索引值 闭区间
data.iloc[1:3] #隐示索引 #推荐 半开区间
'b' in obj2 #索引是否再
zd ={}
ob = Series(zd) #通过字典创建Series
ob1+b2 #在算术运算中自动对齐索引
ob.name = 'pop' #Series本身命名
ob.index.name = 'stata' #为索引命名 和Pandas关系密切
ob.value_counts() #返回Series的唯一值
ob[ob.isin(['c','a'])] #ob中返回c a 的值
##DataFrame
#读取json文件
DF = DataFrame(JSONNAME[DATA],columbs=['name','age'])
df.astype(int) #转换为整数类型
pd.DataFrame('colname1':Series1,'colname2':Series2)
pd.DataFrame({'A':[1,2,3],'B':['a','b','c']})
df.index #获取索引标签
df.columns #存放列标签
pd.DataFrame(data, columns=['year', 'state', 'pop']) #data为数组
#直接使用中括号时，索引表示的是列索引，切片表示的是行切片
df['area'] #检索列，是Series 中括号或者属性是用来访问列
df.area #使用属性形式选择纯字符串列名
df['density'] = data['pop'] / data['area'] #根据列增加列
df.columns = ['state', 'year','ok'] #修改列名
val=Series([],index=[])
df['A']= val #根据索引为df增加一列
df.drop(columns='A') #删除A列
frame2['A'] = frame2.B == 'b' #若B值为b，则为True
df.values[0] #返回第一行
#标签的切片运算是闭区间的,对于切片而言没有单独列切片
df.iloc[:,:2] #列切片
df.iloc[:3, :2] #前两个标签列
df.iloc[:,:2] 
df.iloc[1:3] #第二第三行，左开右闭
df.loc[:'Illinois', :'pop'] #逗号前是索引，逗号以后是标签列
df.loc[['a','b']] #检索行
df['A']['a'] #元素检索，先列后行
df.loc['c']['A'] #元素检索，先行后列 检索行的时候，参数可以是多个，但是列不行
df.loc['c','A'] #元素检索，先行后列
df.loc[data.density > 100, ['pop', 'density']] #掩码
df.iloc[0,2] = 90
df[df.density > 100]
df[df['A'] > 5]
df1['q'].loc['a']=99 #q列a行
d = data[['year','midu']].copy() #选择标签列
data.ix('a',['A','B']) #索引a，AB列标签的值
data.loc[data.index=='Florida', ['area', 'pop']]#选择标签为,列为
ob2 = ob.reindex([index]) #根据index重新排序
ob2 = ob.reindex([index],fill_value=0) #根据index重新排序,没有的索引赋值为0
ob2 = ob.reindex([index],method='ffill') #根据index重新排序,没有索引的值前向填充
ob.drop('c') #删除ob索引c项
ob.drop(['c','d']) #删除索引cd项
ob.drop('c',axis=1) #删除ob的c列
ob.drop(['c','d'],axis=1) #删除ob的cd项
#Pandas数值运算方法
#axis=0,以列为单位操作，参数必须是列
#axis=1,以行为单位操作，参数必须是行
df+series #使用列索引进行相加
df1.add(s1,axis = 0) #想要一列相加时
df1.loc['a'] += 100 #a行加100
A + B #对齐两个对象的索引进行运算，取并集，缺失值存在用NAN填充
A.add(B, fill_value=0)#自定义A或者B的缺失值

fill = A.stack().mean() #先将A的二维数据降维一维
A.add(B, fill_value=fill) #用A的均值填充缺失值

# PY运算符与Pandas方法的映射关系
Python Operator	Pandas Method(s)
+	        add()
-	        sub(), subtract()
*	        mul(), multiply()
/	        truediv(), div(), divide()
//	        floordiv()
%	        mod()
**	        pow()
df.subtract(df['R'], axis=0) #按列计算
df - df.iloc[0] #按行计算，用索引
df1.add(df2,fill_value=0) #缺失值处理
#缺失值处理
np.nan None #创建缺失值
isnull()
notnull()
dropna() #删除含有任何缺失值的整行
df.dropna(axis='columns')  #删除含有任何缺失值的整列
df.dropna(axis='columns', how='all') #删除含有缺失值的整行和整列
df.dropna(axis='rows', thresh=3) #规定了非缺失值的最小数量
fillna()
data.fillna(0)
data.fillna({'a':0,'b':1}) #a列填充o b列填充1
data.fillna(method='ffill') #向后填充
data.fillna(method='bfill') #向前填充
df.fillna(method='ffill', axis=1) #按行向后填充
##多级索引
Series['A', 'B'] #第一个参数，多层索引的第一维，第二个参数，第二维
#1 index参数至少二维
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df
#2 将元组作为键的字典传递给Pandas
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)
#3 显示构造,然后使用将这些对象作为index参数，或者通过reindex方法更新索引
df = pd.DataFrame(np.random.rand(4, 2),
                  columns=['data1', 'data2'],
                  index=)
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]]) #数组
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]) #元组
pd.MultiIndex.from_product([['a', 'b'], [1, 2]]) #笛卡尔积

pop.index.names = ['state', 'year'] #多级列索引
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
health_data = pd.DataFrame(data, index=index, columns=columns)
#多级索引的累计方法
pd.mean(level='year') #对行索引进行累计操作
pd.mean(axis = 1,level='name') #对列索引进行累计操作
health_data
#df多维索引，使用ix(),loc()函数，对于二维索引，如果包含中文，切片可能有BUG
df.loc['a',1] #二级索引
df.loc['a'].loc['1'] #二级索引
df.loc['a'] 
df.loc['a':'c'] #一级多个索引
df.loc[['a','b']] #一级多个索引
df[:, 2000]
df[df > 22000000]
df['a'] #第一层次的索引
df['a':'c']
df.ix[['a':'c']]
df[:,2] #第一层次的全部，第二层次的2
df.loc[:,'data1':'data2'] #多个列索引
df.index.names = ['key1', 'key2'] #命名索引
df.columns.names = ['state', 'color'] #命名列名
df['key1'] #进行列索引

df.swaplevel('key1', 'key2') #互换索引的级别
df.sortlevel(1) #对单个级别的值对数据进行排序

df.sum(level='key1') #根据key1对数据进行求和
frame.sum(level='color', axis=1) #根据color对行求和

frame2 = frame.set_index(['c', 'd']) #将DF的列变为索引
frame.set_index(['c', 'd'], drop=False) #保留列
frame2.reset_index() #将索引列变为列
#索引改名
i = {0:6}
df.rename(index=i) #将索引0改为6

i = {'A':'VV'}
df.rename(columns=i) #将列索引0改为6

df.index = df.index.map(str.uppr) #将索引变为大写
index
df.rename(index=str.title, columns=str.upper)

df.rename(index={'OHIO': 'INDIANA'}, #索引改名
            columns={'three': 'peekaboo'}) #列标签改名

_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True) #就地修改
#根据A列的值进行排序
df.sort_values(by='B')
#对索引进行排序
data = data.sort_index() #a-z 0-...
data = data.sort_index(axis=1) #对列名排序
data.sort_index(axis=1, ascending=False) #降序
data = data.sort_index(by='b') #根据列标签b进行排序
data = data.sort_index(by=['a','b']) #根据列标签b进行排序 
ob.rank(method = 'first') #增设一个排名值，1，2，3，4，5，6
ob.rank() #1，2，3，4.5，4.5，6
ob.rank(ascending=False, method='max') #降序
ob.sort_index(axis=1) #一行排序
#索引的堆 
#使用stack的时候，level等于哪一个，哪一个就消失，出现在行里，默认level= -1，将列变成行
#使用unstack的时候，level等于哪一个，哪一个就消失，出现在列里，将行变成列
#对多层索引的列而言0，1，2，从上往下计数
df.unstack(level=0)
df.unstack(level=1)
df.unstack().stack()
data_mean = health_data.mean(level='year') #根据year求均值
data_mean.mean(axis=1, level='type') #继续根据tpye求均值
#合并，级联
df1.append(df2) #直接追加数据，更加方便 
pd.concat([A,B]) #行合并
pd.concat([A,B],axis='col') #列合并
pd.concat([A,B]，verify_integrith=True) #有重复索引时返回错误
pd.concat([A,B]，ignore_index=True) #忽略索引创建一个新的整数索引
pd.concat([A,B]，keys=['x','y']) #增加多级索引，可以使合并数据更加清晰
pd.concat([A,B]，join='inner') #实现对列名的交集合并
pd.concat([A,B]，join='outer') #实现对列的并集合并，所有缺失值都用NaN填充
pd.concat([df5, df6], join_axes=[df5.columns]) #以某一个df的列索引为新的列索引
pd.merge(df3, df4) #一对一 多对一 多对多
pd.merge(df1, df2, on='employee') #keimport array
pd.merge(df1, df2, on=['key1'.'key2']) #根据两个键合并
pd.merge(df1, df3, left_on="employee", right_on="name") #The left_on and right_on keywords
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1) #删除重复KEYWORD
pd.merge(df1a, df2a, left_index=True, right_index=True) #用索引合并
df1a.join(df2a) #使用索引合并
df1a.join(df2a,on='key') #左用索引，右用key
pd.merge(df1a, df3, left_index=True, right_on='name') #左用索引和右用Keyword进行合并
pd.merge(df6, df7, how='inner') #结果只包含两个输入合集的交集
pd.merge(df6, df7, how='outer') 
pd.merge(df6, df7, how='left') /'right'
pd.merge(df8, df9, on="name") #重复列名合并
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]) #自定义重复列名
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True) #层次化索引合并 
np.where(pd.isnull(a), b, a) #若a为NULL，输入b，否则是a
df1.combine_first(df2) #df1中若有NAN，用df2中的值补充
df.duplicated() #查看各行是否有重复值 列名别重复
df.drop_duplicates() #删除重复行
df.drop_duplicates(['k1']) #删除指定重复行，第一个
df.drop_duplicates(['k1', 'k2'], take_last=True) #删除制定行的重复值，并保留最后一个
df.drop('name',1) #删除name列
df.drop([1,3]) #删除1和3的索引行
df.replace(-999,np.nan) #将-999替代为缺失值
df.replace([-999, -1000], [np.nan, 0])
df.replace({-999: np.nan, -1000: 0})
df.var1.value_counts() #返回df中var1的唯一值及数量
pd.get_dummies(data) #将data的变量转换为虚拟变量，将所有数字看作连续的变量，不会为其创建虚拟变量
#map()函数，新建一列,由已有的列生成一个新列，适合处理单独的列
#map一个一个映射 map中不能使用sum之类的函数
df['A']=df['A'].map(lambda x:x+10) #A列+10

def squart(item):
    return item**2
df['A']=df['A'].map(squart)

df['A']=df['B'].map(lambda x:x+10) #新建一列
#tranform该方法根据某种规则进行运算
df['A']=df['A'].transform(squart)
#take函数 排序
df.take(per) #以per的顺序进行排序

##累计和分组
df.head() #和R一样
df.tail()
f = lambda x: x.max() - x.min() #自定义函数
df.apply(f,axis=1) #按行运行

Aggregation	    Description
count()	            Total number of items
first(), last()	    First and last item
mean(), median()    Mean and median
min(), max()	    Minimum and maximum
std(), var()	    Standard deviation and variance
mad()	            Mean absolute deviation
prod()	            Product of all items
sum()	            Sum of all items
df.idxmax() #返回最大值的索引
df.idxmin()
df.cumsum() #累计求和
df.mean(skipna=False) #有na时返回na值
df.mean(axis='columns') #对每一行求均值
df.dropna().describe()  #统计性表述,可用于非数值型数据
#GROUPBY
#分组执行后将SE转变为DF，然后merge
df.groupby('key') #返回DATAFRAMEGROUPBY对象
df.groupby('key').sum() #进行分组求和计算
df.groupby('key1')['key2'].fun #根据key1分组计算key2的值
group1 = df['A'].groupby(df['key1']) #利用key1对A进行分组变成一个GROUP对象
group1.fun()
df.groupby(len).sum() #根据函数分组，根据索引的长度进行分组，求总
df.groupby(level="A",axis=1).count() #根据索引的级别进行分组
df.groupby('key1')['key2'].median() #根据1分组计算2的均值 多层索引
titanic.groupby(['1', '2'])['3'].aggregate('mean').unstack() #按1和2分组计算3的均值
planets.groupby('method')['year'].describe().unstack() #按组的统计性表述
for (method, group) in planets.groupby('method'): #按组迭代
    print("{0:30s} shape={1}".format(method, group.shape))
planets.groupby('method')['year'].describe().unstack() #按组调用
df.groupby('key').aggregate(['min', np.median, max]) #按组返回3个函数的值
df.groupby('key').filter(filter_func) #过滤 要定义过滤函数

for name, group in df.groupby('key1'):#分组名和数据块
    print(name)
for (k1, k2), group in df.groupby(['key1', 'key2']): #由键值组成的元组
    print((k1, k2))

pieces = dict(list(df.groupby('key1')))
pieces['b'] #只有type

L = [0,2,1,2,3,4,4,5,6,1]
a1.groupby(L).sum() #根据L进行分组，对每一列进行分组求和

mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
df2.groupby(mapping).sum() #ABC为索引 根据字典的值分组求和


#apply和transform都可以使用，但是transform传输到所有的索引，apply值保留分组索引
def fun(A):
a = lambda x :x...
df.groupby('key1').apply(A) #根据key1分组，分别应用A函数，有分组
df.groupby('key1').transform(A) #求和之后对所有数据进行一一对应 注意与apply的区别
df.groupby('key1', group_keys=False).apply(A) #禁止KEY1键的层次化索引



#将长数据变为宽数据
wide_data = long_data.pivot('A','B','C') #A是索引，B变宽的因子，C是值
wide_data = long_data.pivot('A','B'） #C1和C1自动变宽
#将宽数据变为长数据
mydata1=mydata.melt(
id_vars=["Name","Conpany"], #要保留的主字段
var_name="Year",#拉长的分类变量
value_name="Sale"#拉长的度量值名称
#面元划分
B = [18, 25, 35, 60, 100]
cats = pd.cut(A, B) #划分
cats.labels #标签化 1 0 3 2 1 2
cats = pd.cut(A, 4,precision=2) #均匀分布4个区间，2个小数点
cats = pd.qcut(data, 4) # 四分位划分
pd.cut(A, B, right=False) #闭区间

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)  #为面元命名


#数据透视表
df.pivot_table('A', index='B', columns='C') #A为值 B为列分组 C为行分组 默认计算均值

age = pd.cut(df['C'], [0, 18, 80]) #将C分段
titanic.pivot_table('A', ['B', C], 'D') 

fare = pd.qcut(titanic['fare'], 2) #将fare等分2份
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

DataFrame.pivot_table(data, values=None, index=None, columns=None,
                      aggfunc='mean', fill_value=None, margins=False,
                      dropna=True, margins_name='All') 
aggfunc='mean'、'sum','count','min','max'

titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'}) #用字典为不同的列指定不同的函数

titanic.pivot_table('survived', index='sex', columns='class', margins=True) #计算每一组的总数
#交叉表
pd.crosstab(df.A,df.B,margins=True) #根据A和B进行统计频数

#向量化字符串操作
'word' in val #字串定位
df.str().+
len()	    lower()	    translate()	    islower()
ljust()	    upper()	    startswith()    isupper()
rjust()	    find()	    endswith()	    isnumeric()
center()    rfind()	    isalnum()	    isdecimal()
zfill()	    index()	    isalpha()	    split()
strip()	    rindex()	    isdigit()	    rsplit()
rstrip()    capitalize()    isspace()	    partition()
lstrip()    swapcase()      istitle()	    rpartition()
val.index(',') #找不到时返回错误
val.find(':') #找不到时返回-1
Se.str.contain('X') #SERIES对象中是否存在X
#使用正则表达式的方法
match()	        Call re.match() on each element, returning a boolean.
extract()	Call re.match() on each element, returning matched groups as strings.
findall()	Call re.findall() on each element
replace()	Replace occurrences of pattern with some other string
contains()	Call re.search() on each element, returning a boolean
count()	        Count occurrences of pattern
split()     	Equivalent to str.split(), but accepts regexps
rsplit()	Equivalent to str.rsplit(), but accepts regexps
val.count(',') #返回子串出现的次数
val.replace('A','B') #用B替代A
import re #re模块负责对字符串应用正则表达式
re.split('\s+',text) #拆分字符串，不管分隔符为数量不一定的空白符

regex = re.compile('\s+')
regex.findall(text) #找出所有的分隔模式
#其他字符串方法
get()	        Index each element
slice()	        Slice each element
slice_replace()	Replace slice in each element with passed value
cat()	        Concatenate strings
repeat()	Repeat values
normalize()	Return Unicode form of string
pad()	        Add whitespace to left, right, or both sides of strings
wrap()	        Split long strings into lines with length less than a given width
join()	        Join strings in each element of the Series with passed separator
'::'.join(pieces) #'a::b::c'
get_dummies()	extract dummy variables as a dataframe
full_monte['info'].str.get_dummies('|') #将info按照|分开，并生成虚拟变量
#面板数据
df.ix[A,B,C] #插入条件进行索引，:,"a","5/22/2012"

##时间序列
data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
from datetime import detetime 
from datetime import timedelta
now = datetime.now()
delta = now + timedelta(12)
delta = datetime(2011,1,7)-datetime(2008,6,4)
delta.strftime('') #定义时间格式 格式定义代码《利用PY进行数据分析》305
pd.to_datetime() #解析多重不同的日期形式
from dateutil.parser import parse
parse('2011-01-03') #进行日期解析
date = parser.parse("4th of July, 2015") #对字符串格式的日期进行解析
parse('6/12/2011', dayfirst=True) #日出现在月前面时
#datetime dateutil
from datetime import datetime
datetime(year=2015, month=7, day=4)

#索引、选取和子集
ts['2001'] #ts的索引为时间，选取2001年的数据
ts['2001-05'] #选取2001年5月的数据
ts[datetime(2011,1,7):] #选取2011年1月7号以后的
ts['1/6/2011':'1/11/2011']
ts.truncate(after='1/9/2011')
long_df.ix['5-2001'] #2001年5月以后
#重复索引
ts.index.is_unique
grouped = dup_ts.groupby(level=0) #level=0是固定的
grouped.sum() #求和
grouped.count() #重复出现的次数
#日期的范围、频率及移动 基础频率 《利用PY进行数据分析》315
ts.resample('D') #频率为每日，没有的按照缺失值计算
ts.resample('1M').max() #获取每分钟的最大值
rng = pd.period_range('1/1/2000', '6/30/2000', freq='M') #按月生成
rng = pd.date_range('1/1/2012', periods=100, freq='S') #从2012年1月1日起生成100个样本 时间跨度为1s
index = pd.date_range('4/1/2012', '6/1/2012') #生成日期的范围
pd.date_range(start='4/1/2012', periods=20) 
pd.date_range(end='6/1/2012', periods=20)
pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True) #规范化
pd.date_range('1/1/2000', '1/3/2000 23:59', freq='8h') #在这两个时间中每8个小时生成一个索引 30min

import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64) #NP的datetime64类型
date + np.arange(12) #快速向量化运算
#eval
pd.eval('A'+'B') #使用算术运算符
result2 = pd.eval('df1 < df2 <= df3 != df4') #比较运算符
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)') #支持&|
result2 = pd.eval('df2.T[0] + df3.iloc[1]') #支持索引
df1 = df.eval('(A+B)/(C-1)') #列间运算
df.eval('D=(A+B)/c',inplace = True) #增加一列
#query
result2 = df.query('A < 0.5 and B < 0.5') #过滤


a = {}
for i,j in ob.to_dict(orient='dict')['d'].items():
    a[i]=str(j)[0:2]

def s(item):
    a =str(item)
    return a[:2]
df['A']=df['A'].map(s) #取A列的每行前两位

def mk(cols,inds):
    data = {c:[c+str(i) for i in inds] for c in cols}
    return pd.DataFrame(data,index = inds,columns = cols)



##数据存储与提取
import pickle
a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
#数据提取
with open('pickle_example.pickle', 'rb') as file:
    a_dict1 =pickle.load(file)
print(a_dict1)
