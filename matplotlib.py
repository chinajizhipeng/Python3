import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
#可以输出中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axe.unicode_minus'] = False
plt.show() #显示图片

##数据可视化
############################
#折线图
import matplotlib.pyplot as plt

input_values = [1, 2, 3, 4, 5]
squares = [1, 4, 9, 16, 25]
plt.plot(input_values, squares, linewidth=5) #设置线条粗细

# Set chart title and label axes.
plt.title("Square Numbers", fontsize=24) #标题，fontsize设置文字大小
plt.xlabel("Value", fontsize=14) #x轴
plt.ylabel("Square of Value", fontsize=14) #y轴标签

# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14) #刻度标记的大小

plt.show()
############################
#散点图
import matplotlib.pyplot as plt

x_values = list(range(1, 1001))
y_values = [x**2 for x in x_values]
# edgecolor='none'为删除数据点的轮廓 c=(0, 0, 0.8)为设置颜色
plt.scatter(x_values, y_values, c=(0, 0, 0.8), edgecolor='none', s=40)

# Set chart title, and label axes.
plt.title("Square Numbers", fontsize=24)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square of Value", fontsize=14)

# Set size of tick labels.
plt.tick_params(axis='both', which='major', labelsize=14)

# Set the range for each axis.
plt.axis([0, 1100, 0, 1100000]) #前两个值设置x轴的最小值和最大值，后面是y轴

plt.show()
####保存文件
plt.savefig('name.png',bbox_inches='tight') #第二个参数将周围多余的空白区域裁剪掉
##线形图
%matplotlib inline
series.plot()

df.plot() #注意参数，不识别中文。

##柱状图
df.plot(kind='bar')  #垂直方向
df.plot(kind='barh') #水平方向

##直方图
Series.hist() #注意参数
Series.hist(bins=100) #100个间隔
Series.plot(kind = 'kde') #曲线图

s1 = np.random.normal(0,2,100)
s2 = np.random.normal(0,2,100)
nd = np.concatenate([s1,s2])
s = Series(nd)
s.hist(bins = 100,normed = True)
s.plot(kind = 'kde')            #画在一张图上

##散点图
df.plot('X','Y',kind = 'scatter') #输入列索引
pd.plotting.scatter_matrix(nd,diagonal='kde') 
#直方图
plt.hist(data)
#2维密度图
plt.hist2d(x1,x2,bins = 30,cmap = 'Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
### MATPLOTLIB
#散点图
plt.scatter(x,y,marker = 'o') #可以设置更多属性
#单曲线
plt.plot(x)  #横坐标自动从0开始
plt.plot(x,x**2)  
#1多个曲线
plt.plot(x,x**2)
plt.plot(x,x/2)
plt.plot(x,x**3) #一起输出
#2多个曲线
plt.plot(x,x**2,x,5*x,x,x/3)
#设置网格线
x = np.arange(-np.pi,np.pi,0.01)
plt.plot(x,np.sin(x),x,np.cos(x))
plt.grid(True)

#plt面向对象的方法
#创建，图形就是所谓的对象
plt.subplot()#注意设定参数
a1 = plt.subplot(1,3,1) #一行三列第1个子视图
a1.p-lot(x,np.sin(x))

a2 = plt.subplot(1,3,2) #一行三列第2个子视图
a2.plot(x,np.cos(x))

a3 = plt.subplot(1,3,3) #一行三列第3个子视图
a3.plot(x,np.arccos(x))


##坐标轴的界限
x = np.random.randn(10)
plt.axis([-5,15,-5,15]) #xmin.xmax,ymin,ymax
plt.plot(x)

x = np.random.randn(10)
plt.axis('off') #坐标轴关闭，on打开，tight，equal,注意看解释
plt.plot(x)

y = np.arange(0,10,1) #用xlim和ylim
plt.plot(y)
plt.xlim(-2,12)
plt.ylim(2,12)

##坐标轴标签 注意看参数
x = np.arange(0,10,2)
y = x**2+5
plt.plot(x,y)
plt.ylabel('f(x) = x**2+5')
plt.xlabel('s',size = 20)

##标题 注意看参数
x = np.arange(0,10,2)
plt.plot(x)
plt.title('JIZHIPENG',fontsize = 20)    

##图例
#1
x = np.arange(0,10,1)
plt.plot(x,x,x,x*2,x,x/3)
plt.legend(['A','B','C'])  #参数传递需要中括号
#2
plt.plot(x,x/2,label = 'Normal')
plt.plot(x,x/3,label = 'OK') #label = '_OK' #加下划线不展示 
plt.plot(x,x,label = 'KK')
plt.legend()

#loc参数，设定图例位置
x = np.arange(0,10,1)
plt.plot(x,x,x,x*2,x,x/3)
plt.legend(['A','B','C'],loc = 0 ) #看参数解释！！ 
plt.legend(['A','B','C'],loc = (1.2,0))

#ncol参数，设定图例有几列

##linestyle,color,marker 线条、颜色、
plt.plot(x2.cumsum(),c ='black',linestyle = '--',marker = 'o')

##保存图片
plt.savefig('name.jpg',dpi = 500)  #注意看参数，dpi是像素


##设置风格
#颜色
c
#透明度
alpha
#背景
#线型
ls
lw #线宽
dashes #破折号各段的宽度
marker #点型
makersize #点型大小
#多参数连用
'r--o'
#多条曲线同一设置
plt.plot(x,x**2,x,5*x,x,x/3,ls = '--',lw = 3,c = 'r') #注意声明属性名称
plt.plot(x,x**2,'r--o',x,5*x,'r--o',x,x/3,'r--o')

#设置X轴
plt.xticks(np.linspace(0,10,5),list('ABCDE'),fortsize = 5,rotation = 90) #调整大小

##面向对象的方法
a = plt.subplot(111)
a.plot(x.cumsum())
#设置刻度值
a.set_xticks([0,25,50,75,100]) #先设置刻度值，再设置标签
#设置标签
a.set_xticklabels(list('ABCDE'))



















