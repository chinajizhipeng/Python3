%matplotlib inline
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
#sklearn样本生成
sklearn.datasets.make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,  
                    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,  
                    flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0,   
                    shuffle=True, random_state=None)
n_features :特征个数= n_informative（） + n_redundant + n_repeated
n_informative：多信息特征的个数
n_redundant：冗余信息，informative特征的随机线性组合
n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
n_classes：分类类别
n_clusters_per_class ：某一个类别是由几个cluster构成的
#生成高斯分布（正态分布）
sklearn.datasets.make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3,  
                    shuffle=True, random_state=None)
#生成环形数据
sklearn.datasets.make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8)  
#生成半环形
sklearn.datasets.make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)  


#Scikit-Learn
#特征矩阵（解释变量）和目标数组（被解释变量）
#特征矩阵应该为二维，若是一维，增加维度X =x[:,np.newaxis]
#目标数组应为一维函数

#######数据清理
###从CSV变为机器学习可以用的数据
#1、读取CSV变为DF
import pandas as pd
from Ipython.display import display #可以在Jupyter输入漂亮的格式
data = pd.read_csv("",header = None,index_col = False,names = ['var1','var2','var3']) #若没有变量名取None
data = data[['var1','var2']] #选取有用的列
#2、DF转变为NUMPY数组
feature = data.ix[:,'解释变量1':'解释变量n'] #读取解释变量
X_numpy = feature.values
Y_numpy = data['被解释变量'].values
#3.设置训练集和测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_numpy, Y_numpy, random_state=0)
###设置虚拟变量，现在用pd转化呈虚拟变量，在转换为机器学习数据
data_dummy = pd.get_dummies(data) #除了数值变量以外的变量全设置为虚拟变量
data_dummy = pd.get_dummies(data,colums = ['var1']) #将需要的变量转换为虚拟变量

data['num'] = data['num'].astype(str) #若是数值型变量，先转为字符串变量再转化虚拟变量
data_dummy = pd.get_dummies(data,colums = ['num']) #将需要的变量转换为虚拟变量
###分箱，将连续变量离散化
bins = np.linspace(-3,3,11) #设置为11-1个箱子
data_bin = np.digitize(X,bins= bins) #根据箱子X的变量分为第几个箱子里的数
#例子，将age分箱并添加在pd数据中
bins = np.linspace(20,60,6)
data_bins = np.digitize(data['age'].values,bins = bins)
data['age_bin']=data_bins
###线性变换
data.values[:,1] = np.log(data.values[:,1]) #变换第二列
log_data = np.log(data.values) #变换全部
##自动化特征选择
#单变量统计 单独考虑每个特征值，设定阈值（P，或者变量数量）来选择特征
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=50) #选择50%的变量
select.fit(X_train, y_train) #对训练数据进行拟合，选择数据
X_train_selected = select.transform(X_train) #留下选择后的变量
mask = select.get_support() #查看哪些变量被选中
print(mask)
X_test_selected = select.transform(X_test) #对测试数据进行转换
lr = LogisticRegression()
lr.fit(X_train, y_train) #拟合模型
##基于模型的特征选择，同时考虑所有特征的重要性，设定选择特定的变量数量
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), #用100个RF来选择变量
    threshold="median") #选择一半的特征值
select.fit(X_train, y_train) #拟合数据
X_train_l1 = select.transform(X_train) #转换数据
X_test_l1 = select.transform(X_test) #转换测试数据
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test) #拟合数据并得分
##迭代特征选择，所有变量进行建模，舍弃部分变量，再建模，在舍弃，知道留下设定数量的变量
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), #用100个RF
             n_features_to_select=40) #留下40个变量
select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train) #进行数据转换，留下特定变量
X_test_rfe = select.transform(X_test) #对预测变量进行数据转换，拟合模型
##标准化处理数据
#最大最小值
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) #根据X_train拟合标准化模型
X_train_scaled = scaler.transform(X_train) #变换模型
X_test_scaled = scaler.transform(X_test) #测试集 
#标准化
#1
from sklearn import preprocessing #标准化数据模块
X = preprocessing.scale(X) #X为特征矩阵
#2
from sklearn.prepeocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#预测得分
#1
from sklearn.metrics import accuracy_score
y_model = model.predict(Xtest)             # Xtest为训练集自变量
accuracy_score(ytest, y_model) #预测的准确率 ytest为测试集因变量
#2
model.score(X_train, y_train) #训练集的得分
model.score(X_test, y_test)   #验证集的得分






#####最近邻
##适用于小型数据集，容易解释
###K邻近分类器
KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’,
                     leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score #交叉验证模块
from sklearn.cross_validation import LeaveOneOut

X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)
knn = KNeighborsClassifier(n_neighbors=1) #新数据的标签与其最接近的1个训练数据的标签相同
knn.fit(X1, y1) #X1为训练特征矩阵 y1为训练标签数组
y2_pre = knn.predict(X2) #对验证特征矩阵进行验证
y2_pre.score(X2,y2) #测试精度
cross_val_score(knn, X, y, cv=5) #交叉检验 5轮 
scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X))) #LOO交叉检验 只留一个样本检验
scores #查看得分
scores.mean() #查看平均得分

##多次调整参数 看复杂度和精确度
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#####线性模型
#可靠的首选算法 适用于非常大的数据集，也适用于高维数据
#简单线性回归
from sklearn.linear_model import LinearRegression #载入线性回归模型类
model = LinearRegression(fit_intercept=True) #实例化，并设置有截距
lr = model.fit(X_train,Y_train) #注意二维，一维 #X为解释变量，Y为被解释变量
lr.coef_ #斜率
lr.intercept_ #截距
y_predict = model.predict(X_test) #Xfit为预测数据的特征矩阵
lr.score(X_train,Y_train) #训练集的得分
lr.score(X_test,Y_test)   #验证集的得分
#岭回归（应对多重共线性）
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1).fit(X_train, y_train) #调整alpha，增大会提高泛化能力，降低训练集性能
ridge.score(X_train, y_train) #训练集的得分
ridge.score(X_test, y_test)   #验证集的得分
#Lasso回归
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train) #alpha小能降低欠拟合，max_iter为最大迭代次数
lasso.score(X_train, y_train)
lasso.score(X_test, y_test)
#LOGISTIC回归
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1).fit(X_train, y_train) #C越大，越容易过拟合
logreg.score(X_train, y_train)#训练集的得分
logreg.score(X_test, y_test) #验证集的得分
#LinerSVC 多分类的线性模型
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
linear_svm.score(X_train, y_train) #训练集的得分
linear_svm.score(X_test, y_test)    #验证集的得分
#####朴素贝叶斯
#只适用于分类问题，速度比线性模型快，适用于非常大的数据集和高维数据集，精确度低于线性模型
#高斯朴素贝叶斯分类
#GaussianNB 用于任意连续数据
#BernoulliNB  用于二分类数据 BM可加alpha，越大平滑性越强，复杂度越低
#MultinomialNB 用于计数数据
from sklearn.cross_validation import train_test_split #分为训练集和预测集
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, #X为特征矩阵
                                                random_state=1) #Y为目标数组
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model) #预测的准确率 ytest为测试集

#####决策树
#速度快 不需要数据缩放 可以可视化 容易解释
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=0) #max_depth最大层数
tree.fit(X_train, y_train)
tree.score(X_train, y_train) 
tree.score(X_test, y_test)
#可视化 http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
from sklearn.tree import export_graphviz
import graphviz
export_graphviz(tree, out_file="tree.dot")
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))
#树的重要性
tree.feature_importances_

#####随机森林
#比单棵决策树要好，鲁棒性很好，不需要数据缩放，不适用于高维稀疏数据
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2) #5课树
forest.fit(X_train, y_train)
forest.score(X_train, y_train) 
forest.score(X_test, y_test)
#####梯度提升决策树：树，预测，纠正错误，树
#精度比随机森林略高，训练速度比RF慢，预测更快，内存少，比RF调节的参数更多。
from sklearn.ensemble import GradientBoostingClassifier
#learning_rate为学习率降低过拟合，max_depth为最大深度用来加强预剪枝
gbrt = GradientBoostingClassifier(max_depth=1,learning_rate=0.01,random_state=0)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train) 
gbrt.score(X_test, y_test)
#####支持向量机
#对特征含义相似的中等大小的数据集很强大，需要数据缩放，对参数敏感。
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

#####神经网络
#可以构建非常复杂的模型，尤其对大型数据集而言，对数据缩放敏感，对参数调节敏感，大型网络需要很长的训练时间
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)

###主成分分析
#主成分分析前要标准化
from sklearn.prepeocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. 将数据转换为2维
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False) #画图
#主成分作图
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
#主成分的重要性，并作图
pca.components_
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
###聚类分析
##K均值聚类，，簇点是附近数据点的均值，只能用于简单的形状
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) #聚成三类
kmeans.fit(X)
kmeans.labels_ #显示训练X的标签
kmeans.predict(X) #预测新数据集X的标签
##凝聚聚类，每一个数据点是一个簇，然后合并两个最相似的簇，直到满足某种停止准则
#不能对新数据点预测，只能得到训练集的标签
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X) 
##DBSCAN 具有噪声的基于密度的空间聚类应用
#不需要预先设定簇的个数，可以划分具有复杂度形状的簇，还可以找出不属于任何簇的点
#簇形成数据的密集区域，并由相对铰孔的区域分隔开
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10) #eps最小距离，min_samples 数据点
#数据点的距离eps内有min_samples个样本点，这个点就是和新样本，小于eps的放到一个簇中
clusters = dbscan.fit_predict(X) 


#高斯混合模型 GMM 无监督学习
from sklearn.mixture import GMM      # 1. Choose the model class
model = GMM(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels

iris['cluster'] = y_gmm #预测聚类
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False)

###交叉验证，手动分为训练集和测试集，记得给模型传参数
#我们想知道的是模型对于训练过程中没有见过的数据的预测能力
#普通
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target,cv=5) #验证五轮
scores #得出得分
scores.mean() #平均得分
#分层k折交叉验证
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=True, random_state=0) #分成五折,打乱，随机数
cross_val_score(logreg, iris.data, iris.target, cv=kfold)
#留一法交叉验证，只留下单个样本做测试，对于小型数据
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
#打乱划分交叉验证，对于大型数据
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10) #迭代十次
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))
#分组交叉验证
from sklearn.model_selection import GroupKFold
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3)) #4个分组，测试三次
###网格搜索 调整参数提高泛化能力
#网格搜索，自动化工具设定模型。包括多项式的次数，是否截距，是否进行标准化
#1
from sklearn.grid_search import GridSearchCV
param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
grid.best_params_ #得出模型参数
#简单网格搜索
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    random_state=0)
print("Size of training set: {}   size of test set: {}".format(
      X_train.shape[0], X_test.shape[0]))
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = svm.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
#分为训练集、验证集、和测试集
X_trainval, X_test, y_trainval, y_test = train_test_split( #将数据分为训练+验证和测试集
    iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split( #将训练+验证集分为训练集和验证集
    X_trainval, y_trainval, random_state=1)
##带交叉验证的网格搜索
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, #cv是交叉验证的参数
                          return_train_score=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train) #调入训练集搜索最佳参数
grid_search.score(X_test, y_test) #得出最佳得分（基于训练集）
grid_search.best_params_ #得出最佳参数
grid_search.best_score_  #得出最佳得分 ，基于训练集的交叉验证
grid_search.best_estimator_ #最佳参数对应的模型
#分析交叉验证的结果
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
results.head()
#作图
##混淆矩阵
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_y) #y_test为测试集标签，pred_y为预测标签
print("Confusion matrix:\n{}".format(confusion))
#f-分数，是准确率和召回率的调和平均
from sklearn.metrics import f1_score
print("f1 score logistic regression: {:.2f}".format(
    f1_score(y_test, pred_logreg))) 
#得分报告
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#改变分类决策的阈值
#默认情况下decision_function>0 就划为类别1，小于类别0
from mglearn.datasets import make_blobs
X, y = make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2],
                  random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
y_pred_lower_threshold = svc.decision_function(X_test) > -.8 #将决策阈值变为0.8 
print(classification_report(y_test, y_pred_lower_threshold))
#准确率和召回率曲线，曲线越靠近右上角越好
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))
X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2],
                  random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")

from sklearn.metrics import average_precision_score
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("Average precision of svc: {:.3f}".format(ap_svc)) #平均准确率，总结准确率-召回率曲线。
#受试者工作特征（ROC曲线），用的是假正例率（FPR），和真正例率（召回率）
#曲线越靠近左上角越好
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
close_zero = np.argmin(np.abs(thresholds)) # find threshold closest to zero
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
from sklearn.metrics import roc_auc_score #打印AUC得分 用于非平衡分类得分好，比精度好
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("AUC for SVC: {:.3f}".format(svc_auc))
#在模型选择中使用评估指标
roc_auc =  cross_val_score(SVC(), digits.data, digits.target == 9,
                           scoring="roc_auc")
print("AUC scoring: {}".format(roc_auc))

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0)
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc") 
grid.fit(X_train, y_train)
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
###算法链与管道，将模型、预处理、数据划分集合在一起，数据划为训练部分、验证部分、测试部分
#简单管道
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())]) #scaler是MinMaxScaler()的实例，svm是SVC()的实例
pipe.fit(X_train, y_train)
print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))
#在网格搜索中使用管道
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], #“模型__参数”
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5) #交叉验证5次
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))
#访问网格搜索管道中的属性
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5) #网格
grid.fit(X_train, y_train)
print("Best estimator:\n{}".format(grid.best_estimator_)) #访问最佳模型
print("Logistic regression step:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"])) #访问模型步骤
print("Logistic regression coefficients:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"].coef_)) #访问模型的系数



















