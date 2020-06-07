[TOC]

# 0. 数据集

样本数量 39901

特征：['景点','昵称','等级','时间', '评论', '评分']

使用 '评论', '评分' 分别作为 X 和 Y，其他特征去除

# 1. 情感划分

由于评分 0～3 基本为负面评价，而4、5基本为正面评价，因此按此划分负向情感（0）和正向情感（1）

# 2. 分词

采用结巴分词

# 3. 停用词

对于 BOW 和 tf-idf，利用停用词表去除停用词，对于 word2vec 模型，没有去除停用词

# 4. 向量化

## 4.1 BOW 和 tf-idf

由于这两类向量化后特征过多（转换后将近4万维），因此尝试采用 LSA 潜在语义分析（TruncatedSVD）进行降维

### 4.1.1. tf-idf 实验

使用朴素贝叶斯模型进行二分类

降维前：

```
X_train shape: (31920, 39716)
X_test shape: (7981, 39716)
Y_train shape: (31920,)
Y_test shape: (7981,)
```

![tf-idf-朴素贝叶斯-降维前](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/tf-idf-朴素贝叶斯-降维前.png)

使用 LSA 降维至 200 时，解释方差占比 22%，模型拟合结果如下：

![tf-idf-朴素贝叶斯-降维后](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/tf-idf-朴素贝叶斯-降维后.png)

降维到 3000 时，解释方差占比 77%，但结果与上图没有太大变化

尝试不降维，直接将数据用 cat-boost 模型拟合：

![tf-idf_cat-boost](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/tf-idf_cat-boost.png)

### 4.1.2 BOW 实验

使用朴素贝叶斯模型进行二分类

降维前：

```
X_train shape: (31920, 39716)
X_test shape: (7981, 39716)
Y_train shape: (31920,)
Y_test shape: (7981,)
```



![bow-朴素贝叶斯-降维前](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/bow-朴素贝叶斯-降维前.png)

使用 LSA 降维至 200 时，解释方差占比 38%，模型拟合结果如下：

![bow-朴素贝叶斯-降维后](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/bow-朴素贝叶斯-降维后.png)

可以看到，虽然一定程度上解决了维度爆炸的问题，但是模型精度下降严重

利用降维后的数据，采用十折交叉验证，比较不同模型上的效果如下：

![bow-模型比较](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/bow-模型比较.png)

可以看见，树模型效果较好

## 4.2 word2vec

除词向量大小和训练算法，其他超参数使用默认值

    model = gensim.models.Word2Vec(
        sentences,           # 语料
        size=size,           # 词向量大小
        sg=sg,               # 模型的训练算法: 1: skip-gram; 0: CBOW
        window=5,            # 句子中当前单词和被预测单词的最大距离
        hs=0,                # 1: 采用hierarchical softmax训练模型; 0: 使用负采样
        negative=5,          # 使用负采样，设置多个负采样(通常在5-20之间)
        ns_exponent=0.75,    # 负采样分布指数。1.0样本值与频率成正比，0.0样本所有单词均等，负值更多地采样低频词。
        min_count=5,         # 忽略词频小于此值的单词
        alpha=0.025,         # 初始学习率
        min_alpha=0.0001,    # 随着训练的进行，学习率线性下降到min_alpha
        sample=0.001,        # 高频词随机下采样的配置阈值
        cbow_mean=1,         # 0: 使用上下文单词向量的总和; 1: 使用均值，适用于使用CBOW。
        seed=1,              # 随机种子
        workers=4            # 线程数
    )
训练后将每个句子中的词向量求和取平均，作为算法的输入

### 4.2.1 Skip_Gram

在 Skip_Gram 上尝试词向量大小为 100，使用朴素贝叶斯拟合结果如下：

![Skip_Gram-朴素贝叶斯](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/Skip_Gram-朴素贝叶斯.png)



### 4.2.2 CBOW

在 CBOW 上尝试词向量大小为 200，使用朴素贝叶斯拟合结果如下：

![CBOW-朴素贝叶斯](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/CBOW-朴素贝叶斯.png)

采用十折交叉验证，比较在不同模型上的效果：

![CBOW-模型比较](/Users/apple/Documents/Project/旅游评论情感/result/0-2. Feature Engineering/CBOW-模型比较.png)

可以看见，树模型效果较好