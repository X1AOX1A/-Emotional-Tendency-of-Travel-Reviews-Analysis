# -------- BoW & tf-idf ------- #
def stop_words(data, path='data/', file='stopwords.txt'):
    """
    input:
        data: pd.Series with list element
        path: str, path of stopwords file
        file: str, name of stopwords file

    output:
        data: pd.Series
    """
    s = ''
    with open(path+file, 'r', encoding='utf8') as r:
        for i in r.readlines():
            s += i.strip()
    data = data.map(lambda x:[i for i in x if(i not in s) and (len(i) > 1)])
    return data

from sklearn.feature_extraction.text import CountVectorizer
def bow(X_train, X_test, max_features=None, sparse=True):
    """
    input:
        X_train: pd.Series with list element
        X_test: pd.Series with list element
        max_features: int, default None
    
    output:
        X_train: csr_matrix or ndarray
        X_test: csr_matrix or ndarray
        feature_name: list
    """
    print('BOWing...')
    # 将 pd.Series 转换为 list
    # 每个句子为一个 str
    # 每个单词由空格分开
    X_train = X_train.map(lambda line:' '.join(line))    
    X_test = X_test.map(lambda line:' '.join(line))  
    
    BOW = CountVectorizer(max_features=max_features)
    X_train = BOW.fit_transform(X_train)          
    X_test = BOW.transform(X_test)
    feature_name = BOW.get_feature_names()
    
    if sparse==True:
        print('X is a Sparse Matrix')
    else:
        print('X is a dense Matrix')
        # 将稀疏矩阵转换为 ndarray
        X_train = X_train.A    
        X_test = X_test.A    
    return X_train, X_test, feature_name


from sklearn.feature_extraction.text import TfidfTransformer
def tf_idf(X_train, X_test, max_features=None, sparse=True):
    """
    input:
        X_train: pd.Series with list element
        X_test: pd.Series with list element
        max_features: int, default None
        sparse: bool, return sparse matrix, default True
    
    output:
        X_train: csr_matrix or ndarray
        X_test: csr_matrix or ndarray
        feature_name: list
    """
    X_train, X_test, feature_name = bow(X_train, X_test, max_features)
    print('tf_idfing...')
    TF_idf = TfidfTransformer()
    X_train = TF_idf.fit_transform(X_train)     # 返回的是稀疏矩阵
    X_test = TF_idf.transform(X_test)
    if sparse==True:
        print('X is a Sparse Matrix')
    else:
        print('X is a dense Matrix')
        # 将稀疏矩阵转换为 ndarray
        X_train = X_train.A    
        X_test = X_test.A    
    return X_train, X_test, feature_name

def print_info(X_train, X_test, Y_train, Y_test):
    print('\nX_train shape:', X_train.shape, 'type:', type(X_train))
    print('X_test shape:', X_test.shape, 'type:', type(X_test))
    print('Y_train shape:', Y_train.shape, 'type:', type(Y_train))
    print('Y_test shape:', Y_test.shape, 'type:', type(Y_test))
    

import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path='data/', file='广东(数据清洗4).xlsx', 
              vectorize='tf_idf', drop_stop_words=True, 
              max_features=None, sparse=True):
    """
    input:
        path: str, path of file, default 'data/'
        file: str, name of file, default '广东(数据清洗4).xlsx'
        vectorize: str, ['bow','tf_idf'], default 'tf_idf'
        drop_stop_words: bool, default True
        max_features: int, max_features of X, default None
        sparse: bool, whether 'tf_idf' return sparse matrix, default True
            
    output:
        X_train: csr_matrix or ndarray
        X_test: csr_matrix or ndarray
        Y_train: pd.Series
        Y_test: pd.Series
        feature_name: list
    """
    # 读取数据
    print('Reading data...')
    data = pd.read_excel(path+file, index_col=0)
    data.drop(columns=['景点','昵称','等级','时间'], inplace=True)
    data.columns = ['Corpus', 'Sentiment']
    
    Y = data['Sentiment']
    Y = Y.map({0:0, 1:0, 2:0, 3:0, 4:1, 5:1})       # 将评分转换为情感倾向
    X = data['Corpus']
    
    # 分词
    print('Splitting...')
    X = X.map(lambda x:jieba.lcut(x))    
    
    # 去除停用词
    if(drop_stop_words):
        print('Dropping Stop Words...')
        X = stop_words(X, path, file='stopwords.txt') 
           
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # 向量化
    if(vectorize=='bow'):                           # Bag of Word
        X_train, X_test, feature_name = \
        bow(X_train, X_test, max_features=max_features, sparse=sparse)
    elif(vectorize=='tf_idf'):                      # tf-idf
        X_train, X_test, feature_name = \
        tf_idf(X_train, X_test, max_features=max_features, sparse=sparse)
    elif(vectorize==None):
        print_info(X_train, X_test, Y_train, Y_test)
        return X_train, X_test, Y_train, Y_test
        
    print_info(X_train, X_test, Y_train, Y_test)
    return X_train, X_test, Y_train, Y_test, feature_name


# -------- Word2Vec ------- #
import gensim
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
def w2v_transformer(X, model, size):
    '''
    input:
        X: pd.DataFrame
        model: word2vec model
        size: int, size of X in word2vec model
        
    output:
        X_vec: np.ndarray, shape: (num of sample, num of features)
    '''
    print('Transforming...')
    X_vec = np.zeros((size,))
    for corpus in tqdm(X):
        vec = np.zeros((size,))
        length = 0
        for word in corpus:
            try:
                vec += model[word]
                length += 1
            except KeyError:
                continue
        length = length if length!=0 else 1
        vec = vec/length
        X_vec = np.vstack([X_vec, vec])
    X_vec = np.delete(X_vec, 0, axis=0)
    return X_vec


import os
def save_data(X_train, X_test, Y_train, Y_test, 
              path='data/word2vec/', file='New_data'):
    '''
    input:
        X_train: ndarray
        X_test: ndarray
        Y_train: ndarray
        Y_test: ndarray
    '''
    print('Saving data...')
    if(os.path.exists(path+file+'.npz')):
        for i in range(1,10):
            if(not(os.path.exists(path+file+str(i)+'.npz'))):
                file = file+'_'+str(i)+'.npz'
                break
    else:
        file = file+'.npz'
        
    np.savez(path+file, 
             X_train=X_train, Y_train=Y_train,
             X_test=X_test, Y_test=Y_test) 
    print('Data saved in :', path+file)


def Word2vec(path='data/', file='广东(数据清洗4).xlsx', 
             vectorize='CBOW', size=100,
             drop_stop_words=False, save_file=True):
    """
    input:
        path: str, path of file, default 'data/'
        file: str, name of file, default '广东(数据清洗4).xlsx'
        vectorize: str, ['CBOW', 'Skip_Gram'], default 'CBOW'
        drop_stop_words: bool, default False
        size: int, max_features of X, default 100
        save_file： bool defalut True
            
    output:
        X_train: ndarray
        X_test: ndarray
        Y_train: ndarray
        Y_test: ndarray
    """
    # 读取数据
    print('Reading data...')
    data = pd.read_excel(path+file, index_col=0)
    data.drop(columns=['景点','昵称','等级','时间'], inplace=True)
    data.columns = ['Corpus', 'Sentiment']
    
    Y = data['Sentiment']
    Y = Y.map({0:0, 1:0, 2:0, 3:0, 4:1, 5:1}).values   # 将评分转换为情感倾向
    X = data['Corpus']
    
    # 分词
    print('Splitting...')
    X = X.map(lambda x:jieba.lcut(x))    
    
    # 去除停用词
    if(drop_stop_words):
        print('Dropping Stop Words...')
        X = stop_words(X, path, file='stopwords.txt') 
        
    # word2vec
    if(vectorize=='CBOW'):
        sg=0
    elif(vectorize=='Skip_Gram'):
        sg=1
    else:
        print('vectorize type error')
        return None
    sentences = X.tolist()
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
    
    # 将 X 转换为向量形式
    X_vec = w2v_transformer(X, model, size)
    
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = \
    train_test_split(X_vec, Y, test_size=0.2, random_state=0)
    
    print_info(X_train, X_test, Y_train, Y_test)
    
    if save_file:
        save_data(X_train, X_test, Y_train, Y_test, 
                  file=vectorize+'_'+str(size))
    
    return X_train, X_test, Y_train, Y_test


def load_data_w2v(vectorize, size, path='data/word2vec/'):
    '''
    input:
        vectorize: str, ['CBOW', 'Skip_Gram']
        size: int, max_features of X
    
    output:
        X_train: ndarray
        X_test: ndarray
        Y_train: ndarray
        Y_test: ndarray
    '''
    print('Loading '+str(vectorize)+'_'+str(size)+'.npz')
    data = np.load(path+str(vectorize)+'_'+str(size)+'.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']
    print_info(X_train, X_test, Y_train, Y_test)
    return X_train, X_test, Y_train, Y_test