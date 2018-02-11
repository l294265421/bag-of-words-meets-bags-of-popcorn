from data.raw_data import *
from util.text_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score
import numpy as np

# 预处理数据
label = train_df['sentiment']
train_data = []
for i in range(len(train_df['review'])):
    train_data.append(' '.join(review_to_wordlist(train_df['review'][i])))

test_data = []
for i in range(len(test_df['review'])):
    test_data.append(' '.join(review_to_wordlist(test_df['review'][i])))

# 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
tfidf = TfidfVectorizer(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]

model_NB = MNB()
model_NB.fit(train_x, label)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))

test_predicted = np.array(model_NB.predict(test_x))
nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
nb_output['id'] = test_df['id']
nb_output = nb_output[['id', 'sentiment']]
nb_output.to_csv(base_dir + 'nb.csv', index=False)