# Text
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import brown
from nltk.corpus import stopwords

# Best Friend
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score

#basic
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

#################################
####### TEXT TREATMENT ##########
#################################
pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[a-z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''

en_stopwords = stopwords.words("english")
names = pd.read_table('personnames.txt',sep='\s+',engine='python', header=None, encoding='utf-8')
names = names[0].tolist()
# names = []
punctuation = list(set(string.punctuation))
punctuation.append('--')
punctuation.append('...')
punctuation.append('``')
punctuation.append('`')
punctuation.append("''")
punctuation.append('""')

ignore_list = en_stopwords+names+punctuation

vetorizer = TfidfVectorizer(token_pattern=pattern, stop_words=ignore_list, ngram_range=(1, 3))


#################################
####### DATA ARRANGEMENT ########
#################################
#Two more frequent categories >> news': 44, 'belles_lettres': 75
categories = ['news','belles_lettres']

fileids = brown.fileids(categories=categories)

cat_sizes = {}
for category in categories:
	cat_sizes[category] = len(brown.fileids(categories=category))

cat_idxs = {}
for category in categories:
	cat_idxs[category] = []

data = np.empty(len(fileids),dtype=object)
label = np.empty(len(fileids),dtype=object)

for i, fileid in enumerate(fileids):
	doc_categories = brown.categories(fileid)
	category = doc_categories[0]

	data[i] = brown.raw(fileid)
	label[i] = category
	cat_idxs[category].append(i)

perc_train = 0.66

results = []

for clf, clf_name in (
	(MultinomialNB(alpha=0.01),'MultinomialNB'),
	(KNeighborsClassifier(n_neighbors=10),'KNN'),
	(RandomForestClassifier(n_estimators=100),'RandomForest')
	):

	exps = 30

	conf_matrix_exps = np.zeros([exps,len(categories),len(categories)])
	fbeta_exps = np.zeros([exps])

	for exp in range(exps):

		train_idxs = np.array([], dtype=int)
		for category in categories:

			cat_idxs[category] = np.array(cat_idxs[category],dtype=int)

			cat_train_size = int(cat_sizes[category]*perc_train)			

			cat_train_idxs = np.random.choice(cat_idxs[category],cat_train_size, replace=False)
			train_idxs = np.concatenate((train_idxs, cat_train_idxs)) 

		test_idxs = np.array([i for i in range(len(fileids)) if i not in train_idxs])

		train_ifdf_mtx = vetorizer.fit_transform(data[train_idxs])
		test_ifdf_mtx = vetorizer.transform(data[test_idxs])

		params = {'alpha': 0.01}
		clf = clf.fit(train_ifdf_mtx, label[train_idxs])
		pred = clf.predict(test_ifdf_mtx)
		conf_matrix_exps[exp] = confusion_matrix(label[test_idxs], pred)
		fbeta_exps[exp] = fbeta_score(label[test_idxs], pred, average='weighted', beta=0.5)

	conf_matrix_mean = np.zeros([len(categories),len(categories)])
	conf_matrix_std = np.zeros([len(categories),len(categories)])
	np.mean(conf_matrix_exps, axis=0, out = conf_matrix_mean)
	np.std(conf_matrix_exps, axis=0, out = conf_matrix_std)

	print()
	print('Confusion matrix (mean values for ',exps,' experiments,',clf_name,')' )
	print(conf_matrix_mean)
	print()

	print('Confusion matrix (std values for ',exps,' experiments,',clf_name,')' )
	print(conf_matrix_std)
	print()

	result = dict(
		clf_name=clf_name,
		exps = exps,
		conf_matrix_mean = conf_matrix_mean,
		conf_matrix_std = conf_matrix_std,
		fbeta_mean = np.mean(fbeta_exps),
		fbeta_std = np.std(fbeta_exps),
		)

	results.append(result)

print(results)

# print(cm)

# plt.matshow(cm)
# plt.title('Confusion matrix of MultinomialNB')
# plt.colorbar()

# plt.show()