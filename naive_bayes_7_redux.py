# TODO:
# - make class assignment soft

#
# add support for more than two classes
# multiple class sizes

# Import modules
import numpy as np
from numpy import log
import pdb
import cPickle as pickle
import json
from sklearn.linear_model import LogisticRegression as lr

# Import modules for BOWs
from sklearn.feature_extraction.text import TfidfVectorizer

# Import modules for sparse matrix operations
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

def logsumexp(a):
    a = np.asarray(a)
    a_max = a.max()
    return a_max + np.log((np.exp(a - a_max)).sum())

# Load the data from pickle file, load user_ids from pickle file as well
# data: feature_vector
f = open('feature_vector.pickle', 'r')
feature_vector = pickle.load(f)
f.close()

g = open('irony_matrix.pickle', 'r')
irony_matrix = pickle.load(g)
g.close()

h = open('authors.pickle', 'r')
authors = pickle.load(h)
h.close()

# Create bag of words
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

# extract text from comments
comments_for_bow = []
for row in irony_matrix:
    comments_for_bow.append(row[3])

# bag of words
bow = bigram_vectorizer.fit_transform(comments_for_bow)

# User classes vector. Specify the different amounts of user_classes that you wanna build models with
user_classes_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# --- Set Parameters ---

# set parameter for models, summary and iterations and SD
# number of different models, number of classes to divide users into

models = 1
iterations = 5
sd = 0.0001

# Initialize beta for logistic regression
beta = np.log(np.random.normal(loc=0.5, scale=sd))

# summary data file; contains stats for each model and each iteration
summary = np.zeros([models * iterations * len(user_classes_list), 5])
all_conditional_probabilities = []
all_probabilities_user_in_class = []
all_class_probabilities = []

# Counts is basically the line number each new entry in the summary file
counts = 0

# == For each user_class in user_class_list ==
# for user_class_nr in user_classes_list:
user_classes = user_classes_list[1]  # fix to [0] here, change later, uncomment

# Set model to 1 here, later: for loop
model = 1

# == for each model ==

# for model in xrange(models):

# Initialize last_probability_data_given_parameters_log as negative infinity
last_probability_data_given_parameters_log = -np.inf

# probabilities that a randomly drawn user is in class 0 or 1 [p_c1, p_c2, ..., p_cn]
class_probabilities_log = np.ones(user_classes) * np.log(float(1) / user_classes)

# conditional probabilities for each feature by class
conditional_probabilities_log = np.empty([feature_vector.shape[1], user_classes], dtype=float)

# for one class we initialize them all as log(float(1)/user_classes)
# for the other classes we initialize them as a random close to log(float(1)/user_classes)
for row in xrange(len(conditional_probabilities_log)):
    conditional_probabilities_log[row, 0] = log(float(1) / user_classes)
    for column in xrange(1, user_classes):
        conditional_probabilities_log[row, column] = log(np.random.normal(loc=float(1) / user_classes, scale=sd))

# Define for easier access
nr_of_users = feature_vector.shape[0]
nr_of_features = feature_vector.shape[1]

# probability_user_in_class_log rows: users, cols: log([p(class0), p(class1), ..., p(classn)])
# initialized to a random number
probability_user_in_class_log = np.empty([nr_of_users, user_classes], dtype=float)

# Randomly assigning users to groups
for row in xrange(len(probability_user_in_class_log)):
    for col in xrange(probability_user_in_class_log.shape[1]):
        probability_user_in_class_log[row, col] = np.log(np.random.normal(loc=float(1) / user_classes, scale=sd))

### Initialize the logreg

# Y ~ intercept + user_class + bow + bow*userclass
# Y ~ intercept + P(user_class 0) + P(user_class 1) + ... + P(user_class n) + bow + bow * P(user_class 0) + bow * P(user_class 1) + ... + P(user_class n)

# interaction = np.empty([bow.shape[0], bow.shape[1]*(user_classes)])
intercept = np.ones(len(irony_matrix))
# userclass = np.empty([len(irony_matrix), user_classes])

# sparse matrices
interaction = lil_matrix((0, bow.shape[1]*user_classes))
userclass = lil_matrix((len(irony_matrix), user_classes))


logreg_Y_init = np.empty([len(irony_matrix)])

# for row in irony_matrix:
#     user_ind = wo der user name in authors ist ist, das ist dann die  row im feature_vector
for row in xrange(len(irony_matrix)):
    user_ind = authors.index(irony_matrix[row][5])
    userclass[row] = probability_user_in_class_log[user_ind]
    logreg_Y_init[row] = irony_matrix[row][7]

bow_dense = bow.todense()


# We will use soft assignments for class: P(user in class1)

# d is just a delimeter for the columns
d = interaction.shape[1]/user_classes

# user_classes = user_classes_list[1]  # fix to [0] here, change later, uncomment
for c in xrange(user_classes):
    print c
    part_interaction = bow.multiply(userclass[:, c])
    interaction = hstack((interaction, part_interaction))


# May have to use lil_matrix for X1 too instead of using .toarray on userclass

X1 = np.empty([len(irony_matrix), 1+user_classes])
X1[:,0] = intercept
X1[:,1:user_classes+1] = userclass.toarray()

logreg_X_init = hstack((X1, bow, interaction))

logreg = lr(class_weight='auto')  # Set lr.coef_ = []
logreg.fit(logreg_X_init, logreg_Y_init)
coef_init = logreg.coef_
print('coef_init: ' + str(coef_init))
