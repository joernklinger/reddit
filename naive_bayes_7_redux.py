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
from scipy.sparse import csc_matrix

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

# chose unique authors
authors = list(set(authors))

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
nr_of_users = len(authors)
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

# Filter 'I don't know responses out of the irony_matrix
# Apparently there are not 'I don't know entries'
irony_matrix_filtered = []
for entry in irony_matrix:
    if entry[7] != 0:
        irony_matrix_filtered.append(entry)

# Change coding for irony
irony_matrix = []
for entry in irony_matrix_filtered:
    if entry[7] == -1:
        entry[7] = 0 # no irony
    irony_matrix.append(entry)


for c in xrange(user_classes):
    print c
    part_interaction = bow.multiply(userclass[:, c])
    interaction = hstack((interaction, part_interaction))


# May have to use lil_matrix for X1 too instead of using .toarray on userclass

X1 = np.empty([len(irony_matrix), 1+user_classes])
X1[:,0] = intercept
X1[:,1:user_classes+1] = userclass.toarray()

logreg_X_init = hstack((X1, bow, interaction))
logreg_X_init_dense = logreg_X_init.todense()

logreg = lr(class_weight='auto')  # Set lr.coef_ = []
logreg.fit(logreg_X_init, logreg_Y_init)
coef_init = logreg.coef_
print('coef_init: ' + str(coef_init))

# == Run the model ==
# Iteration loop
iterations = 2

for it in xrange(iterations):

    # == User Loop ==
    # probability_feature_terms_given_class [p1, p2, ..., pn]
    #
    ## BYRON: can we make this faster? vectorize it?
    for user in xrange(nr_of_users):
        print('User: ' + str(user))

        # ATTENTION vectorized version:
        probability_feature_terms_given_class = np.zeros([user_classes, 1], dtype=float)
        # for each user_class, multiply conditional_probabilities_log * feature_vector[user] and take the sum of that; that gives us the sum of all the conditional probabilities, given that the current user is in that class
        for x in xrange(user_classes):
            probability_feature_terms_given_class[x] += np.sum(conditional_probabilities_log[:,x] * feature_vector[user])
        # BYRON: can you verify that the code above does the same as the code below?

        # ATTENTION: old for-loop version:
        # probability_feature_terms_given_class = np.zeros([user_classes, 1], dtype=float)
        # for feature in xrange(nr_of_features):
        #     if feature_vector[user, feature] == 1:
        #         for x in xrange(user_classes):
        #             probability_feature_terms_given_class[x] += conditional_probabilities_log[feature][x]



        # Initialize vector V for the current user

        # We are aggregating the probs of observing the Tweets that we did with respect to ironic or not ironic conditioned on the words in those tweets and the current class assignmenst.

        # NEW: Make soft assignments

        V = np.zeros(user_classes)
        # Vector with probabilities that the current user is in a given class
        current_users_prob_class = probability_user_in_class_log[user]
        ###

        # logreg.predict from intercept, class and bag of words and interaction
        if it == 0:
            logreg_X = logreg_X_init
            logreg_X_dense = logreg_X_init_dense

        # Calculating the probability that a user's tweet is ironic, given the users probability of being in class 0, 1 ..., n
        for comment in irony_matrix:
            if comment[5] == authors[user]:
                for c in xrange(user_classes):
                    if comment[7] == 1:
                        predictors = hstack((hstack((1, current_users_prob_class)), logreg_X_dense[0,(1+user_classes):]))
                        V[c] += np.log(logreg.predict_proba(predictors)[0])[1]

                    elif comment[7] == 0:
                        predictors = hstack((hstack((1, current_users_prob_class)), logreg_X_dense[0,(1+user_classes):]))
                        V[c] += np.log(1 - (logreg.predict_proba(predictors)[0])[1])


        for c in xrange(user_classes):
                    probability_user_in_class_log[user][c] = (probability_feature_terms_given_class[c] + class_probabilities_log[c] + V[c])

        # Normalize

        logs_pn = np.empty(user_classes)
        logs_pn = probability_user_in_class_log[user]
        z = logsumexp(logs_pn)
        logs_pn_normalized = logs_pn - z

        probability_user_in_class_log[user] = logs_pn_normalized

    # == End User Loop ==
    #
    # LOGREG:

    # Y ~ intercept + user_class + bow + bow*userclass

    # intercept is the same as in the initialization
    # userclass needs to be determined again
    # ineraction needs to be recalculated
    # hstack with bow and interaction
    #
    intercept = np.ones(len(irony_matrix))
    interaction = lil_matrix((0, bow.shape[1]*user_classes))
    userclass = lil_matrix((len(irony_matrix), user_classes))

    logreg_Y = logreg_Y_init

    for row in xrange(len(irony_matrix)):
        user_ind = authors.index(irony_matrix[row][5])
        userclass[row] = probability_user_in_class_log[user_ind]
        logreg_Y[row] = irony_matrix[row][7]

    # We will use soft assignments for class: P(user in class1)

    # d is just a delimeter for the columns
    d = interaction.shape[1]/user_classes

    for c in xrange(user_classes):
        print c
        part_interaction = bow.multiply(userclass[:, c])
        interaction = hstack((interaction, part_interaction))

    X1 = np.empty([len(irony_matrix), 1+user_classes])
    X1[:,0] = intercept
    X1[:,1:user_classes+1] = userclass.toarray()

    logreg_X = hstack((X1, bow, interaction))
    logreg_X_dense = logreg_X.todense()

    # ATT: Why is feature_counts = np.ones, not np.zeroes? Avoid 0 probabilitiy?
    feature_counts = np.ones([nr_of_features, user_classes], dtype=float)

    # == Feature Count Loop ==
    for feature in xrange(nr_of_features):
        z = np.zeros(user_classes)
        for user in xrange(nr_of_users):
            if feature_vector[user][feature] == 1:
                feature_counts[feature] += np.exp(probability_user_in_class_log[user])
            z += np.exp(probability_user_in_class_log[user])
        conditional_probabilities_log[feature] = np.log(feature_counts[feature]) - np.log(z)
    # == End Feature Count Loop ==

    # Calculate probability of the entire feature_vector given curren parameters
    probability_user_given_data_log = np.zeros(nr_of_users, dtype=float)

    # == Probability of Entire Data Loop ==
    for user in xrange(nr_of_users):
        for feature in xrange(nr_of_features):
            if feature_vector[user][feature] == 1:
                p_user_given_class_log = np.empty(user_classes)
                for x in xrange(user_classes):
                    p_user_given_class_log[x] = (conditional_probabilities_log[feature][x] + probability_user_in_class_log[user][x])

                probability_user_given_data_log[user] += logsumexp(p_user_given_class_log)
    # == End Probability of Entire Data Loop ==
    probability_data_given_parameters_log = np.sum(probability_user_given_data_log)

    # Update class probailities log
    p_total_log = np.empty(user_classes)

    # == Update Class Probailities Loop ==
    for x in xrange(user_classes):
        p_total_log[x] = logsumexp(probability_user_in_class_log[:, x])
        class_probabilities_log[x] = p_total_log[x] - np.log(nr_of_users)
    # == End Update Class Probailities Loop ==

    # Get raw class probabilities for better legibilitiy
    class_probabilities_raw = np.zeros(user_classes)

    # == Raw Class Probabilities Loop ==
    for x in xrange(user_classes):
        class_probabilities_raw[x] = np.exp(class_probabilities_log[x])
    # == End Raw Class Probabilities Loop ==

    ### ADD logistic regression and re-estimation of beta here
    # Need some help here, but it looks like it's on the right track

    # logreg_data = [0: UserID, 1: TweetID, 2: Intercept, 3: prob0, 4: prob1, 5: Confidence, 6: Irony]
    # SET LR.COEF_ TO the values we initilized beta to
    # INTERCEPT
    ### Predictors: [Intercept, prob1]
    if it == 0:
        logreg.coef_ = coef_init

    logreg.fit(logreg_X, logreg_Y)

    # Now give stats on the model's state
    print 'Nr. of classes: ' + str(user_classes)
    print 'Model: ' + str(model)
    print 'Iteration: ' + str(it)
    print 'Raw Class Probabilities: ' + str(class_probabilities_raw)
    print 'Log Likelihood: ' + str(probability_data_given_parameters_log)
    print 'Logreg Coefs: ' + str(logreg.coef_)

    # Summary: [model, iteration, loglikelihood]
    summary[counts] = [counts, model, it, probability_data_given_parameters_log, user_classes]

    all_conditional_probabilities.append([counts, model, it, conditional_probabilities_log])
    all_probabilities_user_in_class.append([counts, model, it, probability_user_in_class_log])
    all_class_probabilities.append([counts, model, it, class_probabilities_raw])

    counts += 1

    if last_probability_data_given_parameters_log >= probability_data_given_parameters_log:
        print('Local Minimum; Break.')
        break

    last_probability_data_given_parameters_log = probability_data_given_parameters_log

# == End Iteration Loop ==
