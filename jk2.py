# Some comments aren't labeled, so I need to use the labeled_comments_ids subset of comments
# Use that to find the segments corresponding to those comments
# Then do what I'm doing

from annotation_stats import *
import pickle
from pandas import *
from collections import Counter
import pdb
from string import lstrip

labeled_comment_ids = get_labeled_thrice_comments()
sentence_ids, subreddits = get_sentence_ids_for_comments(labeled_comment_ids)

collapse_f = lambda lbl_set: 1 if lbl_set.count(1) >= 2 else -1
sentence_ids, sentence_texts, sentence_lbls = get_texts_and_labels_for_sentences(
        sentence_ids, repeat=False, collapse=collapse_f, add_punctuation_features_to_text=False)


# get all authors
authors = cursor.execute(
            '''select redditor from irony_comment;''').fetchall()

authors2 = []
for a in authors:
    authors2.append(a[0])
authors = authors2

# get past user comments

irony_pastusercomment_tmp = cursor.execute(
            '''select * from irony_pastusercomment;''').fetchall()
irony_pastusercomment = DataFrame(irony_pastusercomment_tmp)
irony_pastusercomment.columns = ['id', 'redditor', 'comment_text', 'subreddit', 'thread_title', 'thread_id', 'thread_url', 'upvotes', 'downvotes', 'date', 'permalink']

# BUILD FEATURE VECTOR
# user_ids
# all users to build feature vector

# feature vector array,
# rows = nr of users
# columns = nr of total subreddits

# Get total number of subreddits
subreddits = list(set(irony_pastusercomment['subreddit'].tolist()))

feature_vector_shape = (len(authors), len(subreddits))
feature_vector = np.zeros(feature_vector_shape)

# for each user, find the subreddits of that user
# find the indices of those subreddits
# make those columns in the feature vector 1

a = authors[0]
subreddits_of_a=list(set(irony_pastusercomment[irony_pastusercomment['redditor'] == a]['subreddit'].tolist()))
indices_of_subreddits = [i for i, x in enumerate(subreddits) if x in subreddits_of_a]
row = authors.index(a)
feature_vector[row,indices_of_subreddits] = 1

# BUILD IRONY DATA FRAME
# sentence_id, author_id, sentence_text, label

# id, reddit_id, subreddit, thread_title, thread_urlm thread_id, redditor, parent_comment_id, to_label, downvotes, upvotes, permalink
irony_comment = cursor.execute(
            '''select * from irony_comment;''').fetchall()
irony_comment2 = [list(u) for u in irony_comment]
irony_comment = DataFrame(irony_comment2)
irony_comment.columns = ['id', 'reddit_id', 'subreddit', 'thread_title', 'thread_url', 'thread_id', 'redditor', 'parent_comment_id', 'to_label', 'downvotes', 'upvotes', 'permalink']

# id, segment_id, comment_id, labeler_id, label, forced_decision, viewed_thread, viewed_page, confidence
irony_label = cursor.execute(
            '''select * from irony_label;''').fetchall()
irony_label2 = [list(u) for u in irony_label]
irony_label = DataFrame(irony_label2)
irony_label.columns = ['id', 'segment_id', 'comment_id', 'labeler_id', 'label', 'forced_decision', 'date', 'viewed_thread', 'viewed_page', 'confidence']


# id, comment_id, segment_id, text
irony_segment = cursor.execute(
            '''select * from irony_commentsegment;''').fetchall()
irony_segment2 = [list(u) for u in irony_segment]
irony_segment = DataFrame(irony_segment2)
irony_segment.columns = ['id', 'comment_id', 'segment_id', 'text']

# WEIRD, sometimes list index out of range
# id, comment_id, text, *user_id, subreddit, irony*
irony_matrix = []

# irony_segment['id'] (not ['segment_id'] !!!) corresponds to irony_label['segment_id']

for s in irony_segment['id']:
    # print s
    c_id = int(irony_segment[irony_segment['id'] == s]['comment_id'])

    # if comment is in labeled_comments then do below
    if c_id in labeled_comment_ids:
        ind = int(str(irony_segment[irony_segment['id'] == s]['text'])[0:5])
        s_text = irony_segment[irony_segment['id'] == s]['text'][ind]

        # get majority label
        s_label_tmp = list(irony_label[(irony_label['comment_id'] == c_id) & (irony_label['segment_id'] == s)]['label'])
        s_label = Counter(s_label_tmp).most_common()[0][0]

        # get subreddit
        ind = int(str(irony_comment[irony_comment['id'] == c_id]['subreddit'])[0:5])
        c_subreddit = irony_comment[irony_comment['id'] == c_id]['subreddit'][ind]

        # get author
        ind = int(str(irony_comment[irony_comment['id'] == c_id]['redditor'])[0:5])
        c_author = irony_comment[irony_comment['id'] == c_id]['redditor'][ind]
        c_author_index = authors.index(c_author)

        append_to_irony_matrix = [s, c_id, s_id, s_text, c_subreddit, c_author, c_author_index, s_label]
        irony_matrix.append(append_to_irony_matrix)



