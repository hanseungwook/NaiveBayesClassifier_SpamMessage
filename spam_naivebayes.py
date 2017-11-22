"""
COMS W4701 Artificial Intelligence - Programming Homework 3

Naive Bayes Spam Classifier

@author: Seungwook Han (sh3264)
"""

# Default (k=1, c=1): Precision:0.9516129032258065 Recall:0.8805970149253731 F-Score:0.9147286821705426 Accuracy:0.9802513464991023
# Tuned (k=2, c=0.3): Precision:0.953125 Recall:0.9104477611940298 F-Score:0.931297709923664 Accuracy:0.9838420107719928
# Stop-words: Precision:0.9523809523809523 Recall:0.8955223880597015 F-Score:0.923076923076923 Accuracy:0.9820466786355476

import sys
import string
import re
import math

# Lower-case the text, remove any punctuation, and then split it on white spaces
def extract_words(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    preprocess = (text.translate(translator)).lower()
    extractList = preprocess.split()
    return extractList


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}
        self.word_given_label = {}
        self.stop_words = set()

        # The stopwords file is given as the third command line argument of the program
        # Read in stop words and then put into the set
        with open(stopword_file, 'r') as f:
            for line in f:
                newLine = line.replace('\n','')
                self.stop_words.add(newLine)

        self.collect_attribute_types(training_filename, 2)
        self.train(training_filename)

    # Compute vocabulary consisting of the set of unique words occurring at least k times
    def collect_attribute_types(self, training_filename, k):
        vocab = dict()

        with open(training_filename, 'r') as f:
            for line in f:
                tabSplit = re.split(r'\t+', line)
                extractList = extract_words(tabSplit[1])
                for word in extractList:
                    # Ignore stop words
                    if word in self.stop_words:
                        continue
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            for key in vocab.keys():
                if(vocab[key] >= k):
                    self.attribute_types.add(key)


    def train(self, training_filename):
        with open(training_filename, 'r') as f:
            # Initializing counters
            spamCounter = 0
            hamCounter = 0
            spamWGL = 0
            hamWGL = 0
            totalWGL = None

            # Initializing smoothing c parameter & vocabSize
            c = 0.3
            vocabSize = len(self.attribute_types)

            for line in f:
                spamLabel = 'spam'
                hamLabel = 'ham'
                # Split the line by tab
                tabSplit = re.split(r'\t+', line)
                label = tabSplit[0]

                # Get proprocessed words with extract_words
                extractList = extract_words(tabSplit[1])

                # If the label is a spam, then increment # of spam and add the respective
                # (word, label) in the word_given_label dict (or increment)
                if (label == spamLabel):
                    spamCounter += 1
                    for word in extractList:
                        if word in self.attribute_types:
                            spamWGL += 1
                            if (word, spamLabel)  in self.word_given_label:
                                self.word_given_label[(word, label)] += 1
                            else:
                                self.word_given_label[(word, label)] = 1

                # If the label is a ham, then increment # of ham and add the respective
                # (word, label) in the word_given_label dict (or increment)
                elif (label == hamLabel):
                    hamCounter += 1
                    for word in extractList:
                        if word in self.attribute_types:
                            hamWGL += 1
                            if (word, hamLabel) in self.word_given_label:
                                self.word_given_label[(word, label)] += 1
                            else:
                                self.word_given_label[(word, label)] = 1

                # If the first word separated by tab is not the expected labels, error
                else:
                    sys.exit('Label error')

            # Put P(spam) and P(ham) into respective dict
            total = spamCounter + hamCounter
            self.label_prior['spam'] = spamCounter / total
            self.label_prior['ham'] = hamCounter / total

            attrPair = (None, None)
            # Calculating P(Word=w|Label=y) with smoothing and assume 0 occurrences for unseen pairs
            for attribute in self.attribute_types:
                totalWGL = spamWGL
                attrPair = (attribute, spamLabel)
                if attrPair in self.word_given_label:
                    self.word_given_label[attrPair] = (self.word_given_label[attrPair] + c) / (totalWGL + c * vocabSize)
                else:
                    self.word_given_label[attrPair] = c / (totalWGL + c * vocabSize)

                totalWGL = hamWGL
                attrPair = (attribute, hamLabel)
                if attrPair in self.word_given_label:
                    self.word_given_label[attrPair] = (self.word_given_label[attrPair] + c) / (totalWGL + c * vocabSize)
                else:
                    self.word_given_label[attrPair] = c / (totalWGL + c * vocabSize)

            # testSum1 = testSum2 = 0
            # for attribute in self.attribute_types:
            #     pair1 = (attribute, 'spam')
            #     testSum1 += self.word_given_label[pair1]
            #     pair2 = (attribute, 'ham')
            #     testSum2 += self.word_given_label[pair2]


    def predict(self, text):
        predictWGL = dict()
        extractList = extract_words(text)
        spamLabel = 'spam'
        hamLabel = 'ham'

        sumSpam = math.log(self.label_prior[spamLabel])
        sumHam = math.log(self.label_prior[hamLabel])

        # Sum all the probabilities if the word is in attributes
        for word in extractList:
            if word in self.attribute_types:
                sumSpam += math.log(self.word_given_label[(word, spamLabel)])
                sumHam += math.log(self.word_given_label[(word, hamLabel)])

        predictWGL[spamLabel] = sumSpam
        predictWGL[hamLabel] = sumHam

        return predictWGL


    def evaluate(self, test_filename):
        TP = FP = FN = TN = 0
        spamLabel = 'spam'
        hamLabel = 'ham'

        with open(test_filename, 'r') as f:
            # For each of the text/messages, use predict() to calculate the
            # prediction and compare it to actual value in order to assess
            # the # of TP, FP, FN, TN
            for line in f:
                tabSplit = re.split(r'\t+', line)
                actualLabel = tabSplit[0]
                predictProb = self.predict(tabSplit[1])
                prediction = None
                if predictProb[spamLabel] > predictProb[hamLabel]:
                    prediction = spamLabel
                else:
                    prediction = hamLabel
                if actualLabel == spamLabel:
                    if prediction == actualLabel:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if prediction == actualLabel:
                        TN += 1
                    else:
                        FP += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fscore = (2 * precision * recall) / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    # Pass the stop words file as the third command line argument
    classifier = NbClassifier(sys.argv[1], sys.argv[3])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)

