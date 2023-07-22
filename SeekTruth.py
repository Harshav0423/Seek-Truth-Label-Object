# SeekTruth.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
#
# Name: Harsha Valiveti
# Based on skeleton code by D. Crandall, October 2021
#

import sys
import math
def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data):
    # This is just dummy code -- put yours here!
    
    stop_words = {'a','an','are','by','were','is','if','the','then','through','this',
                  'they','them','their','on','in','out','our','you','your','after','any','also','go',
                  'my','me','here','he','she','her','has','how','what','where','when','with','should','to','from','form','for','do',' ','.',',','1',
                  '0','1','2','3','4','5','6','7','8','9'} 
    
    del train_data['classes']
    
    train_word_t = {}
    train_word_d = {}
    
    for i in range(len(train_data['objects'])):
        words =  train_data['objects'][i].split()
        if train_data['labels'][i] == 'truthful':
            for wd in words:
                w = wd.lower()
                
                if w in train_word_t:
                    train_word_t[w] += 1
                else:
                    train_word_t[w] = 1
            
        if train_data['labels'][i] == 'deceptive':
            for wd in words:
                w = wd.lower()    
                if w in train_word_d:
                    train_word_d[w] += 1
                else:
                    train_word_d[w] = 1
    
    # deleting the stop words from Truth and deceptive ditionary
    # Having these stop words increases the both probability which is not ideal to use
    # in the text classifier, which may classify worngly
    for s in stop_words:
        if s in train_word_t:
            del train_word_t[s]
        if s in train_word_d:
            del train_word_d[s]
    
    #print(train_word_t[' '])
    
   
    # https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
    train_tlen = sum(train_word_t.values())
    train_dlen = sum(train_word_d.values())
    
    tot = len(train_word_t) + len(train_word_d)
    test_class = [] # to store label for each review
    test_prob_t = {}
    test_prob_d = {}
    
    #calculating prob of a word when it is found is count of it, if not it is 1
    #reason: since we are considering likelihood of every word in the truthful/deceptive
    #it is best give the prob as 1 if it is not present
    #Called Zero-Probability
    #Laplace smoothing can be p(w`/truth) = count of w` + alpha/no.of truth + k*alpha
    # alpha = smoothing (1 will be ideal)
    # k = no. of features
    
    for j in range(len(test_data['objects'])): # calculating prob for every word in sentence
        words =  test_data['objects'][j].split()
        test_prob_t = {}
        test_prob_d = {}
        for wd in words:
            w = wd.lower()
            
            test_prob_t[w] = train_word_t.get(w,1)/(train_tlen + tot)
            test_prob_d[w] = train_word_d.get(w,1)/(train_dlen + tot)
        
        s = 0
        for k in test_prob_t.keys():
            s += math.log(test_prob_t[k]) #uderflow of prob if we use multiplication
       
        
        truth = s
        s = 0
        for k in test_prob_d.keys():
            s += math.log(test_prob_d[k]) 
        decep = s
        
        if truth > decep:
            test_class.append('truthful')
            
        else:
            test_class.append('deceptive')
        
    print("test class",test_class.count("truthful"),test_class.count('deceptive'))
    return test_class


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}
    
    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    #print(correct_ct)
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
