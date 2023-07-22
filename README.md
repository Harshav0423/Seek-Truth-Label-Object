# This is from EAI Assignment @ IUB MSCS

Classification of Text, documents, emails and tweets. Naive Bayes classifier is used to label the object/text, in this algorithm it uses probability of the events for its purpose. It is based on the Bayes Theorem which assumes that there is no interdependence amongst the variables, against dependence of events in Bayes Nets. With this we can build Truthful and Deceptive bags of words. Following Naive approach, using trianed bag of words and labelling the reviews and testing it against the words. 

          The Bayes Theorem:
          P(A/B) =P(A) x P(B/A) / P(B)

Approach using Naive Bayes classifier:

            -creating frequency of words from the train dataset, into truthful and deceptive
            -calculating likelihood of each word, through the number of words present in each label, divide it by number of words present in a label
            -as the prior probability will be constant which doesn't change the probability
            -The higher probability leads towards the label.
            -Considered, Multinomial Naive bayes, which gets the integer of words. 

 
Tweaks:

             -words which are not present in the bag are considered as 1, if not the total prob becomes zero.
             -deleted stop words, which may affect the probability outcome like 'are','by','were','is','if', and numbers
             -calculating prob of a word when it is found is count of it, if not it is 1
               reason: since we are considering likelihood of every word in the truthful/deceptive
             -it is best give the prob as 1 if it is not present Called Zero-Probability
             - SMOOTHING: Laplace smoothing can be p(w`/truth) = count of w` + alpha/(no.of truth + k*alpha)
               alpha = smoothing (1 will be ideal) and k = no. of features (tru_count+tru_decep)
             - probability are logged and then summed, it may cause uderflow of prob if we use multiplication, which is not good.
 
 The final accuracy achieved for this approach is 84.75% classifying (test into 186 as truthful and 214 as deceptive out of 400).