#!/usr/bin/python
from random import shuffle
from random import seed
import numpy as np
def get_data(filepath):
    raw_data = open(filepath,encoding='utf8').readlines()
    all_sequences = []
    current_sequence = []
    all_tags = []
    #all_words = []
    for instance in raw_data:
        if instance != '\n':
            cols = instance.split()
            if cols[1] != '_':
                if cols[3] == 'satın':
                    current_sequence.append((cols[1], 'Noun'))
                else:    
                    current_sequence.append((cols[1], cols[3]))
                    if cols[3] not in all_tags:
                        all_tags.append(cols[3])
                    # if cols[1].lower() not in all_words:
                    #   all_words.append(cols[1])   
        else:
            all_sequences.append(current_sequence)
            current_sequence = []   
    print('All tags')
    print(','.join(all_tags))
    return all_sequences, all_tags

def split_data(data, percent):
    shuffle(data)
    train_size = int(len(data) * percent / 100)
    return data[:train_size], data[train_size:]

def training(train_data, all_tags):
    tag_dict = enumerate_list(all_tags)
    distinct_words = set([word.lower() for sequence in train_data for word, tag in sequence])
    word_dict = enumerate_list(list(distinct_words))
    transition_matrix = np.zeros((len(tag_dict), len(tag_dict))) 
    #[[0 for i in range(len(tag_dict))] for j in range(len(tag_dict))]
    observations = np.zeros((len(tag_dict), len(word_dict)))
    #[[0 for j in range(len(word_dict))] for i in range(len(tag_dict))]
    initial_probability = np.zeros(len(tag_dict))
    #[0 for i in range(len(tag_dict))]
    for sequence in train_data:
        prev_tag = 'None'
        for word, tag in sequence:
            word_lower = word.lower()
            tag_id = tag_dict[tag]
            word_id = word_dict[word_lower]
            if prev_tag != 'None':
                transition_matrix[tag_dict[prev_tag], tag_id] += 1  
                prev_tag = tag
            else:
                prev_tag = tag
                initial_probability[tag_id] += 1    
            if word_id not in observations[tag_id]:
                observations[tag_id, word_id] = 1
            else:
                observations[tag_id, word_id] += 1
    return normalize(transition_matrix), initial_probability/initial_probability.sum(), normalize(observations), tag_dict, word_dict, distinct_words, all_tags



def normalize(matrix):
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]


def test(data, tag_dict, word_dict, distinct_words, distinct_tags):
    correct_sentence = 0
    wrong_sentence = 0
    correct_word = 0
    wrong_word = 0
    conf_mat = np.zeros((len(tag_dict), len(tag_dict)))
    [[0 for i in range(len(tag_dict))] for j in range(len(tag_dict)) ]
    for sequence in data:
        #print(sequence)
        actual_tags = list(map(lambda x: x[1], sequence))
        predicted_tags = viterbi(initial_prob, list(map(lambda x: word_dict.get(x[0].lower(), -1), sequence)), transition_matrix, obs)
        #print(' '.join(actual_tags))
        print(' '.join(map(lambda x: x[0], sequence)))
        print(' '.join([distinct_tags[tag] for tag in predicted_tags]))
        print(' '.join(actual_tags))
        if actual_tags == map(lambda x: distinct_tags[x], predicted_tags):
            correct_sentence += 1
        else: 
            wrong_sentence += 1
        for x in range(len(predicted_tags)):
            print(actual_tags[x])
            print(tag_dict[actual_tags[x]] )
            if tag_dict[actual_tags[x]] == predicted_tags[x]:
                print('correct')
                correct_word += 1
            else: 
                print('wrong')
                wrong_word += 1
            conf_mat[tag_dict[actual_tags[x]], predicted_tags[x]] += 1  
        # for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
        #     print('asdflksjfsd')
        #     if tag_dict[actual_tag] == predicted_tag:
        #         print('correct')
        #         correct_word += 1
        #     else: 
        #         print('wrong')
        #         wrong_word += 1
        #         conf_mat[tag_dict[actual_tag]][predicted_tag] += 1  
    print(tag_dict)
    print(correct_word )
    print(wrong_word)
    print(conf_mat)
            

def enumerate_list(data):
    return {instance: index for index, instance in enumerate(data)}

# observations
# states
# p_init initial probabilities
# sequence observations in time
# Tr transitions
# p_obs probability of seeing observation on a state

def viterbi(pi, sequence, transition_matrix, emission_prob):
    num_of_states = transition_matrix.shape[0]
    seq_len = len(sequence)
    V = np.zeros((num_of_states, seq_len))
    B = np.zeros((num_of_states, seq_len))    
    V[:, 0] = pi * emission_prob[:, sequence[0]]
    # print('initial')
    # print(V[:, 0])
    for t in range(1, seq_len):
        for s in range(num_of_states):
            # print('a')
            # print(V[:, t-1])
            # print(transition_matrix[:, s])
            # print(emission_prob[s, sequence[t]])
            result = V[:, t-1] * transition_matrix[:, s] * emission_prob[s, sequence[t]]
            # print('seq ' + str(t))
            # print('state ' + str(s))
            # print(result)
            V[s, t] = max(result)
            B[s, t] = np.argmax(result)
   
    V[:, seq_len-1] = max(V[:, seq_len-1])  
    B[:, seq_len-1] = np.argmax(V[:, seq_len-1])    

    x = np.empty(seq_len, 'B')
    x[-1] = np.argmax(V[:, seq_len - 1])
    for i in reversed(range(1, seq_len)):
        x[i - 1] = B[x[i], i]
    return x 
seed(5)
all_sequences, all_tags = get_data('../Project (Application 1) (MetuSabanci Treebank).conll')
train_data, test_data = split_data(all_sequences, 90)

transition_matrix, initial_prob, obs, tag_dict, word_dict, distinct_words, distinct_tags = training(train_data, all_tags)
#print(obs)
test(test_data, tag_dict, word_dict, distinct_words, distinct_tags)
#print(transition_matrix)
#print(initial_prob)
#print(obs)
