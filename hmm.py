#!/usr/bin/python
from random import shuffle

def get_data(filepath):
	raw_data = open(filepath).readlines()
	all_sequences = []
	current_sequence = []
	all_tags = []
	#all_words = []
	for instance in raw_data:
		if instance != '\n':
			cols = instance.split()
			if cols[1] != '_':
				current_sequence.append((cols[1], cols[3]))
				if cols[3] not in all_tags:
					all_tags.append(cols[3])
				# if cols[1].lower() not in all_words:
				# 	all_words.append(cols[1])	
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
	transition_matrix = [[0 for i in range(len(tag_dict))] for j in range(len(tag_dict))]
	observations = [{} for i in range(len(tag_dict))]
	initial_probability = [0 for i in range(len(tag_dict))]
	for sequence in train_data:
		prev_tag = 'None'
		for word, tag in sequence:
			word_lower = word.lower()
			tag_id = tag_dict[tag]
			word_id = word_dict[word_lower]
			if prev_tag != 'None':
				transition_matrix[tag_dict[prev_tag]][tag_id] += 1	
				prev_tag = tag
			else:
				prev_tag = tag
				initial_probability[tag_id] += 1 	
			if word_id not in observations[tag_id]:
				observations[tag_id][word_id] = 1
			else:
				observations[tag_id][word_id] += 1
	normalized_obs = [normalize_dict(output_probs) for output_probs in observations]			
	return normalize(transition_matrix, 2), normalize(initial_probability), normalized_obs, tag_dict, word_dict


def normalize_dict(d):
	return {key:value/sum(d.values()) for key,value in d.items()}

def normalize(matrix, dim = 1):
	if dim == 2:
		return [[float(i)/sum(row) for i in row] for row in matrix]
	else:	
		return [float(i)/sum(matrix) for i in matrix]


def test(data, tag_dict, word_dict): 
	correct_sentence = 0
	wrong_sentence = 0
	correct_word = 0
	wrong_word = 0
	conf_mat = [[0 for i in range(len(tag_dict))] for j in range(len(tag_dict)) ]
	for sequence in data:
		actual_tags = map(lambda x: x[1], sequence)
		predicted_tags = viterbi(map(lambda x: word_dict[x[0]], sequence))
		if actual_tags == predicted_tags:
			correct_sentence += 1
		else: 
			wrong_sentence += 1
		for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
			if actual_tag == predicted_tag:
				correct_word += 1
			else: 
				wrong_word += 1
				conf_mat[tag_dict[actual_tag]][tag_dict[predicted_tag]] += 1	
			

def enumerate_list(data):
	return {instance: index for index, instance in enumerate(data)}

all_sequences, all_tags = get_data('Project (Application 1) (MetuSabanci Treebank).conll')
train_data, test_data = split_data(all_sequences, 90)
transition_matrix, initial_prob, obs, tag_dict, word_dict = training(train_data, all_tags)
test(test_data, tag_dict, word_dict)
#print(transition_matrix)
#print(initial_prob)
#print(obs)
