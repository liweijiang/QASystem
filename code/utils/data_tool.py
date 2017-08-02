# Filename: data_tool.py
# Date: 19/07/2017
# Author: Liwei Jiang
# Description: parsing tools for sample data of QA system.

import numpy as np
from os.path import join as pjoin
import logging

logging.basicConfig(level=logging.INFO)


class Config():
	def __init__(self, data_dir = 'data', data_name = 'test_sample'):
		self.answer_filename = pjoin(data_dir, data_name + '.answer')
		self.question_filename = pjoin(data_dir, data_name + '.question')
		self.context_filename = pjoin(data_dir, data_name + '.context')
		self.span_filename = pjoin(data_dir, data_name + '.span')
		self.id_context_filename = pjoin(data_dir, data_name + '.ids.context')
		self.id_question_filename = pjoin(data_dir, data_name + '.ids.question')
		self.temp_filename = pjoin(data_dir, data_name + '.tempfile')


# find the phrase given the answer span within a given paragraph
def find_phrase_given_span(context_string, answer_span):
	context_list = context_string.split()
	answer_list = [context_list[w] for w in range(answer_span[0], answer_span[1]+1)]
	predict_answer = ' '.join(answer_list)
	return predict_answer

# check if the string ifnum is a float number
def isdigit(ifnum):
    try:
        float(ifnum)
        return True
    except ValueError:
        return False

# tokenize a sentence
def add_space_between_word_char_sentence(data_sentence):
	string_modified = ''
	data_sentence_split = data_sentence.split()
	for data in data_sentence_split:
		if data == 'i.e.':
			string_modified += data
		elif data == 'e.g.':
			string_modified += data
		elif isdigit(data):
			string_modified += data
		else:
			for letter in data:
				if (   letter == '('
					or letter == ')'
					or letter == '"'
					or letter == '?'
					or letter == '!'
					or letter == ';'
					or letter == ':'):
					if len(string_modified) != 0 and string_modified[-1] != ' ':
						string_modified += ' '

				elif (letter == '.' or letter == ','):
					if len(string_modified) > 1 and not isdigit(string_modified[-1]) and string_modified[-1] != ' ':
						string_modified += ' '
				elif (len(string_modified) > 1 and letter == ' ' and string_modified[-1] == ' '):
					string_modified = string_modified[0:-1]

				else:
					if len(string_modified) > 2 and (string_modified[-1] == '(' or string_modified[-1] == '"') and string_modified[-2] != ' ':
						string_modified += ' '
					if len(string_modified) > 2 and letter == 's' and string_modified[-1] == "'" and string_modified[-2] != " ":
						string_modified = string_modified[0:-1]
						string_modified += " '"
				string_modified += letter
		string_modified += ' '
	string_modified += '\n'
	return string_modified

# manipulate, parse, and tune data
class DataTool():

	def __init__(self):
		return

	# check if the span index is correct
	def check_correctness_of_span(self, context_filename, answer_filename, span_filename):
		context_file = open(context_filename, 'r')
		answer_file = open(answer_filename, 'r')
		span_file = open(span_filename, 'r')

		context_list = context_file.readlines()
		context_list = [elem for elem in context_list if elem.strip()]
		answer_list = answer_file.readlines()
		answer_list = [elem for elem in answer_list if elem.strip()]
		span_list = span_file.readlines()
		span_list = [elem for elem in span_list if elem.strip()]

		if (len(answer_list) != len(context_list)) or (len(answer_list) != len(span_list)) or (len(span_list) != len(context_list)):
			logging.info("ERROR: the number of answer does not match the number of context")
			return

		for i in range(len(context_list)):
			context_sublist = context_list[i].split()
			context_sublist = [elem for elem in context_sublist if elem.strip()]
			answer_sublist = answer_list[i].split()
			answer_sublist = [elem for elem in answer_sublist if elem.strip()]
			span_sublist = span_list[i].split()
			span_sublist = [elem for elem in span_sublist if elem.strip()]

			if context_sublist[int(span_sublist[0])] != answer_sublist[0]:
				logging.info("ERROR 1: answer span index is not correct: line {}".format(i))
			if context_sublist[int(span_sublist[1])] != answer_sublist[-1]:
				logging.info("ERROR 2: answer span index is not correct: line {}".format(i))

		context_file.close()
		answer_file.close()
		span_file.close()

	# find the beginning and the ending index of the input phrase within a given paragraph
	def find_span_given_phrase(self, answer_string, context_string):
		start_index = 0
		end_index = 0

		answer_string = answer_string[:-1].strip()
		if context_string.find(answer_string) != -1:
			string_letter_index = context_string.find(answer_string)

			# logging.info("String letter index: {}".format(string_letter_index))
			answer_list = answer_string.split()
			context_list = context_string.split()
			if (context_string.count(answer_string) == 1):
				# logging.info('True >>> Occured time: {}'.format(str(context_string.count(answer_string[:-1]))))
				string_letter_counter = 0
				for j in range(len(context_list)):
					context_word = context_list[j]
					string_letter_counter += len(context_word) + 1
					if string_letter_counter > string_letter_index:
						start_index = j
						end_index = start_index + len(answer_list) - 1
						break
			else:
				logging.info('False >>> Occured time: {}'.format(str(context_string.count(answer_string))))
				start_index = 0
				end_index = 0
		else:
			logging.info("NOTE: answer_string {} not in context_string".format(answer_string))
			start_index = 0
			end_index = 0

		return([start_index, end_index])

	# find answer's index within a context file and generate a answer.span file
	def find_answer_span_index(self, answer_filename, context_filename, span_filename):
		answer_file = open(answer_filename)
		context_file = open(context_filename)
		span_file = open(span_filename, 'w')

		context_list = context_file.readlines()
		context_list = [elem for elem in context_list if elem.strip()]
		answer_list = answer_file.readlines()
		answer_list = [elem for elem in answer_list if elem.strip()]

		for answer in answer_list:
			if (answer == '\n'):
				answer_list.remove(answer)

		for context in context_list:
			if (context == '\n'):
				context_list.remove(context)

		if len(answer_list) != len(context_list):
			logging.info("ERROR: the number of answer does not match the number of context")
			return

		for i in range(len(answer_list)):
			span = self.find_span_given_phrase(answer_list[i], context_list[i])
			span_file.writelines(str(span[0]) + ' ' + str(span[1]) + '\n')

		answer_file.close()
		context_file.close()
		span_file.close()

	# add space between special characters and letters
	def add_space_between_word_char(self, input_filename, output_filename):
		input_file = open(input_filename, 'r')
		output_file = open(output_filename, 'w')

		dataset = input_file.readlines()
		for data_sentence in dataset:
			string_modified = add_space_between_word_char_sentence(data_sentence)
			output_file.writelines(string_modified)

		output_file.close()
		input_file.close()

if __name__ == "__main__":
	config = Config('data', 'test_sample')
	data_tool = DataTool()
	# data_tool.find_answer_span_index(config.answer_filename, config.context_filename, config.span_filename)
	data_tool.check_correctness_of_span(config.context_filename, config.answer_filename, config.span_filename)

	# data_tool.add_space_between_word_char(config.context_filename, config.temp_filename)
	# data_tool.add_space_between_word_char(config.answer_filename, config.temp_filename)
	# data_tool.add_space_between_word_char(config.question_filename, config.temp_filename)
