#from scipy import mode 
# coding=utf-8
import csv
import math
import operator
import copy
#import numpy as np
from collections import Counter

class node(object):
	"""A class for decision tree structure

	Attributes:
		val: value of the node, may be list to store condition
		child: list of subtree
	"""
	def __init__(self, val, child):
		self.val = val
		self.child = child
	def __str__(self):
		return 'val:'+str(self.val) +'\n'+'child:'+ str(self.child)

class C4_5(object):
	"""A class for all the implementation of c4.5 dtree algorithm

	Attributes:
		tree: store the decision tree as tree structure
		format: the form of attributes, i.e. ['numeric', 'nominal', 'nominal']
		attributes: a list of attributes of the input data
		train_set: the input training data
		data_len: the length of train_set
		rule: the generated rule
	"""

	def __init__(self, train_set, format, rule):
		"""Initialize C4_5 with training data, format, rule(can be empty)"""
		self.tree = node(None, [])
		"""store the headerline in attributes
	    store the data in train_set 
	    """
		train_set = list(train_set)
		self.format = format
		self.attributes = train_set[0][:-1]
		self.train_set = train_set[1:]
		self.data_len = len(self.train_set)
		self.rule = rule
		"""it's a recursive method,
	    when first call it, the input tree should be self.tree
	    attributes the same 
	    examples should be self.train_set use list comprehension to get col
	    maybe : operand is more optimized, but they're both O(n) in principle
	    zip(*train_set
	    """
	def train(self, examples, tree, attributes, default):
		# when no examples return default
		if len(examples) == 0:
			#print 'A'
			return default
		# if all examples have the same classification return the classification
		elif allequal([item[-1] for item in examples]):
			#print 'B'
			return examples[0][-1]
		# when attributes is empty then return mode(examples)
		elif len(attributes) == 0 or len(examples) < 100:
			#print 'C'
			return mode([item[-1] for item in examples])[0][0]
		else:
			best = self.choose_attr(attributes, examples)
			if best == 0:
				return mode([item[-1] for item in examples])[0][0]
			print best
			tree.val = best[0]
			#attributes.remove(best[0])
			idx = self.attributes.index(best[0])
			if best[1] == 'nom':
				attributes.remove(best[0])
				for v in unique([item[idx] for item in examples]):
					exp = [item for item in examples if item[idx] == v]
					subtree = self.train(exp, node(None, []), list(attributes), mode([item[-1] for item in examples])[0][0])
					branch = node([best[0], '==', v], [subtree])
					tree.child.append(branch)
			else:
				#print 'A'
				#print 'examples: '+ str(len(examples))
				exp1 = [item for item in examples if float(item[idx]) > best[2]]
				#print 'exp1: '+str(len(exp1))
				default = mode([item[-1] for item in examples])[0][0]
				if len(exp1) == len(examples):
					return default
				subtree1 = self.train(exp1, node(None, []), list(attributes), default)
				branch1 = node([best[0], '>', str(best[2])], [subtree1])
				tree.child.append(branch1)
				exp2 = [item for item in examples if float(item[idx]) <= best[2]]
				#print 'exp2: '+str(len(exp2))
				subtree2 = self.train(exp2, node(None, []), list(attributes), default)
				branch2 = node([best[0], '<=', str(best[2])], [subtree2])
				tree.child.append(branch2)
			return tree
          
   	def choose_attr(self, attributes, examples):
   		max_gain = -10
   		data_len = float(len(examples))
   		group = [item[-1] for item in examples]
   		infor = entropy([group.count('1')/data_len, group.count('0')/data_len])
   		for attr in attributes:
   			idx = self.attributes.index(attr)
   			gain = infor
   			if self.format[idx] == 'nominal':
   				mode_exp = mode([item[idx] for item in examples])[0][0]
   				#mode_0 = mode([item[idx] for item in examples if item[-1]=='0'])[0][0]
   				for item in examples:
   					if item[idx] == '?':
   						item[idx] = mode_exp
   				for i in unique([item[idx] for item in examples]):
   					exp = [item[-1] for item in examples if item[idx] == i]
   					split_len = float(len(exp))
   					gain -= split_len/data_len * entropy([ exp.count('1')/split_len, exp.count('0')/split_len ])
   				if gain > max_gain:
   					max_gain = gain
   					best = [attr, 'nom']
   			else:
   				num_lst = [float(item[idx]) for item in examples if item[idx] != '?']
   				#num_lst_0 = [float(item[idx]) for item in examples if item[idx] != '?' and item[-1]=='0']
   				mean = sum(num_lst) / len(num_lst)
   				#mean_0 = sum(num_lst_0) / len(num_lst_0)
   				#mode_exp = float(mode([item[idx] for item in examples])[0][0])
   				exps = list(examples)
   				for item in exps:
   					if item[idx] == '?':
   						item[idx] = mean
   					else:
   						item[idx] = float(item[idx])
   				#exps = list(examples)
   				#for item in exps:
   				#	item[idx] = float(item[idx])
   				exps.sort(key=operator.itemgetter(idx))
   				split_candidate = self.split(exps, idx)
   				for thresh in split_candidate:
   					#print '*************************************'
   					#print thresh
   					gain = infor
   					exp1 = [item[-1] for item in exps if float(item[idx]) > thresh]
   					exp2 = [item[-1] for item in exps if float(item[idx]) <= thresh]
   					len1 = float(len(exp1))
   					len2 = float(len(exp2))
   					if len1==0 or len2==0:
   						gain = 0
   					else:
   						gain -= len1/data_len * entropy([ exp1.count('1')/len1, exp1.count('0')/len1 ])
   						gain -= len2/data_len * entropy([ exp2.count('1')/len2, exp2.count('0')/len2 ])
   					if gain > max_gain:
   						max_gain = gain
   						best = [attr, 'num', thresh]
   		print max_gain
   		if max_gain <= 0:
   			return 0
   		return best

   	
   	def rep_miss(self, examples):
   		exp = copy.deepcopy(examples)
   		for attr in self.attributes:
   			idx = self.attributes.index(attr)
   			if self.format[idx] == 'nominal':
   				mode_exp = mode([item[idx] for item in exp])[0][0]
   				#mode_0 = mode([item[idx] for item in exp if item[-1]=='0'])[0][0]
   				for item in exp:
   					if item[idx] == '?':
   						item[idx] = mode_exp
   			else:
   				num_lst = [float(item[idx]) for item in exp if item[idx] != '?']
   				#num_lst_0 = [float(item[idx]) for item in exp if item[idx] != '?' and item[-1] == '0']
   				mean = sum(num_lst) / len(num_lst)
   				#mean_0 = sum(num_lst_0) / len(num_lst_0)
   				#print mean
   				#mode_exp = mode([item[idx] for item in exp])[0][0]
   				for item in exp:
   					if item[idx] == '?':
   						item[idx] = mean
   		return exp
   		
    
  

	def split(self, lst, idx):
		split_candidate = []
		for x, y in zip(lst, lst[1:]):
			if (x[-1] != y[-1]) and (x[idx] != y[idx]):
				split_candidate.append( (x[idx] + y[idx]) / 2 )
		return split_candidate
	"""
	def rep_miss(self, examples):
		exp = copy.deepcopy(examples)
		for attr in self.attributes:
			idx = self.attributes.index(attr)
			mode_1 = mode([item[idx] for item in examples if item[-1] == '1'])[0][0]
			mode_0 = mode([item[idx] for item in examples if item[-1] == '0'])[0][0]
			for item in exp:
				if item[idx] == '?':
					if item[-1] == '1':
						item[idx] = mode_1
					else:
						item[idx] = mode_0
		return exp
    """
	def validate(self, valid_file):
		valid_set = list(read_data(valid_file))
		valid_data = valid_set[1:]
		acc = 0.0
		exp = self.rep_miss(valid_data)
		#print exp
		for sample in exp:
			rlt = self.test(sample)
			#print rlt
			#print 'rlt: '+rlt
			#print 'smp: '+sample[-1]
			if rlt == sample[-1]:
				acc += 1.0
		return acc/len(valid_data)

	def preprocess(self, valid_file):
		valid_set = list(read_data(valid_file))
		valid_data = valid_set[1:]
		exp = self.rep_miss(valid_data)
		return exp

	def valid_prune(self, exp):
		acc = 0
		for sample in exp:
			if self.test(sample) == sample[-1]:
				acc += 1
		return float(acc)/len(exp)

	def predict(self, test_file):
		test_set = read_data(test_file)
		test_data = test_set[1:]
		exp = self.rep_miss(test_data)
		for i in range(0, len(exp)):
			rlt = self.test(exp[i])
			test_data[i].append(rlt)
		header = test_set[0]
		header.append('predict '+header[-1])
		test_data.insert(0, header)
		with open('outfile.csv', 'wb') as f:
			writer = csv.writer(f)
			writer.writerows(test_data)

	
	def fast_test(self, tree, sample):
		if tree.child:
			if tree.child[0] == '1':
				return '1'
			elif tree.child[0] == '0':
				return '0'
			else:
				for item in tree.child:
					if isinstance(item.val, list):
						idx = self.attributes.index(item.val[0])
						attr = sample[idx]
						if self.format[idx] == 'nominal':
							if attr == item.val[2]:
								return self.fast_test(item, sample)
						else:
							if item.val[1] == '>':
								if float(attr) > float(item.val[2]):
									return self.fast_test(item, sample)
							else:
								if float(attr) <= float(item.val[2]):
									return self.fast_test(item, sample)
					else:
						return self.fast_test(item, sample)
    


	def prune(self, valid_file):
		exp = self.preprocess(valid_file)
		for single_rule in self.rule:
			before = self.valid_prune(exp)
			old_rule = copy.deepcopy(self.rule)
			self.rule.remove(single_rule)
			after = self.valid_prune(exp)
			print before
			print after
			#if after > before:
			#	self.prune(valid_file)
			if after < before:
				self.rule = old_rule

	def test(self, sample):
		#print '.'
		for single_rule in self.rule:
			for stump in single_rule:
				if stump[0] == 'class':
					return stump[2]
				idx = self.attributes.index(stump[0])
				attr = sample[idx]
				if self.format[idx] == 'nominal':
					if attr == stump[2]:
						pass
					else:
						break
				else:
					if stump[1] == '>':
						if float(attr) > float(stump[2]):
							pass
						else:
							break
					else:
						if float(attr) <= float(stump[2]):
							pass
						else:
							break
		return '0'
	

	def __str__(self):
		#print self.rule
		rule_str = []
		for single_rule in self.rule:
			single_rule_str = []
			for item in single_rule:
				 single_rule_str.append('('+' '.join(item)+')')
			rule_str.append('('+' ∨ '.join(single_rule_str)+')')
		return ' ∧ '.join(rule_str)

	def rule_generator(self, tree, single_rule):
		#if flag:
		#	self.rule = []
		#print tree
		if tree.child:
			if isinstance(tree.val, list):
				single_rule.append(tree.val)
			if tree.child[0] == '1':
				single_rule.append(['class', '=', '1'])
				self.rule.append(single_rule)
			elif tree.child[0] == '0':
				single_rule.append(['class', '=', '0'])
				self.rule.append(single_rule)
			else:
				for item in tree.child:
					self.rule_generator(item, list(single_rule))

	"""
	def rule_generator(self, tree, rule_str, rule):
		if tree.child:
			if isinstance(tree.val, list):
				rule_str.append(''.join(tree.val))
			if tree.child[0] == str(1):
				rule.append('('+' AND '.join(rule_str)+')')
			elif tree.child[0] == str(0):
				pass
			else:
				for item in tree.child:
					self.rule_generator(item, list(rule_str), rule)
		return ' OR '.join(rule)
    """

def read_data(inputfile):
    csvfile = open(inputfile, 'rb')
    data = list(csv.reader(csvfile))
    csvfile.close()
    return data

def entropy(lst):
	if lst.count(0):
		return 0
	entrop = 0
	for p in lst:
		entrop -= p * math.log(p, 2)
	return entrop
"""produce the list of unique values of a list
"""
def unique(seq):
    keys = {}
    for e in seq:
    	keys[e] = 1
    return keys.keys()

def allequal(seq):
	flag = seq[0]
	for item in seq:
		if item != flag:
			return 0
	return 1

def mode(lst):
	frequent = Counter(lst)
	mostfrequent = frequent.most_common(2)
	if mostfrequent[0][0] == '?':
		mostfrequent = mostfrequent[1:]
	return mostfrequent


