from scipy.stats import mode
import math
import operator

class C4_5:
	def __init__(self, train_set):
		self.tree = node();
		"""store the headerline in attributes
	    store the data in train_set 
	    """
        self.attributes = train_set[0][:-1]
        self.train_set = train_set[1:]
        self.data_len = len(self.train_set)
        """it's a recursive method, 
        when first call it, the input tree should be self.tree
        attributes the same 
        examples should be self.train_set use list comprehension to get col
        maybe : operand is more optimized, but they're both O(n) in principle
        zip(*train_set)
        """
	def train(self, examples, tree, attributes, default, format):
		# when no examples return default
		if len(examples) == 0:
			return default
		# if all examples have the same classification return the classification
		elif len(unique([item[-1] for item in examples])) == 1:
			return examples[0][-1]
		# when attributes is empty then return mode(examples)
		elif len(attributes) == 0:
             return mode([item[-1] for item in examples])[0][0] 
		else:
			best = self.choose_attr(self, attributes, examples, format)
            tree.val = best[0]
            attributes.remove(best[0])
            idx = self.attributes.index(best[0])
            # for nominal attr
            if best[1] == 'nom':
                for v in unique([item[idx] for item in examples]):
              	    exp = [item for item in examples if item[idx] == v]
            	    subtree = self.train(exp, node(), attributes, mode([item[-1] for item in examples])[0][0], format)
            	    branch = node(best[0], '=='+v, [subtree])
            	    tree.child.append(branch)
            # for numberic attr
            else:
            	# >
            	exp = [item for item in examples if item[idx] > best[2]]
            	default = mode([item[-1] for item in examples])[0][0]
            	subtree1 = self.train(exp1, node(), attributes, default, format)
            	branch1 = node(best[0], '>'+thresh, [subtree1])
            	tree.child.append(branch1)
            	# <=
            	exp = [item for item in examples if item[idx] <= best[2]]
            	subtree2 = self.train(exp2, node(), attributes, default, format)
            	branch2 = node(best[0], '<='+thresh, [subtree2])
            	tree.child.append(branch2)
            	# TO DO rethink about the interface
            # TO DO how to print dtree?
            # TO DO how to deal with numeric attr?
            return tree

	def choose_attr(self, attributes, examples, format):
        min_rem = 1
        for attr in attributes:
        	idx = self.attributes.index(attr)
        	#data_len = len(examples[0])
        	if format[idx] == 'nominal':
        		rem = 0
                for i in unique([item[idx] for item in examples]):
                	exp = [item[-1] for item in examples if item[idx] == i]
                	data_len = len(exp)
                	rem += data_len/self.data_len * entropy([ exp.count(1)/data_len, exp.count(0)/data_len ]) 
                if rem < min_rem:
                	min_rem = rem
                	best = [attr, 'nom']
        	else:    # numberic
        		# order by numberic attr
        		exps = list(examples)
        		exps.sort(key=operator.itemgetter(idx))
        		# get split point when adjcent class and val are different
                split_candidate = self.split(exps, idx)
                rem = 0
                for thresh in split_candidate:
                	exp1 = [item[-1] for item in exps if item[idx] > thresh]
                	exp2 = [item[-1] for item in exps if item[idx] <= thresh]
                	len1 = len(exp1)
                	len2 = len(exp2)
                    rem = len1/self.data_len * entropy([ exp1.count(1)/len1, exp1.count(0)/len1 ])
                    rem += len2/self.data_len * entropy([ exp2.count(1)/len2, exp2.count(0)/len2 ])
                    if rem < min_rem:
                        min_rem = rem
                        best = [attr, 'num', thresh]
		return best

	def split(lst, idx):
		split_candidate = []
		for x, y in zip(lst, lst[1:]):
			if x[idx] != y[idx] and x[-1] != y[-1]
			    split_candidate.append(x[idx])
	    return split_candidate

	def validate(self, valid_set):
	def predict(self, test_set):
	def pruning(self):
	def __str__(self):

class node(object):
	def __init__(self, val, label, child):
		self.val = val
		self.lable = self.label
		self.child = child
	def __str__(self):
		print self.val, self.child

def entropy(lst):
        entrop = 0
        for p in lst:
        	entrop += -p * math.log(p, 2)
		return entrop
"""produce the list of unique values of a list
"""
def unique(seq):
    keys = {}
    for e in seq:
    	keys[e] = 1
    return keys.keys()


