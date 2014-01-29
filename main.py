from c4_5 import C4_5
from c4_5 import read_data
#import operator

# the main func
def dtree(inputfile, pruning, task, validfile, testfile, format):
    """A wrapper function using C4_5 for general train and test procedure
    It will print best attributes during training. Print the tree generated(
    before and after pruning). Print validate accuracy after validation and
    during pruning.    

    Args:
        inputfile: The path of input file(training data)
        pruning: whether implementing pruning on the tree(1 or 0)
        task: "validate" or "predict"
        validfile: path
        testfile: path
        format: the form of attributes, list of "nominal" and "numeric"
    """
    train_set = read_data(inputfile)
    classifier = C4_5(train_set, format, [])
    classifier.train(classifier.train_set, classifier.tree, list(classifier.attributes), 0)
    classifier.rule_generator(classifier.tree, [])
    print '------------------------'
    print 'without pruning'
    print '------------------------'
    print classifier
    f = open('unprune_rule.txt', 'wb')
    f.write(classifier.__str__())
    f.close()
    print classifier.validate(validfile)
    if pruning:
        classifier.prune(validfile)
        print '------------------------'
        print 'after pruning'
        print '------------------------'
        print classifier
    if task == 'valid':
        print classifier.validate(validfile)
    elif task == 'predict':
        classifier.predict(testfile)

# define the format of attributes of csv data
format = []
for i in range(12):
    format.append("numeric")
for i in range(3,8):
    format[i] = "nominal"
#print format
# test for read_data
inputfile = '../train.csv'
validfile = '../validate.csv'
testfile = '../test.csv'
train_set = dtree(inputfile, 1, 'predict', validfile, testfile, format)

