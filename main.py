from c4_5 import C4_5
from c4_5 import read_data
#import operator

# the main func
def dtree(inputfile, pruning, task, outfile, format):
    train_set = read_data(inputfile)
    classifier = C4_5(train_set, format, [])
    classifier.train(classifier.train_set, classifier.tree, list(classifier.attributes), 0)
    classifier.rule_generator(classifier.tree, [], 1)
    print 'without pruning'
    print classifier
    print classifier.validate(outfile)
    if pruning:
        classifier.prune(outfile)
        print 'after pruning'
        print classifier
    if task == 'valid':
        print classifier.validate(outfile)
    elif task == 'predict':
        classifier.predict(outfile)

# define the format of attributes of csv data
format = []
for i in range(12):
    format.append("numeric")
for i in range(3,6):
    format[i] = "nominal"
#print format
# test for read_data
inputfile = '/Users/wangkexue/Google Drive/14Winter/EECS 349/ps2/train.csv'
validfile = '/Users/wangkexue/Google Drive/14Winter/EECS 349/ps2/validate.csv'
testfile = '/Users/wangkexue/Google Drive/14Winter/EECS 349/ps2/test.csv'
train_set = dtree(inputfile, 1, 'validate', inputfile, format)

