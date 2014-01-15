import csv
#import c4_5.py
#import operator

# define the format of attributes of csv data
format = []
for i in range(12):
    format.append("numeric")
for i in range(3,6):
    format[i] = "nominal"
#print format

def read_data(inputfile):
    csvfile = open(inputfile, 'rb')
    train_set = list(csv.reader(csvfile))
    csvfile.close()
    return train_set

# the main func
def dtree(inputfile, pruning, task):
    train_set = read_data(inputfile)
    """
    ins = C4.5(train_set)
    ins.train(ins.train_set, ins.tree, ins.attributes, 0, format)
    if task == '':

    else:
    """
    """
    train_set.sort(key=operator.itemgetter(1))
    #DTL = C4_5(train_set)
    for row in train_set:
        print ','.join(row)
    """
#    print type(train_set[1][1])
 #   print train_set[0]
 #   print len(train_set)
 #	print attributes



# test for read_data
train_set = dtree('/Users/wangkexue/Google Drive/14Winter/EECS 349/ps2/train.csv', 0, 0)

#for row in train_set:
#   print ','.join(row)
#print len(train_set)
