def ngram(s,n=[]):
    subs = []

    for i in n:
        grams = []
        for j in range(len(s)-i+1):
            grams.append(s[j:j+i])
        subs += grams
    return subs

from collections import defaultdict
trxf = './data/train_source.txt'
tryf = './data/train_target.txt'
xlines = ['start ' + line.strip() for line in open(trxf).readlines()]
ylines = [line.strip() +' end' for line in open(tryf).readlines()]

from nltk import ConditionalFreqDist,ConditionalProbDist, MLEProbDist
import pickle

max_in_len = max([line.count(' ') for line in xlines[:len(ylines)]])
max_out_len = max([line.count(' ') for line in ylines])

print(max_in_len, max_out_len)
max_n = 1
lines = []
for i, line in enumerate(xlines[:len(ylines)]):
    lines.append(xlines[i]+' '+ylines[i])
#print(lines[0].split())

ngrams = []
for line in lines:
    tokens = line.split()
    try:
        ngrams += ngram(tokens, list(range(2, max_n+1)))
    except:
        print(tokens, len(tokens))
print(ngrams[0])
print(ngrams[-1])
print(len(ngrams))
cfd = ConditionalFreqDist([(' '.join(t[:-1]), t[-1]) for t in ngrams])
print(dict(cfd['start 601']))
with open('probs/'+str(max_n)+'-gram_freq.pkl', 'wb') as pf:
    pickle.dump(cfd, pf)

