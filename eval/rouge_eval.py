# -*- coding: utf-8 -*-


def rouge_score(h, r):
    from rouge import Rouge
    sorted_vocab = dict([(w, str(i)) for i, w in enumerate(list(sorted(set(r + h))))])

    h = ' '.join([sorted_vocab[w] for w in h])
    r = ' '.join([sorted_vocab[w] for w in r])
    rouge = Rouge()
    scores = rouge.get_scores(h, r)[0]

    return scores


import os
result_dir = '../../results/n_gram/'
files = [result_dir + fname for fname in os.listdir(result_dir)]

for f in files:#
    print(f.replace(result_dir, ''))
    r1_scores = {'p': [], 'r':[], 'f':[]}
    r2_scores = {'p': [], 'r':[], 'f':[]}
    rl_scores = {'p': [], 'r': [], 'f': []}
    with open(f) as rf:
        lines = rf.readlines()#[:2]
        for i, line in enumerate(lines):#[:10]:
            try:

                y_hat, y = line.strip().split(' | ')
                y_hat = y_hat.strip().split()
                y = y.strip().split()
                """
                y_hat = line.replace('[','').replace(']','').replace(',','').split()
                y = open(files[0]).readlines()[i].split(' | ')[1].split()
                #print(y_hat)
                #print(y)
                """
                scores = rouge_score(y_hat, y)
                r1_scores['p'].append(scores['rouge-1']['p'])
                r1_scores['r'].append(scores['rouge-1']['r'])
                r1_scores['f'].append(scores['rouge-1']['f'])

                r2_scores['p'].append(scores['rouge-2']['p'])
                r2_scores['r'].append(scores['rouge-2']['r'])
                r2_scores['f'].append(scores['rouge-2']['f'])

                rl_scores['p'].append(scores['rouge-l']['p'])
                rl_scores['r'].append(scores['rouge-l']['r'])
                rl_scores['f'].append(scores['rouge-l']['f'])

            except:

                r1_scores['p'].append(0.0)
                r1_scores['r'].append(0.0)
                r1_scores['f'].append(0.0)

                r2_scores['p'].append(0.0)
                r2_scores['r'].append(0.0)
                r2_scores['f'].append(0.0)

                rl_scores['p'].append(0.0)
                rl_scores['r'].append(0.0)
                rl_scores['f'].append(0.0)



        r1_scores['p'] = str(round(sum(r1_scores['p']) / len(r1_scores['p']), 4))
        r1_scores['r'] = str(round(sum(r1_scores['r']) / len(r1_scores['r']), 4))
        r1_scores['f'] = str(round(sum(r1_scores['f']) / len(r1_scores['f']), 4))

        r2_scores['p'] = str(round(sum(r2_scores['p']) / len(r2_scores['p']), 4))
        r2_scores['r'] = str(round(sum(r2_scores['r']) / len(r2_scores['r']), 4))
        r2_scores['f'] = str(round(sum(r2_scores['f']) / len(r2_scores['f']), 4))

        rl_scores['p'] = str(round(sum(rl_scores['p']) / len(rl_scores['p']), 4))
        rl_scores['r'] = str(round(sum(rl_scores['r']) / len(rl_scores['r']), 4))
        rl_scores['f'] = str(round(sum(rl_scores['f']) / len(rl_scores['f']), 4))

        print(r1_scores['p']+'\t'+r1_scores['r']+'\t'+r1_scores['f']+'\t'+ \
              r2_scores['p'] + '\t' + r2_scores['r'] + '\t' + r2_scores['f']+'\t'+ \
              rl_scores['p'] + '\t' + rl_scores['r'] + '\t' + rl_scores['f']
        )



