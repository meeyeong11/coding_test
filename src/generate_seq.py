
import pickle
import datetime


log_lines = []
for z in [17]:

    log_lines.append('max n\t'+str(z))
    print('max n\t'+str(z))
    log_lines.append(str(datetime.datetime.now()))
    print(str(datetime.datetime.now()))


    max_n = z

    cfd = './probs/'+str(max_n)+'-gram_freq.pkl'

    cfd = pickle.load(open(cfd, 'rb'))


    trxf = './data/train_source.txt'
    tstxf = './data/test_source.txt'
    tstyf = './data/test_target.txt'
    """
    missing_start = len(open('./data/train_target.txt').readlines())
    xlines = open(trxf).readlines()[missing_start:]
    """
    #data_pairs = dict([[xline,yline] for xline, yline in zip(open(tstxf).readlines(), open(tstyf).readlines())])
    data_pairs = [[xline.strip(), yline.strip()] for xline, yline in zip(open(tstxf).readlines(), open(tstyf).readlines())]

    max_out_len = 47

    y_lines = []
    for di, (line, y) in enumerate(data_pairs):
    #for di, line in enumerate(xlines):
        in_seq = line.split()
        #print(in_seq)
        y_hat = []
        if di %50 ==0:
            print(di, str(datetime.datetime.now()))
        else:
            print('=====',di, )

        while True:
            out_tok = ''
            _max_n = max_n
            if max_n > len(in_seq):
                _max_n = len(in_seq)
            for i in range(1,_max_n):

                if len(in_seq)>= _max_n-i:
                    #print(in_seq[-max_out_len+i:])
                    #print(in_seq[-max_n+i:])
                    next_tok = cfd[' '.join(in_seq[-_max_n+i:])].most_common(1)


                    if len(next_tok)>0:
                        out_tok = next_tok[0][0]
                        in_seq.append(out_tok)
                        y_hat.append(out_tok)
                        break


            if len(y_hat) >=max_out_len:
                break
            if len(y_hat) ==0:
                y_hat = ['']
                break
            #print([out_tok])
            if out_tok == 'end':
                y_hat = y_hat[:-1]
                break
        y_lines.append(' '.join(y_hat)+' | '+y+'\n')

    with open('./results/n_gram/'+str(max_n)+'_gram.res', 'w') as wf:
        wf.writelines(y_lines)
    print(str(datetime.datetime.now()))
    log_lines.append(str(datetime.datetime.now()))

    print('\n'.join(log_lines))

    with open('./logs/ngram_lm-2.log','w') as wf:
        wf.write('\n'.join(log_lines))
