
readme.md
readme.md_
Package discription
./coding_test/

./data/
<train_source.txt>
<train_target.txt>
<test_source.txt>
<test_target.txt>
<new_train_target.txt>: filled missing data with 37-gram model generated sequences
./eval/
<rouge_eval.py>
./probs/: n-gram probability files stored (obtained only using <train_source.txt> & <train_target.txt>)
./results/
Result file consists of lines containing ' | '. The sequence before ' | ' is the predicted sequence, and the sequence after ' | ' is the correct output sequence.
./ngram/: n-gram generated sequence results
./nn/: neural net generated sequences
file name contains 'bi_lstm': bi_directional lstm model results
file name contains 'full_data': used <new_train_target.txt> data in training
file name contians 'simple_lstm': simple lstm model results
./nn_models/: neural net checkpoint files stored
./src/
<ngram_prob.py>: model ngram probabilites
<ngram_gen.py>: use probability file stroed in <./probs/> to model target sequence
<seq2seq-lstm.py>: encoder-decoder model with simple lstm layer
<seq2seq-bi_dir-lstm.py>: encoder-decoder model using bi-directional lstm layer
Models
N-gram 대체 https://drive.google.com/file/d/1mtTb3saso5_F3YwBcGeV3ArxbqKtpdl1/view?usp=sharing

Encoder-Decoder Model https://drive.google.com/file/d/1_eR6c03AR6MdzlLgdQDqLAknBOCsQ7lA/view?usp=sharing

Bidirectional Encoder-Decoder Model https://drive.google.com/file/d/1l82PWgmwRfBgFMd-Q3u-gLqIiXosgaTD/view?usp=sharing

Results
1. n-gram generator
rouge-1			rouge-2			rouge-l		
p	r	f	p	r	f	p	r	f
3-gram	0.2207	0.2187	0.2126	0.0794	0.0916	0.0817	0.1845	0.1837	0.1692
5-gram	0.2234	0.2453	0.2201	0.103	0.1026	0.0967	0.1978	0.2198	0.1813
7-gram	0.2743	0.356	0.2812	0.1575	0.1963	0.1537	0.2512	0.3303	0.2344
13-gram	0.4163	0.6301	0.4488	0.3101	0.5074	0.3289	0.4033	0.6151	0.3931
15-gram	0.4403	0.6907	0.4806	0.3297	0.5744	0.3565	0.4279	0.6768	0.4207
17-gram	0.435	0.6679	0.4706	0.3302	0.5495	0.3526	0.4218	0.6529	0.413
23-gram	0.4457	0.7156	0.4874	0.3361	0.6005	0.364	0.4325	0.7009	0.4258
25-gram	0.4472	0.7172	0.4882	0.3377	0.6022	0.3647	0.434	0.7026	0.4265
27-gram	0.4524	0.7184	0.4929	0.3429	0.6034	0.3698	0.4391	0.7037	0.4319
33-gram	0.4519	0.7203	0.4935	0.342	0.6058	0.3701	0.4399	0.7071	0.4329
35-gram	0.4461	0.6874	0.4845	0.3404	0.5729	0.3659	0.4343	0.674	0.4278
37-gram	0.4533	0.721	0.495	0.3428	0.6059	0.371	0.4411	0.7075	0.4342
43-gram	0.4335	0.6872	0.4736	0.3274	0.5716	0.3532	0.4211	0.6728	0.414
45-gram	0.4471	0.6867	0.4852	0.3409	0.5723	0.3664	0.4352	0.6733	0.4285
47-gram	0.4362	0.6856	0.4757	0.3298	0.5709	0.3556	0.4239	0.6716	0.4166
Simple LSTM Model vs. Bi-directional LSTM Model
