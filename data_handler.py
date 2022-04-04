# vocab:
# 	word to index
# 	index to word
# labels:
# 	class to index
# transform data:
# 	input data
# 	target data

import pandas as pd
from pandas import DataFrame as df
from numba import jit

vocab_to_idx={}
idx_to_vocab={}
label_to_idx={}
idx_to_label={}
datafiles=["extracted_data/dev.txt", "extracted_data/train.txt"]
outfile_loc="finData/daaata.csv"
labels_loc="finData/labeaaaals.csv"

# @jit
def treat(datafiles):

	maxlen=0

	with open(outfile_loc, 'w') as data_file:

		with open(labels_loc, 'w') as labels_file:
			
			for tmp_f in datafiles:

				curline=[]
				curlabels=[]
			
				with open(tmp_f, 'r', encoding="utf-8") as f:

					print(tmp_f)
			
					for line in f.readlines():

						line=line.strip("\n")

						if line=="":
							if len(curline)!=0:
								maxlen=max(maxlen, len(curline))
								data_file.write(",".join(list(map(str, curline)))+"\n")
								labels_file.write(", ".join(list(map(str, curlabels)))+"\n")
								curline=[]
								curlabels=[]
							continue

						a, b, word, label=line.split(" ")

						if word not in vocab_to_idx:
							i=len(vocab_to_idx)+1
							vocab_to_idx[word]=i
							idx_to_vocab[i]=word

						if label not in label_to_idx:
							i=len(label_to_idx)
							label_to_idx[label]=i
							idx_to_label[i]=label

						curline.append(vocab_to_idx[word])
						curlabels.append(label_to_idx[label])
						# if word==".":
						# 	o.write(",".join(list(map(str, curline)))+"\n")
						# 	curline=[]

	df1, df2=df(), df()
	df1["indices"]=list(idx_to_vocab.keys())
	df1["words"]=[idx_to_vocab[i] for i in df1["indices"]]

	df2["indices"]=list(idx_to_label.keys())
	df2["labels"]=[idx_to_label[i] for i in df2["indices"]]

	df1.to_csv("vocab_map.csv", index=False)
	df2.to_csv("label_map.csv", index=False)

	print("The maximum sequence lenght is: {}".format(maxlen))
	return

treat(datafiles)