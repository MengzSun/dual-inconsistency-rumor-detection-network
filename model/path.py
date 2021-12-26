import os
import sys

class path_set_BERT():
    def __init__(self,dataset):
        self.path_data_dir = os.path.join('..','data/{}'.format(dataset))
        #text\image\label
        if dataset == 'twitter':
            self.path_txt_data_train = os.path.join(self.path_data_dir,'en_train.csv')
            self.path_txt_data_test = os.path.join(self.path_data_dir,'en_test.csv')
            self.path_img_data = os.path.join(self.path_data_dir,'/images/')
        else:
            self.path_txt_data = os.path.join(self.path_data_dir,'pheme_final_fb.csv')
            self.path_img_data = os.path.join(self.path_data_dir,'/images/')


        #BERT_PATH
        self.path_bert = '../bert-base-uncased/'
        self.VOCAB = 'vocab.txt'
        #TransE path
        self.path_transe = '../Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        self.path_dic = '../Freebase/knowledge graphs/entity2id.txt'