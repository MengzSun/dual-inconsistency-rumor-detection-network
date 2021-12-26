import csv
import os

import numpy as np
import torch
import torch.nn as nn
import re
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from transformers import BertModel, BertTokenizer, BertConfig

class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        # string = re.sub(self.other_char, " ", string)
        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\(", " \( ", string)
        # string = re.sub(r"\)", " \) ", string)
        # string = re.sub(r"\?", " \? ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr,' ',string)
        string = re.sub(r4,' ',string)
        string = string.strip().lower()
        string = self.remove_stopword(string)

        return string

    def clean_str_zh(self, string):
        # string = re.sub(self.other_char, " ", string)
        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\(", " \( ", string)
        # string = re.sub(r"\)", " \) ", string)
        # string = re.sub(r"\?", " \? ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        string = string.strip()
        string = self.remove_stopword_zh(string)
        return string

    def clean_str_BERT(self,string):
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
        r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        # string = re.sub(r1, ' ', string)
        # string = re.sub(r2, ' ', string)
        # string = re.sub(r3, ' ', string)
        string = re.sub(r4, ' ', string)
        return string

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def remove_stopword_zh(self, string):
        stopwords = []
        with open('../data/weibo/stop_words.txt', 'r', encoding='utf-8')as f:
            txt = f.readlines()
        for line in txt:
            # print(line.strip('\n'))
            stopwords.append(line.strip('\n'))

        # if self.stop_words is None:
        #     from nltk.corpus import stopwords
        #     self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = jieba.cut(string)

        new_string = list()
        for word in string:
            if word in stopwords:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

def pre_training_pheme(pathset, config):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    mids = []
    with open(pathset.path_txt_data, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[0])
            text_id.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            tweet.append(line[1].strip('\t'))
            image_id.append(line[2].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            label.append(int(line[3].strip('\t')))
            mids.append(line[4].strip('\t'))
    # print(len(text_id),len(tweet),len(image_id),len(label)) 总数据16417个
    # print(label)

    # r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
    # r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    # r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    #
    # for i in range(len(tweet)):
    #     sentence = tweet[i].split('http')[0]
    #     cleanr = re.compile('<.*?>')
    #     sentence = re.sub(cleanr, ' ', sentence)
    #     sentence = re.sub(r4, '', sentence)
    #     tweet[i] = sentence
    #     # print(sentence)
    # # print(tweet)
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)
        tokens_ids.append(tokenizer_encoding)


    mids_all = []
    for i in range(len(text_id)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))
        mids_all.append(mid)


    X_txt = tokens_ids
    X_img = np.array(image_id)

    X_kg = np.array(mids_all)

    # X = np.array(tokens_ids)
    y = np.array(label)

    return X_txt, X_img, X_kg, y


