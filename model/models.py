import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable,Function
from layers import Bert_lstm,Resnet_Encoder

from layers import *
from transformers import BertModel, BertConfig

class Scaled_Dot_Product_Attention_pos(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_pos, self).__init__()

    def forward(self, Q, K, V, scale,kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta,dim = -1)
        # print('beta size:',beta.size()) #128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)
        # print('v after attention:',context.size()) #128,1,80
        return context

class Scaled_Dot_Product_Attention_neg(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_neg, self).__init__()

    def forward(self, Q, K, V, scale, kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = -1*torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = -1*F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta, dim=-1)
        # print('beta size:', beta.size())  # 128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)
        # print('v after attention:', context.size())  # 128,1,80
        # context = torch.matmul(attention, V)
        return context

class inconsistency_model(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())


    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim,args):
        txt = self.txtenc(txt_token, txt_masks)
        img = self.imgenc(img)
        txt = F.leaky_relu(self.ln_txt(txt))
        img = F.leaky_relu(self.ln_img(img))
        txt_share = self.ln_shr(txt)
        img_share = self.ln_shr(img)
        txt_uniq = self.ln_uniq_txt(txt)
        img_uniq = self.ln_uniq_img(img)
        modal_shr = torch.cat([txt_share, img_share], -1)  # 80

        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)
        sub = txt_uniq - img_uniq
        hadmard = torch.mul(txt_share, img_share)
        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 120
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 120
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class