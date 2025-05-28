import json
import numpy as np
import bert_model
from transformers import BertTokenizer
import torch
import re


def main():
    # create model
    print("program started ...")
    bert_encoder = bert_model.RelationClassification.from_pretrained(
        "../bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        #num_labels=768*3,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    tokenizer = get_tokenizer()
    bert_encoder.resize_token_embeddings(len(tokenizer))

    # check if entity type is available
    hasType = 1

    for i in range(0, 2):
        if i == 0:
            print("train started ...")
            sentence_train = json.load(open('retacred_mix_half/retacred_train_sentence_half.json', 'r'))
            sentence_train_label = json.load(open('retacred_mix_half/retacred_train_label_half.json', 'r'))

            rel_representations = get_rel_representations(bert_encoder, sentence_train, sentence_train_label, hasType)
            with open('retacred_train_half_rel_representations.json', 'w') as outF:
                json.dump(rel_representations, outF)

        # commented when doing experiment
        else:
            print("test started ...")
            sentence_train = json.load(open('retacred_mix_half/retacred_test_sentence_half.json', 'r'))
            sentence_train_label = json.load(open('retacred_mix_half/retacred_test_label_half.json', 'r'))

            rel_representations = get_rel_representations(bert_encoder, sentence_train, sentence_train_label, hasType)
            with open('retacred_test_half_rel_representations.json', 'w') as outF:
                json.dump(rel_representations, outF)



def get_rel_representations(bert_encoder, sentence_train, sentence_train_label, type):
    input_ids, attention_masks, labels, e1_pos, e2_pos, index_arr, rm_lst = pre_processing(sentence_train, sentence_train_label, type)
    print(rm_lst)
    rel_representations = []
    for idx, (input_ids_i, attention_masks_i, e1_pos_i, e2_pos_i) in enumerate(
            zip(input_ids, attention_masks, e1_pos, e2_pos)):
        if idx % 600 == 0:
            print(str(idx))
        input_ids_i = input_ids_i.unsqueeze(0)
        attention_masks_i = attention_masks_i.unsqueeze(0)
        e1_pos_i = e1_pos_i.unsqueeze(0)
        e2_pos_i = e2_pos_i.unsqueeze(0)
        output = bert_encoder(input_ids=input_ids_i, attention_mask=attention_masks_i, e1_pos=e1_pos_i, e2_pos=e2_pos_i)
        output = output.squeeze(0)
        rel_representations.append(output.detach().numpy().tolist())
    return rel_representations


def get_tokenizer():
    """ Tokenize all the sentences and map the tokens to their word ids."""
    tokenizer = BertTokenizer.from_pretrained('../co-training/bert-base-uncased', do_lower_case=True)
    special_tokens = []
    special_tokens.append('<e1>')
    special_tokens.append('</e1>')
    special_tokens.append('<e2>')
    special_tokens.append('</e2>')

    ent_type = ['PERSON', 'ORGANIZATION', 'NUMBER', 'DATE', 'NATIONALITY', 'LOCATION', 'TITLE', 'CITY', 'MISC',
                'COUNTRY', 'CRIMINAL_CHARGE', 'RELIGION', 'DURATION', 'URL', 'STATE_OR_PROVINCE', 'IDEOLOGY',
                'CAUSE_OF_DEATH']  # TACRED
    for r in ent_type:
        special_tokens.append('<e1:' + r + '>')
        special_tokens.append('<e2:' + r + '>')
        special_tokens.append('</e1:' + r + '>')
        special_tokens.append('</e2:' + r + '>')

    special_tokens_dict ={'additional_special_tokens': special_tokens }    # add special token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_token_word_id(sen, tokenizer, type):
    """ Get special token word id """
    if not type:
        # return 2487,2475
        e1 = '<e1>'
        e2 = '<e2>'
    else:
        e1 = re.search('(<e1:.*?>)', sen).group(1)
        e2 = re.search('(<e2:.*?>)', sen).group(1)
        #e1e = re.search('(</e1:.*?>)', sen).group(1)
        #e2e = re.search('(</e2:.*?>)', sen).group(1)
    e1_tks_id = tokenizer.convert_tokens_to_ids(e1)
    e2_tks_id = tokenizer.convert_tokens_to_ids(e2)
    #e1e_id = tokenizer.convert_tokens_to_ids(e1e)
    #e2e_id = tokenizer.convert_tokens_to_ids(e2e)
    return e1_tks_id, e2_tks_id
    pass



def pre_processing(sentence_train, sentence_train_label, type):
    """Main function for pre-processing data """
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    index_arr = []
    exp_lst = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = get_tokenizer()
    counter = 0

    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            truncation=True,        # explicitely truncate examples to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        try:
            # Find e1(id:2487) and e2(id:2475) position
            e1_tks_id, e2_tks_id = get_token_word_id(sentence_train[i], tokenizer, type)
            pos1 = (encoded_dict['input_ids'] == e1_tks_id).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == e2_tks_id).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            index_arr.append(counter)
            counter += 1

        except Exception as e:
            print(i)
            exp_lst.append(i)
            print(e)
            pass

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)
    index_arr = torch.tensor(index_arr)
    print(input_ids.size())
    print(input_ids.size(0))
    print()

    return input_ids, attention_masks, labels, e1_pos, e2_pos, index_arr, exp_lst



if __name__ == '__main__':
    main()



