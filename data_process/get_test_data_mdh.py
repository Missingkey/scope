from tokenizers import BertWordPieceTokenizer
from data_process.get_train_data import hanzi2pinyin
import pickle
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--text", required=True, type=str, help="请输入要改错的文本")
    return parser

def get_test_data_mdh(vocab_path):
    parser = get_parser()
    args = parser.parse_args()
    #TODO 参数：data_path, vocab_path, max_len
    data = []
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = True)
    item_raw = ['114514']
    item_raw.append(args.text)
    item_raw.append(args.text)
    # Field: id, src, tgt
    item = {
        'id': item_raw[0],
        'src': item_raw[1],
        'tgt': item_raw[2],
    }
    assert len(item['src']) == len(item['tgt'])
    data.append(item)

    # Field: tokens_size
    encoded = tokenizer.encode(item['src'])
    tokens = encoded.tokens[1:-1]
    tokens_size = []
    for t in tokens:
        if t == '[UNK]':
            tokens_size.append(1)
        elif t.startswith('##'):
            tokens_size.append(len(t) - 2)
        else:
            tokens_size.append(len(t))
    item['tokens_size'] = tokens_size

    # Field: src_idx
    item['src_idx'] = encoded.ids
    token2pinyin = hanzi2pinyin('/home/mdh19/004_csc_sl/SCOPE/FPT')
    # item['pinyin_ids'] = token2pinyin.convert_sentence_to_pinyin_ids(item['src'], encoded)

    # Field: tgt_idx
    encoded = tokenizer.encode(item['tgt'])
    item['tgt_idx'] = encoded.ids
    # item['tgt_pinyin_ids'] = token2pinyin.convert_sentence_to_pinyin_ids(item['tgt'], encoded)
    # item['pinyin_label'] = token2pinyin.convert_sentence_to_shengmu_yunmu_shengdiao_ids(item['tgt'], encoded)
    assert len(item['src_idx']) == len(item['tgt_idx'])
    lengths = len(item['src'])
    item['lengths'] = lengths

    return data


if __name__ == '__main__':
    # get arguments

    out = get_test_data_mdh(vocab_path='/home/mdh19/004_csc_sl/SCOPE/FPT/vocab.txt')
    print(out)
    with open("/home/mdh19/004_csc_sl/SCOPE/data_process/list_data.pkl", 'wb') as fo:
        pickle.dump(out, fo)
    print("生成新数据文件")
    