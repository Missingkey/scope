import argparse
import os
from pytorch_lightning import Trainer
from datasets.bert_csc_dataset import TestCSCDataset
from finetune.train import CSCTask
from torch.utils.data.dataloader import DataLoader
from functools import partial
import torch
from datasets.collate_functions import collate_to_max_length_with_id
import re
from flask import Flask, render_template, request
from flask_cors import CORS





app = Flask(__name__)
CORS(app)
correction_result = ""

#* 对中文中的“的地得”处理
def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)

#* 处理参数
def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--ckpt_path", required=True, type=str, help="ckpt file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default='/home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv',
         type=str, help="label file")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument("--save_topk", default=5, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default='train', type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser

def create_data_loader(args):
    dataset = TestCSCDataset(
        data_path='/home/mdh19/004_csc_sl/SCOPE/data_process/list_data.pkl',
        chinese_bert_path=args.bert_path,
        max_length=args.max_length,
    )
        # self.tokenizer = dataset.tokenizer
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
        drop_last=False,
    )
    return dataloader

def my_predict(dataloader, args):
    for i, data in enumerate(dataloader):
        print(data)
        output = [model.predict_step(data, i)]
        from metrics.metric import Metric
        metric = Metric(vocab_path=args.bert_path)
        pred_txt_path = os.path.join(args.save_path, "preds.txt")
        pred_lbl_path = os.path.join(args.save_path, "labels.txt")
        out = metric.metric(
            batches=output,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=args.label_file,
            should_remove_de=True if '13'in args.label_file else False
            )
        correction_result = out[0].split("\t")[1]
        print(out[0].split("\t")[1])
        return correction_result


def main():
    """main, load model"""
    global correction_result
    global model
    global args
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CSCTask(args)

    # trainer = Trainer.from_argparse_args(args)
    model.load_state_dict(torch.load(args.ckpt_path)["state_dict"])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/spell_check', methods=['POST'])
def spell_check():
    data = request.get_json()
    print("*************")
    print(data)
    print("*************")
    txt = data["check_text"]
    print("*************")
    print(txt)
    print("*************")
    os.system(f"python /home/mdh19/004_csc_sl/SCOPE/data_process/get_test_data_mdh.py \
                --text {txt}")
    dataloader = create_data_loader(args=args)
    correction = my_predict(dataloader=dataloader, args=args)
    return f'改错结果是：{correction}'


# @app.route('/submit', methods=['POST'])
# def submit():
#     txt = request.form.get('name')
#     os.system(f"python /home/mdh19/test_projects/SCOPE/data_process/get_test_data_mdh.py \
#                 --text {txt}")
#     #TODO 创建新的dataloader,然后进行predcit
#     dataloader = create_data_loader(args=args)
#     correction = my_predict(dataloader=dataloader, args=args)
#     return f'改错结果是：{correction}'

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    #将model启动起来
    main()
    app.run(host="0.0.0.0", port=11451, debug=False)
    
    