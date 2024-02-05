import argparse
import torch

from trainer import trainer_controllerMem

def parse_config():
    parser = argparse.ArgumentParser(description='[Train] MemoNet on ETH/UCY datasets')
    # Configuration for ETH/UCY dataset.
    parser.add_argument('--cfg', default='cfg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)

    parser.add_argument("--info", type=str, default='', help='Name of training/testing.')
    parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
    parser.add_argument("--loader_num_workers", default=0, type=int)
    #parser.add_argument("--dataset_name", default="zara2", type=str)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--th", default = 5, type=float) # 7.5:124;6.5:393;8.5
    # hotel:9
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32) # 32
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=100) # 60000

    parser.add_argument("--past_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)
    parser.add_argument("--best_k", type=int, default=20)  # 1
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    # parser.add_argument("--model_ae", default='pretrained_models/model_AE/zara1')
    parser.add_argument("--model_ae", default='training/training_ae/2023-11-10 11_/model_ae_epoch_482_2023-11-10 11')
    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset fimle")

    return parser.parse_args()
import os
envpath = '/data/xx/xx/venv/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

def main(config):
    print('Start training writing controller')
    t = trainer_controllerMem.Trainer(config)
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
