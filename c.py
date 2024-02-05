import argparse
import evaluate_MemNet


def parse_config():
    parser = argparse.ArgumentParser(description='[Train] MemoNet on ETH/UCY datasets')
    # Configuration for ETH/UCY dataset.
    parser.add_argument('--cfg', default='eth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)

    parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
    parser.add_argument("--loader_num_workers", default=4, type=int)
    parser.add_argument("--dataset_name", default="eth", type=str)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument("--th", default=8.0, type=float)

    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)  #
    parser.add_argument("--learning_rate", type=int, default=0.0001)  # 0.0001
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)
    parser.add_argument("--preds", type=int, default=20)
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    # MODEL CONTROLLER
    parser.add_argument("--model",
                        default='pretrained_models/CLTPMAN/eth')  # model_controller/hotel_ctr
    parser.add_argument("--saved_memory", default=True)  # True
    parser.add_argument("--saveImages", default=True, help="plot qualitative examples in tensorboard")
    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    v = evaluate_MemNet.Trainer(config)
    print('start evaluation')
    v.online_fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
