import argparse

import torch

from exp.exp_classification import Exp_Classification
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
from utils.tools import get_setting, set_random_seed


def get_args():
    parser = argparse.ArgumentParser(description='TEFN')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True,
                        default='TEFN_FuzzyTCN_Classifier_32',
                        choices=['TEFN', 'TEFN_FuzzyTCN_Classifier_32'],
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                             'predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./out/results/', help='location of experiment results')
    parser.add_argument('--checkpoint', type=str, help='specific checkpoint file for inference')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (percent)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # classification task
    parser.add_argument('--num_class', type=int, default=4, help='number of classes')
    parser.add_argument('--classification_hidden_dim', type=int, default=64,
                        help='hidden size of the classification head')
    parser.add_argument('--fuzzy_levels', type=int, default=3,
                        help='number of explicit Gaussian fuzzy levels per sensor')
    parser.add_argument('--tcn_hidden_dim', type=int, default=8,
                        help='hidden channels of the lightweight causal TCN time branch')
    parser.add_argument('--time_bottleneck_dim', type=int, default=0,
                        help='low-rank time mixer bottleneck for SensorFCM; '
                             '0 keeps the original dense mixer')
    parser.add_argument('--time_mixer_type', type=str, default='dense',
                        choices=['dense', 'separable'],
                        help='SensorFCM time mixer: dense T*F mixing or '
                             'parameter-efficient separate time/state mixing')
    parser.add_argument('--classification_stride', type=int, default=10,
                        help='stride between fire-source classification windows')
    parser.add_argument('--event_gap_seconds', type=float, default=60.0,
                        help='time gap that separates distinct fire experiments')
    parser.add_argument('--max_gap_seconds', type=float, default=20.0,
                        help='maximum consecutive gap allowed inside a sensor window')
    parser.add_argument('--exclude_wood', action='store_true',
                        help='exclude Wood and classify Cable/Candles/Lunts only')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=10, help='data loader prefetch factor')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--min_epochs', type=int, default=10,
                        help='minimum epochs before early stopping is counted')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='AdamW weight decay for classification')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau learning-rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=1,
                        help='validation epochs without macro-F1 improvement before reducing LR')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6,
                        help='minimum learning rate used by ReduceLROnPlateau')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=True,
                        help='use GPU when available')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action=argparse.BooleanOptionalAction, default=False,
                        help='calculate the time-consuming DTW metric')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    parser.add_argument('--noise', action=argparse.BooleanOptionalAction, default=False,
                        help='add input noise during evaluation')

    # TEFN
    parser.add_argument('--use_norm', action=argparse.BooleanOptionalAction, default=True,
                        help="use normalization layer")
    parser.add_argument('--use_T_model', action=argparse.BooleanOptionalAction, default=True,
                        help="use time-dimension evidence module")
    parser.add_argument('--use_C_model', action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to use channel dimension module")
    parser.add_argument('--fusion_method', type=str, default='add', choices=['add', 'concat'],
                        help="evidence fusion method")
    parser.add_argument('--use_residual', action=argparse.BooleanOptionalAction, default=True,
                        help="use residual connection in EvidenceMachineKernel")
    parser.add_argument('--kernel_activation', type=str, default='linear',
                        choices=['linear', 'mlp', 'attn', 'relu', 'gelu', 'swish', 'mish', 'elu', 'tanh'],
                        help="Activation function for EvidenceMachineKernel")
    parser.add_argument('--use_probabilistic_layer', action=argparse.BooleanOptionalAction, default=False,
                        help="use probabilistic dropout layer")
    parser.add_argument('--fcm_steps', type=int, default=2,
                        help='number of FCM cross-sensor propagation steps (TEFN_SensorFCM_Classifier_32)')
    parser.add_argument('--fcm_l1_weight', type=float, default=0.0,
                        help='L1 penalty weight on FCM adjacency edges (TEFN_SensorFCM_Classifier_32); '
                             '0 disables it, try 0.001-0.01 to sharpen the learned graph')
    parser.add_argument('--fcm_temp_min', type=float, default=1.0,
                        help='minimum tanh temperature the FCM adjacency anneals to '
                             '(TEFN_SensorFCM_Classifier_32); default 1.0 disables '
                             'annealing (best measured macro_F1). Lower values (e.g. 0.3) '
                             'sharpen the learned graph for interpretability at a measured '
                             'macro_F1 cost (~0.836 -> ~0.803 at temp_min=0.3)')
    parser.add_argument('--fcm_temp_decay', type=float, default=0.999,
                        help='per-training-step multiplicative decay applied to the FCM '
                             'temperature (TEFN_SensorFCM_Classifier_32); no effect when '
                             'fcm_temp_min=1.0')
    parser.add_argument('--fcm_state_activation', type=str, default='sigmoid',
                        choices=['tanh', 'sigmoid'],
                        help='FCM concept-activation squashing function (TEFN_SensorFCM_Classifier_32): '
                             '"sigmoid" is the best measured setting and maps concepts to fuzzy '
                             'membership in [0,1]; "tanh" is bipolar [-1,1]')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing epsilon for CrossEntropyLoss (0=off, try 0.1)')
    parser.add_argument('--use_focal_loss', action=argparse.BooleanOptionalAction, default=False,
                        help='use focal loss instead of (weighted) CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focal loss focusing parameter gamma (higher = more focus on hard examples)')
    parser.add_argument('--class_weight_override', type=float, nargs='+', default=None,
                        help='manual class weights [Background Fire Nuisance]; '
                             'overrides inverse-frequency weighting when set')
    parser.add_argument('--class_weight_power', type=float, default=1.0,
                        help='power applied to inverse-frequency class weights; '
                             '1.0 is full balancing, 0.5 is square-root balancing')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='incorrect-evidence KL regularization weight')
    parser.add_argument('--edl_loss_weight', type=float, default=0.1,
                        help='weight of the auxiliary evidential loss; the main '
                             'classification loss remains weighted cross entropy')
    parser.add_argument('--annealing_epochs', type=int, default=10,
                        help='epochs used to anneal evidential KL regularization')
    parser.add_argument('--branch_loss_weight', type=float, default=0.2,
                        help='auxiliary evidential loss weight for each branch')

    args = parser.parse_args()

    args.use_gpu = args.use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available())

    print(args.use_gpu)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'classification':
        if args.data not in {'IndoorFireSource', 'IndoorFireTernary'}:
            raise ValueError(
                "classification requires --data IndoorFireSource or IndoorFireTernary"
            )
        if args.enc_in != 14:
            raise ValueError("Indoor fire classification requires --enc_in 14")
        expected_classes = (
            3
            if args.data == 'IndoorFireTernary' or args.exclude_wood
            else 4
        )
        if args.num_class != expected_classes:
            raise ValueError(
                "IndoorFireSource classification requires "
                f"--num_class {expected_classes} when exclude_wood={args.exclude_wood}"
            )
        Exp = Exp_Classification
    else:
        exit()

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = get_setting(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
