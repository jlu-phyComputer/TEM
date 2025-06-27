import torch
import argparse
from experiments.exp_long_term_forecasting_my import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(para_dict):
    model_name = "CDTF_TEM"
    save_path = './checkpoints/' + model_name + "_ecl_v1/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pred_len = para_dict["pred_len"]
    lr = para_dict["lr"]
    weight_decay = para_dict["weight_decay"]
    use_warm_up = para_dict["use_warm_up"]
    warm_up_len = para_dict["warm_up_len"]
    warm_up_factor = para_dict["warm_up_factor"]
    batch_size = para_dict["batch_size"]
    e_layers = 4
    d_model = 512
    # e_layers = para_dict["e_layers"]
    # d_model = para_dict["d_model"]
    # 随机种子
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    print("torch.manual_seed({})".format(fix_seed))


    parser = argparse.ArgumentParser(description='T2B-PE')

    # basic config
    # use_weight_dec
    parser.add_argument('--use_weight_dec', type=bool, required=False,
                        help='use_weight_dec', default=para_dict["use_weight_dec"])
    parser.add_argument('--use_warm_up', type=bool, required=False,
                        help='use_warm_up', default=use_warm_up)
    parser.add_argument('--warm_up_len', type=bool, required=False,
                        help='warm_up_len', default=warm_up_len)
    parser.add_argument('--warm_up_factor', type=bool, required=False,
                        help='warm_up_factor', default=warm_up_factor)
    parser.add_argument('--weight_decay', type=float, required=False,
                        help='weight_decay', default=weight_decay)
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='ECL_96_96', help='model id')
    parser.add_argument('--model', type=str, required=False, default=model_name)

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                             'predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min '
                             'or 3h')
    parser.add_argument('--checkpoints', type=str, default=save_path, help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')  # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=321, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=321, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=321,
                        help='output size')  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=e_layers, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False,
                        help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    # parser.add_argument('--weight_decay', type=float, action='weight_decay', help='wc', default=1e-8)
    # optimization
    parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--learning_rate', type=float, default=lr, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[\'MTSF\', \'partial_train\']')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/',
                        help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False,
                        help='whether to use efficient_training (exp_name should be partial train)')  # See Figure 8 of our paper for the detail

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else:  # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args, weight_decay=weight_decay)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    para_dict = {'pred_len': 96, 'lr': 0.0007,
                 'weight_decay': 9e-06, 'use_warm_up': False, 'warm_up_len': 1100,
                 'warm_up_factor': 0.0008, 'batch_size': 16, 'use_weight_dec': True}
    main(para_dict)
    print(para_dict)

    # lr_list2 = [round(0.006 + 0.0001 * (i - 100), 4) for i in range(50, 201)]
    # lr_list = ([0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
    #             0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009] +
    #            [round(0.006 + 0.0001 * (i - 100), 4) for i in range(50, 60)])
    # # lr_list.reverse()
    # for i in range(len(lr_list)):
    #     lr = lr_list[i]
    #     print("lr=", lr_list[i])
    #     para_dict = {'pred_len': 96, 'lr': lr,
    #                  'weight_decay': 9e-06, 'use_warm_up': False, 'warm_up_len': 1100,
    #                  'warm_up_factor': 0.0008, 'batch_size': 16, 'use_weight_dec': True}
    #     main(para_dict)
    #     print("lr=", lr_list[i])
    #     print(para_dict)

