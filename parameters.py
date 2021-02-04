"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    This is a file about parameters.
"""

import argparse
def getHyperParams():
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--epochs', type=int, default=2000)
    model_parser.add_argument('--l2_reg_G', type=float, default=0.001)
    model_parser.add_argument('--l2_reg_D', type=float, default=0.001)
    model_parser.add_argument('--lr_G', type=float, default=0.0001)
    model_parser.add_argument('--lr_D', type=float, default=0.0001)
    model_parser.add_argument('--batchSize_G', type=str, default=64)
    model_parser.add_argument('--batchSize_D', type=str, default=64)
    model_parser.add_argument('--opt_G', type=str, default='adam')
    model_parser.add_argument('--opt_D', type=str, default='adam')
    model_parser.add_argument('--hdim_D', type=int, default=300)
    model_parser.add_argument('--hlayer_D', type=int, default=1)
    model_parser.add_argument('--step_G', type=int, default=2)
    model_parser.add_argument('--step_D', type=int, default=2)
    model_config = model_parser.parse_args()
    return model_config