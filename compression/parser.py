#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import distiller
import distiller.pruning
import distiller.models as models


def add_cmdline_args(parser):
    parser.add_argument('--save-untrained-model',  action='store_true', default=False,
                        help='Save the randomly-initialized model before training (useful for lottery-ticket method)')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --earlyexit_lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')
    
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='RGB',
                    	choices=["RGB", "tvl1", "lk_flow"],
                    	help='modality: RGB | flow')

    parser.add_argument('--dataset', '-d', default='ucf101',
                    	choices=["ucf101", "hmdb51"],
                    	help='dataset: ucf101 | hmdb51')

    parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    	help='which split of data to work on (default: 1)')

    parser.add_argument('--settings', metavar='DIR', default='./settings',
                    	help='path to datset setting files')

    # parser.add_argument('--new_length', default=1, type=int,
    #                 	metavar='N', help='length of sampled video frames (default: 1)')

    parser.add_argument('--new_width', default=340, type=int,
                    	metavar='N', help='resize width (default: 340)')
    parser.add_argument('--new_height', default=256, type=int,
                    	metavar='N', help='resize height (default: 256)')
    parser.add_argument('--num_classes', default=101, type=int,
                        metavar='N', help='classes number(default: 101)')
    parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
    parser.add_argument('--num-segments', type=int, default=3)
    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no-partialbn', '--npb', default=False, action="store_true")
    parser.add_argument('--flow-prefix', type=str, default='flow_')

    distiller.knowledge_distillation.add_distillation_args(parser, models.ALL_MODEL_NAMES, True)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    return parser
