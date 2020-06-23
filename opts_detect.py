import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/sylo/SegNet/armory_mine/video_classification/ucfTrainTestlist/testlist01_jpg.txt', type=str, help='Input file path')
    parser.add_argument('--video_root', default='/home/sylo/SegNet/UCF-101_jpg', type=str, help='Root path of input videos')    
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=3, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')  
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.add_argument('--sample_duration', default=40, type=int, help='sample duration')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')	
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')	
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--n_epochs', default=7, type=int, help='Number of total epochs to run')
    parser.add_argument(  # /home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/resnet-50-kinetics.pth
	                      # /home/sylo/SegNet/3D-ResNets-PyTorch/my_train/wideresnet_multi/save_4.pth
						  # /home/sylo/SegNet/3D-ResNets-PyTorch/my_train/multi_clean_pgd_roa3/save_5.pth
        '--pretrain_path', default='/home/sylo/SegNet/armory_mine/my_train/multi_clean_pgd_roa3/save_5.pth', type=str, help='Pretrained model (.pth)')
    parser.add_argument(  # /home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/detect_train/resnet50/save_4.pth
                          # /home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/detect_train/wideresnet/save_7.pth	                   
        '--detector_path', default='/home/sylo/SegNet/armory_mine/video_classification/detect_train/resnet50/save_4.pth', type=str, help='detector model (.pth)')        
    parser.add_argument('--ft_begin_index', default=0, type=int, help='Begin block index of fine-tuning')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--model_name', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--batch_size', default=12, type=int, help='Batch Size')
    parser.add_argument('--exp_name', default='test', type=str, help='name of the experiment, results are stored in results/exp_name.txt')
    parser.add_argument('--save_image', action='store_true', help='')
    parser.add_argument('--epsilon', default=4, type=int, help='attack epsilon, eps/256')
    parser.add_argument('--attack_iter', default=5, type=int, help='number of attack iterations')
    parser.add_argument('--step_size', default=1.0, type=float, help='learning rate of attack')
    parser.add_argument('--sparsity', default=40, type=int, help='temporal sparsity of video attacks')
    parser.add_argument('--attack_type', default='clean', type=str, help='attack type')
    parser.add_argument('--roa_size', default=30, type=int, help='roa size')
    parser.add_argument('--roa_stride', default=1, type=int, help='roa stride')
    parser.add_argument('--framing_width', default=5, type=int, help='framing mask')
    parser.add_argument('--attack_bn', default='clean', type=str, help='which bn to attack')
    parser.add_argument('--inf_bn', default='clean', type=str, help='which bn to inference')		
    parser.add_argument('--num_pixel', default=100, type=int, help='number of attack pixels')

    args = parser.parse_args()

    return args
	