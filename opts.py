import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/home/sylo/SegNet/3D-ResNets-PyTorch', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='/home/sylo/SegNet/UCF-101_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/ucf101_01.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='/home/sylo/SegNet/3D-ResNets-PyTorch/my_train/', type=str, help='Result directory path')
    parser.add_argument('--pretrain_path', default='/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/resnet-18-kinetics.pth', type=str, help='Pretrained model (.pth)')	
    parser.add_argument('--dataset', default='ucf101', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=101, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=40, type=int, help='Temporal duration of inputs')
    parser.add_argument('--ft_begin_index', default=0, type=int, help='Begin block index of fine-tuning')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')

    # === Unused by me === #		
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')	
	
    # === Model === #
    parser.add_argument('--model', required=True, type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
	
    # === Training === #
    parser.add_argument('--learning_rate',default=5e-4, type=float, help= 'Initial learning rate (divided by 10 while training by lr scheduler)')
    # batch_size=8: 0.001, batch_size=4: 5e-5	
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')	
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=15, type=int, help='Number of total epochs to run')		

    # === Attack === #	
    parser.add_argument('--exp_name', required=True, type=str, help='name of the experiment, results are stored in results/exp_name.txt')
    parser.add_argument('--epsilon', default=4, type=int, help='attack epsilon, eps/256')
    parser.add_argument('--attack_iter', default=5, type=int, help='number of attack iterations')
    parser.add_argument('--step_size', default=1.0, type=float, help='learning rate of attack')
    parser.add_argument('--sparsity', default=40, type=int, help='temporal sparsity of video attacks')
    parser.add_argument('--attack_type', required=True, type=str, help='attack type')
    parser.add_argument('--roa_size', default=30, type=int, help='roa size')
    parser.add_argument('--roa_stride', default=1, type=int, help='roa stride')
    parser.add_argument('--framing_width', default=5, type=int, help='framing mask')	

    args = parser.parse_args()

    return args
	
	