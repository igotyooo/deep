
% Set lib path only.
global path;
path.lib_matConvNet                     = '/iron/lib/matconvnet_v1.0_beta7/';
path.lib_ilsvrcDevKit                   = '/iron/lib/ILSVRC2014_devkit/';
path.lib_vlfeat                         = '/iron/lib/vlfeat/vlfeat-0.9.19/';
path.lib_jsgd                           = '/iron/lib/jsgd-61/';
path.lib_selectiveSearch                = '/iron/lib/selectiveSearch/SelectiveSearchCodeIJCV/';
% Set root directory of the dst data.
path.dstdir                             = '/nickel/data_newdb'; % '/iron/dataiccv';
% Set image DB path only.
path.db_ramdisk                         = '/ramdisk';
path.db_ramdisk_size                    = '2G';
path.db_voc2007.name                    = 'VOC2007';
path.db_voc2007.funh                    = @DB_VOC2007;
path.db_voc2007.root                    = '/iron/db/VOC2007';
path.db_coco2014.name                   = 'COCO2014';
path.db_coco2014.funh                   = @DB_COCO2014;
path.db_coco2014.root                   = '/iron/db/COCO2014';
path.db_caltech_pedestrian.name         = 'CALTECH_PEDESTRIAN';
path.db_caltech_pedestrian.funh         = @DB_CALTECH_PEDESTRIAN;
path.db_caltech_pedestrian.root         = '/iron/db/CALTECH_PEDESTRIAN';
% Set pre-trained CNN path.
path.extnet_caffeRef                    = '/iron/data/TRAINED_NETS/imagenet-caffe-ref.mat';
path.extnet_alex                        = '/iron/data/TRAINED_NETS/imagenet-caffe-alex.mat';
path.extnet_vgg_f                       = '/iron/data/TRAINED_NETS/imagenet-vgg-f.mat';
path.extnet_vgg_s                       = '/iron/data/TRAINED_NETS/imagenet-vgg-s.mat';
path.extnet_vgg_m                       = '/iron/data/TRAINED_NETS/imagenet-vgg-m.mat';
path.extnet_vgg_m_2048                  = '/iron/data/TRAINED_NETS/imagenet-vgg-m-2048.mat';
path.extnet_vgg_m_1024                  = '/iron/data/TRAINED_NETS/imagenet-vgg-m-1024.mat';
path.extnet_hudet_coco2014              = '/iron/data/TRAINED_NETS/hudet_coco2014.mat';
% Set pre-trained CNN name.
path.extnet_name_caffeRef               = 'CAFFREF';
path.extnet_name_alex                   = 'CAFFALX';
path.extnet_name_vgg_f                  = 'VGGF';
path.extnet_name_vgg_s                  = 'VGGS';
path.extnet_name_vgg_m                  = 'VGGM';
path.extnet_name_vgg_m_2048             = 'VGGM2048';
path.extnet_name_vgg_m_1024             = 'VGGM1024';
path.extnet_name_hudet_coco2014         = 'HUDETCOCO2014';
% Do not touch the following.
run( fullfile( path.lib_vlfeat, 'toolbox/vl_setup.m' ) );       % VLFeat.
run( fullfile( path.lib_matConvNet, 'matlab/vl_setupnn.m' ) );  % MatConvnet.
addpath( fullfile( path.lib_jsgd, '/yael/matlab' ) );           % JSGD dependency.
addpath( fullfile( path.lib_jsgd, '/pqcodes_matlab' ) );        % JSGD dependency.
addpath( fullfile( path.lib_jsgd, '/matlab' ) );                % JSGD main.
addpath( fullfile( path.lib_ilsvrcDevKit, 'evaluation' ) );     % ILSVRC dev kit.
addpath( genpath( path.lib_selectiveSearch ) );                 % Selective search.