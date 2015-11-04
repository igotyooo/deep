
% Set lib path only.
global path;
path.lib.matConvNet                     = '/iron/lib/matconvnet_v1.0_beta12_cuda6.5_cudnn/';
path.lib.ilsvrcDevKit                   = '/iron/lib/ILSVRC2015_devkit/devkit/';
path.lib.caffePaeng                     = '/iron/lib/caffe-paeng/';
% Set dst dir.
path.dstDir                             = '/nickel/data_ilsvrc15/';
% Set image DB path only.
path.db.ilsvrcdet2015.name              = 'ILSVRCDET2015';
path.db.ilsvrcdet2015.funh              = @DB_ILSVRCDET2015;
path.db.ilsvrcdet2015.root              = '/iron/db/ILSVRC2015';
path.db.ilsvrcdet2015te.name            = 'ILSVRCDET2015TE';
path.db.ilsvrcdet2015te.funh            = @DB_ILSVRCDET2015TE;
path.db.ilsvrcdet2015te.root            = '/iron/db/ILSVRC2015';
path.db.ilsvrcclsloc2015.name           = 'ILSVRCCLSLOC2015';
path.db.ilsvrcclsloc2015.funh           = @DB_ILSVRCCLSLOC2015;
path.db.ilsvrcclsloc2015.root           = '/iron/db/ILSVRC2015';
% Set pre-trained Caffe AttNet path only.
path.attNetCaffe.ilsdet.modelPath       = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_e8_x2.caffemodel';
path.attNetCaffe.ilsdet.protoPath       = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet.prototxt';
path.attNetCaffe.ilsdet.protoPathTest   = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_test.prototxt';
path.attNetCaffe.ilsdet.rgbMeanPath     = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_rgbmean.mat';
path.attNetCaffe.ilsdet.modelName       = 'ANET_GOO_ILSDET15';
path.attNetCaffe.ilsdet.patchSide       = 223;
path.attNetCaffe.ilsdet.inputSide       = 224;
path.attNetCaffe.ilsdet.stride          = 32;
path.attNetCaffe.ilsloc.modelPath       = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsclsloc_e4.5.caffemodel';
path.attNetCaffe.ilsloc.protoPath       = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsclsloc.prototxt';
path.attNetCaffe.ilsloc.protoPathTest   = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsclsloc_test.prototxt';
path.attNetCaffe.ilsloc.rgbMeanPath     = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_rgbmean.mat';
path.attNetCaffe.ilsloc.modelName       = 'ANET_GOO_ILSCLSLOC15';
path.attNetCaffe.ilsloc.patchSide       = 223;
path.attNetCaffe.ilsloc.inputSide       = 224;
path.attNetCaffe.ilsloc.stride          = 32;
% Do not touch the following codes.
run( fullfile( path.lib.matConvNet, 'matlab/vl_setupnn.m' ) );  % MatConvnet.
addpath( fullfile( path.lib.ilsvrcDevKit, 'evaluation' ) );     % ILSVRC dev kit.
addpath( fullfile( path.lib.caffePaeng, 'matlab' ) );           % Caffe-Paeng.











