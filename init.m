
% Set lib path only.
global path;
path.lib.matConvNet                     = '/iron/lib/matconvnet_v1.0_beta12_cuda6.5_cudnn/';
path.lib.ilsvrcDevKit                   = '/iron/lib/ILSVRC2015_devkit/devkit/';
path.lib.vlfeat                         = '/iron/lib/vlfeat/vlfeat-0.9.19/';
path.lib.caffePaeng                     = '/iron/lib/caffe-paeng/';
% Set dst dir.
path.dstDir                             = '/nickel/data_ilsvrc15/';
% Set image DB path only.
path.db.voc2007.name                    = 'VOC2007';
path.db.voc2007.funh                    = @DB_VOC2007;
path.db.voc2007.root                    = '/iron/db/VOC2007';
path.db.coco2014.name                   = 'COCO2014';
path.db.coco2014.funh                   = @DB_COCO2014;
path.db.coco2014.root                   = '/iron/db/COCO2014';
path.db.ilsvrcdet2015.name              = 'ILSVRCDET2015';
path.db.ilsvrcdet2015.funh              = @DB_ILSVRCDET2015;
path.db.ilsvrcdet2015.root              = '/iron/db/ILSVRC2015';
path.db.ilsvrcdet2015te.name            = 'ILSVRCDET2015TE';
path.db.ilsvrcdet2015te.funh            = @DB_ILSVRCDET2015TE;
path.db.ilsvrcdet2015te.root            = '/iron/db/ILSVRC2015';
path.db.ilsvrcclsloc2015.name           = 'ILSVRCCLSLOC2015';
path.db.ilsvrcclsloc2015.funh           = @DB_ILSVRCCLSLOC2015;
path.db.ilsvrcclsloc2015.root           = '/iron/db/ILSVRC2015';
path.db.ilsvrcclsloc2015te.name         = 'ILSVRCCLSLOC2015TE';
path.db.ilsvrcclsloc2015te.funh         = @DB_ILSVRCCLSLOC2015TE;
path.db.ilsvrcclsloc2015te.root         = '/iron/db/ILSVRC2015';
path.db.ddsm.name                       = 'DDSM';
path.db.ddsm.funh                       = @DB_DDSM;
path.db.ddsm.root                       = '/iron/db/DDSM';
% Set pre-trained CNN path only.
path.net.caffeRef.name                  = 'CAFFREF';
path.net.caffeRef.path                  = '/iron/net/mat/imagenet-caffe-ref.mat';
path.net.alex.name                      = 'CAFFALX';
path.net.alex.path                      = '/iron/net/mat/imagenet-caffe-alex.mat';
path.net.vgg_f.name                     = 'VGGF';
path.net.vgg_f.path                     = '/iron/net/mat/imagenet-vgg-f.mat';
path.net.vgg_s.name                     = 'VGGS';
path.net.vgg_s.path                     = '/iron/net/mat/imagenet-vgg-s.mat';
path.net.vgg_m.name                     = 'VGGM';
path.net.vgg_m.path                     = '/iron/net/mat/imagenet-vgg-m.mat';
path.net.vgg_m_2048.name                = 'VGGM2048';
path.net.vgg_m_2048.path                = '/iron/net/mat/imagenet-vgg-m-2048.mat';
path.net.ddsm.name                      = 'DDSM';
path.net.ddsm.path                      = '/iron/net/mat/ddsm-ft-alex.mat';
% Set pre-trained Caffe AttNet path only.
path.attNetCaffe.ilsdet.modelPath       = '/iron/net/caffe/attNet_gnet_ilsdet_e8_x2.caffemodel';
path.attNetCaffe.ilsdet.protoPath       = '/iron/net/caffe/attNet_gnet_ilsdet.prototxt';
path.attNetCaffe.ilsdet.protoPathTest   = '/iron/net/caffe/attNet_gnet_ilsdet_test.prototxt';
path.attNetCaffe.ilsdet.rgbMeanPath     = '/iron/net/caffe/attNet_gnet_ilsdet_rgbmean.mat';
path.attNetCaffe.ilsdet.modelName       = 'ANET_GOO_ILSDET15';
path.attNetCaffe.ilsdet.patchSide       = 223;
path.attNetCaffe.ilsdet.inputSide       = 224;
path.attNetCaffe.ilsdet.stride          = 32;
path.attNetCaffe.ilsloc.modelPath       = '/iron/net/caffe/attNet_gnet_ilsclsloc_e4.5.caffemodel';
path.attNetCaffe.ilsloc.protoPath       = '/iron/net/caffe/attNet_gnet_ilsclsloc.prototxt';
path.attNetCaffe.ilsloc.protoPathTest   = '/iron/net/caffe/attNet_gnet_ilsclsloc_test.prototxt';
path.attNetCaffe.ilsloc.rgbMeanPath     = '/iron/net/caffe/attNet_gnet_ilsdet_rgbmean.mat';
path.attNetCaffe.ilsloc.modelName       = 'ANET_GOO_ILSCLSLOC15';
path.attNetCaffe.ilsloc.patchSide       = 223;
path.attNetCaffe.ilsloc.inputSide       = 224;
path.attNetCaffe.ilsloc.stride          = 32;
path.attNetCaffe.vocdet.modelPath       = '/iron/net/caffe/attnet_gnet_voc07.caffemodel';
path.attNetCaffe.vocdet.protoPath       = '/iron/net/caffe/attnet_gnet_voc07.prototxt';
path.attNetCaffe.vocdet.protoPathTest   = '/iron/net/caffe/attnet_gnet_voc07_test.prototxt';
path.attNetCaffe.vocdet.rgbMeanPath     = '/iron/net/caffe/attnet_gnet_voc07_rgbmean.mat';
path.attNetCaffe.vocdet.modelName       = 'ANET_GOO_VOCDET07';
path.attNetCaffe.vocdet.patchSide       = 223;
path.attNetCaffe.vocdet.inputSide       = 224;
path.attNetCaffe.vocdet.stride          = 32;
% Do not touch the following codes.
run( fullfile( path.lib.vlfeat, 'toolbox/vl_setup.m' ) );       % VLFeat.
run( fullfile( path.lib.matConvNet, 'matlab/vl_setupnn.m' ) );  % MatConvnet.
addpath( fullfile( path.lib.ilsvrcDevKit, 'evaluation' ) );     % ILSVRC dev kit.
addpath( fullfile( path.lib.caffePaeng, 'matlab' ) );           % Caffe-Paeng.











