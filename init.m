
% Set lib path only.
global path;
path.lib.matConvNet                     = '/iron/lib/matconvnet_v1.0_beta12/';
path.lib.ilsvrcDevKit                   = '/iron/lib/ILSVRC2014_devkit/';
path.lib.vlfeat                         = '/iron/lib/vlfeat/vlfeat-0.9.19/';
path.lib.jsgd                           = '/iron/lib/jsgd-61/';
path.lib.selectiveSearch                = '/iron/lib/selectiveSearch/SelectiveSearchCodeIJCV/';
% Set dst dir.
path.dstDir                             = '/nickel/data_new/';
% Set image DB path only.
path.db.voc2007.name                    = 'VOC2007';
path.db.voc2007.funh                    = @DB_VOC2007;
path.db.voc2007.root                    = '/iron/db/VOC2007';
path.db.coco2014.name                   = 'COCO2014';
path.db.coco2014.funh                   = @DB_COCO2014;
path.db.coco2014.root                   = '/iron/db/COCO2014';
path.db.caltech_pedestrian.name         = 'CALTECH_PEDESTRIAN';
path.db.caltech_pedestrian.funh         = @DB_CALTECH_PEDESTRIAN;
path.db.caltech_pedestrian.root         = '/iron/db/CALTECH_PEDESTRIAN';
path.db.ddsm.name                       = 'DDSM';
path.db.ddsm.root                       = '/iron/db/DDSM';
path.db.ddsm.funh                       = @DB_DDSM;
% Set pre-trained CNN path only.
path.net.caffeRef.name                  = 'CAFFREF';
path.net.caffeRef.path                  = '/iron/net/imagenet-caffe-ref.mat';
path.net.alex.name                      = 'CAFFALX';
path.net.alex.path                      = '/iron/net/imagenet-caffe-alex.mat';
path.net.vgg_f.name                     = 'VGGF';
path.net.vgg_f.path                     = '/iron/net/imagenet-vgg-f.mat';
path.net.vgg_s.name                     = 'VGGS';
path.net.vgg_s.path                     = '/iron/net/imagenet-vgg-s.mat';
path.net.vgg_m.name                     = 'VGGM';
path.net.vgg_m.path                     = '/iron/net/imagenet-vgg-m.mat';
path.net.ddsm.name                      = 'DDSM';
path.net.ddsm.path                      = '/iron/net/ddsm-ft-alex.mat';
% Do not touch the following codes.
run( fullfile( path.lib.vlfeat, 'toolbox/vl_setup.m' ) );       % VLFeat.
run( fullfile( path.lib.matConvNet, 'matlab/vl_setupnn.m' ) );  % MatConvnet.
addpath( fullfile( path.lib.jsgd, '/yael/matlab' ) );           % JSGD dependency.
addpath( fullfile( path.lib.jsgd, '/pqcodes_matlab' ) );        % JSGD dependency.
addpath( fullfile( path.lib.jsgd, '/matlab' ) );                % JSGD main.
addpath( fullfile( path.lib.ilsvrcDevKit, 'evaluation' ) );     % ILSVRC dev kit.
addpath( genpath( path.lib.selectiveSearch ) );                 % Selective search.

