%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus = 1;
setting.db = path.db.something;
setting.io.net.pretrainedNetName = path.net.vgg_m.name;
setting.io.net.suppressPretrainedLayerLearnRate = 0.1;
setting.io.general.shuffleSequance = true; 
setting.io.general.batchSize = 128;
setting.net.normalizeImage = 'NONE';
setting.net.weightDecay = 0.0005;
setting.net.momentum = 0.9;
setting.net.modelType = 'dropout';
setting.net.learningRate = [ 0.01 * ones( 1, 3 ), 0.001 * ones( 1, 1 ) ];

%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutCls( db, setting.io.net, setting.io.general );
io.init;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, ...
    'visionresearchreport@gmail.com' );
net.fetchBestNet;
[ net, netName ] = net.provdNet;
net.name = netName;
net.normalization.averageImage = io.rgbMean;
