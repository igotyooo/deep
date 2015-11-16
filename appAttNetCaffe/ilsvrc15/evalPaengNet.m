%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '../../' ) ); init_ilsvrc15;
setting.gpus                                        = 1;
setting.db                                          = path.db.voc2007; path.db.ilsvrcdet2015; 
setting.io.tsDb.numScaling                          = 24;
setting.io.tsDb.dilate                              = 1 / 4;
setting.io.tsDb.normalizeImageMaxSide               = 500;
setting.io.tsDb.posGotoMargin                       = 2.4;
setting.io.tsDb.numQuantizeBetweenStopAndGoto       = 3;
setting.io.tsDb.negIntOverObjLessThan               = 0.1;
setting.io.net.pretrainedNetName                    = path.net.vgg_m.name;
setting.io.net.suppressPretrainedLayerLearnRate     = 1 / 4;
setting.io.general.shuffleSequance                  = false; 
setting.io.general.batchSize                        = 128;
setting.io.general.numGoSmaplePerObj                = 1;
setting.io.general.numAnyDirectionSmaplePerObj      = 14; 2; 
setting.io.general.numStopSmaplePerObj              = 1;
setting.io.general.numBackgroundSmaplePerObj        = 16; 4; 

reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutAttNetCornerPerCls2( db, ...
    setting.io.tsDb, ...
    setting.io.net, ...
    setting.io.general );
io.init;
io.makeTsDb;

%% DO THE JOB.
clc; clearvars -except db path setting io;
netInfo.modelPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.caffemodel';
netInfo.protoPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.prototxt';
netInfo.rgbMeanPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07_rgbmean.mat';
netInfo.patchSide = 224; 223;  % <----!!!!!!!!!!!!!!!!!!!!!!!!
netInfo.stride = 32;

io.patchSide = netInfo.patchSide;
rgbMean = load( netInfo.rgbMeanPath, 'rgbMean' );
rgbMean = rgbMean.rgbMean;
caffe.reset_all(  );
caffe.set_mode_gpu(  );
caffe.set_device( setting.gpus - 1 );
net = caffe.Net( netInfo.protoPath, netInfo.modelPath, 'test' );
prefix = 'prob';
clsLyrPostFix = 'cls';
cornerNameTl = 'TL';
cornerNameBr = 'BR';
numCls = db.getNumClass;
numLyr = numel( net.outputs );
lyid02lyid = zeros( numLyr, 1 );
for lyid = 1 : numLyr,
    lname = net.outputs{ lyid };
    if strcmp( lname( end - 2 : end ), clsLyrPostFix ),
        lyid0 = numCls * 2 + 1;
        lyid02lyid( lyid0 ) = lyid;
        continue;
    end;
    data = textscan( lname, strcat( prefix, '%d_%s' ) );
    cid = data{ 1 } + 1;
    cornerName = data{ 2 }{ : };
    switch cornerName,
        case cornerNameTl,
            bias = 1;
        case cornerNameBr,
            bias = 2;
        otherwise,
            error( 'Strange corner name.\n' );
    end;
    lyid0 = ( cid - 1 ) * 2 + bias;
    lyid02lyid( lyid0 ) = lyid;
end;
if numel( unique( lyid02lyid ) ) ~= numCls * 2 + 1,
    error( 'Wrong net output layer.\n' ); end;

% Evaluate net.
batchSize = io.getBatchSize;
numBchVal = io.getNumBatchVal;
metric = 0;
for b = 1 : numBchVal; btime = tic;
    [ im, gt ] = io.provdBchVal;
    [ h, w, c, n ] = size( im );
    im = im( :, :, [ 3, 2, 1 ], : );
    im = permute( im, [ 2, 1, 3, 4 ] );
    im = { im };
    net.blobs( 'data' ).reshape( [ w, h, c, n ] );
    lyid2out = net.forward( im );
    lyid2out = lyid2out( lyid02lyid );
    res( 1 ).x = cat( 3, lyid2out{ : } );
    res( 2 ).x = [  ];
    metric = metric + io.computeTsMetric( res, gt );
    btime = toc( btime );
    fprintf( 'Bch %d/%d, %dims/s.\n', b, numBchVal, round( batchSize / btime ) );
end;
metric = metric / ( batchSize * numBchVal );
tsMetricName = io.getTsMetricName;
for i = 1 : numel( tsMetricName ),
fprintf( '%s: %.4f\n', upper( tsMetricName{ i } ),  metric( i ) * 100 );
end;




