function [ db, imsvrTr, cnn, setting ] = loadVggCnn( cnnPath, cnnName, dbName, dbFunh, dstdir, useGpu )

    srcCnn = load( cnnPath );
    srcImsvrTr = srcCnn.normalization;
    setting.db.dir                   = dstdir;
    setting.db.dbName                = dbName;
    setting.db.funGenDbCls           = dbFunh;
    setting.imsvrTr.itpltn           = srcImsvrTr.interpolation;
    setting.imsvrTr.srcSide          = srcImsvrTr.imageSize( 1 ) + srcImsvrTr.border( 1 );
    setting.imsvrTr.dstSide          = srcImsvrTr.imageSize( 1 );
    setting.imsvrTr.srcCh            = srcImsvrTr.imageSize( 3 );
    setting.imsvrTr.dstCh            = srcImsvrTr.imageSize( 3 );
    setting.imsvrTr.keepAspect       = logical( srcImsvrTr.keepAspect );
    setting.imsvrTr.aug              = 'F25';
    setting.imsvrTr.numAug           = 1;
    setting.imsvrTr.nmlzByAvg        = true;
    setting.cnn.archFunh             = str2func( strcat( 'VCFG_', cnnName ) );
    setting.cnn.batchSize            = 256;
    setting.cnn.numEpch              = 65;
    setting.cnn.weightDecay          = 0.0005;
    setting.cnn.momentum             = 0.9;
    setting.cnn.learnRate            = [ 0.01 * ones( 1, 25 ), ...
                                         0.001 * ones( 1, 25 ), ...
                                         0.0001 * ones( 1, 15 ) ];
    % DB formation.
    db = DbCls( setting.db );
    db.genDbCls;
    [ ~, sgt2tgt ] = sort( srcCnn.classes.name );
    if ~issame( sgt2tgt, 1 : length( db.gtid2gtname ) )
        fprintf( 'Reform db to sync class id.\n' );
        db.iid2gt = mat2cell( sgt2tgt( cell2mat( db.iid2gt ) )', ...
            ones( size( db.iid2gt ) ) );
        db.gtid2gtdesc = srcCnn.classes.description';
        db.gtid2gtname = srcCnn.classes.name';
    end
    % CNN formation.
    net.layers = srcCnn.layers;
    for lid = 1 : numel( net.layers )
        if ~strcmp( net.layers{ lid }.type, 'conv' ),
            continue;
        end
        if ~isfield( net.layers{ lid }, 'filtersMomentum' )
            net.layers{ lid }.filtersMomentum = ...
                zeros( 'like', net.layers{ lid }.filters );
            net.layers{ lid }.biasesMomentum = ...
                zeros( 'like', net.layers{ lid }.biases );
        end
        if ~isfield( net.layers{ lid }, 'filtersWeightDecay' )
            net.layers{ lid }.filtersWeightDecay = 1;
            net.layers{ lid }.biasesWeightDecay = 0;
        end
        if ~isfield( net.layers{ lid }, 'filtersLearningRate' )
            net.layers{ lid }.filtersLearningRate = 1;
            net.layers{ lid }.biasesLearningRate = 2;
        end
    end
    srcCnn.layers = net.layers;
    % Do the job.
    imsvrTr = ImServer( setting.imsvrTr );
    imsvrTr.setNumThreads( 8 );
    cnn = Cnn( db, imsvrTr, setting.cnn );
    cnn.setUseGpu( useGpu );
    cnn.fetchAvgIm( srcImsvrTr.averageImage );
    cnn.fetchCnn( srcCnn.layers, cnnName )
    
end

