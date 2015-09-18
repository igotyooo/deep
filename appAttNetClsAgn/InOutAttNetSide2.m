classdef InOutAttNetSide2 < handle
    properties
        db;                     % A general db.
        tsDb;                   % A task specific db to be made. If it is unnecessary, just fetch db.
        rgbMean;                % A task specific RGB mean to normalize images.
        patchSide;
        stride;
        numChannel;
        scales;
        directions;
        poolTr;                 % Training sample pool, where all samples are used up in an epoch.
        poolVal;                % Validation sample pool, where all samples are used up in an epoch.
        tsMetricName;           % Name of task specific evaluation metric;
        settingTsDb;            % Setting for the task specific db.
        settingTsNet;           % Setting for the task specific net.
        settingGeneral;         % Setting for image processing.
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public interface. Net will be trained with the following functions only. %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = public )
        function this = InOutAttNetSide2...
                ( db, settingTsDb, settingTsNet, settingGeneral )
            this.db = db;
            this.tsMetricName = { 'cls err1'; 'dir err1'; };
            % Parameters for task specific db.
            this.settingTsDb.numScaling                         = 256;
            this.settingTsDb.dilate                             = 1 / 4;
            % Parameters for task specific db: Define directions.
            this.settingTsDb.directionVectorMagnitude           = 10;       % pixcels.
            this.settingTsDb.numMaxRegionPerDirectionPair       = 16;
            % Parameters for task specific db: positive mining.
            this.settingTsDb.posMinMargin                       = 0.1;
            this.settingTsDb.posIntOverRegnMoreThan             = 1 / 4;    % A target object should be large enough.
            this.settingTsDb.posIntOverTarObjMoreThan           = 0.99;     % A target object should be fully-included.
            this.settingTsDb.posIntOverSubObjMoreThan           = 0.7;      % A sub-object should be majorly-included.
            this.settingTsDb.posIntMajorityMoreThan             = 2;        % A target object should large enough w.r.t. the sub-objects.
            % Parameters for task specific db: semi-negative mining.
            this.settingTsDb.snegIntOverRegnMoreThan            = 1 / 36;	% A sub-object should not be too small.
            this.settingTsDb.snegIntOverObjMoreThan             = 0.3;      % The region is truncating the target object.
            this.settingTsDb.snegIntOverObjLessThan             = 0.7;      % The region is truncating the target object.
            % Parameters for task specific db: negative mining.
            this.settingTsDb.negIntOverObjLessThan              = 0.1;      % Very small overlap is allowed for background region.
            % Parameters for task specific net.
            global path;
            pretrainedNet                                       = path.net.vgg_m;
            this.settingTsNet.pretrainedNetName                 = pretrainedNet.name;
            this.settingTsNet.suppressPretrainedLayerLearnRate  = 1 / 10;
            % Parameters to provide batches.
            this.settingGeneral.shuffleSequance                 = false;
            this.settingGeneral.batchSize                       = 256;
            this.settingGeneral.numGoSmaplePerObj               = 1;
            this.settingGeneral.numAnyDirectionSmaplePerObj     = 1;
            this.settingGeneral.numStopSmaplePerObj             = 1;
            this.settingGeneral.numTruncatedSmaplePerObj        = 1;
            this.settingGeneral.numBackgroundSmaplePerObj       = 1;
            this.settingGeneral.maxNumTrainingSamplePerEpoch    = 1000000;
            this.settingGeneral.maxNumValidationSamplePerEpoch  = 10000;
            % Apply user setting.
            this.settingTsDb = setChanges...
                ( this.settingTsDb, settingTsDb, upper( mfilename ) );
            this.settingTsNet = setChanges...
                ( this.settingTsNet, settingTsNet, upper( mfilename ) );
            this.settingTsNet.pretrainedNetPath = pretrainedNet.path;
            this.settingGeneral = setChanges...
                ( this.settingGeneral, settingGeneral, upper( mfilename ) );
        end
        % Prepare for all data to be used.
        function init( this )
            % Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            this.directions.did2vecT = [ [ 1; 0; ], [ 0; 0; ] ];
            this.directions.did2vecL = [ [ 0; 1; ], [ 0; 0; ] ];
            this.directions.did2vecB = - this.directions.did2vecT;
            this.directions.did2vecR = - this.directions.did2vecL;
            this.directions.dpid2dp = [ ...
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2; ...
                1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2; ...
                1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2; ...
                1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2; ];
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Determine patch stride and side.
            fprintf( '%s: Determine stride and patch side.\n', ...
                upper( mfilename ) );
            net = this.provdInitNet;
            [ this.patchSide, this.stride ] = ...
                getNetProperties( net, numel( net.layers ) - 1 );
            this.numChannel = size( net.layers{ 1 }.weights{ 1 }, 3 );
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Compute a rgb mean vector.
            fpath = this.getRgbMeanPath;
            try
                fprintf( '%s: Try to load RGB-mean.\n', upper( mfilename ) );
                data = load( fpath );
                this.rgbMean = data.data.rgbMean;
            catch
                data.rgbMean = this.computeRgbMean;
                this.makeRgbMeanDir;
                save( fpath, 'data' );
                this.rgbMean = data.rgbMean;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Determine scaling factors.
            fpath = this.getScaleFactorPath;
            try
                fprintf( '%s: Try to load scaling factors.\n', upper( mfilename ) );
                data = load( fpath );
                this.scales = data.data.scales;
            catch
                fprintf( '%s: Determine scaling factors.\n', ...
                    upper( mfilename ) );
                posIntOverRegnMoreThan = this.settingTsDb.posIntOverRegnMoreThan;
                numScaling = this.settingTsDb.numScaling;
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
                referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr, numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                fprintf( '%s: Done.\n', upper( mfilename ) );
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Initialize sample pool.
            this.poolTr.sid2iid = [  ];
            this.poolTr.sid2tlbr = [  ];
            this.poolTr.sid2flip = [  ];
            this.poolTr.sid2gt = [  ];
            this.poolVal.sid2iid = [  ];
            this.poolVal.sid2tlbr = [  ];
            this.poolVal.sid2flip = [  ];
            this.poolVal.sid2gt = [  ];
        end
        function makeTsDb( this )
            % Reform the general db to a task-specific format.
            fpath = this.getTsDbPath;
            try
                fprintf( '%s: Try to load ts-db.\n', ...
                    upper( mfilename ) );
                data = load( fpath );
                this.tsDb = data.data.tsDb;
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen ts-db.\n', ...
                    upper( mfilename ) );
                rng( 'shuffle' );
                data.tsDb.tr = this.makeSubTsDb( 1 );
                data.tsDb.val = this.makeSubTsDb( 2 );
                this.tsDb = data.tsDb;
                fprintf( '%s: Save ts-db.\n', ...
                    upper( mfilename ) );
                this.makeTsDbDir;
                save( fpath, 'data', '-v7.3' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end;
        end
        function demo( this, fid, setid )
            if setid == 1,
                [ ims, gts, iids ] = this.provdBchTr;
                tsdb = this.tsDb.tr;
            else
                [ ims, gts, iids ] = this.provdBchVal;
                tsdb = this.tsDb.val;
            end;
            numCls = this.db.getNumClass;
            numDir = 2 * 2 * 2 * 2;
            bsize = this.settingGeneral.batchSize;
            figure( fid );
            set( gcf, 'color', 'w' );
            for s = 1 : bsize,
                im = ims( :, :, :, s );
                gt = gts( :, :, :, s );
                gt = gt( : );
                im = uint8( bsxfun( @plus, im, this.rgbMean ) );
                if gt( 1 ) <= numCls,
                    cname = this.db.cid2name{ gt( 1 ) };
                elseif gt( 1 ) == numCls + 1,
                    cname = 'trunc';
                elseif gt( 1 ) == numCls + 2,
                    cname = 'bgd';
                end;
                if gt( 2 ) <= numDir,
                    dname = num2str( this.directions.dpid2dp( :, gt( 2 ) )' );
                    dname( dname == ' ' ) = '';
                    dname( dname == '1' ) = 'g';
                    dname( dname == '2' ) = 's';
                elseif gt( 2 ) == numDir + 1,
                    dname = 'trunc';
                elseif gt( 2 ) == numDir + 2,
                    dname = 'bgd';
                end;
                [ ~, iid ] = fileparts( tsdb.iid2impath{ iids( s ) } );
                iid = str2double( iid );
                subplot( 1, 2, 1 );
                plottlbr( this.db.oid2bbox( :, this.db.iid2oids{ iid } ), this.db.iid2impath{ iid }, false, 'r' );
                title( 'Ground-truth' );
                subplot( 1, 2, 2 );
                imshow( im ); title( sprintf( '%s, %s (%d/%d)', cname, dname, s, bsize ) );
                waitforbuttonpress;
            end;
        end;
        % Majorly used in net. Provide a tr/val batch of I/O pairs.
        function [ ims, gts, sid2iid ] = provdBchTr( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolTr.sid2iid ),
                % Make training pool to be consumed in an epoch.
                [   this.poolTr.sid2iid, ...
                    this.poolTr.sid2tlbr, ...
                    this.poolTr.sid2flip, ...
                    this.poolTr.sid2gt ] = ...
                    this.getRegnSeqInEpch( 1 );
            end;
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolTr.sid2iid( batchSmpl );
            sid2tlbr = this.poolTr.sid2tlbr( :, batchSmpl );
            sid2flip = this.poolTr.sid2flip( batchSmpl );
            sid2gt = this.poolTr.sid2gt( :, batchSmpl );
            this.poolTr.sid2iid( 1 : batchSize ) = [  ];
            this.poolTr.sid2tlbr( :, 1 : batchSize ) = [  ];
            this.poolTr.sid2flip( 1 : batchSize ) = [  ];
            this.poolTr.sid2gt( :, 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.tr.iid2impath; % tr!!!
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2tlbr, sid2flip, sid2gt );
        end
        function [ ims, gts, sid2iid ] = provdBchVal( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolVal.sid2iid ),
                % Make validation pool to be consumed in an epoch.
                [   this.poolVal.sid2iid, ...
                    this.poolVal.sid2tlbr, ...
                    this.poolVal.sid2flip, ...
                    this.poolVal.sid2gt ] = ...
                    this.getRegnSeqInEpch( 2 );
            end;
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolVal.sid2iid( batchSmpl );
            sid2tlbr = this.poolVal.sid2tlbr( :, batchSmpl );
            sid2flip = this.poolVal.sid2flip( batchSmpl );
            sid2gt = this.poolVal.sid2gt( :, batchSmpl );
            this.poolVal.sid2iid( 1 : batchSize ) = [  ];
            this.poolVal.sid2tlbr( :, 1 : batchSize ) = [  ];
            this.poolVal.sid2flip( 1 : batchSize ) = [  ];
            this.poolVal.sid2gt( :, 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.val.iid2impath; % val!!!
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2tlbr, sid2flip, sid2gt );
        end
        function [ net, netName ] = provdInitNet( this )
            % Set parameters.
            preTrainedNetPath = ...
                this.settingTsNet.pretrainedNetPath;
            suppPtdLyrLearnRate = ...
                this.settingTsNet.suppressPretrainedLayerLearnRate;
            preTrainedFilterLearningRate = 1 * suppPtdLyrLearnRate;
            preTrainedBiasLearningRate = 2 * suppPtdLyrLearnRate;
            initFilterLearningRate = 1;
            initBiasLearningRate = 2;
            filterWeightDecay = 1;
            biasWeightDecay = 0;
            dstSide = 219;
            dstCh = 3;
            % Load pre-trained net.
            srcNet = load( preTrainedNetPath );
            % Initilaize mementum and set learning rate.
            layers = srcNet.layers;
            for lid = 1 : numel( layers )
                if isfield( layers{ lid }, 'weights' ),
                    % Initialize momentum.
                    layers{ lid }.momentum{ 1 } = ...
                        zeros( size( layers{ lid }.weights{ 1 } ), 'single' );
                    layers{ lid }.momentum{ 2 } = ...
                        zeros( size( layers{ lid }.weights{ 2 } ), 'single' );
                    % Set learning rate in the order of [ filter, bias ].
                    layers{ lid }.learningRate = [ ...
                        preTrainedFilterLearningRate, ...
                        preTrainedBiasLearningRate ];
                    layers{ lid }.weightDecay = ...
                        [ filterWeightDecay, biasWeightDecay ];
                end;
            end
            % Initialize the output layer.
            filterChannel = size( layers{ end - 3 }.weights{ 1 }, 4 ); 
            numOutDimCls = numel( this.db.cid2name ) + 2;
            numDirPair = 2 * 2 * 2 * 2;
            numOutDimDir = numDirPair + 2;
            numOutDim = numOutDimCls + numOutDimDir;
            weight = 0.01 * randn( 1, 1, filterChannel, numOutDim, 'single' );
            bias = zeros( 1, numOutDim, 'single' );
            layers{ end - 1 } = struct(...
                'type', 'conv', ...
                'name', 'target-fc', ...
                'weights', { { weight, bias } }, ...
                'stride', 1, ...
                'pad', 0, ...
                'learningRate', [ initFilterLearningRate, initBiasLearningRate ], ...
                'weightDecay', [ filterWeightDecay, biasWeightDecay ] );
            layers{ end - 1 }.momentum{ 1 } = ...
                zeros( size( layers{ end - 1 }.weights{ 1 } ), 'single' );
            layers{ end - 1 }.momentum{ 2 } = ...
                zeros( size( layers{ end - 1 }.weights{ 2 } ), 'single' );
            layers{ end }.type = 'custom';
            layers{ end }.dimCls = 1 : numOutDimCls;
            layers{ end }.dimDir = numOutDimCls + ( 1 : numOutDimDir );
            layers{ end }.forward = @InOutAttNetSide2.forward;
            layers{ end }.backward = @InOutAttNetSide2.backward;
            % Form the net in VGG style.
            net.layers = layers;
            net.normalization.averageImage = [  ];
            net.normalization.keepAspect = false;
            net.normalization.border = [ 0, 0 ];
            net.normalization.imageSize = [ dstSide, dstSide, dstCh ];
            net.normalization.interpolation = 'bicubic';
            net.classes.name = this.db.cid2name;
            net.classes.name{ end + 1 } = 'truncating';
            net.classes.name{ end + 1 } = 'background';
            net.classes.name{ end + 1 } = 'TLBR: go, go, go, go';
            net.classes.name{ end + 1 } = 'TLBR: go, go, go, stop';
            net.classes.name{ end + 1 } = 'TLBR: go, go, stop, go';
            net.classes.name{ end + 1 } = 'TLBR: go, go, stop, stop';
            net.classes.name{ end + 1 } = 'TLBR: go, stop, go, go';
            net.classes.name{ end + 1 } = 'TLBR: go, stop, go, stop';
            net.classes.name{ end + 1 } = 'TLBR: go, stop, stop, go';
            net.classes.name{ end + 1 } = 'TLBR: go, stop, stop, stop';
            net.classes.name{ end + 1 } = 'TLBR: stop, go, go, go';
            net.classes.name{ end + 1 } = 'TLBR: stop, go, go, stop';
            net.classes.name{ end + 1 } = 'TLBR: stop, go, stop, go';
            net.classes.name{ end + 1 } = 'TLBR: stop, go, stop, stop';
            net.classes.name{ end + 1 } = 'TLBR: stop, stop, go, go';
            net.classes.name{ end + 1 } = 'TLBR: stop, stop, go, stop';
            net.classes.name{ end + 1 } = 'TLBR: stop, stop, stop, go';
            net.classes.name{ end + 1 } = 'TLBR: stop, stop, stop, stop';
            net.classes.name{ end + 1 } = 'truncating';
            net.classes.name{ end + 1 } = 'background';
            net.classes.description = net.classes.name;
            netName = this.getTsNetName;
        end
        % Functions to provide information.
        function batchSize = getBatchSize( this )
            batchSize = this.settingGeneral.batchSize;
        end
        function numBchTr = getNumBatchTr( this )
            numAugPerObj = this.getNumSamplePerObj; 
            batchSize = this.settingGeneral.batchSize;
            numObj = numel( this.tsDb.tr.oid2iid );
            numSample = ceil( numObj * numAugPerObj / batchSize ) * batchSize;
            numBchTr = numSample / batchSize;
        end
        function numBchVal = getNumBatchVal( this )
            numAugPerObj = this.getNumSamplePerObj; 
            batchSize = this.settingGeneral.batchSize;
            numObj = numel( this.tsDb.val.oid2iid );
            numSample = ceil( numObj * numAugPerObj / batchSize ) * batchSize;
            numBchVal = numSample / batchSize;
        end
        function numSample = getNumSamplePerObj( this )
            % 1) A pos for proposal, 2) a pos for various direction, 3) a pos for stop, 4) a semi-neg, 5) and neg.
            numGoSmaplePerObj = this.settingGeneral.numGoSmaplePerObj;
            numAnyDirectionSmaplePerObj = this.settingGeneral.numAnyDirectionSmaplePerObj;
            numStopSmaplePerObj = this.settingGeneral.numStopSmaplePerObj;
            numTruncatedSmaplePerObj = this.settingGeneral.numTruncatedSmaplePerObj;
            numBackgroundSmaplePerObj = this.settingGeneral.numBackgroundSmaplePerObj;
            numSample = numGoSmaplePerObj + ...
                numAnyDirectionSmaplePerObj + ...
                numStopSmaplePerObj + ...
                numTruncatedSmaplePerObj + ...
                numBackgroundSmaplePerObj; 
        end
        function tsMetricName = getTsMetricName( this )
            tsMetricName = this.tsMetricName;
        end
        function numTsMetric = getNumTsMetric( this )
            numTsMetric = numel( this.tsMetricName );
        end
        % Majorly used in net. Update energy and task-specific evaluation metric.
        % For this target application, object detection, 
        % the metric is top-1 accuracy.
        function tsMetric = computeTsMetric( this, res, gts )
            numOutDimCls = numel( this.db.cid2name ) + 2;
            numDirPair = 2 * 2 * 2 * 2;
            numOutDimDir = numDirPair + 2;
            dimCls = 1 : numOutDimCls;
            dimDir = numOutDimCls + ( 1 : numOutDimDir );
            output = gather( res( end - 1 ).x );
            gts = gather( gts );
            % Compute object class error.
            pcls = output( :, :, dimCls, : );
            [ ~, pcls ] = sort( pcls, 3, 'descend' );
            errCls = ~bsxfun( @eq, pcls, gts( :, :, 1, : ) );
            errCls = errCls( :, :, 1, : );
            errCls = sum( errCls( : ) );
            % Compute direcion error.
            pdir = output( :, :, dimDir, : );
            [ ~, pdir ] = sort( pdir, 3, 'descend' );
            errDir = ~bsxfun( @eq, pdir, gts( :, :, 2, : ) );
            errDir = errDir( :, :, 1, : );
            errDir = sum( errDir( : ) );
            tsMetric = [ errCls; errDir; ];
        end
        % Function for identification.
        function name = getName( this )
            name = sprintf( 'IOANETSID2_%s_OF_%s', ...
                this.settingGeneral.changes, ...
                this.getTsDbName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%
    % Private interface. %
    %%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        function rgbMean = computeRgbMean( this )
            dstCh = this.numChannel;
            idx2iid = this.db.getTriids;
            numIm = numel( idx2iid );
            ok = randperm( numIm, min( 10000, numIm ) );
            idx2iid = idx2iid( ok );
            numIm = numel( idx2iid );
            batchSize = 256;
            idx2rgb = zeros( 1, 1, dstCh, numIm, 'single' );
            cnt = 0;
            cummt = 0; bcnt = 0;
            for i = 1 : batchSize : numIm,
                bcnt = bcnt + 1;
                btime = tic;
                biids = idx2iid( i : min( i + batchSize - 1, numIm ) );
                bimpaths = this.db.iid2impath( biids );
                ims = vl_imreadjpeg( bimpaths, 'numThreads', 12 );
                for j = 1 : numel( ims ),
                    im = ims{ j };
                    if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                    if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                    cnt = cnt + 1;
                    [ r, c, ~ ] = size( im );
                    idx2rgb( :, :, :, cnt ) = sum( sum( im, 1 ), 2 ) / ( r * c );
                end;
                cummt = cummt + toc( btime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numel( 1 : batchSize : numIm ), bcnt, ...
                    'Compute rgb mean.', cummt );
            end;
            rgbMean = mean( idx2rgb, 4 );
        end
        function [ ims, gts ] = makeImGtPairs...
                ( this, iid2impath, sid2iid, sid2tlbr, sid2flip, sid2gt )
            % Set Params.
            dstSide = this.patchSide;
            dstCh = this.numChannel;
            interpolation = 'bicubic';
            % Read images.
            [ idx2iid, ~, sid2idx ] = unique( sid2iid );
            idx2impath = iid2impath( idx2iid );
            idx2im = vl_imreadjpeg( idx2impath, 'numThreads', 12 );
            % Do the job.
            numSmpl = numel( sid2iid );
            sid2im = zeros( dstSide, dstSide, dstCh, numSmpl, 'single' );
            for sid = 1 : numSmpl;
                wind = sid2tlbr( :, sid );
                im = idx2im{ sid2idx( sid ) };
                if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                im = normalizeAndCropImage( im, wind, this.rgbMean, interpolation );
                if sid2flip( sid ), im = fliplr( im ); end;
                sid2im( :, :, :, sid ) = imresize...
                    ( im, [ dstSide, dstSide ], interpolation );
            end;
            ims = sid2im;
            % Merge ground-truths into a same shape of net output.
            gts = reshape( sid2gt, [ 1, 1, size( sid2gt, 1 ), numSmpl ] );
        end
        function [ sid2iid, sid2tlbr, sid2flip, sid2gt ] = ...
                getRegnSeqInEpch( this, setid )
            rng( 'shuffle' );
            shuffleSequance = this.settingGeneral.shuffleSequance;
            batchSize = this.settingGeneral.batchSize;
            numGoSmaplePerObj = this.settingGeneral.numGoSmaplePerObj;
            numAnyDirectionSmaplePerObj = this.settingGeneral.numAnyDirectionSmaplePerObj;
            numStopSmaplePerObj = this.settingGeneral.numStopSmaplePerObj;
            numTruncatedSmaplePerObj = this.settingGeneral.numTruncatedSmaplePerObj;
            numBackgroundSmaplePerObj = this.settingGeneral.numBackgroundSmaplePerObj;
            if setid == 1, subTsDb = this.tsDb.tr; else subTsDb = this.tsDb.val; end;
            numObj = numel( subTsDb.oid2iid );
            numAugPerObj = this.getNumSamplePerObj;
            numSample = ceil( numObj * numAugPerObj / batchSize ) * batchSize;
            truncIdCls = max( subTsDb.oid2cid ) + 1;
            truncIdDir = 2 * 2 * 2 * 2 + 1;
            bgdIdCls = truncIdCls + 1;
            bgdIdDir = truncIdDir + 1;
            sid2iid = zeros( numSample, 1, 'single' );
            sid2tlbr = zeros( 4, numSample, 'single' );
            sid2flip = zeros( numSample, 1, 'single' );
            sid2gt = zeros( 2, numSample, 'single' );
            sid = 0; oid = 0; iter = 0;
            while true,
                iter = iter + 1;
                if iter > numObj, oid = ceil( numObj * rand ); else oid = oid + 1; end;
                iid = subTsDb.oid2iid( oid );
                % Sample positive regions - for initial proposal.
                for n = 1 : numGoSmaplePerObj,
                    if sid == numSample, break; end;
                    dpid = 1;
                    flip = round( rand );
                    if flip,
                        regns = subTsDb.oid2dpid2posregnsFlip{ oid }{ dpid };
                    else
                        regns = subTsDb.oid2dpid2posregns{ oid }{ dpid };
                    end;
                    numRegn = size( regns, 2 );
                    if numRegn,
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( :, sid ) = [ subTsDb.oid2cid( oid ); dpid ];
                    end;
                end;
                if sid == numSample, break; end;
                % Sample positive regions - for various directions.
                for n = 1 : numAnyDirectionSmaplePerObj,
                    if sid == numSample, break; end;
                    flip = round( rand );
                    if flip,
                        dpid2posregns = subTsDb.oid2dpid2posregnsFlip{ oid };
                    else
                        dpid2posregns = subTsDb.oid2dpid2posregns{ oid };
                    end;
                    dpid2ok = ~cellfun( @isempty, dpid2posregns );
                    dpid2ok( 1 ) = false;
                    dpid2ok( end ) = false;
                    if sum( dpid2ok ),
                        dpid = find( dpid2ok );
                        dpid = dpid( ceil( numel( dpid ) * rand ) );
                        regns = dpid2posregns{ dpid };
                        numRegn = size( regns, 2 );
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( :, sid ) = [ subTsDb.oid2cid( oid ); dpid ];
                    end;
                end;
                if sid == numSample, break; end;
                % Sample positive regions - for stop.
                for n = 1 : numStopSmaplePerObj,
                    if sid == numSample, break; end;
                    dpid = 2 * 2 * 2 * 2;
                    flip = round( rand );
                    if flip,
                        regns = subTsDb.oid2dpid2posregnsFlip{ oid }{ dpid };
                    else
                        regns = subTsDb.oid2dpid2posregns{ oid }{ dpid };
                    end;
                    numRegn = size( regns, 2 );
                    if numRegn,
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( :, sid ) = [ subTsDb.oid2cid( oid ); dpid ];
                    end;
                end;
                if sid == numSample, break; end;
                % Sample semi-negative regions.
                for n = 1 : numTruncatedSmaplePerObj,
                    if sid == numSample, break; end;
                    regns = subTsDb.oid2snegregns{ oid };
                    numRegn = size( regns, 2 );
                    if numRegn,
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = round( rand );
                        sid2gt( :, sid ) = [ truncIdCls; truncIdDir; ];
                    end;
                end;
                if sid == numSample, break; end;
                % Sample negative regions.
                for n = 1 : numBackgroundSmaplePerObj,
                    if sid == numSample, break; end;
                    regns = subTsDb.iid2sid2negregns{ iid };
                    s2ok = ~cellfun( @isempty, regns );
                    if sum( s2ok ),
                        s = find( s2ok );
                        s = s( ceil( numel( s ) * rand ) );
                        regns = regns{ s };
                        numRegn = size( regns, 2 );
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = round( rand );
                        sid2gt( :, sid ) = [ bgdIdCls; bgdIdDir; ];
                    end;
                end;
                if sid == numSample, break; end;
            end;
            if shuffleSequance,
                sids = randperm( numSample )';
                sid2iid = sid2iid( sids );
                sid2tlbr = sid2tlbr( :, sids );
                sid2gt = sid2gt( :, sids );
            end;
        end
        function subTsDb = makeSubTsDb( this, setid )
            % Set parameters.
            numSize = size( this.scales, 2 );
            numDirPair = size( this.directions.dpid2dp, 2 );
            dilate = this.settingTsDb.dilate;
            domainWarp = [ this.patchSide; this.patchSide; ];
            % Parameters for positive mining.
            posMinMargin = this.settingTsDb.posMinMargin;
            posIntOverRegnMoreThan = this.settingTsDb.posIntOverRegnMoreThan;       % A target object should be large enough.
            posIntOverTarObjMoreThan = this.settingTsDb.posIntOverTarObjMoreThan;   % A target object should be fully-included.
            posIntOverSubObjMoreThan = this.settingTsDb.posIntOverSubObjMoreThan;   % A sub-object should be majorly-included.
            posIntMajorityMoreThan = this.settingTsDb.posIntMajorityMoreThan;       % A target object should large enough w.r.t. the sub-objects.
            directionVectorMagnitude = this.settingTsDb.directionVectorMagnitude;
            numMaxRegionPerDirectionPair = this.settingTsDb.numMaxRegionPerDirectionPair;
            % Parameters for semi-negative mining.
            snegIntOverRegnMoreThan = this.settingTsDb.snegIntOverRegnMoreThan;     % A sub-object should not be too small.
            snegIntOverObjMoreThan = this.settingTsDb.snegIntOverObjMoreThan;       % The region is truncating the target object.
            snegIntOverObjLessThan = this.settingTsDb.snegIntOverObjLessThan;       % The region is truncating the target object.
            % Parameters for negative mining.
            negIntOverObjLessThan = this.settingTsDb.negIntOverObjLessThan;         % Very small overlap is allowed for background region.
            % Do the job.
            newiid2iid = find( this.db.iid2setid == setid );
            newiid2iid = newiid2iid( randperm( numel( newiid2iid ) )' );
            newoid2oid = cat( 1, this.db.iid2oids{ newiid2iid } );
            newoid2newiid = zeros( size( newoid2oid ), 'single' );
            newoid2dpid2posregns = cell( size( newoid2oid ) );
            newoid2dpid2posregnsFlip = cell( size( newoid2oid ) );
            newoid2cid = zeros( size( newoid2oid ), 'single' );
            newoid2snegregns = cell( size( newoid2oid ) );
            newiid2sid2negregns = cell( size( newiid2iid ) );
            newiid2impath = this.db.iid2impath( newiid2iid );
            numIm = numel( newiid2iid );
            newoid = 1; cummt = 0; 
            for newiid = 1 : numIm;
                itime = tic;
                iid = newiid2iid( newiid );
                imSize0 = this.db.iid2size( :, iid );
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2oids{ iid } );
                oid2cid = this.db.oid2cid( this.db.iid2oids{ iid } );
                numObj = size( oid2tlbr, 2 );
                % Extract candidate regions and compute bisic informations.
                sid2size = round( bsxfun( @times, this.scales, imSize0 ) );
                rid2tlbr = ...
                    extractDenseRegions( ...
                    imSize0, ...
                    sid2size, ...
                    this.patchSide, ...
                    this.stride, ...
                    dilate );
                if isempty( rid2tlbr ), rid2rect = zeros( 5, 0 ); rid2tlbr = zeros( 5, 0 );
                else rid2rect = tlbr2rect( rid2tlbr ); end;
                rid2area = prod( rid2rect( 3 : 4, : ), 1 )';
                oid2rect = tlbr2rect( oid2tlbr );
                oid2area = prod( oid2rect( 3 : 4, : ), 1 )';
                rid2oid2int = rectint( rid2rect', oid2rect' );
                rid2oid2ioo = bsxfun( @times, rid2oid2int, 1 ./ oid2area' );
                rid2oid2ior = bsxfun( @times, rid2oid2int, 1 ./ rid2area );
                % Positive mining.
                % 1. A target object should be fully-included.
                % 2. A target object should occupy a large enough area.
                % 3. A target object should large enough w.r.t. the sub-objects.
                rid2oid2fullinc = rid2oid2ioo > posIntOverTarObjMoreThan;
                rid2oid2big = rid2oid2ior > posIntOverRegnMoreThan;
                rid2oid2fullandbig = rid2oid2fullinc & rid2oid2big;
                rid2tararea = max( rid2oid2int .* rid2oid2fullandbig, [  ], 2 );
                rid2oid2incarea = rid2oid2int .* ( rid2oid2ioo > posIntOverSubObjMoreThan );
                rid2oid2incarea = bsxfun( @times, rid2oid2incarea, 1 ./ rid2tararea );
                foo = rid2oid2incarea >= ( 1 / posIntMajorityMoreThan );
                rid2oid2fullandbig = rid2oid2fullandbig & foo;
                rid2major = sum( foo, 2 ) == 1;
                [ prid2oid, ~ ] = find( rid2oid2fullandbig( rid2major, : )' );
                rid2ok = any( rid2oid2fullandbig, 2 ) & rid2major;
                prid2tlbr = rid2tlbr( 1 : 4, rid2ok );
                oid2mgtlbr = scaleBoxes( oid2tlbr, 1 + 2 * posMinMargin, 1 + 2 * posMinMargin );
                oid2mgtlbr = round( oid2mgtlbr );
                oid2dpid2posregns = cell( numObj, 1 );
                oid2dpid2posregnsFlip = cell( numObj, 1 );
                for oid = 1 : numObj,
                    oid2dpid2posregns{ oid } = cell( numDirPair, 1 );
                    oid2dpid2posregnsFlip{ oid } = cell( numDirPair, 1 );
                    prids = prid2oid == oid;
                    % Clipping the regeion boundaries.
                    if sum( prids ),
                        objRegns = prid2tlbr( :, prids );
                        objRegns( 1, objRegns( 1, : ) > oid2mgtlbr( 1, oid ) ) = oid2mgtlbr( 1, oid );
                        objRegns( 2, objRegns( 2, : ) > oid2mgtlbr( 2, oid ) ) = oid2mgtlbr( 2, oid );
                        objRegns( 3, objRegns( 3, : ) < oid2mgtlbr( 3, oid ) ) = oid2mgtlbr( 3, oid );
                        objRegns( 4, objRegns( 4, : ) < oid2mgtlbr( 4, oid ) ) = oid2mgtlbr( 4, oid );
                    else
                        objRegns = oid2mgtlbr( :, oid );
                    end;
                    % Compute side directions.
                    dpidBasis = 2 .^ ( 3 : -1 : 0 );
                    mgtlbr = oid2mgtlbr( :, oid );
                    for r = 1 : size( objRegns, 2 ),
                        regnCurr = objRegns( :, r );
                        while true,
                            [ didT, didL, didB, didR, didTFlip, didLFlip, didBFlip, didRFlip, regnNext ] = ...
                                getGtSideDirection( regnCurr, mgtlbr, directionVectorMagnitude, domainWarp );
                            if didT == 2, regnCurr( 1 ) = mgtlbr( 1 ); end;
                            if didL == 2, regnCurr( 2 ) = mgtlbr( 2 ); end;
                            if didB == 2, regnCurr( 3 ) = mgtlbr( 3 ); end;
                            if didR == 2, regnCurr( 4 ) = mgtlbr( 4 ); end;
                            dpid = sum( dpidBasis .* ( [ didT, didL, didB, didR ] - 1 ) ) + 1;
                            dpidFlip = sum( dpidBasis .* ( [ didTFlip, didLFlip, didBFlip, didRFlip ] - 1 ) ) + 1;
                            regn = round( regnCurr );
                            oid2dpid2posregns{ oid }{ dpid } = [ oid2dpid2posregns{ oid }{ dpid }, regn ];
                            oid2dpid2posregnsFlip{ oid }{ dpidFlip } = [ oid2dpid2posregnsFlip{ oid }{ dpidFlip }, regn ];
                            regnCurr = regnNext;
                            if didT == 2 && didL == 2 && didB == 2 && didR == 2, break; end;
                        end;
                    end;
                    % Clipping the number of regeions per a direction pair.
                    for dpid = 1 : numDirPair,
                        numRegnPerDirPair = size( oid2dpid2posregns{ oid }{ dpid }, 2 );
                        ok = randperm( numRegnPerDirPair, min( numRegnPerDirPair, numMaxRegionPerDirectionPair ) );
                        oid2dpid2posregns{ oid }{ dpid } = oid2dpid2posregns{ oid }{ dpid }( :, ok );
                    end;
                    for dpid = 1 : numDirPair,
                        numRegnPerDirPair = size( oid2dpid2posregnsFlip{ oid }{ dpid }, 2 );
                        ok = randperm( numRegnPerDirPair, min( numRegnPerDirPair, numMaxRegionPerDirectionPair ) );
                        oid2dpid2posregnsFlip{ oid }{ dpid } = oid2dpid2posregnsFlip{ oid }{ dpid }( :, ok );
                    end;
                end;
                % Display for debugging.
                % for oid = 1 : numObj,
                %     for dpid = 1 : numDirPair,
                %         im = this.db.iid2impath{ iid };
                %         if isempty( oid2dpid2posregns{ oid }{ dpid } ), continue; end;
                %         figure( 1 ); plottlbr( oid2dpid2posregns{ oid }{ dpid }, im, false, 'r' );
                %         title( sprintf( 'No flip, %s', num2str( this.directions.dpid2dp( :, dpid )' ) ) );
                %         if isempty( oid2dpid2posregnsFlip{ oid }{ dpid } ), continue; end;
                %         figure( 2 ); plottlbr( oid2dpid2posregnsFlip{ oid }{ dpid }, im, false, 'r' );
                %         title( sprintf( 'Flip, %s', num2str( this.directions.dpid2dp( :, dpid )' ) ) );
                %         waitforbuttonpress;
                %     end;
                % end;
                % Semi-negative mining.
                % The region is truncating the target object.
                numMaxRegnPerObj = numMaxRegionPerDirectionPair * numDirPair;
                rid2oid2issub = rid2oid2ior >= snegIntOverRegnMoreThan;
                rid2issub = any( rid2oid2issub, 2 );
                rid2oid2ok = snegIntOverObjMoreThan <= rid2oid2ioo & ...
                    snegIntOverObjLessThan >= rid2oid2ioo;
                rid2ok = rid2issub & all( eq( rid2oid2issub, rid2oid2ok ), 2 );
                snrid2tlbr = rid2tlbr( 1 : 4, rid2ok );
                snrid2oid2ok = rid2oid2ok( rid2ok, : );
                oid2snregns = cell( numObj, 1 );
                for oid = 1 : numObj,
                    snrids = snrid2oid2ok( :, oid );
                    if sum( snrids ),
                        oid2snregns{ oid } = snrid2tlbr( :, snrids );
                    else
                        tlbr = oid2tlbr( :, oid );
                        w = floor( ( tlbr( 4 ) - tlbr( 2 ) + 1 ) / 2 );
                        h = floor( ( tlbr( 3 ) - tlbr( 1 ) + 1 ) / 2 );
                        snregns = repmat( tlbr, 1, 4 );
                        snregns( 1, 1 ) = snregns( 1, 1 ) + h;
                        snregns( 2, 2 ) = snregns( 2, 2 ) + w;
                        snregns( 3, 3 ) = snregns( 3, 3 ) - h;
                        snregns( 4, 4 ) = snregns( 4, 4 ) - w;
                        oid2snregns{ oid } = snregns;
                    end;
                    numRegnPerObj = size( oid2snregns{ oid }, 2 );
                    ok = randperm( numRegnPerObj, min( numRegnPerObj, numMaxRegnPerObj ) );
                    oid2snregns{ oid } = oid2snregns{ oid }( :, ok );
                end;
                % Negative mining.
                % Very small overlap is allowed for background region.
                numMaxRegnPerSize = max( 1, round( numMaxRegionPerDirectionPair * numDirPair / numSize ) );
                rid2ok = all( rid2oid2ioo <= negIntOverObjLessThan, 2 );
                nrid2tlbr = rid2tlbr( :, rid2ok );
                sid2nregns = cell( numSize, 1 );
                for sid = 1 : numSize,
                    nrids = find( nrid2tlbr( 5, : ) == sid );
                    nrids = randsample( nrids, min( numel( nrids ), numMaxRegnPerSize ) );
                    sid2nregns{ sid } = nrid2tlbr( 1 : 4, nrids );
                end;
                newoids = ( newoid : newoid + numObj - 1 )';
                newoid2newiid( newoids ) = newiid;
                newoid2cid( newoids ) = oid2cid;
                newoid2dpid2posregns( newoids ) = oid2dpid2posregns;
                newoid2dpid2posregnsFlip( newoids ) = oid2dpid2posregnsFlip;
                newoid2snegregns( newoids ) = oid2snregns;
                newiid2sid2negregns{ newiid } = sid2nregns;
                newoid = newoid + numObj;
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, newiid, ...
                    sprintf( 'make regions on im %d', newiid ), cummt );
            end;
            subTsDb.iid2impath = newiid2impath;
            subTsDb.oid2iid = newoid2newiid;
            subTsDb.oid2cid = newoid2cid;
            subTsDb.oid2dpid2posregns = newoid2dpid2posregns;
            subTsDb.oid2dpid2posregnsFlip = newoid2dpid2posregnsFlip;
            subTsDb.oid2snegregns = newoid2snegregns;
            subTsDb.iid2sid2negregns = newiid2sid2negregns;
        end
        % Functions for file IO.
        function name = getRgbMeanName( this )
            name = sprintf( ...
                'RGBM_OF_%s', ...
                this.db.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getRgbMeanDir( this )
            dir = this.db.getDir;
        end
        function dir = makeRgbMeanDir( this )
            dir = this.getRgbMeanDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getRgbMeanPath( this )
            fname = strcat( this.getRgbMeanName, '.mat' );
            path = fullfile( this.getRgbMeanDir, fname );
        end
        function name = getScaleFactorName( this )
            numScaling = this.settingTsDb.numScaling;
            name = sprintf( ...
                'SFTR_N%03d_OF_%s', numScaling, this.db.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getScaleFactorDir( this )
            dir = this.db.getDir;
        end
        function dir = makeScaleFactorDir( this )
            dir = this.getScaleFactorDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getScaleFactorPath( this )
            fname = strcat( this.getScaleFactorName, '.mat' );
            path = fullfile( this.getScaleFactorDir, fname );
        end
        function name = getTsDbName( this )
            name = sprintf( ...
                'DBANETSID_%s_OF_%s', ...
                this.settingTsDb.changes, ...
                this.db.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getTsDbDir( this )
            dir = this.db.getDir;
        end
        function dir = makeTsDbDir( this )
            dir = this.getTsDbDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getTsDbPath( this )
            fname = strcat( this.getTsDbName, '.mat' );
            path = fullfile( this.getTsDbDir, fname );
        end
        % Function for task-specific network identification.
        function name = getTsNetName( this )
            name = sprintf( ...
                'ANETSID2_%s_%s', ...
                this.settingTsNet.pretrainedNetName, ...
                this.settingTsNet.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    methods( Static )
        % Majorly used in net, if a layer is custom. Forward function in output layer.
        function res2 = forward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numLy = 2;
            ycls = vl_nnsoftmaxloss...
                ( X( :, :, ly.dimCls, : ), gt( :, :, 1, : ) );
            ydir = vl_nnsoftmaxloss...
                ( X( :, :, ly.dimDir, : ), gt( :, :, 2, : ) );
            res2.x = [ ycls; ydir; ] / numLy;
        end
        % % Majorly used in net, if a layer is custom. Backward function in output layer.
        function res1 = backward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numLy = 2;
            dzdyCls = res2.dzdx / numLy;
            ycls = vl_nnsoftmaxloss...
                ( X( :, :, ly.dimCls, : ), gt( :, :, 1, : ), dzdyCls );
            dzdyDir = res2.dzdx / numLy;
            ydir = vl_nnsoftmaxloss...
                ( X( :, :, ly.dimDir, : ), gt( :, :, 2, : ), dzdyDir );
            res1.dzdx = cat( 3, ycls, ydir );
        end
    end
end