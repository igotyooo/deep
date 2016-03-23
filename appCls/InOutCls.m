classdef InOutCls < handle
    properties
        db;                     % A general db.
        rgbMean;                % A task specific RGB mean to normalize images.
        patchSide;
        stride;
        numChannel;
        poolTr;                 % Training sample pool, where all samples are used up in an epoch.
        poolVal;                % Validation sample pool, where all samples are used up in an epoch.
        tsMetricName;           % Name of task specific evaluation metric;
        settingTsNet;           % Setting for the task specific net.
        settingGeneral;         % Setting for image processing.
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public interface. Net will be trained with the following functions only. %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = public )
        function this = InOutCls...
                ( db, settingTsNet, settingGeneral )
            this.db = db;
            this.tsMetricName = { 'cls err1'; };
            % Parameters for task specific net.
            global path;
            pretrainedNet = path.net.vgg_m;
            this.settingTsNet.pretrainedNetName = pretrainedNet.name;
            this.settingTsNet.suppressPretrainedLayerLearnRate  = 1 / 10;
            % Parameters to provide batches.
            this.settingGeneral.shuffleSequance = true;
            this.settingGeneral.batchSize = 128;
            % Apply user setting.
            this.settingTsNet = setChanges...
                ( this.settingTsNet, settingTsNet, upper( mfilename ) );
            this.settingTsNet.pretrainedNetPath = pretrainedNet.path;
            this.settingGeneral = setChanges...
                ( this.settingGeneral, settingGeneral, upper( mfilename ) );
        end
        % Prepare for all data to be used.
        function init( this )
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
        % Majorly used in net. Provide a tr/val batch of I/O pairs.
        function [ ims, gts, sid2iid ] = provdBchTr( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolTr.sid2iid ),
                % Make training pool to be consumed in an epoch.
                [   this.poolTr.sid2iid, ...
                    this.poolTr.sid2flip, ...
                    this.poolTr.sid2gt ] = ...
                    this.getRegnSeqInEpch( 1 );
            end;
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolTr.sid2iid( batchSmpl );
            sid2flip = this.poolTr.sid2flip( batchSmpl );
            sid2gt = this.poolTr.sid2gt( batchSmpl );
            this.poolTr.sid2iid( 1 : batchSize ) = [  ];
            this.poolTr.sid2flip( 1 : batchSize ) = [  ];
            this.poolTr.sid2gt( 1 : batchSize ) = [  ];
            iid2impath = this.db.iid2impath( this.db.iid2setid == 1 ); % tr!!!
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2flip, sid2gt );
        end
        function [ ims, gts, sid2iid ] = provdBchVal( this )
             batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolVal.sid2iid ),
                % Make training pool to be consumed in an epoch.
                [   this.poolVal.sid2iid, ...
                    this.poolVal.sid2flip, ...
                    this.poolVal.sid2gt ] = ...
                    this.getRegnSeqInEpch( 2 );
            end;
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolVal.sid2iid( batchSmpl );
            sid2flip = this.poolVal.sid2flip( batchSmpl );
            sid2gt = this.poolVal.sid2gt( batchSmpl );
            this.poolVal.sid2iid( 1 : batchSize ) = [  ];
            this.poolVal.sid2flip( 1 : batchSize ) = [  ];
            this.poolVal.sid2gt( 1 : batchSize ) = [  ];
            iid2impath = this.db.iid2impath( this.db.iid2setid == 2 ); % val!!!
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2flip, sid2gt );
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
            numOutDimCls = numel( this.db.cid2name );
            numOutDim = numOutDimCls;
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
            layers{ end }.type = 'softmaxloss';
            % Form the net in VGG style.
            net.layers = layers;
            net.normalization.averageImage = [  ];
            net.normalization.keepAspect = false;
            net.normalization.border = [ 0, 0 ];
            net.normalization.imageSize = [ dstSide, dstSide, dstCh ];
            net.normalization.interpolation = 'bicubic';
            net.classes.name = this.db.cid2name;
            net.classes.description = net.classes.name;
            netName = this.getTsNetName;
        end
        % Functions to provide information.
        function batchSize = getBatchSize( this )
            batchSize = this.settingGeneral.batchSize;
        end
        function numBchTr = getNumBatchTr( this )
            batchSize = this.settingGeneral.batchSize;
            numBchTr = ceil( sum( this.db.iid2setid == 1 ) / batchSize );
        end
        function numBchVal = getNumBatchVal( this )
            batchSize = this.settingGeneral.batchSize;
            numBchVal = ceil( sum( this.db.iid2setid == 2 ) / batchSize );
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
            output = gather( res( end - 1 ).x );
            if any( isnan( output ) ), error( 'Divergence.' ); end;
            gts = gather( gts );
            pcls = output;
            [ ~, pcls ] = sort( pcls, 3, 'descend' );
            errCls = ~bsxfun( @eq, pcls, gts( :, :, 1, : ) );
            errCls = errCls( :, :, 1, : );
            errCls = sum( errCls( : ) );
            tsMetric = errCls;
        end
        % Function for identification.
        function name = getName( this )
            name = sprintf( 'IOCNET_%s_OF_%s', ...
                this.settingGeneral.changes, ...
                this.db.name );
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
                ( this, iid2impath, sid2iid, sid2flip, sid2gt )
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
                im = idx2im{ sid2idx( sid ) };
                [ r, c, ~ ] = size( im );
                wind = [ 1; 1; r; c; ];
                if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                im = normalizeAndCropImage( im, wind, this.rgbMean, interpolation );
                if sid2flip( sid ), im = fliplr( im ); end;
                if r ~= dstSide || c ~= dstSide,
                    sid2im( :, :, :, sid ) = imresize...
                        ( im, [ dstSide, dstSide ], interpolation );
                else
                    sid2im( :, :, :, sid ) = im;
                end;
            end;
            ims = sid2im;
            % Merge ground-truths into a same shape of net output.
            gts = reshape( sid2gt, [ 1, 1, size( sid2gt, 1 ), numSmpl ] );
        end
        function [ sid2iid, sid2flip, sid2gt ] = ...
                getRegnSeqInEpch( this, setid )
            rng( 'shuffle' );
            shuffleSequance = this.settingGeneral.shuffleSequance;
            batchSize = this.settingGeneral.batchSize;
            iid2cid = this.db.oid2cid( cell2mat( this.db.iid2oids ) );
            iid2cid = iid2cid( this.db.iid2setid == setid );
            numim = numel( iid2cid );
            numSample = ceil( numim / batchSize ) * batchSize;
            sid2iid = zeros( numSample, 1, 'single' );
            sid2flip = zeros( numSample, 1, 'single' );
            sid2gt = zeros( 1, numSample, 'single' );
            sid = 0; iid = 0;
            while true,
                sid = sid + 1;
                if sid > numim, iid = ceil( numim * rand ); else iid = iid + 1; end;
                sid2iid( sid ) = iid;
                sid2flip( sid ) = round( rand );
                sid2gt( sid ) = iid2cid( iid );
                if sid == numSample, break; end;
            end;
            if shuffleSequance,
                sids = randperm( numSample )';
                sid2iid = sid2iid( sids );
                sid2gt = sid2gt( sids );
            end;
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
        % Function for task-specific network identification.
        function name = getTsNetName( this )
            name = sprintf( ...
                'CNET_%s_%s', ...
                this.settingTsNet.pretrainedNetName, ...
                this.settingTsNet.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
end
