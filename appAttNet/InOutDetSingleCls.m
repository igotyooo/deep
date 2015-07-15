% IN:  DB.
% OUT: Images-GT paies per epoch.
% Task-specific implementation required.
classdef InOutDetSingleCls < handle
    properties
        srcDb;                  % A general db.
        tsDb;                   % A task specific db to be made. If it is unnecessary, just fetch srcDb.
        rgbMean;                % A task specific RGB mean to normalize images.
        numBchTr;               % Number of training batches in an epoch.
        numBchVal;              % Number of validation batches in an epoch.
        poolTr;                 % Training sample pool, where all samples are used up in an epoch.
        poolVal;                % Validation sample pool, where all samples are used up in an epoch.
        poolRemainTr;           % Remaining training sample pool. It is initilized for each epoch of net training.
        poolRemainVal;          % Remaining validation sample pool. It is initilized for each epoch of net training.
        did2dvecTl;             % Quantized directional vectors for top-left.
        did2dvecBr;             % Quantized directional vectors for bottom-right.
        signStop;               % Class id of stop prediction.
        signNoObj;              % Class id of absence prediction.
        tsMetricName;           % Name of task specific evaluation metric;
        settingTsDb;            % Setting for the task specific db.
        settingTsNet;           % Setting for the task specific net.
        settingGeneral;         % Setting for image processing.
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public interface. Net will be trained with the following functions only. %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = public )
        function this = InOutDetSingleCls( srcDb, settingTsDb, settingTsNet, settingGeneral )
            this.srcDb = srcDb;
            this.tsMetricName = 'Top-1 err';
            % Task specific) Default parameters for task specific db.
            this.settingTsDb.selectClassName                    = 'person';
            this.settingTsDb.stride                             = 32;
            this.settingTsDb.dstSide                            = 227;
            this.settingTsDb.numScale                           = 8;
            this.settingTsDb.scaleStep                          = 2;
            this.settingTsDb.docScaleMag                        = 4;
            this.settingTsDb.numAspect                          = 16;
            this.settingTsDb.confidence                         = 0.97;
            this.settingTsDb.insectOverFgdObj                   = 0.5;
            this.settingTsDb.insectOverFgdObjForMajority        = 0.1;
            this.settingTsDb.fgdObjMajority                     = 1.5;
            this.settingTsDb.insectOverBgdObj                   = 0.2;
            this.settingTsDb.insectOverBgdRegn                  = 0.5;
            this.settingTsDb.insectOverBgdRegnForReject         = 0.9;
            this.settingTsDb.numDirection                       = 3;
            this.settingTsDb.numMaxBgdRegnPerScale              = 100;
            this.settingTsDb.stopSignError                      = 5;
            this.settingTsDb.minObjScale                        = 1 / sqrt( 2 );
            this.settingTsDb.numErode                           = 5;
            % Task specific) Default parameters for task specific net.
            global path;
            this.settingTsNet.pretrainedNetName                 = path.net.vgg_m_2048.name;
            this.settingTsNet.suppressPretrainedLayerLearnRate  = 1 / 10;
            % General) Default parameters to provide batches.
            this.settingGeneral.dstSide                         = 227;
            this.settingGeneral.dstCh                           = 3;
            this.settingGeneral.batchSize                       = 256;
            % Apply user setting.
            this.settingTsDb = setChanges...
                ( this.settingTsDb, settingTsDb, upper( mfilename ) );
            this.settingTsNet = setChanges...
                ( this.settingTsNet, settingTsNet, upper( mfilename ) );
            this.settingGeneral = setChanges...
                ( this.settingGeneral, settingGeneral, upper( mfilename ) );
        end
        % Prepare for all data to be used.
        function init( this )
            % Task specific) Compute a rgb mean vector.
            this.computeRgbMean;
            % Task specific) Define derections.
            numDirection = this.settingTsDb.numDirection;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.did2dvecTl = vertcat...
                ( cos( did2angTl' ), sin( did2angTl' ) );
            this.did2dvecBr = vertcat...
                ( cos( did2angBr' ), sin( did2angBr' ) );
            this.signStop = size( this.did2dvecTl, 2 ) + 1;
            this.signNoObj = this.signStop + 1;
            % General) Reform the general source db format to a task-specific format.
            this.makeTsDb;
            % General) Make training pool to be consumed in an epoch.
            batchSize = this.settingGeneral.batchSize;
            [   this.poolTr.sid2iid, ...
                this.poolTr.sid2regn, ...
                this.poolTr.sid2dp ] = this.getRegnSeqInEpch( 1 );
            this.numBchTr = numel( this.poolTr.sid2iid ) / batchSize;
            % General) Make validation pool to be consumed in an epoch.
            [   this.poolVal.sid2iid, ...
                this.poolVal.sid2regn, ...
                this.poolVal.sid2dp ] = this.getRegnSeqInEpch( 2 );
            this.numBchVal = numel( this.poolVal.sid2iid ) / batchSize;
        end
        % Majorly used in net. Provide a tr/val batch of I/O pairs.
        function [ ims, gts ] = provdBchTr( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolTr.sid2iid )
                % Make training pool to be consumed in an epoch.
                [   this.poolTr.sid2iid, ...
                    this.poolTr.sid2regn, ...
                    this.poolTr.sid2dp ] = this.getRegnSeqInEpch( 1 );
                this.numBchTr = numel( this.poolTr.sid2iid ) / batchSize;
            end
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolTr.sid2iid( batchSmpl );
            sid2regn = this.poolTr.sid2regn( :, batchSmpl );
            sid2dp = this.poolTr.sid2dp( :, batchSmpl );
            this.poolTr.sid2iid( 1 : batchSize ) = [  ];
            this.poolTr.sid2regn( :, 1 : batchSize ) = [  ];
            this.poolTr.sid2dp( :, 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.tr.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2regn, sid2dp );
        end
        function [ ims, gts ] = provdBchVal( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolVal.sid2iid )
                % Make validation pool to be consumed in an epoch.
                [   this.poolVal.sid2iid, ...
                    this.poolVal.sid2regn, ...
                    this.poolVal.sid2dp ] = this.getRegnSeqInEpch( 2 );
                this.numBchVal = numel( this.poolVal.sid2iid ) / batchSize;
            end
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolVal.sid2iid( batchSmpl );
            sid2regn = this.poolVal.sid2regn( :, batchSmpl );
            sid2dp = this.poolVal.sid2dp( :, batchSmpl );
            this.poolVal.sid2iid( 1 : batchSize ) = [  ];
            this.poolVal.sid2regn( :, 1 : batchSize ) = [  ];
            this.poolVal.sid2dp( :, 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.val.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2regn, sid2dp );
        end
        function [ net, netName ] = provdInitNet( this )
            global path;
            % Set parameters.
            suppPtdLyrLearnRate = this.settingTsNet.suppressPretrainedLayerLearnRate;
            preTrainedFilterLearningRate = 1 * suppPtdLyrLearnRate;
            preTrainedBiasLearningRate = 2 * suppPtdLyrLearnRate;
            initFilterLearningRate = 1;
            initBiasLearningRate = 2;
            filterWeightDecay = 1;
            biasWeightDecay = 0;
            dstSide = this.settingTsDb.dstSide;
            dstCh = this.settingGeneral.dstCh;
            numOutDim = ( 3 + 1 + 1 ) * 2;
            % Load pre-trained net.
            srcNet = load( path.net.vgg_m_2048.path );
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
                    layers{ lid }.learningRate = ...
                        [ preTrainedFilterLearningRate, preTrainedBiasLearningRate ];
                    layers{ lid }.weightDecay = [ filterWeightDecay, biasWeightDecay ];
                end;
            end
            % Initialize the directional layer.
            layers{ end - 1 } = struct(...
                'type', 'conv', ...
                'name', 'directional-fc', ...
                'weights', { { 0.01 * randn( 1, 1, 2048, numOutDim, 'single' ), zeros( 1, numOutDim, 'single' ) } }, ...
                'stride', 1, ...
                'pad', 0, ...
                'learningRate', [ initFilterLearningRate, initBiasLearningRate ], ...
                'weightDecay', [ filterWeightDecay, biasWeightDecay ] );
            layers{ end - 1 }.momentum{ 1 } = ...
                zeros( size( layers{ end - 1 }.weights{ 1 } ), 'single' );
            layers{ end - 1 }.momentum{ 2 } = ...
                zeros( size( layers{ end - 1 }.weights{ 2 } ), 'single' );
            % Initialize the loss layer.
            layers{ end }.type = 'custom';
            layers{ end }.backward = @InOutDetSingleCls.backward;
            layers{ end }.forward = @InOutDetSingleCls.forward;
            % Form the net in VGG style.
            net.layers = layers;
            net.normalization.averageImage = [  ];
            net.normalization.keepAspect = false;
            net.normalization.border = [ 0, 0 ];
            net.normalization.imageSize = [ dstSide, dstSide, dstCh ];
            net.normalization.interpolation = 'bicubic';
            net.classes.name = cell( 1, numOutDim );
            net.classes.name{ 1 }   = 'TL: bottom';
            net.classes.name{ 2 }   = 'TL: bottom-right';
            net.classes.name{ 3 }   = 'TL: right';
            net.classes.name{ 4 }   = 'TL: stop';
            net.classes.name{ 5 }   = 'TL: no object';
            net.classes.name{ 6 }   = 'BR: up';
            net.classes.name{ 7 }   = 'BR: up-left';
            net.classes.name{ 8 }   = 'BR: left';
            net.classes.name{ 9 }   = 'BR: stop';
            net.classes.name{ 10 }  = 'BR: no object';
            net.classes.description = net.classes.name;
            netName = this.getTsNetName;
        end
        % Functions to provide information.
        function batchSize = getBatchSize( this )
            batchSize = this.settingGeneral.batchSize;
        end
        function numBchTr = getNumBatchTr( this )
            numBchTr = this.numBchTr;
        end
        function numBchVal = getNumBatchVal( this )
            numBchVal = this.numBchVal;
        end
        function imsize = getImSize( this )
            dstSide = this.settingGeneral.dstSide;
            dstCh = this.settingGeneral.dstCh;
            imsize = [ dstSide; dstSide; dstCh; ];
        end
        function tsMetricName = getTsMetricName( this )
            tsMetricName = this.tsMetricName;
        end
        % Function for identification.
        function name = getName( this )
            name = sprintf( 'IODETSINGLECLS_%s_OF_%s', ...
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
        function computeRgbMean( this )
            fpath = this.getRgbMeanPath;
            try
                data = load( fpath );
                this.rgbMean = data.rgbMean;
            catch
                dstCh = this.settingGeneral.dstCh;
                idx2iid = this.srcDb.getTriids;
                numIm = numel( idx2iid );
                batchSize = 256;
                idx2rgb = zeros( 1, 1, dstCh, numIm, 'single' );
                cnt = 0;
                cummt = 0; bcnt = 0;
                for i = 1 : batchSize : numIm,
                    bcnt = bcnt + 1;
                    btime = tic;
                    biids = idx2iid( i : min( i + batchSize - 1, numIm ) );
                    bimpaths = this.srcDb.iid2impath( biids );
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
                this.makeRgbMeanDir;
                save( fpath, 'rgbMean' );
                this.rgbMean = rgbMean;
            end;
        end
        function [ ims, gts ] = makeImGtPairs...
                ( this, iid2impath, sid2iid, sid2regn, sid2dp )
            % Set Params.
            dstSide = this.settingGeneral.dstSide;
            dstCh = this.settingGeneral.dstCh;
            % Read images.
            [ idx2iid, ~, sid2idx ] = unique( sid2iid );
            idx2impath = iid2impath( idx2iid );
            idx2im = vl_imreadjpeg( idx2impath, 'numThreads', 12 );
            % Do the job.
            numSmpl = numel( sid2iid );
            sid2im = zeros( dstSide, dstSide, dstCh, numSmpl, 'single' );
            for sid = 1 : numSmpl;
                wind = sid2regn( :, sid );
                im = idx2im{ sid2idx( sid ) };
                if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                im = cropAndNormalizeIm( im, size( im ), wind, this.rgbMean );
                if sid2regn( 5, sid ) == 2, im = fliplr( im ); end;
                sid2im( :, :, :, sid ) = imresize...
                    ( im, [ dstSide, dstSide ], 'bicubic' );
            end;
            ims = sid2im;
            % Merge ground-truths into a same shape of net output.
            gts = reshape( sid2dp, [ 1, 1, size( sid2dp, 1 ), size( sid2dp, 2 ) ] );
        end
        function [ sid2iid, sid2regn, sid2dp ] = ...
                getRegnSeqInEpch( this, setid )
            if setid == 1, db = this.tsDb.tr; else db = this.tsDb.val; end;
            batchSize = this.settingGeneral.batchSize;
            rng( 'shuffle' );
            % 1/4: Square regions of fore-ground object.
            oid2iid = db.fgdObj.oid2iid;
            oid2dpid2regns = db.fgdObj.oid2dpid2regnsSqr;
            dpid2dids = db.dpid2dids;
            [   fgdObjSqr.sid2iid, ...
                fgdObjSqr.sid2regn, ...
                fgdObjSqr.sid2dp ] = ...
                InOutDetSingleCls.makeFgdObjRegnSequence...
                ( oid2iid, oid2dpid2regns, dpid2dids );
            % 1/4: Non-square regions of fore-ground object.
            oid2dpid2regns = db.fgdObj.oid2dpid2regnsNsqr;
            [   fgdObjNsqr.sid2iid, ...
                fgdObjNsqr.sid2regn, ...
                fgdObjNsqr.sid2dp ] = ...
                InOutDetSingleCls.makeFgdObjRegnSequence...
                ( oid2iid, oid2dpid2regns, dpid2dids );
            numFgdSmpl = numel( fgdObjSqr.sid2iid ) + ...
                numel( fgdObjNsqr.sid2iid );
            % 1/8: Square regions of back-ground object.
            oid2iid = db.bgdObj.oid2iid;
            oid2regns = db.bgdObj.oid2regnsSqr;
            [   bgdObjSqr.sid2iid, ...
                bgdObjSqr.sid2regn ] = ...
                InOutDetSingleCls.makeBgdObjRegnSequence...
                ( oid2iid, oid2regns, round( numFgdSmpl / 4 ) );
            bgdObjSqr.sid2dp = this.signNoObj * ...
                ones( 2, numel( bgdObjSqr.sid2iid ), 'single' );
            % 1/8: Non-square regions of back-ground object.
            oid2regns = db.bgdObj.oid2regnsNsqr;
            [   bgdObjNsqr.sid2iid, ...
                bgdObjNsqr.sid2regn ] = ...
                InOutDetSingleCls.makeBgdObjRegnSequence...
                ( oid2iid, oid2regns, round( numFgdSmpl / 4 ) );
            bgdObjNsqr.sid2dp = this.signNoObj * ...
                ones( 2, numel( bgdObjNsqr.sid2iid ), 'single' );
            numBgdObjSmpl = numel( bgdObjSqr.sid2iid ) + ...
                numel( bgdObjNsqr.sid2iid );
            % 1/8: Square regions of Back-grounds.
            iid2sid2regns = db.bgd.iid2sid2regnsSqr;
            [   bgdSqr.sid2iid, ...
                bgdSqr.sid2regn ] = ...
                InOutDetSingleCls.makeBgdRegnSequence...
                ( iid2sid2regns, round( ( numFgdSmpl - numBgdObjSmpl ) / 2 ) );
            bgdSqr.sid2dp = this.signNoObj * ...
                ones( 2, numel( bgdSqr.sid2iid ), 'single' );
            % 1/8: Non-square regions of Back-grounds.
            iid2sid2regns = db.bgd.iid2sid2regnsNsqr;
            [   bgdNsqr.sid2iid, ...
                bgdNsqr.sid2regn ] = ...
                InOutDetSingleCls.makeBgdRegnSequence...
                ( iid2sid2regns, round( ( numFgdSmpl - numBgdObjSmpl ) / 2 ) );
            bgdNsqr.sid2dp = this.signNoObj * ...
                ones( 2, numel( bgdNsqr.sid2iid ), 'single' );
            % Concatenation.
            sid2iid = cat( 1, ...
                fgdObjSqr.sid2iid, ...
                fgdObjNsqr.sid2iid, ...
                bgdObjSqr.sid2iid, ...
                bgdObjNsqr.sid2iid, ...
                bgdSqr.sid2iid, ...
                bgdNsqr.sid2iid );
            sid2regn = cat( 2, ...
                fgdObjSqr.sid2regn, ...
                fgdObjNsqr.sid2regn, ...
                bgdObjSqr.sid2regn, ...
                bgdObjNsqr.sid2regn, ...
                bgdSqr.sid2regn, ...
                bgdNsqr.sid2regn );
            sid2dp = cat( 2, ...
                fgdObjSqr.sid2dp, ...
                fgdObjNsqr.sid2dp, ...
                bgdObjSqr.sid2dp, ...
                bgdObjNsqr.sid2dp, ...
                bgdSqr.sid2dp, ...
                bgdNsqr.sid2dp );
            % Supply more samples for suitable number of batchs.
            numSmpl = numel( sid2iid );
            numSupp = batchSize * ceil( numSmpl / batchSize ) - numSmpl;
            suppSids = randsample( numSmpl, numSupp );
            sid2iid = cat( 1, sid2iid, sid2iid( suppSids ) );
            sid2regn = cat( 2, sid2regn, sid2regn( :, suppSids ) );
            sid2dp = cat( 2, sid2dp, sid2dp( :, suppSids ) );
            % Shupple.
            numSmpl = numel( sid2iid );
            perm = randperm( numSmpl );
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
            sid2dp = sid2dp( :, perm );
        end
        function makeTsDb( this )
            fpath = this.getTsDbPath;
            try
                fprintf( '%s: Try to load ts db.\n', ...
                    upper( mfilename ) );
                data = load( fpath );
                db = data.db;
                fprintf( '%s: Ts db loaded.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen ts db.\n', ...
                    upper( mfilename ) );
                % Set parameters.
                db.tr = makeSubTsDb( this, 1 );
                db.val = makeSubTsDb( this, 2 );
                fprintf( '%s: Save sub-db.\n', ...
                    upper( mfilename ) );
                this.makeTsDbDir;
                save( fpath, 'db', '-v7.3' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
            this.tsDb = db;
        end
        function subDb = makeSubTsDb( this, setid )
            targetClsName = this.settingTsDb.selectClassName;
            stride = this.settingTsDb.stride;
            dstSide = this.settingTsDb.dstSide;
            numScale = this.settingTsDb.numScale;
            scaleStep = this.settingTsDb.scaleStep;
            docScaleMag = this.settingTsDb.docScaleMag;
            numAspect = this.settingTsDb.numAspect;
            confidence = this.settingTsDb.confidence;
            insectOverFgdObj = this.settingTsDb.insectOverFgdObj;
            insectOverFgdObjForMajority = this.settingTsDb.insectOverFgdObjForMajority;
            fgdObjMajority = this.settingTsDb.fgdObjMajority;
            insectOverBgdObj = this.settingTsDb.insectOverBgdObj;
            insectOverBgdRegn = this.settingTsDb.insectOverBgdRegn;
            insectOverBgdRegnForReject = this.settingTsDb.insectOverBgdRegnForReject;
            numMaxBgdRegnPerScale = this.settingTsDb.numMaxBgdRegnPerScale;
            stopSignError = this.settingTsDb.stopSignError;
            minObjScale = this.settingTsDb.minObjScale;
            numErode = this.settingTsDb.numErode;
            % Do the job.
            targetClsId = cellfun( @( cname )strcmp( cname, targetClsName ), this.srcDb.cid2name );
            targetClsId = find( targetClsId );
            oid2bbox = this.srcDb.oid2bbox( :, this.srcDb.oid2cid == targetClsId );
            aspects = determineAspectRates( oid2bbox, numAspect - 1, confidence );
            aspects = unique( cat( 1, 1, aspects ) );
            scales = scaleStep .^ ( 0 : 0.5 : 0.5 * ( numScale - 1 ) );
            idx2iid = find( this.srcDb.iid2setid == setid );
            numIm = numel( idx2iid );
            [ didsTl, didsBr ] = meshgrid( 1 : this.signStop, 1 : this.signStop );
            dpid2dids = single( [ didsTl( : ), didsBr( : ) ]' );
            numDirPair = size( dpid2dids, 2 );
            iid2impath = this.srcDb.iid2impath( idx2iid );
            fgdObj.oid2iid = cell( numIm, 1 );
            fgdObj.oid2dpid2regnsSqr = cell( numIm, 1 );
            fgdObj.oid2dpid2regnsNsqr = cell( numIm, 1 );
            bgdObj.oid2iid = cell( numIm, 1 );
            bgdObj.oid2regnsSqr = cell( numIm, 1 );
            bgdObj.oid2regnsNsqr = cell( numIm, 1 );
            bgd.iid2sid2regnsSqr = cell( numIm, numScale );
            bgd.iid2sid2regnsNsqr = cell( numIm, numScale );
            cummt = 0;
            for newiid = 1 : numIm;
                itime = tic;
                iid = idx2iid( newiid );
                imSize = this.srcDb.iid2size( :, iid );
                oidx2oid = find( this.srcDb.oid2iid == iid );
                oidx2cid = this.srcDb.oid2cid( oidx2oid );
                oidx2fgdObj = oidx2cid == targetClsId;
                oidx2bgdObj = ~oidx2fgdObj;
                oidx2tlbr = round( this.srcDb.oid2bbox( :, oidx2oid ) );
                oidx2tlbr = single( oidx2tlbr );
                [ rid2tlbrFgdObj, rid2oidxFgdObj, rid2tlbrBgd, rid2oidxBgd ] = ...
                    extTargetObjRegns( ...
                    oidx2tlbr, ...
                    oidx2fgdObj, ...
                    imSize, ...
                    docScaleMag, ...
                    stride, ...
                    dstSide, ...
                    scales, ...
                    aspects, ...
                    insectOverFgdObj, ...
                    insectOverFgdObjForMajority, ...
                    fgdObjMajority, ...
                    insectOverBgdObj, ...
                    insectOverBgdRegn, ...
                    insectOverBgdRegnForReject );
                % Classify regions in three ways: 
                % 1. Fore-ground object regions: rid2tlbrFgdObj
                % 2. Back-ground object regions: rid2tlbrBgdObj
                % 3. Back-ground regions: rid2tlbrBgd
                rid2bgdObj = logical( rid2oidxBgd );
                rid2oidxBgdObj = rid2oidxBgd( rid2bgdObj );
                rid2oidxBgdObj = rid2oidxBgdObj( : );
                rid2tlbrBgdObj = rid2tlbrBgd( :, rid2bgdObj );
                rid2tlbrBgd( :, rid2bgdObj ) = [  ];
                rid2tlbrFgdObj = single( round( rid2tlbrFgdObj ) );
                rid2tlbrBgdObj = single( round( rid2tlbrBgdObj ) );
                rid2tlbrBgd = single( round( rid2tlbrBgd ) );
                rid2rectBgd = tlbr2rect( rid2tlbrBgd );
                rid2areaBgd = prod( rid2rectBgd( 3 : 4, : ), 1 );
                rid2insctBtwImAndBgd = rectint( tlbr2rect( [ 1; 1; imSize ] )', rid2rectBgd' );
                rid2insctBtwImAndBgd = rid2insctBtwImAndBgd ./ rid2areaBgd;
                rid2okBgd = rid2insctBtwImAndBgd > 0.7;
                rid2tlbrBgd = rid2tlbrBgd( :, rid2okBgd );
                % Fore-ground objects.
                rid2fgdObjTlbr = oidx2tlbr( 1 : 4, rid2oidxFgdObj );
                rid2outOfBndTlr = ( rid2tlbrFgdObj( 1, : ) - rid2fgdObjTlbr( 1, : ) ) > 0;
                rid2outOfBndTlc = ( rid2tlbrFgdObj( 2, : ) - rid2fgdObjTlbr( 2, : ) ) > 0;
                rid2outOfBndBrr = ( rid2tlbrFgdObj( 3, : ) - rid2fgdObjTlbr( 3, : ) ) < 0;
                rid2outOfBndBrc = ( rid2tlbrFgdObj( 4, : ) - rid2fgdObjTlbr( 4, : ) ) < 0;
                rid2fgdObjTlbr( 1, rid2outOfBndTlr ) = rid2tlbrFgdObj( 1, rid2outOfBndTlr );
                rid2fgdObjTlbr( 2, rid2outOfBndTlc ) = rid2tlbrFgdObj( 2, rid2outOfBndTlc );
                rid2fgdObjTlbr( 3, rid2outOfBndBrr ) = rid2tlbrFgdObj( 3, rid2outOfBndBrr );
                rid2fgdObjTlbr( 4, rid2outOfBndBrc ) = rid2tlbrFgdObj( 4, rid2outOfBndBrc );
                rid2fgdObjTlbr( 1 : 2, : ) = rid2fgdObjTlbr( 1 : 2, : ) - rid2tlbrFgdObj( 1 : 2, : ) + 1;
                rid2fgdObjTlbr( 3 : 4, : ) = rid2fgdObjTlbr( 3 : 4, : ) - rid2tlbrFgdObj( 1 : 2, : ) + 1;
                slop = bsxfun( @times, [ dstSide; dstSide; ] - 1, 1 ./ ( rid2tlbrFgdObj( 3 : 4, : ) - rid2tlbrFgdObj( 1 : 2, : ) ) );
                rid2fgdObjTlbr( [ 1, 3 ], : ) = bsxfun( @times, slop( 1, : ), ( rid2fgdObjTlbr( [ 1, 3 ], : ) - 1 ) ) + 1;
                rid2fgdObjTlbr( [ 2, 4 ], : ) = bsxfun( @times, slop( 2, : ), ( rid2fgdObjTlbr( [ 2, 4 ], : ) - 1 ) ) + 1;
                rid2fgdObjTlbr = round( rid2fgdObjTlbr );
                rid2dvecTl = rid2fgdObjTlbr( 1 : 2, : ) - 1;
                rid2dvecNormTl = sqrt( sum( rid2dvecTl .^ 2, 1 ) );
                rid2dvecNormTl( rid2dvecNormTl < stopSignError ) = 0;
                rid2dvecTl = bsxfun( @times, rid2dvecTl, 1./ rid2dvecNormTl );
                rid2stopTl = any( isnan( rid2dvecTl ), 1 );
                [ ~, rid2didTl ] = max( this.did2dvecTl' * rid2dvecTl, [  ], 1 );
                rid2didTl( rid2stopTl ) = this.signStop;
                rid2dvecBr = bsxfun( @minus, rid2fgdObjTlbr( 3 : 4, : ), [ dstSide; dstSide; ] );
                rid2dvecNormBr = sqrt( sum( rid2dvecBr .^ 2, 1 ) );
                rid2dvecNormBr( rid2dvecNormBr < stopSignError ) = 0;
                rid2dvecBr = bsxfun( @times, rid2dvecBr, 1./ rid2dvecNormBr );
                rid2stopBr = any( isnan( rid2dvecBr ), 1 );
                [ ~, rid2didBr ] = max( this.did2dvecBr' * rid2dvecBr, [  ], 1 );
                rid2didBr( rid2stopBr ) = this.signStop;
                rid2dids = [ rid2didTl; rid2didBr; ];
                rid2dpid = zeros( size( rid2dids, 2 ), 1 );
                for dpid = 1 : numDirPair,
                    rid2dpid( logical( prod( bsxfun( @minus, rid2dids, dpid2dids( :, dpid ) ) == 0, 1 ) ) ) = dpid;
                end
                rid2fgdObjTlbrHF = rid2fgdObjTlbr;
                rid2fgdObjTlbrHF( [ 2; 4; ], : ) = dstSide - rid2fgdObjTlbrHF( [ 4; 2; ], : ) + 1;
                rid2dvecTlHF = rid2fgdObjTlbrHF( 1 : 2, : ) - 1;
                rid2dvecNormTlHF = sqrt( sum( rid2dvecTlHF .^ 2, 1 ) );
                rid2dvecNormTlHF( rid2dvecNormTlHF < stopSignError ) = 0;
                rid2dvecTlHF = bsxfun( @times, rid2dvecTlHF, 1./ rid2dvecNormTlHF );
                rid2stopTlHF = any( isnan( rid2dvecTlHF ), 1 );
                [ ~, rid2didTlHF ] = max( this.did2dvecTl' * rid2dvecTlHF, [  ], 1 );
                rid2didTlHF( rid2stopTlHF ) = this.signStop;
                rid2dvecBrHF = bsxfun( @minus, rid2fgdObjTlbrHF( 3 : 4, : ), [ dstSide; dstSide; ] );
                rid2dvecNormBrHF = sqrt( sum( rid2dvecBrHF .^ 2, 1 ) );
                rid2dvecNormBrHF( rid2dvecNormBrHF < stopSignError ) = 0;
                rid2dvecBrHF = bsxfun( @times, rid2dvecBrHF, 1./ rid2dvecNormBrHF );
                rid2stopBrHF = any( isnan( rid2dvecBrHF ), 1 );
                [ ~, rid2didBrHF ] = max( this.did2dvecBr' * rid2dvecBrHF, [  ], 1 );
                rid2didBrHF( rid2stopBrHF ) = this.signStop;
                rid2didsHF = [ rid2didTlHF; rid2didBrHF; ];
                rid2dpidHF = zeros( size( rid2didsHF, 2 ), 1 );
                for dpid = 1 : numDirPair,
                    rid2dpidHF( logical( prod( bsxfun( @minus, rid2didsHF, dpid2dids( :, dpid ) ) == 0, 1 ) ) ) = dpid;
                end
                fgdObj.oid2iid{ newiid } = newiid * ones( sum( oidx2fgdObj ), 1, 'single' );
                % Fore-ground objects: square regions.
                rid2sqrFgdObj = rid2tlbrFgdObj( 6, : ) == 1;
                rid2dpidSqr = [ rid2dpid( rid2sqrFgdObj ), rid2dpidHF( rid2sqrFgdObj ) ];
                rid2tlbr = rid2tlbrFgdObj( 1 : 4, rid2sqrFgdObj );
                rid2tlbr = [ [ rid2tlbr; ones( 1, size( rid2tlbr, 2 ) ) ], ...
                    [ rid2tlbr; 2 * ones( 1, size( rid2tlbr, 2 ) ) ] ];
                rid2oidxFgdObjSqr = [ rid2oidxFgdObj( rid2sqrFgdObj ), ...
                    rid2oidxFgdObj( rid2sqrFgdObj ) ];
                if sum( oidx2fgdObj )
                    fgdObj.oid2dpid2regnsSqr{ newiid } = cell( sum( oidx2fgdObj ), 1 );
                end
                oidxsFgdObj = find( oidx2fgdObj );
                for oidxFgd = 1 : sum( oidx2fgdObj ),
                    oidxFgdObj = oidxsFgdObj( oidxFgd );
                    rids = find( rid2oidxFgdObjSqr == oidxFgdObj );
                    dpids = rid2dpidSqr( rids );
                    fgdObj.oid2dpid2regnsSqr{ newiid }{ oidxFgd } = arrayfun( ...
                        @( dpid )rid2tlbr( :, rids( dpids == dpid ) ), ...
                        1 : numDirPair, 'UniformOutput', false );
                end
                % Fore-ground objects: non-square regions.
                rid2nsqrFgdObj = ~rid2sqrFgdObj;
                rid2dpidNsqr = [ rid2dpid( rid2nsqrFgdObj ), rid2dpidHF( rid2nsqrFgdObj ) ];
                rid2tlbr = rid2tlbrFgdObj( 1 : 4, rid2nsqrFgdObj );
                rid2tlbr = [ [ rid2tlbr; ones( 1, size( rid2tlbr, 2 ) ) ], ...
                    [ rid2tlbr; 2 * ones( 1, size( rid2tlbr, 2 ) ) ] ];
                rid2oidxFgdObjNsqr = [ rid2oidxFgdObj( rid2nsqrFgdObj ), ...
                    rid2oidxFgdObj( rid2nsqrFgdObj ) ];
                if sum( oidx2fgdObj )
                    fgdObj.oid2dpid2regnsNsqr{ newiid } = cell( sum( oidx2fgdObj ), 1 );
                end
                for oidxFgd = 1 : sum( oidx2fgdObj ),
                    oidxFgdObj = oidxsFgdObj( oidxFgd );
                    rids = find( rid2oidxFgdObjNsqr == oidxFgdObj );
                    dpids = rid2dpidNsqr( rids );
                    fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd } = arrayfun( ...
                        @( dpid )rid2tlbr( :, rids( dpids == dpid ) ), ...
                        1 : numDirPair, 'UniformOutput', false );
                    objBbox = oidx2tlbr( 1 : 4, oidxFgdObj );
                    objTl = objBbox( 1 : 2 );
                    objBr = objBbox( 3 : 4 );
                    objCenter = ( objTl + objBr ) / 2;
                    objBoxSide = objBr - objTl;
                    shift = objBoxSide * minObjScale / 2;
                    tlout   = objTl;
                    tlin    = round( objCenter - shift );
                    brout   = objBr;
                    brin    = round( objCenter + shift );
                    tlrs = tlout( 1 ) : tlin( 1 );
                    tlcs = tlout( 2 ) : tlin( 2 );
                    brrs = brin( 1 ) : brout( 1 );
                    brcs = brin( 2 ) : brout( 2 );
                    objBboxEroded = zeros( 4, numErode );
                    for e = 1 : numErode
                        tlr = tlrs( ceil( rand * numel( tlrs ) ) );
                        tlc = tlcs( ceil( rand * numel( tlcs ) ) );
                        brr = brrs( ceil( rand * numel( brrs ) ) );
                        brc = brcs( ceil( rand * numel( brcs ) ) );
                        objBboxEroded( :, e ) = [ tlr; tlc; brr; brc; ];
                    end;
                    objBboxEroded = [ [ objBboxEroded; ones( 1, size( objBboxEroded, 2 ) ); ], ...
                        [ objBboxEroded; 2 * ones( 1, size( objBboxEroded, 2 ) ); ] ];
                    fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd }{ end } = ...
                        cat( 2, fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd }{ end }, objBboxEroded );
                end
                % Back-ground objects.
                bgdObj.oid2iid{ newiid } = newiid * ones( sum( oidx2bgdObj ), 1, 'single' );
                % Back-ground objects: square regions.
                rid2sqrBgdObj = rid2tlbrBgdObj( 6, : ) == 1;
                rid2tlbr = rid2tlbrBgdObj( 1 : 4, rid2sqrBgdObj );
                rid2oidxBgdObjSqr = rid2oidxBgdObj( rid2sqrBgdObj );
                bgdObj.oid2regnsSqr{ newiid } = arrayfun( ...
                    @( oid )rid2tlbr( :, rid2oidxBgdObjSqr == oid ), ...
                    find( oidx2bgdObj ), 'UniformOutput', false );
                % Back-ground objects: non-square regions.
                rid2nsqrBgdObj = rid2tlbrBgdObj( 6, : ) ~= 1;
                rid2tlbr = rid2tlbrBgdObj( 1 : 4, rid2nsqrBgdObj );
                rid2oidxBgdObjNsqr = rid2oidxBgdObj( rid2nsqrBgdObj );
                bgdObj.oid2regnsNsqr{ newiid } = arrayfun( ...
                    @( oid )rid2tlbr( :, rid2oidxBgdObjNsqr == oid ), ...
                    find( oidx2bgdObj ), 'UniformOutput', false );
                % Back-ground: squre regions.
                rid2sqrBgd = rid2tlbrBgd( 6, : ) == 1;
                for s = 1 : numScale,
                    ok = rid2sqrBgd & ( rid2tlbrBgd( 5, : ) == s );
                    rids = find( ok );
                    if numel( rids ) > numMaxBgdRegnPerScale,
                        rids = sort( randsample( rids, numMaxBgdRegnPerScale ) );
                    end;
                    bgd.iid2sid2regnsSqr{ newiid, s } = rid2tlbrBgd( 1 : 4, rids );
                end;
                % Back-ground: non-square regions.
                rid2nsqrBgd = rid2tlbrBgd( 6, : ) ~= 1;
                for s = 1 : numScale,
                    ok = rid2nsqrBgd & ( rid2tlbrBgd( 5, : ) == s );
                    rids = find( ok );
                    if numel( rids ) > numMaxBgdRegnPerScale,
                        rids = sort( randsample( rids, numMaxBgdRegnPerScale ) );
                    end;
                    bgd.iid2sid2regnsNsqr{ newiid, s } = rid2tlbrBgd( 1 : 4, rids );
                end;
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop...
                    (   numIm, newiid, ...
                    sprintf( 'ext sub-winds on im %d', newiid ), cummt );
            end;
            fgdObj.oid2iid = cat( 1, fgdObj.oid2iid{ : } );
            fgdObj.oid2dpid2regnsSqr = cat( 1, fgdObj.oid2dpid2regnsSqr{ : } );
            fgdObj.oid2dpid2regnsSqr = cat( 1, fgdObj.oid2dpid2regnsSqr{ : } );
            fgdObj.oid2dpid2regnsNsqr = cat( 1, fgdObj.oid2dpid2regnsNsqr{ : } );
            fgdObj.oid2dpid2regnsNsqr = cat( 1, fgdObj.oid2dpid2regnsNsqr{ : } );
            bgdObj.oid2iid = cat( 1, bgdObj.oid2iid{ : } );
            bgdObj.oid2regnsSqr = cat( 1, bgdObj.oid2regnsSqr{ : } );
            bgdObj.oid2regnsNsqr = cat( 1, bgdObj.oid2regnsNsqr{ : } );
            subDb.iid2impath = iid2impath;
            subDb.dpid2dids = dpid2dids;
            subDb.fgdObj = fgdObj;
            subDb.bgdObj = bgdObj;
            subDb.bgd = bgd;
        end
        % Functions for file IO.
        function name = getRgbMeanName( this )
            name = sprintf( ...
                'RGBM_OF_%s', ...
                this.srcDb.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getRgbMeanDir( this )
            dir = this.srcDb.getDir;
        end
        function dir = makeRgbMeanDir( this )
            dir = this.getRgbMeanDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getRgbMeanPath( this )
            fname = strcat( this.getRgbMeanName, '.mat' );
            path = fullfile( this.getRgbMeanDir, fname );
        end
        
        function name = getTsDbName( this )
            name = sprintf( ...
                'DBTS_%s_OF_%s', ...
                this.settingTsDb.changes, ...
                this.srcDb.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getTsDbDir( this )
            dir = this.srcDb.getDir;
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
                'ANET_%s_%s', ...
                this.settingTsNet.pretrainedNetName, ...
                this.settingTsNet.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    methods( Static )
        function [ sid2iid, sid2regn, sid2dp ] = ...
                makeFgdObjRegnSequence( oid2iid, oid2dpid2regns, dpid2dids )
            numObj = numel( oid2iid );
            numDirPair = size( oid2dpid2regns, 2 );
            sid2iid = zeros( numObj * numDirPair, 1, 'single' );
            sid2regn = zeros( 5, numObj * numDirPair, 'single' );
            sid2dp = zeros( 2, numObj * numDirPair, 'single' );
            sid = 0;
            for oid = 1 : numObj,
                iid = oid2iid( oid );
                for dpid = 1 : numDirPair,
                    regns = oid2dpid2regns{ oid, dpid };
                    numRegns = size( regns, 2 );
                    if ~numRegns, continue; end;
                    sid = sid + 1;
                    sel = ceil( rand * numRegns );
                    sid2iid( sid ) = iid;
                    sid2regn( :, sid ) = regns( :, sel );
                    sid2dp( :, sid ) = dpid2dids( :, dpid );
                end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
            sid2dp = sid2dp( :, perm );
        end
        function [ sid2iid, sid2regn ] = ...
                makeBgdObjRegnSequence( oid2iid, oid2regns, numSmpl )
            numObj = numel( oid2iid );
            if ~numObj, sid2iid = zeros( 0, 1 ); sid2regn = zeros( 5, 0 ); return; end;
            sid2iid = zeros( numSmpl, 1, 'single' );
            sid2regn = zeros( 5, numSmpl, 'single' );
            sid = 0;
            while true,
                oid = ceil( numObj * rand );
                iid = oid2iid( oid );
                regns = oid2regns{ oid };
                numRegns = size( regns, 2 );
                if ~numRegns, continue; end;
                sid = sid + 1;
                sel = ceil( rand * numRegns );
                sid2iid( sid ) = iid;
                sid2regn( :, sid ) = [ regns( :, sel ); ceil( rand * 2 ); ];
                if sid == numSmpl, break; end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
        end
        function [ sid2iid, sid2regn ] = ...
                makeBgdRegnSequence( iid2sid2regns, numSmpl )
            numIm = size( iid2sid2regns, 1 );
            numScale = size( iid2sid2regns, 2 );
            sid2iid = zeros( numSmpl, 1, 'single' );
            sid2regn = zeros( 5, numSmpl, 'single' );
            sid = 0;
            while true,
                iid = ceil( numIm * rand );
                scale = ceil( numScale * rand );
                regns = iid2sid2regns{ iid, scale };
                numRegns = size( regns, 2 );
                if ~numRegns, continue; end;
                sid = sid + 1;
                sel = ceil( rand * numRegns );
                sid2iid( sid ) = iid;
                sid2regn( :, sid ) = [ regns( :, sel ); ceil( rand * 2 ); ];
                if sid == numSmpl, break; end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
        end
        % Majorly used in net. Forward function in output layer.
        function res2 = forward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numOutLyr = size( gt, 3 );
            numDimPerLyr = size( X, 3 ) / numOutLyr;
            y = gpuArray( zeros( numOutLyr, 1, 'single' ) );
            for lid = 1 : numOutLyr,
                sidx = ( lid - 1 ) * numDimPerLyr + 1;
                eidx = lid * numDimPerLyr;
                y( lid ) = vl_nnsoftmaxloss...
                    ( X( :, :, sidx : eidx, : ), gt( :, :, lid, : ) );
            end
            res2.x = mean( y );
        end
        % Majorly used in net. Backward function in output layer.
        function res1 = backward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numOutLyr = size( gt, 3 );
            numDimPerLyr = size( X, 3 ) / numOutLyr;
            dzdy = res2.dzdx / numOutLyr;
            Y = gpuArray( zeros( size( X ), 'single' ) );
            for lid = 1 : numOutLyr,
                sidx = ( lid - 1 ) * numDimPerLyr + 1;
                eidx = lid * numDimPerLyr;
                Y( :, :, sidx : eidx, : ) = ...
                    vl_nnsoftmaxloss...
                    ( X( :, :, sidx : eidx, : ), gt( :, :, lid, : ), dzdy );
            end
            res1.dzdx = Y;
        end
        % Majorly used in net. Update energy and task-specific evaluation metric.
        % For this target application, object detection, 
        % the metric is top-1 accuracy.
        function tsMetric = computeTsMetric( res, gts )
            output = gather( res( end - 1 ).x );
            gts = gather( gts );
            numOutLyr = size( gts, 3 );
            numDimPerLyr = size( output, 3 ) / numOutLyr;
            err1 = zeros( numOutLyr, 1, 'single' );
            for lid = 1 : numOutLyr,
                sidx = ( lid - 1 ) * numDimPerLyr + 1;
                eidx = lid * numDimPerLyr;
                predictions = output( :, :, sidx : eidx, : );
                [ ~, predictions ] = sort...
                    ( predictions, 3, 'descend' );
                err1_ = ~bsxfun( @eq, predictions, ...
                    gts( :, :, lid, : ) );
                % Take top predictions.
                % If top-5, err1_ = err1_( :, :, 1 : 5, : );
                err1_ = err1_( :, :, 1, : );
                err1( lid ) = sum( err1_( : ) );
            end
            tsMetric = mean( err1 );
        end
    end
end