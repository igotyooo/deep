% IN:  DB.
% OUT: Images-GT paies per epoch.
% Task-specific implementation required.
classdef InOutPropRegn < handle
    properties
        db;                     % A general db.
        tsDb;                   % A task specific db to be made. If it is unnecessary, just fetch db.
        rgbMean;                % A task specific RGB mean to normalize images.
        numBchTr;               % Number of training batches in an epoch.
        numBchVal;              % Number of validation batches in an epoch.
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
        function this = InOutPropRegn...
                ( db, settingTsDb, settingTsNet, settingGeneral )
            this.db = db;
            this.tsMetricName = 'Top-1 err';
            % Task specific) Default parameters for task specific db.
            this.settingTsDb.stride                             = 32;
            this.settingTsDb.patchSide                          = 227;
            this.settingTsDb.numScale                           = 16;
            this.settingTsDb.numAspect                          = 16;
            this.settingTsDb.confidence                         = 0.97;
            this.settingTsDb.violate                            = 1 / 4;
            % Parameters for positive mining.
            this.settingTsDb.posMinMargin                       = 0.1;
            this.settingTsDb.posIntOverRegnMoreThan             = 1 / 3;    % A target object should be large enough.
            this.settingTsDb.posIntOverTarObjMoreThan           = 0.99;     % A target object should be fully-included.
            this.settingTsDb.posIntOverSubObjMoreThan           = 0.6;      % A sub-object should be majorly-included.
            this.settingTsDb.posIntMajorityMoreThan             = 2;        % A target object should large enough w.r.t. the sub-objects.
            % Parameters for semi-negative mining.
            this.settingTsDb.snegIntOverRegnMoreThan            = 1 / 36;	% A sub-object should not be too small.
            this.settingTsDb.snegIntOverObjMoreThan             = 0.3;      % The region is truncating the target object.
            this.settingTsDb.snegIntOverObjLessThan             = 0.7;      % The region is truncating the target object.
            % Parameters for negative mining.
            this.settingTsDb.negIntOverObjLessThan              = 0.1;      % Very small overlap is allowed for background region.
            % Task specific) Default parameters for task specific net.
            global path;
            pretrainedNet                                       = path.net.vgg_m;
            this.settingTsNet.pretrainedNetName                 = pretrainedNet.name;
            this.settingTsNet.suppressPretrainedLayerLearnRate  = 1 / 10;
            % General) Default parameters to provide batches.
            this.settingGeneral.dstSide                         = this.settingTsDb.patchSide;
            this.settingGeneral.dstCh                           = 3;
            this.settingGeneral.batchSize                       = 256;
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
            % Compute a rgb mean vector.
            fpath = this.getRgbMeanPath;
            try
                data = load( fpath );
                this.rgbMean = data.data.rgbMean;
            catch
                data.rgbMean = this.computeRgbMean;
                this.makeRgbMeanDir;
                save( fpath, 'data' );
                this.rgbMean = data.rgbMean;
            end;
            % Reform the general db to a task-specific format.
            fpath = this.getTsDbPath;
            try
                fprintf( '%s: Try to load ts-db.\n', ...
                    upper( mfilename ) );
                data = load( fpath );
                this.tsDb = data.data.tsDb;
                fprintf( '%s: Ts-db loaded.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen ts-db.\n', ...
                    upper( mfilename ) );
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
            % General) Make training pool to be consumed in an epoch.
            batchSize = this.settingGeneral.batchSize;
            [   this.poolTr.sid2iid, ...
                this.poolTr.sid2tlbr, ...
                this.poolTr.sid2gt ] = ...
                this.getRegnSeqInEpch( 1 );
            this.numBchTr = ...
                numel( this.poolTr.sid2iid ) / batchSize;
            % General) Make validation pool to be consumed in an epoch.
            [   this.poolVal.sid2iid, ...
                this.poolVal.sid2tlbr, ...
                this.poolVal.sid2gt ] = ...
                this.getRegnSeqInEpch( 2 );
            this.numBchVal = ...
                numel( this.poolVal.sid2iid ) / batchSize;
        end
        % Majorly used in net. Provide a tr/val batch of I/O pairs.
        function [ ims, gts ] = provdBchTr( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolTr.sid2iid )
                % Make training pool to be consumed in an epoch.
                [   this.poolTr.sid2iid, ...
                    this.poolTr.sid2tlbr, ...
                    this.poolTr.sid2gt ] = ...
                    this.getRegnSeqInEpch( 1 );
                this.numBchTr = ...
                    numel( this.poolTr.sid2iid ) / batchSize;
            end
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolTr.sid2iid( batchSmpl );
            sid2tlbr = this.poolTr.sid2tlbr( :, batchSmpl );
            sid2gt = this.poolTr.sid2gt( batchSmpl );
            this.poolTr.sid2iid( 1 : batchSize ) = [  ];
            this.poolTr.sid2tlbr( :, 1 : batchSize ) = [  ];
            this.poolTr.sid2gt( 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.tr.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2tlbr, sid2gt );
        end
        function [ ims, gts ] = provdBchVal( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolVal.sid2iid )
                % Make validation pool to be consumed in an epoch.
                [   this.poolVal.sid2iid, ...
                    this.poolVal.sid2tlbr, ...
                    this.poolVal.sid2gt ] = ...
                    this.getRegnSeqInEpch( 2 );
                this.numBchVal = ...
                    numel( this.poolVal.sid2iid ) / batchSize;
            end
            batchSmpl = ( labindex : numlabs : batchSize )';
            sid2iid = this.poolVal.sid2iid( batchSmpl );
            sid2tlbr = this.poolVal.sid2tlbr( :, batchSmpl );
            sid2gt = this.poolVal.sid2gt( batchSmpl );
            this.poolVal.sid2iid( 1 : batchSize ) = [  ];
            this.poolVal.sid2tlbr( :, 1 : batchSize ) = [  ];
            this.poolVal.sid2gt( 1 : batchSize ) = [  ];
            iid2impath = this.tsDb.val.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2tlbr, sid2gt );
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
            dstSide = this.settingGeneral.dstSide;
            dstCh = this.settingGeneral.dstCh;
            numOutDim = max( this.tsDb.tr.oid2cid ) + 1;
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
            % filterChannel is not general !!!!!!!!!!!!!!!!!!!!!!
            filterChannel = size( layers{ end - 3 }.weights{ 1 }, 4 ); 
            layers{ end - 1 } = struct(...
                'type', 'conv', ...
                'name', 'target-fc', ...
                'weights', { { 0.01 * randn( 1, 1, filterChannel, numOutDim, 'single' ), zeros( 1, numOutDim, 'single' ) } }, ...
                'stride', 1, ...
                'pad', 0, ...
                'learningRate', [ initFilterLearningRate, initBiasLearningRate ], ...
                'weightDecay', [ filterWeightDecay, biasWeightDecay ] );
            layers{ end - 1 }.momentum{ 1 } = ...
                zeros( size( layers{ end - 1 }.weights{ 1 } ), 'single' );
            layers{ end - 1 }.momentum{ 2 } = ...
                zeros( size( layers{ end - 1 }.weights{ 2 } ), 'single' );
            % Initialize the loss layer.
            layers{ end }.type = 'softmaxloss';
            % Form the net in VGG style.
            net.layers = layers;
            net.normalization.averageImage = [  ];
            net.normalization.keepAspect = false;
            net.normalization.border = [ 0, 0 ];
            net.normalization.imageSize = [ dstSide, dstSide, dstCh ];
            net.normalization.interpolation = 'bicubic';
            net.classes.name = this.db.cid2name;
            net.classes.name{ end + 1 }   = 'background';
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
            name = sprintf( 'IOPROPREGN_%s_OF_%s', ...
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
            dstCh = this.settingGeneral.dstCh;
            idx2iid = this.db.getTriids;
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
                ( this, iid2impath, sid2iid, sid2tlbr, sid2gt )
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
                wind = sid2tlbr( :, sid );
                im = idx2im{ sid2idx( sid ) };
                if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                im = cropAndNormalizeIm( im, size( im ), wind, this.rgbMean );
                if ceil( rand - 0.5 ), im = fliplr( im ); end;
                sid2im( :, :, :, sid ) = imresize...
                    ( im, [ dstSide, dstSide ], 'bicubic' );
            end;
            ims = sid2im;
            % Merge ground-truths into a same shape of net output.
            gts = reshape( sid2gt, [ 1, 1, 1, numSmpl ] );
        end
        function [ sid2iid, sid2tlbr, sid2gt ] = ...
                getRegnSeqInEpch( this, setid )
            rng( 'shuffle' );
            batchSize = this.settingGeneral.batchSize;
            if setid == 1, subTsDb = this.tsDb.tr; else subTsDb = this.tsDb.val; end;
            numObj = numel( subTsDb.oid2iid );
            numSample = ceil( numObj * 3 / batchSize ) * batchSize;
            bgdClsId = max( subTsDb.oid2cid ) + 1;
            sid2iid = zeros( numSample, 1, 'single' );
            sid2tlbr = zeros( 4, numSample, 'single' );
            sid2gt = zeros( numSample, 1, 'single' );
            sid = 0; oid = 0; iter = 0;
            while true,
                iter = iter + 1;
                if iter > numObj, oid = ceil( numObj * rand ); else oid = oid + 1; end;
                iid = subTsDb.oid2iid( oid );
                % Sample a positive region.
                if sid == numSample, break; end;
                regns = subTsDb.oid2posregns{ oid };
                numRegn = size( regns, 2 );
                if ~numRegn, continue; end;
                ridx = ceil( rand * numRegn );
                sid = sid + 1;
                sid2iid( sid ) = iid;
                sid2tlbr( :, sid ) = regns( 1 : 4, ridx );
                sid2gt( sid ) = subTsDb.oid2cid( oid );
                % Sample a semi-negative region.
                if sid == numSample, break; end;
                regns = subTsDb.oid2snegregns{ oid };
                numRegn = size( regns, 2 );
                if ~numRegn, continue; end;
                ridx = ceil( rand * numRegn );
                sid = sid + 1;
                sid2iid( sid ) = iid;
                sid2tlbr( :, sid ) = regns( 1 : 4, ridx );
                sid2gt( sid ) = bgdClsId;
                % Sample a negative region.
                if sid == numSample, break; end;
                regns = subTsDb.iid2sid2negregns{ iid };
                s2ok = ~cellfun( @isempty, regns );
                if ~sum( s2ok ), continue; end;
                s = find( s2ok );
                s = s( ceil( numel( s ) * rand ) );
                regns = regns{ s };
                numRegn = size( regns, 2 );
                ridx = ceil( rand * numRegn );
                sid = sid + 1;
                sid2iid( sid ) = iid;
                sid2tlbr( :, sid ) = regns( 1 : 4, ridx );
                sid2gt( sid ) = bgdClsId;
            end;
        end
        function subTsDb = makeSubTsDb( this, setid )
            % Set parameters.
            stride = this.settingTsDb.stride;
            patchSide = this.settingTsDb.patchSide;
            numScale = this.settingTsDb.numScale;
            numAspect = this.settingTsDb.numAspect;
            confidence = this.settingTsDb.confidence;
            violate = this.settingTsDb.violate;
            % Parameters for positive mining.
            posMinMargin = this.settingTsDb.posMinMargin;
            posIntOverRegnMoreThan = this.settingTsDb.posIntOverRegnMoreThan;       % A target object should be large enough.
            posIntOverTarObjMoreThan = this.settingTsDb.posIntOverTarObjMoreThan;   % A target object should be fully-included.
            posIntOverSubObjMoreThan = this.settingTsDb.posIntOverSubObjMoreThan;   % A sub-object should be majorly-included.
            posIntMajorityMoreThan = this.settingTsDb.posIntMajorityMoreThan;       % A target object should large enough w.r.t. the sub-objects.
            % Parameters for semi-negative mining.
            snegIntOverRegnMoreThan = this.settingTsDb.snegIntOverRegnMoreThan;     % A sub-object should not be too small.
            snegIntOverObjMoreThan = this.settingTsDb.snegIntOverObjMoreThan;       % The region is truncating the target object.
            snegIntOverObjLessThan = this.settingTsDb.snegIntOverObjLessThan;       % The region is truncating the target object.
            % Parameters for negative mining.
            negIntOverObjLessThan = this.settingTsDb.negIntOverObjLessThan;         % Very small overlap is allowed for background region.
            % Determine scales and aspects.
            trboxes = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
            referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
            sid2s = determineScales...
                ( trboxes, referenceSide, numScale, confidence );
            aid2a = determineAspectRates...
                ( trboxes, numAspect, confidence );
            % Do the job.
            newiid2iid = find( this.db.iid2setid == setid );
            newiid2iid = newiid2iid( randperm( numel( newiid2iid ) )' );
            newoid2oid = cat( 1, this.db.iid2oids{ newiid2iid } );
            newoid2newiid = zeros( size( newoid2oid ), 'single' );
            newoid2posregns = cell( size( newoid2oid ) );
            newoid2cid = zeros( size( newoid2oid ), 'single' );
            newoid2snegregns = cell( size( newoid2oid ) );
            newiid2sid2negregns = cell( size( newiid2iid ) );
            newiid2impath = this.db.iid2impath( newiid2iid );
            numIm = numel( newiid2iid );
            newoid = 1; cummt = 0; 
            for newiid = 1 : numIm;
                itime = tic;
                iid = newiid2iid( newiid );
                imSize = this.db.iid2size( :, iid );
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2oids{ iid } );
                oid2cid = this.db.oid2cid( this.db.iid2oids{ iid } );
                numObj = size( oid2tlbr, 2 );
                % Extract candidate regions and compute bisic informations.
                rid2tlbr = extMultiScaleDenseRegions2...
                    ( imSize, stride, patchSide, sid2s, aid2a, violate );
                rid2rect = tlbr2rect( rid2tlbr );
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
                prid2tlbr = rid2tlbr( :, rid2ok );
                oid2mgtlbr = scaleBoxes( oid2tlbr, 1 + 2 * posMinMargin, 1 + 2 * posMinMargin );
                oid2mgtlbr = round( oid2mgtlbr );
                oid2pregns = cell( numObj, 1 );
                for oid = 1 : numObj,
                    prids = prid2oid == oid;
                    if sum( prids ),
                        ptlbrs = prid2tlbr( :, prids );
                        ptlbrs( 1, ptlbrs( 1, : ) > oid2mgtlbr( 1, oid ) ) = oid2mgtlbr( 1, oid );
                        ptlbrs( 2, ptlbrs( 2, : ) > oid2mgtlbr( 2, oid ) ) = oid2mgtlbr( 2, oid );
                        ptlbrs( 3, ptlbrs( 3, : ) < oid2mgtlbr( 3, oid ) ) = oid2mgtlbr( 3, oid );
                        ptlbrs( 4, ptlbrs( 4, : ) < oid2mgtlbr( 4, oid ) ) = oid2mgtlbr( 4, oid );
                        oid2pregns{ oid } = ptlbrs;
                    else
                        oid2pregns{ oid } = [ oid2tlbr( :, oid ); 0; 0; ];
                    end;
                end;
                % Semi-negative mining.
                % The region is truncating the target object.
                rid2oid2issub = rid2oid2ior >= snegIntOverRegnMoreThan;
                rid2issub = any( rid2oid2issub, 2 );
                rid2oid2ok = snegIntOverObjMoreThan <= rid2oid2ioo & ...
                    snegIntOverObjLessThan >= rid2oid2ioo;
                rid2ok = rid2issub & all( eq( rid2oid2issub, rid2oid2ok ), 2 );
                snrid2tlbr = rid2tlbr( :, rid2ok );
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
                        snregns = cat( 1, snregns, zeros( 2, 4, 'single' ) );
                        oid2snregns{ oid } = snregns;
                    end;
                end;
                % Negative mining.
                % Very small overlap is allowed for background region.
                rid2ok = all( rid2oid2ioo <= negIntOverObjLessThan, 2 );
                nrid2tlbr = rid2tlbr( :, rid2ok );
                sid2nregns = cell( numScale, 1 );
                for sid = 1 : numScale,
                    nrids = find( nrid2tlbr( 5, : ) == sid );
                    nrids = randsample( nrids, min( numel( nrids ), 500 ) );
                    sid2nregns{ sid } = nrid2tlbr( :, nrids );
                end;
                newoids = ( newoid : newoid + numObj - 1 )';
                newoid2newiid( newoids ) = newiid;
                newoid2cid( newoids ) = oid2cid;
                newoid2posregns( newoids ) = oid2pregns;
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
            subTsDb.oid2posregns = newoid2posregns;
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
        
        function name = getTsDbName( this )
            name = sprintf( ...
                'DBTS_%s_OF_%s', ...
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
                'ANET_%s_%s', ...
                this.settingTsNet.pretrainedNetName, ...
                this.settingTsNet.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    methods( Static )
        % % Majorly used in net, if a layer is custom. Forward function in output layer.
        % function res2 = forward( ly, res1, res2 )
        % end
        % % Majorly used in net, if a layer is custom. Backward function in output layer.
        % function res1 = backward( ly, res1, res2 )
        % end
        
        % Majorly used in net. Update energy and task-specific evaluation metric.
        % For this target application, object detection, 
        % the metric is top-1 accuracy.
        function tsMetric = computeTsMetric( res, gts )
            prediction = gather( res( end - 1 ).x );
            gts = gather( gts );
            [ ~, prediction ] = sort...
                ( prediction, 3, 'descend' );
            err1_ = ~bsxfun( @eq, prediction, gts );
            % Take top predictions.
            % If top-5, err1_ = err1_( :, :, 1 : 5, : );
            err1_ = err1_( :, :, 1, : );
            tsMetric = sum( err1_( : ) );
        end
    end
end