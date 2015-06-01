classdef Cnn < handle
    properties
        srcDb;
        srcImServer;
        initCnnName;
        isinit;
        avgIm;
        layers;
        trInfo;
        valInfo;
        useGpu;
        setting;
    end
    methods
        function this = Cnn( srcDb, srcImServer, setting )
            this.srcDb                  = srcDb;
            this.srcImServer            = srcImServer;
            this.useGpu                 = true;
            this.initCnnName            = 'RND';
            this.setting.archFunh       = @CFG_VL;
            this.setting.batchSize      = 256;
            this.setting.numEpch        = 65;
            this.setting.weightDecay    = 0.0005;
            this.setting.momentum       = 0.9;
            this.setting.learnRate      = [ 0.01 * ones( 1, 25 ), ...
                0.001 * ones( 1, 25 ), ...
                0.0001 * ones( 1, 15 ) ];
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function setUseGpu( this, useGpu )
            this.useGpu = useGpu;
            if useGpu
                fprintf( '%s: GPU mode.\n', ...
                    upper( mfilename ) );
            else
                fprintf( '%s: CPU mode.\n', ...
                    upper( mfilename ) );
            end
        end
        function computeAvgIms( this )
            fpath = this.getAvgImPath;
            try
                data = load( fpath );
                avgIm = data.avgIm;
                fprintf( '%s: Avg im is loaded.\n', ...
                    upper( mfilename ) );
            catch
                impaths = this.srcDb.getTrIfpaths;
                batchSize = this.setting.batchSize;
                avgIm = this.srcImServer.computeAvgIm...
                    ( impaths, batchSize );
                this.makeAvgImDir;
                save( this.getAvgImPath, 'avgIm' );
            end
            this.fetchAvgIm( avgIm );
        end
        function fetchAvgIm( this, avgIm_ )
            this.avgIm = avgIm_;
        end
        function initCnn( this )
            useGpu_ = this.useGpu;
            this.setting.arch = this.setting.archFunh(  );
            this.layers = cell( size( this.setting.arch ) );
            for l = 1 : length( this.setting.arch )
                this.layers{ l } = ...
                    this.initLayer( this.setting.arch{ l } );
            end
            if useGpu_, this.fetchLyrsOnGpu; end;
            this.trInfo.epch2obj    = [  ];
            this.trInfo.epch2err1   = [  ];
            this.trInfo.epch2err5   = [  ];
            this.valInfo.epch2obj   = [  ];
            this.valInfo.epch2err1  = [  ];
            this.valInfo.epch2err5  = [  ];
            this.isinit = true;
        end
        function fetchCnn( this, layers, cnnName )
            useGpu_ = this.useGpu;
            this.layers = layers;
            this.initCnnName = cnnName;
            if useGpu_, this.fetchLyrsOnGpu; end;
            this.trInfo.epch2obj    = [  ];
            this.trInfo.epch2err1   = [  ];
            this.trInfo.epch2err5   = [  ];
            this.valInfo.epch2obj   = [  ];
            this.valInfo.epch2err1  = [  ];
            this.valInfo.epch2err5  = [  ];
            this.isinit = true;
        end
        function trainCnn( this, addrss )
            this.isinit = false;
            numEpch     = this.setting.numEpch;
            trdb        = this.srcDb.getTrDb;
            valdb       = this.srcDb.getTeDb;
            idx2iid     = trdb.idx2iid;
            % Start training.
            rng( 0 ); ecummt = 0; ecnt = 0;
            for epch = 1 : numEpch; etime = tic;
                % Model I/O.
                if this.isTrainedAt( epch ),
                    if epch == numEpch, 
                        fprintf( '%s: Already trained. Load the model.\n', ...
                            upper( mfilename ) );
                        this.loadCnnAt( epch );
                        fprintf( '%s: Done.\n', ...
                            upper( mfilename ) );
                    end; continue; 
                end;
                if epch > 1 && this.isTrainedAt( epch - 1 ),
                    fprintf( '%s: Load the model of epch %d.\n', ...
                        upper( mfilename ), epch - 1 );
                    this.loadCnnAt( epch - 1 );
                end
                % Train net.
                ecnt = ecnt + 1;
                trdb.idx2iid = idx2iid( randperm( numel( idx2iid ) ) );
                trInfo_ = this.trainCnnAt...
                    ( epch, ecnt, ecummt, trdb );
                % Evaluate net.
                valInfo_ = this.valCnn( valdb );
                % Update energy.
                this.trInfo.epch2obj( end + 1 ) = trInfo_.obj;
                this.trInfo.epch2err1( end + 1 ) = trInfo_.err1;
                this.trInfo.epch2err5( end + 1 ) = trInfo_.err5;
                this.valInfo.epch2obj( end + 1 ) = valInfo_.obj;
                this.valInfo.epch2err1( end + 1 ) = valInfo_.err1;
                this.valInfo.epch2err5( end + 1 ) = valInfo_.err5;
                % Save.
                this.saveCnnAt( epch );
                this.saveFigAt( epch );
                this.saveFiltImAt( epch );
                % Report.
                attch{ 1 } = this.getFigPath( epch );
                for lyid = this.getVisFiltLyers( this.layers ),
                    attch{ end + 1 } = this.getFiltImPath( lyid, epch ); end;
                this.reportTrainStatus...
                    ( this.trInfo, this.valInfo, attch, addrss ); 
                clear attch;
                ecummt = ecummt + toc( etime );
            end % Go to next epoch.
        end
        function trInfo = trainCnnAt...
                ( this, epch, ecnt, ecummt,...
                trdb )
            % Set params.
            useGpu_         = this.useGpu;
            errtype         = trdb.getGtType;
            learnRate       = this.setting.learnRate( epch );
            numEpch         = this.setting.numEpch;
            prevLearnRate   = this.setting.learnRate( max( 1, epch - 1 ) );
            batchSize       = this.setting.batchSize;
            weightDecay     = this.setting.weightDecay;
            momentum        = this.setting.momentum;
            numTrBatch      = numel( 1 : batchSize : numel( trdb.idx2iid ) );
            trInfo.obj      = 0;
            trInfo.err1     = 0;
            trInfo.err5     = 0;
            net.layers      = this.layers;
            if useGpu_, one = gpuArray( single( 1 ) ); else one = single( 1 ); end;
            % Reset momentum if the learning rate is changed.
            if learnRate ~= prevLearnRate
                fprintf('%s: Learn rate changed to %f. Momentum is reset.\n', ...
                    upper( mfilename ), learnRate );
                for l = 1 : numel( net.layers )
                    if ~strcmp( net.layers{ l }.type, 'conv' ), continue; end
                    net.layers{ l }.filtersMomentum = ...
                        0 * net.layers{ l }.filtersMomentum;
                    net.layers{ l }.biasesMomentum = ...
                        0 * net.layers{ l }.biasesMomentum;
                end
            end
            % For each batch,
            bcummt = 0; bcnt = 0; res = [  ];
            for t = 1 : batchSize : numel( trdb.idx2iid ); btime = tic;
                % Define a batch for training, and its corresponding labels.
                biids = trdb.idx2iid...
                    ( t : min( t + batchSize - 1, ...
                    numel( trdb.idx2iid ) ) );
                bimpaths = trdb.iid2ifpath( biids );
                bgts = trdb.iid2gt( biids );
                if strcmp( errtype( 2 : end ), 'CLS' )
                    bgts = cell2mat( bgts' );
                elseif strcmp( errtype, 'REG' )
                    bgts = cat( 4, bgts{ : } );
                end
                % Load and augment the batch images.
                bims = this.srcImServer.impath2cnninput...
                    ( bimpaths, this.avgIm );
                % Let CPU prepare for next batch during the following GPU computing.
                nextBiids = trdb.idx2iid...
                    ( t + batchSize : min( t + 2 * batchSize - 1, ...
                    numel( trdb.idx2iid ) ) );
                if ~isempty( nextBiids )
                    nextBimpaths = trdb.iid2ifpath( nextBiids );
                    this.srcImServer.impath2cnninput...
                        ( nextBimpaths, this.avgIm );
                end
                % Put the batch images to the GPU memory.
                if useGpu_, bims = gpuArray( bims ); end;
                % Attatch the corresponding labels at the end
                % of the CNN to compute the energy.
                net.layers{ end }.class = bgts;
                % Do forward/backward.
                res = my_simplenn...
                    ( net, bims, one, res, ...
                    'conserveMemory', true, ...
                    'sync', true );
                % Record the computed energy and error.
                trInfo = this.updateEnergy( trInfo, res, bgts, errtype );
                % Update w by the gredient step.
                net = this.updateW...
                    ( net, res, momentum, weightDecay, learnRate, numel( biids ) );
                % Print out the status.
                btime = toc( btime ); bcummt = bcummt + btime; bcnt = bcnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numTrBatch, bcnt, ...
                    sprintf( 'done for epoch %d with %dims/sec. (Batch %d/%d)', ...
                    epch, round( batchSize / btime ), bcnt, numTrBatch ), bcummt );
                if ecnt > 1,
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numEpch - epch,  ecnt, ...
                        sprintf( 'done to finish train. (Epch=%d/%d)', ...
                        epch, numEpch ), ecummt );
                end
            end % Go to next training batch.
            % Update the info.
            trInfo.obj = trInfo.obj / numel( trdb.idx2iid );
            trInfo.err1 = trInfo.err1 / numel( trdb.idx2iid );
            trInfo.err5 = trInfo.err5 / numel( trdb.idx2iid );
            this.layers = net.layers;
        end
        function valInfo = valCnn...
                ( this, valdb )
            useGpu_     = this.useGpu;
            net.layers  = this.layers;
            errtype     = valdb.getGtType;
            batchSize   = this.setting.batchSize;
            numValBatch = numel( 1 : batchSize : numel( valdb.idx2iid ) );
            % Initialize energy.
            valInfo.obj     = 0;
            valInfo.err1    = 0;
            valInfo.err5    = 0;
            % In this epoch, validate the model.
            bcummt = 0; bcnt = 0; res = [  ];
            fprintf( '%s: Eval net.\n', upper( mfilename ) );
            for t = 1 : batchSize : numel( valdb.idx2iid ); btime = tic;
                % Define a batch for training, and its corresponding labels.
                biids = valdb.idx2iid( t : min( t + batchSize - 1, numel( valdb.idx2iid ) ) );
                bimpaths = valdb.iid2ifpath( biids );
                bgts = valdb.iid2gt( biids );
                if strcmp( errtype( 2 : end ), 'CLS' )
                    bgts = cell2mat( bgts' );
                elseif strcmp( errtype, 'REG' )
                    bgts = cat( 4, bgts{ : } );
                end
                % Load and augment the batch images.
                bims = this.srcImServer.impath2cnninput( bimpaths, this.avgIm );
                % Let CPU prepare for next batch during the following GPU computing.
                nextBiids = valdb.idx2iid...
                    ( t + batchSize : min( t + 2 * batchSize - 1, numel( valdb.idx2iid ) ) );
                if ~isempty( nextBiids )
                    nextBimpaths = valdb.iid2ifpath( nextBiids );
                    this.srcImServer.impath2cnninput( nextBimpaths, this.avgIm );
                end
                % Put the batch images to the GPU memory.
                if useGpu_, bims = gpuArray( bims ); end;
                % Attatch the corresponding labels at the end
                % of the CNN to compute the energy.
                net.layers{ end }.class = bgts;
                % Compute the energy.
                res = my_simplenn( net, bims, [  ], res,...
                    'disableDropout', true, ...
                    'conserveMemory', true, ...
                    'sync', true );
                % update energy
                valInfo = this.updateEnergy( valInfo, res, bgts, errtype );
                % Compute the remaining time and print out the progress.
                btime = toc( btime ); bcummt = bcummt + btime; bcnt = bcnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numValBatch, bcnt,  'done for val.', bcummt );
            end % Go to next validation batch.
            % Update the info.
            valInfo.obj = valInfo.obj / numel( valdb.idx2iid );
            valInfo.err1 = valInfo.err1 / numel( valdb.idx2iid );
            valInfo.err5 = valInfo.err5 / numel( valdb.idx2iid );
        end
        function res = feedforward...
                ( this, bims, imServer )
            if ~iscell( bims ), bims = { bims }; end;
            useGpu_ = this.useGpu;
            bims = imServer.im2cnninput( bims, this.avgIm );
            if useGpu_, bims = gpuArray( bims ); end;
            typeBackup = this.layers{ end }.type;
            this.layers{ end }.type = 'softmax';
            res = my_simplenn...
                ( this, bims, [  ], 'disableDropout', true );
            this.layers{ end }.type = typeBackup;
        end
        function res = feedforwardUpto...
                ( this, bims, lid, imServer )
            if ~iscell( bims ), bims = { bims }; end;
            useGpu_ = this.useGpu;
            bims = imServer.im2cnninput( bims, this.avgIm );
            if useGpu_, bims = gpuArray( bims ); end;
            typeBackup = this.layers{ end }.type;
            this.layers{ end }.type = 'softmax';
            res = my_simplenn...
                ( this, bims, [  ], [  ], ...
                'disableDropout', true, ...
                'extOnly', true, ...
                'targetLayerId', lid );
            this.layers{ end }.type = typeBackup;
        end
        function fetchLyrsOnGpu( this )
            this = vl_simplenn_move( this, 'gpu' );
            for lid = 1 : numel( this.layers )
                if ~strcmp( this.layers{ lid }.type, 'conv' ),
                    continue;
                end
                this.layers{ lid }.filtersMomentum = ...
                    gpuArray( this.layers{ lid }.filtersMomentum );
                this.layers{ lid }.biasesMomentum = ...
                    gpuArray( this.layers{ lid }.biasesMomentum );
            end
        end
        % Functions for visualization.
        function h = showTrainInfoFig( this )
            errtype = this.srcDb.getGtType;
            numEpch = numel( this.trInfo.epch2obj );
            h = figure( 1 ); clf;
            subplot( 1, 2, 1 );
            semilogy( 1 : numEpch, ...
                this.trInfo.epch2obj, 'k.-' );
            set( gca, 'yscale', 'linear' );
            hold on;
            semilogy( 1 : numEpch, ...
                this.valInfo.epch2obj, 'b.-' );
            xlabel( 'Epoch' ); ylabel( 'Loss' );
            legend( 'Train', 'Val' );
            grid on;
            switch errtype
                case 'MCLS'
                    subplot( 1, 2, 2 );
                    plot( 1 : numEpch, ...
                        this.trInfo.epch2err1, 'k.-' );
                    hold on;
                    plot( 1 : numEpch, ...
                        this.trInfo.epch2err5, 'k.--' );
                    plot( 1 : numEpch, ...
                        this.valInfo.epch2err1, 'b.-' );
                    plot( 1 : numEpch, ...
                        this.valInfo.epch2err5, 'b.--' );
                    xlabel( 'Epoch' ); ylabel( 'Error' );
                    legend( ...
                        { 'Top-1 on train', ...
                        'Top-5 on train', ...
                        'Top-1 on val', ...
                        'Top-5 on val' }, ...
                        'Location', 'Best' );
                    grid on;
                case 'BCLS'
                    subplot( 1, 2, 2 );
                    plot( 1 : numEpch, ...
                        this.trInfo.epch2err1, 'k.' );
                    hold on;
                    plot( 1 : numEpch, ...
                        this.valInfo.epch2err1, 'b.' );
                    xlabel( 'Epoch' ); ylabel( 'Error' );
                    legend( { 'Top-1 on train', ...
                        'Top-1 on val' }, ...
                        'Location', 'Best' );
                    grid on;
                case 'REG'
                    % Nothing to do.
            end
            set( gcf, 'color', 'w' );
        end
        % Functions for report training.
        function [ title, mssg ] = writeTrainReport...
                ( this, trInfo, valInfo )
            epch = length( trInfo.epch2obj );
            title = sprintf( '%s: TRAINING REPORT AT EPOCH %d', ...
                upper( mfilename ), epch );
            mssg = {  };
            mssg{ end + 1 } = '_______________';
            mssg{ end + 1 } = 'TRAINING REPORT';
            mssg{ end + 1 } = sprintf( 'DATABASE: %s', ...
                this.srcDb.dbName );
            mssg{ end + 1 } = sprintf( 'TRAINING IMSERVER: %s', ...
                this.srcImServer.getName );
            mssg{ end + 1 } = sprintf( 'CNN: %s', ...
                this.getCnnName );
            switch this.srcDb.getGtType
                case 'MCLS'
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP1 ERR ON TR SET: %.2f%%', ...
                        trInfo.epch2err1( end ) * 100 );
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP5 ERR ON TR SET: %.2f%%', ...
                        trInfo.epch2err5( end ) * 100 );
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP1 ERR ON VAL SET: %.2f%%', ...
                        valInfo.epch2err1( end ) * 100 );
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP5 ERR ON VAL SET: %.2f%%', ...
                        valInfo.epch2err5( end ) * 100 );
                case 'BCLS'
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP1 ERR ON TR SET: %.2f%%', ...
                        trInfo.epch2err1( end ) * 100 );
                    mssg{ end + 1 } = ...
                        sprintf( 'TOP1 ERR ON VAL SET: %.2f%%', ...
                        valInfo.epch2err1( end ) * 100 );
                case 'REG'
                    mssg{ end + 1 } = ...
                        sprintf( 'LOSS ON TR SET: %f', ...
                        trInfo.epch2obj( end ) );
                    mssg{ end + 1 } = ...
                        sprintf( 'LOSS ON VAL SET: %f', ...
                        valInfo.epch2obj( end ) );
            end
        end
        function reportTrainStatus...
                ( this, trInfo, valInfo, attchFpaths, addrss )
            [ title, mssg ] = ...
                this.writeTrainReport( trInfo, valInfo );
            cellfun( @( str )fprintf( '%s\n', str ), mssg );
            if ~isempty( addrss )
                sendEmail( ...
                    'visionresearchreport@gmail.com', ...
                    'visionresearchreporter', ...
                    addrss, ...
                    title, ...
                    mssg, ...
                    attchFpaths );
            end
        end
        % Functions for CNN I/O.
        function name = getCnnName( this )
            if strcmp( this.initCnnName, 'RND' )
                name = sprintf( 'CNN_%s', ...
                    this.setting.changes );
                name = sprintf( '%s_OF_%s', ...
                    name, ...
                    this.srcImServer.getName );
            else
                if this.isinit
                    name = sprintf( 'CNN_%s', ...
                        this.initCnnName );
                else
                    name = sprintf( 'CNN_%s_STFRM_%s', ...
                        this.setting.changes, ...
                        this.initCnnName );
                    name = sprintf( '%s_OF_%s', ...
                        name, ...
                        this.srcImServer.getName );
                end
            end
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getCnnDir( this )
            dbDir = this.srcDb.dir;
            dir = fullfile( dbDir, this.getCnnName );
        end
        function dir = makeCnnDir( this )
            dir = this.getCnnDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getCnnPath( this, epch )
            fname = sprintf...
                ( 'E%04d.mat', epch );
            fpath = fullfile...
                ( this.getCnnDir, fname );
        end
        function fpath = getFigPath( this, epch )
            fname = sprintf...
                ( 'E%04d.pdf', epch );
            fpath = fullfile...
                ( this.getCnnDir, fname );
        end
        function fpath = getFiltImPath( this, lyid, epch )
            fname = sprintf...
                ( 'E%04d_L%02d.jpg', epch, lyid );
            fpath = fullfile...
                ( this.getCnnDir, fname );
        end
        function is = isTrainedAt( this, epch )
            fpath = this.getCnnPath( epch );
            is = exist( fpath, 'file' );
        end
        function loadCnnAt( this, epch )
            fpath = this.getCnnPath( epch );
            data = load( fpath );
            this.layers = data.layers;
            this.trInfo = data.trInfo;
            this.valInfo = data.valInfo;
        end
        function saveCnnAt( this, epch )
            this.makeCnnDir;
            fpath = this.getCnnPath( epch );
            layers = this.layers;
            trInfo = this.trInfo;
            valInfo = this.valInfo;
            save( fpath, 'layers', 'trInfo', 'valInfo' );
        end
        function saveFigAt...
                ( this, epch )
            this.makeCnnDir;
            fpath = this.getFigPath( epch );
            h = this.showTrainInfoFig;
            print( h, fpath, '-dpdf' );
        end
        function saveFiltImAt( this, epch )
            this.makeCnnDir;
            lyids = this.getVisFiltLyers( this.layers );
            for lyid = lyids
                filters = gather...
                    ( this.layers{ lyid }.filters );
                im = this.drawFilters( filters );
                im = imresize( im, 4, 'nearest' );
                fpath = this.getFiltImPath( lyid, epch );
                imwrite( im, fpath );
            end
        end
        % Functions for average image I/O.
        function name = getAvgImName( this )
            name = sprintf( 'AI_%s', ...
                this.srcImServer.setting.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getAvgImDir( this )
            dir = this.srcDb.dir;
        end
        function dir = makeAvgImDir( this )
            dir = this.getAvgImDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getAvgImPath( this )
            fname = strcat( this.getAvgImName, '.mat' );
            fpath = fullfile...
                ( this.getAvgImDir, fname );
        end
    end
    methods( Static )
        function layer = initLayer( config )
            switch config.type
                case 'conv'
                    layer.type = config.type;
                    layer.filters = 0.01 / ...
                        config.initWScal * ...
                        randn( config.filterSize, ...
                        config.filterSize, ...
                        config.filterDepth, ...
                        config.numFilter, 'single' );
                    layer.biases = config.initB * ...
                        ones( 1, config.numFilter, 'single' );
                    layer.stride = config.stride;
                    layer.pad = config.pad;
                    layer.filtersLearningRate = ...
                        config.filtersLearningRate;
                    layer.biasesLearningRate = ...
                        config.biasesLearningRate;
                    layer.filtersWeightDecay = ...
                        config.filtersWeightDecay;
                    layer.biasesWeightDecay = ...
                        config.biasesWeightDecay;
                    layer.filtersMomentum = ...
                        zeros( 'like', layer.filters );
                    layer.biasesMomentum = ...
                        zeros( 'like', layer.biases );
                case 'relu'
                    layer.type = config.type;
                case 'pool'
                    layer.type = config.type;
                    layer.method = config.method;
                    layer.pool = ...
                        [ config.windowSize, config.windowSize ];
                    layer.stride = config.stride;
                    layer.pad = config.pad;
                case 'normalize'
                    layer.type = config.type;
                    layer.param = [ config.localSize, 1, ...
                        config.alpha / config.localSize, config.beta ];
                case 'dropout'
                    layer.type = config.type;
                    layer.rate = config.rate;
                case 'softmaxloss'
                    layer.type = config.type;
                case 'nneucloss_reg'
                    layer.type = config.type;
            end
        end
        function info = updateEnergy( info, res, bgts, errtype )
            info.obj = info.obj + ...
                sum( double( gather( res( end ).x ) ) );
            switch errtype
                case 'MCLS'
                    predictions = gather( res( end - 1 ).x );
                    sz = size( predictions );
                    n = prod( sz( 1 : 2 ) );
                    [ ~, predictions ] = sort...
                        ( predictions, 3, 'descend' );
                    error = ~bsxfun( @eq, predictions, ...
                        reshape( bgts, 1, 1, 1, [  ] ) );
                    info.err1 = info.err1 +...
                        sum( sum( sum( error( :, :, 1, : ) ) ) ) / n;
                    info.err5 = info.err5 + ...
                        sum( sum( sum( min( error( :, :, 1 : 5, : ), [  ], 3 ) ) ) ) / n;
                case 'BCLS'
                    predictions = gather( res( end - 1 ).x );
                    sz = size( predictions );
                    n = prod( sz( 1 : 2 ) );
                    error = bsxfun( @times, predictions, bgts ) < 0;
                    info.err1 = info.err1...
                        + sum( error( : ) ) / n;
                case 'REG'
            end
        end
        function net = updateW...
                ( net, res, momentum, weightDecay, learnRate, numBatchIm )
            for l = 1 : numel( net.layers )
                if ~strcmp( net.layers{ l }.type, 'conv' ), continue; end;
                % ================================================================================
                % THE RULE FOR UPDATING WEIGHTS
                % ================================================================================
                % 1. The cost function with current weight W0 and data x is
                %    J = wDecay*0.5*W^2 + Loss(W,x).
                % 2. Derivative of the cost w.r.t W0 is
                %    dJ/dW|W0 = wDecay*W0 + gradLoss(W0,x),
                %    which means the inverse direction of W to minimize
                %    the cost such as,bdirsvy
                %    directionW = - learnRate*( wDecay*W0 + gradLoss(W0,x) ).
                % 3. To consider the inertia of the motion of W,
                %    we add a monentum term in the next direction of W such as,
                %    directionW = momentum*directionW - learnRate*( wDecay*W0 + gradLoss(W0,x) ).
                %    where momentum is 0.9, and wDecay is 0.0005.
                % 4. Finally, the current weight W0 is updated such as,
                %    W0' = W0 + directionW.
                % ================================================================================
                % The following codes correspond to 3.
                net.layers{ l }.filtersMomentum = ...
                    momentum * net.layers{ l }.filtersMomentum ...
                    - weightDecay * net.layers{ l }.filtersWeightDecay ...
                    * learnRate * net.layers{ l }.filtersLearningRate * net.layers{ l }.filters ...
                    - learnRate * net.layers{ l }.filtersLearningRate / numBatchIm * res( l ).dzdw{ 1 };
                net.layers{ l }.biasesMomentum = ...
                    momentum * net.layers{ l }.biasesMomentum ...
                    - weightDecay * net.layers{ l }.biasesWeightDecay ...
                    * learnRate * net.layers{ l }.biasesLearningRate * net.layers{ l }.biases ...
                    - learnRate * net.layers{ l }.biasesLearningRate / numBatchIm * res( l ).dzdw{ 2 };
                % The following codes correspond to 4.
                net.layers{ l }.filters = net.layers{ l }.filters + net.layers{ l }.filtersMomentum;
                net.layers{ l }.biases = net.layers{ l }.biases + net.layers{ l }.biasesMomentum;
            end % Go to next layer.
        end
        
        function lyids = ...
                getVisFiltLyers( layers )
            lyids = [  ];
            for l = 1 : numel( layers )
                ly = layers{ l };
                if ~strcmp( ly.type, 'conv' ),
                    continue;
                end;
                [ ~, ~, nch, ~ ] = size...
                    ( gather( ly.filters ) );
                if nch == 1 || nch == 3,
                    lyids( end + 1 ) = l;
                end;
            end
        end
        function im = drawFilters( filters )
            [ fr, fc, fch, fn ] = size( filters );
            num = ceil( sqrt( fn ) );
            layout = cell( num, num );
            for fid = 1 : fn
                layout{ fid } = linscale...
                    ( filters( :, :, :, fid ) );
            end
            for fid = fn + 1 : num^2
                layout{ fid } = ones...
                    ( fr, fc, fch, 'single' );
            end
            layout = layout';
            im = cell( num, 1 );
            for r = 1 : num
                im{ r } = cat( 2, layout{ r, : } );
            end
            im = cat( 1, im{ : } );
        end
    end
end