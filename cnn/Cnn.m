classdef Cnn < handle
    properties
        srcInOut;
        initCnnName;
        currentEpch;
        isinit;
        imStats;
        layers;
        eid2energyTr;
        eid2energyVal;
        eid2metricTr;
        eid2metricVal;
        gpus;
        setting;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public interface. The following functions are used only. %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = public )
        function this = Cnn( srcInOut, setting )
            this.srcInOut                           = srcInOut;
            this.initCnnName                        = 'RND';
            this.currentEpch                        = 0;
            this.setting.normalizeImage             = true;
            this.setting.weightDecay                = 0.0005;
            this.setting.momentum                   = 0.9;
            this.setting.modelType                  = 'dropout'; % 'bnorm' for batch normalization.
            this.setting.learningRate               = [ 0.0100 * ones( 1, 25 ), ...
                                                        0.0010 * ones( 1, 25 ), ...
                                                        0.0001 * ones( 1, 15 ) ];
                                                    % logspace( -2, -4, 60 ); for dropdout.
                                                    % logspace(-1, -4, 20); for bnorm.
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        % Initialize network.
        function init( this, cnnName, layers, imStats_ )
            this.initCnnName = cnnName;
            this.layers = layers;
            if nargin > 3, 
                this.imStats = imStats_; 
            elseif nargin <= 3 && this.setting.normalizeImage,
                this.computeImStats;
            end;
            this.eid2energyTr = [  ];
            this.eid2metricTr = [  ];
            this.eid2energyVal = [  ];
            this.eid2metricVal = [  ];
            this.isinit = true;
        end
        function train( this, gpus, addrss )
            this.isinit = false;
            this.gpus = gpus;
            this.setGpus( gpus );
            numEpch = numel( this.setting.learningRate );
            if this.isTrainedAt( numEpch ),
                fprintf( '%s: Already trained. Load the model.\n', ...
                    upper( mfilename ) );
                this.loadCnnAt( numEpch );
                this.currentEpch = numEpch;
                fprintf( '%s: Done.\n', upper( mfilename ) ); return;
            end;
            % Start training.
            rng( 0 ); ecummt = 0;
            for epch = 1 : numEpch; etime = tic;
                % Model I/O.
                if this.isTrainedAt( epch ), continue; end;
                if epch > 1 && this.isTrainedAt( epch - 1 ),
                    fprintf( '%s: Load the model of epch %d.\n', ...
                        upper( mfilename ), epch - 1 );
                    this.loadCnnAt( epch - 1 );
                    this.currentEpch = epch - 1;
                end
                % Copy layers.
                net.layers = this.layers;
                % Fetch network on gpu.
                if numGpus, net = this.fetchNetOnGpu( net ); end;
                % Training at this epoch.
                if numGpus <= 1,
                    % Train net.
                    [ net, energyTr, metricTr ] = ...
                        this.trainAt( net, epch );
                    % Evaluate net.
                else % Multi-gpu mode.
                    spmd( numGpus ),
                        % Train net.
                        [ net, energyTr, metricTr ] = ...
                            this.trainAt( net, epch );
                        % Evaluate net.
                    end;
                    energyTr = sum( cell2mat( energyTr ) );
                    metricTr = sum( cell2mat( metricTr ) );
                end;
                this.currentEpch = epch; 
                % Fetch network on cpu.
                if numGpus <= 1,
                    net = vl_simplenn_move( net, 'cpu' );
                else % Multi-gpu mode.
                    spmd( numGpus ),
                        net = vl_simplenn_move( net, 'cpu' );
                    end; net = net{ 1 };
                end;
                % Copy layers.
                this.layers = net.layers;
                % Update statistics.
                batchSize = this.srcInOut.getBatchSize;
                numBchTr = this.srcInOut.getNumBatchTr;
                this.eid2energyTr( end + 1 ) = energyTr / ( batchSize * numBchTr );
                this.eid2metricTr( end + 1 ) = metricTr / ( batchSize * numBchTr );
                % Save.
                fprintf( '%s: Save net at epch %d.\n', ...
                    upper( mfilename ), epch );
                this.saveCnnAt( epch );
                fprintf( '%s: Done.\n', upper( mfilename ) );
                this.saveFigAt( epch );
                this.saveFiltImAt( epch );
                % Report.
                attch{ 1 } = this.getFigPath( epch );
                for lyid = this.getVisFiltLyers( this.layers ),
                    attch{ end + 1 } = this.getFiltImPath( lyid, epch ); end;
                this.reportTrainStatus( attch, addrss ); clear attch;
                ecummt = ecummt + toc( etime );
            end % Go to next epoch.
        end
        function fetchBestCnn( this, smoothFactor )
            epch2perf = this.eid2metricVal;
            numEpch = numel( epch2perf );
            filtSize = floor( numEpch / smoothFactor ); 
            pad = floor( filtSize / 2 );
            if ~mod( filtSize, 2 ), filtSize = filtSize - 1; end; % Should be odd.
            epch2perfSmooth = cat( 2, ...
                epch2perf( 1 ) * ones( 1, pad ), ...
                epch2perf, epch2perf( end ) * ones( 1, pad ) );
            filter = ones( 1, filtSize );
            epch2perfSmooth = conv( epch2perfSmooth, filter, 'valid' ) / filtSize;
            [ ~, bestEpch ] = min( epch2perfSmooth );
            bestTsMetricVal = epch2perf( bestEpch );
            tsMetricName = this.srcInOut.getTsMetricName;
            fprintf( ...
                '%s: Load cnn of epch %d. (%s of %.4f)\n', ...
                upper( mfilename ), ...
                bestEpch, ...
                upper( tsMetricName ), ...
                bestTsMetricVal );
            this.loadCnnAt( bestEpch );
            this.currentEpch = bestEpch;
            fprintf( '%s: Fetch done.\n', ...
                upper( mfilename ) );
            plot( 1 : numel( epch2perf ), epch2perf, 'b-' ); hold on;
            plot( 1 : numel( epch2perf ), epch2perfSmooth, 'r-' ); hold off;
        end
        % Function for identificatioin.
        function name = getCnnName( this )
            if strcmp( this.initCnnName, 'RND' )
                name = sprintf( 'CNN_E%03d_%s_OF_%s', ...
                    this.currentEpch, ...
                    this.setting.changes, ...
                    this.srcInOut.getName );
            else
                if this.isinit
                    name = sprintf( 'CNN_E%03d_%s', ...
                        this.currentEpch, ...
                        this.initCnnName );
                else
                    name = sprintf( 'CNN_E%03d_%s_STFRM_%s_OF_%s', ...
                        this.currentEpch, ...
                        this.setting.changes, ...
                        this.initCnnName, ...
                        this.srcInOut.getName );
                end
            end
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function name = getCnnDirName( this )
            name = this.getCnnName;
            name( 4 : 8 ) = '';
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%
    % Private interface. %
    %%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        function computeImStats( this )
            fpath = this.getImStatsPath;
            try
                data = load( fpath );
                this.imStats = data.imStats;
                fprintf( '%s: Im stats loaded.\n', ...
                    upper( mfilename ) );
            catch
                numBchTr = this.srcInOut.numBchTr;
                averageImage = cell( numBchTr, 1 );
                rgbMean = cell( numBchTr, 1 );
                rgbCovariance = cell( numBchTr, 1 );
                cummt = 0;
                for bid = 1 : numBchTr; btime = tic;
                    ims = this.srcInOut.provdBchTr;
                    rgb = reshape( permute( ims, [ 3, 1, 2, 4 ] ), 3, [  ] );
                    numPixel = size( rgb, 2 );
                    averageImage{ bid } = mean( ims, 4 );
                    rgbMean{ bid } = sum( rgb, 2 ) / numPixel;
                    rgbCovariance{ bid } = rgb * rgb' / numPixel;
                    cummt = cummt + toc( btime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numBchTr,  bid, 'compute im stats.', cummt );
                end
                averageImage = mean( cat( 4, averageImage{ : } ), 4 );
                rgbMean = mean( cat( 2, rgbMean{ : } ), 2 );
                rgbCovariance = mean( cat( 3, rgbCovariance{ : } ), 3 );
                rgbCovariance = rgbCovariance - rgbMean * rgbMean';
                imStats.averageImage = averageImage;
                imStats.rgbMean = rgbMean;
                imStats.rgbCovariance = rgbCovariance;
                this.imStats = imStats;
                this.makeImStatsDir;
                save( this.getImStatsPath, 'imStats' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
            [ v, d ] = eig( this.imStats.rgbCovariance );
            this.imStats.rgbVariance = 0.1 * sqrt( d ) * v';
        end
        function [ net, energy, metric ] = trainAt( this, net, epch )
            % Set params.
            numGpus = numel( this.gpus );
            normalizeImage = this.setting.normalizeImage;
            numEpch = numel( this.setting.learningRate );
            learnRate = this.setting.learningRate( epch );
            prevLearnRate = this.setting.learningRate( max( 1, epch - 1 ) );
            weightDecay = this.setting.weightDecay;
            momentum = this.setting.momentum;
            batchSize = this.srcInOut.getBatchSize;
            numBchTr = this.srcInOut.getNumBatchTr;
            energy = 0;
            metric = 0;
            one = single( 1 );
            % Reset momentum if the learning rate is changed.
            if learnRate ~= prevLearnRate,
                fprintf('%s: Change learning rate to %f. Reset momentum.\n', ...
                    upper( mfilename ), learnRate );
                for l = 1 : numel( net.layers ),
                    if ~isfield( net.layers{ l }, 'weights' ), continue; end;
                    net.layers{ l }.filtersMomentum = ...
                        0 * net.layers{ l }.filtersMomentum;
                    net.layers{ l }.biasesMomentum = ...
                        0 * net.layers{ l }.biasesMomentum;
                end;
            end;
            % For each batch,
            res = [  ]; mmap = [  ];
            for b = 1 : numBchTr; btime = tic;
                % Get batch images and corrersponding GT.
                [ ims, gts ] = this.srcInOut.provdBchTr;
                % Put data to GPU memory.
                if numGpus > 0, ims = gpuArray( ims ); gts = gpuArray( gts ); one = gpuArray( one ); end;
                % Normalize input image.
                if normalizeImage, ims = this.normalizeIms( ims ); end;
                % Attatch the GT to network to compute the energy.
                net.layers{ end }.class = gts;
                % Do forward/backward.
                res = vl_simplenn( ...
                    net, ims, one, res, ...
                    'accumulate', false, ...
                    'disableDropout', false, ...
                    'conserveMemory', true, ...
                    'backPropDepth', +inf, ...
                    'sync', true ); % 'sync' makes things slow but on MATLAB 2014a it is necessary.
                % Accumulate energy and task-specific evaluation metric.
                energy = energy + ...
                    sum( double( gather( res( end ).x ) ) );
                metric = metric + ...
                    this.srcInOut.computeTsMetric( res, gts );
                % Update w by gradients.
                if numGpus > 1,
                    % Make a sharable memory to wirite gradients for each gpu.
                    if isempty( mmap ), mmap = this.map_gradients( net, res, numGpus ); end;
                    % Write gradients for each gpu on the memory.
                    this.write_gradients( mmap, net, res );
                    % Wait until every gpu finishes writing.
                    labBarrier(  );
                    % Merge the gradients and update the weights.
                    [ net, res ] = this.updateW...
                        ( net, res, momentum, weightDecay, learnRate, batchSize, mmap );
                else
                    net = this.updateW...
                        ( net, res, momentum, weightDecay, learnRate, batchSize );
                end;
                % Print out the status.
                btime = toc( btime );
                fprintf( '%s: ', upper( mfilename ) );
                fprintf( 'Epch %d/%d, Bch %d/%d, %dims/s.\n', ...
                    epch, numEpch, b, batchSize, round( batchSize / btime ) );
            end; % Go to next training batch.
            % Update the energy and task-specific evaluation metric.
        end
        function layer = initLayer( this, config )
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
                case 'custom'
                    layer.type = config.type;
                    meta = metaclass( this.srcInOut );
                    layer.backward = str2func( strcat( meta.Name, '.backward' ) );
                    layer.forward = str2func( strcat( meta.Name, '.forward' ) );
            end
        end
        function ims = normalizeIms( this, ims )
            % Select offset type among averageImage and averageRGB.
            % offset = this.imStats.averageImage;
            offset = this.imStats.rgbMean;
            rgbVar = this.imStats.rgbVariance;
            % Normalize image.
            for i = 1 : size( ims, 4 ),
                offset_ = bsxfun( @plus, offset, reshape( rgbVar * randn( 3, 1 ), 1, 1, 3 ) );
                ims( :, :, :, i ) = bsxfun( @minus, ims( :, :, :, i ), offset_ );
            end;
        end
        % Function for error plot.
        function h = showTrainInfoFig( this )
            numEpch = numel( this.eid2energyTr );
            h = figure( 1 ); clf;
            % Plot energy.
            subplot( 1, 2, 1 );
            semilogy( 1 : numEpch, ...
                this.eid2energyTr, 'k.-' );
            set( gca, 'yscale', 'linear' );
            hold on;
            semilogy( 1 : numEpch, ...
                this.eid2energyVal, 'b.-' );
            xlabel( 'Epoch' ); ylabel( 'Energy' );
            legend( { 'Train', 'Val' }, 'Location', 'Best' );
            grid on;
            % Plot task-specific evaluation metric.
            tsMetricName = this.srcInOut.getTsMetricName;
            subplot( 1, 2, 2 );
            plot( 1 : numEpch, ...
                this.eid2metricTr, 'k.-' );
            hold on;
            plot( 1 : numEpch, ...
                this.eid2metricVal, 'b.-' );
            xlabel( 'Epoch' ); ylabel( tsMetricName );
            legend( { 'Train', 'Val' }, 'Location', 'Best' );
            grid on;
            set( gcf, 'color', 'w' );
        end
        % Functions for report training.
        function [ title, mssg ] = writeTrainReport( this )
            epch = length( this.eid2energyVal );
            tsMetricName = this.srcInOut.getTsMetricName;
            title = sprintf( '%s: TRAINING REPORT AT EPOCH %d', ...
                upper( mfilename ), epch );
            mssg = {  };
            mssg{ end + 1 } = '_______________';
            mssg{ end + 1 } = 'TRAINING REPORT';
            mssg{ end + 1 } = sprintf( 'DATABASE: %s', ...
                this.srcInOut.srcDb.dbName );
            mssg{ end + 1 } = sprintf( 'INOUT: %s', ...
                this.srcInOut.getName );
            mssg{ end + 1 } = sprintf( 'NET: %s', ...
                this.getCnnDirName );
            mssg{ end + 1 } = ...
                sprintf( 'ENERGY: %.4f%', ...
                this.eid2energyVal( end ) );
            mssg{ end + 1 } = ...
                sprintf( '%s: %.4f%', ...
                upper( tsMetricName ), ...
                this.eid2metricVal( end ) );
            mssg{ end + 1 } = ' ';
        end
        function reportTrainStatus...
                ( this, attchFpaths, addrss )
            [ title, mssg ] = ...
                this.writeTrainReport;
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
        % Functions for network I/O.
        function dir = getCnnDir( this )
            dbDir = this.srcInOut.srcDb.dir;
            dir = fullfile( dbDir, this.getCnnDirName );
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
            this.layers = data.cnn.layers;
            this.eid2energyTr = data.cnn.eid2energyTr;
            this.eid2energyVal = data.cnn.eid2energyVal;
            this.eid2metricTr = data.cnn.eid2metricTr;
            this.eid2metricVal = data.cnn.eid2metricVal;
        end
        function saveCnnAt( this, epch )
            this.makeCnnDir;
            fpath = this.getCnnPath( epch );
            cnn.imStats = this.imStats;
            cnn.layers = this.layers;
            cnn.eid2energyTr = this.eid2energyTr;
            cnn.eid2energyVal = this.eid2energyVal;
            cnn.eid2metricTr = this.eid2metricTr;
            cnn.eid2metricVal = this.eid2metricVal;
            save( fpath, 'cnn' );
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
                    ( this.layers{ lyid }.weights{ 1 } );
                im = this.drawFilters( filters );
                im = imresize( im, 4, 'nearest' );
                fpath = this.getFiltImPath( lyid, epch );
                imwrite( im, fpath );
            end
        end
        % Functions for average image I/O.
        function name = getImStatsName( this )
            name = sprintf( 'IS_OF_%s', ...
                this.srcInOut.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getImStatsDir( this )
            dir = this.srcInOut.srcDb.dstDir;
        end
        function dir = makeImStatsDir( this )
            dir = this.getImStatsDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getImStatsPath( this )
            fname = strcat( this.getImStatsName, '.mat' );
            fpath = fullfile...
                ( this.getImStatsDir, fname );
        end
    end
    methods( Static )
        function setGpus( gpus )
            numGpus = numel( gpus );
            if numGpus > 1,
                if isempty( gcp( 'nocreate' ) ),
                    parpool( 'local', numGpus );
                    spmd, gpuDevice( gpus( labindex ) ), end;
                end;
            elseif numGpus == 1,
                gpuDevice( gpus );
            end;
            fname = fullfile( tempdir, 'matconvnet.bin' );
            if exist( fname, 'file' ), delete( fname ); end;
        end
        function net = fetchNetOnGpu( net, gpus )
            numGpus = numel( gpus );
            if numGpus == 1,
                net = vl_simplenn_move( net, 'gpu' );
            elseif numGpus > 1,
                spmd( numGpus )
                    net = vl_simplenn_move( net, 'gpu' );
                end;
            end;
        end
        function [ net, res ] = updateW...
                ( net, res, momentum, weightDecay, learnRate, batchSize, mmap )
            for l = 1 : numel( net.layers ),
                if ~isfield( net.layers{ l }, 'weights' ), continue; end;
                for j = 1 : numel( res( l ).dzdw ),
                    % Accumualte from multiple GPUs if needed.
                    if nargin >= 7,
                        tag = sprintf( 'l%d_%d', l, j );
                        tmp = zeros( size( mmap.Data( labindex ).( tag ) ), 'single' );
                        for g = setdiff( 1 : numel( mmap.Data ), labindex ),
                            tmp = tmp + mmap.Data( g ).( tag );
                        end; res( l ).dzdw{ j } = res( l ).dzdw{ j } + tmp;
                    end;
                    % ================================================================================
                    % THE RULE FOR UPDATING WEIGHTS
                    % ================================================================================
                    % 1. The cost function with current weight W0 and data x is
                    %    J = wDecay*0.5*W^2 + Loss(W,x).
                    % 2. Derivative of the cost w.r.t W0 is
                    %    dJ/dW|W0 = wDecay*W0 + gradLoss(W0,x),
                    %    which means the inverse direction of W to minimize
                    %    the cost such as,
                    %    directionW = - learnRate*( wDecay*W0 + gradLoss(W0,x) ).
                    % 3. To consider the inertia of the motion of W,
                    %    we add a monentum term in the next direction of W such as,
                    %    directionW = momentum*directionW - learnRate*( wDecay*W0 + gradLoss(W0,x) ).
                    %    where momentum is 0.9, and wDecay is 0.0005.
                    % 4. Finally, the current weight W0 is updated such as,
                    %    W0' = W0 + directionW.
                    % ================================================================================
                    % The following codes correspond to 3.
                    thisDecay = weightDecay * net.layers{ l }.weightDecay( j );
                    thisLR = learnRate * net.layers{ l }.learningRate( j );
                    net.layers{ l }.momentum{ j } = ...
                        momentum * net.layers{ l }.momentum{ j } ...
                        - thisDecay * net.layers{ l }.weights{ j } ...
                        - ( 1 / batchSize ) * res( l ).dzdw{ j };
                    % The following codes correspond to 4.
                    net.layers{ l }.weights{ j } = ...
                        net.layers{ l }.weights{ j } + ...
                        thisLR * net.layers{ l }.momentum{ j };
                end;
            end; % Go to next layer.
        end
        function lyids = ...
                getVisFiltLyers( layers )
            lyids = [  ];
            for l = 1 : numel( layers ),
                if ~isfield( layers{ l }, 'weights' ), continue; end;
                [ ~, ~, nch, ~ ] = size...
                    ( gather( layers{ l }.weights{ 1 } ) );
                if nch == 1 || nch == 3,
                    lyids( end + 1 ) = l;
                end;
            end;
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
        function mmap = map_gradients( net, res, numGpus )
            fname = fullfile( tempdir, 'matconvnet.bin' );
            format = {  };
            for i = 1 : numel( net.layers ),
                for j = 1 : numel( res( i ).dzdw ),
                    format( end + 1, 1 : 3 ) = ...
                        { 'single', size( res( i ).dzdw{ j } ), sprintf( 'l%d_%d', i, j ) };
                end;
            end;
            format( end + 1, 1 : 3 ) = { 'double', [ 3, 1 ], 'errors' };
            if ~exist( fname, 'file' ) && ( labindex == 1 ),
                f = fopen( fname, 'wb' );
                for g = 1 : numGpus,
                    for i = 1 : size( format, 1 ),
                        fwrite( f, zeros( format{ i, 2 }, format{ i, 1 } ), format{ i, 1 } );
                    end;
                end;
                fclose( f );
            end;
            labBarrier(  );
            mmap = memmapfile( fname, 'Format', format, 'Repeat', numGpus, 'Writable', true );
        end
        function write_gradients( mmap, net, res )
            for i = 1 : numel( net.layers )
                for j = 1 : numel( res( i ).dzdw )
                    mmap.Data( labindex ).( sprintf( 'l%d_%d', i, j ) ) = gather( res( i ).dzdw{ j } );
                end
            end
        end
    end
end