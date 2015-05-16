classdef Cnn < handle
    properties
        srcInOut;
        initCnnName;
        currentEpch;
        isinit;
        avgIm;
        layers;
        energyTr;
        energyVal;
        tsMetricTr;
        tsMetricVal;
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
            this.setting.architectureFunction       = @CFG_VL;
            this.setting.normalizeByAverageImage    = true;
            this.setting.weightDecay                = 0.0005;
            this.setting.momentum                   = 0.9;
            this.setting.learningRate               = [ 0.0100 * ones( 1, 25 ), ...
                                                        0.0010 * ones( 1, 25 ), ...
                                                        0.0001 * ones( 1, 15 ) ];
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        % Initialize CNN with random weights.
        % If you use this function, do not use fetch().
        function init( this )
            if this.setting.normalizeByAverageImage,
                this.computeAvgIms;
            end
            this.setting.arch = ...
                this.setting.architectureFunction(  );
            this.layers = cell( size( this.setting.arch ) );
            for l = 1 : length( this.setting.arch )
                this.layers{ l } = ...
                    this.initLayer...
                    ( this.setting.arch{ l } );
            end
            this.energyTr       = [  ];
            this.tsMetricTr     = [  ];
            this.energyVal      = [  ];
            this.tsMetricVal 	= [  ];
            this.isinit         = true;
        end
        % Initialize CNN with pre-trained weights.
        % If you use this function, do not use init().
        function fetch( this, cnnName, layers, avgIm_ )
            this.initCnnName = cnnName;
            this.layers = layers;
            if nargin > 3, 
                this.avgIm = avgIm_; 
            elseif nargin <= 3 && this.setting.normalizeByAverageImage,
                this.computeAvgIms;
            end;
            this.energyTr       = [  ];
            this.tsMetricTr     = [  ];
            this.energyVal      = [  ];
            this.tsMetricVal    = [  ];
            this.isinit         = true;
        end
        function train( this, useGpu, addrss )
            this.isinit = false;
            numEpch     = numel( this.setting.learningRate );
            if this.isTrainedAt( numEpch ),
                fprintf( '%s: Already trained. Load the model.\n', ...
                    upper( mfilename ) );
                this.loadCnnAt( numEpch );
                this.currentEpch = numEpch;
                fprintf( '%s: Done.\n', upper( mfilename ) ); return;
            end;
            % Start training.
            if useGpu, this.fetchLyrsOnGpu; end;
            rng( 0 ); ecummt = 0; ecnt = 0;
            for epch = 1 : numEpch; etime = tic;
                % Model I/O.
                if this.isTrainedAt( epch ), continue; end;
                if epch > 1 && this.isTrainedAt( epch - 1 ),
                    fprintf( '%s: Load the model of epch %d.\n', ...
                        upper( mfilename ), epch - 1 );
                    this.loadCnnAt( epch - 1 );
                    this.currentEpch = epch - 1;
                end
                % Train net.
                ecnt = ecnt + 1;
                this.trainAt( epch, ecnt, ecummt );
                this.currentEpch = epch;
                % Evaluate net.
                this.validate;
                % Save.
                fprintf( '%s: Save cnn at epch %d.\n', upper( mfilename ), epch );
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
            epch2perf = this.tsMetricVal;
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
        function computeAvgIms( this )
            fpath = this.getAvgImPath;
            try
                data = load( fpath );
                this.avgIm = data.avgIm;
                fprintf( '%s: Avg im is loaded.\n', ...
                    upper( mfilename ) );
            catch
                imsize = this.srcInOut.getImSize;
                numBchTr = this.srcInOut.numBchTr;
                r = imsize( 1 ); c = imsize( 2 ); ch = imsize( 3 );
                this.avgIm = zeros( r, c, ch, 'single' );
                numIm = 0; cummt = 0;
                for bid = 1 : numBchTr; btime = tic;
                    ims = this.srcInOut.provdBchTr;
                    this.avgIm = this.avgIm + sum( ims, 4 );
                    numIm = numIm + size( ims, 4 );
                    cummt = cummt + toc( btime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numBchTr,  bid, 'compute avg im.', cummt );
                end
                this.avgIm = this.avgIm / numIm;
                avgIm = this.avgIm;
                this.makeAvgImDir;
                save( this.getAvgImPath, 'avgIm' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
        end
        function trainAt( this, epch, ecnt, ecummt )
            % Set params.
            useGpu          = isa( this.layers{ 1 }.filters, 'gpuArray' );
            nmlzByAvg       = this.setting.normalizeByAverageImage;
            numEpch         = numel( this.setting.learningRate );
            learnRate       = this.setting.learningRate( epch );
            prevLearnRate   = this.setting.learningRate( max( 1, epch - 1 ) );
            weightDecay     = this.setting.weightDecay;
            momentum        = this.setting.momentum;
            batchSize       = this.srcInOut.getBatchSize;
            numBchTr        = this.srcInOut.getNumBatchTr;
            evergyTr_       = 0;
            tsMetricTr_     = 0;
            net.layers      = this.layers;
            one             = single( 1 );
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
            for b = 1 : numBchTr; btime = tic;
                % Get batch images and corrersponding GT.
                [ ims, gts ] = this.srcInOut.provdBchTr;
                % Put data to GPU memory.
                if useGpu, ims = gpuArray( ims ); gts = gpuArray( gts ); one = gpuArray( one ); end;
                if nmlzByAvg, ims = bsxfun( @minus, ims, this.avgIm ); end;
                % Attatch the GT to CNN to compute the energy.
                net.layers{ end }.class = gts;
                % Do forward/backward.
                res = my_simplenn...
                    ( net, ims, one, res, ...
                    'conserveMemory', true, ...
                    'sync', true );
                % Compute the energy and task-specific evaluation metric.
                evergyTr_ = evergyTr_ + ...
                    sum( double( gather( res( end ).x ) ) );
                tsMetricTr_ = tsMetricTr_ + ...
                    this.srcInOut.computeTsMetric( res, gts );
                % Update w by the gredient step.
                net = this.updateW...
                    ( net, res, momentum, weightDecay, learnRate, batchSize );
                % filters = gather( net.layers{ 1 }.filters );
                % im = this.drawFilters( filters );
                % im = imresize( im, 4, 'nearest' );
                % figure( 2 ); imshow( im ); drawnow;
                % Print out the status.
                btime = toc( btime ); bcummt = bcummt + btime; bcnt = bcnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numBchTr, bcnt, ...
                    sprintf( '(%d/%d) done for epch %d. (%dims/s)', ...
                    bcnt, numBchTr, epch, round( batchSize / btime ) ), bcummt );
                if ecnt > 1,
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numEpch - epch + ecnt,  ecnt, ...
                        sprintf( '(Epch=%d/%d) done to finish.', epch, numEpch ), ecummt );
                end
            end % Go to next training batch.
            % Update the energy and task-specific evaluation metric.
            this.energyTr( end + 1 ) = evergyTr_ / ( batchSize * numBchTr );
            this.tsMetricTr( end + 1 ) = tsMetricTr_ / ( batchSize * numBchTr );
            this.layers = net.layers;
        end
        function validate( this )
            useGpu          = isa( this.layers{ 1 }.filters, 'gpuArray' );
            net.layers      = this.layers;
            nmlzByAvg       = this.setting.normalizeByAverageImage;
            batchSize       = this.srcInOut.getBatchSize;
            numBchVal       = this.srcInOut.getNumBatchVal;
            evergyVal_      = 0;
            tsMetricVal_	= 0;
            % In this epoch, validate the model.
            bcummt = 0; bcnt = 0; res = [  ];
            for b = 1 : numBchVal; btime = tic;
                % Get batch images and corrersponding GT.
                [ ims, gts ] = this.srcInOut.provdBchVal;
                % Put data to GPU memory.
                if useGpu, ims = gpuArray( ims ); gts = gpuArray( gts ); end;
                if nmlzByAvg, ims = bsxfun( @minus, ims, this.avgIm ); end;
                % Attatch the GT to CNN to compute the energy.
                net.layers{ end }.class = gts;
                % Do forward only.
                res = my_simplenn...
                    ( net, ims, [  ], res, ...
                    'disableDropout', true, ...
                    'conserveMemory', true, ...
                    'sync', true );
                % Compute the energy and task-specific evaluation metric.
                evergyVal_ = evergyVal_ + ...
                    sum( double( gather( res( end ).x ) ) );
                tsMetricVal_ = tsMetricVal_ + ...
                    this.srcInOut.computeTsMetric( res, gts );
                % Compute the remaining time and print out the progress.
                btime = toc( btime ); bcummt = bcummt + btime; bcnt = bcnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numBchVal, bcnt, ...
                    sprintf( '(%d/%d) done for val. (%dims/s)', ...
                    bcnt, numBchVal, round( batchSize / btime ) ), bcummt );
            end % Go to next validation batch.
            % Update the energy and task-specific evaluation metric.
            this.energyVal( end + 1 ) = evergyVal_ / ( batchSize * numBchVal );
            this.tsMetricVal( end + 1 ) = tsMetricVal_ / ( batchSize * numBchVal );
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
            if ~isempty( this.avgIm ), this.avgIm = gpuArray( this.avgIm ); end;
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
        % Function for error plot.
        function h = showTrainInfoFig( this )
            numEpch = numel( this.energyTr );
            h = figure( 1 ); clf;
            % Plot energy.
            subplot( 1, 2, 1 );
            semilogy( 1 : numEpch, ...
                this.energyTr, 'k.-' );
            set( gca, 'yscale', 'linear' );
            hold on;
            semilogy( 1 : numEpch, ...
                this.energyVal, 'b.-' );
            xlabel( 'Epoch' ); ylabel( 'Energy' );
            legend( { 'Train', 'Val' }, 'Location', 'Best' );
            grid on;
            % Plot task-specific evaluation metric.
            tsMetricName = this.srcInOut.getTsMetricName;
            subplot( 1, 2, 2 );
            plot( 1 : numEpch, ...
                this.tsMetricTr, 'k.-' );
            hold on;
            plot( 1 : numEpch, ...
                this.tsMetricVal, 'b.-' );
            xlabel( 'Epoch' ); ylabel( tsMetricName );
            legend( { 'Train', 'Val' }, 'Location', 'Best' );
            grid on;
            set( gcf, 'color', 'w' );
        end
        % Functions for report training.
        function [ title, mssg ] = writeTrainReport( this )
            epch = length( this.energyVal );
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
            mssg{ end + 1 } = sprintf( 'CNN: %s', ...
                this.getCnnDirName );
            mssg{ end + 1 } = ...
                sprintf( 'ENERGY: %.4f%', ...
                this.energyVal( end ) );
            mssg{ end + 1 } = ...
                sprintf( '%s: %.4f%', ...
                upper( tsMetricName ), ...
                this.tsMetricVal( end ) );
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
        % Functions for CNN I/O.
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
            this.energyTr = data.cnn.energyTr;
            this.energyVal = data.cnn.energyVal;
            this.tsMetricTr = data.cnn.tsMetricTr;
            this.tsMetricVal = data.cnn.tsMetricVal;
        end
        function saveCnnAt( this, epch )
            this.makeCnnDir;
            fpath = this.getCnnPath( epch );
            cnn.avgIm = this.avgIm;
            cnn.layers = this.layers;
            cnn.energyTr = this.energyTr;
            cnn.energyVal = this.energyVal;
            cnn.tsMetricTr = this.tsMetricTr;
            cnn.tsMetricVal = this.tsMetricVal;
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
                    ( this.layers{ lyid }.filters );
                im = this.drawFilters( filters );
                im = imresize( im, 4, 'nearest' );
                fpath = this.getFiltImPath( lyid, epch );
                imwrite( im, fpath );
            end
        end
        % Functions for average image I/O.
        function name = getAvgImName( this )
            name = sprintf( 'AI_OF_%s', ...
                this.srcInOut.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getAvgImDir( this )
            dir = this.srcInOut.srcDb.dir;
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