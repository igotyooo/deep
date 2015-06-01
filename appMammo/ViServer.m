classdef ViServer < handle
    properties
        numThreads;
        setting;
    end
    methods
        function this = ViServer( setting )
            this.numThreads                 = 8;
            this.setting.itpltn             = 'bicubic';
            this.setting.srcSide            = 256;
            this.setting.dstSide            = 227;
            this.setting.srcCh              = 3;
            this.setting.dstCh              = 3;
            this.setting.numFrame           = 10;
            this.setting.keepAspect         = true;
            this.setting.aug                = 'F25';
            this.setting.numAug             = 1;
            this.setting.nmlzByAvg          = true;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function setNumThreads( this, numThreads )
            this.numThreads = numThreads;
            fprintf( '%s: Use %d threads.\n', ...
                upper( mfilename ), numThreads );
        end
        function avgIm = computeAvgIm(...
                this, impaths, batchSize )
            % Set params.
            numThreads_ = this.numThreads;
            itpltn      = this.setting.itpltn;
            srcSide     = this.setting.srcSide;
            dstSide     = this.setting.dstSide;
            srcCh       = this.setting.srcCh;
            dstCh       = this.setting.dstCh;
            numFrame    = this.setting.numFrame;
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            % Do the job.
            iids = 1 : length( impaths );
            numBatch = length( 1 : batchSize : numel( iids ) );
            bid2avgim = cell( numBatch, 1 );
            cummt = 0; bid = 0;
            for beginIdx = 1 : batchSize : numel( iids ); btime = tic;
                endIdx = min...
                    ( beginIdx + batchSize - 1, numel( iids ) );
                biids = iids( beginIdx : endIdx );
                bims = this.loadBatchVideos_JPEG( ...
                    impaths( biids ), ...
                    numFrame, ...
                    numThreads_, ...
                    nargout == 0 );
                bims = this.augVideos...
                    ( bims, numFrame, itpltn, ...
                    srcSide, srcCh, ...
                    dstSide, dstCh, ...
                    keepAspect, aug, numAug );
                bid = bid + 1;
                bid2avgim{ bid } = mean( bims, 4 );
                % Print out.
                cummt = cummt + toc( btime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numBatch, bid, ...
                    sprintf( 'Compute avg im.' ), cummt );
            end
            avgIm = mean( cat( 4, bid2avgim{ : } ), 4 );
        end
        function bims = impath2cnninput( this, bimpaths, avgIm )
            % Set params.
            numThreads_ = this.numThreads;
            nmlzByAvg   = this.setting.nmlzByAvg;
            itpltn      = this.setting.itpltn;
            srcSide     = this.setting.srcSide;
            dstSide     = this.setting.dstSide;
            srcCh       = this.setting.srcCh;
            dstCh       = this.setting.dstCh;
            numFrame    = this.setting.numFrame;
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            prefatch    = nargout == 0;
            % Do the job.
            bims = this.loadBatchVideos_JPEG( ...
                    bimpaths, ...
                    numFrame, ...
                    numThreads_, ...
                    prefatch );
            if prefatch, bims = [  ]; return; end;
            bims = this.augVideos...
                ( bims, numFrame, itpltn, ...
                srcSide, srcCh, ...
                dstSide, dstCh, ...
                keepAspect, aug, numAug );
            if nmlzByAvg, bims = bsxfun( @minus, bims, avgIm ); end;
        end
        function bims = im2cnninput( this, bims, avgIm )
            % Set params.
            nmlzByAvg   = this.setting.nmlzByAvg;
            itpltn      = this.setting.itpltn;
            srcSide     = this.setting.srcSide;
            dstSide     = this.setting.dstSide;
            srcCh       = this.setting.srcCh;
            dstCh       = this.setting.dstCh;
            numFrame    = this.setting.numFrame;
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            % Do the job.
            if nargout == 0, bims = [  ]; return; end;
            bims = this.augVideos...
                ( bims, numFrame, itpltn, ...
                srcSide, srcCh, ...
                dstSide, dstCh, ...
                keepAspect, aug, numAug );
            if nmlzByAvg, bims = bsxfun( @minus, bims, avgIm ); end;
        end
        % Functions for object identification.
        function name = getName( this )
            name = sprintf( 'IS_%s', this.setting.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    methods( Static )
        function ims = loadBatchVideos_JPEG...
                ( impaths, numFrame, numThreads, preFetch )
            numv = numel( impaths );
            framePaths = cell( numv, 1 );
            for vidx = 1 : numv
                strti = randsample( 1 : ( numel( impaths{ vidx } ) - numFrame + 1 ), 1 );
                endi = strti + numFrame - 1;
                framePaths{ vidx } = impaths{ vidx }( strti : endi );
            end
            framePaths = cat( 1, framePaths{ : } );
            if preFetch
                vl_imreadjpeg...
                    ( framePaths, 'numThreads', numThreads, 'prefetch' );
                ims = [  ];
            else
                ims = vl_imreadjpeg...
                    ( framePaths, 'numThreads', numThreads );
            end
        end
        function augVideos = augVideos...
                ( ims, numFrame, itpltn, ...
                srcSide, srcCh, ...
                dstSide, dstCh, ...
                keepAspect, aug, numAug )
            switch aug
                case 'NONE'
                    aid2tf = [ ...
                        0.5;
                        0.5;
                        0.0; ];
                case 'F5'
                    aid2tf = [ ...
                        0.5, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0;
                        0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0;
                        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0; ];
                case 'F25'
                    [ tx, ty ] = meshgrid( linspace( 0, 1, 5 ) );
                    tfs = [ tx( : )'; ty( : )'; zeros( 1, numel( tx ) ) ];
                    tfs_ = tfs; tfs_( 3, : ) = 1;
                    aid2tf = [ tfs, tfs_ ] ;
            end
            % For each video, randomly select augmentation spec.
            numv = numel( ims ) / numFrame;
            [ ~, vidx2aid ] = sort...
                ( rand( size( aid2tf, 2 ), numv ), 1 );
            dstIidx = 1;
            augVideos = zeros...
                ( dstSide, dstSide, srcCh * numFrame, numv * numAug, 'single' );
            % For each video,
            for vidx = 1 : numv
                strti = ( vidx - 1 ) * numFrame + 1;
                endi = strti + numFrame - 1;
                frames = ims( strti : endi );
                % Depth resize.
                fid2ch = cellfun( @( f )size( f, 3 ), frames );
                if dstCh == 1
                    fid2resize = fid2ch ~= 1;
                    frames( fid2resize ) = ...
                        cellfun( @( im )mean( im, 3 ), frames( fid2resize ) );
                elseif dstCh == 3
                    fid2resize = fid2ch ~= 3;
                    frames( fid2resize ) = ...
                        cellfun( @( im )cat( 3, im, im, im ), frames( fid2resize ) );
                end
                vi = cat( 3, frames{ : } );
                % Just resize an image to srcSide.
                w = size( vi, 2 ); h = size( vi, 1 );
                if keepAspect,
                    factor = max( srcSide / h, srcSide / w );
                    if any( abs( factor - 1 ) > 0.0001 )
                        vi = imresize...
                            ( vi, 'scale', factor, 'method', itpltn );
                    end
                else
                    if w ~= srcSide || h ~= srcSide,
                        vi = imresize...
                            ( vi, [ srcSide, srcSide ], 'method', itpltn );
                    end
                end
                % Augment from an image to multiple images.
                w = size( vi, 2 ); h = size( vi, 1 );
                for aidx = 1 : numAug
                    aid = vidx2aid( aidx, vidx );
                    tf = aid2tf( :, aid );
                    % Crop.
                    dx = floor( ( w - dstSide ) * tf( 2 ) ); % Orientation of x-axis. ( = bias of x )
                    dy = floor( ( h - dstSide ) * tf( 1 ) ); % Orientation of y-axis. ( = bias of y )
                    sx = ( 1 : dstSide ) + dx;
                    sy = ( 1 : dstSide ) + dy;
                    % Flip.
                    if tf( 3 ), sx = fliplr( sx ); end
                    im_ = vi( sy, sx, : );
                    augVideos( :, :, :, dstIidx ) = im_;
                    dstIidx = dstIidx + 1;
                end
            end
        end
    end
end