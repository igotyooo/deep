classdef ImServer < handle
    properties
        numThreads;
        setting;
    end
    methods
        function this = ImServer( setting )
            this.numThreads                 = 8;
            this.setting.itpltn             = 'bicubic';
            this.setting.srcSide            = 256;
            this.setting.dstSide            = 227;
            this.setting.srcCh              = 3;
            this.setting.dstCh              = 3;
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
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            % Do the job.
            iids = 1 : length( impaths );
            numBatch = length( 1 : batchSize : numel( iids ) );
            bid2avgim = cell( numBatch, 1 );
            cummt = 0; bid = 0;
            for beginIdx = 1 : batchSize : numel( iids ); btime = tic;
                endIdx = min( beginIdx + batchSize - 1, numel( iids ) );
                biids = iids( beginIdx : endIdx );
                bims = this.loadBatchIms_JPEG_wrapper...
                    ( impaths( biids ), numThreads_ );
                bims = this.augIms...
                    ( bims, itpltn, ...
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
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            % Do the job.
            bims = this.loadBatchIms_JPEG...
                ( bimpaths, numThreads_, nargout == 0 );
            if nargout == 0, bims = [  ]; return; end;
            bims = this.augIms...
                ( bims, itpltn, ...
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
            keepAspect  = this.setting.keepAspect;
            aug         = this.setting.aug;
            numAug      = this.setting.numAug;
            % Do the job.
            if nargout == 0, bims = [  ]; return; end;
            bims = this.augIms...
                ( bims, itpltn, ...
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
        function ims = loadBatchIms_JPEG_wrapper...
                ( impaths, numThreads )
            ims = ImServer.loadBatchIms_JPEG...
                ( impaths, numThreads, nargout == 0 );
        end
        function ims = loadBatchIms_JPEG...
                ( impaths, numThreads, preFetch )
            if preFetch
                vl_imreadjpeg...
                    ( impaths, 'numThreads', numThreads, 'prefetch' );
                ims = [  ];
            else
                ims = vl_imreadjpeg...
                    ( impaths, 'numThreads', numThreads );
            end
        end
        function augIms = augIms...
                ( ims, itpltn, ...
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
            % For each image, randomly select augmentations to be done.
            [ ~, iidx2aid ] = sort...
                ( rand( size( aid2tf, 2 ), numel( ims ) ), 1 );
            dstIidx = 1;
            augIms = zeros...
                ( dstSide, dstSide, srcCh, numel( ims ) * numAug, 'single' );
            % For each image,
            for iidx = 1 : numel( ims )
                im = ims{ iidx };
                % Just resize an image to srcSide.
                w = size( im, 2 ); h = size( im, 1 );
                if keepAspect,
                    factor = max( srcSide / h, srcSide / w );
                    if any( abs( factor - 1 ) > 0.0001 )
                        im = imresize...
                            ( im, 'scale', factor, 'method', itpltn );
                    end
                else
                    if w ~= srcSide || h ~= srcSide,
                        im = imresize...
                            ( im, [ srcSide, srcSide ], 'method', itpltn );
                    end
                end
                % Augment from an image to multiple images.
                w = size( im, 2 ); h = size( im, 1 );
                for aidx = 1 : numAug
                    aid = iidx2aid( aidx, iidx );
                    tf = aid2tf( :, aid );
                    % Crop.
                    dx = floor( ( w - dstSide ) * tf( 2 ) ); % Orientation of x-axis. ( = bias of x )
                    dy = floor( ( h - dstSide ) * tf( 1 ) ); % Orientation of y-axis. ( = bias of y )
                    sx = ( 1 : dstSide ) + dx;
                    sy = ( 1 : dstSide ) + dy;
                    % Flip.
                    if tf( 3 ), sx = fliplr( sx ); end
                    im_ = im( sy, sx, : );
                    % Depth resize.
                    if dstCh == 3 && size( im_, 3 ) == 1
                        im_ = cat( 3, im_, im_, im_ );
                    elseif dstCh == 1 && size( im_, 3 ) == 3
                        im_ = mean( im_, 3 );
                    end
                    augIms( :, :, :, dstIidx ) = im_;
                    dstIidx = dstIidx + 1;
                end
            end
        end
    end
end