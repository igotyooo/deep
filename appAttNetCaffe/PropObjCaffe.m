classdef PropObjCaffe < handle
    properties
        db;
        attNet;
        attNetName;
        rgbMean;
        stride;
        patchSide;
        weights;
        biases;
        directions;
        scales;
        setting;
    end
    methods( Access = public )
        function this = PropObjCaffe( db, setting )
            this.db = db;
            this.setting.normalizeImageMaxSide = 0;
            this.setting.numScaling = 24;
            this.setting.dilate = 1 / 4;
            this.setting.maximumImageSize = 9e6;
            this.setting.posIntOverRegnMoreThan = 1 / 3;
            this.setting.numTopClassification = 1;
            this.setting.numTopDirection = 1;
            this.setting.directionVectorSize = 30;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, netInfo, gpus )
            % Set parameters.
            maxSide = this.setting.normalizeImageMaxSide;
            numScaling = this.setting.numScaling;
            posIntOverRegnMoreThan = this.setting.posIntOverRegnMoreThan;
            % Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            numDirection = 3;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.directions.did2vecTl = [ [ cos( did2angTl' ); sin( did2angTl' ); ], [ 0; 0; ] ];
            this.directions.did2vecBr = [ [ cos( did2angBr' ); sin( did2angBr' ); ], [ 0; 0; ] ];
            % Fetch output layer's parameters on GPU.
            this.attNetName = netInfo.modelName;
            this.patchSide = netInfo.patchSide;
            this.stride = netInfo.stride;
            this.rgbMean = load( netInfo.rgbMeanPath, 'rgbMean' );
            this.rgbMean = this.rgbMean.rgbMean;
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            net = caffe.Net( netInfo.protoPath, netInfo.modelPath, 'test' );
            prefix = 'dir';
            clsLyrName = 'cls';
            cornerNameTl = 'TL';
            cornerNameBr = 'BR';
            numCls = this.db.getNumClass;
            numLyr = numCls * 2 + 1;
            mlid2clid = zeros( numLyr, 1 );
            mlid2name = cell( numLyr, 1 );
            mlid2w = cell( numLyr, 1 );
            mlid2b = cell( numLyr, 1 );
            for lid = 1 : numLyr,
                if lid == numLyr,
                    lname = clsLyrName;
                else
                    if mod( lid, 2 ), cornerName = cornerNameTl; else cornerName = cornerNameBr;  end;
                    lname = sprintf( '%s%d_%s', prefix, ( ( lid - 1 ) - mod( lid - 1, 2 ) ) / 2, cornerName );
                end;
                mlid2clid( lid ) = net.name2layer_index( lname );
                mlid2name{ lid } = lname;
                mlid2w{ lid } = net.params( lname, 1 ).get_data(  );
                mlid2b{ lid } = net.params( lname, 2 ).get_data(  );
                mlid2b{ lid } = mlid2b{ lid }';
            end;
            this.weights = gpuArray( cat( 4, mlid2w{ : } ) );
            this.biases = gpuArray( cat( 2, mlid2b{ : } ) );
            % Fetch att net on GPU without the output layer.
            caffe.reset_all(  );
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            this.attNet = caffe.Net( netInfo.protoPathTest, netInfo.modelPath, 'test' );
            % Determine scaling factors.
            fpath = this.getScaleFactorPath;
            try
                fprintf( '%s: Try to load scaling factors.\n', upper( mfilename ) );
                data = load( fpath );
                this.scales = data.data.scales;
            catch
                fprintf( '%s: Determine scaling factors.\n', ...
                    upper( mfilename ) );
                setid = 2;
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == setid );
                if maxSide,
                    oid2iid = this.db.oid2iid( this.db.iid2setid( this.db.oid2iid ) == setid );
                    oid2imsize = this.db.iid2size( :, oid2iid );
                    numRegn = size( oid2tlbr, 2 );
                    for oid = 1 : numRegn,
                        [ ~, oid2tlbr( :, oid ) ] = normalizeImageSize...
                            ( maxSide, oid2imsize( :, oid ), oid2tlbr( :, oid ) );
                    end;
                end;
                referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr, numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                fprintf( '%s: Done.\n', upper( mfilename ) );
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
        end
        function propObj( this )
            iids = this.db.getTeiids;
            fprintf( '%s: Check if proposals exist.\n', ...
                upper( mfilename ) );
            paths = arrayfun( ...
                @( iid )this.getPath( iid ), iids, ...
                'UniformOutput', false );
            exists = cellfun( ...
                @( path )exist( path, 'file' ), paths );
            this.makeDir;
            iids = iids( ~exists );
            numIm = numel( iids );
            if ~numIm, fprintf( '%s: No im to process.\n', upper( mfilename ) ); end;
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2det( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, ...
                    'Prop obj.', cummt );
            end;
        end
        function [ rid2tlbr, nid2rid, nid2cid ] = iid2det( this, iid, cids )
            fpath = this.getPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                nid2rid = data.nid2rid;
                nid2cid = data.nid2cid;
            catch
                % Initial guess.
                im = imread( this.db.iid2impath{ iid } );
                [ rid2out, rid2tlbr ] = this.im2det( im ); % For localization, cids should be intput to this function.
                % Compute each region score.
                dvecSize = this.setting.directionVectorSize;
                numTopCls = this.setting.numTopClassification;
                numTopDir = this.setting.numTopDirection;
                numCls = this.db.getNumClass;
                signStop = 4;
                signDiag = 2;
                dimCls = numCls * signStop * 2 + ( 1 : ( numCls + 1 ) );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2rank2cid ] = sort( rid2outCls, 1, 'descend' );
                rid2tlbrProp = cell( numCls, 1 );
                if nargin < 3, cids = 1 : numCls; else cids = cids( : )'; end;
                for cid = cids,
                    % Direction: DD condition.
                    dimTl = ( cid - 1 ) * signStop * 2 + 1;
                    dimTl = dimTl : dimTl + signStop - 1;
                    dimBr = dimTl + signStop;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    [ ~, rid2rank2ptl ] = sort( rid2outTl, 1, 'descend' );
                    [ ~, rid2rank2pbr ] = sort( rid2outBr, 1, 'descend' );
                    rid2ptl = rid2rank2ptl( 1, : );
                    rid2pbr = rid2rank2pbr( 1, : );
                    rid2okTl = any( rid2rank2ptl( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2okBr = any( rid2rank2pbr( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2ss = rid2ptl == signStop & rid2pbr == signStop;
                    rid2dd = rid2okTl & rid2okBr;
                    rid2dd = rid2dd & ( ~rid2ss );
                    rid2dd = rid2dd & ( rid2ptl == signDiag | rid2pbr == signDiag );
                    % Classification.
                    rid2bgd = rid2rank2cid( 1, : ) == ( numCls + 1 );
                    rid2okCls = any( rid2rank2cid( 1 : numTopCls, : ) == cid, 1 );
                    rid2okCls = rid2okCls & ( ~rid2bgd );
                    % Update.
                    rid2cont = rid2dd & rid2okCls;
                    numCont = sum( rid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = rid2tlbr( 1 : 4, rid2cont );
                    idx2ptl = rid2ptl( rid2cont );
                    idx2pbr = rid2pbr( rid2cont );
                    idx2tlbrWarp = [ ...
                        this.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.directions.did2vecBr( :, idx2pbr ) * dvecSize + this.patchSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ this.patchSide, this.patchSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                    rid2tlbrProp{ cid } = idx2tlbr;
                end;
                rid2tlbr = round( cat( 2, rid2tlbrProp{ : } ) );
                [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
                nid2cid = rid2tlbr( 5, : )';
                rid2tlbr = rid2tlbr_';
                % Save data.
                this.makeDir;
                save( fpath, 'rid2tlbr', 'nid2rid', 'nid2cid' );
            end;
        end
        function [ rid2out, rid2tlbr ] = im2det( this, im )
            dilate = this.setting.dilate;
            maxSide = this.setting.normalizeImageMaxSide;
            maximumImageSize = this.setting.maximumImageSize;
            [ r, c, ~ ] = size( im );
            imSize0 = [ r; c; ];
            if maxSide, imSize = normalizeImageSize( maxSide, imSize0 ); else imSize = imSize0; end;
            sid2size = round( bsxfun( @times, this.scales, imSize ) );
            rid2tlbr = ...
                extractDenseRegions( ...
                imSize, ...
                sid2size, ...
                this.patchSide, ...
                this.stride, ...
                dilate, ...
                maximumImageSize );
            rid2tlbr = round( resizeTlbr( rid2tlbr, imSize, imSize0 ) );
            rid2out = this.extractDenseActivationsCaffe( im, sid2size );
            if size( rid2out, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
        end
        function demo( this, fid, position, wait, iid )
            if nargin < 5,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            rid2tlbr = this.iid2det( iid );
            rid2tlbr = round( rid2tlbr );
            figure( fid );
            if wait,
                for rid = 1 : size( rid2tlbr, 2 ),
                    plottlbr( rid2tlbr( :, rid ), im, false, 'r' );
                    title( sprintf( 'Object proposal: %d/%d regions. (IID%06d)', ...
                        rid, size( rid2tlbr, 2 ), iid ) );
                    hold off;
                    waitforbuttonpress;
                end;
            else
                plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
                title( sprintf( 'Object proposal: %d regions. (IID%06d)', ...
                    size( rid2tlbr, 2 ), iid ) );
                hold off;
            end;
            set( gcf, 'color', 'w' );
            setFigPos( gcf, position ); drawnow;
        end
        function demoGivenCls( this, fid, position, wait, iid, cids )
            im = imread( this.db.iid2impath{ iid } );
            rid2tlbr = this.iid2det( iid, cids );
            rid2tlbr = round( rid2tlbr );
            figure( fid );
            if wait,
                for rid = 1 : size( rid2tlbr, 2 ),
                    plottlbr( rid2tlbr( :, rid ), im, false, 'r' );
                    title( sprintf( 'Object proposal: %d/%d regions. (IID%06d)', ...
                        rid, size( rid2tlbr, 2 ), iid ) );
                    hold off;
                    waitforbuttonpress;
                end;
            else
                plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
                title( sprintf( 'Object proposal: %d regions. (IID%06d)', ...
                    size( rid2tlbr, 2 ), iid ) );
                hold off;
            end;
            set( gcf, 'color', 'w' );
            setFigPos( gcf, position ); drawnow;
        end
        function rid2out = ...
                extractDenseActivationsCaffe( ...
                this, ...
                originalImage, ...
                targetImageSizes )
            regionDilate = this.setting.dilate;
            maximumImageSize = this.setting.maximumImageSize;
            imageDilate = round( this.patchSide * regionDilate );
            interpolation = 'bilinear';
            numSize = size( targetImageSizes, 2 );
            rid2out = cell( numSize, 1 );
            trsiz = 0;
            tfwd =0;
            for sid = 1 : numSize,
                trsiz_ = tic;
                imSize = targetImageSizes( :, sid );
                if min( imSize ) + 2 * imageDilate < this.patchSide, continue; end;
                if prod( imSize + imageDilate * 2 ) > maximumImageSize,
                    fprintf( '%s: Warning) Im of %s rejected.\n', ...
                        upper( mfilename ), mat2str( imSize ) ); continue;
                end;
                im = imresize( ...
                    originalImage, imSize', ...
                    'method', interpolation );
                im = single( im );
                roi = [ ...
                    1 - imageDilate; ...
                    1 - imageDilate; ...
                    imSize( : ) + imageDilate; ];
                im = normalizeAndCropImage( im, roi, this.rgbMean, interpolation );
                fprintf( '%s: Feed im of %dX%d size.\n', upper( mfilename ), size( im, 1 ), size( im, 2 ) );
                trsiz = trsiz + toc( trsiz_ );
                tfwd_ = tic;
                y = this.feedforwardCaffe( im );
                [ nr, nc, z ] = size( y );
                y = reshape( permute( y, [ 3, 1, 2 ] ), z, nr * nc );
                rid2out{ sid } = y;
                tfwd = tfwd + toc( tfwd_ );
            end % Next scale.
            fprintf( '%s: Preproc t = %.2f sec, Fwd t = %.2f sec.\n', upper( mfilename ), trsiz, tfwd );
            % Aggregate for each layer.
            rid2out = cat( 2, rid2out{ : } );
        end
        function y = feedforwardCaffe( this, im )
            [ h, w, c, n ] = size( im );
            im = im( :, :, [ 3, 2, 1 ], : );
            im = permute( im, [ 2, 1, 3, 4 ] );
            im = { im };
            this.attNet.blobs( 'data' ).reshape( [ w, h, c, n ] );
            x = this.attNet.forward( im );
            if numel( x ) > 1, error( 'Output should be a single.' ); end;
            x = x{ 1 };
            x = permute( x, [ 2, 1, 3, 4 ] );
            x = gpuArray( x );
            y = vl_nnconv( x, this.weights, this.biases, 'pad', 0, 'stride', 1 );
            y = gather( y );
            clear x;
        end
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROPCAF_%s_OF_%s', ...
                this.setting.changes, ...
                this.attNetName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this )
            name = this.getName;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'PROP_', name );
            end
            dir = fullfile...
                ( this.db.dstDir, name );
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this, iid )
            fname = sprintf...
                ( 'ID%06d.mat', iid );
            fpath = fullfile...
                ( this.getDir, fname );
        end
        function name = getScaleFactorName( this )
            numScaling = this.setting.numScaling;
            maxSide = this.setting.normalizeImageMaxSide;
            piormt = this.setting.posIntOverRegnMoreThan;
            piormt = num2str( piormt );
            piormt( piormt == '.' ) = 'P';
            name = sprintf( 'SFTE_N%03d_PIORMT%s_NIMS%d_OF_%s', ...
                numScaling, piormt, maxSide, this.db.getName );
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
    end
end

