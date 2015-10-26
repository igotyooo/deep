classdef PropObjCaffe2 < handle
    properties
        db;
        attNet;
        attNetName;
        rgbMean;
        stride;
        patchSide;
        lyid02lyid;
        scales;
        setting;
    end
    methods( Access = public )
        function this = PropObjCaffe2( db, setting )
            this.db = db;
            this.setting.normalizeImageMaxSide = 0;
            this.setting.numScaling = 24;
            this.setting.dilate = 1 / 4;
            this.setting.posIntOverRegnMoreThan = 1 / 3;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, netInfo, gpus )
            % Set parameters.
            maxSide = this.setting.normalizeImageMaxSide;
            numScaling = this.setting.numScaling;
            posIntOverRegnMoreThan = this.setting.posIntOverRegnMoreThan;
            % Fetch net on GPU.
            this.attNetName = netInfo.modelName;
            this.patchSide = netInfo.patchSide;
            this.stride = netInfo.stride;
            this.rgbMean = load( netInfo.rgbMeanPath, 'rgbMean' );
            this.rgbMean = this.rgbMean.rgbMean;
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            this.attNet = caffe.Net( netInfo.protoPath, netInfo.modelPath, 'test' );
            prefix = 'prob';
            clsLyrPostFix = 'cls';
            cornerNameTl = 'TL';
            cornerNameBr = 'BR';
            numCls = this.db.getNumClass;
            numLyr = numel( this.attNet.outputs );
            this.lyid02lyid = zeros( numLyr, 1 );
            for lyid = 1 : numLyr,
                lname = this.attNet.outputs{ lyid };
                if strcmp( lname( end - 2 : end ), clsLyrPostFix ),
                    lyid0 = numCls * 2 + 1;
                    this.lyid02lyid( lyid0 ) = lyid;
                    continue;
                end;
                data = textscan( lname, strcat( prefix, '%d_%s' ) );
                cid = data{ 1 } + 1;
                cornerName = data{ 2 }{ : };
                switch cornerName,
                    case cornerNameTl,
                        bias = 1;
                    case cornerNameBr,
                        bias = 2;
                    otherwise,
                        error( 'Strange corner name.\n' );
                end;
                lyid0 = ( cid - 1 ) * 2 + bias;
                this.lyid02lyid( lyid0 ) = lyid;
            end;
            if numel( unique( this.lyid02lyid ) ) ~= numCls * 2 + 1, 
                error( 'Wrong net output layer.\n' ); end;
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
        function rid2tlbr = iid2det( this, iid )
            % Initial guess.
            fpath = this.getPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.prop.rid2tlbr;
                rid2out = data.prop.rid2out;
            catch
                im = imread( this.db.iid2impath{ iid } );
                [ rid2out, rid2tlbr ] = ...
                    this.im2det( im );
                prop.rid2tlbr = rid2tlbr;
                prop.rid2out = rid2out;
                this.makeDir;
                save( fpath, 'prop' );
            end;
            % Compute each region score.
            if nargout,
                threshDir = 0.1; 0.15; 2; -Inf; 
                threshCls = -Inf; 
                numDimPerLyr = 4;
                numCls = this.db.getNumClass;
                numDimCls = numCls + 1;
                dimCls = numCls * numDimPerLyr * 2 + ( 1 : numDimCls );
                signDiag = 2;
                rid2tlbrProp = cell( numCls, 1 );
                for cid = 1 : numCls,
                    % Direction.
                    dimTl = ( cid - 1 ) * numDimPerLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerLyr - 1;
                    dimBr = dimTl + numDimPerLyr;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    rid2sTl = rid2outTl( signDiag, : );
                    rid2sBr = rid2outBr( signDiag, : );
                    rid2okTl = rid2sTl > threshDir;
                    rid2okBr = rid2sBr > threshDir;
                    % Classification.
                    rid2outCls = rid2out( dimCls, : );
                    [ rid2sCls, rid2pCls ] = max( rid2outCls, [  ], 1 );
                    rid2sCls = rid2sCls * 2 - sum( rid2outCls, 1 );
                    rid2okCls = ( rid2pCls == cid ) & ( rid2sCls > threshCls );
                    rid2ok = rid2okTl & rid2okBr & rid2okCls;
                    rid2tlbrBff = rid2tlbr( 1 : 4, rid2ok );
                    rid2tlbrProp{ cid } = rid2tlbrBff;
                end;
                rid2tlbr = cat( 2, rid2tlbrProp{ : } );
            end;
        end
        function [ rid2out, rid2tlbr ] = im2det( this, im )
            dilate = this.setting.dilate;
            maxSide = this.setting.normalizeImageMaxSide;
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
                dilate );
            rid2tlbr = round( resizeTlbr( rid2tlbr, imSize, imSize0 ) );
            rid2out = this.extractDenseActivationsCaffe( im, sid2size );
            if size( rid2out, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
        end
        function demo( this, fid, wait, iid )
            if nargin < 4,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            rid2tlbr = this.iid2det( iid );
            rid2tlbr = round( rid2tlbr );
            figure( fid );
            set( gcf, 'color', 'w' );
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
        end
        function rid2out = ...
                extractDenseActivationsCaffe( ...
                this, ...
                originalImage, ...
                targetImageSizes )
            regionDilate = this.setting.dilate;
            imageDilate = round( this.patchSide * regionDilate );
            interpolation = 'bicubic';
            numSize = size( targetImageSizes, 2 );
            rid2out = cell( numSize, 1 );
            trsiz = 0;
            tfwd =0;
            for sid = 1 : numSize,
                trsiz_ = tic;
                imSize = targetImageSizes( :, sid );
                if min( imSize ) + 2 * imageDilate < this.patchSide, continue; end;
                if prod( imSize + imageDilate * 2 ) > 15000000,
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
                lyid2out = this.feedforwardCaffe( im );
                lyid02out = lyid2out( this.lyid02lyid );
                for lyid0 = 1 : numel( lyid02out ),
                    out = lyid02out{ lyid0 };
                    [ nr, nc, z ] = size( out );
                    out = reshape( permute( out, [ 3, 1, 2 ] ), z, nr * nc );
                    lyid02out{ lyid0 } = out;
                end;
                rid2out{ sid } = cat( 1, lyid02out{ : } );
                tfwd = tfwd + toc( tfwd_ );
            end % Next scale.
            fprintf( '%s: Preproc t = %.2f sec, Fwd t = %.2f sec.\n', upper( mfilename ), trsiz, tfwd );
            % Aggregate for each layer.
            rid2out = cat( 2, rid2out{ : } );
        end
        function res = feedforwardCaffe( this, im )
            [ h, w, c, n ] = size( im );
            im = im( :, :, [ 3, 2, 1 ], : );
            im = permute( im, [ 2, 1, 3, 4 ] );
            im = { im };
            this.attNet.blobs( 'data' ).reshape( [ w, h, c, n ] );
            res = this.attNet.forward( im );
            res = cellfun( @( x )permute( x, [ 2, 1, 3, 4 ] ), res, 'UniformOutput', false );
        end;
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROPCAF2_%s_OF_%s', ...
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

