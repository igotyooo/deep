classdef PropObjCornerPerCls < handle
    properties
        db;
        attNet;
        stride;
        patchSide;
        scales;
        setting;
    end
    methods( Access = public )
        function this = PropObjCornerPerCls( db, attNet, setting )
            this.db = db;
            this.attNet = attNet;
            this.setting.normalizeImageMaxSide = 0;
            this.setting.numScaling = 24;
            this.setting.dilate = 1 / 4;
            this.setting.posIntOverRegnMoreThan = 1 / 3;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, gpus )
            % Set parameters.
            maxSide = this.setting.normalizeImageMaxSide;
            numScaling = this.setting.numScaling;
            posIntOverRegnMoreThan = this.setting.posIntOverRegnMoreThan;
            % Fetch net on GPU.
            this.attNet.layers{ end }.type = 'softmax';
            this.attNet = Net.fetchNetOnGpu( this.attNet, gpus );
            % Determine stride and patch side.
            fprintf( '%s: Determine stride and patch side.\n', ...
                upper( mfilename ) );
            [ this.patchSide, this.stride ] = ...
                getNetProperties( this.attNet, numel( this.attNet.layers ) - 1 );
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
                threshDir = 7; -Inf; 
                ov = 1; 0.7; 
                numDimPerLyr = 5;
                numCls = this.db.getNumClass;
                signDiag = 2;
                rid2tlbrProp = cell( numCls, 1 );
                rid2scoreProp = cell( numCls, 1 );
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerLyr - 1;
                    dimBr = dimTl + numDimPerLyr;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    [ rid2sTl, rid2pTl ] = max( rid2outTl, [  ], 1 );
                    [ rid2sBr, rid2pBr ] = max( rid2outBr, [  ], 1 );
                    rid2sTl = rid2sTl * 2 - sum( rid2outTl, 1 );
                    rid2sBr = rid2sBr * 2 - sum( rid2outBr, 1 );
                    rid2s = ( rid2sTl + rid2sBr ) / 2;
                    rid2okTl = ( rid2pTl == signDiag ) & ( rid2sTl > threshDir );
                    rid2okBr = ( rid2pBr == signDiag ) & ( rid2sBr > threshDir );
                    rid2ok = rid2okTl & rid2okBr;
                    rid2tlbrBff = rid2tlbr( 1 : 4, rid2ok );
                    rid2scoreBff = rid2s( rid2ok );
                    if ov ~= 1,
                        pick = nms_iou( [ rid2tlbrBff; rid2scoreBff ]', ov );
                        rid2tlbrBff = rid2tlbrBff( :, pick );
                        rid2scoreBff = rid2scoreBff( pick );
                    end;
                    rid2tlbrProp{ cid } = rid2tlbrBff;
                    rid2scoreProp{ cid } = rid2scoreBff;
                end;
                rid2tlbr = cat( 2, rid2tlbrProp{ : } );
                rid2score = cat( 2, rid2scoreProp{ : } );
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
            rid2out = ...
                extractDenseActivations( ...
                im, ...
                this.attNet, ...
                numel( this.attNet.layers ) - 1, ...
                sid2size, ...
                this.patchSide, ...
                dilate );
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
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROPCORPERCLS_%s_OF_%s', ...
                this.setting.changes, ...
                this.attNet.name );
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

