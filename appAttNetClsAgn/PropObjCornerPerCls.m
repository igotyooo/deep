classdef PropObjCornerPerCls < handle
    properties
        db;
        attNet;
        stride;
        patchSide;
        scales;
        setting;
        settingPost;
    end
    methods( Access = public )
        function this = PropObjCornerPerCls( db, attNet, setting )
            this.db = db;
            this.attNet = attNet;
            this.setting.numScaling = 24;
            this.setting.dilate = 1 / 4;
            this.setting.posIntOverRegnMoreThan = 1 / 3;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, gpus )
            % Set parameters.
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
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
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
                thresh = -Inf; 
                numDimPerLyr = 5;
                numDimPerCls = numDimPerLyr * 2;
                signDiag = 2;
                numLyr = size( rid2out, 1 ) / numDimPerLyr;
                numCls = numLyr / 2;
                rid2ok = false( 1, size( rid2out, 2 ) );
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerCls + 1;
                    dimTl = dimTl : dimTl + numDimPerLyr - 1;
                    dimBr = dimTl + numDimPerLyr;
                    [ rid2sTl, rid2pTl ] = ...
                        max( rid2out( dimTl, : ), [  ], 1 );
                    [ rid2sBr, rid2pBr ] = ...
                        max( rid2out( dimBr, : ), [  ], 1 );
                    rid2okTl = ( rid2pTl == signDiag ) & ( rid2sTl > thresh );
                    rid2okBr = ( rid2pBr == signDiag ) & ( rid2sBr > thresh );
                    rid2ok = rid2ok | ( rid2okTl & rid2okBr );
                end;
                rid2tlbr = rid2tlbr( 1 : 4, rid2ok );
            end;
        end
        function [ rid2out, rid2tlbr ] = im2det( this, im )
            dilate = this.setting.dilate;
            [ r, c, ~ ] = size( im );
            imSize0 = [ r; c; ];
            sid2size = round( bsxfun( @times, this.scales, imSize0 ) );
            rid2out = ...
                extractDenseActivations( ...
                im, ...
                this.attNet, ...
                numel( this.attNet.layers ) - 1, ...
                sid2size, ...
                this.patchSide, ...
                dilate );
            rid2tlbr = ...
                extractDenseRegions( ...
                imSize0, ...
                sid2size, ...
                this.patchSide, ...
                this.stride, ...
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
                'PROPCOR_%s_OF_%s', ...
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
            piormt = this.setting.posIntOverRegnMoreThan;
            piormt = num2str( piormt );
            piormt( piormt == '.' ) = 'P';
            name = sprintf( 'SFTE_N%03d_PIORMT%s_OF_%s', ...
                numScaling, piormt, this.db.getName );
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

