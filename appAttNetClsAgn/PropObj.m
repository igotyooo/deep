classdef PropObj < handle
    properties
        db;
        attNet;
        stride;
        patchSide;
        scales;
        settingMain;
        settingPost;
    end
    methods( Access = public )
        function this = PropObj( db, attNet, settingMain, settingPost )
            this.db = db;
            this.attNet = attNet;
            this.settingMain.numScaling = 24;
            this.settingMain.dilate = 1 / 4;
            this.settingMain.posIntOverRegnMoreThan = 1 / 3;
            this.settingPost.mergingOverlap = 0.7;
            this.settingPost.mergingType = 'OV'; 'NMS';
            this.settingPost.mergingMethod = 'WAVG'; 'MAX';
            this.settingPost.minimumNumSupportBox = 0;
            this.settingMain = setChanges...
                ( this.settingMain, settingMain, upper( mfilename ) );
            this.settingPost = setChanges...
                ( this.settingPost, settingPost, upper( mfilename ) );
        end
        function init( this, gpus )
            % Set parameters.
            numScaling = this.settingMain.numScaling;
            posIntOverRegnMoreThan = this.settingMain.posIntOverRegnMoreThan;
            % Fetch net on GPU.
            this.attNet.layers{ end }.type = 'softmax';
            this.attNet = Net.fetchNetOnGpu( this.attNet, gpus );
            % Determine stride and patch side.
            fprintf( '%s: Determine stride and patch side.\n', ...
                upper( mfilename ) );
            [ this.patchSide, this.stride ] = ...
                getNetProperties( this.attNet, numel( this.attNet.layers ) - 1 );
            % Determine scaling factors.
            fprintf( '%s: Determine scaling factors.\n', ...
                upper( mfilename ) );
            oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
            referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
            [ scalesRow, scalesCol ] = determineImageScaling...
                ( oid2tlbr, numScaling, referenceSide, true );
            this.scales = [ scalesRow, scalesCol ]';
            fprintf( '%s: Done.\n', ...
                upper( mfilename ) );
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
        function [ rid2tlbr, rid2score, rid2cid ] = ...
                iid2det( this, iid )
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
                save( fpath, 'prop' );
            end;
            if nargout,
                % Compute each region score.
                dimCls = this.attNet.layers{ end }.dimCls;
                dimDir = this.attNet.layers{ end }.dimDir;
                rid2outCls = rid2out( dimCls, : );
                rid2outDir = rid2out( dimDir, : );
                [ rid2scoreCls, rid2cidCls ] = ...
                    max( rid2outCls, [  ], 1 );
                [ rid2scoreDir, rid2cidDir ] = ...
                    max( rid2outDir, [  ], 1 );
                rid2okCls = rid2cidCls ~= ( numel( dimCls ) - 1 ) & ...
                    rid2cidCls ~= numel( dimCls );
                rid2okDir = rid2cidDir == 1;
                rid2ok = rid2okCls & rid2okDir;
                rid2outCls = rid2outCls( :, rid2ok );
                rid2outDir = rid2outDir( :, rid2ok );
                rid2scoreCls = rid2scoreCls( rid2ok ) - ...
                    sum( rid2outCls( numel( dimCls ) - 1 : end, : ), 1 );
                rid2scoreDir = rid2scoreDir( rid2ok ) * 2 - sum( rid2outDir, 1 );
                rid2score = rid2scoreCls + rid2scoreDir;
                rid2cid = rid2cidCls( rid2ok );
                rid2tlbr = rid2tlbr( 1 : 4, rid2ok );
                % Post-processing: merge bounding boxes.
                if this.settingPost.mergingOverlap ~= 1,
                    [ rid2tlbr, rid2score, rid2cid ] = ...
                        this.merge( rid2tlbr, rid2score, rid2cid );
                end;
            end;
        end
        function [ rid2out, rid2tlbr ] = im2det( this, im )
            dilate = this.settingMain.dilate;
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
        function [ rid2tlbr, rid2score, rid2cid ] = ...
                merge( this, rid2tlbr, rid2score, rid2cid )
            mergingOverlap = this.settingPost.mergingOverlap;
            mergingType = this.settingPost.mergingType;
            mergingMethod = this.settingPost.mergingMethod;
            minNumSuppBox = this.settingPost.minimumNumSupportBox;
            cids = unique( rid2cid );
            numCls = numel( cids );
            rid2tlbr_ = cell( numCls, 1 );
            rid2score_ = cell( numCls, 1 );
            rid2cid_ = cell( numCls, 1 );
            for cidx = 1 : numCls,
                cid = cids( cidx );
                rid2ok = rid2cid == cid;
                switch mergingType,
                    case 'NMS',
                        [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = nms( ...
                            [ rid2tlbr( :, rid2ok ); rid2score( rid2ok ); ]', ...
                            mergingOverlap, minNumSuppBox, mergingMethod );
                        rid2tlbr_{ cidx } = rid2tlbr_{ cidx }';
                    case 'OV',
                        [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = ov( ...
                            rid2tlbr( :, rid2ok ), rid2score( rid2ok ), ...
                            mergingOverlap, minNumSuppBox, mergingMethod );
                end
                rid2cid_{ cidx } = cid * ones( size( rid2score_{ cidx } ) );
            end;
            rid2tlbr = cat( 2, rid2tlbr_{ : } );
            rid2score = cat( 1, rid2score_{ : } );
            rid2cid = cat( 1, rid2cid_{ : } );
            [ rid2score, idx ] = sort( rid2score, 'descend' );
            rid2tlbr = rid2tlbr( :, idx );
            rid2cid = rid2cid( idx );
        end
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROP_%s_OF_%s', ...
                this.settingMain.changes, ...
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
    end
end

