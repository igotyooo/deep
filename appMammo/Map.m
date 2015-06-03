classdef Map < handle
    properties
        srcDb;
        srcSvm;
        settingPre;
        settingPost;
    end
    methods
        function this = Map...
                ( srcDb, srcSvm, settingPre, settingPost )
            this.srcDb                  = srcDb;
            this.srcSvm                 = srcSvm;
            this.settingPre.scoreThrsh  = 0.0;
            this.settingPre.mapMaxSide  = 500;
            this.settingPost.weightByIm = false;
            this.settingPost.smoothMap  = 5;
            this.settingPost.smoothIm   = 10;
            this.settingPost.mapMaxVal  = 50;
            this.settingPre = setChanges...
                ( this.settingPre, settingPre, upper( mfilename ) );
            this.settingPost = setChanges...
                ( this.settingPost, settingPost, upper( mfilename ) );
        end
        function genMapDb( this )
            % Check if descs exist.
            fprintf( '%s: Check if map exist.\n', ...
                upper( mfilename ) );
            idx2iid = this.srcDb.getTeiids;
            idx2mpath = cellfun( ...
                @( iid )this.getPath( iid ), ...
                num2cell( idx2iid ), ...
                'UniformOutput', false );
            idx2exist = cellfun( ...
                @( path )exist( path, 'file' ), ...
                idx2mpath );
            this.makeDir;
            iids = idx2iid( ~idx2exist );
            if isempty( iids ),
                fprintf( '%s: No im to compute map.\n', ...
                    upper( mfilename ) ); return;
            end
            % Do the job.
            cnt = 0; cummt = 0; numIm = numel( iids );
            for iid = iids'; itime = tic;
                this.iid2map( iid );
                cummt = cummt + toc( itime );
                cnt = cnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, cnt, ...
                    'Desc im regns.', cummt );
            end
        end
        function cid2map = iid2map( this, iid )
            mpath = this.getPath( iid );
            try
                data = load( mpath );
                cid2map = data.cid2map;
            catch
                cid2map = this.genMap( iid );
                save( mpath, 'cid2map' );
            end
            % Post-processing.
            if nargout,
                im = imread( this.srcDb.iid2impath{ iid } );
                [ cid2map, cid2cont ] = this.postProcMap( im, cid2map );
            end
        end
        function cid2map = im2map( this, im )
            cid2map = this.genMap( im );
            % Post-processing.
            if nargout,
                [ cid2map, cid2cont ] = this.postProcMap( im, cid2map );
            end
        end
        function cid2map = genMap( this, src )
            if numel( src ) == 1,
                iid = src;
                im = imread( this.srcDb.iid2impath{ iid } );
            else
                im = src;
            end
            % Preparation.
            scoreThrsh = this.settingPre.scoreThrsh;
            mapMaxSide = this.settingPre.mapMaxSide;
            fisher = this.srcSvm.srcImDscrber.srcDscrber{ 1 };
            cid2w = this.srcSvm.loadSvm;
            imSize = size( im );
            imMaxSide = max( imSize( 1 : 2 ) );
            im_ = imresize( im, mapMaxSide ./ imMaxSide );
            imSize_ = size( im_ );
            r = imSize_( 1 ); c = imSize_( 2 );
            % Extract neural activtions.
            if numel( src ) == 1,
                iid = src;
                [ rid2geo, rid2fisher, ~ ] = ...
                    fisher.iid2descNoAp( iid );
            else
                [ rid2geo, rid2fisher, ~ ] = ...
                    fisher.im2descNoAp( im );
            end
            numRegn = size( rid2geo, 2 );
            rid2geo = resizeTlbr( rid2geo( 1 : 4, : ), imSize, imSize_ );
            rid2geo = bndtlbr( rid2geo, [ 1, 1, imSize_( 1 : 2 ) ] );
            rid2geo = round( rid2geo );
            % Test each fisher.
            cid2rid2score = cid2w' * ...
                cat( 1, rid2fisher, ones( 1, numRegn ) );
            cid2rid2score = gather( cid2rid2score ); clear rid2fisher;
            [ rank2rid2score, rank2rid2cid ] = sort( cid2rid2score, 1, 'descend' );
            % Compute a map for each class.
            numCls = size( cid2rid2score, 1 );
            cid2map = arrayfun( @( cid )zeros( r, c, 'single' ), ...
                ( 1 : numCls )', 'UniformOutput', false );
            rid2ok = ( rank2rid2score( 1, : ) > scoreThrsh )';
            rank2rid2cid = rank2rid2cid( :, rid2ok );
            rid2geo = rid2geo( :, rid2ok );
            numRegn = size( rid2geo, 2 );
            for rid = 1 : numRegn,
                cid = rank2rid2cid( 1, rid );
                r1 = rid2geo( 1, rid ); c1 = rid2geo( 2, rid );
                r2 = rid2geo( 3, rid ); c2 = rid2geo( 4, rid );
                cid2map{ cid }( r1 : r2, c1 : c2 ) = ...
                    cid2map{ cid }( r1 : r2, c1 : c2 ) + 1;
            end;
        end
        function [ cid2map, cid2cont ] = postProcMap( this, im, cid2map )
            cid2scaling = [ 5, 3, 0 ];
            weightByIm = this.settingPost.weightByIm;
            smoothMap = this.settingPost.smoothMap;
            smoothIm = this.settingPost.smoothIm;
            mapMaxVal = this.settingPost.mapMaxVal;
            mapMaxSide = max( size( cid2map{ 1 } ) );
            numCls = numel( cid2map );
            for cid = 1 : numCls,
                map = cid2map{ cid };
                if smoothMap, map = vl_imsmooth( map, smoothMap ); end;
                if weightByIm,
                    imSize = size( im );
                    imMaxSide = max( imSize( 1 : 2 ) );
                    im_ = imresize( im, mapMaxSide ./ imMaxSide );
                    im_ = single( mean( im_, 3 ) / 255 );
                    if smoothIm, im_ = vl_imsmooth( im_, smoothIm ); end;
                    map = map .* im_;
                end
                map = map * cid2scaling( cid );
                imMap = uint8( map * 255 / mapMaxVal );
                imMap = uint8( ind2rgb( gray2ind( imMap ), colormap ) * 255 );
                imMap = imresize( imMap, imSize( 1 : 2 ) );
                cid2map{ cid } = imMap;
                % Detect contours.
                
            end
            cid2cont = '';
        end
        % Functions for data I/O.
        function name = getName( this )
            name = sprintf( 'MAP_%s_OF_%s_BY_%s', ...
                this.settingPre.changes, this.srcDb.name, this.srcSvm.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this )
            name = this.getName;
            dir = fullfile( this.srcDb.dstDir, name );
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getDir, fname );
        end
    end
end