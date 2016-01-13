classdef Map < handle
    properties
        srcDb;
        srcSvm;
        setting;
    end
    methods
        function this = Map...
                ( srcDb, srcSvm, setting )
            this.srcDb = srcDb;
            this.srcSvm = srcSvm;
            this.setting.scoreThrsh = 0;
            this.setting.mapMaxSide = 512;
            this.setting= setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function scoreRegnDb( this )
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
                    'Compute a map.', cummt );
            end
        end
        function cid2map = iid2map( this, iid )
            spath = this.getPath( iid );
            try
                data = load( spath );
                cid2rid2score = data.cid2rid2score;
                rid2tlbr = data.rid2tlbr;
                imSize = data.imSize;
            catch
                [ cid2rid2score, rid2tlbr, imSize ] = this.iid2regnScore( iid );
                this.makeDir;
                save( spath, 'cid2rid2score', 'rid2tlbr', 'imSize' );
            end
            if nargout,
                cid2map = this.vote( cid2rid2score, rid2tlbr, imSize );
            end;
        end
        function [ cid2rid2score, rid2tlbr, imSize ] = iid2regnScore( this, iid )
            fisher = this.srcSvm.srcImDscrber.srcDscrber{ 1 };
            cid2w = this.srcSvm.loadSvm;
            [ rid2tlbr, rid2fisher, imSize ] = fisher.iid2descNoAp( iid );
            cid2rid2score = cid2w' * cat( 1, rid2fisher, ones( 1, size( rid2fisher, 2 ) ) );
        end
        function cid2map = vote( this, cid2rid2score, rid2tlbr, imSize )
            scoreThrsh = this.setting.scoreThrsh;
            mapMaxSide = this.setting.mapMaxSide;
            imSize_ = round( imSize( 1 : 2 ) * mapMaxSide ./ max( imSize( 1 : 2 ) ) );
            r = imSize_( 1 ); c = imSize_( 2 );
            rid2tlbr = resizeTlbr( rid2tlbr( 1 : 4, : ), imSize, imSize_ );
            rid2tlbr = round( rid2tlbr );
            [ rank2rid2score, rank2rid2cid ] = sort( cid2rid2score, 1, 'descend' );
            numCls = size( cid2rid2score, 1 );
            cid2map = arrayfun( @( cid )zeros( r, c, 'single' ), ...
                ( 1 : numCls )', 'UniformOutput', false );
            rid2ok = ( rank2rid2score( 1, : ) > scoreThrsh )';
            rank2rid2cid = rank2rid2cid( :, rid2ok );
            rid2tlbr_ = rid2tlbr( :, rid2ok );
            numRegn = size( rid2tlbr_, 2 );
            for rid = 1 : numRegn,
                cid = rank2rid2cid( 1, rid );
                r1 = rid2tlbr_( 1, rid ); c1 = rid2tlbr_( 2, rid );
                r2 = rid2tlbr_( 3, rid ); c2 = rid2tlbr_( 4, rid );
                cid2map{ cid }( r1 : r2, c1 : c2 ) = ...
                    cid2map{ cid }( r1 : r2, c1 : c2 ) + 1;
            end;
        end
        % Functions for data I/O.
        function name = getName( this )
            name = sprintf( 'RSCORE_OF_%s_BY_%s', ...
                this.srcDb.name, this.srcSvm.getName );
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