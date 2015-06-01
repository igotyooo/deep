classdef DbCls < handle
    properties
        dir;
        dbName;
        funGenDbCls;
        gtid2gtname;
        gtid2gtdesc;
        gtid2diids;
        iid2gt;
        iid2setid;
        iid2ifpath;
        idx2iid;
    end
    methods
        function this = DbCls( setting )
            this.dir = fullfile( setting.dir, setting.dbName );
            this.dbName = setting.dbName;
            this.funGenDbCls = setting.funGenDbCls;
        end
        function [  gtid2gtname, ...
                    gtid2gtdesc, ...
                    gtid2diids, ...
                    iid2gt, ...
                    iid2setid, ...
                    iid2ifpath ] ...
                = genDbCls( this )
            fpath = this.getPath;
            try
                fprintf( '%s: Try to load DB.\n', ...
                    upper( mfilename ) );
                src = load( fpath );
                gtid2gtname = src.gtid2gtname;
                gtid2gtdesc = src.gtid2gtdesc;
                gtid2diids  = src.gtid2diids;
                iid2gt      = src.iid2gt;
                iid2setid   = src.iid2setid;
                iid2ifpath  = src.iid2ifpath;
                fprintf( '%s: DB loaded.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen DB.\n', ...
                    upper( mfilename ) );
                [   gtid2gtname, ...
                    gtid2gtdesc, ...
                    gtid2diids, ...
                    iid2gt, ...
                    iid2setid, ...
                    iid2ifpath ] = this.funGenDbCls(  );
                fprintf( '%s: Save DB.\n', ...
                    upper( mfilename ) );
                this.makeDir;
                save( fpath, ...
                    'gtid2gtname', ...
                    'gtid2gtdesc', ...
                    'gtid2diids', ...
                    'iid2gt', ...
                    'iid2setid', ...
                    'iid2ifpath' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
            this.gtid2gtname    = gtid2gtname;
            this.gtid2gtdesc    = gtid2gtdesc;
            this.gtid2diids     = gtid2diids;
            this.iid2gt         = iid2gt;
            this.iid2setid      = iid2setid;
            this.iid2ifpath     = iid2ifpath;
            this.idx2iid        = 1 : numel( iid2ifpath );
        end
        function numClass = getNumClass( this )
            numClass = length( this.gtid2gtname );
        end
        function numIm = getNumIm( this )
            numIm = length( this.iid2ifpath );
        end
        function gtDim = getGtDim( this )
            gtDim = this.getNumClass;
        end
        function numIm = getNumTrIm( this )
            numIm = length( this.getTriids );
        end
        function numIm = getNumTeIm( this )
            numIm = length( this.getTeiids );
        end
        function iids = getTriids( this )
            iids = find( this.iid2setid == 1 );
        end
        function iids = getTeiids( this )
            iids = find( this.iid2setid == 2 );
        end
        function ifpaths = getTrIfpaths( this )
            ifpaths = this.iid2ifpath( this.getTriids );
        end
        function ifpaths = getTeIfpaths( this )
            ifpaths = this.iid2ifpath( this.getTeiids );
        end
        function trGts = getTrGts( this )
            trGts = this.iid2gt( this.getTriids );
        end
        function teGts = getTeGts( this )
            teGts = this.iid2gt( this.getTeiids );
        end
        function gtType = getGtType( this )
            if length( this.gtid2gtname ) > 2
                gtType = 'MCLS';
            else
                gtType = 'BCLS';
            end
        end
        function trdb = getTrDb( this )
            setting.dir         = '';
            setting.dbName      = this.dbName;
            setting.funGenDbCls = [  ];
            trdb                = DbCls( setting );
            trdb.gtid2gtname    = this.gtid2gtname;
            trdb.gtid2gtdesc    = this.gtid2gtdesc;
            trdb.iid2gt         = this.getTrGts;
            trdb.iid2setid      = ones( size( trdb.iid2gt ) );
            trdb.iid2ifpath     = this.getTrIfpaths;
            trdb.idx2iid        = 1 : numel( trdb.iid2ifpath );
        end
        function tedb = getTeDb( this )
            setting.dir         = '';
            setting.dbName      = this.dbName;
            setting.funGenDbCls = [  ];
            tedb                = DbCls( setting );
            tedb.gtid2gtname    = this.gtid2gtname;
            tedb.gtid2gtdesc    = this.gtid2gtdesc;
            tedb.iid2gt         = this.getTeGts;
            tedb.iid2setid      = 2 * ones( size( tedb.iid2gt ) );
            tedb.iid2ifpath     = this.getTeIfpaths;
            tedb.idx2iid        = 1 : numel( tedb.iid2ifpath );
        end
        function gtid2idxs = gtid2idxs( this, idx2iids )
            idx2gt = this.iid2gt( idx2iids );
            idxthread = cellfun( @( x, y ) y * ones( size( x ) ), ...
                idx2gt, num2cell( 1 : numel( idx2gt ) )', ...
                'UniformOutput', false );
            idxthread = cat( 1, idxthread{ : } );
            gtthread = cat( 1, idx2gt{ : } );
            numGtid = this.getNumClass;
            gtid2idxs = cell( numGtid, 1 );
            for gtid = 1 : numGtid
                gtid2idxs{ gtid } = idxthread( gtthread == gtid );
            end
        end
        function gtid2didxs = gtid2didxs( this, idx2iids )
            gtid2didxs = cell( this.getNumClass, 1 );
            for gtid = 1 : this.getNumClass
                diids = this.gtid2diids{ gtid };
                didxs = arrayfun( ...
                    @( diid )find( idx2iids == diid ), ...
                    diids, ...
                    'UniformOutput', false );
                gtid2didxs{ gtid } = cat( 1, didxs{ : } );
            end
        end
        function ismulti = isMutiLabel( this )
            ismulti = any( cellfun( @length, this.iid2gt ) ~= 1 );
        end
        % Functions for object I/O.
        function name = getName( this )
            name = upper( mfilename );
        end
        function dir = getDir( this )
            dir = this.dir;
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getPath( this )
            fname = strcat( this.getName, '.mat' );
            path = fullfile( this.getDir, fname );
        end
    end
end