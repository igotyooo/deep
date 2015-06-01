classdef DbReg < handle
    properties
        dir;
        dbName;
        funGenDbReg;
        iid2gt;
        iid2setid;
        iid2ifpath;
        idx2iid;
    end
    methods
        function this = DbReg( setting )
            this.dir = fullfile( setting.dir, setting.dbName );
            this.dbName = setting.dbName;
            this.funGenDbReg = setting.funGenDbReg;
        end
        function [  iid2gt, ...
                    iid2setid, ...
                    iid2ifpath ] ...
                = genDbReg( this )
            fpath = this.getPath;
            try
                fprintf( '%s: Try to load DB.\n', ...
                    upper( mfilename ) );
                src = load( fpath );
                iid2gt      = src.iid2gt;
                iid2setid   = src.iid2setid;
                iid2ifpath  = src.iid2ifpath;
                fprintf( '%s: DB loaded.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen DB.\n', ...
                    upper( mfilename ) );
                [   iid2gt, ...
                    iid2setid, ...
                    iid2ifpath ] = this.funGenDbReg(  );
                fprintf( '%s: Save DB.\n', ...
                    upper( mfilename ) );
                this.makeDir;
                save( fpath, ...
                    'iid2gt', ...
                    'iid2setid', ...
                    'iid2ifpath' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
            this.iid2gt         = iid2gt;
            this.iid2setid      = iid2setid;
            this.iid2ifpath     = iid2ifpath;
            this.idx2iid        = 1 : numel( iid2ifpath );
        end
        function numIm = getNumIm( this )
            numIm = length( this.iid2ifpath );
        end
        function gtDim = getGtDim( this )
            gtDim = numel( this.iid2gt{ 1 } );
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
            gtType = 'REG';
        end
        function trdb = getTrDb( this )
            setting.dir         = '';
            setting.dbName      = this.dbName;
            setting.funGenDbReg = [  ];
            trdb                = DbReg( setting );
            trdb.iid2gt         = this.getTrGts;
            trdb.iid2setid      = ones( size( trdb.iid2gt ) );
            trdb.iid2ifpath     = this.getTrIfpaths;
            trdb.idx2iid        = 1 : numel( trdb.iid2ifpath );
        end
        function tedb = getTeDb( this )
            setting.dir         = '';
            setting.dbName      = this.dbName;
            setting.funGenDbReg = [  ];
            tedb                = DbReg( setting );
            tedb.iid2gt         = this.getTeGts;
            tedb.iid2setid      = 2 * ones( size( tedb.iid2gt ) );
            tedb.iid2ifpath     = this.getTeIfpaths;
            tedb.idx2iid        = 1 : numel( tedb.iid2ifpath );
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