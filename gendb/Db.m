classdef Db < handle
    properties
        dstDir;
        name;
        funh;
        cid2name;
        cid2diids;
        iid2impath;
        iid2size;
        iid2cids;
        iid2oids;
        iid2setid;
        oid2cid;
        oid2diff;
        oid2iid;
        oid2bbox;
    end
    methods
        function this = Db( setting, dstDir )
            this.dstDir = fullfile( dstDir, setting.name );
            this.name = setting.name;
            this.funh = setting.funh;
        end
        function genDb( this )
            fpath = this.getPath;
            try
                fprintf( '%s: Try to load db.\n', ...
                    upper( mfilename ) );
                data = load( fpath );
                db = data.db;
                fprintf( '%s: db loaded.\n', ...
                    upper( mfilename ) );
            catch
                fprintf( '%s: Gen db.\n', ...
                    upper( mfilename ) );
                [   db.cid2name, ...
                    db.iid2impath, ...
                    db.iid2size, ...
                    db.iid2setid, ...
                    db.oid2cid, ...
                    db.oid2diff, ...
                    db.oid2iid, ...
                    db.oid2bbox ] = this.funh(  );
                oid2out = 1 - db.oid2bbox( 1, : ) > 0;
                db.oid2bbox( 1, oid2out ) = 1;
                oid2out = 1 - db.oid2bbox( 2, : ) > 0;
                db.oid2bbox( 2, oid2out ) = 1;
                oid2out = db.iid2size( 1, db.oid2iid ) - db.oid2bbox( 3, : ) < 0;
                db.oid2bbox( 3, oid2out ) = db.iid2size( 1, db.oid2iid( oid2out ) );
                oid2out = db.iid2size( 2, db.oid2iid ) - db.oid2bbox( 4, : ) < 0;
                db.oid2bbox( 4, oid2out ) = db.iid2size( 2, db.oid2iid( oid2out ) );
                db.iid2oids = arrayfun( ...
                    @( iid )find( db.oid2iid == iid ), ...
                    1 : numel( db.iid2setid ), ...
                    'UniformOutput', false )';
                db.iid2cids = cellfun( ...
                    @( oids )unique( db.oid2cid( oids ) ), ...
                    db.iid2oids, ...
                    'UniformOutput', false )';
                db.cid2diids = arrayfun( ...
                    @( cid )setdiff( db.oid2iid( db.oid2cid == cid ), db.oid2iid( ~db.oid2diff ) ), ...
                    1 : numel( db.cid2name ), ...
                    'UniformOutput', false )';
                fprintf( '%s: Save DB.\n', ...
                    upper( mfilename ) );
                this.makeDir;
                save( fpath, 'db' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
            this.cid2name   = db.cid2name;
            this.cid2diids  = db.cid2diids;
            this.iid2impath = db.iid2impath;
            this.iid2size   = single( db.iid2size );
            this.iid2cids   = db.iid2cids;
            this.iid2oids   = db.iid2oids;
            this.iid2setid  = db.iid2setid;
            this.oid2cid    = db.oid2cid;
            this.oid2diff   = db.oid2diff;
            this.oid2iid    = db.oid2iid;
            this.oid2bbox   = single( round( db.oid2bbox ) );
        end
        function numClass = getNumClass( this )
            numClass = length( this.cid2name );
        end
        function numIm = getNumIm( this )
            numIm = length( this.iid2impath );
        end
        function numIm = getNumTrIm( this )
            numIm = sum( this.iid2setid == 1 );
        end
        function numIm = getNumTeIm( this )
            numIm = sum( this.iid2setid == 2 );
        end
        function iids = getTriids( this )
            iids = find( this.iid2setid == 1 );
        end
        function iids = getTeiids( this )
            iids = find( this.iid2setid == 2 );
        end
        function ifpaths = getTrImpaths( this )
            ifpaths = this.iid2impath( this.getTriids );
        end
        function ifpaths = getTeImpaths( this )
            ifpaths = this.iid2impath( this.getTeiids );
        end
        function trImCids = getTrImCids( this )
            trImCids = this.iid2cids( this.getTriids );
        end
        function teImCids = getTeImCids( this )
            teImCids = this.iid2cids( this.getTeiids );
        end
        function ismulti = isMutiLabel( this )
            ismulti = any( cellfun( @length, this.iid2cids ) ~= 1 );
        end
        function newdb = reduceDbByCls( this, class )
            if isnumeric( class ), 
                cid = class; 
                cname = this.cid2name{ cid };
            else
                cname = class;
                cid = cellfun( ...
                    @( name )strcmp( cname, name ), ...
                    this.cid2name );
                cid = find( cid );
            end
            iids = this.oid2iid( this.oid2cid == cid );
            iids = unique( iids );
            newdb = this.reduceDb...
                ( iids, cid, strcat( this.name, cname ) );
        end
        function newdb = reduceDb( this, iids, cids, newName )
            setting.dir = fileparts( this.dstDir );
            setting.name = upper( newName );
            setting.funh = this.funh;
            newdb = Db( setting );
            % Reordering.
            numim = numel( this.iid2impath );
            newiid2iid = iids;
            iid2newiid = zeros( numim, 1 );
            iid2newiid( newiid2iid ) = 1 : numel( newiid2iid );
            numclass = numel( this.cid2name );
            newcid2cid = cids;
            cid2newcid = zeros( numclass, 1 );
            cid2newcid( newcid2cid ) = 1 : numel( newcid2cid );
            numobj = numel( this.oid2iid );
            oid2ok = ismember( this.oid2iid, newiid2iid );
            oid2ok = oid2ok & ismember( this.oid2cid, newcid2cid );
            newoid2oid = find( oid2ok );
            oid2newoid = zeros( numobj, 1 );
            oid2newoid( oid2ok ) = 1 : numel( newoid2oid );
            % Reform db.
            newdb.cid2name = this.cid2name( newcid2cid );
            newdb.cid2diids = this.cid2diids( newcid2cid );
            newdb.oid2cid = cid2newcid( this.oid2cid( oid2ok ) );
            newdb.oid2iid = iid2newiid( this.oid2iid( oid2ok ) );
            newdb.oid2diff = this.oid2diff( oid2ok );
            newdb.oid2bbox = this.oid2bbox( :, oid2ok );
            newdb.iid2cids = this.iid2cids( newiid2iid );
            newdb.iid2cids = cellfun( ...
                @( cid )cid2newcid( cid( cid2newcid( cid ) ~= 0 ) ), ...
                newdb.iid2cids, 'UniformOutput', false );
            newdb.iid2impath = this.iid2impath( newiid2iid );
            newdb.iid2oids = this.iid2oids( newiid2iid );
            newdb.iid2oids = cellfun( ...
                @( oid )oid2newoid( oid( oid2newoid( oid ) ~= 0 ) ), ...
                newdb.iid2oids, 'UniformOutput', false );
            newdb.iid2setid = this.iid2setid( newiid2iid );
            newdb.iid2size = this.iid2size( :, newiid2iid );
            this.makeDir;
            fprintf( '%s: reduced to %s\n', upper( mfilename ), upper( newName ) );
        end
        % Functions for object identification.
        function name = getName( this )
            name = upper( mfilename );
        end
        function dir = getDir( this )
            dir = this.dstDir;
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