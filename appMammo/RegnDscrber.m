classdef RegnDscrber < handle
    properties
        srcDb;
        srcCutter;      % Can be DevRegion, DenseRegion, SelSearch, ...
        srcDscrber;     % Can be Fisher, NeuralCode, Sift, Gist, Hog, ...
    end
    methods
        function this = RegnDscrber...
                ( srcDb, srcCutter, srcDscrber )
            this.srcDb = srcDb;
            this.srcCutter = srcCutter;
            this.srcDscrber = srcDscrber;
        end
        function descDb( this )
            numDscrber = numel( this.srcDscrber );
            did2iid2exist = cell( numDscrber, 1 );
            for did = 1 : numDscrber
                fprintf( '%s: Check if descs exist.\n', ...
                    upper( mfilename ) );
                iid2vpath = cellfun( ...
                    @( iid )this.getPath( iid, did ), ...
                    num2cell( 1 : this.srcDb.getNumIm )', ...
                    'UniformOutput', false );
                did2iid2exist{ did } = cellfun( ...
                    @( path )exist( path, 'file' ), ...
                    iid2vpath );
                this.makeDir( did );
            end
            did2iid2exist = cat( 2, did2iid2exist{ : } );
            iid2exist = prod( did2iid2exist, 2 );
            iids = find( ~iid2exist );
            if isempty( iids ),
                fprintf( '%s: No im to desc regns.\n', ...
                    upper( mfilename ) ); return;
            end;
            cnt = 0; cummt = 0; numIm = numel( iids );
            for iid = iids'; itime = tic;
                this.iid2regdesc...
                    ( iid, Inf, 'NONE', 'NONE', [  ], 'NONE' );
                cummt = cummt + toc( itime ); 
                cnt = cnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, cnt, ...
                    'Desc im regns.', cummt );
            end
        end
        function [ rid2geo, rid2desc, imsize ] = ...
                iid2regdesc( this, ...
                iid, ...
                numTargetScale, ...
                kernelBeforePca, ...
                normBeforePca, ...
                pca, ...
                normAfterPca )
            numDscrber = numel( this.srcDscrber );
            did2rid2desc = cell( numDscrber, 1 );
            for did = 1 : numDscrber
                fpath = this.getPath( iid, did );
                try
                    data = load( fpath );
                    rid2geo  = data.rid2geo;
                    rid2desc = data.rid2desc;
                    imsize   = data.imsize;
                catch
                    im = imread...
                        ( this.srcDb.iid2ifpath{ iid } );
                    [ rid2im, rid2geo, imsize ] = ...
                        this.srcCutter.extImRegions( im );
                    rid2desc = ...
                        this.srcDscrber( did ).descIms( rid2im );
                    save( fpath, 'rid2geo', 'rid2desc', 'imsize' );
                end
                [ rid2geo, rid2desc ] = this.filterByScale...
                    ( rid2geo, rid2desc, numTargetScale );
                % Kernelize before dimensionality reduction.
                rid2desc = kernelMap( rid2desc, kernelBeforePca );
                % Normalization before dimensionality reduction.
                rid2desc = nmlzVecs( rid2desc, normBeforePca );
                did2rid2desc{ did } = rid2desc;
            end
            rid2desc = cat( 1, did2rid2desc{ : } );
            % Dimensionality reduction.
            if isstruct( pca ),
                rid2desc = pca.proj * bsxfun( @minus, rid2desc, pca.center ); end;
            % Normalization after dimensionality reduction.
            rid2desc = nmlzVecs( rid2desc, normAfterPca );
        end
        function [ rid2geo, rid2desc, imsize ] = ...
                im2regdesc( this, ...
                im, ...
                numTargetScale, ...
                kernelBeforePca, ...
                normBeforePca, ...
                pca, ...
                normAfterPca )
            numDscrber = numel( this.srcDscrber );
            did2rid2desc = cell( numDscrber, 1 );
            for did = 1 : numDscrber
                [ rid2im, rid2geo, imsize ] = ...
                    this.srcCutter.extImRegions( im );
                rid2desc = ...
                    this.srcDscrber( did ).descIms( rid2im );
                [ rid2geo, rid2desc ] = this.filterByScale...
                    ( rid2geo, rid2desc, numTargetScale );
                % Kernelize before dimensionality reduction.
                rid2desc = kernelMap( rid2desc, kernelBeforePca );
                % Normalization before dimensionality reduction.
                rid2desc = nmlzVecs( rid2desc, normBeforePca );
                did2rid2desc{ did } = rid2desc;
            end
            rid2desc = cat( 1, did2rid2desc{ : } );
            % Dimensionality reduction.
            if isstruct( pca ),
                rid2desc = pca.proj * bsxfun( @minus, rid2desc, pca.center ); end;
            % Normalization after dimensionality reduction.
            rid2desc = nmlzVecs( rid2desc, normAfterPca );
        end
        % Functions for data I/O.
        function name = getName( this, dscrberId )
            if nargin > 1
                name = sprintf( 'RD_OF_%s_OF_%s', ...
                    this.srcDscrber( dscrberId ).getName, ...
                    this.srcCutter.getName );
            else
                numDscrber = numel( this.srcDscrber );
                name = {  };
                name{ end + 1 } = 'RD_OF_';
                for did = 1 : numDscrber
                    if did > 1, name{ end + 1 } = '_AND_'; end;
                    name{ end + 1 } = ...
                        this.srcDscrber( did ).getName;
                end
                name{ end + 1 } = ...
                    strcat( '_OF_', this.srcCutter.getName );
                name = cat( 2, name{ : } );
            end
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this, dscrberId )
            name = this.getName( dscrberId );
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'RD_', name );
            end
            dir = fullfile...
                ( this.srcDb.dir, name );
        end
        function dir = makeDir( this, dscrberId )
            dir = this.getDir( dscrberId );
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this, iid, dscrberId )
            fname = sprintf...
                ( 'ID%06d.mat', iid );
            fpath = fullfile...
                ( this.getDir( dscrberId ), fname );
        end
    end
    methods( Static )
        function rid2desc = postProc( ...
                rid2desc, ...
                kernelBeforePca, ...
                normBeforePca, ...
                pca, ...
                normAfterPca )
            % Kernelize before dimensionality reduction.
            rid2desc = kernelMap( rid2desc, kernelBeforePca );
            % Normalization before dimensionality reduction.
            rid2desc = nmlzVecs( rid2desc, normBeforePca );
            % Dimensionality reduction.
            if isstruct( pca ),
                rid2desc = pca.proj * bsxfun( @minus, rid2desc, pca.center ); end;
            % Normalization after dimensionality reduction.
            rid2desc = nmlzVecs( rid2desc, normAfterPca );
        end
        function [ rid2geo, rid2desc ] = filterByScale...
                ( rid2geo, rid2desc, numTargetScale )
            if isfinite( numTargetScale ),
                rid2scale = rid2geo( end, : );
                scales = unique( rid2scale );
                scaleThresh = scales( numel( scales ) - numTargetScale );
                rid2roi = rid2scale > scaleThresh;
                rid2geo = rid2geo( :, rid2roi );
                rid2desc = rid2desc( :, rid2roi );
            end
        end
    end
end