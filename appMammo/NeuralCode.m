classdef NeuralCode
    properties
        srcDb;
        srcImsvr;
        srcCnn;
        setting;
    end
    methods
        function this = NeuralCode( srcDb, srcImsvr, srcCnn, setting )
            this.srcDb                  = srcDb;
            this.srcImsvr               = srcImsvr;
            this.srcCnn                 = srcCnn;
            this.setting.layerId        = numel( srcCnn.layers );
            this.setting.reform         = 'XYA11';                    % RAW: No reform, XYA: Averaging xy domain. ZA: Averaging z domain.
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        % Off-the-shelf function for image description class.
        function ncodes = iid2desc( this, iid )
            im = imread( this.srcDb.iid2ifpath{ iid } );
            ncodes = this.descIms( im );
        end
        function ncodes = descIms( this, ims )
            if ~iscell( ims ), ims = { ims }; end;
            layerId     = this.setting.layerId;
            maxNumIm    = this.srcCnn.setting.batchSize;
            numIm       = numel( ims );
            numBatch    = ceil( numIm / maxNumIm );
            ncodes      = cell( numBatch, 1 );
            for b = 1 : numBatch
                idxFrom = ( b - 1 ) * maxNumIm + 1;
                idxTo = min( b * maxNumIm, numIm );
                bims = ims( idxFrom : idxTo );
                ncs = this.srcCnn.feedforward...
                    ( bims, this.srcImsvr );
                ncodes{ b } = ...
                    this.reformRawNeuralCodes( ncs, layerId );
                ncodes{ b } = mean( ncodes{ b }, 2 );
                clear ncs;
            end
            ncodes = cat( 2, ncodes{ : } );
        end
        function ncodes = descIms2( this, ims, batchSize )
            if ~iscell( ims ), ims = { ims }; end;
            layerId     = this.setting.layerId;
            maxNumIm    = batchSize;
            numIm       = numel( ims );
            numBatch    = ceil( numIm / maxNumIm );
            ncodes      = cell( numBatch, 1 );
            for b = 1 : numBatch
                idxFrom = ( b - 1 ) * maxNumIm + 1;
                idxTo = min( b * maxNumIm, numIm );
                bims = ims( idxFrom : idxTo );
                ncs = this.srcCnn.feedforward...
                    ( bims, this.srcImsvr );
                ncodes{ b } = ...
                    this.reformRawNeuralCodes( ncs, layerId );
                ncodes{ b } = mean( ncodes{ b }, 2 );
                clear ncs;
            end
            ncodes = cat( 2, ncodes{ : } );
        end
        % Functions for object identification.
        function name = getName( this )
            srcImsvrName = this.srcImsvr.getName;
            srcCnnName = this.srcCnn.getCnnName;
            name = sprintf( 'NC_%s_OF_%s_OF_%s', ...
                this.setting.changes, ...
                srcImsvrName, ...
                srcCnnName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function iid2desc = reformRawNeuralCodes( this, ncs, layerId )
            reform = this.setting.reform;
            if strcmp( reform, 'NONE' )
                iid2desc = this.noReform( ncs, layerId );
            elseif strcmp( reform, 'RAW' )
                    iid2desc = this.rawReform( ncs, layerId );
            elseif strcmp( reform( 1 : 3 ), 'XYA' )
                layout = reform( 4 : end );
                iid2desc = this.xyAvgReform( ncs, layerId, layout );
            elseif strcmp( reform( 1 : 3 ), 'ZA' )
                iid2desc = this.zAvgReform( ncs, layerId );
            end
        end
    end
    methods( Static )
        function iid2desc = noReform( ncs, layerId )
            iid2desc = cell( numel( layerId ), 1 );
            for lidx = 1 : numel( layerId )
                lid = layerId( lidx );
                descs = ncs( lid + 1 ).x;
                iid2desc{ lidx } = gather( descs );
            end
        end
        function iid2desc = rawReform( ncs, layerId )
            iid2desc = cell( numel( layerId ), 1 );
            for lidx = 1 : numel( layerId )
                lid = layerId( lidx );
                descs = ncs( lid + 1 ).x;
                [ r, c, z, n ] = size( descs );
                descs = gather( descs );
                iid2desc{ lidx } = ...
                    reshape( descs, [ r * c * z, n ] );
            end
            iid2desc = cat( 1, iid2desc{ : } );
        end
        function iid2desc = xyAvgReform( ncs, layerId, layout )
            iid2desc = cell( numel( layerId ), 1 );
            for lidx = 1 : numel( layerId )
                lid = layerId( lidx );
                descs = ncs( lid + 1 ).x;
                [ r, c, z, n ] = size( descs );
                if r ~= 1 && c ~= 1,
                    descs = mean( mean( descs, 1 ), 2 ); end;
                descs = gather( descs );
                iid2desc{ lidx } = ...
                    reshape( descs, [ z, n ] );
            end
            iid2desc = cat( 1, iid2desc{ : } );
        end
        function iid2desc = zAvgReform( ncs, layerId )
            iid2desc = cell( numel( layerId ), 1 );
            for lidx = 1 : numel( layerId )
                lid = layerId( lidx );
                descs = ncs( lid + 1 ).x;
                [ r, c, z, n ] = size( descs );
                if z ~= 1,
                    descs = mean( desc, 3 ); end;
                descs = gather( descs );
                iid2desc{ lidx } = ...
                    reshape( descs, [ r * c, n ] );
            end
            iid2desc = cat( 1, iid2desc{ : } );
        end
        function subreg = ndiv2subregs...
                ( nrowDiv, ncolDiv )
            m = nrowDiv;
            n = ncolDiv;
            [ x, y ] = meshgrid...
                ( linspace( 0, 1, n + 1 ), ...
                linspace( 0, 1, m + 1 ) );
            x1 = x( 1 : end - 1, 1 : end - 1 );
            y1 = y( 1 : end - 1, 1 : end - 1 );
            x2 = x( 2 : end, 2 : end );
            y2 = y( 2 : end, 2 : end );
            subreg = [ y1( : )'; x1( : )'; ...
                       y2( : )'; x2( : )'; ];
        end
        function isroi = coor2isroi...
                ( idx2rc, minR, minC, maxR, maxC )
            isroi = ...
                minR <= idx2rc( 1, : ) & ...
                idx2rc( 1, : ) <= maxR  & ...
                minC <= idx2rc( 2, : ) & ...
                idx2rc( 2, : ) <= maxC ;
        end
    end
end

