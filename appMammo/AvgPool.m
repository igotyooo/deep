classdef AvgPool < handle
    properties
        srcRegnDscrber;
        setting;
    end
    methods
        function this = AvgPool...
                ( srcRegnDscrber, setting )
            this.srcRegnDscrber                     = srcRegnDscrber;
            this.setting.normalizeByScale           = true;
            this.setting.spatialPyramid             = '11';
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function avgPool = iid2desc( this, iid )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.iid2regdesc( iid, false );
            avgPool = this.encodeSpAvgPool...
                ( rid2geo, rid2desc, imsize );
        end
        function avgPool = im2desc( this, im )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.im2regdesc( im );
            avgPool = this.encodeSpAvgPool...
                ( rid2geo, rid2desc, imsize );
        end
        function spAvgPool = encodeSpAvgPool...
                ( this, rid2geo, rid2desc, imsize )
            spatialPyramid              = this.setting.spatialPyramid;
            numLevel                    = length( spatialPyramid ) / 2;
            layouts                     = reshape...
                ( spatialPyramid, [ 2, numLevel ] );
            srid2level = cell( numLevel, 1 );
            subreg = cell( numLevel, 1 );
            for l = 1 : numLevel;
                layout = layouts( :, l );
                nr = str2double( layout( 1 ) );
                nc = str2double( layout( 2 ) );
                srid2level{ l } = cat...
                    ( 1, l * ones( nr * nc, 1 ) );
                subreg{ l } = this.ndiv2subregs( nr, nc );
            end; 
            subreg = cat( 2, subreg{ : } );
            numSubreg = size( subreg, 2 );
            srid2avgPool = cell( numSubreg, 1 );
            for srid = 1 : numSubreg
                minR = subreg( 1, srid ) * ( imsize( 1 ) - 1 ) + 1;
                minC = subreg( 2, srid ) * ( imsize( 2 ) - 1 ) + 1;
                maxR = subreg( 3, srid ) * ( imsize( 1 ) - 1 ) + 1;
                maxC = subreg( 4, srid ) * ( imsize( 2 ) - 1 ) + 1;
                rid2center = ...
                    ( rid2geo( 1 : 2, : ) + rid2geo( 3 : 4, : ) ) / 2;
                rid2isroi = this.coor2isroi...
                    ( rid2center, minR, minC, maxR, maxC ); 
                roiGeos = rid2geo( :, rid2isroi );
                roiDescs = rid2desc( :, rid2isroi );
                avgPool = this.descs2avgPool...
                    ( roiDescs, roiGeos );
                srid2avgPool{ srid } = avgPool; 
            end
            spAvgPool = cat( 1, srid2avgPool{ : } );
        end
        function avgPool = descs2avgPool...
                ( this, descs, geos )
            if this.setting.normalizeByScale
                scales = unique( geos( end, : ) )';
                sid2avgPool = cell( size( scales ) );
                for sid = 1 : numel( scales )
                    roi = geos( end, : ) == scales( sid );
                    savgPool = mean( descs( :, roi ), 2 );
                    savgPool = nmlzVecs( savgPool, 'L2' );
                    sid2avgPool{ sid } = savgPool;
                end
                avgPool = mean( cat( 2, sid2avgPool{ : } ), 2 );
            else
                avgPool = mean( descs, 2 );
            end
        end
        % Functions for object identification.
        function name = getName( this )
            name = sprintf( 'AP_%s_OF_%s', ...
                this.setting.changes, ...
                this.srcRegnDscrber.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    methods( Static )
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

