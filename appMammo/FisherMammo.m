classdef FisherMammo < handle
    properties
        srcRegnDscrber;
        setting;
    end
    methods
        function this = FisherMammo...
                ( srcRegnDscrber, setting )
            this.srcRegnDscrber             = srcRegnDscrber;
            this.setting.normalizeByScale   = true;
            this.setting.spatialPyramid     = '11';
            this.setting.regionFiltering    = '';
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function fisher = iid2desc( this, iid )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.iid2regdesc( iid, false );
            if ~isempty( this.setting.regionFiltering )
                im = imread( this.srcDb.iid2impath{ iid } );
                rid2ok = this.setting.regionFiltering( im, rid2geo );
                rid2geo = rid2geo( :, rid2ok );
                rid2desc = rid2desc( :, rid2ok );
            end
            fisher = this.encodeSpFisher...
                ( rid2geo, rid2desc, imsize );
        end
        function fisher = im2desc( this, im )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.im2regdesc( im );
            if ~isempty( this.setting.regionFiltering )
                rid2ok = this.setting.regionFiltering( im, rid2geo );
                rid2geo = rid2geo( :, rid2ok );
                rid2desc = rid2desc( :, rid2ok );
            end
            fisher = this.encodeSpFisher...
                ( rid2geo, rid2desc, imsize );
        end
        function spFisher = encodeSpFisher...
                ( this, rid2geo, rid2desc, imsize )
            spatialPyramid = this.setting.spatialPyramid;
            numLevel = length( spatialPyramid ) / 2;
            layouts = reshape...
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
            srid2fisher = cell( numSubreg, 1 );
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
                fisher = this.descs2fisher...
                    ( roiDescs, roiGeos );
                srid2fisher{ srid } = fisher; 
            end
            spFisher = cat( 1, srid2fisher{ : } );
        end
        function fisher = descs2fisher...
                ( this, descs, geos )
            if this.setting.normalizeByScale
                scales = unique( geos( end, : ) )';
                sid2fisher = cell( size( scales ) );
                for sid = 1 : numel( scales )
                    roi = geos( end, : ) == scales( sid );
                    sfisher = vl_fisher( ...
                        descs( :, roi ), ...
                        this.srcRegnDscrber.gmm.means, ...
                        this.srcRegnDscrber.gmm.covs, ...
                        this.srcRegnDscrber.gmm.priors );
                    sfisher = nmlzVecs( sfisher, 'L2' );
                    sid2fisher{ sid } = sfisher;
                end
                fisher = mean( cat( 2, sid2fisher{ : } ), 2 );
            else
                fisher = vl_fisher( ...
                    descs, ...
                    this.srcRegnDscrber.gmm.means, ...
                    this.srcRegnDscrber.gmm.covs, ...
                    this.srcRegnDscrber.gmm.priors );
            end
            fisher = kernelMap( fisher, 'HELL' );
            fisher = nmlzVecs( fisher, 'L2' );
        end
        function [ rid2geo, rid2fisher, imsize ] = iid2descNoAp( this, iid )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.iid2regdesc( iid, false );
            if ~isempty( this.setting.regionFiltering )
                im = imread( this.srcDb.iid2impath{ iid } );
                rid2ok = this.setting.regionFiltering( im, rid2geo );
                rid2geo = rid2geo( :, rid2ok );
                rid2desc = rid2desc( :, rid2ok );
            end
            numRegn = size( rid2geo, 2 );
            means = this.srcRegnDscrber.gmm.means;
            covs = this.srcRegnDscrber.gmm.covs;
            priors = this.srcRegnDscrber.gmm.priors;
            fisherDim = 2 * size( rid2desc, 1 ) * size( means, 2 );
            rid2fisher = vl_fisher...
                ( rid2desc, means, covs, priors, 'NoAveragePooling' );
            rid2fisher = reshape( rid2fisher, [ fisherDim, numRegn ] );
            rid2fisher = kernelMap( rid2fisher, 'HELL' );
            rid2fisher = nmlzVecs( rid2fisher, 'L2' );
        end
        function [ rid2geo, rid2fisher, imsize ] = im2descNoAp( this, im )
            [ rid2geo, rid2desc, imsize ] = ...
                this.srcRegnDscrber.im2regdesc( im, false );
            if ~isempty( this.setting.regionFiltering )
                rid2ok = this.setting.regionFiltering( im, rid2geo );
                rid2geo = rid2geo( :, rid2ok );
                rid2desc = rid2desc( :, rid2ok );
            end
            numRegn = size( rid2geo, 2 );
            means = this.srcRegnDscrber.gmm.means;
            covs = this.srcRegnDscrber.gmm.covs;
            priors = this.srcRegnDscrber.gmm.priors;
            fisherDim = 2 * size( rid2desc, 1 ) * size( means, 2 );
            rid2fisher = vl_fisher...
                ( rid2desc, means, covs, priors, 'NoAveragePooling' );
            rid2fisher = reshape( rid2fisher, [ fisherDim, numRegn ] );
            rid2fisher = kernelMap( rid2fisher, 'HELL' );
            rid2fisher = nmlzVecs( rid2fisher, 'L2' );
        end
        % Functions for object identification.
        function name = getName( this )
            name = sprintf( 'FVM_%s_OF_%s', ...
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

