classdef PropObj < handle
    properties
        db;
        propNet;
        stride;
        patchSide;
        scales;
        aspects;
        setting;
    end
    methods( Access = public )
        function this = PropObj( db, propNet, setting )
            this.db = db;
            this.propNet = propNet;
            this.setting.numScale               = 6;
            this.setting.numAspect              = 6;
            this.setting.confidence             = 0.90;
            this.setting.dilate                 = 1 / 4;
            this.setting.posIntOverRegnMoreThan = 1 / 3;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, gpus )
            % Set parameters.
            numScale = this.setting.numScale;
            numAspect = this.setting.numAspect;
            confidence = this.setting.confidence;
            posIntOverRegnMoreThan = this.setting.posIntOverRegnMoreThan;
            % Fetch net on GPU.
            this.propNet = Net.fetchNetOnGpu( this.propNet, gpus );
            % Determine stride and patch side.
            [ this.patchSide, this.stride ] = ...
                getNetProperties( this.propNet, numel( this.propNet.layers ) - 1 );
            % Determine multiple scales and aspect rates.
            trboxes = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
            referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
            this.scales = determineScales...
                ( trboxes, referenceSide, numScale, confidence );
            this.aspects = determineAspectRates...
                ( trboxes, numAspect, confidence );
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
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2prop0( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, ...
                    'Prop obj.', cummt );
            end;
        end
        function [ rid2tlbr, rid2out ] = ...
                iid2prop( this, iid )
            % Initial guess.
            fpath = this.getPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.det.rid2tlbr;
                rid2out = data.det.rid2out;
            catch
                im = imread( this.db.iid2impath{ iid } );
                [ rid2tlbr, rid2out ] = ...
                    this.im2prop0( im );
                prop.rid2tlbr = rid2tlbr;
                prop.rid2out = rid2out;
                save( fpath, 'prop' );
            end;
            % % Compute each region score here.
            % % Scale/aspect selection.
            % numScale = this.setting.numScale;
            % numAspect = this.setting.numAspect;
            % sids = this.settingInitMrg.selectScaleIds;
            % aids = this.settingInitMrg.selectAspectIds;
            % if numScale ~= numel( sids ),
            %     did2ok = ismember( did2det( 5, : ), sids );
            %     did2det = did2det( :, did2ok );
            %     did2score = did2score( :, did2ok );
            % end;
            % if numAspect ~= numel( aids ),
            %     did2ok = ismember( did2det( 6, : ), aids );
            %     did2det = did2det( :, did2ok );
            %     did2score = did2score( :, did2ok );
            % end;
            % % NMS.
            % overlap = this.settingInitMrg.overlap;
            % [ did2det, did2score, ~ ] = ...
            %     nms( [ did2det; did2score ]', overlap );
            % did2det = did2det';
        end
        function [ rid2det, rid2out ] = im2prop0( this, im )
            % Pre-filtering by initial guess.
            fprintf( '%s: Initial guess.\n', upper( mfilename ) );
            [ rid2out, rid2tlbr, imGlobal, sideMargin ] = ...
                this.initGuess( im );
            % Convert to original image domain.
            imGlobalSize = [ size( imGlobal, 1 ); size( imGlobal, 2 ); ];
            bnd = [ sideMargin + 1; imGlobalSize - sideMargin; ];
            [ rid2det, ok ] = bndtlbr( rid2det, bnd );
            rid2out = rid2out( :, ok );
            rid2det( 1 : 4, : ) = bsxfun( @minus, rid2det( 1 : 4, : ), [ sideMargin; sideMargin; ] );
        end
        function [ rid2out, rid2tlbr, imGlobal, sideMargin ] = ...
                initGuess( this, im ) % This image is original.
            % Prepare settings and data.
            patchMargin = this.setting.patchMargin;
            numAspect = numel( this.aspects );
            numScale = numel( this.scales );
            interpolation = this.propNet.normalization.interpolation;
            lyid = numel( this.propNet.layers ) - 1;
            % Do the job.
            imSize = size( im ); imSize = imSize( 1 : 2 ); imSize = imSize( : );
            pside0 = min( imSize ) / patchMargin;
            cnt = 0;
            rid2tlbr = cell( numScale * numAspect, 1 );
            rid2out = cell( numScale * numAspect, 1 );
            for s = 1 : numScale,
                pside = pside0 / this.scales( s );
                for a = 1 : numAspect,
                    psider = pside;
                    psidec = pside / this.aspects( a );
                    mar2im = round( [ psider; psidec ] * patchMargin );
                    bnd = [ 1 - mar2im; imSize + mar2im ];
                    srcr = bnd( 3 ) - bnd( 1 ) + 1;
                    srcc = bnd( 4 ) - bnd( 2 ) + 1;
                    dstr = round( this.patchSize * srcr / psider );
                    dstc = round( this.patchSize * srcc / psidec );
                    if dstr * dstc > 2859 * 5448,
                        fprintf( '%s: Warning) Im of s%d a%d rejected.\n', ...
                            upper( mfilename ), s, a ); continue;
                    end;
                    im_ = cropAndNormalizeIm...
                        ( single( im ), imSize, bnd, this.rgbMean );
                    im_ = imresize( im_, [ dstr, dstc ], ...
                        'method', interpolation );
                    % Feed-foreward.
                    if isa( this.propNet.layers{ 1 }.weights{ 1 }, 'gpuArray' ), 
                        im_ = gpuArray( im_ ); end;
                    res = my_simplenn( ...
                        this.propNet, im_, [  ], [  ], ...
                        'accumulate', false, ...
                        'disableDropout', true, ...
                        'conserveMemory', true, ...
                        'backPropDepth', +inf, ...
                        'targetLayerId', lyid, ...
                        'sync', true ); clear im_;
                    % Form activations.
                    outs = gather( res( lyid + 1 ).x ); clear res;
                    [ nr, nc, z ] = size( outs );
                    outs = reshape( permute( outs, [ 3, 1, 2 ] ), z, nr * nc );
                    % Form geometries.
                    r = ( ( 1 : nr ) - 1 ) * this.stride + 1;
                    c = ( ( 1 : nc ) - 1 ) * this.stride + 1;
                    [ c, r ] = meshgrid( c, r );
                    regns = cat( 3, r, c );
                    regns = cat( 3, regns, regns + this.patchSize - 1 );
                    regns = reshape( permute( regns, [ 3, 1, 2 ] ), 4, nr * nc );
                    regns = cat( 1, regns, ...
                        s * ones( 1, nr * nc  ), ...
                        a * ones( 1, nr * nc  ) );
                    % Back projection.
                    regns = resizeTlbr( regns, [ dstr; dstc; ], [ srcr; srcc; ] );
                    regns( 1 : 4, : ) = round( bsxfun( @minus, regns( 1 : 4, : ), [ mar2im; mar2im; ] ) );
                    cnt = cnt + 1;
                    rid2tlbr{ cnt } = regns;
                    rid2out{ cnt } = outs;
                end;
            end;
            rid2tlbr = cat( 2, rid2tlbr{ : } );
            rid2out = cat( 2, rid2out{ : } );
            mar2im = 1 - min( rid2tlbr( 1 : 2, : ), [  ], 2 );
            rid2tlbr( 1 : 4, : ) = bsxfun( @plus, rid2tlbr( 1 : 4, : ), [ mar2im; mar2im; ] );
            imGlobal = cropAndNormalizeIm...
                ( single( im ), imSize, [ 1 - mar2im; imSize + mar2im ], this.rgbMean );
            sideMargin = mar2im;
        end
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROP_%s_OF_%s', ...
                this.setting.changes, ...
                this.propNet.getNetName );
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

