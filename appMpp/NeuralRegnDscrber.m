classdef NeuralRegnDscrber < handle
    properties
        db;
        net;
        gmm;
        pca;
        patchSide;
        stride;
        settingDesc;
        settingDic;
    end
    methods
        function this = NeuralRegnDscrber...
                ( db, net, settingDesc, settingDic )
            this.db                             = db;
            this.net                            = net;
            this.settingDesc.layerId            = numel( net.layers ) - 2;              % FC7 in AlexNet.
            this.settingDesc.scalingCriteria    = 'MIN';                                % MIN, MAX, WIDTH, HEIGHT, AREA.
            this.settingDesc.scaleId2numPixel   = round( 227 * 2 .^ ( 0 : 0.5 : 3 ) );
            this.settingDesc.maximumImageSize   = 15e6;
            this.settingDesc.pcaDim             = 128;
            this.settingDesc.kernelBeforePca    = 'NONE';
            this.settingDesc.normBeforePca      = 'L2';
            this.settingDesc.normAfterPca       = 'L2';
            this.settingDesc.regionFiltering    = '';
            this.settingDic.numTargetScale      = Inf;
            this.settingDic.numGaussian         = 256;
            this.settingDesc = setChanges...
                ( this.settingDesc, settingDesc, upper( mfilename ) );
            this.settingDic = setChanges...
                ( this.settingDic, settingDic, upper( mfilename ) );
        end
        function init( this, gpu )
            if gpu,
                this.net = vl_simplenn_move...
                    ( this.net, 'gpu' );
            end;
            idx = strfind( this.net.layers{end}.type, 'loss' );
            if ~isempty( idx ),
                this.net.layers{ end }.type( idx : end ) = [  ];
            end;
            [ this.patchSide, this.stride ] = ...
                getNetProperties( this.net, this.settingDesc.layerId );
        end
        function trainDic( this )
            fpath = this.getDicPath;
            try
                data = load( fpath );
                this.gmm = data.gmm;
                this.pca = data.pca;
                fprintf( '%s: Dic loaded.\n', ...
                    upper( mfilename ) );
            catch
                kernelBeforePca = this.settingDesc.kernelBeforePca;
                normBeforePca = this.settingDesc.normBeforePca;
                pcaDim = this.settingDesc.pcaDim;
                normAfterPca = this.settingDesc.normAfterPca;
                numGaussian = this.settingDic.numGaussian;
                this.pca = [  ];
                this.gmm = [  ];
                if isfinite( pcaDim ) || numGaussian,
                    % Get descriptors.
                    descs = this.sampleDescs;
                    descs = kernelMap...
                        ( descs, kernelBeforePca );
                    descs = nmlzVecs...
                        ( descs, normBeforePca );
                    % Learn PCA and reduce dim.
                    if isfinite( pcaDim ),
                        fprintf( '%s: Train PCA.\n', upper( mfilename ) );
                        [ this.pca.proj, this.pca.center ] = ...
                            this.learnPca...
                            ( pcaDim, descs, false, 0 );
                        descs = this.pca.proj * ...
                            bsxfun( @minus, descs, this.pca.center );
                    end
                    descs = nmlzVecs( descs, normAfterPca );
                    % Learn GMM.
                    if numGaussian,
                        fprintf( '%s: Train GMM.\n', upper( mfilename ) );
                        [ this.gmm.means, this.gmm.covs, this.gmm.priors ] = ...
                            this.leanDicByGmm( descs, numGaussian );
                    end
                end                
                fprintf( '%s: Save dic.\n', upper( mfilename ) );
                gmm = this.gmm;
                pca = this.pca;
                save( fpath, 'gmm', 'pca' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end
        end
        function descDb( this )
            % Check if descs exist.
            fprintf( '%s: Check if descs exist.\n', ...
                upper( mfilename ) );
            iid2vpath = cellfun( ...
                @( iid )this.getDescPath( iid ), ...
                num2cell( 1 : this.db.getNumIm )', ...
                'UniformOutput', false );
            iid2exist = cellfun( ...
                @( path )exist( path, 'file' ), ...
                iid2vpath );
            this.makeDescDir;
            iids = find( ~iid2exist );
            if isempty( iids ),
                fprintf( '%s: No im to desc.\n', ...
                    upper( mfilename ) ); return;
            end
            % Do the job.
            cnt = 0; cummt = 0; numIm = numel( iids );
            for iid = iids'; itime = tic;
                this.iid2regdesc( iid, true );
                cummt = cummt + toc( itime );
                cnt = cnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, cnt, ...
                    'Desc im regns.', cummt );
            end
        end
        function [ rid2geo, rid2desc, imsize ] = ...
                iid2regdesc( this, iid, saveData )
            kernelBeforePca = this.settingDesc.kernelBeforePca;
            normBeforePca = this.settingDesc.normBeforePca;
            normAfterPca = this.settingDesc.normAfterPca;
            fpath = this.getDescPath( iid );
            try
                data = load( fpath );
                rid2desc = data.rid2desc;
                rid2geo = data.rid2geo;
                imsize = data.imsize;
            catch
                [ rid2geo, rid2desc, imsize ] = ...
                    this.iid2rawregdesc( iid );
                rid2desc = kernelMap...
                    ( rid2desc, kernelBeforePca );
                rid2desc = nmlzVecs...
                    ( rid2desc, normBeforePca );
                if ~isempty( this.pca )
                    rid2desc = this.pca.proj * ...
                        bsxfun( @minus, rid2desc, this.pca.center );
                end
                if saveData,
                    save( fpath, 'rid2geo', 'rid2desc', 'imsize' ); end;
            end
            rid2desc = nmlzVecs...
                ( rid2desc, normAfterPca );
        end
        function [ rid2geo, rid2desc, imsize ] = ...
                im2regdesc( this, im )
            kernelBeforePca = this.settingDesc.kernelBeforePca;
            normBeforePca = this.settingDesc.normBeforePca;
            normAfterPca = this.settingDesc.normAfterPca;
            [ rid2geo, rid2desc, imsize ] = ...
                this.im2rawregdesc( im );
            rid2desc = kernelMap...
                ( rid2desc, kernelBeforePca );
            rid2desc = nmlzVecs...
                ( rid2desc, normBeforePca );
            if ~isempty( this.pca )
                rid2desc = this.pca.proj * ...
                    bsxfun( @minus, rid2desc, this.pca.center );
            end
            rid2desc = nmlzVecs...
                ( rid2desc, normAfterPca );
        end
        function [ rid2geo, rid2desc, imsize ] = ...
                iid2rawregdesc( this, iid )
            im = imread...
                ( this.db.iid2impath{ iid } );
            [ rid2geo, rid2desc, imsize ] = ...
                this.im2rawregdesc( im );
        end
        function [ rid2tlbr, rid2desc, imSize ] = ...
                im2rawregdesc( this, im )
            lyid = this.settingDesc.layerId;
            criteria = this.settingDesc.scalingCriteria;
            sid2npix = this.settingDesc.scaleId2numPixel;
            maximumImageSize = this.settingDesc.maximumImageSize;
            keepAspect = this.net.normalization.keepAspect;
            regionFiltering = this.settingDesc.regionFiltering;
            dilate = 0;
            % Do the job.
            imSize = size( im );
            imSize = imSize( 1 : 2 )';
            if keepAspect,
                sid2size = scaleImage( sid2npix, criteria, imSize );
            else
                sid2size = [ sid2npix; sid2npix; ];
            end;
            rid2desc = extractDenseActivations( ...
                im, ...
                this.net, ...
                lyid, ...
                sid2size, ...
                this.patchSide, ...
                dilate, ...
                maximumImageSize );
            rid2tlbr = extractDenseRegions( ...
                imSize, ...
                sid2size, ...
                this.patchSide, ...
                this.stride, ...
                dilate, ...
                maximumImageSize );
            if size( rid2desc, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
            % Filtering if needed.
            if ~isempty( regionFiltering )
                regionFiltering = str2func( regionFiltering );
                rid2ok = regionFiltering( im, rid2tlbr );
                rid2tlbr = rid2tlbr( :, rid2ok );
                rid2desc = rid2desc( :, rid2ok );
            end;
        end
        function descs = sampleDescs( this )
            fpath = this.getSmplDescPath;
            numGaussian = this.settingDic.numGaussian;
            try
                fprintf( '%s: Try to load training regn descs.\n', ...
                    upper( mfilename ) );
                data = load( fpath );
                descs = data.descs;
                fprintf( '%s: Loaded.\n', ...
                    upper( mfilename ) );
            catch
                this.makeSmplDescDir;
                maxNumSrcIm = 5000;
                numSamplePerGaussian = 1000;
                numIm = min( maxNumSrcIm, this.db.getNumTrIm );
                numDescPerIm = ceil( numGaussian * numSamplePerGaussian / numIm );
                iids = randsample( this.db.getTriids, numIm );
                descs = cell( numIm, 1 );
                cummt = 0;
                for i = 1 : numIm; itime = tic; iid = iids( i );
                    [ ~, rid2desc, ~ ] = ...
                        this.iid2rawregdesc( iid );
                        numDesc = size( rid2desc, 2 );
                        ridx = randsample( 1 : numDesc, ...
                            min( numDescPerIm, numDesc ) );
                        descs{ i } = rid2desc( :, ridx );
                    cummt = cummt + toc( itime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, i, ...
                        sprintf( 'Sample training regn descs.' ), cummt );
                end
                descs = cat( 2, descs{ : } );
                fprintf( '%s: Save regn descs.\n', ...
                    upper( mfilename ) );
                save( fpath, 'descs', '-v7.3' );
                fprintf( '%s: Done.\n', ...
                    upper( mfilename ) );
            end
        end
        % Functions for sample descriptor I/O.
        function name = getSmplDescName( this )
            regionFiltering = this.settingDesc.regionFiltering;
            scalingCriteria = this.settingDesc.scalingCriteria;
            scaleId2numPixel = this.settingDesc.scaleId2numPixel;
            if isempty( regionFiltering )
                name = sprintf( 'NRDMSMPL_LI%d_SC%s_NP%s_OF_%s', ...
                    this.settingDesc.layerId, ...
                    scalingCriteria, ...
                    mat2str( scaleId2numPixel ), ...
                    this.net.name );
            else
                name = sprintf( 'NRDMSMPL_LI%d_SC%s_NP%s_RF%s_OF_%s', ...
                    this.settingDesc.layerId, ...
                    scalingCriteria, ...
                    mat2str( scaleId2numPixel ), ...
                    upper( regionFiltering ), ...
                    this.net.name );
            end
            name( strfind( name, '__' ) ) = '';
        end
        function dir = getSmplDescDir( this )
            dir = this.db.dstDir;
        end
        function dir = makeSmplDescDir( this )
            dir = this.getSmplDescDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getSmplDescPath( this )
            name = this.getSmplDescName;
            fname = strcat( name, '.mat' );
            fpath = fullfile...
                ( this.getSmplDescDir, fname );
        end
        % Functions for dictionary I/O.
        function name = getName( this )
            name = sprintf( 'DIC_%s_OF_%s', ...
                this.settingDic.changes, ...
                this.getDescName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDicDir( this )
            dir = this.db.dstDir;
        end
        function dir = makeDicDir( this )
            dir = this.getDicDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDicPath( this )
            name = this.getName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name ); 
                name = strcat( 'DIC_', name );
            end
            fname = strcat( name, '.mat' );
            fpath = fullfile...
                ( this.getDicDir, fname );
        end
        % Functions for descriptor I/O.
        function name = getDescName( this )
            name = sprintf( 'NRDM_%s_OF_%s', ...
                this.settingDesc.changes, ...
                this.net.name );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDescDir( this )
            name = this.getDescName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'NRDM_', name );
            end
            dir = fullfile...
                ( this.db.dstDir, name );
        end
        function dir = makeDescDir( this )
            dir = this.getDescDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDescPath( this, iid )
            fname = sprintf...
                ( 'ID%06d.mat', iid );
            fpath = fullfile...
                ( this.getDescDir, fname );
        end
    end
    methods( Static )
        function [ rid2geo, rid2desc ] = filterByScale...
                ( rid2geo, rid2desc, numTargetScale )
            if isfinite( numTargetScale ),
                rid2scale = rid2geo( end, : );
                scales = unique( rid2scale );
                scaleThresh = scales( numel( scales ) - numTargetScale + 1 );
                rid2roi = rid2scale >= scaleThresh;
                rid2geo = rid2geo( :, rid2roi );
                rid2desc = rid2desc( :, rid2roi );
            end
        end
        function [ proj, center ] = learnPca...
                ( dim, vecs, whitening, whiteningRegular )
            center = mean( vecs, 2 );
            x = bsxfun( @minus, vecs, center );
            X = x * x' / size( x, 2 );
            [ V, D ] = eig( X );
            d = diag( D );
            [ d, perm ] = sort( d, 'descend' );
            m = min( dim, size( vecs, 1 ) );
            V = V( :, perm );
            if whitening
                d = d + whiteningRegular * max( d );
                proj = diag( 1 ./ sqrt( d( 1 : m ) ) ) * V( :, 1 : m )';
            else
                proj = V( :, 1 : m )';
            end
        end
        function [ means, covs, priors ] = leanDicByGmm( vecs, k )
            v = var( vecs, [  ], 2 );
            gmmSetting = { ...
                'verbose', ...
                'Initialization', 'KMEANS', ...
                'NumRepetitions', 1, ...
                'CovarianceBound', double( max( v ) * 0.0001 ) };
            [ means, covs, priors ] = ...
                vl_gmm( vecs, k, gmmSetting{ : } );
        end
    end
end