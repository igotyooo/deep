classdef Dic < handle
    properties
        srcDb;
        srcRegnDscrber;
        gmm;
        pca;
        setting;
    end
    methods
        function this = Dic( srcDb, srcRegnDscrber, setting )
            this.srcDb                          = srcDb;
            this.srcRegnDscrber                 = srcRegnDscrber;
            this.setting.numTargetScale         = Inf;
            this.setting.kernel                 = 'NONE';
            this.setting.norm                   = 'L2';
            this.setting.pcaDim                 = 128;
            this.setting.whitening              = false;
            this.setting.whiteningRegular       = 0;
            this.setting.normAfterPca           = 'L2';
            this.setting.k                      = 256;
            this.setting.numSamplePerGaussian   = 1000;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function trainDic( this )
            fpath = this.getPath;
            try
                data = load( fpath );
                this.gmm = data.gmm;
                this.pca = data.pca;
                fprintf( '%s: Dic is loaded.\n', ...
                    upper( mfilename ) );
            catch
                k                       = this.setting.k;
                pcaDim                  = this.setting.pcaDim;
                whitening               = this.setting.whitening;
                whiteningRegular        = this.setting.whiteningRegular;
                normAfterPca            = this.setting.normAfterPca;
                % Get descriptors.
                descs = this.sampleDescs;
                % Learn PCA and reduce dim.
                fprintf( '%s: Train PCA.\n', upper( mfilename ) );
                this.pca = [  ];
                if isfinite( pcaDim )
                    [ this.pca.proj, this.pca.center ] = this.learnPca...
                        ( pcaDim, descs, whitening, whiteningRegular );
                    descs = this.pca.proj * bsxfun( @minus, descs, this.pca.center );
                    descs = nmlzVecs( descs, normAfterPca );
                end
                % Learn GMM.
                fprintf( '%s: Train GMM.\n', upper( mfilename ) );
                [ this.gmm.means, this.gmm.covs, this.gmm.priors ] = ...
                    this.leanDicByGmm( descs, k );
                fprintf( '%s: Save dic.\n', upper( mfilename ) );
                gmm = this.gmm;
                pca = this.pca;
                save( fpath, 'gmm', 'pca' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end
        end
        function descs = sampleDescs( this )
            numTargetScale          = this.setting.numTargetScale;
            kernel                  = this.setting.kernel;
            norm                    = this.setting.norm;
            k                       = this.setting.k;
            numSamplePerGaussian    = this.setting.numSamplePerGaussian;
            numIm                   = min( 5000, this.srcDb.getNumTrIm );
            numDescPerIm            = ceil( k * numSamplePerGaussian / numIm );
            iids                    = randsample( this.srcDb.getTriids, numIm );
            descs = cell( 1, numIm );
            progress = 0; cummt = 0;
            for i = 1 : numIm; itime = tic; iid = iids( i );
                [ rid2geo, rid2desc, ~ ] = ...
                    this.srcRegnDscrber.iid2regdesc...
                    ( iid, numTargetScale, kernel, norm, [  ], 'NONE' );
                ridx = randsample( 1 : size( rid2geo, 2 ), ...
                    min( numDescPerIm, size( rid2geo, 2 ) ) );
                descs{ i } = rid2desc( :, ridx );
                progress = progress + 1;
                cummt = cummt + toc( itime );
                if ( progress / numIm ) >= 0.1; progress = 0;
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, i, ...
                        sprintf( 'Ext region descs to train GMM.' ), cummt );
                end
            end
            descs = cat( 2, descs{ : } );
        end
        % Functions for data I/O.
        function name = getName( this )
            name = sprintf( 'DIC_%s_OF_%s', ...
                this.setting.changes, ...
                this.srcRegnDscrber.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this )
            dir = this.srcDb.dir;
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this )
            name = this.getName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name ); 
                name = strcat( 'DIC_', name );
            end
            fname = strcat( name, '.mat' );
            fpath = fullfile...
                ( this.getDir, fname );
        end
    end
    methods( Static )
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

