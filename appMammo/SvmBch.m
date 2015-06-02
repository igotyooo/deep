classdef SvmBch < handle
    properties
        srcDb;
        srcImDscrber;
        prll;
        setting;
    end    
    methods
        function this = SvmBch( srcDb, srcImDscrber, setting )
            this.srcDb                      = srcDb;
            this.srcImDscrber               = srcImDscrber;
            this.setting.kernel             = 'NONE';
            this.setting.norm               = 'L2';
            this.setting.c                  = 10;
            this.setting.trAug              = 'SA';
            this.setting.teAug              = 'SMP';
            this.setting.epsilon            = 1e-3;
            this.setting.biasMultiplier     = 1;
            this.setting.biasLearningRate   = 0.5;
            this.setting.loss               = 'HINGE';
            this.setting.solver             = 'SDCA';
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function trainSvm( this )
            fprintf( '%s: Check if svm exist.\n', ...
                upper( mfilename ) );
            cid2path = cellfun( ...
                @( cid )this.getSvmPath( cid ), ...
                num2cell( 1 : this.srcDb.getNumClass )', ...
                'UniformOutput', false );
            cid2exist = cellfun( ...
                @( path )exist( path, 'file' ), ...
                cid2path );
            cids = find( ~cid2exist );
            if isempty( cids ), 
                fprintf( '%s: No svm to train.\n', ...
                    upper( mfilename ) ); return; 
            end;
            idx2iid = this.srcDb.getTriids;
            idx2iid = idx2iid( randperm( numel( idx2iid ) )' );
            idx2desc = this.loadDbDescs( idx2iid );
            cid2idxs = this.srcDb.getCid2idxs( idx2iid );
            cid2didxs = this.srcDb.getCid2didxs( idx2iid );
            this.makeDir;
            cnt = 0; cummt = 0; numIm = numel( cids );
            for cid = cids'; itime = tic;
                this.cid2svm( cid, idx2desc, cid2idxs, cid2didxs );
                cummt = cummt + toc( itime ); 
                cnt = cnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, cnt, ...
                    sprintf( 'train svm of class %d.', cid ), cummt );
            end
        end
        function evalSvm( this, addrss )
            [ idx2cscore, cid2idxs, cid2didxs ] ...
                = this.testSvm;
            mssg = this.writeReport...
                ( idx2cscore, cid2idxs, cid2didxs );
            cellfun( @( str )fprintf( '%s\n', str ), mssg );
            title = sprintf( '%s: TEST REPORT', ...
                upper( mfilename ) );
            if ~isempty( addrss )
                sendEmail( ...
                    'visionresearchreport@gmail.com', ...
                    'visionresearchreporter', ...
                    addrss, title, mssg, [  ] );
            end
        end
        function cid2score = predictIm( this, im )
            kernel = this.setting.kernel;
            norm = this.setting.norm;
            cid2w = this.loadSvm;
            desc = this.srcImDscrber.im2desc...
                ( im, kernel, norm );
            desc = cat( 1, desc, 1 );
            cid2score = cid2w' * desc;
        end
        function descs = loadDbDescs( this, iids )
            kernel = this.setting.kernel;
            norm = this.setting.norm;
            numIm = numel( iids );
            descs = cell( numIm, 1 );
            prgrss = 0; cummt = 0; numIm = numel( iids );
            for i = 1 : numIm; itime = tic; iid = iids( i );
                desc = this.srcImDscrber.iid2desc...
                    ( iid, kernel, norm );
                descs{ i } = desc;
                cummt = cummt + toc( itime ); prgrss = prgrss + 1;
                if ( prgrss / numIm ) >= 0.1; prgrss = 0;
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, i, 'Load descs.', cummt );
                end
            end
            descs = cat( 2, descs{ : } );
        end
        function w = cid2svm( this, cid, idx2desc, cid2idxs, cid2didxs )
            fpath = this.getSvmPath( cid );
            try
                data = load( fpath );
                w = data.w;
            catch
                c = this.setting.c;
                epsilon = this.setting.epsilon;
                biasMultiplier = this.setting.biasMultiplier;
                biasLearningRate = this.setting.biasLearningRate;
                loss = this.setting.loss;
                solver = this.setting.solver;
                numDesc = size( idx2desc, 2 );
                lambda = 1 / ( numDesc * c );
                svmSetting = { ...
                    'Epsilon', epsilon, ...
                    'BiasMultiplier', biasMultiplier, ...
                    'BiasLearningRate', biasLearningRate, ...
                    'Loss', loss, ...
                    'Solver', solver };
                idx2y = -1 * ones( numDesc, 1 );
                idx2y( cid2idxs{ cid } )    = 1;
                idx2y( cid2didxs{ cid } )   = 0;
                [ w, b ] = vl_svmtrain( idx2desc, idx2y', lambda, svmSetting{ : } );
                w = cat( 1, w, b );
                save( fpath, 'w' );
            end
        end
        function [ idx2cscore, cid2idxs, cid2didxs ] = ...
                testSvm( this )
            path = this.getTestPath;
            idx2iid = this.srcDb.getTeiids;
            cid2idxs = this.srcDb.getCid2idxs( idx2iid );
            cid2didxs = this.srcDb.getCid2didxs( idx2iid );
            try
                data = load( path );
                idx2cscore = data.idx2cscore;
            catch
                idx2desc = this.loadDbDescs( idx2iid );
                cid2w = this.loadSvm;
                idx2desc = cat( 1, idx2desc, ...
                    ones( 1, size( idx2desc, 2 ) ) );
                idx2cscore = cid2w' * idx2desc;
                this.makeDir;
                save( path, 'idx2cscore' );
            end
        end
        function cid2w = loadSvm( this )
            fprintf( '%s: load svm.\n', upper( mfilename ) );
            cids = num2cell( 1 : this.srcDb.getNumClass );
            cid2w = cellfun...
                ( @( cid )this.cid2svm( cid, [  ], [  ], [  ] ), ...
                cids, 'UniformOutput', false );
            cid2w = cat( 2, cid2w{ : } );
        end
        % Functions for evaluation metrics.
        function metric = computeTopAcc( this, idx2cscore, topn )
            idx2iid = this.srcDb.getTeiids;
            idx2cid = cell2mat( this.srcDb.iid2gt( idx2iid ) );
            [ ~, idx2topcid ] = sort( idx2cscore, 1, 'descend' );
            idx2topncid = idx2topcid( 1 : topn, : );
            idx2hit = ~prod( idx2topncid - repmat( idx2cid', topn, 1 ), 1 );
            metric = mean( idx2hit );
        end
        function [ cid2ap, cid2ap11 ] = computeAp...
                ( this, idx2cscore, cid2idxs, cid2didxs )
            numClass = this.srcDb.getNumClass;
            numDesc = size( idx2cscore, 2 );
            cid2ap = zeros( numClass, 1 );
            cid2ap11 = zeros( numClass, 1 );
            for cid = 1 : numClass
                idx2y = -1 * ones( numDesc, 1 );
                idx2y( cid2idxs{ cid } ) = 1;
                idx2y( cid2didxs{ cid } ) = 0;
                [ ~, ~, info ] = vl_pr( idx2y, idx2cscore( cid, : )' );
                cid2ap( cid ) = info.ap;
                cid2ap11( cid ) = info.ap_interp_11;
            end
        end
        function cid2ap = computeVocAp...
                ( this, idx2cscore, cid2idxs, cid2didxs )
            numDesc = size( idx2cscore, 2 );
            cid2ap = zeros( size( cid2idxs ) );
            for cid = 1 : numel( cid2idxs )
                gt = -1 * ones( numDesc, 1 );
                gt( cid2idxs{ cid } ) = 1;
                gt( cid2didxs{ cid } ) = 0;
                [ ~, si ] = sort( -idx2cscore( cid, : ) );
                tp = gt( si ) > 0;
                fp = gt( si ) < 0;
                fp = cumsum( fp );
                tp = cumsum( tp );
                rec = tp / sum( gt > 0 );
                prec = tp ./ ( fp + tp );
                ap = 0;
                for t = 0 : 0.1 : 1
                    p = max( prec( rec >= t ) );
                    if isempty( p )
                        p = 0;
                    end
                    ap = ap + p / 11;
                end
                cid2ap( cid ) = ap;
            end
        end
        % Functions to report the result.
        function mssg = writeReport...
                ( this, idx2cscore, cid2idxs, cid2didxs )
            mssg = {  };
            mssg{ end + 1 } = '___________';
            mssg{ end + 1 } = 'TEST REPORT';
            mssg{ end + 1 } = sprintf( 'DATABASE: %s', this.srcDb.name );
            mssg{ end + 1 } = sprintf( 'IMAGE DESCRIBER: %s', this.srcImDscrber.getName );
            mssg{ end + 1 } = sprintf( 'CLASSIFIER: %s', this.getSvmName );
            if this.srcDb.isMutiLabel
                % cid2ap = this.computeVocAp...
                %     ( idx2cscore, cid2idxs, cid2didxs );
                % mssg{ end + 1 } = sprintf( 'MAP: %.2f%%', ...
                %     mean( cid2ap ) * 100 );
                [ cid2ap, cid2ap11 ] = computeAp...
                    ( this, idx2cscore, cid2idxs, cid2didxs );
                mssg{ end + 1 } = sprintf( 'MAP: %.2f%%', ...
                    mean( cid2ap ) * 100 );
                mssg{ end + 1 } = sprintf( 'MAP11: %.2f%%', ...
                    mean( cid2ap11 ) * 100 );
            else
                mssg{ end + 1 } = sprintf( 'TOP1 ACCURACY: %.2f%%', ...
                    this.computeTopAcc( idx2cscore, 1 ) * 100 );
            end
        end
        % Functions for data I/O.
        function name = getSvmName( this )
            name = sprintf( 'SB_%s_OF_%s', ...
                this.setting.changes, ...
                this.srcImDscrber.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this )
            name = this.getSvmName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name ); 
                name = strcat( 'SB_', name );
            end
            dir = fullfile...
                ( this.srcDb.dstDir, name );
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getSvmPath( this, cid )
            fname = sprintf...
                ( 'ID%04d.mat', cid );
            fpath = fullfile...
                ( this.getDir, fname );
        end
        function fpath = getTestPath( this )
            fname = 'TE.mat';
            fpath = fullfile...
                ( this.getDir, fname );
        end
    end
end

