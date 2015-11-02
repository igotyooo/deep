classdef AttNetCaffe < handle
    properties
        db;
        attNet;
        attNetName;
        rgbMean;
        inputSide;
        weights;
        biases;
        propObj;
        directions;
        settingMain;
        settingPost;
    end
    methods( Access = public )
        function this = AttNetCaffe( ...
                db, ...
                propObj, ...
                settingMain, ...
                settingPost )
            this.db = db;
            this.propObj = propObj;
            this.settingMain.rescaleBox = 1;
            this.settingMain.directionVectorSize = 30;
            this.settingMain.numTopClassification = 1;
            this.settingMain.numTopDirection = 1;
            this.settingMain.numMaxTest = 50;
            this.settingPost.mergingOverlap = 0.7;
            this.settingPost.mergingType = 'OV'; 'NMS';
            this.settingPost.mergingMethod = 'WAVG'; 'MAX';
            this.settingPost.minimumNumSupportBox = 0;
            this.settingMain = setChanges( ...
                this.settingMain, ...
                settingMain, ...
                upper( mfilename ) );
            this.settingPost = setChanges( ...
                this.settingPost, ...
                settingPost, ...
                upper( mfilename ) );
        end
        function init( this, netInfo, gpus )
            % Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            numDirection = 3;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.directions.did2vecTl = [ [ cos( did2angTl' ); sin( did2angTl' ); ], [ 0; 0; ] ];
            this.directions.did2vecBr = [ [ cos( did2angBr' ); sin( did2angBr' ); ], [ 0; 0; ] ];
            % Fetch output layer's parameters on GPU.
            this.attNetName = netInfo.modelName;
            this.inputSide = netInfo.patchSide;
            this.rgbMean = load( netInfo.rgbMeanPath, 'rgbMean' );
            this.rgbMean = this.rgbMean.rgbMean;
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            net = caffe.Net( netInfo.protoPath, netInfo.modelPath, 'test' );
            prefix = 'dir';
            clsLyrName = 'cls';
            cornerNameTl = 'TL';
            cornerNameBr = 'BR';
            numCls = this.db.getNumClass;
            numLyr = numCls * 2 + 1;
            mlid2clid = zeros( numLyr, 1 );
            mlid2name = cell( numLyr, 1 );
            mlid2w = cell( numLyr, 1 );
            mlid2b = cell( numLyr, 1 );
            for lid = 1 : numLyr,
                if lid == numLyr,
                    lname = clsLyrName;
                else
                    if mod( lid, 2 ), cornerName = cornerNameTl; else cornerName = cornerNameBr;  end;
                    lname = sprintf( '%s%d_%s', prefix, ( ( lid - 1 ) - mod( lid - 1, 2 ) ) / 2, cornerName );
                end;
                mlid2clid( lid ) = net.name2layer_index( lname );
                mlid2name{ lid } = lname;
                mlid2w{ lid } = net.params( lname, 1 ).get_data(  );
                mlid2b{ lid } = net.params( lname, 2 ).get_data(  );
                mlid2b{ lid } = mlid2b{ lid }';
            end;
            this.weights = gpuArray( cat( 4, mlid2w{ : } ) );
            this.biases = gpuArray( cat( 2, mlid2b{ : } ) );
            % Fetch att net on GPU without the output layer.
            caffe.reset_all(  );
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            this.attNet = caffe.Net( netInfo.protoPathTest, netInfo.modelPath, 'test' );
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det( this, iid )
            % Get initial proposals.
            [ rid2tlbr0, nid2rid0, nid2cid0 ] = this.propObj.iid2det( iid );
            % Pre-processing: box re-scaling.
            rescaleBox = this.settingMain.rescaleBox;
            rid2tlbr0 = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            rid2tlbr0 = round( rid2tlbr0 );
            % Do detection on each region.
            interpolation = 'bilinear';
            im = imread( this.db.iid2impath{ iid } );
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0( 1 : 4, : ) = bsxfun( @minus, rid2tlbr0( 1 : 4, : ), [ imTl; imTl; ] ) + 1;
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], this.rgbMean, interpolation );
            [ rid2tlbr, rid2score, rid2cid ] = this.detFromRegns( rid2tlbr0, nid2rid0, nid2cid0, imGlobal );
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
            if nargout,
                % Post-processing: merge bounding boxes.
                if this.settingPost.mergingOverlap ~= 1,
                    [ rid2tlbr, rid2score, rid2cid ] = ...
                        this.merge( rid2tlbr, rid2score, rid2cid );
                end;
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2detFromCls( this, iid, cids )
            % Get initial proposals.
            [ rid2tlbr0, nid2rid0, nid2cid0 ] = this.propObj.iid2det( iid, cids );
            % Pre-processing: box re-scaling.
            rescaleBox = this.settingMain.rescaleBox;
            rid2tlbr0 = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            rid2tlbr0 = round( rid2tlbr0 );
            % Do detection on each region.
            interpolation = 'bilinear';
            im = imread( this.db.iid2impath{ iid } );
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0( 1 : 4, : ) = bsxfun( @minus, rid2tlbr0( 1 : 4, : ), [ imTl; imTl; ] ) + 1;
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], this.rgbMean, interpolation );
            [ rid2tlbr, rid2score, rid2cid ] = this.detFromRegnsAndCls( rid2tlbr0, nid2rid0, nid2cid0, imGlobal );
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
            if nargout,
                % Post-processing: merge bounding boxes.
                if this.settingPost.mergingOverlap ~= 1,
                    [ rid2tlbr, rid2score, rid2cid ] = ...
                        this.merge( rid2tlbr, rid2score, rid2cid );
                end;
            end;
        end
        function demo( this, fid, wait, iid )
            if nargin < 4,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            [ rid2tlbr, rid2score, rid2cid ] = this.iid2det( iid );
            figure( fid );
            set( gcf, 'color', 'w' );
            if wait,
                for rid = 1 : numel( rid2score ),
                    name = sprintf( '%s, %.2f', ...
                        this.db.cid2name{ rid2cid( rid ) }, rid2score( rid ) );
                    plottlbr( round( rid2tlbr( :, rid ) ), im, false, 'r', { name } ); 
                    title( sprintf( 'Detection. (IID%06d)', iid ) );
                    waitforbuttonpress;
                end;
            else
                % plottlbr( round( rid2tlbr ), im, false, { 'r'; 'g'; 'b'; 'y' } );
                % title( sprintf( 'Detection. (IID%06d)', iid ) );
                name = arrayfun( @( cid, score )sprintf( '%s, %.1f', this.db.cid2name{ cid }, score ), rid2cid, rid2score, 'UniformOutput', false );
                plottlbr( round( rid2tlbr ), im, false, 'r', name );
                title( sprintf( 'Detection. (IID%06d)', iid ) );
            end;
        end
        function demo2( this, fid, position, iid )
            if nargin < 4,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            tic;
            [ rid2tlbr, ~, rid2cid ] = this.iid2det( iid );
            toc;
            rid2tlbr = round( rid2tlbr );
            cids = unique( rid2cid );
            numCls = numel( cids );
            figure( fid );
            set( gcf, 'color', 'w' );
            for c = 1 : numCls,
                cid = cids( c );
                rid2ok = rid2cid == cid;
                plottlbr( rid2tlbr( :, rid2ok ), im, false, { 'r'; 'g'; 'b'; 'y' } );
                title( sprintf( '%s, IID%06d', this.db.cid2name{ cid }, iid ) );
                hold off;
                setFigPos( gcf, position ); drawnow;
                waitforbuttonpress;
                newrid2tlbr = ov( rid2tlbr( :, rid2ok ), ones( sum( rid2ok ), 1 )', 0.6, 1, 'WAVG' );
                if isempty( newrid2tlbr ), continue; end;
                plottlbr( round( newrid2tlbr ), im, false, 'c' );
                title( sprintf( '%s, IID%06d', this.db.cid2name{ cid }, iid ) );
                hold off;
                setFigPos( gcf, position ); drawnow;
                waitforbuttonpress;
            end;
        end
        function demo3( this, fid, position, iid, cids )
            if nargin < 4,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            [ rid2tlbr, ~, rid2cid ] = this.iid2detFromCls( iid, cids );
            rid2tlbr = round( rid2tlbr );
            cids = unique( rid2cid );
            numCls = numel( cids );
            figure( fid );
            set( gcf, 'color', 'w' );
            for c = 1 : numCls,
                cid = cids( c );
                rid2ok = rid2cid == cid;
                plottlbr( rid2tlbr( :, rid2ok ), im, false, { 'r'; 'g'; 'b'; 'y' } );
                title( sprintf( '%s, IID%06d', this.db.cid2name{ cid }, iid ) );
                hold off;
                setFigPos( gcf, position ); drawnow;
                waitforbuttonpress;
                newrid2tlbr = ov( rid2tlbr( :, rid2ok ), ones( sum( rid2ok ), 1 )', 0.6, 1, 'WAVG' );
                if isempty( newrid2tlbr ), continue; end;
                plottlbr( round( newrid2tlbr ), im, false, 'c' );
                title( sprintf( '%s, IID%06d', this.db.cid2name{ cid }, iid ) );
                hold off;
                setFigPos( gcf, position ); drawnow;
                waitforbuttonpress;
            end;
        end
        function detSubDb( this, numDiv, divId )
            iids = this.db.getTeiids;
            numIm = numel( iids );
            divSize = ceil( numIm / numDiv );
            sidx = divSize * ( divId - 1 ) + 1;
            eidx = min( sidx + divSize - 1, numIm );
            iids = iids( sidx : eidx );
            this.makeDir;
            numIm = numel( iids );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2det0( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, ...
                    'Init det.', cummt );
            end;
        end
        function detDb( this )
            fpath = this.getDetResPath( 0 );
            try
                fprintf( '%s: Try to load det res.\n', upper( mfilename ) ); 
                data = load( fpath );
                res = data.res;
                fprintf( '%s: Done.\n', upper( mfilename ) );
            catch
                iids = this.db.getTeiids;
                fprintf( '%s: Check if detections exist.\n', ...
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
                    this.iid2det0( iid );
                    cummt = cummt + toc( itime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        'Init det.', cummt );
                end
                iids = this.db.getTeiids;
                numIm = numel( iids );
                res.did2iid = cell( numIm, 1 );
                res.did2tlbr = cell( numIm, 1 );
                res.did2score = cell( numIm, 1 );
                cummt = 0;
                for iidx = 1 : numIm; itime = tic;
                    iid = iids( iidx );
                    [ res.did2tlbr{ iidx }, res.did2score{ iidx } ] = ...
                        this.iid2det( iid );
                    if isempty( res.did2tlbr{ iidx } ), continue; end;
                    res.did2iid{ iidx } = iid * ...
                        ones( size( res.did2tlbr{ iidx }, 2 ), 1 );
                    cummt = cummt + toc( itime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        'Det.', cummt );
                end
                res.did2iid = cat( 1, res.did2iid{ : } );
                res.did2tlbr = cat( 2, res.did2tlbr{ : } );
                res.did2score = cat( 1, res.did2score{ : } );
                fprintf( '%s: Save det results.\n', upper( mfilename ) );
                save( fpath, 'res' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end
            this.did2iid = res.did2iid;
            this.did2tlbr = res.did2tlbr;
            this.did2score = res.did2score;
            this.refineId = 0;
        end
        function dir = getDir( this )
            name = this.getName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'DET_', name );
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
        function fpath = getDetResPath( this, refineId )
            name = sprintf( 'DETRES%d_%s', ...
                refineId, this.settingInitMrg.changes );
            if refineId > 0, name  = strcat( name, '_', this.settingRefine.changes ); end;
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
            fname = sprintf( '%s.mat', name );
            fpath = fullfile...
                ( this.getDir, fname );
        end
    end
    methods( Access = private )
        function [ rid2tlbr, rid2score, rid2cid ] = ...
                merge( this, rid2tlbr, rid2score, rid2cid )
            mergingOverlap = this.settingPost.mergingOverlap;
            mergingType = this.settingPost.mergingType;
            mergingMethod = this.settingPost.mergingMethod;
            minNumSuppBox = this.settingPost.minimumNumSupportBox;
            cids = unique( rid2cid );
            numCls = numel( cids );
            rid2tlbr_ = cell( numCls, 1 );
            rid2score_ = cell( numCls, 1 );
            rid2cid_ = cell( numCls, 1 );
            for cidx = 1 : numCls,
                cid = cids( cidx );
                rid2ok = rid2cid == cid;
                switch mergingType,
                    case 'NMS',
                        [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = nms( ...
                            [ rid2tlbr( :, rid2ok ); rid2score( rid2ok ); ]', ...
                            mergingOverlap, minNumSuppBox, mergingMethod );
                        rid2tlbr_{ cidx } = rid2tlbr_{ cidx }';
                    case 'OV',
                        [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = ov( ...
                            rid2tlbr( :, rid2ok ), rid2score( rid2ok ), ...
                            mergingOverlap, minNumSuppBox, mergingMethod );
                end
                rid2cid_{ cidx } = cid * ones( size( rid2score_{ cidx } ) );
            end;
            rid2tlbr = cat( 2, rid2tlbr_{ : } );
            rid2score = cat( 1, rid2score_{ : } );
            rid2cid = cat( 1, rid2cid_{ : } );
            [ rid2score, idx ] = sort( rid2score, 'descend' );
            rid2tlbr = rid2tlbr( :, idx );
            rid2cid = rid2cid( idx );
        end
        function [ did2tlbr, did2score, did2cid ] = detFromRegns( this, rid2tlbr, nid2rid, nid2cid, im )
            % Preparing for data.
            testBatchSize = 256 / 2;
            numMaxFeed = this.settingMain.numMaxTest;
            interpolation = 'bilinear';
            dvecSize = this.settingMain.directionVectorSize;
            numTopCls = this.settingMain.numTopClassification;
            numTopDir = this.settingMain.numTopDirection;
            inputCh = size( im, 3 );
            numDimPerDirLyr = 4;
            numCls = this.db.getNumClass;
            numDimClsLyr = numCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numCls + numDimClsLyr;
            signStop = numDimPerDirLyr;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            numRegn = size( rid2tlbr, 2 );
            buffSize = 5000;
            if ~numRegn, 
                did2tlbr = zeros( 4, 0, 'single' ); 
                did2score = zeros( 0, 1, 'single' ); 
                did2cid = zeros( 0, 1, 'single' ); 
                return; 
            end;
            % Detection on each region.
            did2tlbr = zeros( 4, buffSize, 'single' );
            did2score = zeros( buffSize, 1, 'single' );
            did2cid = zeros( buffSize, 1, 'single' );
            did2fill = false( 1, buffSize );
            did = 1;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                trsiz = 0;
                tfwd =0;
                for r = 1 : testBatchSize : numRegn,
                    trsiz_ = tic;
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2im = zeros( this.inputSide, this.inputSide, inputCh, bsize, 'single' );
                    for brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ this.inputSide, this.inputSide ], 'method', interpolation );
                    end;
                    trsiz = trsiz + toc( trsiz_ );
                    % Feedforward.
                    tfwd_ = tic;
                    brid2out = this.feedforwardCaffe( brid2im );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                    tfwd = tfwd + toc( tfwd_ );
                end;
                fprintf( '%s: Preproc t = %.2f sec, Fwd t = %.2f sec.\n', upper( mfilename ), trsiz, tfwd );
                % Do the job.
                nrid2tlbr = cell( numCls, 1 );
                signDiag = 2;
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2rank2pCls ] = sort( rid2outCls, 1, 'descend' );
                rid2pCls = rid2rank2pCls( 1, : );
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    [ ~, rid2rank2ptl ] = sort( rid2outTl, 1, 'descend' );
                    [ ~, rid2rank2pbr ] = sort( rid2outBr, 1, 'descend' );
                    rid2ptl = rid2rank2ptl( 1, : );
                    rid2pbr = rid2rank2pbr( 1, : );
                    rid2ss = rid2ptl == signStop & rid2pbr == signStop;
                    rid2okTl = any( rid2rank2ptl( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2okBr = any( rid2rank2pbr( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2dd = rid2okTl & rid2okBr;
                    rid2dd = rid2dd & ( ~rid2ss );
                    rid2dd = rid2dd & ( rid2ptl == signDiag | rid2pbr == signDiag );
                    rid2bgd = rid2pCls == ( numCls + 1 );
                    rid2high = any( rid2rank2pCls( 1 : numTopCls, : ) == cid, 1 );
                    rid2high = rid2high & ( ~rid2bgd );
                    rid2top = rid2pCls == cid;
                    nid2purebred = nid2cid == cid;
                    rid2purebred = false( 1, numRegn );
                    rid2purebred( nid2rid( nid2purebred ) ) = true;
                    % Find and store detections.
                    rid2det = rid2ss & rid2purebred & rid2top; 
                    numDet = sum( rid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = rid2tlbr( :, rid2det );
                    did2cid( dids ) = cid;
                    didx2outTl = rid2outTl( :, rid2det );
                    didx2outBr = rid2outBr( :, rid2det );
                    didx2outCls = rid2outCls( :, rid2det );
                    didx2scoreTl = didx2outTl( signStop, : ) * 2 - sum( didx2outTl, 1 );
                    didx2scoreBr = didx2outBr( signStop, : ) * 2 - sum( didx2outBr, 1 );
                    didx2scoreCls = didx2outCls( cid, : ) * 2 - sum( didx2outCls, 1 );
                    did2score( dids ) = ( didx2scoreTl + didx2scoreBr ) / 2 + didx2scoreCls;
                    did2fill( dids ) = true;
                    did = did + numDet;
                    % Find and store regiones to be continued.
                    rid2purebredCont = ( ~rid2det ) & rid2high & rid2purebred & ( ~rid2ss );
                    rid2branchCont = rid2high & rid2dd & ~rid2purebred;
                    rid2cont = rid2purebredCont | rid2branchCont;
                    numCont = sum( rid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = rid2tlbr( :, rid2cont );
                    idx2ptl = rid2ptl( rid2cont );
                    idx2pbr = rid2pbr( rid2cont );
                    idx2tlbrWarp = [ ...
                        this.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.directions.did2vecBr( :, idx2pbr ) * dvecSize + this.inputSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ this.inputSide, this.inputSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                    nrid2tlbr{ cid } = idx2tlbr;
                end;
                rid2tlbr = round( cat( 2, nrid2tlbr{ : } ) );
                if isempty( rid2tlbr ), break; end;
                [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
                nid2cid = rid2tlbr( 5, : )';
                rid2tlbr = rid2tlbr_';
                numRegn = size( rid2tlbr, 2 );
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2score = did2score( did2fill );
            did2cid = did2cid( did2fill );
        end
        function [ did2tlbr, did2score, did2cid ] = ...
                detFromRegnsAndCls( this, rid2tlbr, nid2rid, nid2cid, im )
            % Preparing for data.
            testBatchSize = 256 / 2;
            numMaxFeed = this.settingMain.numMaxTest;
            interpolation = 'bilinear';
            dvecSize = this.settingMain.directionVectorSize;
            numTopCls = this.settingMain.numTopClassification;
            inputCh = size( im, 3 );
            numDimPerDirLyr = 4;
            numCls = this.db.getNumClass;
            numDimClsLyr = numCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numCls + numDimClsLyr;
            signStop = numDimPerDirLyr;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            numRegn = size( rid2tlbr, 2 );
            buffSize = numel( nid2rid );
            if ~numRegn, 
                did2tlbr = zeros( 4, 0, 'single' ); 
                did2score = zeros( 0, 1, 'single' ); 
                did2cid = zeros( 0, 1, 'single' ); 
                return; 
            end;
            % Detection on each region.
            did2tlbr = zeros( 4, buffSize, 'single' );
            did2score = zeros( buffSize, 1, 'single' );
            did2cid = zeros( buffSize, 1, 'single' );
            did2fill = false( 1, buffSize );
            did = 1;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                trsiz = 0;
                tfwd =0;
                for r = 1 : testBatchSize : numRegn,
                    trsiz_ = tic;
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2im = zeros( this.inputSide, this.inputSide, inputCh, bsize, 'single' );
                    for brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ this.inputSide, this.inputSide ], 'method', interpolation );
                    end;
                    trsiz = trsiz + toc( trsiz_ );
                    % Feedforward.
                    tfwd_ = tic;
                    brid2out = this.feedforwardCaffe( brid2im );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                    tfwd = tfwd + toc( tfwd_ );
                end;
                fprintf( '%s: Preproc t = %.2f sec, Fwd t = %.2f sec.\n', upper( mfilename ), trsiz, tfwd );
                % Do the job.
                nrid2tlbr = cell( numCls, 1 );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2rank2pCls ] = sort( rid2outCls, 1, 'descend' );
                rid2pCls = rid2rank2pCls( 1, : );
                for cid = 1 : numCls,
                    rid2tar = false( size( nid2rid ) );
                    rid2tar( nid2rid( nid2cid == cid ) ) = true;
                    if ~sum( rid2tar ), continue; end;
                    crid2tlbr = rid2tlbr(  :, rid2tar );
                    crid2out = rid2out( :, rid2tar );
                    crid2outCls = rid2outCls( :, rid2tar );
                    
                    crid2pCls = rid2pCls( rid2tar );
                    crid2rank2pCls = rid2rank2pCls( :, rid2tar );
                    crid2bgd = crid2pCls == ( numCls + 1 );
                    crid2high = any( crid2rank2pCls( 1 : numTopCls, : ) == cid, 1 );
                    crid2high = crid2high & ( ~crid2bgd );
                    
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    crid2outTl = crid2out( dimTl, : );
                    crid2outBr = crid2out( dimBr, : );
                    [ ~, crid2ptl ] = max( crid2outTl, [  ], 1 );
                    [ ~, crid2pbr ] = max( crid2outBr, [  ], 1 );
                    crid2ss = crid2ptl == signStop & crid2pbr == signStop;
                    crid2scoreTl = crid2outTl( signStop, : ) * 2 - sum( crid2outTl, 1 );
                    crid2scoreBr = crid2outBr( signStop, : ) * 2 - sum( crid2outBr, 1 );
                    crid2scoreCls = crid2outCls( cid, : ) * 2 - sum( crid2outCls, 1 );
                    crid2score = ( crid2scoreTl + crid2scoreBr ) / 2 + crid2scoreCls;
                    % Find and store detections.
                    crid2det = crid2ss; % Add more conditions!!!
                    numDet = sum( crid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = crid2tlbr( :, crid2det );
                    did2score( dids ) = crid2score( crid2det );
                    did2cid( dids ) = cid;
                    did2fill( dids ) = true;
                    did = did + numDet;
                    % Find and store regiones to be continued.
                    crid2cont = ~crid2det; % Add more conditions!!!
                    numCont = sum( crid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = crid2tlbr( :, crid2cont );
                    idx2ptl = crid2ptl( crid2cont );
                    idx2pbr = crid2pbr( crid2cont );
                    idx2tlbrWarp = [ ...
                        this.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.directions.did2vecBr( :, idx2pbr ) * dvecSize + this.inputSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ this.inputSide, this.inputSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                    nrid2tlbr{ cid } = idx2tlbr;
                end;
                rid2tlbr = round( cat( 2, nrid2tlbr{ : } ) );
                if isempty( rid2tlbr ), break; end;
                [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
                nid2cid = rid2tlbr( 5, : )';
                rid2tlbr = rid2tlbr_';
                numRegn = size( rid2tlbr, 2 );
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2score = did2score( did2fill );
            did2cid = did2cid( did2fill );
        end
        function y = feedforwardCaffe( this, im )
            
            [ h, w, c, n ] = size( im );
            im = im( :, :, [ 3, 2, 1 ], : );
            im = permute( im, [ 2, 1, 3, 4 ] );
            im = { im };
            this.attNet.blobs( 'data' ).reshape( [ w, h, c, n ] );
            x = this.attNet.forward( im );
            if numel( x ) > 1, error( 'Output should be a single.' ); end;
            x = x{ 1 };
            x = permute( x, [ 2, 1, 3, 4 ] );
            x = gpuArray( x );
            y = vl_nnconv( x, this.weights, this.biases, 'pad', 0, 'stride', 1 );
            y = gather( y );
            clear x;
        end
    end
end