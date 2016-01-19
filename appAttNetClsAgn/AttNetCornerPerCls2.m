classdef AttNetCornerPerCls2 < handle
    properties
        db;
        attNet;
        propObj;
        directions;
        settingMain;
        settingPost;
    end
    methods( Access = public )
        function this = AttNetCornerPerCls2( ...
                db, ...
                attNet, ...
                propObj, ...
                settingMain, ...
                settingPost )
            this.db = db;
            this.attNet = attNet;
            this.propObj = propObj;
            this.settingMain.rescaleBox = 1;
            this.settingMain.directionVectorSize = 30;
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
        function init( this, gpus )
            % Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            numDirection = 3;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.directions.did2vecTl = [ [ cos( did2angTl' ); sin( did2angTl' ); ], [ 0; 0; ] ];
            this.directions.did2vecBr = [ [ cos( did2angBr' ); sin( did2angBr' ); ], [ 0; 0; ] ];
            % Fetch net on GPU.
            this.attNet.layers{ end }.type = 'softmax';
            this.attNet = Net.fetchNetOnGpu( this.attNet, gpus );
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det( this, iid )
            % Get initial proposals.
            rid2tlbr0 = this.propObj.iid2det( iid );
            % Pre-processing: box re-scaling.
            rescaleBox = this.settingMain.rescaleBox;
            rid2tlbr0 = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            rid2tlbr0 = round( rid2tlbr0 );
            % Do detection on each region.
            rgbMean = this.attNet.normalization.averageImage;
            interpolation = this.attNet.normalization.interpolation;
            im = imread( this.db.iid2impath{ iid } );
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0( 1 : 4, : ) = bsxfun( @minus, rid2tlbr0( 1 : 4, : ), [ imTl; imTl; ] ) + 1;
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], rgbMean, interpolation );
            [ rid2tlbr, rid2out ] = this.detMultiRegns( rid2tlbr0, imGlobal );
            % [ rid2tlbr, rid2out ] = this.detMultiRegnsDebug( rid2tlbr0, imGlobal );
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
            if nargout,
                % Compute each region score.
                buffSize = 5000;
                numDimPerDirLyr = 4;
                numCls = this.db.getNumClass;
                numDimClsLyr = numCls + 1;
                signStop = numDimPerDirLyr;
                dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
                did2tlbr = zeros( 4, buffSize, 'single' );
                did2score = zeros( 1, buffSize, 'single' );
                did2cid = zeros( 1, buffSize, 'single' );
                did2fill = false( 1, buffSize );
                did = 1;
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    rid2outCls = rid2out( dimCls, : );
                    [ rid2stl, rid2ptl ] = max( rid2outTl, [  ], 1 );
                    [ rid2sbr, rid2pbr ] = max( rid2outBr, [  ], 1 );
                    [ rid2sCls, rid2pCls ] = max( rid2outCls, [  ], 1 );
                    rid2stl = rid2stl * 2 - sum( rid2outTl, 1 );
                    rid2sbr = rid2sbr * 2 - sum( rid2outBr, 1 );
                    rid2sCLs = rid2sCls * 2 - sum( rid2outCls, 1 );
                    rid2s = ( rid2stl + rid2sbr ) / 2 + rid2sCLs;
                    rid2stop = rid2ptl == signStop & rid2pbr == signStop;
                    rid2fgd = rid2pCls == cid;
                    rid2ok = rid2stop & rid2fgd;
                    numDet = sum( rid2ok );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = rid2tlbr( :, rid2ok );
                    did2score( dids ) = rid2s( rid2ok );
                    did2cid( dids ) = cid;
                    did2fill( dids ) = true;
                    did = did + numDet;
                end;
                rid2tlbr = did2tlbr( :, did2fill );
                rid2score = did2score( :, did2fill );
                rid2cid = did2cid( :, did2fill );
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
        function demo2( this, fid, iid )
            if nargin < 3,
                iid = this.db.getTeiids;
                iid = randsample( iid', 1 );
            end;
            im = imread( this.db.iid2impath{ iid } );
            [ rid2tlbr, ~, rid2cid ] = this.iid2det( iid );
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
        function [ ap, rank2iid, rank2bbox, rank2tp, rank2fp ] = ...
                computeAp( this, addrss )
            % Set parameters.
            clsName = this.attNet.srcInOut.settingTsDb.selectClassName;
            minoverlap = 0.5;
            minDetArea = 500;
            clsId = cellfun( ...
                @( cname )strcmp( cname, clsName ), ...
                this.db.cid2name );
            clsId = find( clsId );
            % Prepare data.
            iidx2iid = this.db.getTeiids;
            numTeIm = numel( iidx2iid );
            oid2target = this.db.oid2cid == clsId;
            % Get ground-truth.
            iidx2oidx2bbox = cell( numTeIm, 1 );
            iidx2oidx2diff = cell( numTeIm, 1 );
            iidx2oidx2det = cell( numTeIm, 1 );
            numPos = 0;
            fprintf( '%s: Compute gt.\n', upper( mfilename ) );
            for iidx = 1 : numTeIm;
                iid = iidx2iid( iidx );
                oidx2oid = ( this.db.oid2iid == iid ) & oid2target;
                iidx2oidx2bbox{ iidx } = this.db.oid2bbox( :, oidx2oid );
                iidx2oidx2diff{ iidx } = this.db.oid2diff( oidx2oid );
                iidx2oidx2det{ iidx } = false( sum( oidx2oid ), 1 );
                numPos = numPos + sum( ~iidx2oidx2diff{ iidx } );
            end
            % Cut detection results by minimum area.
            did2rect = tlbr2rect( this.did2tlbr );
            did2area = prod( did2rect( 3 : 4, : ), 1 );
            did2ok = did2area >= minDetArea;
            did2score_ = this.did2score( did2ok );
            did2iid_ = this.did2iid( did2ok );
            did2tlbr_ = this.did2tlbr( :, did2ok );
            % Sort detection results.
            [ ~, rank2did ] = sort( - did2score_ );
            rank2iid = did2iid_( rank2did );
            rank2bbox = did2tlbr_( :, rank2did );
            % Determine TP/FP/DONT-CARE.
            numDet = numel( rank2did );
            rank2tp = zeros( numDet, 1 );
            rank2fp = zeros( numDet, 1 );
            for r = 1 : numDet,
                iid = rank2iid( r );
                iidx = find( iidx2iid == iid );
                detBbox = rank2bbox( :, r );
                ovMax = -Inf;
                for oidx = 1 : size( iidx2oidx2bbox{ iidx }, 2 ),
                    gtBbox = iidx2oidx2bbox{ iidx }( :, oidx );
                    insectBbox = [  ...
                        max( detBbox( 1 ), gtBbox( 1 ) ); ...
                        max( detBbox( 2 ), gtBbox( 2 ) ); ...
                        min( detBbox( 3 ), gtBbox( 3 ) ); ...
                        min( detBbox( 4 ), gtBbox( 4 ) ); ];
                    insectW = insectBbox( 3 ) - insectBbox( 1 ) + 1;
                    insectH = insectBbox( 4 ) - insectBbox( 2 ) + 1;
                    if insectW > 0 && insectH > 0,
                        union = ...
                            ( detBbox( 3 ) - detBbox( 1 ) + 1 ) * ...
                            ( detBbox( 4 ) - detBbox( 2 ) + 1 ) + ...
                            ( gtBbox( 3 ) - gtBbox( 1 ) + 1 ) * ...
                            ( gtBbox( 4 ) - gtBbox( 2 ) + 1 ) - ...
                            insectW * insectH;
                        ov = insectW * insectH / union;
                        if ov > ovMax, ovMax = ov; oidxMax = oidx; end;
                    end
                end
                if ovMax >= minoverlap,
                    if ~iidx2oidx2diff{ iidx }( oidxMax ),
                        if ~iidx2oidx2det{ iidx }( oidxMax ),
                            rank2tp( r ) = 1;
                            iidx2oidx2det{ iidx }( oidxMax ) = true;
                        else
                            rank2fp( r ) = 1;
                        end
                    end
                else
                    rank2fp( r ) = 1;
                end
            end
            % Compute AP.
            rank2fpCum = cumsum( rank2fp );
            rank2tpCum = cumsum( rank2tp );
            rec = rank2tpCum / numPos;
            prec = rank2tpCum ./ ( rank2fpCum + rank2tpCum );
            ap=0;
            for t = 0 : 0.1 : 1,
                p = max( prec( rec >= t ) );
                if isempty( p ), p = 0; end;
                ap = ap + p / 11;
            end
            this.reportTestResult( ap, addrss );
        end
        function reportTestResult...
                ( this, ap, addrss )
            title = sprintf( '%s: TEST REPORT', ...
                upper( mfilename ) );
            mssg = {  };
            mssg{ end + 1 } = '___________';
            mssg{ end + 1 } = 'TEST REPORT';
            mssg{ end + 1 } = sprintf( 'DATABASE: %s', ...
                this.db.name );
            mssg{ end + 1 } = sprintf( 'TARGET CLASS: %s', ...
                upper( this.attNet.srcInOut.settingTsDb.selectClassName ) );
            mssg{ end + 1 } = sprintf( 'INOUT: %s', ...
                this.attNet.srcInOut.getName );
            mssg{ end + 1 } = sprintf( 'CNN: %s', ...
                this.attNet.getNetName );
            mssg{ end + 1 } = sprintf( 'DETECTOR: %s', ...
                this.getName );
            mssg{ end + 1 } = sprintf( 'INIT MERGE: %s', ...
                this.settingInitMrg.changes );
            mssg{ end + 1 } = sprintf( 'REFINE ID: %d', ...
                    this.refineId );
            if this.refineId > 0,
                mssg{ end + 1 } = sprintf( 'REFINE: %s', ...
                    this.settingRefine.changes );
            end
            mssg{ end + 1 } = ...
                sprintf( 'AP: %.2f%', ap * 100 );
            mssg{ end + 1 } = ' ';
            cellfun( @( str )fprintf( '%s\n', str ), mssg );
            if ~isempty( addrss )
                sendEmail( ...
                    'visionresearchreport@gmail.com', ...
                    'visionresearchreporter', ...
                    addrss, ...
                    title, ...
                    mssg, ...
                    '' );
            end
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
        function [ did2tlbr, did2out ] = detMultiRegns( this, rid2tlbr, im )
            % Preparing for data.
            testBatchSize = 256;
            numMaxFeed = this.settingMain.numMaxTest;
            interpolation = this.attNet.normalization.interpolation;
            dvecSize = this.settingMain.directionVectorSize;
            inputSide = this.attNet.normalization.imageSize( 1 );
            inputCh = size( im, 3 );
            useGpu = isa( this.attNet.layers{ 1 }.weights{ 1 }, 'gpuArray' );
            numOutDim = size( this.attNet.layers{ end - 1 }.weights{ 1 }, 4 );
            targetLyrId = numel( this.attNet.layers ) - 1;
            numDimPerDirLyr = 4;
            numCls = this.db.getNumClass;
            numDimClsLyr = numCls + 1;
            signStop = numDimPerDirLyr;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            numRegn = size( rid2tlbr, 2 );
            buffSize = 5000;
            thresh = 2; 1; +Inf; 3; 0.4; 
            if ~numRegn, 
                did2tlbr = zeros( 4, 0, 'single' ); 
                did2out = zeros( numOutDim, 0, 'single' ); return; 
            end;
            % Detection on each region.
            did2tlbr = zeros( 4, buffSize, 'single' );
            did2out = zeros( numOutDim, buffSize, 'single' );
            did2fill = false( 1, buffSize );
            did = 1;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                for r = 1 : testBatchSize : numRegn,
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2im = zeros( inputSide, inputSide, inputCh, bsize, 'single' );
                    parfor brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ inputSide, inputSide ], 'method', interpolation );
                    end;
                    % Feedforward for MatConvNet.
                    if useGpu, brid2im = gpuArray( brid2im ); end;
                    brid2out = my_simplenn( this.attNet, brid2im, [  ], [  ], ...
                        'accumulate', false, 'disableDropout', true, ...
                        'conserveMemory', true, 'backPropDepth', +inf, ...
                        'targetLayerId', targetLyrId, 'sync', true ); clear brid2im;
                    % Feedforward for Caffe.
                    
                    brid2out = brid2out( end - 1 ).x;
                    brid2out( :, :, dimCls, : ) = ...
                        vl_nnsoftmax( brid2out( :, :, dimCls, : ) );
                    brid2out = gather( brid2out );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                nrid2tlbr = zeros( 4, buffSize, 'single' );
                nrid2fill = false( 1, buffSize );
                nrid = 1;
                [ ~, rid2pCls ] = max( rid2out( dimCls, : ), [  ], 1 );
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    [ rid2stl, rid2ptl ] = max( rid2out( dimTl, : ), [  ], 1 );
                    [ rid2sbr, rid2pbr ] = max( rid2out( dimBr, : ), [  ], 1 );
                    rid2stl = rid2stl * 2 - sum( rid2out( dimTl, : ), 1 );
                    rid2sbr = rid2sbr * 2 - sum( rid2out( dimBr, : ), 1 );
                    % Find and store detections.
                    rid2stop = rid2ptl == signStop & rid2pbr == signStop;
                    rid2fgd = rid2pCls == cid;
                    rid2det = rid2stop & rid2fgd;
                    numDet = sum( rid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = rid2tlbr( :, rid2det );
                    did2out( :, dids ) = rid2out( :, rid2det );
                    did2fill( dids ) = true;
                    did = did + numDet;
                    % Find and store regiones to be continued.
                    rid2fgd2 = ...
                        ( rid2pCls ~= ( numCls + 1 ) ) & ...
                        ( rid2ptl == 2 & rid2pbr == 2 ) & ...
                        ( rid2stl > thresh & rid2sbr > thresh );
                    rid2cont = ( rid2fgd | rid2fgd2 ) & ~rid2det;
                    numCont = sum( rid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = rid2tlbr( :, rid2cont );
                    idx2ptl = rid2ptl( rid2cont );
                    idx2pbr = rid2pbr( rid2cont );
                    idx2tlbrWarp = [ ...
                        this.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.directions.did2vecBr( :, idx2pbr ) * dvecSize + inputSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ inputSide, inputSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    nrids = nrid : nrid + numCont - 1;
                    nrid2tlbr( :, nrids ) = idx2tlbr;
                    nrid2fill( nrids ) = true;
                    nrid = nrid + numCont;
                end;
                rid2tlbr = round( nrid2tlbr( :, nrid2fill ) );
                rid2tlbr = unique( rid2tlbr', 'rows', 'stable' );
                rid2tlbr = rid2tlbr';
                numRegn = size( rid2tlbr, 2 );
                if ~numRegn, break; end;
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2out = did2out( :, did2fill );
        end
    end
end






