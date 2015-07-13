classdef DetSingleCls < handle
    properties
        db;
        attNet;
        rgbMean;
        scales;
        aspects;
        stride;
        patchSize;
        did2dvecTl;
        did2dvecBr;
        signStop;
        signNoObj;
        refineId;
        did2iid;
        did2tlbr;
        did2score;
        settingInitDet;
        settingInitMrg;
        settingRefine;
    end
    methods( Access = public )
        function this = DetSingleCls( ...
                db, ...
                attNet, ...
                settingInitDet, ...
                settingInitMrg, ...
                settingRefine )
            this.db = db;
            this.attNet = attNet;
            this.refineId = 0;
            this.settingInitDet.scaleStep           = 2;
            this.settingInitDet.numScale            = 6;
            this.settingInitDet.dvecLength          = 30;
            this.settingInitDet.numMaxTest          = 50;
            this.settingInitDet.patchMargin         = 0.5;
            this.settingInitDet.numAspect           = 16 / 2;
            this.settingInitDet.confidence          = 0.97;
            this.settingInitMrg.method              = 'OV'; % 'NMS'
            this.settingInitMrg.overlap             = 0.7;
            this.settingInitMrg.minNumSuppBox       = 1;
            this.settingInitMrg.mergeType           = 'WAVG';
            this.settingInitMrg.scoreType           = 'AVG';
            this.settingRefine.dvecLength          = 30;
            this.settingRefine.boxScaleMag          = 2.5;
            this.settingRefine.method               = 'OV'; % 'NMS'
            this.settingRefine.overlap              = 0.4;
            this.settingRefine.minNumSuppBox        = 0;
            this.settingRefine.mergeType            = 'WAVG';
            this.settingRefine.scoreType            = 'AVG';
            this.settingInitDet = setChanges( ...
                this.settingInitDet, ...
                settingInitDet, ...
                upper( mfilename ) );
            this.settingInitMrg = setChanges( ...
                this.settingInitMrg, ...
                settingInitMrg, ...
                upper( mfilename ) );
            this.settingRefine = setChanges( ...
                this.settingRefine, ...
                settingRefine, ...
                upper( mfilename ) );
        end
        function init( this, gpus )
            % Set RGB mean vector.
            this.rgbMean = this.attNet.srcInOut.rgbMean;
            % Determine multiple scales.
            scaleStep = this.settingInitDet.scaleStep;
            numScale = this.settingInitDet.numScale;
            this.scales = scaleStep .^ ( 0 : 0.5 : ( 0.5 * ( numScale - 1 ) ) )';
            % Determine multiple aspect rates.
            numAspect = this.settingInitDet.numAspect;
            confidence = this.settingInitDet.confidence;
            clsName = this.attNet.srcInOut.settingTsDb.selectClassName;
            cid = find( cellfun( @( name )strcmp( name, clsName ), this.db.cid2name ) );
            oid2bbox = this.db.oid2bbox( :, this.db.oid2cid == cid );
            this.aspects = determineAspectRates( oid2bbox, numAspect - 1, confidence );
            this.aspects = unique( cat( 1, 1, this.aspects ) );
            % Set directional unit vectors.
            this.did2dvecTl = this.attNet.srcInOut.did2dvecTl;
            this.did2dvecBr = this.attNet.srcInOut.did2dvecBr;
            this.did2dvecTl  = round( cat( 2, this.did2dvecTl, [ 0; 0; ] ) );
            this.did2dvecBr  = round( cat( 2, this.did2dvecBr, [ 0; 0; ] ) );
            % Set non-directional class ids.
            this.signStop = this.attNet.srcInOut.signStop;
            this.signNoObj = this.attNet.srcInOut.signNoObj;
            % Fetch net on GPU.
            this.attNet = Net.fetchNetOnGpu( this.attNet, gpus );
            % Determine patch size and stride for activation map.
            this.determineInOutRelations( gpus );
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
        function refineDet( this, refineId )
            fpath = this.getDetResPath( refineId );
            try
                fprintf( '%s: Try to load det res. (Refine %d)\n', upper( mfilename ), refineId ); 
                data = load( fpath );
                res = data.res;
                fprintf( '%s: Done.\n', upper( mfilename ) ); 
            catch
                iidx2iid = this.db.getTeiids;
                numIm = numel( iidx2iid );
                res.did2iid = cell( numIm, 1 );
                res.did2tlbr = cell( numIm, 1 );
                res.did2score = cell( numIm, 1 );
                cnt = 0; cummt = 0;
                for iidx = 1 : numIm; itime = tic;
                	iid = iidx2iid( iidx );
                    dids = this.did2iid == iid;
                    did2det = this.did2tlbr( :, dids );
                    if ~isempty( did2det );
                        % Do the job.
                        im = imread( this.db.iid2impath{ iid } );
                        [ did2redet, did2rescore ] = this.im2redet( im, did2det );
                        res.did2iid{ iidx } = iid * ones( size( did2redet, 2 ), 1 );
                        res.did2score{ iidx } = did2rescore;
                        res.did2tlbr{ iidx } = did2redet;
                    end;
                    cummt = cummt + toc( itime );
                    cnt = cnt + 1;
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        sprintf( 'Det. (Refine %d)', refineId ), cummt );
                end
                res.did2iid = cat( 1, res.did2iid{ : } );
                res.did2tlbr = cat( 2, res.did2tlbr{ : } );
                res.did2score = cat( 1, res.did2score{ : } );
                fprintf( '%s: Save det results. (Refine %d)\n', upper( mfilename ), refineId ); 
                save( fpath, 'res' );
                fprintf( '%s: Done.\n', upper( mfilename ) ); 
            end
            this.did2iid = res.did2iid;
            this.did2tlbr = res.did2tlbr;
            this.did2score = res.did2score;
            this.refineId = refineId;
        end
        function [ did2redet, did2rescore ] = im2redet( this, im, did2det )
            % Set parameters.
            docScaleMag = 4;
            boxScaleMag = this.settingRefine.boxScaleMag;
            method = this.settingRefine.method;
            overlap = this.settingRefine.overlap;
            minNumSuppBox = this.settingRefine.minNumSuppBox;
            mergeType = this.settingRefine.mergeType;
            scoreType = this.settingRefine.scoreType;
            dvecLength = this.settingRefine.dvecLength;
            % Augment document size.
            imSize = size( im ); imSize = imSize( 1 : 2 );
            imTargetSize = round( imSize * sqrt( docScaleMag ) );
            center = round( imTargetSize / 2 );
            imTl = center - round( imSize / 2 ) + 1;
            imBr = imTl + imSize - 1;
            imBgd = imresize( uint8( this.rgbMean ), imTargetSize );
            imBgd( imTl( 1 ) : imBr( 1 ), imTl( 2 ) : imBr( 2 ), : ) = im;
            imMag = imBgd;
            did2det = bsxfun( @plus, did2det, [ imTl( : ); imTl( : ); ] ) - 1;
            % Make re-initilized bounding boxes.
            [ nr, nc, ~ ] = size( imMag );
            did2size = tlbr2rect( did2det );
            did2size = did2size( [ 4, 3 ], : );
            did2tsize = did2size * sqrt( boxScaleMag );
            did2center = [ sum( did2det( [ 1, 3 ], : ), 1 ); sum( did2det( [ 2, 4 ], : ), 1 ) ] / 2;
            did2reinit = zeros( size( did2det ) );
            did2reinit( 1 : 2, : ) = did2center - round( did2tsize / 2 );
            did2reinit( 3 : 4, : ) = did2center + round( did2tsize / 2 );
            did2reinit = round( bndtlbr( did2reinit, [ 1; 1; nr; nc; ] ) );
            % Restart detection.
            testBatchSize = 256;
            numDet = size( did2reinit, 2 );
            did2redet = cell( numel( 1 : testBatchSize : numDet ), 1 );
            did2reout = cell( numel( 1 : testBatchSize : numDet ), 1 );
            cnt = 0;
            for d = 1 : testBatchSize : numDet,
                bdids = d : min( d + testBatchSize - 1, numDet );
                bregns = did2reinit( :, bdids );
                cnt = cnt + 1;
                [ did2redet{ cnt }, did2reout{ cnt } ] = ...
                    this.detMultiRegns( bregns, imMag, dvecLength );
            end;
            did2redet = cat( 2, did2redet{ : } );
            did2reout = cat( 2, did2reout{ : } );
            % Back to original image domain.
            bnd = [ imTl( : ); imBr( : ); ];
            [ did2redet, ok ] = bndtlbr( did2redet, bnd );
            did2reout = did2reout( :, ok );
            did2redet = bsxfun( @minus, did2redet, [ imTl( : ); imTl( : ); ] ) + 1;
            % Compute detection scores.
            did2outTl = did2reout( 1 : this.signNoObj, : );
            did2outBr = did2reout( this.signNoObj + 1 : end, : );
            did2rescore = ...
                did2outTl( this.signStop, : ) + ...
                did2outBr( this.signStop, : ) - ...
                did2outTl( this.signNoObj, : ) - ...
                did2outBr( this.signNoObj, : ) - ...
                sum( did2outTl( 1 : this.signStop - 1, : ), 1 ) - ...
                sum( did2outBr( 1 : this.signStop - 1, : ), 1 );
            % Merge again.
            switch method,
                case 'NMS',
                    [ did2redet, did2rescore ] = ...
                        this.mergeDetBoxsByNms( ...
                        did2redet, ...
                        did2rescore, ...
                        overlap, ...
                        minNumSuppBox, ...
                        mergeType, ...
                        scoreType );
                case 'OV',
                    [ did2redet, did2rescore ] = ...
                        this.mergeDetBoxsByOvlp( ...
                        did2redet, ...
                        did2rescore, ...
                        overlap, ...
                        minNumSuppBox, ...
                        mergeType, ...
                        scoreType );
            end
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
        function [ did2tlbr, did2out ] = ...
                iid2det0( this, iid )
            fpath = this.getPath( iid );
            try
                data = load( fpath );
                did2tlbr = data.det.did2tlbr;
                did2out = data.det.did2out;
            catch
                im = imread( this.db.iid2impath{ iid } );
                [ did2tlbr, did2out ] = ...
                    this.im2det0( im );
                det.did2tlbr = did2tlbr;
                det.did2out = did2out;
                save( fpath, 'det' );
            end;
        end
        function [ did2det, did2score ] = ...
                iid2det( this, iid )
            % Initial detection.
            [ did2det, did2out ] = ...
                this.iid2det0( iid );
            outLyrDim = size( did2out, 1 ) / 2;
            did2outTl = did2out( 1 : outLyrDim, : );
            did2outBr = did2out( outLyrDim + 1 : end, : );
            did2score = ...
                did2outTl( this.signStop, : ) + ...
                did2outBr( this.signStop, : ) - ...
                did2outTl( this.signNoObj, : ) - ...
                did2outBr( this.signNoObj, : ) - ...
                sum( did2outTl( 1 : this.signStop - 1, : ), 1 ) - ...
                sum( did2outBr( 1 : this.signStop - 1, : ), 1 );
            % Merge.
            method = this.settingInitMrg.method;
            overlap = this.settingInitMrg.overlap;
            minNumSuppBox = this.settingInitMrg.minNumSuppBox;
            mergeType = this.settingInitMrg.mergeType;
            scoreType = this.settingInitMrg.scoreType;
            switch method,
                case 'NMS',
                    [ did2det, did2score ] = ...
                        this.mergeDetBoxsByNms( ...
                        did2det, ...
                        did2score, ...
                        overlap, ...
                        minNumSuppBox, ...
                        mergeType, ...
                        scoreType );
                case 'OV',
                    [ did2det, did2score ] = ...
                        this.mergeDetBoxsByOvlp( ...
                        did2det, ...
                        did2score, ...
                        overlap, ...
                        minNumSuppBox, ...
                        mergeType, ...
                        scoreType );
            end;
        end
        function [ rid2det, rid2out ] = im2det0( this, im )
            % Pre-filtering by initial guess.
            fprintf( '%s: Initial guess.\n', upper( mfilename ) );
            [ rid2out, rid2tlbr, imGlobal, sideMargin ] = ...
                this.initGuess( im );
            predsTl = rid2out( 1 : this.signNoObj, : );
            predsBr = rid2out( this.signNoObj + 1 : end, : );
            [ ~, predsTl ] = max( predsTl, [  ], 1 );
            [ ~, predsBr ] = max( predsBr, [  ], 1 );
            rid2ok = predsTl == 2 & predsBr == 2;
            rid2tlbr = rid2tlbr( 1 : 4, rid2ok );
            if isempty( rid2tlbr ), 
                fprintf( '%s: No object.\n', upper( mfilename ) );
                rid2det = zeros( 4, 0 );
                rid2out = zeros( this.signNoObj * 2, 0 );
                return;
            end;
            % Do detection on each region.
            testBatchSize = 256;
            dvecLength = this.settingInitDet.dvecLength;
            numRegn = size( rid2tlbr, 2 );
            rid2det = cell( numel( 1 : testBatchSize : numRegn ), 1 );
            rid2out = cell( numel( 1 : testBatchSize : numRegn ), 1 );
            cnt = 0;
            for r = 1 : testBatchSize : numRegn,
                brids = r : min( r + testBatchSize - 1, numRegn );
                bregns = rid2tlbr( :, brids );
                cnt = cnt + 1;
                [ rid2det{ cnt }, rid2out{ cnt } ] = ...
                    this.detMultiRegns( bregns, imGlobal, dvecLength );
            end;
            rid2det = cat( 2, rid2det{ : } );
            rid2out = cat( 2, rid2out{ : } );
            % Convert to original image domain.
            imGlobalSize = [ size( imGlobal, 1 ); size( imGlobal, 2 ); ];
            bnd = [ sideMargin + 1; imGlobalSize - sideMargin; ];
            [ rid2det, ok ] = bndtlbr( rid2det, bnd );
            rid2out = rid2out( :, ok );
            rid2det = bsxfun( @minus, rid2det, [ sideMargin; sideMargin; ] );
        end
        function [ did2tlbr, did2out ] = detMultiRegns( this, rid2tlbr, im, dvecLength )
            % Set parameters.
            useGpu = isa( this.attNet.layers{ 1 }.weights{ 1 }, 'gpuArray' );
            inputSide = this.attNet.srcInOut.settingGeneral.dstSide;
            inputCh = this.attNet.srcInOut.settingGeneral.dstCh;
            numMaxFeed  = this.settingInitDet.numMaxTest;
            targetLyrId = numel( this.attNet.layers ) - 1;
            numRegn = size( rid2tlbr, 2 );
            % Detection on each region.
            im = single( im );
            rgbMean_ = this.rgbMean;
            if useGpu, rgbMean_ = gpuArray( rgbMean_ ); end;
            idx2im = zeros( inputSide, inputSide, inputCh, numRegn, 'single' );
            parfor r = 1 : numRegn,
                imRegn = im( ...
                    rid2tlbr( 1, r ) : rid2tlbr( 3, r ), ...
                    rid2tlbr( 2, r ) : rid2tlbr( 4, r ), : );
                idx2im( :, :, :, r ) = ...
                    imresize( imRegn, [ inputSide, inputSide ] );
            end;
            rid2out = zeros( this.signNoObj * 2, numRegn, 'single' );
            idx2rid = 1 : numRegn;
            for feed = 1 : numMaxFeed,
                numIm = size( idx2im, 4 );
                if ~numIm, break; end;
                fprintf( '%s: %dth feed. %d ims.\n', upper( mfilename ), feed, numIm );
                % Feed-forward.
                if useGpu, idx2im = gpuArray( single( idx2im ) ); end;
                idx2im = bsxfun( @minus, idx2im, rgbMean_ );
                idx2out = my_simplenn( ...
                    this.attNet, idx2im, [  ], [  ], ...
                    'accumulate', false, ...
                    'disableDropout', true, ...
                    'conserveMemory', true, ...
                    'backPropDepth', +inf, ...
                    'targetLayerId', targetLyrId, ...
                    'sync', true ); clear idx2im;
                idx2out = gather( idx2out( end - 1 ).x );
                idx2out = permute( idx2out, [ 3, 4, 1, 2 ] );
                rid2out( :, idx2rid ) = idx2out;
                [ ~, idx2predTl ] = max( idx2out( 1 : this.signNoObj, : ), [  ], 1 );
                [ ~, idx2predBr ] = max( idx2out( this.signNoObj + 1 : end, : ), [  ], 1 );
                % Leave regions to be continued.
                idx2stop = idx2predTl == this.signStop & idx2predBr == this.signStop;
                idx2false = idx2predTl == this.signNoObj & idx2predBr == this.signNoObj;
                idx2false = idx2false | ( idx2predTl == this.signNoObj & idx2predBr == this.signStop );
                idx2false = idx2false | ( idx2predTl == this.signStop & idx2predBr == this.signNoObj );
                rid2tlbr( :, idx2rid( idx2false ) ) = 0;
                idx2continue = ~( idx2stop | idx2false );
                numCont = sum( idx2continue );
                idx2rid = idx2rid( idx2continue );
                idx2predTl = idx2predTl( idx2continue );
                idx2predBr = idx2predBr( idx2continue );
                % Update regions.
                if feed == numMaxFeed, rid2tlbr( :, idx2rid ) = 0; break; end;
                if ~numCont, break; end;
                idx2predTl( idx2predTl == this.signNoObj ) = this.signStop;
                idx2predBr( idx2predBr == this.signNoObj ) = this.signStop;
                idx2tlbr = [ this.did2dvecTl( :, idx2predTl ) * dvecLength + 1; ...
                    this.did2dvecBr( :, idx2predBr ) * dvecLength + inputSide ];
                for idx = 1 : numCont,
                    rid = idx2rid( idx );
                    w = rid2tlbr( 4, rid ) - rid2tlbr( 2, rid ) + 1;
                    h = rid2tlbr( 3, rid ) - rid2tlbr( 1, rid ) + 1;
                    tlbr = idx2tlbr( :, idx );
                    tlbr = resizeTlbr( tlbr, [ inputSide, inputSide ], [ h, w ] );
                    rid2tlbr( :, rid ) = tlbr + [ rid2tlbr( 1 : 2, rid ); rid2tlbr( 1 : 2, rid ) ] - 1;
                end;
                rid2tlbr = round( rid2tlbr );
                % Preparing for next feed-forward.
                idx2im = zeros( inputSide, inputSide, inputCh, numCont, 'single' );
                parfor idx = 1 : numCont,
                    rid = idx2rid( idx );
                    imRegn = im( ...
                        rid2tlbr( 1, rid ) : rid2tlbr( 3, rid ), ...
                        rid2tlbr( 2, rid ) : rid2tlbr( 4, rid ), : );
                    idx2im( :, :, :, idx ) = ...
                        imresize( imRegn, [ inputSide, inputSide ] );
                end;
            end;  clear rgbMean_;
            rid2ok = ~sum( rid2tlbr, 1 );
            rid2tlbr( :, rid2ok ) = [  ];
            rid2out( :, rid2ok ) = [  ];
            did2tlbr = rid2tlbr;
            did2out = rid2out;
        end
        % Next task 1) Modify this.avgIm part.
        % Next task 2) Add scale/aspect selection option for further tuning.
        function [ rid2out, rid2tlbr, imGlobal, sideMargin ] = ...
                initGuess( this, im ) % This image is original.
            % Prepare settings and data.
            patchMargin = this.settingInitDet.patchMargin;
            numAspect = numel( this.aspects );
            numScale = numel( this.scales );
            interpolation = this.attNet.normalization.interpolation;
            lyid = numel( this.attNet.layers ) - 1;
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
                    if isa( this.attNet.layers{ 1 }.weights{ 1 }, 'gpuArray' ), 
                        im_ = gpuArray( im_ ); end;
                    res = my_simplenn( ...
                        this.attNet, im_, [  ], [  ], ...
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
        function determineInOutRelations( this, gpus )
            % Determine patch size.
            targetLyrId = numel( this.attNet.layers ) - 1;
            psize = this.attNet.srcInOut.settingTsDb.dstSide;
            ch = numel( this.rgbMean );
            while true,
                try
                    im = zeros( psize, psize, ch, 'single' );
                    if ~isempty( gpus ), im = gpuArray( im ); end;
                    my_simplenn( ...
                        this.attNet, im, [  ], [  ], ...
                        'targetLayerId', targetLyrId );
                    clear im; clear ans;
                    psize = psize - 1;
                catch
                    psize = psize + 1;
                    break;
                end;
            end
            % Determine patch stride.
            strd = 0;
            while true,
                im = zeros( psize + strd, psize, ch, 'single' );
                if ~isempty( gpus ), im = gpuArray( im ); end;
                res = my_simplenn( ...
                    this.attNet, im, [  ], [  ], ...
                    'targetLayerId', targetLyrId );
                desc = res( targetLyrId + 1 ).x;
                if size( desc, 1 ) == 2, break; end;
                strd = strd + 1;
                clear im; clear res;
            end;
            this.patchSize = psize;
            this.stride = strd;
        end
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'DET_%s_OF_%s', ...
                this.settingInitDet.changes, ...
                this.attNet.getNetName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
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
    methods( Static )
        function [ newdid2det, newdid2score ] = ...
                mergeDetBoxsByNms( ...
                did2det, ...
                did2score, ...
                overlap, ...
                minNumSuppBox, ...
                mergeType, ...
                scoreType )
            [ newdid2det, newdid2score, ~ ] = ...
                DetSingleCls.nms( ...
                [ did2det; did2score ]', ...
                overlap, ...
                minNumSuppBox, ...
                mergeType, ...
                scoreType );
            newdid2det = newdid2det';
        end
        function [ newdid2det, newdid2score ] = ...
                mergeDetBoxsByOvlp( ...
                did2det, ...
                did2score, ...
                overlap, ...
                minNumSuppBox, ...
                mergeType, ...
                scoreType )
            if isempty( did2det ), newdid2det = zeros( 4, 0 ); newdid2score = zeros( 0, 1 ); return; end;
            if size( did2det, 2 ) == 1, 
                if minNumSuppBox >= 1, newdid2det = zeros( 4, 0 ); newdid2score = zeros( 0, 1 ); return; end;
                newdid2det = did2det; newdid2score = did2score; return; 
            end;
            numCand = size( did2det, 2 );
            did2detRect = tlbr2rect( did2det );
            did2area = prod( did2detRect( 3 : 4, : ), 1 )';
            did2did2int = rectint( did2detRect', did2detRect' );
            did2did2uni = repmat( did2area, 1, numCand ) + ...
                repmat( did2area, 1, numCand )' - did2did2int;
            did2did2ov = did2did2int ./ did2did2uni;
            did2did2ov = triu( did2did2ov ) - eye( numCand );
            did2did2mrg = did2did2ov > overlap;
            [ newdid2did1, newdid2did2 ] = find( did2did2mrg );
            newdid2dids = [ newdid2did1, newdid2did2 ]';
            sdids = setdiff( 1 : numCand, unique( newdid2dids( : ) ) );
            did2newdid2is = false( numCand, numel( newdid2did1 ) + numel( sdids ) );
            did2newdid2is( ...
                [ newdid2did1', sdids ] + ...
                numCand * ( 0 : numel( newdid2did1 ) + numel( sdids ) - 1 ) ) = true;
            did2newdid2is( newdid2did2' + ...
                numCand * ( 0 : numel( newdid2did2 ) - 1 ) ) = true;
            for did = 1 : numCand,
                newdid2mrg = did2newdid2is( did, : );
                did2is = any( did2newdid2is( :, newdid2mrg ), 2 );
                did2newdid2is( :, newdid2mrg ) = [  ];
                did2newdid2is = [ did2is, did2newdid2is ];
            end;
            did2newdid2is = did2newdid2is...
                ( :, sum( did2newdid2is ) > minNumSuppBox );
            numDet = size( did2newdid2is, 2 );
            newdid2det = zeros( 4, numDet );
            newdid2score = zeros( numDet, 1 );
            for newdid = 1 : numDet,
                boxes = did2det( :, did2newdid2is( :, newdid ) );
                scores = did2score( did2newdid2is( :, newdid ) );
                switch scoreType,
                    case 'AVG',
                        score = mean( scores );
                    case 'MAX'
                        score = max( scores );
                    case 'SUM',
                        score = sum( scores );
                    case 'NUM',
                        score = numel( scores );
                end
                switch mergeType,
                    case 'WAVG',
                        box = sum( bsxfun( @times, boxes, scores ), 2 ) / ...
                            sum( scores );
                    case 'AVG',
                        box = mean( boxes, 2 );
                    case 'MAXSCORE',
                        [ ~, is ] = max( scores );
                        box = boxes( :, is );
                end
                newdid2det( :, newdid ) = box;
                newdid2score( newdid ) = score;
            end;
        end
        function [ outboxes, scores, assigns ] = nms( ...
                boxes, ...
                overlap, ...
                minNumSuppBox, ...
                mergeType, ...
                scoreType )
            if isempty( boxes ), outboxes = [  ]; scores = [  ]; assigns = [  ]; return; end;
            outboxes = zeros( size( boxes, 1 ), 4 );
            assigns = cell( size( boxes, 1 ), 1 );
            scores = zeros( size( boxes, 1 ), 1 );
            x1 = boxes( :, 1 );
            y1 = boxes( :, 2 );
            x2 = boxes( :, 3 );
            y2 = boxes( :, 4 );
            s = boxes( :, end );
            area = ( x2 - x1 + 1 ) .* ( y2 - y1 + 1 );
            [ ~, I ] = sort( s );
            pick = s * 0;
            counter = 1;
            while ~isempty( I )
                last = length( I );
                i = I( last );
                xx1 = max( x1( i ), x1( I( 1 : last - 1 ) ) );
                yy1 = max( y1( i ), y1( I( 1 : last - 1 ) ) );
                xx2 = min( x2( i ), x2( I( 1 : last - 1 ) ) );
                yy2 = min( y2( i ), y2( I( 1 : last - 1 ) ) );
                w = max( 0.0, xx2 - xx1 + 1 );
                h = max( 0.0, yy2 - yy1 + 1 );
                o = w .* h ./ area( I( 1 : last - 1 ) );
                supp = find( o > overlap );
                if numel(  supp  ) >= minNumSuppBox,
                    pick( counter ) = i;
                    idx = [ last; supp ];
                    switch mergeType,
                        case 'WAVG',
                            outboxes( counter, : ) = ...
                                sum( bsxfun( @times, boxes( I( idx ), 1 : 4 ), boxes( I( idx ), end ) ), 1 ) / ...
                                sum( boxes( I( idx ), end ) );
                        case 'AVG',
                            outboxes( counter, : ) = ...
                                mean( boxes( I( idx ), 1 : 4 ), 1 );
                        case 'MAXSCORE',
                            outboxes( counter, : ) = boxes( i, 1 : 4 );
                    end
                    ss = boxes( I( idx ), end );
                    switch scoreType,
                        case 'AVG',
                            scores( counter ) = mean( ss );
                        case 'MAX'
                            scores( counter ) = max( ss );
                        case 'SUM',
                            scores( counter ) = sum( ss );
                        case 'NUM',
                            scores( counter ) = numel( ss );
                    end
                    assigns{ counter } = I( idx );
                    I( idx ) = [  ];
                    counter = counter + 1;
                else
                    I( last ) = [  ];
                end
            end
            if counter == 1,
                outboxes = [  ]; scores = [  ]; assigns = [  ];
            else
                outboxes = outboxes( 1 : ( counter - 1 ), : );
                assigns = assigns( 1 : counter - 1 );
                scores = scores( 1 : counter - 1 );
            end
        end
        function rid2geo = ExtDevGrid( imSize, scaleStep, numScale )
            % Extract multi-scale regions from image layout.
            % IN
            %   imSize:     [ numRow, numCols ]
            %   numScale:   Number of sacles of the regions to be extracted.
            %   scaleStep:  Rate of area values of neighboring regions.
            % OUT
            %   rid2geo:    First row is the top left row index in the image domain.
            %               Second row is the top left column index in the image domain.
            %               Third row is the bottom right row index in the image domain.
            %               Fourth row is the bottom right column index in the image domain.
            %               Fifth row is the length of each region box when we consider small-side-length of the image as 1.
            flagAddWhole = false;
            [ minLen, minIdx ] = min( imSize( 1 : 2 ) );
            [ maxLen, maxIdx ] = max( imSize( 1 : 2 ) );
            if minIdx == maxIdx, minIdx = 1; maxIdx = 2; end;
            scales = ( scaleStep .^ ( 0 : -0.5 : ( -0.5 * ( numScale - 1 ) ) ) )';
            patchSize = minLen * scales;
            maxNumDiv = round( maxLen * 2 ./ patchSize );
            minNumDiv = round( minLen * 2 ./ patchSize );
            sidx2starts = cell( length( patchSize ), 1 );
            sidx2ends = cell( length( patchSize ), 1 );
            sidx2scales = cell( length( patchSize ), 1 );
            for sidx = 1 : length( patchSize )
                maxAxis = 1 : ( ( maxLen - 1 ) / maxNumDiv( sidx ) ) : maxLen;
                minAxis = 1 : ( ( minLen - 1 ) / minNumDiv( sidx ) ) : minLen;
                maxStart = cat( 2, maxAxis( 1 ), round( maxAxis( 2 : end - 2 ) ) + 1 );
                maxEnd = cat( 2, round( maxAxis( 3 : end - 1 ) ), maxAxis( end ) );
                minStart = cat( 2, minAxis( 1 ), round( minAxis( 2 : end - 2 ) ) + 1 );
                minEnd = cat( 2, round( minAxis( 3 : end - 1 ) ), minAxis( end ) );
                sidx2starts{ sidx } = cat( 1, ...
                    reshape( repmat( maxStart, length( minStart ), 1 ), [ 1, length( minStart ) * length( maxStart ) ] ), ...
                    repmat( minStart, 1, length( maxStart ) ) );
                sidx2ends{ sidx } = cat( 1, ...
                    reshape( repmat( maxEnd, length( minEnd ), 1 ), [ 1, length( minEnd ) * length( maxEnd ) ] ), ...
                    repmat( minEnd, 1, length( maxEnd ) ) );
                sidx2scales{ sidx } = sidx * ones( 1, size( sidx2ends{ sidx }, 2 ) );
                if sidx == 1 && size( sidx2ends{ sidx }, 2 ) > 1, flagAddWhole = true; end;
            end
            fid2tl = cat( 2, sidx2starts{ : } );
            fid2tl( [ maxIdx; minIdx ], : ) = fid2tl;
            fid2br = cat( 2, sidx2ends{ : } );
            fid2br( [ maxIdx; minIdx ], : ) = fid2br;
            fid2s = cat( 2, sidx2scales{ : } );
            rid2geo = cat( 1, fid2tl, fid2br, fid2s );
            if flagAddWhole, 
                rid2geo( end, : ) = rid2geo( end, : ) + 1; 
                rid2geo = [ [ 1; 1; imSize( 1 ); imSize( 2 ); 1; ], rid2geo ];
            end;
        end
    end
end

