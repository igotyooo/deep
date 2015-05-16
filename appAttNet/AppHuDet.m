classdef AppHuDet < handle
    properties
        srcDb;
        srcCnn;
        avgIm;
        did2dvecTl;
        did2dvecBr;
        signStop;
        signNoObj;
        refineId;
        did2iid;
        did2bbox;
        did2score;
        settingInitDet;
        settingInitMrg;
        settingRefine;
    end
    methods( Access = public )
        function this = AppHuDet( ...
                srcDb, ...
                srcCnn, ...
                settingInitDet, ...
                settingInitMrg, ...
                settingRefine )
            this.srcDb = srcDb;
            this.srcCnn = srcCnn;
            this.refineId = 0;
            this.settingInitDet.scaleStep       = 2;
            this.settingInitDet.numScale        = 6;
            this.settingInitDet.dvecLength      = 30;
            this.settingInitDet.numMaxTest      = 50;
            this.settingInitDet.scaleMag        = 4;
            this.settingInitDet.startHorzScale	= 1;
            this.settingInitDet.horzScaleStep   = 0.5;
            this.settingInitDet.endHorzScale    = 2;
            this.settingInitMrg.method          = 'OV'; % 'NMS'
            this.settingInitMrg.overlap         = 0.7;
            this.settingInitMrg.minNumSuppBox   = 1;
            this.settingInitMrg.mergeType       = 'WAVG';
            this.settingInitMrg.scoreType       = 'AVG';
            this.settingRefine.scaleMag         = 2.5;
            this.settingRefine.method           = 'OV'; % 'NMS'
            this.settingRefine.overlap          = 0.4;
            this.settingRefine.minNumSuppBox    = 0;
            this.settingRefine.mergeType        = 'WAVG';
            this.settingRefine.scoreType        = 'AVG';
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
        function init( this )
            dvecLength = this.settingInitDet.dvecLength;
            this.avgIm = this.srcCnn.avgIm;
            this.did2dvecTl = this.srcCnn.srcInOut.did2dvecTl * dvecLength;
            this.did2dvecBr = this.srcCnn.srcInOut.did2dvecBr * dvecLength;
            this.did2dvecTl  = round( cat( 2, this.did2dvecTl, [ 0; 0; ] ) );
            this.did2dvecBr  = round( cat( 2, this.did2dvecBr, [ 0; 0; ] ) );
            this.signStop = this.srcCnn.srcInOut.signStop;
            this.signNoObj = this.srcCnn.srcInOut.signNoObj;
        end
        function detDb( this )
            fpath = this.getDetResPath( 0 );
            try
                fprintf( '%s: Try to load det res.\n', upper( mfilename ) ); 
                data = load( fpath );
                res = data.res;
                fprintf( '%s: Done.\n', upper( mfilename ) );
            catch
                iids = this.srcDb.getTeiids;
                fprintf( '%s: Check if human detections exist.\n', ...
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
                    this.iid2detInit( iid );
                    cummt = cummt + toc( itime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        'Init det.', cummt );
                end
                iids = this.srcDb.getTeiids;
                numIm = numel( iids );
                res.did2iid = cell( numIm, 1 );
                res.did2bbox = cell( numIm, 1 );
                res.did2score = cell( numIm, 1 );
                cummt = 0;
                for iidx = 1 : numIm; itime = tic;
                    iid = iids( iidx );
                    [ res.did2bbox{ iidx }, res.did2score{ iidx } ] = ...
                        this.iid2det( iid );
                    if isempty( res.did2bbox{ iidx } ), continue; end;
                    res.did2iid{ iidx } = iid * ones( size( res.did2bbox{ iidx }, 2 ), 1 );
                    cummt = cummt + toc( itime );
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        'Det.', cummt );
                end
                res.did2iid = cat( 1, res.did2iid{ : } );
                res.did2bbox = cat( 2, res.did2bbox{ : } );
                res.did2score = cat( 1, res.did2score{ : } );
                fprintf( '%s: Save det results.\n', upper( mfilename ) );
                save( fpath, 'res' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end
            this.did2iid = res.did2iid;
            this.did2bbox = res.did2bbox;
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
                scaleMag = this.settingRefine.scaleMag;
                method = this.settingRefine.method;
                overlap = this.settingRefine.overlap;
                minNumSuppBox = this.settingRefine.minNumSuppBox;
                mergeType = this.settingRefine.mergeType;
                scoreType = this.settingRefine.scoreType;
                iidx2iid = this.srcDb.getTeiids;
                numIm = numel( iidx2iid );
                res.did2iid = cell( numIm, 1 );
                res.did2bbox = cell( numIm, 1 );
                res.did2score = cell( numIm, 1 );
                cnt = 0; cummt = 0;
                for iidx = 1 : numIm; itime = tic;
                	iid = iidx2iid( iidx );
                    dids = this.did2iid == iid;
                    did2det = this.did2bbox( :, dids );
                    if ~isempty( did2det );
                        % Make re-initilized bounding boxes.
                        im = imread( this.srcDb.iid2impath{ iid } );
                        [ nr, nc, ~ ] = size( im );
                        did2size = tlbr2rect( did2det );
                        did2size = did2size( [ 4, 3 ], : );
                        did2tsize = did2size * sqrt( scaleMag );
                        did2center = [ sum( did2det( [ 1, 3 ], : ), 1 ); sum( did2det( [ 2, 4 ], : ), 1 ) ] / 2;
                        did2reinit = zeros( size( did2det ) );
                        did2reinit( 1 : 2, : ) = did2center - round( did2tsize / 2 );
                        did2reinit( 3 : 4, : ) = did2center + round( did2tsize / 2 );
                        did2reinit = round( bndtlbr( did2reinit, [ 1; 1; nr; nc; ] ) );
                        % Restart detection.
                        numDet = size( did2reinit, 2 );
                        did2redet = zeros( 4, numDet );
                        did2reout = zeros( this.signNoObj * 2, numDet );
                        did2ok = true( numDet, 1 );
                        for did = 1 : numDet,
                            regn = did2reinit( :, did );
                            imRegn = im( regn( 1 ) : regn( 3 ), regn( 2 ) : regn( 4 ), : );
                            [ det, out ] = this.detSingleHuman...
                                ( imRegn, false );
                            if isempty( det ), did2ok( did ) = false; continue; end;
                            det = det + [ regn( 1 : 2 ); regn( 1 : 2 ); ] - 1;
                            did2redet( :, did ) = det;
                            did2reout( :, did ) = out;
                        end
                        did2redet = did2redet( :, did2ok );
                        did2reout = did2reout( :, did2ok );
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
                        res.did2iid{ iidx } = iid * ones( size( did2redet, 2 ), 1 );
                        res.did2score{ iidx } = did2rescore;
                        res.did2bbox{ iidx } = did2redet;
                    end;
                    cummt = cummt + toc( itime );
                    cnt = cnt + 1;
                    fprintf( '%s: ', upper( mfilename ) );
                    disploop( numIm, iidx, ...
                        sprintf( 'Det. (Refine %d)', refineId ), cummt );
                end
                res.did2iid = cat( 1, res.did2iid{ : } );
                res.did2bbox = cat( 2, res.did2bbox{ : } );
                res.did2score = cat( 1, res.did2score{ : } );
                fprintf( '%s: Save det results. (Refine %d)\n', upper( mfilename ), refineId ); 
                save( fpath, 'res' );
                fprintf( '%s: Done.\n', upper( mfilename ) ); 
            end
            this.did2iid = res.did2iid;
            this.did2bbox = res.did2bbox;
            this.did2score = res.did2score;
            this.refineId = refineId;
        end
        function [ ap, rank2iid, rank2bbox, rank2tp, rank2fp ] = ...
                computeAp( this, addrss )
            % Set parameters.
            clsName = this.srcCnn.srcInOut.settingTsDb.selectClassName;
            minoverlap = 0.5;
            minDetArea = 500;
            clsId = cellfun( ...
                @( cname )strcmp( cname, clsName ), ...
                this.srcDb.cid2name );
            clsId = find( clsId );
            % Prepare data.
            iidx2iid = this.srcDb.getTeiids;
            numTeIm = numel( iidx2iid );
            oid2target = this.srcDb.oid2cid == clsId;
            % Get ground-truth.
            iidx2oidx2bbox = cell( numTeIm, 1 );
            iidx2oidx2diff = cell( numTeIm, 1 );
            iidx2oidx2det = cell( numTeIm, 1 );
            numPos = 0;
            fprintf( '%s: Compute gt.\n', upper( mfilename ) );
            for iidx = 1 : numTeIm;
                iid = iidx2iid( iidx );
                oidx2oid = ( this.srcDb.oid2iid == iid ) & oid2target;
                iidx2oidx2bbox{ iidx } = this.srcDb.oid2bbox( :, oidx2oid );
                iidx2oidx2diff{ iidx } = this.srcDb.oid2diff( oidx2oid );
                iidx2oidx2det{ iidx } = false( sum( oidx2oid ), 1 );
                numPos = numPos + sum( ~iidx2oidx2diff{ iidx } );
            end
            % Cut detection results by minimum area.
            did2rect = tlbr2rect( this.did2bbox );
            did2area = prod( did2rect( 3 : 4, : ), 1 );
            did2ok = did2area >= minDetArea;
            did2score_ = this.did2score( did2ok );
            did2iid_ = this.did2iid( did2ok );
            did2bbox_ = this.did2bbox( :, did2ok );
            % Sort detection results.
            [ ~, rank2did ] = sort( - did2score_ );
            rank2iid = did2iid_( rank2did );
            rank2bbox = did2bbox_( :, rank2did );
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
        function [ rid2det, rid2out, rid2regn ] = ...
                iid2detInit( this, iid )
            fpath = this.getPath( iid );
            im = imread( this.srcDb.iid2impath{ iid } );
            [ r, c, ~ ] = size( im );
            try
                data = load( fpath );
                rid2regn = data.det.rid2regn;
                rid2det = data.det.rid2det;
                rid2out = data.det.rid2out;
            catch
                startHorzScale = this.settingInitDet.startHorzScale;
                horzScaleStep = this.settingInitDet.horzScaleStep;
                endHorzScale = this.settingInitDet.endHorzScale;
                % Image pre-processing.
                horzScales = startHorzScale : horzScaleStep : endHorzScale;
                numHorzScale = numel( horzScales );
                hid2rid2regn = cell( numHorzScale, 1 );
                hid2rid2det = cell( numHorzScale, 1 );
                hid2rid2out = cell( numHorzScale, 1 );
                for h = 1 : numHorzScale;
                    hs = horzScales( h );
                    sourceSize = [ r, c ];
                    targetSize = [ r, c * hs ];
                    im_ = imresize( im, targetSize );
                    [ rid2regn, rid2det, rid2out ] = ...
                        this.detSingleHumanDensely( im_ );
                    hid2rid2regn{ h } = resizeTlbr( rid2regn, targetSize, sourceSize );
                    hid2rid2det{ h } = resizeTlbr( rid2det, targetSize, sourceSize );
                    hid2rid2out{ h } = rid2out;
                end;
                rid2regn = cat( 2, hid2rid2regn{ : } );
                rid2det = cat( 2, hid2rid2det{ : } );
                rid2out = cat( 2, hid2rid2out{ : } );
                det.rid2regn = rid2regn;
                det.rid2det = rid2det;
                det.rid2out = rid2out;
                save( fpath, 'det' );
            end
        end
        function [ did2det, did2score ] = ...
                iid2det( this, iid )
            % Initial detection.
            [ did2det, did2out, ~ ] = ...
                this.iid2detInit( iid );
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
            end
        end
        function rid2geo = extSubRegions( this, imSize )
            scaleStep = this.settingInitDet.scaleStep;
            numScale = this.settingInitDet.numScale;
            rid2geo = this.ExtDevGrid( imSize, scaleStep, numScale );
        end
        function [ rid2regn, rid2det, rid2out ] = ...
                detSingleHumanDensely( this, im )
            scaleMag = this.settingInitDet.scaleMag;
            imSize = size( im ); imSize = imSize( 1 : 2 );
            imTargetSize = round( imSize * sqrt( scaleMag ) );
            center = round( imTargetSize / 2 );
            imTl = center - round( imSize / 2 ) + 1;
            imBr = imTl + imSize - 1;
            imBgd = imresize( uint8( gather( this.srcCnn.avgIm ) ), imTargetSize );
            imBgd( imTl( 1 ) : imBr( 1 ), imTl( 2 ) : imBr( 2 ), : ) = im;
            im = imBgd;
            % Pre-filtering by initial guess.
            [ rid2pred, rid2regn ] = this.initGuess( im );
            outLyrDim = size( rid2pred, 1 ) / 2;
            predsTl = rid2pred( 1 : outLyrDim, : );
            predsBr = rid2pred( outLyrDim + 1 : end, : );
            [ ~, predsTl ] = max( predsTl, [  ], 1 );
            [ ~, predsBr ] = max( predsBr, [  ], 1 );
            rid2ok = predsTl == 2 & predsBr == 2;
            rid2regn = rid2regn( :, rid2ok );
            % Detection on each region.
            numRegn = size( rid2regn, 2 );
            rid2det = zeros( 4, numRegn );
            rid2out = zeros( outLyrDim * 2, numRegn );
            rid2ok = true( numRegn, 1 );
            for rid = 1 : numRegn,
                regn = rid2regn( :, rid );
                imRegn = im( regn( 1 ) : regn( 3 ), regn( 2 ) : regn( 4 ), : );
                [ det, score ] = this.detSingleHuman...
                    ( imRegn, false );
                if isempty( det ), rid2ok( rid ) = false; continue; end;
                det = det + [ regn( 1 : 2 ); regn( 1 : 2 ); ] - 1;
                rid2det( :, rid ) = det;
                rid2out( :, rid ) = score;
            end
            rid2regn = rid2regn( :, rid2ok );
            rid2det = rid2det( :, rid2ok );
            rid2out = rid2out( :, rid2ok );
            % Convert to original image domain.
            bnd = [ imTl( : ); imBr( : ); ];
            [ rid2det, ok ] = bndtlbr( rid2det, bnd );
            rid2regn = rid2regn( :, ok );
            rid2out = rid2out( :, ok );
            [ rid2regn, ~ ] = bndtlbr( rid2regn, bnd );
            rid2regn( 1 : 4, : ) = bsxfun( ...
                @minus, ...
                rid2regn( 1 : 4, : ), ...
                [ imTl( : ); imTl( : ); ] ) + 1;
            rid2det = bsxfun( ...
                @minus, ...
                rid2det, ...
                [ imTl( : ); imTl( : ); ] ) + 1;
        end
        function [ rid2pred, rid2regn ] = initGuess( this, im )
            % Prepare settings and data.
            lyid            = numel( this.srcCnn.layers ) - 1;
            scaleStep       = this.settingInitDet.scaleStep;
            numScale        = this.settingInitDet.numScale;
            dstSide         = this.srcCnn.srcInOut.settingGeneral.dstSide;
            useGpu          = isa( this.srcCnn.layers{ 1 }.filters, 'gpuArray' );
            stride = 1;
            numLyr = numel( this.srcCnn.layers );
            for lid = 1 : numLyr,
                lyr = this.srcCnn.layers{ lid };
                if isfield( lyr, 'stride' ), 
                    stride = stride * lyr.stride( 1 ); end; 
            end;
            % Do the job.
            imSize = size( im )';
            dstImSize = imSize( 1 : 2 ) / min( imSize( 1 : 2 ) ) * dstSide;
            sid2imsize = dstImSize * ...
                ( scaleStep .^ ( 0 : 0.5 : 0.5 * ( numScale - 1 ) ) );
            sid2imsize = round( sid2imsize );
            sid2tlbrs = cell( numScale, 1 );
            sid2preds = cell( numScale, 1 );
            for sid = 1 : numScale
                imRowSize = sid2imsize( 1, sid );
                imColSize = sid2imsize( 2, sid );
                avgImRe = imresize( gather( this.avgIm ), ...
                    [ imRowSize, imColSize ] );
                imRe = imresize( im, ...
                    [ imRowSize, imColSize ] );
                imRe = single( imRe ) - avgImRe;
                try
                    if useGpu, imRe = gpuArray( imRe ); end;
                    res = my_simplenn...
                        ( this.srcCnn, imRe, [  ], [  ], ...
                        'disableDropout', true, ...
                        'conserveMemory', true, ...
                        'extOnly', true, ...
                        'targetLayerId', lyid, ...
                        'sync', true ); clear imRe;
                catch
                    continue;
                end
                res = gather( res( lyid + 1 ).x );
                [ r, c, z ] = size( res );
                % Form cnn predictions.
                depthid = ( ( ( 1 : z ) - 1 ) * ( r * c ) )';
                rcid = 1 : ( r * c );
                [ depthid, rcid ] = meshgrid( depthid, rcid );
                idx = reshape( ( depthid + rcid )', z * r * c, 1 );
                rid2desc = reshape( res( idx ), z, r * c );
                sid2preds{ sid } = rid2desc; clear res;
                % Form geometries.
                [ cs, rs ] = meshgrid( 1 : c, 1 : r );
                geo = cat( 3, rs, cs, rs, cs, dstSide * ones( r, c ) );
                [ r, c, z ] = size( geo );
                depthid = ( ( ( 1 : z ) - 1 ) * ( r * c ) )';
                rcid = 1 : ( r * c );
                [ depthid, rcid ] = meshgrid( depthid, rcid );
                idx = reshape( ( depthid + rcid )', z * r * c, 1 );
                geo = reshape( geo( idx ), z, r * c );
                geo( 1 : 2, : ) = ( geo( 1 : 2, : ) - 1 ) * stride + 1;
                geo( 3 : 4, : ) = geo( 1 : 2, : ) + dstSide - 1;
                geo = resizeTlbr( geo( 1 : 4, : ), sid2imsize( :, sid ), imSize( 1 : 2 ) );
                geo( 1 : 2, : ) = max( 1, geo( 1 : 2, : ) );
                geo( 3, : ) = min( imSize( 1 ), geo( 3, : ) );
                geo( 4, : ) = min( imSize( 2 ), geo( 4, : ) );
                sid2tlbrs{ sid } = cat( 1, ...
                    round( geo( 1 : 4, : ) ), ...
                    sid * ones( 1, size( geo, 2 ) ) );
            end % Next scale.
            rid2pred = cat( 2, sid2preds{ : } );
            rid2regn = cat( 2, sid2tlbrs{ : } );
        end
        function [ tlbr, pred ] = detSingleHuman...
                ( this, im, getIntermediateRect )
            numMaxFeed  = this.settingInitDet.numMaxTest;
            dstSide     = this.srcCnn.srcInOut.settingGeneral.dstSide;
            imin = gpuArray( single( im ) );
            srcSize = size( im );
            targetLyrId = numel( this.srcCnn.layers ) - 1;
            fid2tlbr = zeros( 4, numMaxFeed );
            for feed = 1 : numMaxFeed;
                imin = myimresize( imin, dstSide );
                imin_ = bsxfun( @minus, imin, this.avgIm );
                res = my_simplenn...
                    ( this.srcCnn, imin_, [  ], [  ], ...
                    'disableDropout', true, ...
                    'conserveMemory', true, ...
                    'extOnly', true, ...
                    'targetLayerId', targetLyrId, ...
                    'sync', true ); clear imin_;
                res = gather( res( end - 1 ).x( : ) );
                outTl = res( 1 : this.signNoObj );
                outBr = res( this.signNoObj + 1 : end );
                [ ~, predTl ] = max( outTl );
                [ ~, predBr ] = max( outBr );
                humanDetected = predTl == this.signStop && predBr == this.signStop;
                noHuman = predTl == this.signNoObj && predBr == this.signNoObj;
                if humanDetected, break; end;
                if noHuman, tlbr = [  ]; pred = [  ]; return; end;
                if predTl == this.signNoObj, predTl = this.signStop; end;
                if predBr == this.signNoObj, predBr = this.signStop; end;
                dvecTl = this.did2dvecTl( :, predTl );
                dvecBr = this.did2dvecBr( :, predBr );
                tlbr = [ 1; 1; dstSide; dstSide; ] + [ dvecTl; dvecBr ];
                imin = imin( tlbr( 1 ) : tlbr( 3 ), tlbr( 2 ) : tlbr( 4 ), : );
                fid2tlbr( :, feed ) = tlbr;
            end
            if ~humanDetected, tlbr = [  ]; pred = [  ]; return; end;
            pred = [ outTl; outBr; ];
            if feed == 1, tlbr = [ 1; 1; srcSize( 1 : 2 )'; ]; return; end;
            fid2tlbr = fid2tlbr( :, 1 : feed - 1 );
            if getIntermediateRect,
                fid2rectproj = cell( feed - 1, 1 );
                for f = 1 : feed - 1
                    fid2rectproj{ f } = this.backProj...
                        ( fid2tlbr( :, 1 : f ), [ dstSide, dstSide ], srcSize( 1 : 2 ) );
                end
                tlbr = cat( 2, fid2rectproj{ : } );
            else
                tlbr = this.backProj( fid2tlbr, [ dstSide, dstSide ], srcSize( 1 : 2 ) );
            end
        end
        function reportTestResult...
                ( this, ap, addrss )
            title = sprintf( '%s: TEST REPORT', ...
                upper( mfilename ) );
            mssg = {  };
            mssg{ end + 1 } = '___________';
            mssg{ end + 1 } = 'TEST REPORT';
            mssg{ end + 1 } = sprintf( 'DATABASE: %s', ...
                this.srcDb.dbName );
            mssg{ end + 1 } = sprintf( 'TARGET CLASS: %s', ...
                upper( this.srcCnn.srcInOut.settingTsDb.selectClassName ) );
            mssg{ end + 1 } = sprintf( 'INOUT: %s', ...
                this.srcCnn.srcInOut.getName );
            mssg{ end + 1 } = sprintf( 'CNN: %s', ...
                this.srcCnn.getCnnDirName );
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
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'HUDET_%s_OF_%s', ...
                this.settingInitDet.changes, ...
                this.srcCnn.getCnnName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
            trDbName = this.srcCnn.srcInOut.srcDb.dbName;
            if ~strcmp( this.srcDb.dbName, trDbName )
                name = strcat( name, '_', trDbName );
            end
        end
        function dir = getDir( this )
            name = this.getName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'HUDET_', name );
            end
            dir = fullfile...
                ( this.srcDb.dir, name );
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
        function tlbr = backProj( fid2tlbr, srcDomainSize, dstDomainSize )
            if size( fid2tlbr, 2 ) > 1,
                tlbr = AppHuDet.backProjBbox( fid2tlbr, srcDomainSize );
            else
                tlbr = fid2tlbr;
            end
            tlbr = resizeTlbr( tlbr, srcDomainSize, dstDomainSize );
        end
        function tlbrProj = backProjBbox( fid2tlbr, lastDomainSize )
            tlbrCurr = fid2tlbr( :, end );
            tlbrPrev = fid2tlbr( :, end - 1 );
            srcDomainSize = lastDomainSize;
            dstDomainSize = [ tlbrPrev( 3 ) - tlbrPrev( 1 ), tlbrPrev( 4 ) - tlbrPrev( 2 ) ] + 1;
            tlbrProj = resizeTlbr( tlbrCurr, srcDomainSize, dstDomainSize );
            tlbrProj = tlbrProj + [ tlbrPrev( 1 : 2 ); tlbrPrev( 1 : 2 ) ] - 1;
            fid2tlbr( :, end ) = [  ];
            fid2tlbr( :, end ) = tlbrProj;
            if size( fid2tlbr, 2 ) == 1,
                return;
            else
                tlbrProj = AppHuDet.backProjBbox( fid2tlbr, lastDomainSize );
            end
        end
        function [ newdid2det, newdid2score ] = ...
                mergeDetBoxsByNms( ...
                did2det, ...
                did2score, ...
                overlap, ...
                minNumSuppBox, ...
                mergeType, ...
                scoreType )
            [ newdid2det, newdid2score, ~ ] = ...
                AppHuDet.nms( ...
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

