classdef AttNetCaffe < handle
    properties
        db;
        attNet;
        attNetName;
        rgbMean;
        patchSide;
        inputSide;
        stride;
        weights;
        biases;
        scales;
        directions;
        settingProp;
        settingDet0;
        settingMrg0;
        settingDet1;
        settingMrg1;
    end
    methods( Access = public )
        function this = AttNetCaffe...
                ( db, settingProp, settingDet0, settingMrg0, settingDet1, settingMrg1 )
            this.db                                   = db;
            this.settingProp.numBaseProposal          = 0;
            this.settingProp.normalizeImageMaxSide    = 500;
            this.settingProp.numScaling               = 12;
            this.settingProp.dilate                   = 1 / 2;
            this.settingProp.posIntOverRegnMoreThan   = 1 / 8;
            this.settingProp.maximumImageSize         = 9e6;
            this.settingProp.numTopClassification     = 1;
            this.settingProp.numTopDirection          = 1;
            this.settingProp.directionVectorSize      = 30;
            this.settingDet0.type                     = 'DYNAMIC';
            this.settingDet0.rescaleBox               = 1;
            this.settingDet0.numTopClassification     = 1;          % Ignored if 'STATIC'.
            this.settingDet0.numTopDirection          = 1;          % Ignored if 'STATIC'.
            this.settingDet0.directionVectorSize      = 30;
            this.settingMrg0.mergingOverlap           = 0.8;
            this.settingMrg0.mergingType              = 'OV';
            this.settingMrg0.mergingMethod            = 'WAVG';
            this.settingMrg0.minimumNumSupportBox     = 1;          % Ignored if mergingOverlap = 1.
            this.settingDet1.type                     = 'DYNAMIC';
            this.settingDet1.rescaleBox               = 2.5;
            this.settingDet1.numTopClassification     = 1;          % Ignored if 'STATIC'.
            this.settingDet1.numTopDirection          = 1;          % Ignored if 'STATIC'.
            this.settingDet1.directionVectorSize      = 30;
            this.settingMrg1.mergingOverlap           = 0.6;
            this.settingMrg1.mergingType              = 'OV';
            this.settingMrg1.mergingMethod            = 'WAVG';
            this.settingMrg1.minimumNumSupportBox     = 0;          % Ignored if mergingOverlap = 1.
            this.settingProp = setChanges...
                ( this.settingProp, settingProp, upper( mfilename ) );
            this.settingDet0 = setChanges...
                ( this.settingDet0, settingDet0, upper( mfilename ) );
            this.settingMrg0 = setChanges...
                ( this.settingMrg0, settingMrg0, upper( mfilename ) );
            this.settingDet1 = setChanges...
                ( this.settingDet1, settingDet1, upper( mfilename ) );
            this.settingMrg1 = setChanges...
                ( this.settingMrg1, settingMrg1, upper( mfilename ) );
        end
        function init( this, netInfo, gpus )
            % 1. Fetch Caffe parameters of output layer on GPU.
            fprintf( '%s: Fetch Caffe parameters of output layer on GPU.\n', upper( mfilename ) );
            this.attNetName = netInfo.modelName;
            this.patchSide = netInfo.patchSide;
            this.inputSide = netInfo.inputSide;
            this.stride = netInfo.stride;
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
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % 2. Fetch Caffe net on GPU without the output layer.
            fprintf( '%s: Fetch Caffe net on GPU without the output layer..\n', upper( mfilename ) );
            caffe.reset_all(  );
            caffe.set_mode_gpu(  );
            caffe.set_device( gpus - 1 );
            this.attNet = caffe.Net( netInfo.protoPathTest, netInfo.modelPath, 'test' );
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % 3. Determine scaling factors.
            maxSide = this.settingProp.normalizeImageMaxSide;
            numScaling = this.settingProp.numScaling;
            posIntOverRegnMoreThan = this.settingProp.posIntOverRegnMoreThan;
            fpath = this.getScalesPath;
            try
                fprintf( '%s: Try to load scaling factors.\n', upper( mfilename ) );
                data = load( fpath );
                this.scales = data.data.scales;
            catch
                fprintf( '%s: Determine scaling factors.\n', ...
                    upper( mfilename ) );
                setid = 2;
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == setid );
                if maxSide,
                    oid2iid = this.db.oid2iid( this.db.iid2setid( this.db.oid2iid ) == setid );
                    oid2imsize = this.db.iid2size( :, oid2iid );
                    numRegn = size( oid2tlbr, 2 );
                    for oid = 1 : numRegn,
                        [ ~, oid2tlbr( :, oid ) ] = normalizeImageSize...
                            ( maxSide, oid2imsize( :, oid ), oid2tlbr( :, oid ) );
                    end;
                end;
                referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr, numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                fprintf( '%s: Done.\n', upper( mfilename ) );
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % 4. Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            numDirection = 3;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.directions.did2vecTl = [ [ cos( did2angTl' ); sin( did2angTl' ); ], [ 0; 0; ] ];
            this.directions.did2vecBr = [ [ cos( did2angBr' ); sin( did2angBr' ); ], [ 0; 0; ] ];
            fprintf( '%s: Done.\n', upper( mfilename ) );
        end
        function [ rid2tlbr, nid2rid, nid2cid ] = iid2prop( this, iid, cidx2cid )
            fpath = this.getPropPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                nid2rid = data.nid2rid;
                nid2cid = data.nid2cid;
            catch
                [ rid2tlbr, nid2rid, nid2cid ] = this.iid2propWrapper( iid, cidx2cid );
                this.makePropDir;
                save( fpath, 'rid2tlbr', 'nid2rid', 'nid2cid' );
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det0( this, iid, cidx2cid )
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2score = data.rid2score;
                rid2cid = data.rid2cid;
            catch
                % 1. Get regions.
                [ rid2tlbr, nid2rid, nid2cid ] = this.iid2prop( iid, cidx2cid );
                % 2. Tighten regions.
                [ rid2tlbr, rid2score, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2score', 'rid2cid' );
            end;
            if nargout,
                % 3. Merge regions.
                [ rid2tlbr, rid2score, rid2cid ] = this.merge...
                    ( rid2tlbr, rid2score, rid2cid, this.settingMrg0 );
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det1( this, iid, cidx2cid )
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2score = data.rid2score;
                rid2cid = data.rid2cid;
            catch
                % 1. Get regions.
                [ rid2tlbr, ~, rid2cid ] = this.iid2det0( iid, cidx2cid );
                nid2rid = 1 : numel( rid2cid );
                nid2cid = rid2cid;
                % 2. Tighten regions.
                [ rid2tlbr, rid2score, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2score', 'rid2cid' );
            end;
            if nargout,
                % 3. Merge regions.
                [ rid2tlbr, rid2score, rid2cid ] = this.merge...
                    ( rid2tlbr, rid2score, rid2cid, this.settingMrg1 );
            end;
        end
        function demoDet( this, iid, cidx2cid )
            im = imread( this.db.iid2impath{ iid } );
            % Demo 1: proposals.
            [ rid2tlbr, nid2rid, nid2cid ] = this.iid2prop( iid, cidx2cid );
            rid2tlbr = round( rid2tlbr );
            figure; set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Proposals, IID%06d', iid ) ); hold off;
            setFigPos( gcf, [ 3, 6, 1, 2 ] ); drawnow;
            % Demo 2: detection0.
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2score = data.rid2score;
                rid2cid = data.rid2cid;
            catch
                [ rid2tlbr, rid2score, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2score', 'rid2cid' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure; set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection0, IID%06d', iid ) ); hold off;
            setFigPos( gcf, [ 3, 6, 1, 3 ] ); drawnow;
            % Demo 3: Merge0.
            [ rid2tlbr, ~, rid2cid ] = this.merge...
                ( rid2tlbr, rid2score, rid2cid, this.settingMrg0 );
            rid2tlbr = round( rid2tlbr );
            figure; set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, 'c', this.db.cid2name( rid2cid ) );
            title( sprintf( 'Merge0, IID%06d', iid ) ); hold off;
            setFigPos( gcf, [ 3, 6, 1, 4 ] ); drawnow;
            % Demo 4: detection1.
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2score = data.rid2score;
                rid2cid = data.rid2cid;
            catch
                nid2rid = 1 : numel( rid2cid );
                nid2cid = rid2cid;
                [ rid2tlbr, rid2score, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2score', 'rid2cid' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure; set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection1, IID%06d', iid ) ); hold off;
            setFigPos( gcf, [ 3, 6, 1, 5 ] ); drawnow;
            % Demo 5: Merge1.
            [ rid2tlbr, ~, rid2cid ] = this.merge...
                ( rid2tlbr, rid2score, rid2cid, this.settingMrg1 );
            rid2tlbr = round( rid2tlbr );
            figure; set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, 'c', this.db.cid2name( rid2cid ) );
            title( sprintf( 'Merge1, IID%06d', iid ) ); hold off;
            setFigPos( gcf, [ 3, 6, 1, 6 ] ); drawnow;
        end
        function subDbDet0( this, cidx2cid, numDiv, divId )
            iids = this.db.getTeiids;
            numIm = numel( iids );
            divSize = ceil( numIm / numDiv );
            sidx = divSize * ( divId - 1 ) + 1;
            eidx = min( sidx + divSize - 1, numIm );
            iids = iids( sidx : eidx );
            fprintf( '%s: Check if detections exist.\n', upper( mfilename ) );
            paths = arrayfun( @( iid )this.getDet0Path( iid ), iids, 'UniformOutput', false );
            exists = cellfun( @( path )exist( path, 'file' ), paths );
            this.makeDet0Dir;
            iids = iids( ~exists );
            numIm = numel( iids );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2det0( iid, cidx2cid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Det0 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
        end
    end
    methods( Access = private )
        function [ rid2tlbr, nid2rid, nid2cid ] = iid2propWrapper( this, iid, cidx2cid )
            % Initial guess.
            cidx2cid = cidx2cid( : )';
            im = imread( this.db.iid2impath{ iid } );
            [ rid2out, rid2tlbr ] = this.initGuess( im, cidx2cid );
            % Compute each region score.
            dvecSize = this.settingProp.directionVectorSize;
            numTopCls = this.settingProp.numTopClassification;
            numTopDir = this.settingProp.numTopDirection;
            signStop = 4;
            signDiag = 2;
            numTarCls = numel( cidx2cid );
            numDimPerDirLyr = 4;
            numDimClsLyr = numTarCls + 1;
            dimCls = numTarCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            rid2outCls = rid2out( dimCls, : );
            [ ~, rid2rank2cidx ] = sort( rid2outCls, 1, 'descend' );
            rid2tlbrProp = cell( numTarCls, 1 );
            for cidx = 1 : numTarCls,
                % Direction: DD condition.
                dimTl = ( cidx - 1 ) * numDimPerDirLyr * 2 + 1;
                dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                dimBr = dimTl + numDimPerDirLyr;
                rid2outTl = rid2out( dimTl, : );
                rid2outBr = rid2out( dimBr, : );
                [ ~, rid2rank2ptl ] = sort( rid2outTl, 1, 'descend' );
                [ ~, rid2rank2pbr ] = sort( rid2outBr, 1, 'descend' );
                rid2ptl = rid2rank2ptl( 1, : );
                rid2pbr = rid2rank2pbr( 1, : );
                rid2okTl = any( rid2rank2ptl( 1 : numTopDir, : ) == signDiag, 1 );
                rid2okBr = any( rid2rank2pbr( 1 : numTopDir, : ) == signDiag, 1 );
                rid2ss = rid2ptl == signStop & rid2pbr == signStop;
                rid2dd = rid2okTl & rid2okBr;
                rid2dd = rid2dd & ( ~rid2ss );
                rid2dd = rid2dd & ( rid2ptl == signDiag | rid2pbr == signDiag );
                % Classification.
                rid2bgd = rid2rank2cidx( 1, : ) == ( numTarCls + 1 );
                rid2okCls = any( rid2rank2cidx( 1 : min( numTopCls, numDimClsLyr ), : ) == cidx, 1 );
                rid2okCls = rid2okCls & ( ~rid2bgd );
                % Update.
                rid2cont = rid2dd & rid2okCls;
                numCont = sum( rid2cont );
                if ~numCont, continue; end;
                idx2tlbr = rid2tlbr( 1 : 4, rid2cont );
                idx2ptl = rid2ptl( rid2cont );
                idx2pbr = rid2pbr( rid2cont );
                idx2tlbrWarp = [ ...
                    this.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                    this.directions.did2vecBr( :, idx2pbr ) * dvecSize + this.patchSide; ];
                for idx = 1 : numCont,
                    w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                    h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                    tlbrWarp = idx2tlbrWarp( :, idx );
                    tlbr = resizeTlbr( tlbrWarp, [ this.patchSide, this.patchSide ], [ h, w ] );
                    idx2tlbr( :, idx ) = tlbr - 1 + ...
                        [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                end;
                idx2tlbr = cat( 1, idx2tlbr, cidx2cid( cidx ) * ones( 1, numCont ) );
                rid2tlbrProp{ cidx } = idx2tlbr;
            end;
            rid2tlbr = round( cat( 2, rid2tlbrProp{ : } ) );
            [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
            nid2cid = rid2tlbr( 5, : )';
            rid2tlbr = rid2tlbr_';
            % Add base proposals.
            numAdd = this.settingProp.numBaseProposal;
            if numAdd,
                numProp = size( rid2tlbr, 2 );
                [ r, c, ~ ] = size( im );
                scaling = sqrt( 2 .^ ( -1 : ( -1 + numAdd - 1 ) ) );
                newProp = round( scaleBoxes( repmat( [ 1; 1; r; c; ], 1, numAdd ), scaling, scaling ) );
                rid2tlbr = cat( 2, rid2tlbr, newProp );
                nid2cid = cat( 1, nid2cid, repmat( cidx2cid', numAdd, 1 ) );
                newRids = repmat( numProp + ( 1 : numAdd ), numTarCls, 1 );
                nid2rid = cat( 1, nid2rid, newRids( : ) );
            end;
            if isempty( rid2tlbr ),
                rid2tlbr = zeros( 4, 0 );
                nid2rid = zeros( 0, 1 );
                nid2cid = zeros( 0, 1 );
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det...
                ( this, iid, rid2tlbr0, nid2rid0, nid2cid0, detParams )
            if isempty( rid2tlbr0 ),
                rid2tlbr = zeros( 4, 0 );
                rid2score = zeros( 0, 1 );
                rid2cid = zeros( 0, 1 );
                return;
            end;
            % Pre-processing: box re-scaling.
            detType = detParams.type;
            rescaleBox = detParams.rescaleBox;
            rid2tlbr0 = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            rid2tlbr0 = round( rid2tlbr0 );
            % Do detection on each region.
            interpolation = 'bilinear';
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0( 1 : 4, : ) = bsxfun( @minus, rid2tlbr0( 1 : 4, : ), [ imTl; imTl; ] ) + 1;
            im = imread( this.db.iid2impath{ iid } );
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], this.rgbMean, interpolation );
            switch detType,
                case 'STATIC',
                    [ rid2tlbr, rid2score, rid2cid ] = this.staticFitting...
                        ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
                case 'DYNAMIC',
                    [ rid2tlbr, rid2score, rid2cid ] = this.dynamicFitting...
                        ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
            end;
            if isempty( rid2tlbr ),
                rid2tlbr = zeros( 4, 0 );
                rid2score = zeros( 0, 1 );
                rid2cid = zeros( 0, 1 );
                return;
            end;
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
        end
        function [ rid2out, rid2tlbr ] = initGuess( this, im, cidx2cid )
            dilate = this.settingProp.dilate;
            maxSide = this.settingProp.normalizeImageMaxSide;
            maximumImageSize = this.settingProp.maximumImageSize;
            [ r, c, ~ ] = size( im );
            imSize0 = [ r; c; ];
            if maxSide, imSize = normalizeImageSize( maxSide, imSize0 ); else imSize = imSize0; end;
            sid2size = round( bsxfun( @times, this.scales, imSize ) );
            rid2tlbr = extractDenseRegions...
                ( imSize, sid2size, this.patchSide, this.stride, dilate, maximumImageSize );
            rid2tlbr = round( resizeTlbr( rid2tlbr, imSize, imSize0 ) );
            rid2out = this.extractDenseActivationsCaffe( im, cidx2cid, sid2size );
            if size( rid2out, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
        end
        function rid2out = extractDenseActivationsCaffe...
                ( this, originalImage, cidx2cid, targetImageSizes )
            regionDilate = this.settingProp.dilate;
            maximumImageSize = this.settingProp.maximumImageSize;
            imageDilate = round( this.patchSide * regionDilate );
            interpolation = 'bilinear';
            numSize = size( targetImageSizes, 2 );
            rid2out = cell( numSize, 1 );
            for sid = 1 : numSize,
                imSize = targetImageSizes( :, sid );
                if min( imSize ) + 2 * imageDilate < this.patchSide, continue; end;
                if prod( imSize + imageDilate * 2 ) > maximumImageSize,
                    fprintf( '%s: Warning) Im of %s rejected.\n', ...
                        upper( mfilename ), mat2str( imSize ) ); continue;
                end;
                im = imresize( ...
                    originalImage, imSize', ...
                    'method', interpolation );
                im = single( im );
                roi = [ ...
                    1 - imageDilate; ...
                    1 - imageDilate; ...
                    imSize( : ) + imageDilate; ];
                im = normalizeAndCropImage( im, roi, this.rgbMean, interpolation );
                fprintf( '%s: Feed im of %dX%d size.\n', ...
                    upper( mfilename ), size( im, 1 ), size( im, 2 ) );
                y = this.feedforwardCaffe( im, cidx2cid );
                [ nr, nc, z ] = size( y );
                y = reshape( permute( y, [ 3, 1, 2 ] ), z, nr * nc );
                rid2out{ sid } = y;
            end;
            rid2out = cat( 2, rid2out{ : } );
        end
        function y = feedforwardCaffe( this, im, cidx2cid )
            cidx2cid = cidx2cid( : );
            numCls = this.db.getNumClass;
            targetDimDir = bsxfun( @plus, repmat( ( cidx2cid' - 1 ) * 4 * 2, 4 * 2, 1 ), ( 1 : ( 4 * 2 ) )' );
            targetDimCls = [ cidx2cid; numCls + 1; ] + 4 * 2 * numCls;
            targetDim = [ targetDimDir( : ); targetDimCls; ];
            weight = this.weights( :, :, :, targetDim );
            bias = this.biases( :, targetDim );
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
            y = vl_nnconv( x, weight, bias, 'pad', 0, 'stride', 1 );
            y = gather( y );
            clear x;
        end
        function [ did2tlbr, did2score, did2cid ] = dynamicFitting...
                ( this, rid2tlbr, nid2rid, nid2cid, im, detParams )
            % Preparing for data.
            numTopCls = detParams.numTopClassification;
            numTopDir = detParams.numTopDirection;
            dvecSize = detParams.directionVectorSize;
            testBatchSize = 256 / 2;
            numMaxFeed = 50;
            interpolation = 'bilinear';
            inputCh = size( im, 3 );
            cidx2cid = unique( nid2cid );
            numTarCls = numel( cidx2cid );
            numDimPerDirLyr = 4;
            numDimClsLyr = numTarCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numTarCls + numDimClsLyr;
            dimCls = numTarCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            signStop = numDimPerDirLyr;
            signDiag = 2;
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
                for r = 1 : testBatchSize : numRegn,
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
                    brid2out = this.feedforwardCaffe( brid2im, cidx2cid );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                nrid2tlbr = cell( numTarCls, 1 );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2rank2cidx ] = sort( rid2outCls, 1, 'descend' );
                rid2cidx = rid2rank2cidx( 1, : );
                for cidx = 1 : numTarCls,
                    cid = cidx2cid( cidx );
                    dimTl = ( cidx - 1 ) * numDimPerDirLyr * 2 + 1;
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
                    rid2bgd = rid2cidx == ( numTarCls + 1 );
                    rid2high = any( rid2rank2cidx( 1 : min( numTopCls, numDimClsLyr ), : ) == cidx, 1 );
                    rid2high = rid2high & ( ~rid2bgd );
                    rid2top = rid2cidx == cidx;
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
                    didx2scoreTl = didx2outTl( signStop, : );
                    didx2scoreBr = didx2outBr( signStop, : );
                    didx2scoreCls = didx2outCls( cidx, : );
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
                    nrid2tlbr{ cidx } = idx2tlbr;
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
        function [ did2tlbr, did2score, did2cid ] = staticFitting...
                ( this, rid2tlbr, nid2rid, nid2cid, im, detParams )
            % Preparing for data.
            dvecSize = detParams.directionVectorSize;
            testBatchSize = 256 / 2;
            numMaxFeed = 50;
            interpolation = 'bilinear';
            inputCh = size( im, 3 );
            cidx2cid = unique( nid2cid );
            numTarCls = numel( cidx2cid );
            numDimPerDirLyr = 4;
            numDimClsLyr = numTarCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numTarCls + numDimClsLyr;
            dimCls = numTarCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            signStop = numDimPerDirLyr;
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
                for r = 1 : testBatchSize : numRegn,
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
                    brid2out = this.feedforwardCaffe( brid2im, cidx2cid );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                nrid2tlbr = cell( numTarCls, 1 );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2cidx ] = max( rid2outCls, [  ], 1 );
                for cidx = 1 : numTarCls,
                    cid = cidx2cid( cidx );
                    rid2tar = false( size( nid2rid ) );
                    rid2tar( nid2rid( nid2cid == cid ) ) = true;
                    if ~sum( rid2tar ), continue; end;
                    crid2tlbr = rid2tlbr(  :, rid2tar );
                    crid2out = rid2out( :, rid2tar );
                    crid2outCls = rid2outCls( :, rid2tar );
                    crid2cidx = rid2cidx( rid2tar );
                    crid2fgd = crid2cidx ~= ( numTarCls + 1 );
                    dimTl = ( cidx - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    crid2outTl = crid2out( dimTl, : );
                    crid2outBr = crid2out( dimBr, : );
                    [ ~, crid2ptl ] = max( crid2outTl, [  ], 1 );
                    [ ~, crid2pbr ] = max( crid2outBr, [  ], 1 );
                    crid2ss = crid2ptl == signStop & crid2pbr == signStop;
                    crid2scoreTl = crid2outTl( signStop, : );
                    crid2scoreBr = crid2outBr( signStop, : );
                    crid2scoreCls = crid2outCls( cidx, : );
                    crid2score = ( crid2scoreTl + crid2scoreBr ) / 2 + crid2scoreCls;
                    % Find and store detections.
                    crid2det = crid2ss & crid2fgd; % Add more conditions!!!
                    numDet = sum( crid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = crid2tlbr( :, crid2det );
                    did2score( dids ) = crid2score( crid2det );
                    did2cid( dids ) = cid;
                    did2fill( dids ) = true;
                    did = did + numDet;
                    % Find and store regiones to be continued.
                    crid2cont = ~crid2det & ~crid2ss;
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functions for file identification %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 1. Scales file.
        function name = getScalesName( this )
            numScaling = this.settingProp.numScaling;
            maxSide = this.settingProp.normalizeImageMaxSide;
            piormt = this.settingProp.posIntOverRegnMoreThan;
            piormt = num2str( piormt );
            piormt( piormt == '.' ) = 'P';
            name = sprintf( 'SCL_TE_N%03d_PIORMT%s_NIMS%d_OF_%s', ...
                numScaling, piormt, maxSide, this.db.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getScalesDir( this )
            dir = this.db.getDir;
        end
        function dir = makeScalesDir( this )
            dir = this.getScalesDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getScalesPath( this )
            fname = strcat( this.getScalesName, '.mat' );
            path = fullfile( this.getScalesDir, fname );
        end
        % 2. Proposal file.
        function name = getPropName( this )
            name = sprintf( 'ANETCAF_PROP_%s_OF_%s', ...
                this.settingProp.changes, this.attNetName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getPropDir( this )
            name = this.getPropName;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'ANETCAF_PROP_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makePropDir( this )
            dir = this.getPropDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPropPath( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getPropDir, fname );
        end
        % 3. Detection0 file.
        function name = getDet0Name( this )
            name = sprintf( 'ANETCAF_DET0_%s_OF_%s', ...
                this.settingDet0.changes, this.getPropName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDet0Dir( this )
            name = this.getDet0Name;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'ANETCAF_DET0_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makeDet0Dir( this )
            dir = this.getDet0Dir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDet0Path( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getDet0Dir, fname );
        end
        % 4. Detection1 file.
        function name = getDet1Name( this )
            name = sprintf( 'ANETCAF_DET1_%s_OF_%s_OF_%s', ...
                this.settingDet1.changes, this.settingMrg0.changes, this.getDet0Name );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDet1Dir( this )
            name = this.getDet1Name;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'ANETCAF_DET1_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makeDet1Dir( this )
            dir = this.getDet1Dir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDet1Path( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getDet1Dir, fname );
        end
    end
    methods( Static )
       function [ rid2tlbr, rid2score, rid2cid ] = merge...
               ( rid2tlbr, rid2score, rid2cid, mrgParams )
            mergingOverlap = mrgParams.mergingOverlap;
            mergingType = mrgParams.mergingType;
            mergingMethod = mrgParams.mergingMethod;
            minNumSuppBox = mrgParams.minimumNumSupportBox;
            if mergingOverlap == 1, return; end;
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
    end
end