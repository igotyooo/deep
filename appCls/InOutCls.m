% IN:  DB.
% OUT: Images-GT paies per epoch.
% Task-specific implementation required.
classdef InOutCls < handle
    properties
        % General properties.
        srcDb;                  % A general db.
        tsDb;                   % A task specific db to be made. If it is unnecessary, just fetch srcDb.
        numBchTr;               % Number of training batches in an epoch.
        numBchVal;              % Number of validation batches in an epoch.
        poolTr;                 % Training sample pool, where all samples are used up in an epoch.
        poolVal;                % Validation sample pool, where all samples are used up in an epoch.
        poolRemainTr;           % Remaining training sample pool. It is initilized for each epoch of CNN training.
        poolRemainVal;          % Remaining validation sample pool. It is initilized for each epoch of CNN training.
        tsMetricName;           % Name of task specific evaluation metric;
        settingGeneral;         % Setting for image processing.
        settingTsDb;            % Setting for the task specific db.
        settingTsAug;           % Setting for the task specific augmentation.
        % Task-specific properties.
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public interface. Cnn will be trained with the following functions only. %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = public )
        function this = InOutCls( srcDb, settingGeneral, settingTsDb, settingTsAug )
            this.srcDb = srcDb;
            this.tsMetricName = 'Top-1 err';
            % General) Default parameters to provide batches.
            this.settingGeneral.dstImSide               = 227;
            this.settingGeneral.numImChannel            = 3;
            this.settingGeneral.batchSize               = 256;
            this.settingGeneral.normalizeByAverageImage = true;
            % Task specific) Default parameters for task specific db.
            % Task specific) Default parameters for task specific augmentation.
            this.settingTsAug.interpolationMethod       = 'bicubic';
            this.settingTsAug.srcImSide                 = 256; % Set it to dstImSide for no crop-augmentation.
            this.settingTsAug.rotation                  = 0;
            this.settingTsAug.keepAspectRate            = true;
            % Apply user setting.
            this.settingGeneral = setChanges...
                ( this.settingGeneral, settingGeneral, upper( mfilename ) );
            this.settingTsDb = setChanges...
                ( this.settingTsDb, settingTsDb, upper( mfilename ) );
            this.settingTsAug = setChanges...
                ( this.settingTsAug, settingTsAug, upper( mfilename ) );
        end
        % Prepare for all data to be used.
        function init( this )
            % Set Params.
            batchSize = this.settingGeneral.batchSize;
            % Reform the general source db format to a task-specific format.
            this.makeTsDb;
            % Make training pool to be consumed in an epoch.
            [   this.poolTr.sid2iid, ...
                this.poolTr.sid2regn, ...
                this.poolTr.sid2cid ] = this.getRegnSeqInEpch( 1 );
            this.numBchTr = numel( this.poolTr.sid2iid ) / batchSize;
            % Make validation pool to be consumed in an epoch.
            [   this.poolVal.sid2iid, ...
                this.poolVal.sid2regn, ...
                this.poolVal.sid2cid ] = this.getRegnSeqInEpch( 2 );
            this.numBchVal = numel( this.poolVal.sid2iid ) / batchSize;
        end
        % Majorly used in CNN. Provide a tr/val batch of I/O pairs.
        function [ ims, gts ] = provdBchTr( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolTr.sid2iid )
                % Make training pool to be consumed in an epoch.
                [   this.poolTr.sid2iid, ...
                    this.poolTr.sid2regn, ...
                    this.poolTr.sid2cid ] = this.getRegnSeqInEpch( 1 );
                this.numBchTr = numel( this.poolTr.sid2iid ) / batchSize;
            end
            batchSmpl = ( 1 : batchSize )';
            sid2iid = this.poolTr.sid2iid( batchSmpl );
            sid2regn = this.poolTr.sid2regn( :, batchSmpl );
            sid2cid = this.poolTr.sid2cid( :, batchSmpl );
            this.poolTr.sid2iid( batchSmpl ) = [  ];
            this.poolTr.sid2regn( :, batchSmpl ) = [  ];
            this.poolTr.sid2cid( :, batchSmpl ) = [  ];
            iid2impath = this.tsDb.tr.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2regn, sid2cid );
        end
        function [ ims, gts ] = provdBchVal( this )
            batchSize = this.settingGeneral.batchSize;
            if isempty( this.poolVal.sid2iid )
                % Make validation pool to be consumed in an epoch.
                [   this.poolVal.sid2iid, ...
                    this.poolVal.sid2regn, ...
                    this.poolVal.sid2cid ] = this.getRegnSeqInEpch( 2 );
                this.numBchVal = numel( this.poolVal.sid2iid ) / batchSize;
            end
            batchSmpl = ( 1 : batchSize )';
            sid2iid = this.poolVal.sid2iid( batchSmpl );
            sid2regn = this.poolVal.sid2regn( :, batchSmpl );
            sid2cid = this.poolVal.sid2cid( :, batchSmpl );
            this.poolVal.sid2iid( batchSmpl ) = [  ];
            this.poolVal.sid2regn( :, batchSmpl ) = [  ];
            this.poolVal.sid2cid( :, batchSmpl ) = [  ];
            iid2impath = this.tsDb.val.iid2impath;
            [ ims, gts ] = this.makeImGtPairs...
                ( iid2impath, sid2iid, sid2regn, sid2cid );
        end
        % Functions to provide information.
        function batchSize = getBatchSize( this )
            batchSize = this.settingGeneral.batchSize;
        end
        function numBchTr = getNumBatchTr( this )
            numBchTr = this.numBchTr;
        end
        function numBchVal = getNumBatchVal( this )
            numBchVal = this.numBchVal;
        end
        function imsize = getImSize( this )
            dstImSide = this.settingGeneral.dstImSide;
            dstCh = this.settingGeneral.dstCh;
            imsize = [ dstImSide; dstImSide; dstCh; ];
        end
        function tsMetricName = getTsMetricName( this )
            tsMetricName = this.tsMetricName;
        end
        % Function for identification.
        function name = getName( this )
            name = sprintf( 'IODETSINGLECLS_%s_OF_%s', ...
                this.settingGeneral.changes, ...
                this.getTsDbName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Private interface. %
    %%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        function [ ims, gts ] = makeImGtPairs...
                ( this, iid2impath, sid2iid, sid2regn, sid2cid )
            % Set Params.
            dstImSide = this.settingGeneral.dstImSide;
            dstCh = this.settingGeneral.dstCh;
            % Read images.
            [ idx2iid, ~, sid2idx ] = unique( sid2iid );
            idx2impath = iid2impath( idx2iid );
            idx2im = vl_imreadjpeg( idx2impath, 'numThreads', 8 );
            % Do the job.
            numSmpl = numel( sid2iid );
            sid2im = gpuArray( zeros( dstImSide, dstImSide, dstCh, numSmpl, 'single' ) );
            for sid = 1 : numSmpl;
                % Prepare data.
                wind = sid2regn( :, sid );
                im = idx2im{ sid2idx( sid ) };
                im = im( wind( 1 ) : wind( 3 ), wind( 2 ) : wind( 4 ), : );
                if dstCh == 1 && size( im, 3 ) == 3, im = mean( im, 3 ); end;
                if dstCh == 3 && size( im, 3 ) == 1, im = cat( 3, im, im, im ); end;
                if sid2regn( 5, sid ) == 2, im = fliplr( im ); end;
                sid2im( :, :, :, sid ) = myimresize( gpuArray( im ), dstImSide );
                % sid2im( :, :, :, sid ) = imresize...
                %     ( im, [ dstImSide, dstImSide ], interpolationMethod );
            end
            ims = sid2im;
            % Merge ground-truths into a same shape of cnn output.
            gts = reshape( sid2cid, [ 1, 1, size( sid2cid, 1 ), size( sid2cid, 2 ) ] );
        end
        function [ sid2iid, sid2regn, sid2cid ] = ...
                getRegnSeqInEpch( this, setid )
            dstImSide = this.settingGeneral.dstImSide;
            srcImSide = this.settingTsAug.srcImSide;
            rotation = this.settingTsAug.rotation;
            keepAspectRate = this.settingTsAug.keepAspectRate;
            sid2iid = find( this.srcDb.iid2setid == setid );
            sid2size = this.srcDb.iid2size( :, sid2iid );
            
            rng( 'shuffle' );
            
            
            
            
            
            
            
            
            
            
            
            % Supply more samples for suitable number of batchs.
            numSmpl = numel( sid2iid );
            numSupp = batchSize * ceil( numSmpl / batchSize ) - numSmpl;
            suppSids = randsample( numSmpl, numSupp );
            sid2iid = cat( 1, sid2iid, sid2iid( suppSids ) );
            sid2regn = cat( 2, sid2regn, sid2regn( :, suppSids ) );
            sid2cid = cat( 2, sid2cid, sid2cid( :, suppSids ) );
            % Shupple.
            numSmpl = numel( sid2iid );
            perm = randperm( numSmpl );
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
            sid2cid = sid2cid( :, perm );
        end
        function makeTsDb( this )
            this.tsDb = this.srcDb;
        end
        function subDb = makeSubTsDb( this, setid )
            targetClsName = this.settingTsAug.selectClassName;
            stride = this.settingTsAug.stride;
            dstImSide = this.settingTsAug.dstImSide;
            numScale = this.settingTsAug.numScale;
            scaleStep = this.settingTsAug.scaleStep;
            startHorzScale = this.settingTsAug.startHorzScale;
            horzScaleStep = this.settingTsAug.horzScaleStep;
            endHorzScale = this.settingTsAug.endHorzScale;
            startVertScale = this.settingTsAug.startVertScale;
            vertScaleStep = this.settingTsAug.vertScaleStep;
            endVertScale = this.settingTsAug.endVertScale;
            insectOverFgdObj = this.settingTsAug.insectOverFgdObj;
            insectOverFgdObjForMajority = this.settingTsAug.insectOverFgdObjForMajority;
            fgdObjMajority = this.settingTsAug.fgdObjMajority;
            insectOverBgdObj = this.settingTsAug.insectOverBgdObj;
            insectOverBgdRegn = this.settingTsAug.insectOverBgdRegn;
            insectOverBgdRegnForReject = this.settingTsAug.insectOverBgdRegnForReject;
            numMaxBgdRegnPerScale = this.settingTsAug.numMaxBgdRegnPerScale;
            stopSignError = this.settingTsAug.stopSignError;
            minObjScale = this.settingTsAug.minObjScale;
            numErode = this.settingTsAug.numErode;
            % Do the job.
            targetClsId = cellfun( @( cname )strcmp( cname, targetClsName ), this.srcDb.cid2name );
            targetClsId = find( targetClsId );
            idx2iid = find( this.srcDb.iid2setid == setid );
            numIm = numel( idx2iid );
            [ didsTl, didsBr ] = meshgrid( 1 : this.signStop, 1 : this.signStop );
            dpid2dids = single( [ didsTl( : ), didsBr( : ) ]' );
            numDirPair = size( dpid2dids, 2 );
            iid2impath = this.srcDb.iid2impath( idx2iid );
            fgdObj.oid2iid = cell( numIm, 1 );
            fgdObj.oid2dpid2regnsSqr = cell( numIm, 1 );
            fgdObj.oid2dpid2regnsNsqr = cell( numIm, 1 );
            bgdObj.oid2iid = cell( numIm, 1 );
            bgdObj.oid2regnsSqr = cell( numIm, 1 );
            bgdObj.oid2regnsNsqr = cell( numIm, 1 );
            bgd.iid2sid2regnsSqr = cell( numIm, numScale );
            bgd.iid2sid2regnsNsqr = cell( numIm, numScale );
            cummt = 0;
            for newiid = 1 : numIm;
                itime = tic;
                iid = idx2iid( newiid );
                oidx2oid = find( this.srcDb.oid2iid == iid );
                oidx2cid = this.srcDb.oid2cid( oidx2oid );
                oidx2fgdObj = oidx2cid == targetClsId;
                oidx2bgdObj = ~oidx2fgdObj;
                oidx2tlbr = this.srcDb.oid2bbox( :, oidx2oid );
                oidx2tlbr = single( round( oidx2tlbr ) );
                imSize = this.srcDb.iid2size( :, iid );
                [ rid2tlbrFgdObj, rid2oidxFgdObj, rid2tlbrBgd, rid2oidxBgd ] = ...
                    extTargetObjRegns( ...
                    oidx2tlbr, ...
                    oidx2fgdObj, ...
                    imSize, ...
                    stride, ...
                    dstImSide, ...
                    numScale, ...
                    scaleStep, ...
                    startHorzScale, ...
                    horzScaleStep, ...
                    endHorzScale, ...
                    startVertScale, ...
                    vertScaleStep, ...
                    endVertScale, ...
                    insectOverFgdObj, ...
                    insectOverFgdObjForMajority, ...
                    fgdObjMajority, ...
                    insectOverBgdObj, ...
                    insectOverBgdRegn, ...
                    insectOverBgdRegnForReject );
                rid2bgdObj = logical( rid2oidxBgd );
                rid2oidxBgdObj = rid2oidxBgd( rid2bgdObj );
                rid2oidxBgdObj = rid2oidxBgdObj( : );
                rid2tlbrBgdObj = rid2tlbrBgd( :, rid2bgdObj );
                rid2tlbrBgd( :, rid2bgdObj ) = [  ];
                rid2tlbrFgdObj = single( round( rid2tlbrFgdObj ) );
                rid2tlbrBgdObj = single( round( rid2tlbrBgdObj ) );
                rid2tlbrBgd = single( round( rid2tlbrBgd ) );
                % Fore-ground objects.
                rid2fgdObjTlbr = oidx2tlbr( 1 : 4, rid2oidxFgdObj );
                rid2outOfBndTlr = ( rid2tlbrFgdObj( 1, : ) - rid2fgdObjTlbr( 1, : ) ) > 0;
                rid2outOfBndTlc = ( rid2tlbrFgdObj( 2, : ) - rid2fgdObjTlbr( 2, : ) ) > 0;
                rid2outOfBndBrr = ( rid2tlbrFgdObj( 3, : ) - rid2fgdObjTlbr( 3, : ) ) < 0;
                rid2outOfBndBrc = ( rid2tlbrFgdObj( 4, : ) - rid2fgdObjTlbr( 4, : ) ) < 0;
                rid2fgdObjTlbr( 1, rid2outOfBndTlr ) = rid2tlbrFgdObj( 1, rid2outOfBndTlr );
                rid2fgdObjTlbr( 2, rid2outOfBndTlc ) = rid2tlbrFgdObj( 2, rid2outOfBndTlc );
                rid2fgdObjTlbr( 3, rid2outOfBndBrr ) = rid2tlbrFgdObj( 3, rid2outOfBndBrr );
                rid2fgdObjTlbr( 4, rid2outOfBndBrc ) = rid2tlbrFgdObj( 4, rid2outOfBndBrc );
                rid2fgdObjTlbr( 1 : 2, : ) = rid2fgdObjTlbr( 1 : 2, : ) - rid2tlbrFgdObj( 1 : 2, : ) + 1;
                rid2fgdObjTlbr( 3 : 4, : ) = rid2fgdObjTlbr( 3 : 4, : ) - rid2tlbrFgdObj( 1 : 2, : ) + 1;
                slop = bsxfun( @times, [ dstImSide; dstImSide ] - 1, 1 ./ ( rid2tlbrFgdObj( 3 : 4, : ) - rid2tlbrFgdObj( 1 : 2, : ) ) );
                rid2fgdObjTlbr( [ 1, 3 ], : ) = bsxfun( @times, slop( 1, : ), ( rid2fgdObjTlbr( [ 1, 3 ], : ) - 1 ) ) + 1;
                rid2fgdObjTlbr( [ 2, 4 ], : ) = bsxfun( @times, slop( 2, : ), ( rid2fgdObjTlbr( [ 2, 4 ], : ) - 1 ) ) + 1;
                rid2fgdObjTlbr = round( rid2fgdObjTlbr );
                rid2dvecTl = rid2fgdObjTlbr( 1 : 2, : ) - 1;
                rid2dvecNormTl = sqrt( sum( rid2dvecTl .^ 2, 1 ) );
                rid2dvecNormTl( rid2dvecNormTl < stopSignError ) = 0;
                rid2dvecTl = bsxfun( @times, rid2dvecTl, 1./ rid2dvecNormTl );
                rid2stopTl = any( isnan( rid2dvecTl ), 1 );
                [ ~, rid2didTl ] = max( this.did2dvecTl' * rid2dvecTl, [  ], 1 );
                rid2didTl( rid2stopTl ) = this.signStop;
                rid2dvecBr = bsxfun( @minus, rid2fgdObjTlbr( 3 : 4, : ), [ dstImSide; dstImSide; ] );
                rid2dvecNormBr = sqrt( sum( rid2dvecBr .^ 2, 1 ) );
                rid2dvecNormBr( rid2dvecNormBr < stopSignError ) = 0;
                rid2dvecBr = bsxfun( @times, rid2dvecBr, 1./ rid2dvecNormBr );
                rid2stopBr = any( isnan( rid2dvecBr ), 1 );
                [ ~, rid2didBr ] = max( this.did2dvecBr' * rid2dvecBr, [  ], 1 );
                rid2didBr( rid2stopBr ) = this.signStop;
                rid2dids = [ rid2didTl; rid2didBr; ];
                rid2dpid = zeros( size( rid2dids, 2 ), 1 );
                for dpid = 1 : numDirPair,
                    rid2dpid( logical( prod( bsxfun( @minus, rid2dids, dpid2dids( :, dpid ) ) == 0, 1 ) ) ) = dpid;
                end
                rid2fgdObjTlbrHF = rid2fgdObjTlbr;
                rid2fgdObjTlbrHF( [ 2; 4; ], : ) = dstImSide - rid2fgdObjTlbrHF( [ 4; 2; ], : ) + 1;
                rid2dvecTlHF = rid2fgdObjTlbrHF( 1 : 2, : ) - 1;
                rid2dvecNormTlHF = sqrt( sum( rid2dvecTlHF .^ 2, 1 ) );
                rid2dvecNormTlHF( rid2dvecNormTlHF < stopSignError ) = 0;
                rid2dvecTlHF = bsxfun( @times, rid2dvecTlHF, 1./ rid2dvecNormTlHF );
                rid2stopTlHF = any( isnan( rid2dvecTlHF ), 1 );
                [ ~, rid2didTlHF ] = max( this.did2dvecTl' * rid2dvecTlHF, [  ], 1 );
                rid2didTlHF( rid2stopTlHF ) = this.signStop;
                rid2dvecBrHF = bsxfun( @minus, rid2fgdObjTlbrHF( 3 : 4, : ), [ dstImSide; dstImSide; ] );
                rid2dvecNormBrHF = sqrt( sum( rid2dvecBrHF .^ 2, 1 ) );
                rid2dvecNormBrHF( rid2dvecNormBrHF < stopSignError ) = 0;
                rid2dvecBrHF = bsxfun( @times, rid2dvecBrHF, 1./ rid2dvecNormBrHF );
                rid2stopBrHF = any( isnan( rid2dvecBrHF ), 1 );
                [ ~, rid2didBrHF ] = max( this.did2dvecBr' * rid2dvecBrHF, [  ], 1 );
                rid2didBrHF( rid2stopBrHF ) = this.signStop;
                rid2didsHF = [ rid2didTlHF; rid2didBrHF; ];
                rid2dpidHF = zeros( size( rid2didsHF, 2 ), 1 );
                for dpid = 1 : numDirPair,
                    rid2dpidHF( logical( prod( bsxfun( @minus, rid2didsHF, dpid2dids( :, dpid ) ) == 0, 1 ) ) ) = dpid;
                end
                fgdObj.oid2iid{ newiid } = newiid * ones( sum( oidx2fgdObj ), 1, 'single' );
                % Fore-ground objects: square regions.
                rid2sqrFgdObj = rid2tlbrFgdObj( 6, : ) == 1;
                rid2sqrFgdObj = rid2sqrFgdObj & ( rid2tlbrFgdObj( 7, : ) == 1 );
                rid2dpidSqr = [ rid2dpid( rid2sqrFgdObj ), rid2dpidHF( rid2sqrFgdObj ) ];
                rid2tlbr = rid2tlbrFgdObj( 1 : 4, rid2sqrFgdObj );
                rid2tlbr = [ [ rid2tlbr; ones( 1, size( rid2tlbr, 2 ) ) ], ...
                    [ rid2tlbr; 2 * ones( 1, size( rid2tlbr, 2 ) ) ] ];
                rid2oidxFgdObjSqr = [ rid2oidxFgdObj( rid2sqrFgdObj ), ...
                    rid2oidxFgdObj( rid2sqrFgdObj ) ];
                if sum( oidx2fgdObj )
                    fgdObj.oid2dpid2regnsSqr{ newiid } = cell( sum( oidx2fgdObj ), 1 );
                end
                oidxsFgdObj = find( oidx2fgdObj );
                for oidxFgd = 1 : sum( oidx2fgdObj ),
                    oidxFgdObj = oidxsFgdObj( oidxFgd );
                    rids = find( rid2oidxFgdObjSqr == oidxFgdObj );
                    dpids = rid2dpidSqr( rids );
                    fgdObj.oid2dpid2regnsSqr{ newiid }{ oidxFgd } = arrayfun( ...
                        @( dpid )rid2tlbr( :, rids( dpids == dpid ) ), ...
                        1 : numDirPair, 'UniformOutput', false );
                end
                % Fore-ground objects: non-square regions.
                rid2nsqrFgdObj = ~rid2sqrFgdObj;
                rid2dpidNsqr = [ rid2dpid( rid2nsqrFgdObj ), rid2dpidHF( rid2nsqrFgdObj ) ];
                rid2tlbr = rid2tlbrFgdObj( 1 : 4, rid2nsqrFgdObj );
                rid2tlbr = [ [ rid2tlbr; ones( 1, size( rid2tlbr, 2 ) ) ], ...
                    [ rid2tlbr; 2 * ones( 1, size( rid2tlbr, 2 ) ) ] ];
                rid2oidxFgdObjNsqr = [ rid2oidxFgdObj( rid2nsqrFgdObj ), ...
                    rid2oidxFgdObj( rid2nsqrFgdObj ) ];
                if sum( oidx2fgdObj )
                    fgdObj.oid2dpid2regnsNsqr{ newiid } = cell( sum( oidx2fgdObj ), 1 );
                end
                for oidxFgd = 1 : sum( oidx2fgdObj ),
                    oidxFgdObj = oidxsFgdObj( oidxFgd );
                    rids = find( rid2oidxFgdObjNsqr == oidxFgdObj );
                    dpids = rid2dpidNsqr( rids );
                    fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd } = arrayfun( ...
                        @( dpid )rid2tlbr( :, rids( dpids == dpid ) ), ...
                        1 : numDirPair, 'UniformOutput', false );
                    objBbox = oidx2tlbr( 1 : 4, oidxFgdObj );
                    objTl = objBbox( 1 : 2 );
                    objBr = objBbox( 3 : 4 );
                    objCenter = ( objTl + objBr ) / 2;
                    objBoxSide = objBr - objTl;
                    shift = objBoxSide * minObjScale / 2;
                    tlout   = objTl;
                    tlin    = round( objCenter - shift );
                    brout   = objBr;
                    brin    = round( objCenter + shift );
                    tlrs = tlout( 1 ) : tlin( 1 );
                    tlcs = tlout( 2 ) : tlin( 2 );
                    brrs = brin( 1 ) : brout( 1 );
                    brcs = brin( 2 ) : brout( 2 );
                    objBboxEroded = zeros( 4, numErode );
                    for e = 1 : numErode
                        tlr = tlrs( ceil( rand * numel( tlrs ) ) );
                        tlc = tlcs( ceil( rand * numel( tlcs ) ) );
                        brr = brrs( ceil( rand * numel( brrs ) ) );
                        brc = brcs( ceil( rand * numel( brcs ) ) );
                        objBboxEroded( :, e ) = [ tlr; tlc; brr; brc; ];
                    end;
                    objBboxEroded = [ [ objBboxEroded; ones( 1, size( objBboxEroded, 2 ) ); ], ...
                        [ objBboxEroded; 2 * ones( 1, size( objBboxEroded, 2 ) ); ] ];
                    fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd }{ end } = ...
                        cat( 2, fgdObj.oid2dpid2regnsNsqr{ newiid }{ oidxFgd }{ end }, objBboxEroded );
                end
                % Back-ground objects.
                bgdObj.oid2iid{ newiid } = newiid * ones( sum( oidx2bgdObj ), 1, 'single' );
                % Back-ground objects: square regions.
                rid2sqrBgdObj = rid2tlbrBgdObj( 6, : ) == 1;
                rid2sqrBgdObj = rid2sqrBgdObj & ( rid2tlbrBgdObj( 7, : ) == 1 );
                rid2tlbr = rid2tlbrBgdObj( 1 : 4, rid2sqrBgdObj );
                rid2oidxBgdObjSqr = rid2oidxBgdObj( rid2sqrBgdObj );
                bgdObj.oid2regnsSqr{ newiid } = arrayfun( ...
                    @( oid )rid2tlbr( :, rid2oidxBgdObjSqr == oid ), ...
                    find( oidx2bgdObj ), 'UniformOutput', false );
                % Back-ground objects: non-square regions.
                rid2nsqrBgdObj = rid2tlbrBgdObj( 6, : ) ~= 1;
                rid2nsqrBgdObj = rid2nsqrBgdObj | ( rid2tlbrBgdObj( 7, : ) ~= 1 );
                rid2tlbr = rid2tlbrBgdObj( 1 : 4, rid2nsqrBgdObj );
                rid2oidxBgdObjNsqr = rid2oidxBgdObj( rid2nsqrBgdObj );
                bgdObj.oid2regnsNsqr{ newiid } = arrayfun( ...
                    @( oid )rid2tlbr( :, rid2oidxBgdObjNsqr == oid ), ...
                    find( oidx2bgdObj ), 'UniformOutput', false );
                % Back-ground: squre regions.
                rid2sqrBgd = rid2tlbrBgd( 6, : ) == 1;
                rid2sqrBgd = rid2sqrBgd & ( rid2tlbrBgd( 7, : ) == 1 );
                for s = 1 : numScale,
                    ok = rid2sqrBgd & ( rid2tlbrBgd( 5, : ) == s );
                    rids = find( ok );
                    if numel( rids ) > numMaxBgdRegnPerScale,
                        rids = sort( randsample( rids, numMaxBgdRegnPerScale ) );
                    end
                    bgd.iid2sid2regnsSqr{ newiid, s } = rid2tlbrBgd( 1 : 4, rids );
                end
                % Back-ground: non-square regions.
                rid2nsqrBgd = rid2tlbrBgd( 6, : ) ~= 1;
                rid2nsqrBgd = rid2nsqrBgd | ( rid2tlbrBgd( 7, : ) ~= 1 );
                for s = 1 : numScale,
                    ok = rid2nsqrBgd & ( rid2tlbrBgd( 5, : ) == s );
                    rids = find( ok );
                    if numel( rids ) > numMaxBgdRegnPerScale,
                        rids = sort( randsample( rids, numMaxBgdRegnPerScale ) );
                    end
                    bgd.iid2sid2regnsNsqr{ newiid, s } = rid2tlbrBgd( 1 : 4, rids );
                end
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop...
                    (   numIm, newiid, ...
                    sprintf( 'ext sub-winds on im %d', newiid ), cummt );
            end;
            fgdObj.oid2iid = cat( 1, fgdObj.oid2iid{ : } );
            fgdObj.oid2dpid2regnsSqr = cat( 1, fgdObj.oid2dpid2regnsSqr{ : } );
            fgdObj.oid2dpid2regnsSqr = cat( 1, fgdObj.oid2dpid2regnsSqr{ : } );
            fgdObj.oid2dpid2regnsNsqr = cat( 1, fgdObj.oid2dpid2regnsNsqr{ : } );
            fgdObj.oid2dpid2regnsNsqr = cat( 1, fgdObj.oid2dpid2regnsNsqr{ : } );
            bgdObj.oid2iid = cat( 1, bgdObj.oid2iid{ : } );
            bgdObj.oid2regnsSqr = cat( 1, bgdObj.oid2regnsSqr{ : } );
            bgdObj.oid2regnsNsqr = cat( 1, bgdObj.oid2regnsNsqr{ : } );
            subDb.iid2impath = iid2impath;
            subDb.dpid2dids = dpid2dids;
            subDb.fgdObj = fgdObj;
            subDb.bgdObj = bgdObj;
            subDb.bgd = bgd;
        end
        % Functions for file IO.
        function name = getTsDbName( this )
            name = sprintf( ...
                'DBTS_%s_OF_%s', ...
                this.settingTsAug.changes, ...
                this.srcDb.getName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getTsDbDir( this )
            dir = this.srcDb.getDir;
        end
        function dir = makeTsDbDir( this )
            dir = this.getTsDbDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getTsDbPath( this )
            fname = strcat( this.getTsDbName, '.mat' );
            path = fullfile( this.getTsDbDir, fname );
        end
    end
    methods( Static )
        function [ sid2iid, sid2regn, sid2cid ] = ...
                makeFgdObjRegnSequence( oid2iid, oid2dpid2regns, dpid2dids )
            numObj = numel( oid2iid );
            numDirPair = size( oid2dpid2regns, 2 );
            sid2iid = zeros( numObj * numDirPair, 1, 'single' );
            sid2regn = zeros( 5, numObj * numDirPair, 'single' );
            sid2cid = zeros( 2, numObj * numDirPair, 'single' );
            sid = 0;
            for oid = 1 : numObj,
                iid = oid2iid( oid );
                for dpid = 1 : numDirPair,
                    regns = oid2dpid2regns{ oid, dpid };
                    numRegns = size( regns, 2 );
                    if ~numRegns, continue; end;
                    sid = sid + 1;
                    sel = ceil( rand * numRegns );
                    sid2iid( sid ) = iid;
                    sid2regn( :, sid ) = regns( :, sel );
                    sid2cid( :, sid ) = dpid2dids( :, dpid );
                end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
            sid2cid = sid2cid( :, perm );
        end
        function [ sid2iid, sid2regn ] = ...
                makeBgdObjRegnSequence( oid2iid, oid2regns, numSmpl )
            numObj = numel( oid2iid );
            if ~numObj, sid2iid = zeros( 0, 1 ); sid2regn = zeros( 5, 0 ); return; end;
            sid2iid = zeros( numSmpl, 1, 'single' );
            sid2regn = zeros( 5, numSmpl, 'single' );
            sid = 0;
            while true,
                oid = ceil( numObj * rand );
                iid = oid2iid( oid );
                regns = oid2regns{ oid };
                numRegns = size( regns, 2 );
                if ~numRegns, continue; end;
                sid = sid + 1;
                sel = ceil( rand * numRegns );
                sid2iid( sid ) = iid;
                sid2regn( :, sid ) = [ regns( :, sel ); ceil( rand * 2 ); ];
                if sid == numSmpl, break; end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
        end
        function [ sid2iid, sid2regn ] = ...
                makeBgdRegnSequence( iid2sid2regns, numSmpl )
            numIm = size( iid2sid2regns, 1 );
            numScale = size( iid2sid2regns, 2 );
            sid2iid = zeros( numSmpl, 1, 'single' );
            sid2regn = zeros( 5, numSmpl, 'single' );
            sid = 0;
            while true,
                iid = ceil( numIm * rand );
                scale = ceil( numScale * rand );
                regns = iid2sid2regns{ iid, scale };
                numRegns = size( regns, 2 );
                if ~numRegns, continue; end;
                sid = sid + 1;
                sel = ceil( rand * numRegns );
                sid2iid( sid ) = iid;
                sid2regn( :, sid ) = [ regns( :, sel ); ceil( rand * 2 ); ];
                if sid == numSmpl, break; end;
            end;
            perm = randperm( sid )';
            sid2iid = sid2iid( perm );
            sid2regn = sid2regn( :, perm );
        end
        % Majorly used in CNN. Forward/backward function in output layer.
        function res2 = forward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numOutLyr = size( gt, 3 );
            numDimPerLyr = size( X, 3 ) / numOutLyr;
            y = gpuArray( zeros( numOutLyr, 1, 'single' ) );
            for lyid = 1 : numOutLyr,
                sidx = ( lyid - 1 ) * numDimPerLyr + 1;
                eidx = lyid * numDimPerLyr;
                y( lyid ) = vl_nnsoftmaxloss...
                    ( X( :, :, sidx : eidx, : ), gt( :, :, lyid, : ) );
            end
            res2.x = mean( y );
        end
        % Majorly used in CNN. Forward/backward function in output layer.
        function res1 = backward( ly, res1, res2 )
            X = res1.x;
            gt = ly.class;
            numOutLyr = size( gt, 3 );
            numDimPerLyr = size( X, 3 ) / numOutLyr;
            dzdy = res2.dzdx / numOutLyr;
            Y = gpuArray( zeros( size( X ), 'single' ) );
            for lyid = 1 : numOutLyr,
                sidx = ( lyid - 1 ) * numDimPerLyr + 1;
                eidx = lyid * numDimPerLyr;
                Y( :, :, sidx : eidx, : ) = ...
                    vl_nnsoftmaxloss...
                    ( X( :, :, sidx : eidx, : ), gt( :, :, lyid, : ), dzdy );
            end
            res1.dzdx = Y;
        end
        % Majorly used in CNN. Update energy and task-specific evaluation metric.
        % For this target application, object detection, 
        % the metric is top-1 accuracy.
        function tsMetric = computeTsMetric( res, gts )
            output = gather( res( end - 1 ).x );
            gts = gather( gts );
            numOutLyr = size( gts, 3 );
            numDimPerLyr = size( output, 3 ) / numOutLyr;
            err1 = zeros( numOutLyr, 1, 'single' );
            for lyid = 1 : numOutLyr,
                sidx = ( lyid - 1 ) * numDimPerLyr + 1;
                eidx = lyid * numDimPerLyr;
                predictions = output( :, :, sidx : eidx, : );
                [ ~, predictions ] = sort...
                    ( predictions, 3, 'descend' );
                err1_ = ~bsxfun( @eq, predictions, ...
                    gts( :, :, lyid, : ) );
                % Take top predictions.
                % If top-1, err1_ = err1_( :, :, 1, : );
                % If top-5, err1_ = err1_( :, :, 1 : 5, : );
                err1_ = err1_( :, :, 1, : );
                err1( lyid ) = sum( err1_( : ) );
            end
            tsMetric = mean( err1 );
        end
        function [ layers, cnnName ] = provdInitCnn
            global path;
            cnnName = 'PTVGGM';
            % Load pre-trained CNN.
            srcCnn = load( path.extnet_vgg_m );
            layers = srcCnn.layers;
            for lid = 1 : numel( layers )
                if ~strcmp( layers{ lid }.type, 'conv' ),
                    continue;
                end
                if ~isfield( layers{ lid }, 'filtersMomentum' )
                    layers{ lid }.filtersMomentum = ...
                        zeros( 'like', layers{ lid }.filters );
                    layers{ lid }.biasesMomentum = ...
                        zeros( 'like', layers{ lid }.biases );
                end
                if ~isfield( layers{ lid }, 'filtersLearningRate' )
                    layers{ lid }.filtersLearningRate = 1 / 10;     % Caution!
                    layers{ lid }.biasesLearningRate = 2 / 10;      % Caution!
                end
                if ~isfield( layers{ lid }, 'filtersWeightDecay' )
                    layers{ lid }.filtersWeightDecay = 1;
                    layers{ lid }.biasesWeightDecay = 0;
                end
            end
            % Re-initialize the last convolutional layer.
            cfgConv.type = 'conv';
            cfgConv.filterSize = 1;
            cfgConv.filterDepth = 2048;
            cfgConv.numFilter = ( 3 + 1 + 1 ) * 2 * 1; % ( numDirection + stopSignal + noObjectSignal ) * ( TL + BR ) * numObjClass.
            cfgConv.stride = [ 1, 1 ];
            cfgConv.pad = [ 0, 0, 0, 0 ];
            cfgConv.initWScal = 1;
            cfgConv.initB = 0;
            cfgConv.filtersLearningRate = 1;    % Caution!
            cfgConv.biasesLearningRate = 2;     % Caution!
            cfgConv.filtersWeightDecay = 1;
            cfgConv.biasesWeightDecay = 0;
            layerConv.type = cfgConv.type;
            layerConv.filters = 0.01 / ...
                cfgConv.initWScal * ...
                randn( cfgConv.filterSize, ...
                cfgConv.filterSize, ...
                cfgConv.filterDepth, ...
                cfgConv.numFilter, 'single' );
            layerConv.biases = cfgConv.initB * ...
                ones( 1, cfgConv.numFilter, 'single' );
            layerConv.stride = cfgConv.stride;
            layerConv.pad = cfgConv.pad;
            layerConv.filtersLearningRate = ...
                cfgConv.filtersLearningRate;
            layerConv.biasesLearningRate = ...
                cfgConv.biasesLearningRate;
            layerConv.filtersWeightDecay = ...
                cfgConv.filtersWeightDecay;
            layerConv.biasesWeightDecay = ...
                cfgConv.biasesWeightDecay;
            layerConv.filtersMomentum = ...
                zeros( 'like', layerConv.filters );
            layerConv.biasesMomentum = ...
                zeros( 'like', layerConv.biases );
            % Re-initialize the last layer.
            layerCustom.type = 'custom';
            layerCustom.backward = @InOutCls.backward;
            layerCustom.forward = @InOutCls.forward;
            % Replacement.
            layers{ end - 1 } = layerConv;
            layers{ end } = layerCustom;
        end
    end
end