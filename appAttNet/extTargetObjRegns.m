function [ rid2tlbr, rid2oidx, rid2tlbrBgd, rid2oidxBgd ] = ...
    extTargetObjRegns( ...
    oidx2tlbr, ...
    oidx2fgd, ...
    imSize, ...
    stride, ...
    dstSide, ...
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
    insectOverBgdRegnForReject )
    % Extract dense regions in original image.
    rid2tlbr = extDenseRegns...
        ( imSize, stride, dstSide, scaleStep, numScale );
    numRegn = size( rid2tlbr, 2 );
    rid2tlbr = cat( 1, rid2tlbr, ones( 2, numRegn ) );
    % Extract dense regions in different horizontal scales.
    horzScales = setdiff( startHorzScale : horzScaleStep : endHorzScale, 1 );
    numHorzScale = numel( horzScales );
    rid2tlbrHorz = cell( 1, numHorzScale );
    for hsid = 1 : numHorzScale;
        s = horzScales( hsid );
        imSize_ = imSize;
        imSize_( 2 ) = round( imSize( 2 ) * s );
        rid2tlbrHorz{ hsid } = extDenseRegns...
            ( imSize_, stride, dstSide, scaleStep, numScale );
        numRegn = size( rid2tlbrHorz{ hsid }, 2 );
        rid2tlbrHorz{ hsid } = resizeTlbr( rid2tlbrHorz{ hsid }, imSize_, imSize );
        rid2tlbrHorz{ hsid } = cat( 1, rid2tlbrHorz{ hsid }, ones( 1, numRegn ), s * ones( 1, numRegn ) );
    end; rid2tlbrHorz = cat( 2, rid2tlbrHorz{ : } );
    % Extract dense regions in different horizontal scales.
    vertScales = setdiff( startVertScale : vertScaleStep : endVertScale, 1 );
    numVertScale = numel( vertScales );
    rid2tlbrVert = cell( 1, numVertScale );
    for vsid = 1 : numVertScale;
        s = vertScales( vsid );
        imSize_ = imSize;
        imSize_( 1 ) = round( imSize( 1 ) * vertScales( vsid ) );
        rid2tlbrVert{ vsid } = extDenseRegns...
            ( imSize_, stride, dstSide, scaleStep, numScale );
        numRegn = size( rid2tlbrVert{ vsid }, 2 );
        rid2tlbrVert{ vsid } = resizeTlbr( rid2tlbrVert{ vsid }, imSize_, imSize );
        rid2tlbrVert{ vsid } = cat( 1, rid2tlbrVert{ vsid }, s * ones( 1, numRegn ), ones( 1, numRegn ) );
    end; rid2tlbrVert = cat( 2, rid2tlbrVert{ : } );
    rid2tlbr = cat( 2, rid2tlbr, rid2tlbrHorz, rid2tlbrVert ); 
    clear rid2tlbrHorz rid2tlbrVert;
    % Compute information.
    foid2oidx = find( oidx2fgd );
    foid2tlbr = oidx2tlbr( :, oidx2fgd );
    numTarget = size( foid2tlbr, 2 );
    foid2rect = tlbr2rect( foid2tlbr );
    rid2rect = tlbr2rect( rid2tlbr );
    foid2area = prod( foid2rect( 3 : 4, : ), 1 )';
    foid2rid2insect = rectint( foid2rect', rid2rect' );
    % Make background regions.
    boid2oidx = find( ~oidx2fgd );
    boid2tlbr = oidx2tlbr( :, ~oidx2fgd );
    boid2rect = tlbr2rect( boid2tlbr );
    boid2area = prod( boid2rect( 3 : 4, : ), 1 )';
    rid2bgd = ~sum( foid2rid2insect, 1 );
    rid2tlbrBgd = rid2tlbr( :, rid2bgd );
    rid2rectBgd = tlbr2rect( rid2tlbrBgd );
    rid2areaBgd = prod( rid2rectBgd( 3 : 4, : ), 1 )';
    boid2rid2insect = rectint( boid2rect', rid2rectBgd' );
    boid2rid2insectOverBgdObj = bsxfun( @times, boid2rid2insect, 1./ boid2area );
    boid2rid2insectOverBgdRegn = bsxfun( @times, boid2rid2insect, 1./ rid2areaBgd' );
    boid2rid2ok1 = boid2rid2insectOverBgdObj > insectOverBgdObj;
    boid2rid2ok2 = boid2rid2insectOverBgdRegn > insectOverBgdRegn;
    boid2rid2ok = boid2rid2ok1 & boid2rid2ok2;
    rid2rejectBgd = any( ( boid2rid2insectOverBgdRegn > insectOverBgdRegnForReject ) & ( ~boid2rid2ok1 ), 1 );
    boid2rid2insect = boid2rid2ok .* boid2rid2insect;
    [ ~, rid2boid ] = max( boid2rid2insect, [  ], 1 );
    rid2oidxBgd = boid2oidx( rid2boid );
    rid2oidxBgd( ~any( boid2rid2ok, 1 ) ) = 0;
    rid2oidxBgd( rid2rejectBgd ) = [  ];
    rid2tlbrBgd( :, rid2rejectBgd ) = [  ];
    if ~numTarget, rid2tlbr = zeros( 7, 0 ); rid2oidx = [  ]; return; end;
    % Filter by intersection over object.
    foid2rid2insectOverTarget = bsxfun( @times, foid2rid2insect, 1./ foid2area );
    foid2rid2ok = foid2rid2insectOverTarget > insectOverFgdObj;
    rid2ok = any( foid2rid2ok, 1 );
    foid2rid2insect_ = foid2rid2ok .* foid2rid2insect;
    foid2rid2insect_ = foid2rid2insect_( :, rid2ok );
    foid2rid2insect = foid2rid2insect( :, rid2ok );
    rid2tlbr = rid2tlbr( :, rid2ok );
    [ rid2tinsect, rid2foid ] = max( foid2rid2insect_, [  ], 1 );
    foid2rid2insect( ( ( 1 : sum( rid2ok ) ) - 1 ) * numTarget + rid2foid ) = 0;
    % Filter by object majority.
    foid2rid2insectOverTarget = bsxfun( @times, foid2rid2insect, 1./ foid2area );
    foid2rid2ok = foid2rid2insectOverTarget > insectOverFgdObjForMajority;
    foid2rid2insect = foid2rid2insect .* foid2rid2ok;
    topInsect = max( foid2rid2insect, [  ], 1 );
    rid2ok = ( rid2tinsect ./ max( 1, topInsect( 1, : ) ) ) > fgdObjMajority;
    rid2foid = rid2foid( rid2ok );
    rid2oidx = foid2oidx( rid2foid );
    rid2tlbr = rid2tlbr( :, rid2ok );
end

