function [ rid2tlbr, rid2oidx, rid2tlbrBgd, rid2oidxBgd ] = ...
    extTargetObjRegns( ...
    oidx2tlbr, ...
    oidx2fgd, ...
    imSize, ...
    docScaleMag, ...
    stride, ...
    dstSide, ...
    scales, ...
    aspects, ...
    insectOverFgdObj, ...
    insectOverFgdObjForMajority, ...
    fgdObjMajority, ...
    insectOverBgdObj, ...
    insectOverBgdRegn, ...
    insectOverBgdRegnForReject )
    % Extract multi-scale & multi-aspect dense regions.
    imSize = imSize( 1 : 2 );
    imSize = imSize( : );
    rid2tlbr = extMultiScaleDenseRegions...
        ( imSize, docScaleMag, stride, dstSide, scales, aspects );
    % Compute information.
    foid2oidx = find( oidx2fgd );
    foid2tlbr = oidx2tlbr( :, oidx2fgd );
    numFgd = size( foid2tlbr, 2 );
    foid2rect = tlbr2rect( foid2tlbr );
    rid2rect = tlbr2rect( rid2tlbr );
    foid2area = prod( foid2rect( 3 : 4, : ), 1 )';
    foid2rid2insect = rectint( foid2rect', rid2rect' );
    % Make background regions.
    boid2oidx = find( ~oidx2fgd );
    boid2tlbr = oidx2tlbr( :, boid2oidx );
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
    if ~numFgd, rid2tlbr = zeros( 6, 0 ); rid2oidx = [  ]; return; end;
    % Filter by intersection over object.
    foid2rid2insectOverTarget = bsxfun( @times, foid2rid2insect, 1./ foid2area );
    foid2rid2ok = foid2rid2insectOverTarget > insectOverFgdObj;
    rid2ok = any( foid2rid2ok, 1 );
    foid2rid2insect_ = foid2rid2ok .* foid2rid2insect;
    foid2rid2insect_ = foid2rid2insect_( :, rid2ok );
    foid2rid2insect = foid2rid2insect( :, rid2ok );
    rid2tlbr = rid2tlbr( :, rid2ok );
    [ rid2tinsect, rid2foid ] = max( foid2rid2insect_, [  ], 1 );
    foid2rid2insect( ( ( 1 : sum( rid2ok ) ) - 1 ) * numFgd + rid2foid ) = 0;
    % Filter by object majority.
    foid2rid2insectOverTarget = bsxfun( @times, foid2rid2insect, 1./ foid2area ); % 
    foid2rid2ok = foid2rid2insectOverTarget > insectOverFgdObjForMajority;
    foid2rid2insect = foid2rid2insect .* foid2rid2ok;
    topInsect = max( foid2rid2insect, [  ], 1 );
    rid2ok = ( rid2tinsect ./ max( 1, topInsect( 1, : ) ) ) > fgdObjMajority;
    rid2foid = rid2foid( rid2ok );
    rid2oidx = foid2oidx( rid2foid );
    rid2tlbr = rid2tlbr( :, rid2ok );
end

