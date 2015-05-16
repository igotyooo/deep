function [ oids_out, tlbrs_out ] = extSignleObjWindows( oid2tlbr, imsize, numgrid, invaderate, minoccupy )
    numr = imsize( 1 );
    numc = imsize( 2 );    
    oid2area = prod( ( oid2tlbr( 3 : 4, : ) - oid2tlbr( 1 : 2, : ) ), 1 );
    % Do the job.
    numobj = size( oid2tlbr, 2 );
    oid2rect = tlbr2rect( oid2tlbr );
    ovlmat = bsxfun( @times, rectint( oid2rect', oid2rect' ), 1 ./ oid2area );
    oid2target = sum( ovlmat < invaderate, 2 ) == ( numobj - 1 );
    step = sqrt( numr * numc / numgrid  );
    gridr = 1 : ( ( numr - 1 ) / round( ( numr - 1 ) / step ) ) : numr;
    gridc = 1 : ( ( numc - 1 ) / round( ( numc - 1 ) / step ) ) : numc;
    oids = ( 1 : numobj )';
    oids_out = find( oid2target );
    tlbrs_out = zeros( 4, numel( oids_out ) ); cnt = 0;
    for oid = oids_out'; cnt = cnt + 1;
        tlbr = oid2tlbr( :, oid );
        tlr = horzcat( gridr( gridr < tlbr( 1 ) ), tlbr( 1 ) );
        tlc = horzcat( gridc( gridc < tlbr( 2 ) ), tlbr( 2 ) );
        brr = horzcat( tlbr( 3 ), gridr( gridr > tlbr( 3 ) ) );
        brc = horzcat( tlbr( 4 ), gridc( gridc > tlbr( 4 ) ) );
        [ i1, i2, i3, i4 ] = ndgrid( 1 : numel( tlr ), 1 : numel( tlc ), 1 : numel( brr ), 1 : numel( brc ) );
        rid2tlbr = vertcat( tlr( i1( : )' ), tlc( i2( : )' ), brr( i3( : )' ), brc( i4( : )' ) );
        rid2rect = tlbr2rect( rid2tlbr );
        rid2area = prod( rid2rect( 3 : 4, : ), 1 );
        noids = setdiff( oids, oid );
        rid2oid2ovl = bsxfun( @times, rectint( oid2rect( :, noids )', rid2rect' ), 1 ./ prod( oid2rect( 3 : 4, noids ) )' );
        cand = find( prod( rid2oid2ovl < invaderate, 1 ) );
        if isempty( cand )
            rid2val = sum( abs( rid2oid2ovl - invaderate ), 1 );
            cand = find( rid2val == min( rid2val ) );
        end
        [ ~, idx ] = max( rid2area( cand ) );
        rid = cand( idx );
        tlbrs_out( :, cnt ) = rid2tlbr( :, rid );
    end
    % Filter by occupy constraint.
    imarea = numr * numc;
    oid2ok = ( ( oid2area / imarea ) >= minoccupy ) & oid2target';
    tlbrs_out = tlbrs_out( :, ismember( oids_out, find( oid2ok ) ) );
    oids_out = find( oid2ok );
end

