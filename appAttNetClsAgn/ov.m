function [ newrid2tlbr, newrid2score ] = ...
    ov( rid2tlbr, rid2score, overlap, minNumSuppBox, mergingMethod )
    rid2score = rid2score( : )';
    if isempty( rid2tlbr ), 
        newrid2tlbr = zeros( 4, 0 ); 
        newrid2score = zeros( 0, 1 ); 
        return; 
    end;
    if size( rid2tlbr, 2 ) == 1,
        if minNumSuppBox >= 1, 
            newrid2tlbr = zeros( 4, 0 ); 
            newrid2score = zeros( 0, 1 ); 
            return; 
        end;
        newrid2tlbr = rid2tlbr; 
        newrid2score = rid2score; 
        return;
    end;
    numCand = size( rid2tlbr, 2 );
    rid2tlbrRect = tlbr2rect( rid2tlbr );
    rid2area = prod( rid2tlbrRect( 3 : 4, : ), 1 )';
    rid2rid2int = rectint( rid2tlbrRect', rid2tlbrRect' );
    rid2rid2uni = repmat( rid2area, 1, numCand ) + ...
        repmat( rid2area, 1, numCand )' - rid2rid2int;
    rid2rid2ov = rid2rid2int ./ rid2rid2uni;
    rid2rid2ov = triu( rid2rid2ov ) - eye( numCand );
    rid2rid2mrg = rid2rid2ov > overlap;
    [ newrid2did1, newrid2rid2 ] = find( rid2rid2mrg );
    newrid2dids = [ newrid2did1, newrid2rid2 ]';
    sdids = setdiff( 1 : numCand, unique( newrid2dids( : ) ) );
    rid2newrid2is = false( numCand, numel( newrid2did1 ) + numel( sdids ) );
    rid2newrid2is( ...
        [ newrid2did1', sdids ] + ...
        numCand * ( 0 : numel( newrid2did1 ) + numel( sdids ) - 1 ) ) = true;
    rid2newrid2is( newrid2rid2' + ...
        numCand * ( 0 : numel( newrid2rid2 ) - 1 ) ) = true;
    for did = 1 : numCand,
        newrid2mrg = rid2newrid2is( did, : );
        rid2is = any( rid2newrid2is( :, newrid2mrg ), 2 );
        rid2newrid2is( :, newrid2mrg ) = [  ];
        rid2newrid2is = [ rid2is, rid2newrid2is ];
    end;
    rid2newrid2is = rid2newrid2is...
        ( :, sum( rid2newrid2is ) > minNumSuppBox );
    numDet = size( rid2newrid2is, 2 );
    newrid2tlbr = zeros( size( rid2tlbr, 1 ), numDet );
    newrid2score = zeros( numDet, 1 );
    for newdid = 1 : numDet,
        boxes = rid2tlbr( :, rid2newrid2is( :, newdid ) );
        scores = rid2score( rid2newrid2is( :, newdid ) );
        switch mergingMethod,
            case 'WAVG',
                box = sum( bsxfun( @times, boxes, scores ), 2 ) / ...
                    sum( scores );
                score = mean( scores );
            case 'MAX',
                [ ~, is ] = max( scores );
                box = boxes( :, is );
                score = max( scores );
        end
        newrid2tlbr( :, newdid ) = box;
        newrid2score( newdid ) = score;
    end;
end