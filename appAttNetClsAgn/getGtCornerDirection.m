function [ didTl, didBr, didTlFlip, didBrFlip, newRegnOrg ] = getGtCornerDirection...
    ( region, gt, did2vecTl, did2vecBr, directionVectorMagnitude, domainWarp )
    numDir = size( did2vecTl, 2 );
    patchSide = domainWarp( 1 );
    generousStop = round( patchSide / 20 ); % directionVectorMagnitude / 2;
    bias = region( 1 : 2 ) - 1;
    domainOrg = region( 3 : 4 ) - bias;
    gtWarp = resizeTlbr( gt - [ bias; bias; ], domainOrg, domainWarp );
    dirTl = max( 0, gtWarp( 1 : 2 ) - 1 );
    dirBr = min( 0, gtWarp( 3 : 4 ) - domainWarp );
    if sqrt( sum( dirTl .^ 2 ) ) < generousStop, 
        didTl = numDir; else [ ~, didTl ] = max( dirTl' * did2vecTl ); end;
    if sqrt( sum( dirBr .^ 2 ) ) < generousStop, 
        didBr = numDir; else [ ~, didBr ] = max( dirBr' * did2vecBr ); end;
    newRegnWarp = [ 1; 1; domainWarp; ] + [ did2vecTl( :, didTl ); did2vecBr( :, didBr ); ] * directionVectorMagnitude;
    newRegnOrg = resizeTlbr( newRegnWarp, domainWarp, domainOrg ) + [ bias; bias; ];
    % Flipping.
    gtWarpFlip = [ gtWarp( 1 ); patchSide - gtWarp( 4 ) + 1; gtWarp( 3 ); patchSide - gtWarp( 2 ) + 1; ];
    dirTlFlip = max( 0, gtWarpFlip( 1 : 2 ) - 1 );
    dirBrFlip = min( 0, gtWarpFlip( 3 : 4 ) - domainWarp );
    if sqrt( sum( dirTlFlip .^ 2 ) ) < generousStop, 
        didTlFlip = numDir; else [ ~, didTlFlip ] = max( dirTlFlip' * did2vecTl ); end;
    if sqrt( sum( dirBrFlip .^ 2 ) ) < generousStop, 
        didBrFlip = numDir; else [ ~, didBrFlip ] = max( dirBrFlip' * did2vecBr ); end;
end