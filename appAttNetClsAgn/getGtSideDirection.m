function [ didT, didL, didB, didR, didTFlip, didLFlip, didBFlip, didRFlip, newRegnOrg ] = ...
    getGtSideDirection( region, gt, directionVectorMagnitude, domainWarp )
    patchSide = domainWarp( 1 );
    generousStop = round( patchSide / 20 ); % directionVectorMagnitude / 2;
    bias = region( 1 : 2 ) - 1;
    domainOrg = region( 3 : 4 ) - bias;
    gtWarp = resizeTlbr( gt - [ bias; bias; ], domainOrg, domainWarp );
    
    dirT = max( 0, gtWarp( 1 ) - 1 );
    dirL = max( 0, gtWarp( 2 ) - 1 );
    dirB = max( 0, patchSide - gtWarp( 3 ) );
    dirR = max( 0, patchSide - gtWarp( 4 ) );
    newRegnWarp = [ 1; 1; domainWarp; ];
    if dirT < generousStop, didT = 2; else didT = 1; newRegnWarp( 1 ) = newRegnWarp( 1 ) + directionVectorMagnitude; end;
    if dirL < generousStop, didL = 2; else didL = 1; newRegnWarp( 2 ) = newRegnWarp( 2 ) + directionVectorMagnitude; end;
    if dirB < generousStop, didB = 2; else didB = 1; newRegnWarp( 3 ) = newRegnWarp( 3 ) - directionVectorMagnitude; end;
    if dirR < generousStop, didR = 2; else didR = 1; newRegnWarp( 4 ) = newRegnWarp( 4 ) - directionVectorMagnitude; end;
    newRegnOrg = resizeTlbr( newRegnWarp, domainWarp, domainOrg ) + [ bias; bias; ];
    % Flipping.
    didTFlip = didT;
    didLFlip = didR;
    didBFlip = didB;
    didRFlip = didL;
end