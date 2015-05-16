function [ rid2tlbr, rid2scale ] = extDensRegn( imsize, numPix, strd, scaleStep, numScale )
    sid2scale = ( scaleStep .^ ( 0 : -0.5 : ( -0.5 * ( numScale - 1 ) ) ) )';
    rOrg = imsize( 1 );
    cOrg = imsize( 2 );
    sf = sqrt( numPix / ( rOrg * cOrg ) );
    r = ceil( rOrg * sf );
    c = ceil( cOrg * sf );
    maxLen = min( r, c );
    r = r - 1;
    c = c - 1;
    sid2len = maxLen * sid2scale;
    rid2tlbr = cell( numScale, 1 );
    rid2scale = cell( numScale, 1 );
    for sid = 1 : numScale
        len = sid2len( sid );
        rgrd = 0 : strd : ( r - len + 1 );
        cgrd = 0 : strd : ( c - len + 1 );
        [ rgrd, cgrd ] = meshgrid( rgrd, cgrd );
        numReg = numel( rgrd );
        rid2tlbr{ sid } = cat( 1, ...
            reshape( rgrd, 1, numReg ), ...
            reshape( cgrd, 1, numReg ) );
        rid2tlbr{ sid } = cat( 1, ...
            rid2tlbr{ sid }, ...
            rid2tlbr{ sid } + len - 1 );
        rid2scale{ sid } = ones( 1, numReg ) * sid;
    end
    rid2tlbr = cat( 2, rid2tlbr{ : } );
    rid2scale = cat( 2, rid2scale{ : } )';
    rid2tlbr( 1 : 4, : ) = round( rid2tlbr( 1 : 4, : ) / sf + 1 );
end