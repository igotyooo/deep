function rid2tlbr = extMultiScaleDenseRegions...
    ( imSize, docScaleMag, stride, regionSide, scales, aspects )
    imSize = imSize( 1 : 2 ); imSize = imSize( : );
    imSizeMag = round( imSize * sqrt( docScaleMag ) );
    numScale = numel( scales ); numAspect = numel( aspects );
    rid2tlbr = cell( 1, numScale * numAspect ); cnt = 0;
    for s = 1 : numScale,
        for a = 1 : numAspect,
            cnt = cnt + 1;
            rside = round( min( imSizeMag ) / scales( s ) );
            cside = round( rside / aspects( a ) );
            rstrd = round( stride * rside / regionSide );
            cstrd = round( stride * cside / regionSide );
            r = 1 : rstrd : ( imSizeMag( 1 ) - rside + 1 );
            c = 1 : cstrd : ( imSizeMag( 2 ) - cside + 1 );
            if isempty( r ), r = round( ( imSizeMag( 1 ) - rside ) / 2 ) + 1; end;
            if isempty( c ), c = round( ( imSizeMag( 2 ) - cside ) / 2 ) + 1; end;
            nr = numel( r ); nc = numel( c );
            [ c, r ] = meshgrid( c, r );
            tlbr = cat( 3, r, c, r + rside - 1, c + cside - 1 );
            tlbr = reshape( permute( tlbr, [ 3, 1, 2 ] ), 4, nr * nc );
            rid2tlbr{ cnt } = cat( 1, tlbr, ...
                s * ones( 1, nr * nc ), ...
                ( aspects( a ) == 1 ) * ones( 1, nr * nc ) );
        end;
    end;
    rid2tlbr = cat( 2, rid2tlbr{ : } );
    rid2tlbr( 1 : 4, : ) = bsxfun( @plus, ...
        rid2tlbr( 1 : 4, : ), ...
        - repmat( ( imSizeMag - imSize ) / 2, 2, 1 ) );
end