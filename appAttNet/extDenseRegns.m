function rid2regn = extDenseRegns( imSize, stride, dstSide, scaleStep, numScale )
    sid2imsize = imSize / min( imSize ) * dstSide;
    sid2imsize = sid2imsize * ...
        ( scaleStep .^ ( 0 : 0.5 : 0.5 * ( numScale - 1 ) ) );
    sid2imsize = round( sid2imsize );
    rid2regn = cell( numScale, 1 );
    for sid = 1 : numScale
        r = 1 : stride : ( sid2imsize( 1, sid ) - dstSide + 1 );
        c = 1 : stride : ( sid2imsize( 2, sid ) - dstSide + 1 );
        nr = numel( r ); nc = numel( c );
        [ c, r ] = meshgrid( c, r );
        regn = cat( 3, r, c, r + dstSide - 1, c + dstSide - 1 );
        zid = ( ( ( 1 : 4 ) - 1 ) * ( nr * nc ) )';
        rcid = 1 : ( nr * nc );
        [ zid, rcid ] = meshgrid( zid, rcid );
        idx = reshape( ( zid + rcid )', 4 * nr * nc, 1 );
        regn = reshape( regn( idx ), 4, nr * nc );
        regn = resizeTlbr( regn, sid2imsize( :, sid ), imSize );
        rid2regn{ sid } = cat( 1, round( regn ), sid * ones( 1, size( regn, 2 ) ) );
    end % Next scale.
    rid2regn = cat( 2, rid2regn{ : } );
    rid2regn( 1 : 2, : ) = max( 1, rid2regn( 1 : 2, : ) );
    rid2regn( 3, : ) = min( imSize( 1 ), rid2regn( 3, : ) );
    rid2regn( 4, : ) = min( imSize( 2 ), rid2regn( 4, : ) );
end

