function sid2s = determineScales( oid2tlbr, referenceSide, numScale, confidence )
    resolution = 1000;
    oid2h = ( oid2tlbr( 3, : ) - oid2tlbr( 1, : ) + 1 )';
    oid2w = ( oid2tlbr( 4, : ) - oid2tlbr( 2, : ) + 1 )';
    oid2side = max( [ oid2h, oid2w ], [  ], 2 );
    oid2s = referenceSide ./ oid2side;
    idx2s = logspace( -2, 2, resolution );
    idx2pm = histc( oid2s, idx2s );
    idx2cm = cumsum( idx2pm );
    for c = 0 : 1 : max( idx2pm ),
        if sum( idx2pm( idx2pm >= c ) ) / sum( idx2pm ) < confidence,
            indice = find( idx2pm >= c );
            mincm = idx2cm( min( indice ) );
            maxcm = idx2cm( max( indice ) );
            break;
        end;
    end;
    sid2cm = ( mincm : ( maxcm - mincm ) / ( numScale - 1 ) : maxcm )';
    [ ~, sid2idx ] = min( abs( bsxfun( @minus, sid2cm, repmat( idx2cm', numel( sid2cm ), 1 ) ) ), [], 2 );
    sid2idx = unique( sid2idx );
    sid2s = idx2s( sid2idx )';
end