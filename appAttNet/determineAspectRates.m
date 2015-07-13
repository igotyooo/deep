function aid2how = determineAspectRates( oid2bbox, numAspect, confidence )
    resolution = 1000;
    oid2h = ( oid2bbox( 3, : ) - oid2bbox( 1, : ) + 1 )';
    oid2w = ( oid2bbox( 4, : ) - oid2bbox( 2, : ) + 1 )';
    oid2how = oid2h ./ oid2w;
    idx2how = logspace( -2, 2, resolution );
    idx2pm = histc( oid2how, idx2how );
    idx2cm = cumsum( idx2pm );
    for c = 0 : 1 : max( idx2pm ),
        if sum( idx2pm( idx2pm >= c ) ) / sum( idx2pm ) < confidence,
            indice = find( idx2pm >= c );
            mincm = idx2cm( min( indice ) );
            maxcm = idx2cm( max( indice ) );
            break;
        end;
    end;
    aid2cm = ( mincm : ( maxcm - mincm ) / ( numAspect - 1 ) : maxcm )';
    [ ~, aid2idx ] = min( abs( bsxfun( @minus, aid2cm, repmat( idx2cm', numel( aid2cm ), 1 ) ) ), [], 2 );
    aid2idx = unique( aid2idx );
    aid2how = idx2how( aid2idx )';
end

