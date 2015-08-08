function rid2tlbr = extMultiScaleDenseRegions2...
    ( imSize, stride, patchSide, sid2s, aid2a, dilate )
    imSize = imSize( 1 : 2 ); imSize = imSize( : );
    numScale = numel( sid2s );
    numAspect = numel( aid2a );
    rid2tlbr = cell( 1, numScale * numAspect ); 
    cnt = 0;
    for s = 1 : numScale,
        for a = 1 : numAspect,
            % Fix height, warp width.
            pr = patchSide / sid2s( s );
            pc = patchSide / sid2s( s ) / aid2a( a );
            strdr = stride / sid2s( s );
            strdc = stride / sid2s( s ) / aid2a( a );
            sr = 1 - pr * dilate;
            sc = 1 - pc * dilate;
            er = imSize( 1 ) - pr * ( 1 - dilate ) + 1;
            ec = imSize( 2 ) - pc * ( 1 - dilate ) + 1;
            r = sr : strdr : er;
            c = sc : strdc : ec;
            nr = numel( r );
            nc = numel( c );
            [ c, r ] = meshgrid( c, r );
            tlbr = cat( 3, r, c, r + pr - 1, c + pc - 1 );
            tlbr = single( tlbr );
            tlbr = reshape( permute( tlbr, [ 3, 1, 2 ] ), 4, nr * nc );
            cnt = cnt + 1;
            rid2tlbr{ cnt } = cat( 1, tlbr, ...
                s * ones( 1, nr * nc, 'single' ), ...
                a * ones( 1, nr * nc, 'single' ) );
        end;
    end;
    rid2tlbr = cat( 2, rid2tlbr{ : } );
    rid2tlbr = round( rid2tlbr );
end