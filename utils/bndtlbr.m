function [ tlbr, idx ] = bndtlbr( tlbr, bnd )
    tlbr( 1, tlbr( 1, : ) < bnd( 1 ) ) = bnd( 1 );
    tlbr( 2, tlbr( 2, : ) < bnd( 2 ) ) = bnd( 2 );
    tlbr( 3, tlbr( 3, : ) > bnd( 3 ) ) = bnd( 3 );
    tlbr( 4, tlbr( 4, : ) > bnd( 4 ) ) = bnd( 4 );
    ok = ( tlbr( 3, : ) - tlbr( 1, : ) ) > 0;
    ok = ok & ...
        ( ( tlbr( 4, : ) - tlbr( 2, : ) ) > 0 );
    tlbr = tlbr( :, ok );
    idx = find( ok );
end