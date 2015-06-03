function [ im, mask ] = contourOnIm( im, cont, color )
    cont = splitContour( cont );
    cont = round( cat( 2, cont{ : } ) );
    [ r, c, ~ ] = size( im );
    rs = cont( 2, : )';
    cs = cont( 1, : )';
    idx = ( cs - 1 ) * r + rs;
    idx = cat( 1, idx, idx - 1, idx + 1, idx + r, idx - r );
    im( idx ) = color( 1 );
    im( idx + r * c ) = color( 2 );
    im( idx + r * c * 2 ) = color( 3 );
end

