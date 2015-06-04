function [ im, mask ] = contourOnIm( im, cont, color )
    cont = round( cont );
    [ r, c, ~ ] = size( im );
    rs = cont( 2, : )';
    cs = cont( 1, : )';
    idx = ( cs - 1 ) * r + rs;
    idx = cat( 1, idx, idx - 1, idx + 1, idx + r, idx - r );
    idx( idx < 1 ) = [  ];
    im( min( idx, r * c ) ) = color( 1 );
    im( min( idx + r * c, r * c * 2 ) ) = color( 2 );
    im( min( idx + r * c * 2, r * c * 3 ) ) = color( 3 );
    if nargout > 1,
        mask = false( r, c );
        mask( min( idx, r * c ) ) = true;
    end
end