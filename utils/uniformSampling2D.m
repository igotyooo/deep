function [ x0, y0 ] = uniformSampling2D( idx2x, idx2y, dist )

distCol = sum( dist, 1 );
distCol = distCol / sum( distCol );
ind1 = gendist( distCol, 1 );
x0 = idx2x( ind1 );

[ ~, ind_temp ] = sort( ( x0 - idx2x ) .^ 2 );
distRow = dist( :, ind_temp( 1 ) );
distRow = distRow / sum( distRow );
ind2 = gendist( distRow, 1 );
y0 = idx2y( ind2 );



function T  =  gendist( P, N )
if size( P, 1 ) > 1, P = P';end;
Pnorm = [ 0 P ] / sum( P );
Pcum = cumsum( Pnorm );
R = rand( 1, N );
V = 1 : length( P );
[ ~, inds ]  =  histc( R, Pcum ); 
T  =  V( inds );