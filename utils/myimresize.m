function imre = myimresize( im, dstSide )
    [ r, c, ~ ] = size( im );
    scale = dstSide / min( r, c );
    imre = imresize( im, scale );
    [ r, c, ~ ] = size( imre );
    if c > dstSide, imre = imre( :, round( 1 : ( ( c - 1 ) / ( dstSide - 1 ) ) : c ), : ); end;
    if r > dstSide, imre = imre( round( 1 : ( ( r - 1 ) / ( dstSide - 1 ) ) : r ), :, : ); end;
    clear im;
end