function [ d, h, m, s ] = myclock( cummt, curriter, totiter )

    remain = ( cummt / curriter ) * ( totiter - curriter );
    d = floor( remain / ( 24 * 3600 ) );
    h = floor( ( remain - d * 24 * 3600 ) / 3600 );
    m = floor( ( remain - h * 3600 ) / 60 );
    s = floor( ( remain - h * 3600 ) - m * 60 );

end