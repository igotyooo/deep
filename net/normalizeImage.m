function image = normalizeImage( image, averageImage, interpolation )
    [ r, c, ch ] = size( image );
    [ rav, cav, chav ] = size( averageImage );
    if ch ~= chav, warning( 'Inconsistent channels.' ); end;
    if chav == 1 && ch == 3, image = rgb2gray( image ); end;
    if chav == 3 && ch == 1, image = cat( 3, image, image, image ); end;
    if ( rav * cav ~= 1 ) && ( r ~= rav || c ~= cav ), ...
            averageImage = imresize( averageImage, [ r, c ], 'method', interpolation ); end;
    image = bsxfun( @minus, image, averageImage );
end