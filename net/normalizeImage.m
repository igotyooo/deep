function image = normalizeImage( image, averageImage, interpolation )
    [ r, c, ch ] = size( image );
    [ rav, cav, chav ] = size( averageImage );
    if ch ~= chav, error( 'Inconsistent channels.\n' ); end;
    if ( rav * cav ~= 1 ) && ( r ~= rav || c ~= cav ), ...
            averageImage = imresize( averageImage, [ r, c ], 'method', interpolation ); end;
    image = bsxfun( @minus, image, averageImage );
end