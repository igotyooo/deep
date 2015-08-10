function image = normalizeImage( image, averageImage, interpolation )
    [ r, c, ~ ] = size( image );
    [ rav, cav, ~ ] = size( averageImage );
    if rav * cav == 1, image = bsxfun( @minus, image, averageImage ); return; end;
    if r ~= rav || c ~= cav, averageImage = imresize( averageImage, [ r, c ], 'method', interpolation ); end;
    image = image - averageImage;
end