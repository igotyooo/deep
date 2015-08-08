function scaleId2imSize = scaleImage( scaleId2numPixels, scalingCriteria, imSize )
    scaleId2numPixels = scaleId2numPixels( : );
    numScale = numel( scaleId2numPixels );
    imSize = imSize( 1 : 2 ); 
    imSize = imSize( : );
    switch scalingCriteria,
        case 'MIN',
            sid2scale = scaleId2numPixels ./ min( imSize );
        case 'MAX',
            sid2scale = scaleId2numPixels ./ max( imSize );
        case 'WIDTH',
            sid2scale = scaleId2numPixels ./ imSize( 2 );
        case 'HEIGHT',
            sid2scale = scaleId2numPixels ./ imSize( 1 );
        case 'AREA',
            sid2scale = scaleId2numPixels ./ sqrt( prod( imSize( 1 : 2 ) ) );
    end;
    scaleId2imSize = round( repmat( imSize, 1, numScale ) .* repmat( sid2scale', 2, 1 ) );
end