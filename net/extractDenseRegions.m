function rid2tlbr = ...
    extractDenseRegions( ...
    originalImageSize, ...
    targetImageSizes, ...
    regionSide, ...
    regionStride, ...
    regionDilate, ...
    maximumImageSize )
    imageDilate = round( regionSide * regionDilate );
    imSize0 = originalImageSize( : );
    numSize = size( targetImageSizes, 2 );
    rid2tlbr = cell( numSize, 1 );
    for sid = 1 : numSize,
        imSize = targetImageSizes( :, sid );
        if min( imSize ) + 2 * imageDilate < regionSide, continue; end;
        if prod( imSize + imageDilate * 2 ) > maximumImageSize, continue; end;
        roi = [ ...
            1 - imageDilate; ...
            1 - imageDilate; ...
            imSize( : ) + imageDilate; ];
        % Form geometries.
        r = roi( 1 ) : regionStride : ( roi( 3 ) - regionSide + 1 );
        c = roi( 2 ) : regionStride : ( roi( 4 ) - regionSide + 1 );
        nr = numel( r );
        nc = numel( c );
        [ c, r ] = meshgrid( c, r );
        regns = cat( 3, r, c );
        regns = cat( 3, regns, regns + regionSide - 1 );
        regns = reshape( permute( regns, [ 3, 1, 2 ] ), 4, nr * nc );
        regns = cat( 1, regns, ...
            sid * ones( 1, nr * nc  ) );
        % Back projection.
        regns = resizeTlbr( regns, imSize, imSize0 );
        regns( 1 : 4, : ) = round( regns( 1 : 4, : ) );
        rid2tlbr{ sid } = regns;
    end % Next scale.
    % Aggregate for each layer.
    rid2tlbr = cat( 2, rid2tlbr{ : } );
end

