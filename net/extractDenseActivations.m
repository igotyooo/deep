function rid2out = ...
    extractDenseActivations( ...
    originalImage, ...
    network, ...
    targetLayer, ...
    targetImageSizes, ...
    regionSide, ...
    regionDilate, ...
    maximumImageSize )
    imageDilate = round( regionSide * regionDilate );
    interpolation = network.normalization.interpolation;
    averageImage = network.normalization.averageImage;
    numSize = size( targetImageSizes, 2 );
    rid2out = cell( numSize, 1 );
    for sid = 1 : numSize,
        imSize = targetImageSizes( :, sid );
        if min( imSize ) + 2 * imageDilate < regionSide, continue; end;
        if prod( imSize + imageDilate * 2 ) > maximumImageSize,
            fprintf( '%s: Warning) Im of %s rejected.\n', ...
                upper( mfilename ), mat2str( imSize ) ); continue;
        end;
        im = imresize( ...
            originalImage, imSize', ...
            'method', interpolation );
        im = single( im );
        roi = [ ...
            1 - imageDilate; ...
            1 - imageDilate; ...
            imSize( : ) + imageDilate; ];
        im = normalizeAndCropImage( ...
            im, roi, ...
            averageImage, ...
            interpolation );
        if isa( network.layers{ 1 }.weights{ 1 }, 'gpuArray' ), 
            im = gpuArray( im ); end;
        res = my_simplenn( ...
            network, im, [  ], [  ], ...
            'accumulate', false, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'backPropDepth', +inf, ...
            'targetLayerId', targetLayer, ...
            'sync', true ); clear im;
        % Form activations.
        outs = gather( res( targetLayer + 1 ).x ); clear res;
        [ nr, nc, z ] = size( outs );
        outs = reshape( permute( outs, [ 3, 1, 2 ] ), z, nr * nc );
        rid2out{ sid } = outs;
    end % Next scale.
    % Aggregate for each layer.
    rid2out = cat( 2, rid2out{ : } );
end

