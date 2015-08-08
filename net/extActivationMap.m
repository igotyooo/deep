function [ rid2tlbr, rid2desc ] = ...
    extActivationMap( ...
    originalImage, ...
    net, ...
    targetLayerId, ...
    targetImSizes, ...
    interpolationMethod, ...
    patchSide, ...
    stride )

    im = originalImage;
    avgIm = net.normalization.averageImage;
    lid = targetLayerId;
    sid2size = targetImSizes;
    itpltn = interpolationMethod;
    numScale = size( targetImSizes, 2 );
    imSize = size( im );
    imSize = imSize( 1 : 2 )';
    useGpu = isa( net.layers{ 1 }.weights{ 1 }, 'gpuArray' );
    rid2tlbr = cell( numScale, 1 );
    rid2desc = cell( numScale, 1 );
    for sid = 1 : numScale,
        imSize_ = sid2size( :, sid );
        im_ = imresize( im, imSize_', 'method', itpltn );
        im_ = single( im_ );
        avgIm_ = imresize( avgIm, imSize_', 'method', itpltn );
        im_ = im_ - avgIm_;
        if useGpu, im_ = gpuArray( im_ ); end;
        res = my_simplenn( ...
            net, im_, [  ], [  ], ...
            'accumulate', false, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'backPropDepth', +inf, ...
            'targetLayerId', lid, ...
            'sync', true ); clear im_;
        % Form activations.
        outs = gather( res( lid + 1 ).x ); clear res;
        [ nr, nc, z ] = size( outs );
        outs = reshape( permute( outs, [ 3, 1, 2 ] ), z, nr * nc );
        % Form geometries.
        r = ( ( 1 : nr ) - 1 ) * stride + 1;
        c = ( ( 1 : nc ) - 1 ) * stride + 1;
        [ c, r ] = meshgrid( c, r );
        regns = cat( 3, r, c );
        regns = cat( 3, regns, regns + patchSide - 1 );
        regns = reshape( permute( regns, [ 3, 1, 2 ] ), 4, nr * nc );
        regns = cat( 1, regns, ...
            sid * ones( 1, nr * nc  ) );
        % Back projection.
        regns = resizeTlbr( regns, imSize_, imSize );
        regns( 1 : 4, : ) = round( regns( 1 : 4, : ) );
        rid2tlbr{ sid } = regns;
        rid2desc{ sid } = outs;
    end % Next scale.
    % Aggregate for each layer.
    rid2desc = cat( 2, rid2desc{ : } );
    rid2tlbr = cat( 2, rid2tlbr{ : } );
end

