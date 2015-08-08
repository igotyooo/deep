function [ patchSide, stride ] = ...
    getNetProperties( net, targetLyrId )
    patchSide = net.normalization.imageSize( 1 );
    numChannel = net.normalization.imageSize( 3 );
    % Determine patch size.
    useGpu = isa( net.layers{ 1 }.weights{ 1 }, 'gpuArray' );
    while true,
        try
            im = zeros...
                ( patchSide, patchSide, numChannel, 'single' );
            if useGpu, im = gpuArray( im ); end;
            my_simplenn( ...
                net, im, [  ], [  ], ...
                'targetLayerId', targetLyrId );
            clear im; clear ans;
            patchSide = patchSide - 1;
        catch
            patchSide = patchSide + 1;
            break;
        end;
    end
    % Determine patch stride.
    stride = 0;
    while true,
        im = zeros...
            ( patchSide + stride, patchSide, numChannel, 'single' );
        if useGpu, im = gpuArray( im ); end;
        res = my_simplenn( ...
            net, im, [  ], [  ], ...
            'targetLayerId', targetLyrId );
        desc = gather( res( targetLyrId + 1 ).x );
        clear im; clear res;
        if size( desc, 1 ) == 2, break; end;
        stride = stride + 1;
    end;
end

