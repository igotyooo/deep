function [ rid2geo, rid2desc ] = ...
    extActivationMap( ...
    image, ...
    averageImage, ...
    net, ...
    targetLayerId, ...
    targetImSizes, ...
    interpolationMethod, ...
    patchSize, ...
    stride, ...
    regionFilteringFunctionName )

    im = image;
    avgIm = averageImage;
    lid = targetLayerId;
    sid2size = targetImSizes;
    itpltn = interpolationMethod;
    funName = regionFilteringFunctionName;
    numSize = size( targetImSizes, 2 );
    orgImSize = size( im );
    sid2rid2geo = cell( numSize, 1 );
    sid2rid2desc = cell( numSize, 1 );
    for sid = 1 : numSize,
        im_ = imresize( im, sid2size( :, sid )', 'method', itpltn );
        im_ = single( im_ );
        imSize_ = size( im_ );
        r = imSize_( 1 ); c = imSize_( 2 );
        avgIm_ = imresize( avgIm, [ r, c ], 'method', itpltn );
        im_ = im_ - avgIm_;
        if isa( net.layers{ 1 }.weights{ 1 }, 'gpuArray' ), im_ = gpuArray( im_ ); end;
        respns = my_simplenn( ...
            net, im_, [  ], [  ], ...
            'accumulate', false, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'backPropDepth', +inf, ...
            'targetLayerId', lid, ...
            'sync', true ); clear im_;
        % Form descriptors.
        desc = respns( lid + 1 ).x;
        [ r, c, z ] = size( desc );
        depthid = ( ( ( 1 : z ) - 1 ) * ( r * c ) )';
        rcid = 1 : ( r * c );
        [ depthid, rcid ] = meshgrid( depthid, rcid );
        idx = reshape( ( depthid + rcid )', z * r * c, 1 );
        rid2desc = gather( reshape( desc( idx ), z, r * c ) );
        sid2rid2desc{ sid } = rid2desc; clear respns;
        % Form geometries.
        [ cs, rs ] = meshgrid( 1 : c, 1 : r );
        geo = cat( 3, rs, cs, rs, cs, sid * ones( r, c ) );
        [ r, c, z ] = size( geo );
        depthid = ( ( ( 1 : z ) - 1 ) * ( r * c ) )';
        rcid = 1 : ( r * c );
        [ depthid, rcid ] = meshgrid( depthid, rcid );
        idx = reshape( ( depthid + rcid )', z * r * c, 1 );
        geo = reshape( geo( idx ), z, r * c );
        geo( 1, : ) = ( geo( 1, : ) - 1 ) * stride( 1 ) + 1;
        geo( 2, : ) = ( geo( 2, : ) - 1 ) * stride( 2 ) + 1;
        geo( 3, : ) = geo( 1, : ) + patchSize( 1 ) - 1;
        geo( 4, : ) = geo( 2, : ) + patchSize( 2 ) - 1;
        geo( 1 : 4, : ) = resizeTlbr( geo( 1 : 4, : ), imSize_, orgImSize );
        geo( 1 : 2, : ) = max( 1, geo( 1 : 2, : ) );
        geo( 3, : ) = min( orgImSize( 1 ), geo( 3, : ) );
        geo( 4, : ) = min( orgImSize( 2 ), geo( 4, : ) );
        sid2rid2geo{ sid } = round( geo );
    end % Next scale.
    % Aggregate for each layer.
    rid2desc = cat( 2, sid2rid2desc{ : } );
    rid2geo = cat( 2, sid2rid2geo{ : } );
    % Filtering if needed.
    if ~isempty( funName )
        fh = str2func( funName );
        rid2ok = fh( im, rid2geo );
        rid2geo = rid2geo( :, rid2ok );
        rid2desc = rid2desc( :, rid2ok );
    end;
end

