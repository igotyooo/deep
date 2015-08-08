function augIms = augmentImages( ...
    images, ...
    augmentationType, ...
    numAugmentation, ...
    interpolation, ...
    srcSide, ...
    srcChannel, ...
    dstSide, ...
    dstChannel, ...
    keepAspect )
    if ~iscell( images ), images = { images }; end;
    switch augmentationType
        case 'NONE'
            aid2tf = [ ...
                0.5;
                0.5;
                0.0; ];
        case 'F5'
            aid2tf = [ ...
                0.5, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0;
                0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0;
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0; ];
        case 'F25'
            [ tx, ty ] = meshgrid( linspace( 0, 1, 5 ) );
            tfs = [ tx( : )'; ty( : )'; zeros( 1, numel( tx ) ) ];
            tfs_ = tfs; tfs_( 3, : ) = 1;
            aid2tf = [ tfs, tfs_ ] ;
    end
    % For each image, randomly select augmentations to be done.
    [ ~, iidx2aid ] = sort...
        ( rand( size( aid2tf, 2 ), numel( images ) ), 1 );
    dstIidx = 1;
    augIms = zeros...
        ( dstSide, dstSide, srcChannel, numel( images ), 'uint8' );
    % For each image,
    for iidx = 1 : numel( images )
        im = images{ iidx };
        % Just resize an image to srcSide.
        w = size( im, 2 ); h = size( im, 1 );
        if keepAspect,
            factor = max( srcSide / h, srcSide / w );
            if any( abs( factor - 1 ) > 0.0001 )
                im = imresize...
                    ( im, 'scale', factor, 'method', interpolation );
            end
        else
            if w ~= srcSide || h ~= srcSide,
                im = imresize...
                    ( im, [ srcSide, srcSide ], 'method', interpolation );
            end
        end
        % Augment from an image to multiple images.
        w = size( im, 2 ); h = size( im, 1 );
        for aidx = 1 : numAugmentation,
            aid = iidx2aid( aidx, iidx );
            tf = aid2tf( :, aid );
            % Crop.
            dx = floor( ( w - dstSide ) * tf( 2 ) ); % Orientation of x-axis. ( = bias of x )
            dy = floor( ( h - dstSide ) * tf( 1 ) ); % Orientation of y-axis. ( = bias of y )
            sx = ( 1 : dstSide ) + dx;
            sy = ( 1 : dstSide ) + dy;
            % Flip.
            if tf( 3 ), sx = fliplr( sx ); end
            im_ = im( sy, sx, : );
            % Depth resize.
            if dstChannel == 3 && size( im_, 3 ) == 1
                im_ = cat( 3, im_, im_, im_ );
            elseif dstChannel == 1 && size( im_, 3 ) == 3
                im_ = mean( im_, 3 );
            end
            augIms( :, :, :, dstIidx ) = im_;
            dstIidx = dstIidx + 1;
        end
    end
end