function im = normalizeAndCropImage...
    ( image, roi, averageImage, interpolation )
    % 1. Crop image.
    [ r, c, ~ ] = size( image );
    imc = image( ...
        max( 1, roi( 1 ) ) : min( r, roi( 3 ) ), ...
        max( 1, roi( 2 ) ) : min( c, roi( 4 ) ), : );
    % 2. Normalization.
    if nargin > 2, imc = normalizeImage( imc, averageImage, interpolation ); end;
    if roi( 1 ) >= 1 && roi( 2 ) >= 1 && r >= roi( 3 ) && c >= roi( 4 ), im = imc; return; end;
    % 3. Dilate image if necessary.
    h = roi( 3 ) - roi( 1 ) + 1;
    w = roi( 4 ) - roi( 2 ) + 1;
    im = zeros( h, w, size( imc, 3 ), 'single' );
    im( max( 2 - roi( 1 ), 1 ) : min( r - roi( 3 ) + h, h ), ...
        max( 2 - roi( 2 ), 1 ) : min( c - roi( 4 ) + w, w ), : ) = imc;
end