function [ imageSize, boxes, image ] = normalizeImageSize( maxSide, imageSize0, boxes0, image0 )
    maxSide0 = max( imageSize0( 1 ), imageSize0( 2 ) );
    imageSize = [ imageSize0( 1 ); imageSize0( 2 ); ] / maxSide0 * maxSide;
    if nargin > 2, boxes = round( resizeTlbr( boxes0, imageSize0, imageSize ) ); end;
    imageSize = round( imageSize );
    if nargin > 3, image = imresize( image0, imageSize', 'method', 'bicubic' ); end;
end