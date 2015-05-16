% 1. Y = MY_NNEUCLOSS_REG( X, GT ) applies the euclidian loss to the data X, such as Y=|X-GT|.
% 	X has dimension H x W x D x N, packing N arrays of W x H D-dimensional vectors.
%   GT contains the ground-truth vector, which should be fix-length vectors of single type.
%   GT can be an array with H x W x 1 x N dimensions. H, W and D can be any integer number.
% 2. DZDX = MY_NNEUCLOSS_REG( X, GT, DZDY ) computes the derivative dZ/dX of the
%   CNN with respect to the input X given the derivative dZ/dY with respect to the block output Y.
%   dZ/dX has the same dimension as X.
function Y = my_nneucloss_reg( X, gt, dzdy )
    n = numel( X( :, :, :, 1 ) );
    if nargin <= 2
        Y = sum( sum( sum( ( X - gt ) .^ 2, 1 ), 2 ), 3 ) / n;
    else
        % dZ/dX = dZ/dY * dY/dX
        Y = dzdy * ( 2 * ( X - gt ) ) / n;
    end
end
