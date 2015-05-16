function Y = my_softmaxloss_bibranch(X,c,dzdy)
dim = size( X, 3 ) / 2;
if nargin <= 2
    y1 = vl_nnsoftmaxloss( X( :, :, 1 : dim, : ), c( 1, : ) );
    y2 = vl_nnsoftmaxloss( X( :, :, dim + 1 : end, : ), c( 2, : ) );
    Y = ( y1 + y2 ) / 2;
else
    y1 = vl_nnsoftmaxloss( X( :, :, 1 : dim, : ), c( 1, : ), dzdy );
    y2 = vl_nnsoftmaxloss( X( :, :, dim + 1 : end, : ), c( 2, : ), dzdy );
    Y = cat( 3, y1, y2 );
end
