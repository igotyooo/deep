function Y = my_softmax_bibranch(X,dzdY)
dim = size( X, 3 ) / 2;
if nargin <= 1
    y1 = vl_nnsoftmax( X( :, :, 1 : dim, : ) );
    y2 = vl_nnsoftmax( X( :, :, dim + 1 : end, : ) );
else
    y1 = vl_nnsoftmaxloss( X( :, :, 1 : dim, : ), dzdY );
    y2 = vl_nnsoftmaxloss( X( :, :, dim + 1 : end, : ), dzdY );
end
Y = cat( 3, y1, y2 );