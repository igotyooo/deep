function rid2ok = imThr( im, rid2tlbr )
    thr = 20;
    im = uint8( mean( im, 3 ) );
    maskIm = im > thr;
    centerR = round( mean( rid2tlbr( [ 1, 3 ], : ), 1 ) )';
    centerC = round( mean( rid2tlbr( [ 2, 4 ], : ), 1 ) )';
    rid2idx = ( centerC - 1 ) * size( im, 1 ) + centerR;
    rid2ok = maskIm( rid2idx );
end

