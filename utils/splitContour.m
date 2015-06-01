function c_ = splitContour( c )
    n = 0;
    c_ = {  };
    while ~isempty( c );
        n = n + 1;
        numItem = c( 2, 1 );
        c_{ n } = c( :, 2 : 1 + numItem );
        c( :, 1 : 1 + numItem ) = [  ];
    end
end

