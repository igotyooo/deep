function idx = tlbr2idx( imSize, tlbr )
    imr = imSize( 1 ); 
    imc = imSize( 2 ); 
    imz = imSize( 3 );
    tlr = tlbr( 1 );
    tlc = tlbr( 2 );
    brr = tlbr( 3 );
    brc = tlbr( 4 );
    bias = imr * ( tlc - 1 );
    ( tlbr( 1 ) : tlbr( 3 ) ) + 
end

