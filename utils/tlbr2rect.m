function rect = tlbr2rect( tlbr ), rect = [ flipud( tlbr( 1 : 2, : ) ); flipud( tlbr( 3 : 4, : ) - tlbr( 1 : 2, : ) ) ]; end
