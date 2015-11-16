function setFigPos( figh, position )
    nr = position( 1 );
    nc = position( 2 );
    tr = position( 3 );
    tc = position( 4 );
    tr = nr - tr + 1;
    scr = get( 0, 'screensize' );
    rstrd = floor( scr( 4 ) / nr );
    cstrd = floor( scr( 3 ) / nc );
    rtl = ( tr - 1 ) * rstrd + 1;
    ctl = ( tc - 1 ) * cstrd + 1;
    rbr = rtl + rstrd - 1;
    cbr = ctl + cstrd - 1;
    set( figh, 'Position', [ ctl, rtl, cbr - ctl, rbr - rtl ] );
end

