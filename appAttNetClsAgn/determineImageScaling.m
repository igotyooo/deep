function [ sid2hs, sid2ws ] = determineImageScaling...
    ( oid2tlbr, numSize, referenceSide, displayResult )
    oid2h = ( oid2tlbr( 3, : ) - oid2tlbr( 1, : ) + 1 )';
    oid2w = ( oid2tlbr( 4, : ) - oid2tlbr( 2, : ) + 1 )';
    oid2hs = referenceSide ./ oid2h;
    oid2ws = referenceSide ./ oid2w;
    oid2hls = log10( oid2hs );
    oid2wls = log10( oid2ws );
    numTrial = 3;
    e = +Inf;
    for i = 1 : numTrial,
        [ ~, c_, e_ ] = kmeans( ...
            [ oid2hls, oid2wls ], numSize, ...
            'MaxIter', 1000 );
        e_ = sum( e_ );
        if e > e_, e = e_; c = c_; end;
    end;
    sid2hls = c( :, 1 );
    sid2wls = c( :, 2 );
    sid2hs = 10 .^ sid2hls;
    sid2ws = 10 .^ sid2wls;
    if displayResult,
        figure;
        plot( oid2wls, oid2hls, 'k.' ); hold on;
        plot( sid2wls, sid2hls, 'rx' );
        xlabel( 'log( width scaling )' );
        ylabel( 'log( height scaling )' );
        title( sprintf( 'Determine image scaling (k=%d)', numSize ) );
        legend( { 'data', 'center' }, 'location', 'best' );
        set( gcf, 'color', 'w' );
        grid on; hold off;
        figure;
        plot( oid2ws, oid2hs, 'k.' ); hold on;
        plot( sid2ws, sid2hs, 'rx' );
        xlabel( 'width scaling' );
        ylabel( 'height scaling' );
        title( sprintf( 'Determine image scaling (k=%d)', numSize ) );
        legend( { 'data', 'center' }, 'location', 'best' );
        set( gcf, 'color', 'w' );
        grid on; hold off;
    end;
end