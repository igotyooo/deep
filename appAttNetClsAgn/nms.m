function [ outboxes, scores, assigns ] = ...
    nms( boxes, overlap, minNumSuppBox, mergingMethod )
    if isempty( boxes ), 
        outboxes = [  ]; 
        scores = [  ]; 
        assigns = [  ]; 
        return; 
    end;
    outboxes = zeros( size( boxes, 1 ), 4 );
    assigns = cell( size( boxes, 1 ), 1 );
    scores = zeros( size( boxes, 1 ), 1 );
    x1 = boxes( :, 1 );
    y1 = boxes( :, 2 );
    x2 = boxes( :, 3 );
    y2 = boxes( :, 4 );
    s = boxes( :, end );
    area = ( x2 - x1 + 1 ) .* ( y2 - y1 + 1 );
    [ ~, I ] = sort( s );
    pick = s * 0;
    counter = 1;
    while ~isempty( I )
        last = length( I );
        i = I( last );
        xx1 = max( x1( i ), x1( I( 1 : last - 1 ) ) );
        yy1 = max( y1( i ), y1( I( 1 : last - 1 ) ) );
        xx2 = min( x2( i ), x2( I( 1 : last - 1 ) ) );
        yy2 = min( y2( i ), y2( I( 1 : last - 1 ) ) );
        w = max( 0.0, xx2 - xx1 + 1 );
        h = max( 0.0, yy2 - yy1 + 1 );
        o = w .* h ./ area( I( 1 : last - 1 ) );
        supp = find( o > overlap );
        if numel(  supp  ) >= minNumSuppBox,
            pick( counter ) = i;
            idx = [ last; supp ];
            ss = boxes( I( idx ), end );
            switch mergingMethod,
                case 'WAVG',
                    outboxes( counter, : ) = ...
                        sum( bsxfun( @times, boxes( I( idx ), 1 : 4 ), boxes( I( idx ), end ) ), 1 ) / ...
                        sum( boxes( I( idx ), end ) );
                    scores( counter ) = mean( ss );
                case 'MAX',
                    outboxes( counter, : ) = boxes( i, 1 : 4 );
                    scores( counter ) = max( ss );
            end
            assigns{ counter } = I( idx );
            I( idx ) = [  ];
            counter = counter + 1;
        else
            I( last ) = [  ];
        end
    end
    if counter == 1,
        outboxes = [  ]; scores = [  ]; assigns = [  ];
    else
        outboxes = outboxes( 1 : ( counter - 1 ), : );
        assigns = assigns( 1 : counter - 1 );
        scores = scores( 1 : counter - 1 );
    end
end