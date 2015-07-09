function plottlbr( tlbrs, im, wait, color, cnames )
    if nargin < 4, color = 'c'; end;
    
    lineWidth = 2;
    numBox = size( tlbrs, 2 );
    imshow( im ); 
    if ~numBox, return; end;
    rects = tlbr2rect( tlbrs );
    hold on; cnt = 0;
    for i = 1 : numBox; cnt = cnt + 1;
        tlbr = tlbrs( :, i );
        rect = rects( :, i );
        if iscell( color ) && numel( color ) == 4,
            line( ...
                [ tlbr( 2 ), tlbr( 4 ) ], ...
                [ tlbr( 1 ), tlbr( 1 ) ], ...
                'color', color{ 1 }, ...
                'lineWidth', lineWidth ); % Top.
            line( ...
                [ tlbr( 2 ), tlbr( 2 ) ], ...
                [ tlbr( 1 ), tlbr( 3 ) ], ...
                'color', color{ 2 }, ...
                'lineWidth', lineWidth ); % Left.
            line( ...
                [ tlbr( 2 ), tlbr( 4 ) ], ...
                [ tlbr( 3 ), tlbr( 3 ) ], ...
                'color', color{ 3 }, ...
                'lineWidth', lineWidth ); % Bottom.
            line( ...
                [ tlbr( 4 ), tlbr( 4 ) ], ...
                [ tlbr( 1 ), tlbr( 3 ) ], ...
                'color', color{ 4 }, ...
                'lineWidth', lineWidth ); % Right.
        else
            rectangle( ...
                'Position', rect, ...
                'EdgeColor', color, ...
                'LineWidth', lineWidth );
        end;
        if nargin > 4,
            text( ...
                double( rect( 1 ) ), ...
                double( rect( 2 ) ), ...
                cnames{ cnt }, ...
                'color', 'k', ...
                'FontSize', 14, ...
                'BackgroundColor', color );
        end
        if wait, waitforbuttonpress; hold off; imshow( im ); hold on; end;
    end; 
    if nargin > 4 && ~wait, cnt = 0;
        for rect = rects; cnt = cnt + 1;
            text( ...
                double( rect( 1 ) ), ...
                double( rect( 2 ) ), ...
                cnames{ cnt }, ...
                'color', 'k', ...
                'FontSize', 14, ...
                'BackgroundColor', color );
        end;
    end; hold off;
end

