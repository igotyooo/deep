function [ imgs ] = catImgs( iidLayout, iid2ifpath, cellSize )
    
    nr = size( iidLayout, 1 );
    nc = size( iidLayout, 2 );
    imgs = cell( nr, nc );
    rateCell = cellSize( 2 ) / cellSize( 1 );
    for r = 1 : nr
        for c = 1 : nc
            if isnumeric( iidLayout )
                im = imread( iid2ifpath{ iidLayout( r, c ) } );
            elseif iscell( iidLayout )
                im = iidLayout{ r, c };
            end
            imSize = size( im );
            if size( im, 3 ) == 1
                im = cat( 3, im, im, im );
            end
            rateIm = imSize( 2 ) / imSize( 1 );
            if rateCell >= rateIm
                scale = cellSize( 1 ) / imSize( 1 );
                tsize = [ cellSize( 1 ), round( scale * imSize( 2 ) ) ];
                im = imresize( im, tsize );
                margin = cellSize( 2 ) - tsize( 2 );
                if mod( margin, 2 )
                    marginLeft = ( margin - 1 ) / 2;
                    marginRight = marginLeft + 1;
                else
                    marginLeft = margin / 2;
                    marginRight = margin / 2;
                end
                imgs{ r, c } = cat( 2, uint8( ones( tsize( 1 ), marginLeft, 3 ) ) * 255, im, uint8( ones( tsize( 1 ), marginRight, 3 ) ) * 255 );
            else
                scale = cellSize( 2 ) / imSize( 2 );
                tsize = [ round( scale * imSize( 1 ) ), cellSize( 2 ) ];
                im = imresize( im, tsize );
                margin = cellSize( 1 ) - tsize( 1 );
                if mod( margin, 2 )
                    marginUp = ( margin - 1 ) / 2;
                    marginDown = marginUp + 1;
                else
                    marginUp = margin / 2;
                    marginDown = margin / 2;
                end
                imgs{ r, c } = cat( 1, uint8( ones( marginUp, tsize( 2 ), 3 ) ) * 255, im, uint8( ones( marginDown, tsize( 2 ), 3 ) ) * 255 );
            end
        end
        imgs{ r, 1 } = cat( 2, imgs{ r, : } );
    end
    imgs = cat( 1, imgs{ :, 1 } );

end