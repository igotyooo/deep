function bbox = contour2bbox( cont )

    cont = flipud( cont ); % Order of rc.
    tl = min( cont, [  ], 2 );
    br = max( cont, [  ], 2 );
    bbox = [ tl; br; ];

end

