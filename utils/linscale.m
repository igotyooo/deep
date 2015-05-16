function dst = linscale( src )
    minval = min( min( min( src ) ) );
    dst = src - minval;
    maxval = max( max( max( dst ) ) );
    dst = dst / maxval;
end

