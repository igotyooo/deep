function str = serializeCellStr( strs )
    if ~iscell( strs ), strs = { strs }; end;
    str = cellfun( @( str )[ ', ', [ upper( str( 1 ) ), str( 2 : end ) ] ], strs, 'UniformOutput', false );
    str = cat( 2, str{ : } );
    str = [ str( 3 : end ), '.' ];
end

