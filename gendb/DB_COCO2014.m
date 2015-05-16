function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox ] = DB_COCO2014

    global path;
    srcpath = path.db_coco2014.root;

    % Convert json to mat.
    setnames = { 'train2014', 'val2014' };
    for set = setnames
        % Read json.
        fprintf( 'DB_COCO2014: [%s] Read json annotations.\n', ...
            upper( set{ : } ) );
        annopath = fullfile( srcpath, 'annotations', ...
            sprintf( 'instances_%s.json', set{ : } ) );
        fid = fopen( annopath );
        data = textscan( fid, '%s', 'Delimiter', ...
            { ',', ': ', '[', ']', '{', '}' } );
        fclose( fid );
        data = data{ 1 }( cellfun( @numel, data{ 1 } ) ~= 0 );
        % Image info.
        fprintf( 'DB_COCO2014: [%s] Extract image info.\n', ...
            upper( set{ : } ) );
        imlist = ismember( data, '"license"' );
        imlines = find( imlist );
        imnumline = imlines( 2 ) - imlines( 1 );
        imlines = imlines( 1 ) : imlines( end ) + imnumline - 1;
        imlist = data( imlines );
        images.id = imlist...
            ( find( ismember( imlist, '"id"' ) ) + 1 );
        images.id = cellfun( @str2double, images.id );
        images.id = uint32( images.id );
        images.impath = imlist...
            ( find( ismember( imlist, '"file_name"' ) ) + 1 );
        images.impath = cellfun( ...
            @( x )x( 2 : end - 1 ), ...
            images.impath, ...
            'UniformOutput', false );
        images.impath = fullfile...
            ( srcpath, 'images', set{ : }, images.impath );
        images.size = imlist...
            ( find( ismember( imlist, '"height"' ) ) + 1 );
        images.size = cellfun( @str2double, images.size );
        images.size = cat( 2, images.size, ...
            cellfun( @str2double, ...
            imlist( find( ismember( imlist, '"width"' ) ) + 1 ) ) )';
        images.size = uint32( images.size );
        % Category info.
        fprintf( 'DB_COCO2014: [%s] Extract category info.\n', ...
            upper( set{ : } ) );
        catlist = ismember( data, '"name"' );
        catlines = find( catlist );
        catlines = ( catlines( 1 ) - 2 ) : catlines( end ) + 1;
        catlist = data( catlines );
        categories.id = catlist...
            ( find( ismember( catlist, '"id"' ) ) + 1 );
        categories.id = cellfun...
            ( @str2double, categories.id );
        categories.id = uint32( categories.id );
        categories.name = catlist...
            ( find( ismember( catlist, '"name"' ) ) + 1 );
        categories.name = cellfun( ...
            @( x )x( 2 : end - 1 ), ...
            categories.name, ...
            'UniformOutput', false );
        % Instance info.
        fprintf( 'DB_COCO2014: [%s] Extract instance info.\n', ...
            upper( set{ : } ) );
        inslines = ( find( ismember( data, '"instances"' ) ) + 1 )...
            : numel( data );
        inslist = data( inslines );
        instances.id = inslist...
            ( find( ismember( inslist, '"id"' ) ) + 1 );
        instances.id = cellfun...
            ( @str2double, instances.id );
        instances.id = uint32( instances.id );
        instances.iid = inslist...
            ( find( ismember( inslist, '"image_id"' ) ) + 1 );
        instances.iid = cellfun...
            ( @str2double, instances.iid );
        instances.iid = uint32( instances.iid );
        instances.cid = inslist...
            ( find( ismember( inslist, '"category_id"' ) ) + 1 );
        instances.cid = cellfun...
            ( @str2double, instances.cid );
        instances.cid = uint32( instances.cid );
        bbidx = find( ismember( inslist, '"bbox"' ) );
        bbidx = cat( 2, ...
            bbidx + 1, bbidx + 2, ...
            bbidx + 3, bbidx + 4 );
        instances.bbox = cellfun...
            ( @str2double, inslist( bbidx ) );
        instances.bbox = single( instances.bbox );
        data_.images = images;
        data_.categories = categories;
        data_.instances = instances;
        anno.( set{ : } ) = data_;
        fprintf( 'DB_COCO2014: [%s] Done.\n', upper( set{ : } ) );
    end;

    % Convert raw annotation to my form.
    for set = setnames;
        fprintf( 'DB_COCO2014: [%s] Convert raw anno to my anno.\n', ...
            upper( set{ : } ) );
        images = anno.( set{ : } ).images;
        categories = anno.( set{ : } ).categories;
        instances = anno.( set{ : } ).instances;
        % Reorder ids.
        newiid2iid = images.id;
        iid2newiid = zeros( max( newiid2iid ), 1, 'uint32' );
        iid2newiid( newiid2iid ) = 1 : numel( newiid2iid );
        newcid2cid = categories.id;
        cid2newcid = zeros( max( newcid2cid ), 1, 'uint32' );
        cid2newcid( newcid2cid ) = 1 : numel( newcid2cid );
        % Form object info.
        oid2cid = cid2newcid( instances.cid );
        oid2iid = iid2newiid( instances.iid );
        oid2bbox = instances.bbox;
        oid2bbox = cat( 2, ...
            oid2bbox( :, 2 ), ...
            oid2bbox( :, 1 ), ...
            oid2bbox( :, 2 ) + oid2bbox( :, 4 ), ...
            oid2bbox( :, 1 ) + oid2bbox( :, 3 ) )';
        % Form image info.
        iid2impath = images.impath;
        iid2size = images.size;
        switch set{ : }( 1 : 3 )
            case 'tra'
                iid2setid = ones( size( iid2impath ) );
            case 'val'
                iid2setid = 2 * ones( size( iid2impath ) );
        end
        % Form class info.
        cid2name = categories.name;
        % Merge info.
        myanno.( set{ : } ).iid2setid = iid2setid;
        myanno.( set{ : } ).iid2impath = iid2impath;
        myanno.( set{ : } ).iid2size = iid2size;
        myanno.( set{ : } ).cid2name = cid2name;
        myanno.( set{ : } ).oid2iid = oid2iid;
        myanno.( set{ : } ).oid2cid = oid2cid;
        myanno.( set{ : } ).oid2bbox = oid2bbox;
    end;
    % Merge train and val.
    fprintf( 'DB_COCO2014: Merge sets.\n' );
    numim = numel( myanno.( setnames{ 1 } ).iid2setid );
    myanno.( setnames{ 2 } ).oid2iid = arrayfun( ...
        @( x )plus( x, numim ), ...
        myanno.( setnames{ 2 } ).oid2iid );
    iid2setid = vertcat( ...
        myanno.( setnames{ 1 } ).iid2setid, ...
        myanno.( setnames{ 2 } ).iid2setid );
    iid2impath = vertcat( ...
        myanno.( setnames{ 1 } ).iid2impath, ...
        myanno.( setnames{ 2 } ).iid2impath );
    iid2size = horzcat( ...
        myanno.( setnames{ 1 } ).iid2size, ...
        myanno.( setnames{ 2 } ).iid2size );
    if ~any( ~cellfun( ...
            @( x, y )strcmp( x, y ), ...
            myanno.train2014.cid2name, ...
            myanno.val2014.cid2name ) )
        cid2name = myanno.train2014.cid2name;
    else
        error( 'Inconsistent class.\n' );
    end
    oid2iid = vertcat( ...
        myanno.( setnames{ 1 } ).oid2iid, ...
        myanno.( setnames{ 2 } ).oid2iid );
    oid2cid = vertcat( ...
        myanno.( setnames{ 1 } ).oid2cid, ...
        myanno.( setnames{ 2 } ).oid2cid );
    oid2bbox = horzcat( ...
        myanno.( setnames{ 1 } ).oid2bbox, ...
        myanno.( setnames{ 2 } ).oid2bbox );
    oid2diff = false( size( oid2iid ) );
    fprintf( 'Done.\n' );

end

