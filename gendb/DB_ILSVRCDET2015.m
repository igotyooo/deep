function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox, ...
            oid2cont ] = DB_ILSVRCDET2015

    global path;    
    dirRoot = path.db.ilsvrcdet2015.root;
    fprintf( 'Read training image list.\n' );
    fp = fopen( fullfile( dirRoot, 'ImageSets/DET/train.txt' ), 'r' );
    iid2imnameTr = textscan( fp, '%s %d' );
    fclose( fp );
    iid2imnameTr = iid2imnameTr{ 1 };
    iid2impathTr = cell( size( iid2imnameTr ) );
    iid2annopathTr = cell( size( iid2imnameTr ) );
    parfor iid = 1 : numel( iid2imnameTr ),
        iid2impathTr{ iid } = fullfile( dirRoot, 'Data/DET/train', iid2imnameTr{ iid } );
        iid2annopathTr{ iid } = fullfile( dirRoot, 'Annotations/DET/train', iid2imnameTr{ iid } );
    end;
    fprintf( 'Done.\n' );
    fprintf( 'Read validation image list.\n' );
    fp = fopen( fullfile( dirRoot, 'ImageSets/DET/val.txt' ), 'r' );
    iid2imnameVal = textscan( fp, '%s %d' );
    fclose( fp );
    iid2imnameVal = iid2imnameVal{ 1 };
    iid2impathVal = cell( size( iid2imnameVal ) );
    iid2annopathVal = cell( size( iid2imnameVal ) );
    parfor iid = 1 : numel( iid2imnameVal ),
        iid2impathVal{ iid } = fullfile( dirRoot, 'Data/DET/val', iid2imnameVal{ iid } );
        iid2annopathVal{ iid } = fullfile( dirRoot, 'Annotations/DET/val', iid2imnameVal{ iid } );
    end;
    fprintf( 'Done.\n' );
    fprintf( 'Form file path.\n' );
    iid2impath = cat( 1, iid2impathTr, iid2impathVal );
    iid2annopath = cat( 1, iid2annopathTr, iid2annopathVal );
    iid2setid = cat( 1, ones( size( iid2impathTr ) ), 2 * ones( size( iid2impathVal ) ) );
    parfor iid = 1 : numel( iid2impath ),
        iid2impath{ iid } = strcat( iid2impath{ iid }, '.JPEG' );
        iid2annopath{ iid } = strcat( iid2annopath{ iid }, '.xml' );
    end;
    fprintf( 'Done.\n' );
    clearvars -except iid2impath iid2annopath iid2setid path dirRoot;
    fprintf( 'Read class information.\n' );
    synsets = load( fullfile( path.lib.ilsvrcDevKit, 'data/meta_det.mat' ) );
    synsets = synsets.synsets;
    cid2name = { synsets.name }';
    cid2wnid = { synsets.WNID }';
    cid2wnid = cellfun( @( x )str2double( x( 2 : end ) ), cid2wnid );
    wnid2cid = zeros( max( cid2wnid ), 1 );
    wnid2cid( cid2wnid ) = 1 : numel( cid2wnid );
    fprintf( 'Done.\n' );
    fprintf( 'Read bounding boxes.\n' );
    numIm = numel( iid2impath );
    iid2size = zeros( 2, numIm );
    oid2wnid = cell( numIm, 1 );
    oid2bbox = cell( numIm, 1 );
    iid2numObj = zeros( numIm, 1 );
    iid2pos = false( numIm, 1 );
    fprintf( 'Read xmls.\n' );
    parfor iid = 1 : numIm,
        try
            annopath = iid2annopath{ iid };
            anno = VOCreadxml( annopath );
            anno = anno.annotation;
            oid2wnid{ iid } = { anno.object.name }';
            oid2bbox{ iid } = { anno.object.bndbox }';
            iid2numObj( iid ) = numel( oid2bbox{ iid } );
            iid2size( :, iid ) = [ str2double( anno.size.height ); str2double( anno.size.width ); ];
            iid2pos( iid ) = true;
        catch
            im = imread( iid2impath{ iid } );
            iid2size( :, iid ) = [ size( im, 1 ); size( im, 2 ); ];
            fprintf( 'Read xmls: No obj in iid %06d.\n', iid );
        end;
    end;
    iid2ok = iid2pos | ( iid2setid == 2 );
    iid2impath = iid2impath( iid2ok );
    iid2setid = iid2setid( iid2ok );
    iid2numObj = iid2numObj( iid2ok );
    iid2size = iid2size( :, iid2ok );
    oid2wnid = oid2wnid( iid2ok );
    oid2bbox = oid2bbox( iid2ok );
    oid2wnid = cat( 1, oid2wnid{ : } );
    oid2bbox = cat( 1, oid2bbox{ : } );
    numIm = numel( iid2impath );
    oid2iid = cell( numIm, 1 );
    parfor iid = 1 : numIm,
        oid2iid{ iid } = iid * ones( iid2numObj( iid ), 1 );
    end;
    oid2iid = cat( 1, oid2iid{ : } );
    fprintf( 'Done.\n' );
    fprintf( 'Form bbox.\n' );
    numObj = numel( oid2wnid );
    oid2cid = zeros( numObj, 1 );
    parfor oid = 1 : numObj,
        wnid = str2double( oid2wnid{ oid }( 2 : end ) );
        obj = oid2bbox{ oid };
        oid2cid( oid ) = wnid2cid( wnid );
        oid2bbox{ oid } = [ ...
            str2double( obj.ymin ); ...
            str2double( obj.xmin ); ...
            str2double( obj.ymax ); ...
            str2double( obj.xmax ); ];
    end;
    oid2bbox = cat( 2, oid2bbox{ : } );
    oid2bbox = oid2bbox + 1;
    fprintf( 'Done.\n' );
    fprintf( 'Reject images if defined as "part".\n' );
    iid2ispart = false( numIm, 1 );
    iid2name = cell( numIm, 1 );
    parfor iid = 1 : numIm,
        [ ~, iid2name{ iid } ] = fileparts( iid2impath{ iid } );
    end;
    numClass = numel( cid2name );
    partImList = cell( numClass, 1 );
    imSetPath = dir( fullfile( dirRoot, 'ImageSets/DET/train_*.txt' ) );
    parfor cid = 1 : numClass,
        fp = fopen( fullfile( dirRoot, 'ImageSets/DET/', imSetPath( cid ).name ) );
        data = textscan( fp, '%s %d' );
        fclose( fp );
        partImList{ cid } = data{ 1 }( ( data{ 2 } == 0 ) );
    end;
    partImList = cat( 1, partImList{ : } );
    partImList = unique( partImList );
    numPartIm = numel( partImList );
    for pid = 1 : numPartIm,
        [ ~, part ] = fileparts( partImList{ pid } );
        partiid = ismember( iid2name, part );
        if sum( partiid ) ~= 1, error( 'Error.\n' ); end;
        iid2ispart( partiid ) = true;
    end;
    newiid2iid = setdiff( ( 1 : numIm )', find( iid2ispart ) );
    iid2newiid = zeros( numIm, 1 );
    iid2newiid( newiid2iid ) = 1 : numel( newiid2iid );
    iid2impath = iid2impath( newiid2iid );
    iid2size = iid2size( :, newiid2iid );
    iid2setid = iid2setid( newiid2iid );
    newoid2oid = find( ~ismember( oid2iid, find( iid2ispart ) ) );
    oid2cid = oid2cid( newoid2oid );
    oid2iid = iid2newiid( oid2iid( newoid2oid ) );
    oid2bbox = oid2bbox( :, newoid2oid );
    fprintf( 'Done.\n' );
    % Reject wrong bounding boxes.
    oid2nr = oid2bbox( 3, : ) - oid2bbox( 1, : ) + 1;
    oid2nc = oid2bbox( 4, : ) - oid2bbox( 2, : ) + 1;
    oid2wrong = oid2nr <= 1 | oid2nc <= 1;
    newoid2oid = find( ~oid2wrong );
    oid2cid = oid2cid( newoid2oid );
    oid2iid = oid2iid( newoid2oid );
    oid2bbox = oid2bbox( :, newoid2oid );
    oid2cont = cell( size( oid2cid ) );
    oid2diff = false( size( oid2cid ) );
    % Data type conversion.
    iid2size = single( iid2size );
    iid2setid = single( iid2setid );
    oid2cid = single( oid2cid );
    oid2iid = single( oid2iid );
    oid2bbox = single( oid2bbox );
end

