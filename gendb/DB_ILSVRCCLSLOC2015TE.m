function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox, ...
            oid2cont ] = DB_ILSVRCCLSLOC2015TE
    global path;    
    fprintf( 'Read test images.\n' );
    dirRoot = path.db.ilsvrcclsloc2015te.root;
    iid2impath = dir( fullfile( dirRoot, 'Data/CLS-LOC/test/*.JPEG' ) );
    iid2impath = { iid2impath.name }';
    iid2impath = fullfile( dirRoot, 'Data/CLS-LOC/test', iid2impath );
    numIm = numel( iid2impath );
    iid2size = zeros( 2, numIm );
    parfor iid = 1 : numIm,
        try
            [ r, c, ~ ] = size( imread( iid2impath{ iid } ) );
            iid2size( :, iid ) = [ r; c; ];
        catch
            error( 'Cannot open IID%d.', iid );
        end;
    end;
    fprintf( 'Done.\n' );
    fprintf( 'Read class information.\n' );
    synsets = load( fullfile( path.lib.ilsvrcDevKit, 'data/meta_clsloc.mat' ) );
    synsets = synsets.synsets;
    cid2name = { synsets.name }';
    fprintf( 'Done.\n' );
    iid2setid = 3 * ones( size( iid2impath ) );
    oid2cid = zeros( 0, 1 );
    oid2iid = zeros( 0, 1 );
    oid2diff = false( 0, 1 );
    oid2bbox = zeros( 4, 0 );
    oid2cont = cell( 0, 1 );
    iid2size = single( iid2size );
    iid2setid = single( iid2setid );
    oid2cid = single( oid2cid );
    oid2iid = single( oid2iid );
    oid2bbox = single( oid2bbox );
end

