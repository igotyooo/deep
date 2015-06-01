function [ cid2name, iid2impath, iid2size, iid2setid, oid2cid, oid2diff, oid2iid, oid2bbox ]...
    = DB_DDSM

    global path;
    rootdir = path.db.ddsm.root;
    
    dir_bentr = fullfile( rootdir, 'tr', 'benign' );
    dir_cantr = fullfile( rootdir, 'tr', 'cancer' );
    dir_nortr = fullfile( rootdir, 'tr', 'normal' );
    
    dir_bente = fullfile( rootdir, 'te', 'benign' );
    dir_cante = fullfile( rootdir, 'te', 'cancer' );
    dir_norte = fullfile( rootdir, 'te', 'normal' );
    
    cid2name = { 'benign'; 'cancer'; 'normal'; };

    iid2impath_bentr = dir( dir_bentr );
    iid2impath_bentr = fullfile( dir_bentr, { iid2impath_bentr( 3 : end ).name }' );
    iid2impath_cantr = dir( dir_cantr );
    iid2impath_cantr = fullfile( dir_cantr, { iid2impath_cantr( 3 : end ).name }' );
    iid2impath_nortr = dir( dir_nortr );
    iid2impath_nortr = fullfile( dir_nortr, { iid2impath_nortr( 3 : end ).name }' );
    
    iid2impath_bente = dir( dir_bente );
    iid2impath_bente = fullfile( dir_bente, { iid2impath_bente( 3 : end ).name }' );
    iid2impath_cante = dir( dir_cante );
    iid2impath_cante = fullfile( dir_cante, { iid2impath_cante( 3 : end ).name }' );
    iid2impath_norte = dir( dir_norte );
    iid2impath_norte = fullfile( dir_norte, { iid2impath_norte( 3 : end ).name }' );
    
    iid2impath = cat( 1, ...
        iid2impath_bentr, iid2impath_cantr, iid2impath_nortr, ...
        iid2impath_bente, iid2impath_cante, iid2impath_norte );
    
    numIm = numel( iid2impath );
    iid2size = zeros( 2, numIm );
    for iid = 1 : numIm,
        fprintf( 'Read im %d/%d.\n', iid, numIm );
        im = imread( iid2impath{ iid } );
        [ r, c, ~ ] = size( im );
        iid2size( :, iid ) = [ r; c; ];
    end
    
    oid2cid_bentr = 1 * ones( size( iid2impath_bentr ) );
    oid2cid_cantr = 2 * ones( size( iid2impath_cantr ) );
    oid2cid_nortr = 3 * ones( size( iid2impath_nortr ) );
    
    oid2cid_bente = 1 * ones( size( iid2impath_bente ) );
    oid2cid_cante = 2 * ones( size( iid2impath_cante ) );
    oid2cid_norte = 3 * ones( size( iid2impath_norte ) );
    
    oid2cid = cat( 1, ...
        oid2cid_bentr, oid2cid_cantr, oid2cid_nortr, ...
        oid2cid_bente, oid2cid_cante, oid2cid_norte );
    
    oid2diff = false( numIm, 1 );
    oid2iid = ( 1 : numIm )';
    oid2bbox = cat( 1, ones( 2, numIm ), iid2size );

    iid2setid_bentr = 1 * ones( size( iid2impath_bentr ) );
    iid2setid_cantr = 1 * ones( size( iid2impath_cantr ) );
    iid2setid_nortr = 1 * ones( size( iid2impath_nortr ) );
    
    iid2setid_bente = 2 * ones( size( iid2impath_bente ) );
    iid2setid_cante = 2 * ones( size( iid2impath_cante ) );
    iid2setid_norte = 2 * ones( size( iid2impath_norte ) );
    
    iid2setid = cat( 1, ...
        iid2setid_bentr, iid2setid_cantr, iid2setid_nortr, ...
        iid2setid_bente, iid2setid_cante, iid2setid_norte );

end


