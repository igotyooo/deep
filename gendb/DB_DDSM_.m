function [ cid2name, iid2impath, iid2size, iid2setid, oid2cid, oid2diff, oid2iid, oid2bbox, oid2cont ]...
    = DB_DDSM

    global path;
    rootdir = path.db.ddsm.root;
    imForm = '*.jpg';
    cid2name = { 'benign_cc'; 'cancer_cc'; 'normal_cc'; 'benign_mlo'; 'cancer_mlo'; 'normal_mlo'; };
    
    dir_bentr_cc = fullfile( rootdir, 'tr', 'benign_cc' );
    dir_cantr_cc = fullfile( rootdir, 'tr', 'cancer_cc' );
    dir_nortr_cc = fullfile( rootdir, 'tr', 'normal_cc' );
    dir_bente_cc = fullfile( rootdir, 'te', 'benign_cc' );
    dir_cante_cc = fullfile( rootdir, 'te', 'cancer_cc' );
    dir_norte_cc = fullfile( rootdir, 'te', 'normal_cc' );
    
    dir_bentr_mlo = fullfile( rootdir, 'tr', 'benign_mlo' );
    dir_cantr_mlo = fullfile( rootdir, 'tr', 'cancer_mlo' );
    dir_nortr_mlo = fullfile( rootdir, 'tr', 'normal_mlo' );
    dir_bente_mlo = fullfile( rootdir, 'te', 'benign_mlo' );
    dir_cante_mlo = fullfile( rootdir, 'te', 'cancer_mlo' );
    dir_norte_mlo = fullfile( rootdir, 'te', 'normal_mlo' );

    iid2impath_bentr_cc = dir( fullfile( dir_bentr_cc, imForm ) );
    iid2impath_bentr_cc = fullfile( dir_bentr_cc, { iid2impath_bentr_cc( : ).name }' );
    iid2impath_cantr_cc = dir( fullfile( dir_cantr_cc, imForm ) );
    iid2impath_cantr_cc = fullfile( dir_cantr_cc, { iid2impath_cantr_cc( : ).name }' );
    iid2impath_nortr_cc = dir( fullfile( dir_nortr_cc, imForm ) );
    iid2impath_nortr_cc = fullfile( dir_nortr_cc, { iid2impath_nortr_cc( : ).name }' );
    iid2impath_bente_cc = dir( fullfile( dir_bente_cc, imForm ) );
    iid2impath_bente_cc = fullfile( dir_bente_cc, { iid2impath_bente_cc( : ).name }' );
    iid2impath_cante_cc = dir( fullfile( dir_cante_cc, imForm ) );
    iid2impath_cante_cc = fullfile( dir_cante_cc, { iid2impath_cante_cc( : ).name }' );
    iid2impath_norte_cc = dir( fullfile( dir_norte_cc, imForm ) );
    iid2impath_norte_cc = fullfile( dir_norte_cc, { iid2impath_norte_cc( : ).name }' );
    
    iid2impath_bentr_mlo = dir( fullfile( dir_bentr_mlo, imForm ) );
    iid2impath_bentr_mlo = fullfile( dir_bentr_mlo, { iid2impath_bentr_mlo( : ).name }' );
    iid2impath_cantr_mlo = dir( fullfile( dir_cantr_mlo, imForm ) );
    iid2impath_cantr_mlo = fullfile( dir_cantr_mlo, { iid2impath_cantr_mlo( : ).name }' );
    iid2impath_nortr_mlo = dir( fullfile( dir_nortr_mlo, imForm ) );
    iid2impath_nortr_mlo = fullfile( dir_nortr_mlo, { iid2impath_nortr_mlo( : ).name }' );
    iid2impath_bente_mlo = dir( fullfile( dir_bente_mlo, imForm ) );
    iid2impath_bente_mlo = fullfile( dir_bente_mlo, { iid2impath_bente_mlo( : ).name }' );
    iid2impath_cante_mlo = dir( fullfile( dir_cante_mlo, imForm ) );
    iid2impath_cante_mlo = fullfile( dir_cante_mlo, { iid2impath_cante_mlo( : ).name }' );
    iid2impath_norte_mlo = dir( fullfile( dir_norte_mlo, imForm ) );
    iid2impath_norte_mlo = fullfile( dir_norte_mlo, { iid2impath_norte_mlo( : ).name }' );
    
    iid2impath = cat( 1, ...
        iid2impath_bentr_cc, iid2impath_cantr_cc, iid2impath_nortr_cc, ...
        iid2impath_bente_cc, iid2impath_cante_cc, iid2impath_norte_cc, ...
        iid2impath_bentr_mlo, iid2impath_cantr_mlo, iid2impath_nortr_mlo, ...
        iid2impath_bente_mlo, iid2impath_cante_mlo, iid2impath_norte_mlo );
    
    numIm = numel( iid2impath );
    iid2size = zeros( 2, numIm );
    oid2cont = cell( numIm, 1 );
    for iid = 1 : numIm,
        fprintf( 'Read im %d/%d.', iid, numIm );
        impath = iid2impath{ iid };
        im = imread( impath );
        [ r, c, ~ ] = size( im );
        iid2size( :, iid ) = [ r; c; ];
        [ direct, name, ~ ] = fileparts( impath );
        name = strcat( name, '_CONTOUR.mat' );
        contpath = fullfile( direct, name );
        try
            data = load( contpath );
            oid2cont{ iid } = data.imct;
            fprintf( '\n' );
        catch
            fprintf( ' No contour.\n' );
        end
    end
    
    oid2cid_bentr_cc = 1 * ones( size( iid2impath_bentr_cc ) );
    oid2cid_cantr_cc = 2 * ones( size( iid2impath_cantr_cc ) );
    oid2cid_nortr_cc = 3 * ones( size( iid2impath_nortr_cc ) );
    oid2cid_bente_cc = 1 * ones( size( iid2impath_bente_cc ) );
    oid2cid_cante_cc = 2 * ones( size( iid2impath_cante_cc ) );
    oid2cid_norte_cc = 3 * ones( size( iid2impath_norte_cc ) );
    
    oid2cid_bentr_mlo = 4 * ones( size( iid2impath_bentr_mlo ) );
    oid2cid_cantr_mlo = 5 * ones( size( iid2impath_cantr_mlo ) );
    oid2cid_nortr_mlo = 6 * ones( size( iid2impath_nortr_mlo ) );
    oid2cid_bente_mlo = 4 * ones( size( iid2impath_bente_mlo ) );
    oid2cid_cante_mlo = 5 * ones( size( iid2impath_cante_mlo ) );
    oid2cid_norte_mlo = 6 * ones( size( iid2impath_norte_mlo ) );
    
    oid2cid = cat( 1, ...
        oid2cid_bentr_cc, oid2cid_cantr_cc, oid2cid_nortr_cc, ...
        oid2cid_bente_cc, oid2cid_cante_cc, oid2cid_norte_cc, ...
        oid2cid_bentr_mlo, oid2cid_cantr_mlo, oid2cid_nortr_mlo, ...
        oid2cid_bente_mlo, oid2cid_cante_mlo, oid2cid_norte_mlo );
    
    oid2diff = false( numIm, 1 );
    oid2iid = ( 1 : numIm )';
    oid2bbox = cat( 1, ones( 2, numIm ), iid2size );
    
    iid2setid_bentr_cc = 1 * ones( size( iid2impath_bentr_cc ) );
    iid2setid_cantr_cc = 1 * ones( size( iid2impath_cantr_cc ) );
    iid2setid_nortr_cc = 1 * ones( size( iid2impath_nortr_cc ) );
    iid2setid_bente_cc = 2 * ones( size( iid2impath_bente_cc ) );
    iid2setid_cante_cc = 2 * ones( size( iid2impath_cante_cc ) );
    iid2setid_norte_cc = 2 * ones( size( iid2impath_norte_cc ) );
    
    iid2setid_bentr_mlo = 1 * ones( size( iid2impath_bentr_mlo ) );
    iid2setid_cantr_mlo = 1 * ones( size( iid2impath_cantr_mlo ) );
    iid2setid_nortr_mlo = 1 * ones( size( iid2impath_nortr_mlo ) );
    iid2setid_bente_mlo = 2 * ones( size( iid2impath_bente_mlo ) );
    iid2setid_cante_mlo = 2 * ones( size( iid2impath_cante_mlo ) );
    iid2setid_norte_mlo = 2 * ones( size( iid2impath_norte_mlo ) );
    
    iid2setid = cat( 1, ...
        iid2setid_bentr_cc, iid2setid_cantr_cc, iid2setid_nortr_cc, ...
        iid2setid_bente_cc, iid2setid_cante_cc, iid2setid_norte_cc, ...
        iid2setid_bentr_mlo, iid2setid_cantr_mlo, iid2setid_nortr_mlo, ...
        iid2setid_bente_mlo, iid2setid_cante_mlo, iid2setid_norte_mlo );

end


