function [ cid2name, iid2impath, iid2size, iid2setid, oid2cid, oid2diff, oid2iid, oid2bbox, oid2cont ]...
    = DB_DDSM

    global path;
    rootdir = path.db.ddsm.root;
    imForm = '*.jpg';
    cid2name = { 'benign_cc'; 'cancer_cc'; 'normal_cc'; 'benign_mlo'; 'cancer_mlo'; 'normal_mlo'; };
    % Make iid2impath.
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
    % Make iid2setid.
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
    
    iid2cid_bentr_cc = 1 * ones( size( iid2impath_bentr_cc ) );
    iid2cid_cantr_cc = 2 * ones( size( iid2impath_cantr_cc ) );
    iid2cid_nortr_cc = 3 * ones( size( iid2impath_nortr_cc ) );
    iid2cid_bente_cc = 1 * ones( size( iid2impath_bente_cc ) );
    iid2cid_cante_cc = 2 * ones( size( iid2impath_cante_cc ) );
    iid2cid_norte_cc = 3 * ones( size( iid2impath_norte_cc ) );
    iid2cid_bentr_mlo = 4 * ones( size( iid2impath_bentr_mlo ) );
    iid2cid_cantr_mlo = 5 * ones( size( iid2impath_cantr_mlo ) );
    iid2cid_nortr_mlo = 6 * ones( size( iid2impath_nortr_mlo ) );
    iid2cid_bente_mlo = 4 * ones( size( iid2impath_bente_mlo ) );
    iid2cid_cante_mlo = 5 * ones( size( iid2impath_cante_mlo ) );
    iid2cid_norte_mlo = 6 * ones( size( iid2impath_norte_mlo ) );
    iid2cid = cat( 1, ...
        iid2cid_bentr_cc, iid2cid_cantr_cc, iid2cid_nortr_cc, ...
        iid2cid_bente_cc, iid2cid_cante_cc, iid2cid_norte_cc, ...
        iid2cid_bentr_mlo, iid2cid_cantr_mlo, iid2cid_nortr_mlo, ...
        iid2cid_bente_mlo, iid2cid_cante_mlo, iid2cid_norte_mlo );
    
    numIm = numel( iid2impath );
    iid2size = zeros( 2, numIm );
    oid2cid = cell( numIm, 1 );
    oid2diff = cell( numIm, 1 );
    oid2iid = cell( numIm, 1 );
    oid2bbox = cell( numIm, 1 );
    oid = 0;
    for iid = 1 : numIm,
        % Make iid2size.
        fprintf( 'Read im %d/%d.', iid, numIm );
        impath = iid2impath{ iid };
        im = imread( impath );
        [ r, c, ~ ] = size( im );
        iid2size( :, iid ) = [ r; c; ];
        % Make objects.
        cid = iid2cid( iid );
        [ direct, name, ~ ] = fileparts( impath );
        name = strcat( name, '_CONTOUR.mat' );
        contpath = fullfile( direct, name );
        try
            data = load( contpath );
            cont = data.imct;
            cont = splitContour( cont );
            if isempty( cont )
                % Make oid2cid.
                oid2cid{ iid } = cid;
                % Make oid2diff.
                oid2diff{ iid } = false;
                % Make oid2iid.
                oid2iid{ iid } = iid;
                % Make oid2cont.
                oid = oid + 1;
                oid2cont{ oid } = [  ];
                % Make oid2bbox.
                oid2bbox{ iid } = [ 1; 1; r; c; ];
                fprintf( ' No contour.\n' );
            else
                numObj = numel( cont );
                % Make oid2cid.
                oid2cid{ iid } = cid * ones( numObj, 1 );
                % Make oid2diff.
                oid2diff{ iid } = false( numObj, 1 );
                % Make oid2iid.
                oid2iid{ iid } = iid * ones( numObj, 1 );
                % Make oid2cont.
                for n = 1 : numObj,
                    oid = oid + 1;
                    oid2cont{ oid } = cont{ n };
                end
                % Make oid2bbox.
                bbox = zeros( 4, numObj );
                for n = 1 : numObj, bbox( :, n ) = contour2bbox( cont{ n } ); end;
                oid2bbox{ iid } = bbox;
                fprintf( '\n' );
            end
        catch
            % Make oid2cid.
            oid2cid{ iid } = cid;
            % Make oid2diff.
            oid2diff{ iid } = false;
            % Make oid2iid.
            oid2iid{ iid } = iid;
            % Make oid2cont.
            oid = oid + 1;
            oid2cont{ oid } = [  ];
            % Make oid2bbox.
            oid2bbox{ iid } = [ 1; 1; r; c; ];
            fprintf( ' No contour.\n' );
        end
    end
    oid2cid = cat( 1, oid2cid{ : } );
    oid2diff = cat( 1, oid2diff{ : } );
    oid2iid = cat( 1, oid2iid{ : } );
    oid2bbox = cat( 2, oid2bbox{ : } );

end




