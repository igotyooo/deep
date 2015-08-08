function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox, ...
            oid2cont ] = DB_INDOOR_DEVICES
    global path;
    numTePerCls = 100;
    rootDir = path.db.indoor_devices.root;
    cid2name = dir( rootDir );
    cid2name = { cid2name( 3 : end ).name }';
    numCls = numel( cid2name );
    iid2impath = cell( numCls, 1 );
    iid2setid = cell( numCls, 1 );
    oid2cid = cell( numCls, 1 );
    for cid = 1 : numCls,
        clsDir = fullfile( rootDir, cid2name{ cid } );
        impaths = dir( clsDir );
        impaths = fullfile( clsDir, { impaths( 3 : end ).name }' );
        iid2impath{ cid } = impaths;
        iid2setid{ cid } = ones( size( impaths ) );
        iid2setid{ cid }( 1 : numTePerCls ) = 2;
        oid2cid{ cid } = cid * ones( size( impaths ) );
    end;
    iid2impath = cat( 1, iid2impath{ : } );
    numIm = numel( iid2impath );
    iid2setid = cat( 1, iid2setid{ : } );
    oid2cid = cat( 1, oid2cid{ : } );
    oid2diff = false( numIm, 1 );
    oid2iid = ( 1 : numIm )';
    oid2cont = cell( numIm, 1 );
    iid2size = zeros( 2, numIm );
    parfor i = 1 : numIm,
        [ r, c, ~ ] = size( imread( iid2impath{ i } ) );
        iid2size( :, i ) = [ r; c; ];
    end;
    oid2bbox = [ ones( 2, numIm ); iid2size ];
end