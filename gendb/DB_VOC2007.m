function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox ] = DB_VOC2007

    global path;
    triids = dlmread( fullfile( path.db_voc2007.root, 'ImageSets', 'Main', 'trainval.txt' ) );
    teiids = dlmread( fullfile( path.db_voc2007.root, 'ImageSets', 'Main', 'test.txt' ) );
    numIm = length( triids ) + length( teiids );
    imdir = fullfile( path.db_voc2007.root, 'JPEGImages' );
    annodir = fullfile( path.db_voc2007.root, 'Annotations' );
    iid2impath = arrayfun( @( iid )fullfile( imdir, sprintf( '%06d.jpg', iid ) ), 1 : numIm, 'UniformOutput', false )';
    iid2annopath = arrayfun( @( iid )fullfile( annodir, sprintf( '%06d.xml', iid ) ), 1 : numIm, 'UniformOutput', false )';
    iid2setid = zeros( size( iid2impath ) );
    iid2setid( triids ) = 1;
    iid2setid( teiids ) = 2;
    iid2size = cell( numIm, 1 );
    oid2name = cell( numIm, 1 );
    oid2diff = cell( numIm, 1 );
    oid2bbox_ = cell( numIm, 1 );
    parfor iid = 1 : numIm
        anno = VOCreadxml( iid2annopath{ iid } );
        anno = anno.annotation;
        iid2size{ iid } = [ str2double( anno.size.height ); str2double( anno.size.width ); ];
        oid2name{ iid } = { anno.object.name }';
        oid2diff{ iid } = logical( str2double( { anno.object.difficult }' ) );
        oid2bbox_{ iid } = { anno.object.bndbox }';
    end
    oid2name = cat( 1, oid2name{ : } );
    oid2diff = cat( 1, oid2diff{ : } );
    iid2size = cat( 2, iid2size{ : } );
    numObj = numel( oid2diff );

    oid2bbox = cell( numObj, 1 );
    oid2iid = zeros( numObj, 1 );
    oid = 0;
    for iid = 1 : numIm,
        no = numel( oid2bbox_{ iid } );
        for oidx = 1 : no
            oid = oid + 1;
            oid2iid( oid ) = iid;
            oid2bbox{ oid } = ...
                [   str2double( oid2bbox_{ iid }{ oidx }.ymin ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.xmin ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.ymax ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.xmax ); ];
        end
    end
    oid2bbox = cat( 2, oid2bbox{ : } );
    [  cid2name, ~, oid2cid ] = unique( oid2name );

end

