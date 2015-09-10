function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox, ...
            oid2cont ] = DB_ILSVRCDET2014

    global path;
    dirRoot = path.db.ilsvrcdet2014.root;
    dirTrIm = fullfile( dirRoot, 'TR' );
    dirTrBb = fullfile( dirRoot, 'TR_BBOX' );
    dirValIm = fullfile( dirRoot, 'VAL' );
    dirValBb = fullfile( dirRoot, 'VAL_BBOX' );
    subDirs1 = dir( fullfile( dirTrIm, 'ILSVRC2014*' ) );
    subDirs1 = { subDirs1.name }';
    subDirs2 = dir( fullfile( dirTrIm, 'n*' ) );
    subDirs2 = { subDirs2.name }';
    subDirs = cat( 1, subDirs1, subDirs2 );
    numSubDir = numel( subDirs );
    % Training images - positive images.
    iid2impathTr = cell( numSubDir, 1 );
    iid2bbpathTr = cell( numSubDir, 1 );
    fprintf( 'Find training images.\n' );
    parfor i = 1 : numSubDir,
        imdir = fullfile( dirTrIm, subDirs{ i } );
        impath = dir( imdir );
        impath = fullfile( imdir, { impath( 3 : end ).name }' );
        iid2impathTr{ i } = impath;
        bbdir = fullfile( dirTrBb, subDirs{ i } );
        bbpath = dir( bbdir );
        bbpath = fullfile( bbdir, { bbpath( 3 : end ).name }' );
        iid2bbpathTr{ i } = bbpath;
        if numel( impath ) ~= numel( bbpath ), error( 'Inconsistent number of im-gt pairs.' ); end;
    end;
    iid2impathTr = cat( 1, iid2impathTr{ : } );
    iid2bbpathTr = cat( 1, iid2bbpathTr{ : } );
    fprintf( 'Done.\n' );
    % Validation images.
    fprintf( 'Find validation images.\n' );
    iid2impathVal = dir( dirValIm );
    iid2impathVal = fullfile( dirValIm, { iid2impathVal( 3 : end ).name }' );
    iid2bbpathVal = dir( dirValBb );
    iid2bbpathVal = fullfile( dirValBb, { iid2bbpathVal( 3 : end ).name }' );
    if numel( iid2impathVal ) ~= numel( iid2bbpathVal ), error( 'Inconsistent number of im-gt pairs.' ); end;
    fprintf( 'Done.\n' );
    iid2impath = cat( 1, iid2impathTr, iid2impathVal );
    iid2bbpath = cat( 1, iid2bbpathTr, iid2bbpathVal );
    iid2setid = cat( 1, ones( size( iid2impathTr ) ), 2 * ones( size( iid2impathVal ) ) );
    % Read xmls.
    numIm = numel( iid2impath );
    iid2size = cell( numIm, 1 );
    oid2name = cell( numIm, 1 );
    oid2bbox_ = cell( numIm, 1 );
    fprintf( 'Read xmls.\n' );
    parfor iid = 1 : numIm,
        try
            bbpath = iid2bbpath{ iid };
            anno = VOCreadxml( bbpath );
            anno = anno.annotation;
            iid2size{ iid } = [ str2double( anno.size.height ); str2double( anno.size.width ); ];
            oid2name{ iid } = { anno.object.name }';
            oid2bbox_{ iid } = { anno.object.bndbox }';
        catch
            fprintf( 'Read xmls: No obj in iid %06d.\n', iid );
        end;
    end;
    oid2name = cat( 1, oid2name{ : } );
    iid2size = cat( 2, iid2size{ : } );
    numObj = numel( oid2name );
    fprintf( 'Done.\n' );
    % Compute bbox.
    fprintf( 'Compute bbox.\n' );
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
    fprintf( 'Done.\n' );
    fprintf( 'Compute class.\n' );
    [  cid2name, ~, oid2cid ] = unique( oid2name );
    meta = load( fullfile( path.lib.ilsvrcDevKit, 'data/meta_det.mat' ) );
    numMeta = numel( meta.synsets );
    numClass = numel( cid2name );
    mid2name = cell( numMeta, 1 );
    for mid = 1 : numMeta,
        cname = meta.synsets( mid ).name;
        len = numel( cname );
        idx = strfind( cname, ', ' );
        if ~isempty( idx ),
            sidx = [ 1, idx + 2 ];
            eidx = [ idx - 1, len ];
            wlen = eidx - sidx + 1;
            [ ~, sel ] = min( wlen );
            cname = cname( sidx( sel ) : eidx( sel ) );
        end;
        mid2name{ mid } = cname;
    end;
    mid2wnid = { meta.synsets.WNID }';
    for cid = 1 : numClass,
        cname = cid2name{ cid };
        mid = ismember( mid2wnid, cname );
        cid2name{ cid } = mid2name{ mid };
    end;
    fprintf( 'Done.\n' );
    % Reject images if defined as "part".
    fprintf( 'Reject images if defined as "part".\n' );
    numIm = numel( iid2impath );
    iid2name = cell( numIm, 1 );
    parfor iid = 1 : numIm,
        [ ~, iid2name{ iid } ] = fileparts( iid2impath{ iid } );
    end;
    pid2path = dir( fullfile( path.lib.ilsvrcDevKit, 'data/det_lists/*_part_*' ) );
    pid2path = fullfile( path.lib.ilsvrcDevKit, 'data/det_lists', { pid2path.name }' );
    iid2ispart = false( numIm, 1 );
    numPart = numel( pid2path );
    for pid = 1 : numPart,
        fp = fopen( pid2path{ pid }, 'r' );
        imlist = textscan( fp, '%s' );
        imlist = imlist{ : };
        fclose( fp );
        for l = 1 : numel( imlist ),
            [ ~, partname ] = fileparts( imlist{ l } );
            partid = ismember( iid2name, partname );
            if sum( partid ) ~= 1, error( 'Error.\n' ); end;
            iid2ispart( partid ) = true;
        end;
    end;
    % iid2impath, iid2size, iid2setid, oid2cid, oid2iid, oid2bbox
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
    % Reject too small bounding boxes.
    % minObjRate = 0.002;
    % iid2area = prod( iid2size, 1 );
    % oid2imarea = iid2area( oid2iid );
    % oid2small = ( oid2area ./ oid2imarea ) < minObjRate;
    % oid = randsample( find( oid2small ), 1 );
    % iid = oid2iid( oid );
    % figure( 1 ); plottlbr( oid2bbox( :, oid2iid == iid ), iid2impath{ iid }, false, 'r' );
    % figure( 2 ); plottlbr( oid2bbox( :, oid ), iid2impath{ iid }, false, 'r' );
    oid2cont = cell( size( oid2cid ) );
    oid2diff = false( size( oid2cid ) );
    % Data type conversion.
    iid2size = single( iid2size );
    iid2setid = single( iid2setid );
    oid2cid = single( oid2cid );
    oid2iid = single( oid2iid );
    oid2bbox = single( oid2bbox );
end

