%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all;
addpath( genpath( '..' ) ); init;
setting.db                                      = path.db.voc2007; path.db.ilsvrcclsloc2015; path.db.ilsvrcdet2015; 
setting.io.tsDb.numScaling                      = 24; 
setting.io.tsDb.dilate                          = 1 / 4;
setting.io.tsDb.normalizeImageMaxSide           = 500; 0;
setting.io.tsDb.posGotoMargin                   = 2.4;
setting.io.tsDb.numQuantizeBetweenStopAndGoto   = 3;
setting.io.tsDb.negIntOverObjLessThan           = 0.1;
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutAttNetCornerPerCls( db, setting.io.tsDb, [  ], [  ] );
io.init;
io.makeTsDb;

%% DO THE JOB.
clearvars -except db io path setting;
rootDir = '/nickel/data_attnet_clsagn';
dstName = strcat( setting.db.name, '_PAENG' );
dbRoot = setting.db.root;
numEpoch = 10;
for setid = 1 : 2;
    if dbRoot( end ) == '/', dbRoot( end ) = ''; end;
    switch setid,
        case 1,
            setName = 'train';
            subTsDb = io.tsDb.tr;
        case 2,
            setName = 'val';
            numEpoch = 1;
            subTsDb = io.tsDb.val;
    end;
    bgdClsId = 5;
    dpid2dp = io.directions.dpid2dp;
    numDirPair = size( dpid2dp, 2 );
    numIm = max( subTsDb.oid2iid );
    iid2oids = arrayfun( @( iid )find( subTsDb.oid2iid == iid ), ( 1 : numIm )', 'UniformOutput', false );
    if any( ~cellfun( @numel, iid2oids ) ), error( 'Several images have no object.\n' ); end;
    dstDir = fullfile( rootDir, dstName );
    if ~exist( dstDir, 'dir' ), mkdir( dstDir ); end;
    dstPath = fullfile( rootDir, dstName, [ setName, '.txt' ] );
    fp = fopen( dstPath, 'w' );
    cummt = 0;
    for epch = 1 : numEpoch, 
        for iid = 1 : numIm, itime = tic;
            oids = iid2oids{ iid };
            cids = subTsDb.oid2cid( oids );
            numObj = numel( oids );
            bgds = subTsDb.iid2sid2negregns{ iid };
            bgdScales = find( ~cellfun( @isempty, bgds ) );
            noBgd = isempty( bgdScales );
            numSample = numObj * numDirPair * 2;
            sid2tlbr = zeros( 4, numSample, 'single' );
            sid2gt = zeros( 2, numSample, 'single' );
            sid2cid = zeros( numSample, 'single' );
            sid2flip = zeros( numSample, 1, 'single' );
            sid = 0;
            for o = 1 : numObj,
                oid = oids( o );
                cid = cids( o );
                snegregns = subTsDb.oid2snegregns{ oid };
                numSobj = min( round( numDirPair / 2 ), size( snegregns, 2 ) ); %%
                if noBgd, numBgd = 0; else numBgd = numDirPair - numSobj; end;
                numFgd = numDirPair * 2 - numSobj - numBgd;
                % Foreground sampling.
                for n = 1 : numFgd,
                    dpid = mod( n, numDirPair );
                    if ~dpid, dpid = numDirPair; end;
                    flip = round( rand );
                    if flip,
                        frid2tlbr = subTsDb.oid2dpid2posregnsFlip{ oid }{ dpid };
                    else
                        frid2tlbr = subTsDb.oid2dpid2posregns{ oid }{ dpid };
                    end;
                    frid = ceil( size( frid2tlbr, 2 ) * rand );
                    sid = sid + 1;
                    sid2tlbr( :, sid ) = frid2tlbr( :, frid );
                    sid2gt( :, sid ) = dpid2dp( :, dpid );
                    sid2cid( sid ) = cid;
                    sid2flip( sid ) = flip;
                end;
                % Sub-object sampling.
                for n = 1 : numSobj,
                    sid = sid + 1;
                    snrid = ceil( size( snegregns, 2 ) * rand ); %%
                    sid2tlbr( :, sid ) = snegregns( :, snrid ); %%
                    sid2gt( :, sid ) = [ bgdClsId; bgdClsId; ];
                    sid2cid( sid ) = cid;
                    sid2flip( sid ) = round( rand );
                end;
                % Background sampling.
                for n = 1 : numBgd,
                    s = bgdScales( ceil( numel( bgdScales ) * rand ) );
                    brid2tlbr = bgds{ s };
                    brid = ceil( size( brid2tlbr, 2 ) * rand );
                    sid = sid + 1;
                    sid2tlbr( :, sid ) = brid2tlbr( :, brid );
                    sid2gt( :, sid ) = [ bgdClsId; bgdClsId; ];
                    sid2cid( sid ) = cid;
                    sid2flip( sid ) = round( rand );
                end;
            end;
            if sid ~= numSample, error( 'Wrong # of smaples per im.\n' ); end;
            % Writing.
            f.impath = subTsDb.iid2impath{ iid };
            f.impath( 1 : numel( dbRoot ) + 1 ) = '';
            f.imidx = iid - 1;
            f.numBox = sid;
            f.bbox = sid2tlbr( [ 2; 1; 4; 3; ], : ) - 1;
            f.flip = sid2flip;
            f.gt = sid2gt - 1;
            f.cid = sid2cid - 1;
            fprintf( fp, '# %d\n', f.imidx );
            fprintf( fp, '%s\n', f.impath );
            fprintf( fp, '%d\n', f.numBox );
            for sid = 1 : f.numBox,
                fprintf( fp, '%d ', f.bbox( :, sid )' );
                fprintf( fp, '%d ', f.flip( sid ) );
                fprintf( fp, '%d ', f.gt( :, sid )' );
                fprintf( fp, '%d ', f.cid( sid )' );
                fprintf( fp, '\n' );
            end;
            % Display status.
            cummt = cummt + toc( itime );
            disploop( numIm * numEpoch, ( epch - 1 ) * numIm + iid, ...
                sprintf( 'Make pang data for epch %d/%d. %d/%d in %s.', epch, numEpoch, iid, numIm, lower( setName ) ), cummt );
        end;
    end;
    fclose( fp );
    fprintf( 'Done for %s.\n', lower( setName ) );
end;





%% VERIFICATION OF PAENG DATA.
clc; fclose all; clearvars -except db io path setting;
rootDir = '/nickel/data_attnet_clsagn';
dstName = strcat( setting.db.name, '_PAENG' );
dbRoot = setting.db.root;
setid = 1; 2; 

switch setid, case 1, setName = 'train'; case 2, setName = 'val'; end;
srcPath = fullfile( rootDir, dstName, [ setName, '.txt' ] );
fp = fopen( srcPath, 'r' );
figure( 1 );
set( gcf, 'color', 'w' );
while true,
    string = fgets( fp );
    iid = sscanf( string, '# %d\n' );
    string = fgets( fp );
    impath = sscanf( string, '%s\n' );
    string = fgets( fp );
    numBox = sscanf( string, '%d\n' );
    im = imread( fullfile( dbRoot, impath ) );
    for sid = 1 : numBox,
        string = fgets( fp );
        nums = sscanf( string, '%d ' );
        bbox = nums( 1 : 4 );
        nums( 1 : 4 ) = [  ];
        flip = nums( 1 );
        nums( 1 ) = [  ];
        gtDir = nums( 1 : 2 );
        nums( 1 : 2 ) = [  ];
        gtCls = nums( 1 );
        nums( 1 ) = [  ];
        if ~isempty( nums ), error( 'Wrong txt length.\n' ); end;
        switch gtDir( 1 ) + 1,
            case 1, dnameTl = 'down';
            case 2, dnameTl = 'diag';
            case 3, dnameTl = 'right';
            case 4, dnameTl = 'stop';
            case 5, dnameTl = 'bgd';
        end;
        switch gtDir( 2 ) + 1,
            case 1, dnameBr = 'up';
            case 2, dnameBr = 'diag';
            case 3, dnameBr = 'left';
            case 4, dnameBr = 'stop';
            case 5, dnameBr = 'bgd';
        end;
        cname = db.cid2name{ gtCls + 1 };
        iid0 = find( ismember( db.iid2impath, fullfile( dbRoot, impath ) ) );
        subplot( 1, 2, 1 );
        plottlbr( db.oid2bbox( :, db.iid2oids{ iid0 } ), im, false, 'r' );
        title( 'Ground-truth' );
        hold off;
        subplot( 1, 2, 2 );
        imRegn = uint8( normalizeAndCropImage( im, bbox( [ 2; 1; 4; 3; ], : ) + 1, uint8( cat( 3, 0, 0, 0 ) ), 'bicubic' ) );
        imRegn = imresize( imRegn, [ io.patchSide, io.patchSide ] );
        if flip, imRegn = fliplr( imRegn ); end;
        imshow( imRegn ); 
        title( sprintf( 'IID%d, %s, %s/%s (%d/%d)', iid + 1, cname, dnameTl, dnameBr, sid, numBox ) );
        hold off;
        waitforbuttonpress; 
    end;
end;






