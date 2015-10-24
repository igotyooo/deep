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
io = InOutAttNetCornerPerCls2( db, setting.io.tsDb, [  ], [  ] );
io.init;
io.makeTsDb;

%% DO THE JOB.
clearvars -except db io path setting;
rootDir = '/nickel/data_attnet_clsagn';
dstName = strcat( setting.db.name, '_PAENG2' );
dbRoot = setting.db.root;
numEpoch = 10;
for setid = 1 : 2;
    if dbRoot( end ) == '/', dbRoot( end ) = ''; end;
    numDirPair = size( io.directions.dpid2dp, 2 );
    numBgdPerObj = numDirPair;
    switch setid,
        case 1,
            setName = 'train';
            subTsDb = io.tsDb.tr;
        case 2,
            setName = 'val';
            numEpoch = 1;
            subTsDb = io.tsDb.val;
    end;
    numClass = db.getNumClass;
    numObj = numel( subTsDb.oid2iid );
    bgdClsId = numClass + 1;
    dpid2dp = io.directions.dpid2dp;
    numLyr = numClass * 2 + 1;
    rng( 'shuffle' );
    if max( subTsDb.oid2iid ) ~= numel( subTsDb.iid2impath ), error( 'In consistent # of im.\n' ); end;
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
            % Processing.
            oids = iid2oids{ iid };
            cids = subTsDb.oid2cid( oids );
            numObj = numel( oids );
            bgds = subTsDb.iid2sid2negregns{ iid };
            bgdScales = find( ~cellfun( @isempty, bgds ) );
            fillBgdByFgd = isempty( bgdScales );
            numSample = numObj * ( numDirPair + numBgdPerObj );
            sid2tlbr = zeros( 4, numSample, 'single' );
            sid2gt = 5 * ones( 3, numSample, 'single' );
            sid2flip = zeros( numSample, 1, 'single' );
            sid = 0;
            for o = 1 : numObj,
                oid = oids( o );
                cid = cids( o );
                lids = [ cid * 2 - 1; cid * 2 ];
                for fillFgd = 1 : 1 + fillBgdByFgd,
                    for dpid = 1 : numDirPair,
                        flip = round( rand );
                        if flip,
                            frid2tlbr = subTsDb.oid2dpid2posregnsFlip{ oid }{ dpid };
                        else
                            frid2tlbr = subTsDb.oid2dpid2posregns{ oid }{ dpid };
                        end;
                        frid = ceil( size( frid2tlbr, 2 ) * rand );
                        sid = sid + 1;
                        sid2tlbr( :, sid ) = frid2tlbr( :, frid );
                        sid2gt( 1 : 2, sid ) = dpid2dp( :, dpid );
                        sid2gt( end, sid ) = cid;
                        sid2flip( sid ) = flip;
                    end;
                end;
                if ~fillBgdByFgd,
                    for n = 1 : numBgdPerObj,
                        s = bgdScales( ceil( numel( bgdScales ) * rand ) );
                        brid2tlbr = bgds{ s };
                        brid = ceil( size( brid2tlbr, 2 ) * rand );
                        sid = sid + 1;
                        sid2tlbr( :, sid ) = brid2tlbr( :, brid );
                        sid2gt( end, sid ) = bgdClsId;
                        sid2flip( sid ) = round( rand );
                    end;
                end;
            end;
            if sid ~= numSample, error( 'In consistent # of samples.\n' ); end;
            % Writing.
            f.impath = subTsDb.iid2impath{ iid };
            f.impath( 1 : numel( dbRoot ) + 1 ) = '';
            f.imidx = iid - 1;
            f.numBox = sid;
            f.bbox = sid2tlbr( [ 2; 1; 4; 3; ], : ) - 1;
            f.flip = sid2flip;
            f.gt = sid2gt - 1;
            fprintf( fp, '# %d\n', f.imidx );
            fprintf( fp, '%s\n', f.impath );
            fprintf( fp, '%d\n', f.numBox );
            for sid = 1 : f.numBox,
                fprintf( fp, '%d ', f.bbox( :, sid )' );
                fprintf( fp, '%d ', f.flip( sid ) );
                fprintf( fp, '%d ', f.gt( :, sid )' );
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
dstName = strcat( setting.db.name, '_PAENG2' );
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
        gtDir = nums( 1 : end - 1 );
        nums( 1 : end - 1 ) = [  ];
        gtCls = nums;
        nums( 1 ) = [  ];
        if ~isempty( nums ), error( 'Wrong txt length.\n' ); end;
        bbox = bbox( [ 2; 1; 4; 3; ], : );
        bbox = bbox + 1;
        gtDir = gtDir + 1;
        gtDir = gtDir( gtDir ~= 5 );
        gtCls = gtCls + 1;
        if ~isempty( gtDir ),
            switch gtDir( 1 )
                case 1, dnameTl = 'down';
                case 2, dnameTl = 'diag';
                case 3, dnameTl = 'right';
                case 4, dnameTl = 'stop';
            end;
            switch gtDir( 2 )
                case 1, dnameBr = 'up';
                case 2, dnameBr = 'diag';
                case 3, dnameBr = 'left';
                case 4, dnameBr = 'stop';
            end;
        else
            if gtCls ~= ( db.getNumClass + 1 ), error( 'Wrong fgd/bgd label.' ); end;
            dnameTl = 'bgd';
            dnameBr = 'bgd';
        end;
        if gtCls == ( db.getNumClass + 1 ),
            cname = 'bgd';
        else
            cname = db.cid2name{ gtCls };
        end;
        iid0 = find( ismember( db.iid2impath, fullfile( dbRoot, impath ) ) );
        subplot( 1, 2, 1 );
        plottlbr( db.oid2bbox( :, db.iid2oids{ iid0 } ), im, false, 'r' );
        title( 'Ground-truth' );
        hold off;
        subplot( 1, 2, 2 );
        imRegn = uint8( normalizeAndCropImage( im, bbox, uint8( cat( 3, 0, 0, 0 ) ), 'bicubic' ) );
        imRegn = imresize( imRegn, [ io.patchSide, io.patchSide ] );
        if flip, imRegn = fliplr( imRegn ); end;
        imshow( imRegn ); 
        title( sprintf( 'IID%d, %s, %s/%s (%d/%d)', iid, cname, dnameTl, dnameBr, sid, numBox ) );
        hold off;
        waitforbuttonpress; 
    end;
end;






