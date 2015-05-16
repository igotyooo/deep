clc; close all; fclose all; clear all;
setpath;
db = setdb( 'ILSVRC_CLSLOC' );


%% PREPARE DATA.
setting.label.dbName = db.name;
setting.label.dstdir = fullfile( path.dstData, db.name );
setting.label.funGenLabel = db.funh;
label = Label( setting.label );
label.setLabel;

%% VERIFY ALL IMAGE DATA. V2.
bid = 0; invalFlist = {  }; numInvalid = 0; batchSize = 256; cummt = 0;
iids = 1 : label.getNumIm;
numThreads = feature( 'numCores' );
numBatch = numel( 1 : batchSize : numel( iids ) );
for beginIdx = 1 : batchSize : numel( iids ); tic;
    bid = bid + 1;
    biids = iids( beginIdx : min( beginIdx + batchSize - 1, numel( iids ) ) );
    bifpaths = label.iid2ifpath( biids );
    batchIms = vl_imreadjpeg( bifpaths, 'numThreads', numThreads );
    invalIdx = find( cellfun( @isempty, batchIms )' );
    for i = invalIdx
        numInvalid = numInvalid + 1;
        invalFlist{ end + 1 } = bifpaths{ i };
    end
    cummt = cummt + toc;
    disploop( numBatch, bid, ...
        sprintf( 'Verify im of batch %06d(/%d). # of invalid im: %d now.', bid, numBatch, numInvalid ), cummt );
end
fprintf( 'Done.\n' );

%% COPY INVALID IMAGES TO ANOTHER DIRECTORY.
dstdir = '/iron/db/ILSVRC_CLSLOC/INVALID/WRONGJPEG';
if ~exist( dstdir, 'dir' ), mkdir( dstdir ); end;
orgFlist = invalFlist;
for f = orgFlist
    [ ~, fname ] = fileparts( f{ : } );
    fname = strcat( fname, '.JPEG' );
    copyfile( f{ : }, fullfile( dstdir, fname ) );
end

%% MODIFY THE INVALID IMAGES.
srcdir = '/iron/db/ILSVRC_CLSLOC/INVALID/WRONGJPEG';
dstdir = '/iron/db/ILSVRC_CLSLOC/INVALID/VALIDJPEG';
if ~exist( dstdir, 'dir' ), mkdir( dstdir ); end;
orgFlist = invalFlist;
for f = orgFlist
    [ ~, fname ] = fileparts( f{ : } );
    fname = strcat( fname, '.JPEG' );
    im = imread( fullfile( srcdir, fname ) );
    imshow( im ); waitforbuttonpress;
    imwrite( im, fullfile( dstdir, fname ), 'JPEG' );
end



%% MOVE MODIFIED VALID IMAGES TO THE ORIGINAL LOCATION.
srcdir = '/iron/db/ILSVRC_CLSLOC/INVALID/VALIDJPEG';
dstFlist = orgFlist;
for f = dstFlist
    [ ~, fname ] = fileparts( f{ : } );
    fname = strcat( fname, '.JPEG' );
    im = vl_imreadjpeg( { fullfile( srcdir, fname ) } );
    imshow( uint8( im{ : } ) );
    waitforbuttonpress;
    copyfile( fullfile( srcdir, fname ), f{ : } );
end

%% VARIFICATION
for f = orgFlist
    imshow( f{ : } );
    im = vl_imreadjpeg( f );
    imshow( uint8( im{ : } ) );
    waitforbuttonpress;
end




