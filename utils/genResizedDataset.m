
clc; close all; fclose all; clear all; reset( gpuDevice );
setpath;
dbPath = path.db_ilsvrc_clsloc;
setting.db.dir                   = path.dstdir;
setting.db.dbName                = dbPath.name;
setting.db.funGenDbCls           = dbPath.funh;
db = DbCls( setting.db );
db.genDbCls;


%% RESIZE
srcdir = '/iron/db/ILSVRC_CLSLOC/';
dstdir = '/db/ILSVRC_CLSLOC_256WC/';    % RAM disk.
dstsize = [ 256, 256, 3 ];              % 256C
keepAspect = false;                     % 256C for true, 256WC for false.
tw = dstsize( 2 ); th = dstsize( 1 ); tz = dstsize( 3 );
iid2ifpath = db.iid2ifpath;
numIm = db.getNumIm;
parfor iid = 1 : numIm
    srcfpath = iid2ifpath{ iid };
    im = imread( srcfpath );
    if size( im, 3 ) ~= 1 && size( im, 3 ) ~= 3
        error( 'Not supported dimension: %s\n', srcfpath );
    end
    % Resize z.
    if tz == 1 && size( im, 3 ) == 3
        im = rgb2gray( im );
    elseif tz == 3 && size( im, 3 ) == 1
        im = cat( 3, im, im, im );
    end
    % Resize xy.
    w = size( im, 2 ); h = size( im, 1 );
    factor = [ th / h, tw / w ];
    if keepAspect, factor = max( factor ); end
    im = imresize( im, 'scale', factor, 'method', 'bicubic' );
    % Save it to the destination.
    dstfpath = strcat( dstdir, srcfpath( numel( srcdir ) + 1 : end ) );
    dstdir_ = fileparts( dstfpath );
    if ~exist( dstdir_, 'dir' ), mkdir( dstdir_ ); end;
    imwrite( im, dstfpath );
    fprintf( '[%.2f%%] Im %06d/%d: Done.\n', iid / numIm * 100, iid, numIm );
end